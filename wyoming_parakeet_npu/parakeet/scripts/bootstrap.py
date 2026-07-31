"""Bootstrap helper: fetch what the server needs into /data, and only that.

Run on every start (from docker-entrypoint.sh). Idempotent, and deliberately
frugal: of the 3.2 GB the model repository offers, a default install keeps
about 1.3 GB. Three of the published files are only ever means to an end:

  * the INT8 encoder and decoder/joint drive onnx-asr's CPU pipeline, which
    this add-on does not build at all (see pipeline.py) — never downloaded;
  * the FP32 decoder/joint is the source for the static OpenVINO IR — fetched
    once, dropped as soon as the IR exists;
  * the 2.5 GB FP32 encoder graph is the source for the compiled NPU blobs —
    needed only for bucket sizes that have no precompiled blob to download.

The last one is why bucket bookkeeping lives in the installed package rather
than here: the server has to reach the same verdict, or it would look for a
graph this script decided to delete.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from wyoming_parakeet_npu import MODEL_NAME, buckets

HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
# Pinned repo revision (2026-02-17) so upstream changes can't slip in silently.
HF_REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"

# Optional manifest with prebuilt NPU encoder blobs (shipped by the add-on).
BLOB_MANIFEST = Path("/app/blobs.json")

# Always needed: the tokenizer vocabulary and the shape/feature config.
BASE_FILES = ["config.json", "vocab.txt"]

# Source for the static decoder IR (~72 MB), dropped once the IR is built.
DECODER_SOURCE = ["decoder_joint-model.onnx"]

# Source for on-device encoder compilation (~2.5 GB), kept only while some
# configured bucket has no precompiled blob.
ENCODER_SOURCE = ["encoder-model.onnx", "encoder-model.onnx.data"]

# Downloaded by add-on versions up to 1.5.0, unused since 1.6.0. Removed so
# existing installs reclaim the space on their next start.
SUPERSEDED_FILES = [
    "encoder-model.int8.onnx",        # 652 MB — the INT8 pipeline is never built
    "decoder_joint-model.int8.onnx",  # 18 MB — same
    "nemo128.onnx",                   # the mel preprocessor runs in NumPy
]


def download(model_dir: Path, filenames: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    model_dir.mkdir(parents=True, exist_ok=True)
    for fname in filenames:
        target = model_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"[bootstrap] Downloading {fname} from {HF_REPO} ...", flush=True)
        t0 = time.perf_counter()
        hf_hub_download(
            repo_id=HF_REPO,
            filename=fname,
            revision=HF_REVISION,
            local_dir=str(model_dir),
        )
        print(
            f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
            f"({target.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )


def drop(model_dir: Path, filenames: list[str], reason: str) -> None:
    """Delete model files that are no longer needed, reporting what was freed."""
    freed = 0
    for fname in filenames:
        target = model_dir / fname
        if target.exists():
            freed += target.stat().st_size
            target.unlink()
    if freed:
        print(f"[bootstrap] Freed {freed / 1e6:.0f} MB — {reason}", flush=True)


def _fetch_resumable(url: str, tmp: Path, attempts: int = 3) -> None:
    """Download to ``tmp``, resuming a partial file via HTTP Range."""
    import urllib.request

    for attempt in range(1, attempts + 1):
        pos = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if pos:
            req.add_header("Range", f"bytes={pos}-")
            print(f"[bootstrap]   resuming from {pos / 1e6:.1f} MB", flush=True)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if pos and resp.status != 206:
                    pos = 0  # server ignored Range; start over
                with open(tmp, "ab" if pos else "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            return
        except Exception as exc:  # noqa: BLE001 — retried below
            if attempt == attempts:
                raise
            print(f"[bootstrap]   download interrupted ({exc}); retrying "
                  f"({attempt}/{attempts})", file=sys.stderr)
            time.sleep(5)


def download_prebuilt_blobs(cache_dir: Path, wanted: list[int], device: str) -> None:
    """Fetch prebuilt NPU blobs from the manifest so no on-device compile
    (with its multi-GB memory peak) is ever needed for the listed buckets."""
    import hashlib
    import json

    if not BLOB_MANIFEST.exists():
        return
    names = {buckets.blob_name(T, device) for T in wanted}
    for entry in json.loads(BLOB_MANIFEST.read_text()).get("encoder_blobs", []):
        fname = entry["file"]
        if fname not in names:
            continue
        target = cache_dir / fname
        if target.exists() and target.stat().st_size > 0:
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        print(f"[bootstrap] Downloading precompiled NPU blob {fname} ...", flush=True)
        try:
            t0 = time.perf_counter()
            _fetch_resumable(entry["url"], tmp)
            digest = hashlib.sha256()
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                print(f"[bootstrap]   checksum mismatch for {fname}; "
                      "discarding (will compile on device instead)", file=sys.stderr)
                tmp.unlink()
                continue
            tmp.rename(target)
            print(
                f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
                f"({target.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — blob prefetch is best-effort
            # Keep the .part file: the next start resumes where this one left off.
            print(f"[bootstrap]   blob download failed: {exc} "
                  "(will retry/resume on next start, or compile on device)",
                  file=sys.stderr)


def build_static_decoder_ir(model_dir: Path, out_dir: Path) -> None:
    """Static-reshape the FP32 decoder/joint to fixed shapes and save as OV IR."""
    out_xml = out_dir / "decoder-static.xml"
    out_bin = out_dir / "decoder-static.bin"

    def ir_complete() -> bool:
        # Both halves, or the IR is unusable and the ONNX source must stay.
        return all(p.exists() and p.stat().st_size > 0 for p in (out_xml, out_bin))

    if ir_complete():
        drop(model_dir, DECODER_SOURCE, "static decoder IR already built")
        return

    download(model_dir, DECODER_SOURCE)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[bootstrap] Building static decoder IR ...", flush=True)
    import openvino as ov  # imported here so docker layer ordering doesn't matter

    src = model_dir / DECODER_SOURCE[0]
    if not src.exists():
        print(f"[bootstrap] FATAL: {src} not found", file=sys.stderr)
        sys.exit(1)

    core = ov.Core()
    model = core.read_model(str(src))
    # Static shapes match the per-token decoder loop exactly: one frame of
    # encoder output, one previous token, two LSTM states.
    model.reshape({
        "encoder_outputs": [1, 1024, 1],
        "targets": [1, 1],
        "target_length": [1],
        "input_states_1": [2, 1, 640],
        "input_states_2": [2, 1, 640],
    })
    ov.save_model(model, str(out_xml))
    print(f"[bootstrap]   saved {out_xml}", flush=True)

    # Only now is the 72 MB ONNX source expendable.
    if ir_complete():
        drop(model_dir, DECODER_SOURCE, "static decoder IR built")


def sync_encoder_source(model_dir: Path, cache_dir: Path,
                        wanted: list[int], device: str) -> None:
    """Keep the FP32 encoder graph only while a bucket still needs compiling."""
    to_compile = buckets.missing_blobs(str(cache_dir), device, wanted)
    if to_compile:
        print(f"[bootstrap] Buckets without a precompiled blob: {to_compile}; "
              "fetching the FP32 encoder graph to compile them on device",
              flush=True)
        download(model_dir, ENCODER_SOURCE)
    else:
        drop(model_dir, ENCODER_SOURCE,
             "every configured bucket has a precompiled blob")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/data"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = data_dir / MODEL_NAME
    cache_dir = data_dir / "ov_cache"

    device = os.environ.get("DEVICE", "NPU")
    wanted = buckets.configured_frames(os.environ)
    print(f"[bootstrap] device={device} buckets(frames)={sorted(set(wanted))}",
          flush=True)

    download(model_dir, BASE_FILES)
    build_static_decoder_ir(model_dir, data_dir / "static_decoder")
    download_prebuilt_blobs(cache_dir, wanted, device)
    sync_encoder_source(model_dir, cache_dir, wanted, device)
    drop(model_dir, SUPERSEDED_FILES, "unused since add-on 1.6.0")
    print("[bootstrap] All assets ready.", flush=True)


if __name__ == "__main__":
    main()
