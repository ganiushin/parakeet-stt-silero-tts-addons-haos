"""Bootstrap helper: download the Silero TTS model packages.

Run once on first start (from docker-entrypoint.sh) to populate /data.
Idempotent — a present, size-nonzero model file is trusted (it was SHA-256
verified when first downloaded; re-hashing 237 MB on every start is wasted
startup time on the low-power boxes this add-on targets).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

BASE_URL = "https://models.silero.ai/models/tts/ru/"
# Digests of the upstream packages are pinned so a silently re-published model
# can't slip in.
MODELS = (
    # The 29 ru_ voices (2026-08-13).
    ("v5_cis_base.pt",
     "ba41b18f6a707ad93605a162998865e7c087153d2e010a26dd02229dab0e672a"),
    # Five more voices, plus the Russian stress/homograph model that
    # v5_cis_base does not ship (2026-07-05).
    ("v5_5_ru.pt",
     "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437"),
)


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


def download_model(model_dir: Path, name: str, sha256: str) -> None:
    target = model_dir / name
    if target.exists() and target.stat().st_size > 0:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    # Two full tries: a checksum mismatch discards the file and re-downloads
    # once; a second mismatch is a hard error (the model is mandatory).
    for attempt in (1, 2):
        print(f"[bootstrap] Downloading {name} ...", flush=True)
        t0 = time.perf_counter()
        _fetch_resumable(BASE_URL + name, tmp)
        digest = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() == sha256:
            tmp.rename(target)
            print(
                f"[bootstrap]   done in {time.perf_counter()-t0:.1f}s "
                f"({target.stat().st_size / 1e6:.1f} MB)",
                flush=True,
            )
            return
        print(f"[bootstrap]   checksum mismatch for {name}; "
              f"discarding (attempt {attempt}/2)", file=sys.stderr)
        tmp.unlink()

    print(f"[bootstrap] FATAL: {name} failed checksum verification twice",
          file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/data"))
    args = ap.parse_args()

    model_dir = Path(args.data_dir) / "silero"
    for name, sha256 in MODELS:
        download_model(model_dir, name, sha256)
    print("[bootstrap] All assets ready.", flush=True)


if __name__ == "__main__":
    main()
