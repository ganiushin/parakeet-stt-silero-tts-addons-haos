# Changelog

## 1.6.0

- **Disk usage drops from ~4.5 GB to ~1.3 GB** for a default install, and
  upgrading reclaims the difference on the first start. The add-on now keeps
  only what it runs on: the INT8 encoder and decoder/joint (670 MB) are never
  downloaded at all, the FP32 decoder/joint (72 MB) is deleted once the static
  IR is built from it, and the 2.5 GB FP32 encoder graph is fetched only for
  bucket sizes that have no precompiled blob. Change `encoder_buckets` away
  from the default and the next start downloads it; change it back and the
  next start removes it again.
- **The onnx-asr pipeline is now assembled explicitly** around the NPU shims
  (`pipeline.py`) instead of being loaded for CPU and patched afterwards. No
  ONNX Runtime session is created any more — previously nine were: the INT8
  encoder (stubbed), the INT8 decoder/joint, and seven resamplers, each with
  its own thread pool sized to the whole CPU. The 16 kHz → 16 kHz resampler
  path is a no-op and is only built if a client ever sends another rate.
- **Correct encoder bucket dispatch.** The mel preprocessor pads the waveform
  by half a window on each side, so it emits exactly one frame more than the
  audio contains. Sizing the bucket by that width missed the matching bucket
  by one frame every time: audio of exactly 10.00 s fell through to the next
  bucket (or, with a single bucket configured, logged a spurious "truncating"
  warning on every full window). Dispatch now uses the true speech length.
- **A corrupt encoder blob no longer needs a 2.5 GB re-download**: it is
  removed on import failure so the next start re-fetches just the blob.
- Defaults for `encoder_buckets` / `language` now agree across `config.yaml`,
  `run.sh`, the Dockerfile and the server (10 s, `ru`).
- **The `GPU` device option is gone.** It could never work: the OpenVINO GPU
  plugin needs the host's `/dev/dri` mapped into the add-on, which this add-on
  does not do, so selecting it always failed the device check at startup.
  `NPU` and `CPU` remain.
- Dead code removed: two unused `_out_names` attributes in the shims, and the
  handler's `_reset()` (wyoming disconnects after `AudioStop`, so the handler
  instance never survives to be reused). The model directory name is now a
  single shared constant instead of a literal in two files, and the redundant
  `[hub]` extra was dropped from the `onnx-asr` pin — `huggingface_hub` is
  already a direct dependency, and onnx-asr's downloader is no longer used.

## 1.5.0

- **New option `force_language` (default on)**: locks decoding to the alphabet
  of the pipeline language. Parakeet TDT v3 ignores the requested language and
  auto-detects it per utterance, which occasionally turned Russian speech into
  English gibberish; the decoder shim now suppresses foreign-script token
  logits (`ru`/`uk`/`bg` → Cyrillic only, `el` → Greek only, others → Latin
  only). Digits, punctuation and special tokens are unaffected. Disable for
  mixed-language dictation.

## 1.4.0

- **Long audio is no longer truncated**: utterances longer than the bucket are
  transcribed in bucket-sized windows split at quiet points and stitched
  together (up to 60 s total).
- **Faster, lighter startup**: the throwaway ~650 MB INT8 encoder session is
  stubbed out during pipeline init (loads in under a second instead of tens;
  falls back to the plain loader automatically if anything goes wrong).
- **Handler robustness**: audio is buffered in memory instead of a temp WAV
  file; recognition errors return an empty transcript instead of an
  "ERROR: ..." pseudo-phrase; `AudioStop` without audio no longer crashes the
  connection; runaway audio input is capped at 60 s.
- **Blob download resumes** after network interruptions (HTTP Range, 3
  retries; a partial file survives restarts).
- **Supply chain fully pinned**: Hugging Face model revision (`8f23f0c`) and
  exact `onnx-asr==0.11.0`.

## 1.3.0

- Default configuration is now the recommended one: `encoder_buckets: "10"`
  (prebuilt blob, no on-device compile), `encoder_lazy_buckets` empty.
- Blob manifest trimmed to the single 10 s bucket; other bucket sizes compile
  on the device (one-time ~5 GB RAM peak, then cached).
- Documentation overhaul: explicit list of every external download with its
  verification, memory sizing guidance for Proxmox VMs.

## 1.2.1

- Blob import no longer buffers the whole 1.2 GB file in Python: the blob is
  mmap-fed to OpenVINO, roughly halving the import memory peak (fixes OOM
  restarts on 4 GB VMs).
- Note: an int8 encoder variant was evaluated and rejected — the NPU 3720
  compiler always materializes FP16 weights in the executable blob, so int8
  gives no memory savings on Meteor/Arrow Lake.

## 1.2.0

- Warm starts no longer read the 2.4 GB FP32 encoder: after a successful
  compile the NPU blob is exported to `/data/ov_cache/encoder_T<frames>_<device>.blob`
  and imported directly on later starts (~1.5 GB peak instead of ~3.5 GB).
- Prebuilt blobs listed in `blobs.json` (SHA-256 verified) are downloaded by
  the bootstrap, so listed bucket sizes never need an on-device compile at all.
  Prebuilt blobs are provided for 6, 7, 8, 9 and 10 second buckets
  (compiled offline for NPU 3720 — Meteor/Arrow Lake).
- Stale blobs for no-longer-configured buckets are cleaned up at startup.
- `openvino` pinned to 2026.2.1: blobs are only importable by the exact
  OpenVINO version that exported them.
- Intel NPU driver updated to v1.33.0 — same version the prebuilt blobs are
  compiled with; also ships the NPU compiler loader that OpenVINO 2026.2
  expects for ahead-of-time compilation.

## 1.1.2

- Pull the ubuntu:24.04 base image from mirror.gcr.io (Google's Docker Hub
  library mirror) instead of docker.io, which is unreachable or unstable on
  some networks.

## 1.1.1

- Remove the supervisor watchdog and Docker HEALTHCHECK: the first start
  downloads ~3.2 GB of models before port 10300 opens, so the probes could
  restart the add-on mid-download and caused a harmless but confusing
  "Timeout while waiting for app to start" warning.

## 1.1.0

- Image is now built entirely from source code vendored in this repository
  (`parakeet/`, upstream commit `238b52b`, MIT) — no prebuilt third-party
  images are pulled.
- Intel NPU driver and Level Zero loader downloads are pinned and verified
  by SHA-256 at build time.

## 1.0.0

- Initial release.
- Wraps `ghcr.io/cibernox/wyoming-parakeet-on-intel-npu:latest`.
- Options: `language`, `device`, `encoder_buckets`, `encoder_lazy_buckets`.
- Maps `/dev/accel/accel0` (Intel NPU) into the container.
- Registers Wyoming discovery with Home Assistant automatically.
