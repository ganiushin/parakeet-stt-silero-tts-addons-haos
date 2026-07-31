# Origin and attribution

This directory started as a copy of
[cibernox/wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu),
vendored into this add-on repository so the image is built entirely from
code kept here (no prebuilt third-party images).

- Forked from upstream commit: `238b52b0c47346295599417abe53ea02e2343f59` (2026-05-25)
- License: MIT (see [LICENSE](./LICENSE), © Miguel Camba)
- `src/wyoming_parakeet_npu/handler.py` additionally adapts code from
  [tboby/wyoming-onnx-asr](https://github.com/tboby/wyoming-onnx-asr) (MIT)

**It is no longer a straight vendored copy.** Since the fork the add-on has
diverged far enough that syncing by diffing upstream is not practical:

- `pipeline.py` — assembles onnx-asr's TDT model directly around the NPU
  shims, so no ONNX Runtime session is ever created (added here)
- `buckets.py` — bucket/blob bookkeeping shared with `scripts/bootstrap.py`
  (added here)
- `script_mask.py` — script-level language locking for the decoder (added here)
- `shims.py` — blob export/import with mmap-backed loading, multi-bucket
  dispatch, lazy buckets
- `handler.py` — windowed transcription of long audio, in-memory buffering
- `scripts/bootstrap.py` — prebuilt blob manifest, pinned model revision, and
  fetching only the model files a given configuration actually needs

Treat upstream as the origin and the licence holder, not as a branch to merge
from; changes here are made directly. Note that `pyproject.toml` still carries
upstream's `authors` and `project.urls` from the fork point.
