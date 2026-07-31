"""Encoder bucket bookkeeping, shared by the server and the bootstrap script.

A bucket is one static-shape compiled encoder, identified by the number of mel
frames ``T`` it accepts. Its compiled blob is named after ``T`` and the
OpenVINO device, so three places have to agree on that mapping: the shim that
imports or compiles blobs, the sweeper that drops stale ones, and the
bootstrap step that decides whether the 2.5 GB FP32 encoder graph is still
needed. Keeping the mapping here is what makes that last decision safe.
"""
from __future__ import annotations

import os
from typing import Iterable

# The NeMo mel preprocessor emits one frame per 10 ms of audio.
FRAMES_PER_SECOND = 100

# Defaults for the bucket configuration, in seconds. One 10 s eager bucket is
# the size the add-on ships a prebuilt blob for; keep these in step with
# config.yaml, or a default install would download 2.5 GB it does not need.
DEFAULT_EAGER = "10"
DEFAULT_LAZY = ""


def parse_seconds(spec: str | None) -> list[float]:
    """Parse a comma-separated bucket spec (``"5,20"``) into seconds."""
    return [float(x) for x in spec.split(",") if x.strip()] if spec else []


def frames(seconds: float) -> int:
    """Mel frames in a bucket of ``seconds`` seconds."""
    return int(seconds * FRAMES_PER_SECOND)


def blob_name(T: int, device: str) -> str:
    return f"encoder_T{T}_{device}.blob"


def blob_path(cache_dir: str, T: int, device: str) -> str:
    return os.path.join(cache_dir, blob_name(T, device))


def has_blob(cache_dir: str, T: int, device: str) -> bool:
    path = blob_path(cache_dir, T, device)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def missing_blobs(cache_dir: str, device: str, frame_counts: Iterable[int]) -> list[int]:
    """Bucket sizes that would have to be compiled from the FP32 ONNX graph."""
    return sorted(T for T in set(frame_counts) if not has_blob(cache_dir, T, device))


def configured_frames(env: dict) -> list[int]:
    """Frame counts for the buckets the current configuration asks for.

    Reads the same environment variables the server's command line defaults
    to, so the bootstrap step and the server can never disagree about which
    blobs are wanted — that agreement is what lets bootstrap delete the FP32
    encoder graph.
    """
    seconds = parse_seconds(env.get("ENCODER_BUCKETS", DEFAULT_EAGER))
    seconds += parse_seconds(env.get("ENCODER_LAZY_BUCKETS", DEFAULT_LAZY))
    return [frames(s) for s in seconds]
