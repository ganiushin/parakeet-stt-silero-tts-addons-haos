"""Wyoming server for Parakeet TDT 0.6B v3 multilingual on Intel NPU.

Builds OpenVINO NPU-compiled shims for the encoder and decoder/joint, then
assembles the onnx-asr pipeline around them (see pipeline.py — no ORT session
for either graph is ever opened). The encoder uses multi-bucket dispatch, one
compiled blob per audio length picked at request time, with optional lazy
loading for large buckets.

Designed for one specific use case: smart-home / dictation STT on Intel
Core Ultra CPUs with the AI Boost NPU. No model selection, no quantization
selection — just Parakeet TDT 0.6B v3.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial

import openvino as ov
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import MODEL_NAME, __version__, buckets, pipeline
from .handler import ParakeetEventHandler
from .shims import OpenVinoDecoderShim, OpenVinoEncoderShim

_LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = (
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk",
    "bg", "hr", "cs", "da", "et", "fi", "el", "hu", "lv", "lt",
    "mt", "ro", "sk", "sl", "sv",
)


def _build_wyoming_info(default_language: str) -> Info:
    return Info(asr=[AsrProgram(
        name="parakeet-npu",
        description="Parakeet TDT 0.6B v3 multilingual on Intel NPU (OpenVINO)",
        attribution=Attribution(
            name="cibernox/wyoming-parakeet-on-intel-npu",
            url="https://github.com/cibernox/wyoming-parakeet-on-intel-npu",
        ),
        installed=True,
        version=__version__,
        models=[AsrModel(
            name=MODEL_NAME,
            description=f"Multilingual ASR (default: {default_language})",
            attribution=Attribution(
                name="NVIDIA NeMo (model) + onnx-asr (pipeline)",
                url="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx",
            ),
            installed=True,
            languages=list(SUPPORTED_LANGUAGES),
            version="0.1",
        )],
    )])


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wyoming-parakeet-npu",
        description="Wyoming STT server for Parakeet TDT on Intel NPU",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10300"),
        help="Wyoming server URI (default: tcp://0.0.0.0:10300)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "/data"),
        help="Directory holding the model files and OpenVINO cache (default: /data)",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("LANGUAGE", "ru"),
        choices=SUPPORTED_LANGUAGES,
        help="Default transcription language when the client does not specify one",
    )
    parser.add_argument(
        "--force-language",
        default=os.environ.get("FORCE_LANGUAGE", "true"),
        choices=["true", "false"],
        help="Restrict decoding to the transcription language's alphabet "
             "(Parakeet TDT auto-detects the language per utterance and "
             "otherwise sometimes drifts into the wrong one; default: true)",
    )
    parser.add_argument(
        "--encoder-buckets",
        default=os.environ.get("ENCODER_BUCKETS", buckets.DEFAULT_EAGER),
        help=f"Comma-separated EAGER bucket sizes in seconds "
             f"(default: {buckets.DEFAULT_EAGER})",
    )
    parser.add_argument(
        "--encoder-lazy-buckets",
        default=os.environ.get("ENCODER_LAZY_BUCKETS", buckets.DEFAULT_LAZY),
        help="Comma-separated LAZY bucket sizes in seconds (default: none)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "NPU"),
        choices=["CPU", "NPU"],
        help="OpenVINO device for both encoder and decoder (default: NPU). "
             "GPU is not offered: the add-on does not map /dev/dri.",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("wyoming-parakeet-on-intel-npu v%s", __version__)
    available = ov.Core().available_devices
    _LOGGER.info("OpenVINO devices available: %s", available)
    if args.device not in available:
        _LOGGER.error(
            "Requested device %s not available (have %s). "
            "Make sure the NPU device is passed through to the container.",
            args.device, available,
        )
        sys.exit(1)

    model_dir = os.path.join(args.data_dir, MODEL_NAME)
    encoder_onnx = os.path.join(model_dir, "encoder-model.onnx")
    decoder_ir = os.path.join(args.data_dir, "static_decoder", "decoder-static.xml")
    cache_dir = os.path.join(args.data_dir, "ov_cache")

    eager = buckets.parse_seconds(args.encoder_buckets)
    lazy = buckets.parse_seconds(args.encoder_lazy_buckets)
    _LOGGER.info("Encoder buckets: eager=%s lazy=%s on %s", eager, lazy, args.device)

    required = [(os.path.join(model_dir, "config.json"), "model config"),
                (os.path.join(model_dir, "vocab.txt"), "vocabulary"),
                (decoder_ir, "static decoder IR")]
    # The 2.5 GB FP32 encoder graph is only kept on disk while some bucket
    # still has to be compiled on device; buckets covered by a precompiled
    # blob never touch it. Bootstrap decides this from the same helper.
    to_compile = buckets.missing_blobs(
        cache_dir, args.device, [buckets.frames(s) for s in eager + lazy]
    )
    if to_compile:
        _LOGGER.info("Buckets without a precompiled blob: %s", to_compile)
        required.append((encoder_onnx, "FP32 encoder ONNX"))

    for path, label in required:
        if not os.path.exists(path):
            _LOGGER.error(
                "%s not found at %s. The entrypoint should fetch it on first run; "
                "if you bypassed the entrypoint, run scripts/bootstrap.py manually.",
                label, path,
            )
            sys.exit(1)

    # 1. Compile / import the NPU shims first, so the pipeline can be built
    #    around them without onnx-asr ever opening an ORT session.
    encoder = OpenVinoEncoderShim(
        onnx_path=encoder_onnx,
        device=args.device,
        cache_dir=cache_dir,
        eager_seconds=eager,
        lazy_seconds=lazy,
    )
    force_language = args.force_language == "true"
    decoder_joint = OpenVinoDecoderShim(
        ir_path=decoder_ir,
        device=args.device,
        cache_dir=cache_dir,
        vocab_path=os.path.join(model_dir, "vocab.txt") if force_language else None,
    )

    # 2. Assemble the onnx-asr pipeline (vocab + NumPy mel preprocessor) on top.
    model = pipeline.build(model_dir, encoder, decoder_joint)

    if force_language:
        _LOGGER.info(
            "Language forcing ON: decoder locked to the script of the "
            "requested language (default: %s)", args.language,
        )

    # 3. Run the wyoming server.
    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    info = _build_wyoming_info(args.language)
    _LOGGER.info("Ready. Listening on %s", args.uri)
    await server.run(partial(
        ParakeetEventHandler, info, model, model_lock,
        default_language=args.language,
        window_seconds=max(eager + lazy),
        force_language=force_language,
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
