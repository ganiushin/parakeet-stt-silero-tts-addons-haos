"""Wyoming server for Silero v5 Russian text-to-speech on CPU.

Loads the torch.package model downloaded by scripts/bootstrap.py and serves
it over the Wyoming protocol with streaming synthesis support.

Designed for one specific use case: natural Russian voices for Home
Assistant Assist pipelines on modest x86/ARM CPUs. No model selection — just
Silero v5: the Russian half of v5_cis_base for the voices, plus silero-stress
for the stress marks they need.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from functools import partial
from importlib import resources
from typing import List

import torch
# Importing silero_stress calls torch.set_num_threads(1); keep it up here so
# that runs before main() sets the configured thread count, not after.
from silero_stress import load_accentor as _silero_load_accentor
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .handler import SileroEventHandler

_LOGGER = logging.getLogger(__name__)

MODEL_FILE = "v5_cis_base.pt"

# Rows of the stress model's embedding table converted per block; 4096 rows is
# ~5 MB of fp32, small enough to disappear into the allocator's reuse.
DEQUANT_ROWS = 4096

# v5_cis_base is a multilingual model; every voice was recorded by a native
# speaker of one of its languages, and the `ru_` ones are those same people
# reading Russian. Whose voice it is explains the accent you hear, so that is
# what the voice list shows.
_VOICE_ORIGINS = {
    "gamat": "Azerbaijani",
    "zara": "Armenian",
    "aigul": "Bashkir", "alfia": "Bashkir", "alfia2": "Bashkir",
    "miyau": "Bashkir", "ramilia": "Bashkir",
    "dmitriy": "Belarusian",
    "vika": "Georgian",
    "eduard": "Kabardian",
    "zhadyra": "Kazakh", "zhazira": "Kazakh",
    "kejilgan": "Kalmyk", "kermen": "Kalmyk",
    "nurgul": "Kyrgyz",
    "oksana": "Moksha",
    "onaoy": "Tajik", "safarhuja": "Tajik",
    "albina": "Tatar", "marat": "Tatar",
    "bogdan": "Udmurt",
    "saida": "Uzbek",
    "igor": "Ukrainian", "roman": "Ukrainian",
    "karina": "Khakas", "sibday": "Khakas",
    "ekaterina": "Chuvash",
    "alexandr": "Erzya",
    "zinaida": "Yakut",
}


def _voice_description(speaker: str) -> str | None:
    origin = _VOICE_ORIGINS.get(speaker.removeprefix("ru_"))
    return f"Russian, {origin} speaker" if origin else None


def _load_accentor():
    """Load silero-stress without the memory spike its own loader leaves.

    The stress model ships with its homograph BERT's embedding table packed
    as int8. silero_stress.load_accentor() unpacks it in a single expression,
    `scale * (weight.clone() - zero_point)`, which allocates three ~105 MB
    temporaries to produce a ~105 MB result; the allocator then keeps that
    high-water mark for the life of the process — ~120 MB of resident memory
    that is never used again.

    Converting the table a block at a time gives bit-identical weights for
    one ~5 MB temporary. Identical was checked, not assumed: over 9,128
    sentences covering all 2,208 homographs, this accentor and the library's
    produced the same output every time.

    The unpacking reaches into silero-stress's internals, so it is pinned to
    an exact version in pyproject.toml and falls back to the library's own
    loader — costing only the memory — if the model is not shaped as expected.
    """
    try:
        path = resources.files("silero_stress.data").joinpath("accentor.pt")
        with open(path, "rb") as f:
            accentor = torch.package.PackageImporter(f).load_pickle(
                "accentor_models", "accentor")
        bert = accentor.homosolver.model.bert
        packed = bert.embeddings.word_embeddings.weight
        if packed.dtype is not torch.int8:
            raise TypeError(f"expected an int8 table, found {packed.dtype}")
        with torch.no_grad():
            table = torch.empty(packed.shape, dtype=torch.float32)
            for row in range(0, packed.shape[0], DEQUANT_ROWS):
                block = packed.data[row:row + DEQUANT_ROWS].to(torch.float32)
                table[row:row + DEQUANT_ROWS] = block.sub_(bert.zero_point).mul_(bert.scale)
            bert.embeddings.word_embeddings.weight.data = table
        return accentor
    except Exception:
        _LOGGER.warning(
            "silero-stress is not shaped as expected; loading it the library's "
            "own way, which costs ~120 MB more resident memory", exc_info=True)
        return _silero_load_accentor()


def _build_wyoming_info(speakers: List[str]) -> Info:
    silero_attribution = Attribution(
        name="Silero (snakers4/silero-models)",
        url="https://github.com/snakers4/silero-models",
    )
    return Info(tts=[TtsProgram(
        name="silero",
        description="Silero v5 Russian text-to-speech on CPU",
        attribution=Attribution(
            name="ganiushin/parakeet-stt-silero-tts-addons-haos",
            url="https://github.com/ganiushin/parakeet-stt-silero-tts-addons-haos",
        ),
        installed=True,
        version=__version__,
        supports_synthesize_streaming=True,
        voices=[
            TtsVoice(
                name=speaker,
                description=_voice_description(speaker),
                attribution=silero_attribution,
                installed=True,
                version="v5-cis",
                languages=["ru"],
            )
            for speaker in speakers
        ],
    )])


def _env_flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wyoming-silero-tts",
        description="Wyoming TTS server for Silero v5 Russian voices",
    )
    parser.add_argument(
        "--uri",
        default=os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10200"),
        help="Wyoming server URI (default: tcp://0.0.0.0:10200)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", "/data"),
        help="Directory holding the model files (default: /data)",
    )
    parser.add_argument(
        "--voice",
        default=os.environ.get("VOICE", "ru_zhadyra"),
        help="Default speaker when the client does not specify one",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.environ.get("SAMPLE_RATE", "48000")),
        choices=[8000, 24000, 48000],
        help="Output sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("THREADS", "2")),
        help="Torch CPU threads for synthesis (default: 2)",
    )
    parser.add_argument(
        "--no-transliterate",
        action="store_true",
        default=not _env_flag("TRANSLITERATE"),
        help="Do not transliterate Latin words to Cyrillic",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG-level logging")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _LOGGER.info("wyoming-silero-tts v%s", __version__)
    torch.set_num_threads(max(1, args.threads))

    model_dir = os.path.join(args.data_dir, "silero")
    model_path = os.path.join(model_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        _LOGGER.error(
            "Model not found at %s. The entrypoint should fetch it on "
            "first run; if you bypassed the entrypoint, run "
            "scripts/bootstrap.py manually.",
            model_path,
        )
        sys.exit(1)

    # v5_cis_base carries no stress model: it wants every word already
    # stressed ("к+ошка") and guesses badly otherwise. silero-stress ships its
    # weights inside the wheel, so this reads from the image, not from /data.
    _LOGGER.info("Loading Russian stress model (silero-stress) ...")
    t0 = time.perf_counter()
    accentor = _load_accentor()
    _LOGGER.info("Stress model loaded in %.1f s", time.perf_counter() - t0)

    _LOGGER.info("Loading Silero model from %s ...", model_path)
    t0 = time.perf_counter()
    importer = torch.package.PackageImporter(model_path)
    model = importer.load_pickle("tts_models", "model")
    model.to("cpu")
    # The package is multilingual; only the Russian voices are offered.
    speakers = [s for s in model.speakers if s.startswith("ru_")]
    _LOGGER.info("Model loaded in %.1f s; Russian speakers: %s",
                 time.perf_counter() - t0, speakers)

    voice = args.voice
    if voice not in speakers:
        _LOGGER.warning("Voice %r not among the Russian speakers; using %s",
                        voice, speakers[0])
        voice = speakers[0]

    # First apply_tts call pays one-time lazy-init costs (~10x a normal
    # request); warm up now so the first real request is instant. Doubles as
    # a self-test that model and stress model work together.
    t0 = time.perf_counter()
    model.apply_tts(text=accentor("Голосовой сервер запущен."), speaker=voice,
                    sample_rate=args.sample_rate)
    _LOGGER.info("Warm-up synthesis took %.2f s", time.perf_counter() - t0)

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    info = _build_wyoming_info(speakers)
    _LOGGER.info("Ready. Listening on %s (voice=%s rate=%d)",
                 args.uri, voice, args.sample_rate)
    await server.run(partial(
        SileroEventHandler, info, model, model_lock,
        accentor=accentor,
        voice=voice,
        speakers=speakers,
        sample_rate=args.sample_rate,
        transliterate=not args.no_transliterate,
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
