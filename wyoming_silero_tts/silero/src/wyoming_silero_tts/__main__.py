"""Wyoming server for Silero v5 Russian text-to-speech on CPU.

Loads the torch.package models downloaded by scripts/bootstrap.py and serves
them over the Wyoming protocol with streaming synthesis support.

Designed for one specific use case: natural Russian voices for Home
Assistant Assist pipelines on modest x86/ARM CPUs. Two Silero v5 packages
are loaded and their Russian voices offered as one list: the 29 `ru_`
speakers of the multilingual v5_cis_base, and the 5 of the Russian-only
v5_5_ru, which also provides the stress model the former lacks.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from functools import partial
from typing import Dict, NamedTuple

import torch
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .handler import SileroEventHandler, Voice

_LOGGER = logging.getLogger(__name__)

CIS_FILE = "v5_cis_base.pt"
RU_FILE = "v5_5_ru.pt"

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


# The five Russian-only voices, described the way Silero describes them.
_V5_5_DESCRIPTIONS = {
    "xenia": "Female, neutral",
    "baya": "Female, soft",
    "kseniya": "Female, bright",
    "aidar": "Male, neutral",
    "eugene": "Male, low",
}


def _voice_description(speaker: str) -> str | None:
    origin = _VOICE_ORIGINS.get(speaker.removeprefix("ru_"))
    if origin:
        return f"Russian, {origin} speaker"
    return _V5_5_DESCRIPTIONS.get(speaker)


def _load(path: str):
    model = torch.package.PackageImporter(path).load_pickle("tts_models", "model")
    model.to("cpu")
    return model


def _build_voices(cis, ru5) -> Dict[str, Voice]:
    """Map every offered speaker to the model that can say it.

    v5_cis_base is multilingual; only its Russian speakers are offered, since
    the add-on's normalization and stress model are Russian-only. It also has
    no accentor of its own and wants text that arrives already stressed —
    v5_5_ru stresses internally instead, exactly as it always has.
    """
    voices = {s: Voice(model=cis, version="v5-cis", accent=True)
              for s in cis.speakers if s.startswith("ru_")}
    voices.update({s: Voice(model=ru5, version="v5.5", accent=False)
                   for s in ru5.speakers})
    return voices


def _build_wyoming_info(voices: Dict[str, Voice]) -> Info:
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
                version=voice.version,
                languages=["ru"],
            )
            for speaker, voice in voices.items()
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
    paths = [os.path.join(model_dir, name) for name in (CIS_FILE, RU_FILE)]
    for path in paths:
        if not os.path.exists(path):
            _LOGGER.error(
                "Model not found at %s. The entrypoint should fetch it on "
                "first run; if you bypassed the entrypoint, run "
                "scripts/bootstrap.py manually.",
                path,
            )
            sys.exit(1)

    t0 = time.perf_counter()
    _LOGGER.info("Loading Silero models from %s ...", model_dir)
    cis, ru5 = (_load(path) for path in paths)
    accentor = ru5.packages[0].accentor
    voices = _build_voices(cis, ru5)
    _LOGGER.info("Models loaded in %.1f s; voices: %s",
                 time.perf_counter() - t0, list(voices))

    voice = args.voice
    if voice not in voices:
        default = next(iter(voices))
        _LOGGER.warning("Voice %r is not one of the offered voices; using %s",
                        voice, default)
        voice = default

    # The first apply_tts call on a model pays one-time lazy-init costs (~10x
    # a normal request), so warm both up now rather than make whoever
    # switches voices wait for it. Doubles as a self-test that the models and
    # the accentor work together.
    t0 = time.perf_counter()
    warmed = []
    for speaker in (voice, *voices):
        entry = voices[speaker]
        if any(entry.model is model for model in warmed):
            continue
        text = "Голосовой сервер запущен."
        entry.model.apply_tts(text=accentor(text) if entry.accent else text,
                              speaker=speaker, sample_rate=args.sample_rate)
        warmed.append(entry.model)
    _LOGGER.info("Warm-up synthesis of %d models took %.2f s",
                 len(warmed), time.perf_counter() - t0)

    server = AsyncServer.from_uri(args.uri)
    model_lock = asyncio.Lock()
    info = _build_wyoming_info(voices)
    _LOGGER.info("Ready. Listening on %s (voice=%s rate=%d)",
                 args.uri, voice, args.sample_rate)
    await server.run(partial(
        SileroEventHandler, info, voices, model_lock,
        accentor=accentor,
        voice=voice,
        sample_rate=args.sample_rate,
        transliterate=not args.no_transliterate,
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
