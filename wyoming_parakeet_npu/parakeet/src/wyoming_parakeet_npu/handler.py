"""Wyoming protocol event handler.

Adapted with thanks from https://github.com/tboby/wyoming-onnx-asr (MIT).
Buffers AudioChunk events in memory; on AudioStop, runs ASR. Audio longer
than the encoder bucket is transcribed in bucket-sized windows split at
low-energy (quiet) points and the texts are stitched together.
"""
import io
import logging
import wave
from typing import List, Optional

import numpy as np
import soundfile as sf
from onnx_asr.adapters import AsrAdapter
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Hard cap on buffered audio, a guard against runaway clients. 60 s of
# 16 kHz 16-bit mono is ~1.9 MB, so the cap is about latency, not memory.
MAX_AUDIO_SECONDS = 60.0


def split_at_low_energy(
    waveform: np.ndarray,
    rate: int,
    window_seconds: float,
    search_seconds: float = 1.5,
    frame_seconds: float = 0.1,
) -> List[np.ndarray]:
    """Split audio into windows of at most ``window_seconds``.

    Each cut is placed at the quietest 100 ms frame within the last
    ``search_seconds`` of the window, so words are not sliced mid-syllable.
    """
    window = int(window_seconds * rate)
    if len(waveform) <= window:
        return [waveform]

    segments = []
    start = 0
    while len(waveform) - start > window:
        lo = max(start + window - int(search_seconds * rate), start + 1)
        hi = start + window
        frame = max(1, int(frame_seconds * rate))
        best_cut, best_energy = hi, None
        for f0 in range(lo, hi - frame + 1, frame):
            energy = float(np.mean(waveform[f0:f0 + frame] ** 2))
            if best_energy is None or energy < best_energy:
                best_energy, best_cut = energy, f0 + frame // 2
        segments.append(waveform[start:best_cut])
        start = best_cut
    segments.append(waveform[start:])
    return segments


class ParakeetEventHandler(AsyncEventHandler):
    """Wyoming event handler for Parakeet TDT 0.6B v3 multilingual."""

    def __init__(
        self,
        wyoming_info: Info,
        model: AsrAdapter,
        model_lock,
        *args,
        default_language: str = "ru",
        window_seconds: float = 10.0,
        force_language: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.model_lock = model_lock
        self.default_language = default_language
        self.window_seconds = window_seconds
        self.force_language = force_language
        self.request_language: Optional[str] = None
        self._audio = bytearray()
        self._rate: Optional[int] = None
        self._width = 2
        self._channels = 1
        self._overflow_logged = False

    def _decode_audio(self) -> np.ndarray:
        if self._width == 2:
            waveform = np.frombuffer(bytes(self._audio), dtype=np.int16)
            waveform = waveform.astype(np.float32) / 32768.0
        else:
            # Uncommon sample widths go through a WAV round-trip in memory.
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setframerate(self._rate)
                w.setsampwidth(self._width)
                w.setnchannels(self._channels)
                w.writeframes(bytes(self._audio))
            buf.seek(0)
            waveform, _ = sf.read(buf, dtype="float32")
        if self._channels > 1:
            waveform = waveform.reshape(-1, self._channels).mean(axis=1)
        return waveform

    async def _transcribe(self, waveform: np.ndarray, lang: str) -> str:
        segments = split_at_low_energy(waveform, self._rate, self.window_seconds)
        if len(segments) > 1:
            _LOGGER.debug("Long audio: transcribing %d windows", len(segments))
        texts = []
        async with self.model_lock:
            if self.force_language:
                # Parakeet TDT ignores onnx_asr's language argument; the
                # decoder shim enforces the language via a script logit mask.
                decoder = getattr(getattr(self.model, "asr", None),
                                  "_decoder_joint", None)
                if hasattr(decoder, "set_language"):
                    decoder.set_language(lang)
            for segment in segments:
                text = self.model.recognize(
                    segment, language=lang, sample_rate=self._rate
                )
                if text and text.strip():
                    texts.append(text.strip())
        return " ".join(texts)

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            if self._rate is None:
                self._rate = chunk.rate
                self._width = chunk.width
                self._channels = chunk.channels
            limit = int(MAX_AUDIO_SECONDS * self._rate * self._width * self._channels)
            if len(self._audio) < limit:
                self._audio.extend(chunk.audio)
            elif not self._overflow_logged:
                _LOGGER.warning(
                    "Audio exceeds %.0f s; further input ignored", MAX_AUDIO_SECONDS
                )
                self._overflow_logged = True
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped. Transcribing.")
            text = ""
            if self._audio and self._rate:
                lang = self.request_language or self.default_language
                try:
                    text = await self._transcribe(self._decode_audio(), lang)
                except Exception:
                    _LOGGER.exception("Recognition failed")
                    text = ""
                _LOGGER.info("%s: %s", lang, text)
            else:
                _LOGGER.warning("AudioStop without audio; returning empty transcript")
            await self.write_event(Transcript(text=text).event())
            # False ends wyoming's read loop and disconnects, so this handler
            # instance is done — one connection carries one utterance.
            return False

        if Transcribe.is_type(event.type):
            self.request_language = Transcribe.from_event(event).language
            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        return True
