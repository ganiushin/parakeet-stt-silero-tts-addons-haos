"""Wyoming protocol event handler for Silero TTS.

Supports both classic one-shot Synthesize and streaming synthesis
(SynthesizeStart/Chunk/Stop, wyoming >= 1.6): incoming text is split into
sentences and each sentence is synthesized and streamed as soon as it is
complete, so long LLM answers start playing before they finish generating.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional

import torch
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .normalize import normalize

_LOGGER = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r'[.!?…]+["»)\]]*\s+')
# A sentence VITS can chew comfortably; longer runs are cut at whitespace.
MAX_SENTENCE_CHARS = 400
# ~100 ms of audio per AudioChunk event.
CHUNK_SECONDS = 0.1


class SentenceSplitter:
    """Incremental sentence splitter for streamed text chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    def add(self, text: str) -> List[str]:
        """Absorb a text chunk, return any sentences it completed."""
        self._buffer += text
        sentences = []
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if match is not None:
                sentences.append(self._buffer[: match.end()].strip())
                self._buffer = self._buffer[match.end():]
                continue
            # Punctuation-free runaway (URLs, rambling LLM output): cut at
            # the last whitespace before the cap so memory stays bounded.
            if len(self._buffer) > MAX_SENTENCE_CHARS:
                cut = self._buffer.rfind(" ", 0, MAX_SENTENCE_CHARS)
                if cut <= 0:
                    cut = MAX_SENTENCE_CHARS
                sentences.append(self._buffer[:cut].strip())
                self._buffer = self._buffer[cut:]
                continue
            return sentences

    def flush(self) -> str:
        remainder, self._buffer = self._buffer.strip(), ""
        return remainder


class SileroEventHandler(AsyncEventHandler):
    """Wyoming event handler for Silero v5 Russian TTS."""

    def __init__(
        self,
        wyoming_info: Info,
        model,
        model_lock: asyncio.Lock,
        *args,
        accentor,
        voice: str,
        speakers: List[str],
        sample_rate: int = 48000,
        transliterate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        self.model = model
        self.speakers = speakers
        self.accentor = accentor
        self.model_lock = model_lock
        self.default_voice = voice
        self.sample_rate = sample_rate
        self.transliterate = transliterate

        self._streaming = False
        self._splitter = SentenceSplitter()
        self._request_voice: Optional[str] = None
        self._audio_started = False

    def _resolve_voice(self) -> str:
        requested = self._request_voice
        if requested and requested in self.speakers:
            return requested
        if requested:
            _LOGGER.warning("Unknown voice %r; using %s", requested, self.default_voice)
        return self.default_voice

    async def _speak(self, sentence: str) -> None:
        """Synthesize one sentence and stream its PCM to the client."""
        text = normalize(sentence, transliterate=self.transliterate)
        if not text:
            return
        voice = self._resolve_voice()
        loop = asyncio.get_running_loop()
        async with self.model_lock:
            try:
                # The accentor runs under the same lock and in the same
                # executor call: it is one more torch model, and the voices
                # need the stress marks it adds ("к+ошка") to pronounce
                # anything correctly.
                audio = await loop.run_in_executor(
                    None,
                    lambda: self.model.apply_tts(
                        text=self.accentor(text), speaker=voice,
                        sample_rate=self.sample_rate,
                    ),
                )
            except Exception:
                # Silero raises bare ValueError on unspeakable leftovers;
                # skip the sentence rather than kill the whole response.
                _LOGGER.exception("Synthesis failed for %r", text)
                return

        # clamp() already copies, so scale in place on that copy rather than
        # allocating a second one — ~11 MB per minute of 48 kHz speech. The
        # tensor Silero returns is a view into its own output and is left
        # untouched.
        pcm = torch.clamp(audio, -1.0, 1.0).mul_(32767).to(torch.int16).numpy().tobytes()
        if not self._audio_started:
            await self.write_event(
                AudioStart(rate=self.sample_rate, width=2, channels=1).event()
            )
            self._audio_started = True

        step = int(self.sample_rate * CHUNK_SECONDS) * 2
        for offset in range(0, len(pcm), step):
            await self.write_event(
                AudioChunk(
                    audio=pcm[offset:offset + step],
                    rate=self.sample_rate,
                    width=2,
                    channels=1,
                ).event()
            )

    async def _finish_audio(self) -> None:
        """Close the audio stream; open it first if nothing was speakable."""
        if not self._audio_started:
            await self.write_event(
                AudioStart(rate=self.sample_rate, width=2, channels=1).event()
            )
        await self.write_event(AudioStop().event())
        self._audio_started = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._streaming = True
            self._splitter = SentenceSplitter()
            self._request_voice = start.voice.name if start.voice else None
            self._audio_started = False
            return True

        if SynthesizeChunk.is_type(event.type):
            chunk = SynthesizeChunk.from_event(event)
            for sentence in self._splitter.add(chunk.text):
                await self._speak(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            remainder = self._splitter.flush()
            if remainder:
                await self._speak(remainder)
            await self._finish_audio()
            await self.write_event(SynthesizeStopped().event())
            self._streaming = False
            return True

        if Synthesize.is_type(event.type):
            if self._streaming:
                # Streaming clients also send the assembled Synthesize for
                # backward compatibility; it was already spoken chunk by chunk.
                return True
            synthesize = Synthesize.from_event(event)
            self._request_voice = (
                synthesize.voice.name if synthesize.voice else None
            )
            self._audio_started = False
            splitter = SentenceSplitter()
            for sentence in splitter.add(synthesize.text):
                await self._speak(sentence)
            remainder = splitter.flush()
            if remainder:
                await self._speak(remainder)
            await self._finish_audio()
            return True

        return True
