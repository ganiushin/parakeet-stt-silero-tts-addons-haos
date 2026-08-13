# Wyoming Silero TTS

Silero v5 Russian text-to-speech served over the Wyoming protocol — a
natural-sounding replacement for Piper's Russian voices in Assist pipelines.

- 34 Russian voices — 29 from Silero's multilingual `v5_cis_base`, 5 from
  the Russian-only `v5_5_ru`, whose stress and homograph model both use
- Streaming synthesis: long answers start playing sentence by sentence
- Runs on CPU far faster than real time; no GPU or NPU needed
- Expands numbers to words and transliterates Latin — the Silero model would
  otherwise silently skip them
- Auto-discovered by the Wyoming Protocol integration

Configuration and troubleshooting: see [DOCS.md](./DOCS.md).
Project overview and provenance of all downloads: see the
[repository README](https://github.com/ganiushin/parakeet-stt-silero-tts-addons-haos#readme).
