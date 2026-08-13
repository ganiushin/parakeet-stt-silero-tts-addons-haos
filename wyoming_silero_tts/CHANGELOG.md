# Changelog

## 1.2.0

- **29 new voices: the Russian speakers of Silero's multilingual
  `v5_cis_base`, alongside the five of `v5_5_ru`** — 34 in total, and the
  default is now `ru_zhadyra`. Every voice in `v5_cis_base` was recorded by a
  native speaker of one of its languages, and the `ru_` ones are those same
  people reading Russian; the voice list names the speaker's language, since
  that is where the light accent comes from. Both packages stay loaded, so
  switching between the two families is a choice in the UI, not a rebuild.
- `v5_cis_base` carries no stress model of its own and mispronounces plain
  text, so text for its voices is stressed first with `v5_5_ru`'s stress and
  homograph model — the same one `v5_5_ru` applies internally for its own
  five. Nothing changes for those five.
- Keeping both packages resident costs ~300 MB: ~950 MB once voices from
  both have been used, against ~653 MB for 1.1.0's `v5_5_ru` alone (same
  server, same ten requests, two runs each). Serving only the `ru_` voices
  and dropping `v5_5_ru`'s five would instead *save* ~70 MB (~585 MB) — an
  option once a favourite voice is settled on. Disk grows to ~250 MB, since
  both packages are downloaded either way.

## 1.1.0

- **Measurements are now read out with correct Russian grammar.** Assist
  announces temperatures and percentages constantly, and both were wrong:
  - Fractions are proper fractions: "21,5°C" → *двадцать одна целая пять
    десятых градуса* (was *двадцать один и пять градусов*).
  - Units agree with the count: *один процент* / *два процента* / *пять
    процентов* (was *процентов* always), and a fractional quantity takes the
    genitive singular.
  - Numerals agree in gender with the noun that follows for the units Assist
    emits: *одна минута*, *две минуты*, *одна тысяча* (was *один минута*).
  - Decimals are parsed as decimals rather than floats, so "21,50" stays
    *пятьдесят сотых* and binary rounding can never leak into speech.
- Synthesized audio is converted to PCM without two intermediate float copies
  (~11 MB each per minute of 48 kHz speech).

## 1.0.0

- Initial release: Silero `v5_5_ru` (5 Russian voices) over the Wyoming
  protocol, with streaming synthesis (sentence-by-sentence playback).
- Text normalization: numbers, times and decimals are expanded to Russian
  words; Latin words are transliterated to Cyrillic (both would otherwise be
  silently dropped by the model).
- Model download is SHA-256 verified and resumable; torch is installed from
  the official CPU wheel index, all versions pinned.
