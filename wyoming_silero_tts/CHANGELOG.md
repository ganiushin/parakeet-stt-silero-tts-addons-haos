# Changelog

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
