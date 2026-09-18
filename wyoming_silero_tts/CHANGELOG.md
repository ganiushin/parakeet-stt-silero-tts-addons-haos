# Changelog

## 1.4.0

- **Stress and homographs now come from `silero-stress` 1.5**, the library
  Silero split out of its TTS packages, instead of from `v5_5_ru`. Same job,
  same `+` marks, better result: 2,208 homographs against the old model's
  1,924, a ~4M-word stress dictionary, and predictions that no longer shift
  when the punctuation around a word changes — Assist sends plenty of that.
  «Я уже открыл замок в старом замке» is the short version: 1.3.0 said
  «+Я +уже откр+ыл з+амок в ст+аром з+амке», 1.4.0 says «+Я уж+е откр+ыл
  зам+ок в ст+аром з+амке».
- **It costs about 100 MB.** Measured on a dev machine, both engines driven
  through the same sequence, resident memory settles at ~591 MB on 1.3.0 and
  ~692 MB here. On a 4 GB Home Assistant VM the add-on's own RAM figure reads
  ~15% just after start and **~17% (~700 MB)** once it has served a few
  requests — torch allocates its synthesis buffers on first use, in both
  versions alike, and then stops. Plan for ~900 MB free.
- **~120 MB of that was avoidable, and is avoided.** The stress model's
  homograph BERT ships with its embedding table packed as int8;
  `silero_stress.load_accentor()` unpacks it as `scale * (weight.clone() -
  zero_point)`, three ~105 MB temporaries for a ~105 MB result, and the
  allocator keeps that high-water mark for the life of the process. The
  add-on unpacks the same table a block at a time instead — one ~5 MB
  temporary, and **bit-identical weights**: across 9,128 sentences covering
  all 2,208 homographs, this loader and the library's agreed every time.
  Loading straight through `silero-stress` would be ~786 MB rather than
  ~666 MB. Since this reaches into the library's internals, the version is
  pinned exactly and the add-on falls back to the library's own loader — at
  the old cost — if a future release is shaped differently.
- **The `v5_5_ru` download is gone.** It was 145 MB fetched purely to lift
  one object out of it, and `silero-stress` carries its own weights inside
  the wheel. First start now downloads **~92 MB instead of ~237 MB**, and the
  data directory holds the same; on upgrade the stale `v5_5_ru.pt` is deleted
  from `/data` on the next start. The image grows by ~67 MB in exchange, so
  the trade is roughly 145 MB of downloads and data for 67 MB of image.
- **The add-on is now permissively licensed end to end.** `v5_5_ru` was
  CC BY-NC-SA 4.0 — free for home use, but not for commercial deployments,
  and the one non-permissive piece left. `silero-stress` is **MIT**, like
  `v5_cis_base` and like this add-on.
- **Upgrading:** nothing to reconfigure — voices, options and pipelines are
  untouched. Check free memory first.

## 1.3.0

- **The five `v5_5_ru` voices are gone; the 29 `ru_` ones stay.** They were
  offered in 1.2.0 so the two families could be compared in the UI; that
  comparison is over. `v5_5_ru` is still downloaded and opened, but only to
  lift its stress and homograph model out — its voices are dropped again
  right away, and only `v5_cis_base` stays warm.
- **Resident memory drops by roughly a third**: ~585 MB against 1.2.0's
  ~950 MB, and below 1.1.0's ~653 MB (same server, same ten requests, two
  runs each). Start-up is also about half as long, since only one package is
  warmed up. Disk is unchanged at ~250 MB — the stress model still ships
  inside the whole `v5_5_ru` package.
- **Upgrading:** a saved `voice: xenia` (or `baya`, `kseniya`, `aidar`,
  `eugene`) no longer validates — open the add-on's Configuration tab and
  pick a voice once. Assist pipelines pinned to one of those need re-picking
  too, under **Settings → Voice assistants**.

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
