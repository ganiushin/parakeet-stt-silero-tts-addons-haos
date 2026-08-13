# Wyoming Silero TTS

Runs [Silero](https://github.com/snakers4/silero-models) `v5_cis_base`
Russian text-to-speech on the CPU and exposes its 29 Russian voices over the
Wyoming protocol. Use it as the TTS engine in Assist voice pipelines instead
of Piper — the voices are markedly more natural than Piper's Russian ones.

## Requirements

- Any amd64 or aarch64 machine; no GPU or NPU needed. Synthesis runs ~50–100×
  faster than real time on two CPU threads.
- ~250 MB free disk space in the add-on data volume (the model files).
- ~750 MB of free RAM for the add-on (~585 MB resident after warm-up).

## First start

On first start the add-on downloads two model packages (~237 MB in total,
SHA-256 verified, resumable) into its persistent data directory: the voices
(`v5_cis_base`) and the Russian-only `v5_5_ru`, which is opened solely for
its stress and homograph model — `v5_cis_base` ships none of its own and
wants every word already stressed, and `v5_5_ru`'s own voices are freed again
as soon as that model is out. Later starts still take a while: on a modest
CPU, loading and warming up the voices before the port opens can take half a
minute.

Once the Wyoming server is listening, the add-on registers itself with Home
Assistant and the **Wyoming Protocol** integration is offered under
**Settings → Devices & Services** (accept it, or add it manually with the
host IP and port `10200`).

Then select the new TTS engine and voice in **Settings → Voice assistants**
for your pipeline.

> **Port conflict with Piper:** this add-on uses host port `10200`, the same
> as the official Piper add-on. Stop Piper first, or change the host port on
> this add-on's Configuration → Network panel if you want to run both.

## Options

### `voice`

Default speaker, used when the pipeline does not specify one. `v5_cis_base`
is a multilingual model, and each voice was recorded by a native speaker of
one of its languages; the 29 offered here are those same people reading
Russian, which is where the light accents come from:

| voice | recorded by a speaker of |
|---|---|
| `ru_zhadyra` (default), `ru_zhazira` | Kazakh |
| `ru_aigul`, `ru_alfia`, `ru_alfia2`, `ru_miyau`, `ru_ramilia` | Bashkir |
| `ru_albina`, `ru_marat` | Tatar |
| `ru_igor`, `ru_roman` | Ukrainian |
| `ru_dmitriy` | Belarusian |
| `ru_kejilgan`, `ru_kermen` | Kalmyk |
| `ru_onaoy`, `ru_safarhuja` | Tajik |
| `ru_karina`, `ru_sibday` | Khakas |
| `ru_gamat` | Azerbaijani |
| `ru_zara` | Armenian |
| `ru_vika` | Georgian |
| `ru_eduard` | Kabardian |
| `ru_nurgul` | Kyrgyz |
| `ru_oksana` | Moksha |
| `ru_bogdan` | Udmurt |
| `ru_saida` | Uzbek |
| `ru_ekaterina` | Chuvash |
| `ru_alexandr` | Erzya |
| `ru_zinaida` | Yakut |

All 29 are always installed; the pipeline can pick any of them per request.
The model's non-Russian voices are not offered — the add-on's text
normalization and stress model are Russian-only.

### `sample_rate`

Output sample rate: `48000` (default, best quality), `24000` or `8000`.

### `threads`

Torch CPU threads for synthesis (default `2`). Two threads already
synthesize far faster than real time; raise only if responses feel slow on a
very weak CPU.

### `transliterate`

The Silero model silently skips Latin script. When enabled (default),
Latin words — device names, "Wi-Fi", "Spotify" — are transliterated to
Cyrillic so they are spoken instead of dropped. The transliteration is
letter-based and approximate; disable it if you prefer Latin words silent.

## Text normalization

The model also drops bare digits, so the add-on expands them before
synthesis: integers and decimals become Russian words with the unit in
agreement (`21,5°C` → «двадцать одна целая пять десятых градуса»), times are
read as hours and minutes (`13:45` → «тринадцать сорок пять»), and `%`, `°C`,
`°F`, `№` are spelled out. The result then goes through the stress model,
which marks the stressed vowel of every word and resolves homographs
(«з+амок» vs «зам+ок») — `v5_cis_base` has no stress of its own and would
otherwise guess.

## Model license

The add-on code is MIT. So are the `v5_cis_base` voice weights
([LICENSE_CIS](https://github.com/snakers4/silero-models/blob/master/LICENSE_CIS)).
The `v5_5_ru` package, which the add-on downloads for its stress model, is
distributed under **CC BY-NC-SA 4.0** — free for personal, non-commercial
use, which is what a home Assist pipeline is. Commercial deployments need a
license from Silero.

## Troubleshooting

- **The add-on speaks numbers oddly.** Russian number agreement (gender and
  case) is approximated; «двадцать один градус» comes out fine, some
  combinations less so. Open an issue with the exact phrase.
- **A sentence is skipped entirely.** After normalization nothing speakable
  remained (e.g. only emoji or punctuation) — this is by design, the stream
  continues with the next sentence.
- **First response after a restart is slow.** The first synthesis pays
  one-time initialization; the add-on warms up at startup, but if you query
  it during the model download/load window the reply waits for that.
