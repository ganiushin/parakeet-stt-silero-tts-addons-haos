# Wyoming Parakeet (Intel NPU)

Runs [wyoming-parakeet-on-intel-npu](https://github.com/cibernox/wyoming-parakeet-on-intel-npu): NVIDIA Parakeet TDT 0.6B v3 multilingual speech-to-text, accelerated on Intel NPUs (AI Boost) via OpenVINO and exposed over the Wyoming protocol. Use it as the STT engine in Assist voice pipelines instead of Whisper.

## Requirements

- Intel Core Ultra CPU with an AI Boost NPU (Arrow Lake verified; Meteor Lake and Lunar Lake should work).
- `/dev/accel/accel0` must exist on the Home Assistant OS host (`intel_vpu` kernel driver). Check from the SSH add-on with protection mode off: `ls /dev/accel/`.
- ~1.5 GB free disk space in the add-on data volume with the default 10 s bucket (NPU blob, decoder IR, vocabulary). Bucket sizes that have no precompiled blob additionally need ~2.5 GB for the FP32 encoder graph they are compiled from.
- ~2 GB of free RAM for the add-on. On Proxmox with NPU passthrough give the HAOS VM at least 5 GB — passthrough pins the guest RAM, and a 4 GB VM running a typical add-on set OOM-kills the model load (exit code 137).

## Running HAOS in a Proxmox VM

Two things are required for the NPU to work inside a VM:

1. **Pass the NPU through** to the VM as a PCI device (Proxmox: VM → Hardware → Add → PCI Device → the Intel NPU / AI Boost device).
2. **Add `intel_vpu.force_snoop=1` to the HAOS kernel command line.** Inside a VM the IOMMU does not provide cache-coherent (snooped) DMA for the NPU, so the driver must be forced into snooping mode — without this the device misbehaves or fails to initialize. From the HAOS console (or SSH on port 22222):

   ```
   vi /mnt/boot/cmdline.txt      # append: intel_vpu.force_snoop=1
   reboot
   ```

   After reboot verify: `cat /proc/cmdline` should show the parameter and `ls /dev/accel/accel0` should succeed.

## First start

On first start the add-on downloads a ~1.2 GB precompiled NPU blob (SHA-256 verified) plus ~72 MB of model files into its persistent data directory — watch the progress in the add-on log. No on-device model compilation happens for the default 10 s bucket. Later starts take seconds.

The add-on only keeps what it actually runs on. The FP32 decoder/joint graph is deleted as soon as the static IR is built from it, and the 2.5 GB FP32 encoder graph is fetched only when a configured bucket has no precompiled blob — change `encoder_buckets` away from the default and the next start will download it, change it back and the next start will remove it again. Upgrading from 1.5.0 or earlier reclaims about 3.2 GB on the first start.

Once the Wyoming server is listening, the add-on registers itself with Home Assistant and the **Wyoming Protocol** integration is offered under **Settings → Devices & Services** (accept it, or add it manually with the host IP and port `10300`).

Then select the new STT engine in **Settings → Voice assistants** for your pipeline.

## Options

### `language`

Default transcription language, used when the pipeline does not specify one. Parakeet TDT 0.6B v3 supports 25 European languages: `bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro ru sk sl es sv uk`.

### `force_language`

Parakeet TDT v3 has **no language input**: it auto-detects the language of
every utterance on its own (the Wyoming pipeline language is ignored by the
model), and on quiet or noisy audio the guess occasionally lands on the wrong
language — e.g. a Russian command transcribed as English gibberish.

With this option enabled (default), decoding is locked to the alphabet of the
requested language by suppressing other scripts' tokens: `ru`/`uk`/`bg` allow
only Cyrillic, `el` only Greek, all other languages only Latin letters. Digits
and punctuation are always allowed.

Note the lock is per *script*, not per language: it cannot separate two
Latin-script languages (e.g. English vs. German). Disable it if you dictate
mixed-language text (e.g. Russian with inline English terms) and prefer proper
Latin spelling over forced transliteration.

### `device`

OpenVINO device to run inference on: `NPU` (default) or `CPU`. `CPU` is a useful fallback to verify the pipeline works if the NPU is not detected. GPU is not offered: it would need the host's `/dev/dri` mapped into the add-on, which this add-on does not do.

### `encoder_buckets` / `encoder_lazy_buckets`

Comma-separated audio bucket sizes in seconds. Eager buckets are prepared at startup; lazy buckets on first use. Audio shorter than the bucket is padded (and still costs a full-bucket inference — ~200 ms on the NPU for 10 s); audio longer than the largest bucket is transcribed in bucket-sized windows split at quiet points and stitched together (up to 60 s total, each window costing one inference).

The default **10 s** bucket uses a prebuilt, SHA-256-verified NPU blob (compiled offline for NPU 3720 — Meteor/Arrow Lake) downloaded on first start, so no on-device compilation is needed. Any other size falls back to a one-time on-device compile: the add-on then downloads the **~2.5 GB FP32 encoder graph** it compiles from, and the compile itself peaks at **~5 GB of RAM**. The resulting blob is cached in `/data/ov_cache`, later starts are cheap again, and the FP32 graph is deleted once no configured bucket needs it. Each configured eager bucket keeps its own ~1.2 GB compiled copy in memory — on small VMs stick to a single bucket.

## Notes and limitations

- No streaming transcription: the whole utterance is transcribed at once (normal for Assist pipelines).
- amd64 only — the Intel NPU is x86 by definition.
- The image is built locally from source code vendored in this repository (`parakeet/`, see `parakeet/VENDORED.md` for provenance); the first install therefore takes several minutes while Python dependencies and the Intel NPU driver packages (SHA-256 verified) are installed.

## Troubleshooting

- **`/dev/accel/accel0 not found` in the log** — the host kernel does not have `intel_vpu` loaded or the NPU is unsupported. Set `device: CPU` to test the rest of the pipeline.
- **Discovery did not appear** — add the Wyoming integration manually: **Settings → Devices & Services → Add integration → Wyoming Protocol**, host = your HA machine IP, port = `10300`.
- **A word garbled in long dictation around 10 s marks** — audio longer than the bucket is split into windows at the quietest point near each boundary; continuous speech without pauses can occasionally clip a word at a seam. Speak with natural pauses or configure a larger bucket (one-time on-device compile, ~5 GB RAM peak).
