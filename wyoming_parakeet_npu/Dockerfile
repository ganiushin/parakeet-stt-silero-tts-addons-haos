# Wyoming Parakeet (Intel NPU) — built entirely from sources in this repository.
# The application code lives in ./parakeet (vendored, MIT — see parakeet/VENDORED.md).
# The only build-time downloads are Intel's official NPU driver packages and the
# oneAPI Level Zero loader, pinned by version and verified by SHA-256.
# Base image comes from Google's mirror of the Docker Hub library (same image,
# same digests) because direct docker.io access is unreliable on some networks.
FROM mirror.gcr.io/library/ubuntu:24.04

ARG LIBZE_URL="https://github.com/oneapi-src/level-zero/releases/download/v1.28.6/libze1_1.28.6+u24.04_amd64.deb"
ARG LIBZE_SHA256="8af9dc06d9684a20a3f43754fe109968fba2ba085cfa572a4cc6ef0272f9bcd1"
ARG NPU_DRIVER_URL="https://github.com/intel/linux-npu-driver/releases/download/v1.33.0/linux-npu-driver-v1.33.0.20260529-26625960453-ubuntu2404.tar.gz"
ARG NPU_DRIVER_SHA256="48cc219a5bd882a7cadaef523d9bb0c1e03c1b9b5b8562a29ad5ee4b9bfbb3bc"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/data/huggingface \
    TZ=Etc/UTC \
    DATA_DIR=/data \
    LANGUAGE=ru \
    DEVICE=NPU \
    ENCODER_BUCKETS=10 \
    ENCODER_LAZY_BUCKETS=""

# System packages (jq is used by the add-on's run.sh to read options.json)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq tzdata \
        python3 python3-pip python3-venv \
        libtbb12 libtbbmalloc2 ocl-icd-libopencl1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Level Zero loader >= 1.17 (Ubuntu 24.04 ships 1.16, which can't discover
# the modern intel-level-zero-npu library)
RUN mkdir /tmp/lz && cd /tmp/lz \
    && curl -fL -o libze1.deb "$LIBZE_URL" \
    && echo "$LIBZE_SHA256  libze1.deb" | sha256sum -c - \
    && apt-get update \
    && apt-get install -y --no-install-recommends ./libze1.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/lz

# Intel NPU userspace driver + firmware + compiler
RUN mkdir /tmp/npu && cd /tmp/npu \
    && curl -fL -o npu.tar.gz "$NPU_DRIVER_URL" \
    && echo "$NPU_DRIVER_SHA256  npu.tar.gz" | sha256sum -c - \
    && tar xzf npu.tar.gz \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ./intel-driver-compiler-npu_*.deb \
        ./intel-fw-npu_*.deb \
        ./intel-level-zero-npu_*.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/npu

# Python environment + the vendored application
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY parakeet/pyproject.toml ./
COPY parakeet/src ./src
COPY parakeet/scripts ./scripts
COPY parakeet/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY blobs.json /app/blobs.json

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && chmod +x /usr/local/bin/docker-entrypoint.sh \
    # The pip openvino wheel lacks the NPU compiler loader; the driver package
    # ships it. Symlink it where the plugin looks, enabling offline (AOT)
    # compilation paths as well.
    && ln -sf /usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler_loader.so \
        /opt/venv/lib/python3.12/site-packages/openvino/libs/libopenvino_intel_npu_compiler_loader.so \
    && ln -sf /usr/lib/x86_64-linux-gnu/libopenvino_intel_npu_compiler.so \
        /opt/venv/lib/python3.12/site-packages/openvino/libs/libopenvino_intel_npu_compiler.so

EXPOSE 10300

# No HEALTHCHECK / supervisor watchdog: the first start downloads ~3.2 GB of
# models before the port opens, which takes far longer than any sane probe
# window and would get the add-on killed mid-download.

COPY run.sh /run.sh
RUN chmod +x /run.sh

ENTRYPOINT ["/run.sh"]
