# Wyoming Silero TTS — built entirely from sources in this repository.
# The application code lives in ./silero (original, MIT).
# The only build-time downloads are Python packages: torch comes from the
# official PyTorch CPU wheel index (the PyPI x86_64 wheel bundles CUDA and is
# ~10x larger), everything else from PyPI, versions pinned in pyproject.toml.
# Base image comes from Google's mirror of the Docker Hub library (same image,
# same digests) because direct docker.io access is unreliable on some networks.
FROM mirror.gcr.io/library/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    DATA_DIR=/data \
    VOICE=ru_zhadyra \
    SAMPLE_RATE=48000 \
    THREADS=2 \
    TRANSLITERATE=true

# System packages (jq is used by the add-on's run.sh to read options.json)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq tzdata \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Python environment + the application
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY silero/pyproject.toml ./
COPY silero/src ./src
COPY silero/scripts ./scripts
COPY silero/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-cache-dir --upgrade pip \
    # CPU-only torch first, so the pyproject "torch" requirement is already
    # satisfied and pip never reaches for the CUDA-bundled PyPI wheel.
    && pip install --no-cache-dir torch==2.8.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 10200

# No HEALTHCHECK / supervisor watchdog: the first start downloads the models
# (~237 MB) before the port opens; on a slow line that outlives probe windows.

COPY run.sh /run.sh
RUN chmod +x /run.sh

ENTRYPOINT ["/run.sh"]
