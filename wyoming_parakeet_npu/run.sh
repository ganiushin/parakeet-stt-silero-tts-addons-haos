#!/usr/bin/env bash
set -euo pipefail

OPTIONS=/data/options.json

opt() {
    jq -r --arg k "$1" --arg d "$2" '.[$k] // $d | tostring' "$OPTIONS" 2>/dev/null || echo "$2"
}

export LANGUAGE="$(opt language ru)"
export FORCE_LANGUAGE="$(opt force_language true)"
export DEVICE="$(opt device NPU)"
export ENCODER_BUCKETS="$(opt encoder_buckets 10)"
export ENCODER_LAZY_BUCKETS="$(opt encoder_lazy_buckets '')"
export DATA_DIR=/data
export WYOMING_URI=tcp://0.0.0.0:10300

echo "[addon] language=${LANGUAGE} force_language=${FORCE_LANGUAGE} device=${DEVICE} buckets=${ENCODER_BUCKETS} lazy_buckets=${ENCODER_LAZY_BUCKETS}"

if [ "${DEVICE}" = "NPU" ] && [ ! -e /dev/accel/accel0 ]; then
    echo "[addon] WARNING: /dev/accel/accel0 not found inside the container." >&2
    echo "[addon] The intel_vpu driver is missing on the host, or the NPU is unsupported." >&2
    echo "[addon] Set the 'device' option to CPU as a fallback." >&2
fi

# Register with Home Assistant via Supervisor discovery once the Wyoming
# server starts accepting connections, so the Wyoming integration is offered
# automatically in Settings -> Devices & Services.
(
    for _ in $(seq 1 120); do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 10300)); s.close()" 2>/dev/null; then
            payload="{\"service\": \"wyoming\", \"config\": {\"uri\": \"tcp://$(hostname):10300\"}}"
            if curl -sf -X POST \
                -H "Authorization: Bearer ${SUPERVISOR_TOKEN:-}" \
                -H "Content-Type: application/json" \
                -d "${payload}" \
                http://supervisor/discovery > /dev/null; then
                echo "[addon] Registered Wyoming discovery at tcp://$(hostname):10300"
            else
                echo "[addon] Discovery registration failed (add the Wyoming integration manually)." >&2
            fi
            exit 0
        fi
        sleep 5
    done
    echo "[addon] Server did not come up within 10 minutes; skipping discovery." >&2
) &

exec /usr/local/bin/docker-entrypoint.sh "$@"
