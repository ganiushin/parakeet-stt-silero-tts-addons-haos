#!/usr/bin/env bash
set -euo pipefail

OPTIONS=/data/options.json

opt() {
    jq -r --arg k "$1" --arg d "$2" '.[$k] // $d | tostring' "$OPTIONS" 2>/dev/null || echo "$2"
}

export VOICE="$(opt voice ru_zhadyra)"
export SAMPLE_RATE="$(opt sample_rate 48000)"
export THREADS="$(opt threads 2)"
export TRANSLITERATE="$(opt transliterate true)"
export DATA_DIR=/data
export WYOMING_URI=tcp://0.0.0.0:10200

echo "[addon] voice=${VOICE} sample_rate=${SAMPLE_RATE} threads=${THREADS} transliterate=${TRANSLITERATE}"

# Register with Home Assistant via Supervisor discovery once the Wyoming
# server starts accepting connections, so the Wyoming integration is offered
# automatically in Settings -> Devices & Services.
(
    for _ in $(seq 1 120); do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', 10200)); s.close()" 2>/dev/null; then
            payload="{\"service\": \"wyoming\", \"config\": {\"uri\": \"tcp://$(hostname):10200\"}}"
            if curl -sf -X POST \
                -H "Authorization: Bearer ${SUPERVISOR_TOKEN:-}" \
                -H "Content-Type: application/json" \
                -d "${payload}" \
                http://supervisor/discovery > /dev/null; then
                echo "[addon] Registered Wyoming discovery at tcp://$(hostname):10200"
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
