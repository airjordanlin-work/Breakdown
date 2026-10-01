#!/usr/bin/env bash
# Start Breakdown for local development with one command.
#
#   ./dev.sh             backend + frontend
#   ./dev.sh --sensors   backend + frontend + IMU bridge (forwarding to the backend)
#
# Ctrl+C stops everything. Run from the repo root.

set -euo pipefail
cd "$(dirname "$0")"

SENSORS=false
[[ "${1:-}" == "--sensors" ]] && SENSORS=true

if [[ ! -f backend/.venv/bin/activate ]]; then
  echo "No backend venv found. Set it up first (see README, Running it > Backend)."
  exit 1
fi
if [[ ! -d frontend/node_modules ]]; then
  echo "Frontend packages missing. Run: (cd frontend && npm install)"
  exit 1
fi

# Label each process's output so you can tell them apart in one terminal.
label() { local tag=$1 color=$2; while IFS= read -r line; do printf "\033[%sm%-8s\033[0m %s\n" "$color" "$tag" "$line"; done; }

pids=()
cleanup() {
  echo; echo "Stopping..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(
  cd backend && source .venv/bin/activate
  exec python -m uvicorn api.server:app --reload --port 8000
) > >(label backend 36) 2>&1 &
pids+=($!)

(
  cd frontend
  exec npm run dev
) > >(label frontend 35) 2>&1 &
pids+=($!)

if $SENSORS; then
  # Give the backend a moment so the bridge's first connection succeeds.
  sleep 3
  (
    cd backend && source .venv/bin/activate
    exec python -u scripts/imu_bridge.py --forward
  ) > >(label sensors 33) 2>&1 &
  pids+=($!)
fi

echo "Running. Open http://localhost:5173  (Ctrl+C to stop)"
wait
