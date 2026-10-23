#!/usr/bin/env bash
# Start Breakdown for local development with one command.
#
#   ./dev.sh             backend + frontend
#   ./dev.sh --sensors   backend + frontend + IMU bridge (forwarding to the backend)
#   ./dev.sh --tunnel    backend + frontend + cloudflared tunnel; prints a phone link and QR code
#                        (flags can be combined; the link carries ACCESS_TOKEN, so keep it private)
#
# Ctrl+C stops everything. Run from the repo root.

set -euo pipefail
cd "$(dirname "$0")"

SENSORS=false
TUNNEL=false
for arg in "$@"; do
  case "$arg" in
    --sensors) SENSORS=true ;;
    --tunnel) TUNNEL=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

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

if $TUNNEL; then
  if ! command -v cloudflared >/dev/null; then
    echo "cloudflared not found. Install it first: brew install cloudflared"
    exit 1
  fi
  # ACCESS_TOKEN from the environment, else from backend/.env or .env.
  TOKEN="${ACCESS_TOKEN:-}"
  for f in backend/.env .env; do
    if [[ -z "$TOKEN" && -f "$f" ]]; then
      TOKEN=$(sed -n 's/^ACCESS_TOKEN=//p' "$f" | tail -n 1 | tr -d '"'"'"' \r')
    fi
  done
  if [[ -z "$TOKEN" ]]; then
    echo "ACCESS_TOKEN is not set. Set it in .env before starting a public tunnel (see .env.example)."
    exit 1
  fi
  TUNNEL_LOG=$(mktemp)
  cloudflared tunnel --url http://localhost:5173 > "$TUNNEL_LOG" 2>&1 &
  pids+=($!)
  (
    url=""
    for _ in $(seq 1 60); do
      url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" | head -n 1 || true)
      [[ -n "$url" ]] && break
      sleep 1
    done
    if [[ -z "$url" ]]; then
      echo "Tunnel URL did not appear. See $TUNNEL_LOG"
    else
      # A new hostname takes a little while to resolve. Printing it early gives
      # "server can't be found" on the phone, so wait until it answers here.
      echo "Waiting for $url to come online..."
      code=000
      for _ in $(seq 1 45); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" || true)
        [[ "$code" != "000" && "$code" != "530" && "$code" != "502" ]] && break
        sleep 2
      done
      [[ "$code" == "000" || "$code" == "530" || "$code" == "502" ]] && echo "Tunnel still not answering; try the link anyway, or restart."
      link="$url/?token=$TOKEN"
      echo "Phone link (contains your token, do not share it):"
      echo "$link"
      if command -v qrencode >/dev/null; then
        qrencode -t ANSIUTF8 "$link"
        echo "Scan the QR code with the iPhone camera. If Safari says the server can't be found, wait 30 seconds and reload."
      else
        echo "For a QR code, run: brew install qrencode  (then restart ./dev.sh --tunnel)"
      fi
    fi
  ) > >(label tunnel 32) 2>&1 &
fi

echo "Running. Open http://localhost:5173  (Ctrl+C to stop)"
wait
