#!/usr/bin/env bash
# Stop app.py and the Cloudflare tunnel, regardless of how they were launched.
# Cleans up stale PID files if the processes are already gone.

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/app.pid"
CLOUDFLARED_PID_FILE="$APP_DIR/cloudflared.pid"
KILLED=0

# ── 1. Stop app.py by PID file ────────────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[stop] Stopping app.py PID $PID..."
        kill "$PID" && KILLED=1
    else
        echo "[stop] PID $PID in app.pid is already gone — removing stale file."
    fi
    rm -f "$PID_FILE"
fi

# ── 2. Catch any remaining app.py processes not tracked by PID file ───────
while IFS= read -r PID; do
    echo "[stop] Stopping app.py PID $PID..."
    kill "$PID" 2>/dev/null && KILLED=1
done < <(pgrep -f "python.*app\.py" 2>/dev/null)

# ── 3. Stop Cloudflare tunnel by PID file ─────────────────────────────────
if [[ -f "$CLOUDFLARED_PID_FILE" ]]; then
    CF_PID=$(cat "$CLOUDFLARED_PID_FILE")
    if kill -0 "$CF_PID" 2>/dev/null; then
        echo "[stop] Stopping Cloudflare tunnel PID $CF_PID..."
        kill "$CF_PID" 2>/dev/null && KILLED=1
    else
        echo "[stop] Cloudflare tunnel PID $CF_PID already gone — removing stale file."
    fi
    rm -f "$CLOUDFLARED_PID_FILE"
fi

# ── 4. Catch any remaining cloudflared processes for port 7860 ────────────
pkill -f "cloudflared.*7860" 2>/dev/null && KILLED=1 || true

if [[ $KILLED -eq 1 ]]; then
    echo "[stop] Done."
else
    echo "[stop] Nothing was running."
fi
