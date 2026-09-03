#!/usr/bin/env bash
# ============================================================
#  NewGen VPS Launcher  --  run this on the VPS
#
#  Usage:
#    bash autorun.sh           # start app.py (survives SSH disconnect)
#    bash autorun.sh stop      # kill it
#    bash autorun.sh status    # show PID
#    bash autorun.sh logs      # tail live log
#    bash autorun.sh restart   # stop then start
# ============================================================

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$APP_DIR/app.py"
LOG="$APP_DIR/newgen.log"
PID_FILE="$APP_DIR/app.pid"
APP_VENV="$APP_DIR/.app-venv"

# Prefer the isolated venv python (has all deps installed by setup.sh).
# Fall back to the PYTHON env var or bare python3 only if the venv doesn't exist.
# IMPORTANT: do NOT export PYTHONPATH pointing at the system site-packages here.
# setup.sh places directory symlinks for torch/torchvision/torchaudio inside the
# venv's own site-packages AND writes a zzz_system_torch_path.pth file so
# sys.path already includes the system torch location at runtime — no PYTHONPATH
# needed.  A PYTHONPATH export pointing at the whole system dist-packages dir
# would shadow every same-named package in the venv with the system copy (e.g.
# it would make the app import system diffusers 0.33.1 instead of the venv's
# correctly-installed 0.37.1).
if [ -f "$APP_VENV/bin/python" ]; then
    PYTHON="$APP_VENV/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

do_stop() {
    # Kill via PID file
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[autorun] Stopping PID $PID..."
            kill "$PID"
        fi
        rm -f "$PID_FILE"
    fi
    # Also catch anything missed (nohup, direct python3, venv python, etc.)
    pkill -f "app\.py" 2>/dev/null || true
    echo "[autorun] Stopped."
}

do_start() {
    if is_running; then
        echo "[autorun] Already running (PID $(cat "$PID_FILE")). Run:  bash autorun.sh restart"
        exit 0
    fi
    [[ -f "$APP" ]] || { echo "[autorun] ERROR: $APP not found."; exit 1; }

    echo "[autorun] Starting app.py..."
    echo "[autorun] Log  -> $LOG"

    # nohup + setsid: process survives SSH disconnect and terminal close
    nohup setsid "$PYTHON" "$APP" > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"

    echo "[autorun] Waiting for startup..."
    for i in $(seq 1 30); do
        sleep 2
        if grep -q "Running on local URL" "$LOG" 2>/dev/null; then
            echo "[autorun] Started (PID $(cat "$PID_FILE")). Gradio is up."
            echo "[autorun] Confirm AutorunAPI:"
            grep "AutorunAPI\|Running on" "$LOG" | tail -5
            echo ""
            echo "[autorun] ── Live log (Ctrl+C to detach, app keeps running) ──"
            tail -n 80 -f "$LOG"
            return
        fi
        if ! is_running; then
            echo "[autorun] ERROR: process died. Last log:"
            tail -20 "$LOG"
            exit 1
        fi
    done
    echo "[autorun] Still starting (taking longer than usual)."
    echo ""
    echo "[autorun] ── Live log (Ctrl+C to detach, app keeps running) ──"
    tail -n 80 -f "$LOG"
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 2; do_start ;;
    status)
        if is_running; then
            echo "[autorun] Running (PID $(cat "$PID_FILE"))"
        else
            echo "[autorun] Not running."
        fi
        ;;
    logs)
        echo "[autorun] Tailing $LOG  (Ctrl+C to stop)..."
        tail -f "$LOG"
        ;;
    *)
        echo "Usage: bash autorun.sh [start|stop|restart|status|logs]"
        exit 1
        ;;
esac
