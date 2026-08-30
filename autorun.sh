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
PYTHON="${PYTHON:-python3}"

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
    # Also catch anything missed (nohup, direct python3, etc.)
    pkill -f "python.*app\.py" 2>/dev/null || true
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
            echo "[autorun] Tail log with:  bash autorun.sh logs"
            return
        fi
        if ! is_running; then
            echo "[autorun] ERROR: process died. Last log:"
            tail -20 "$LOG"
            exit 1
        fi
    done
    echo "[autorun] Still starting (taking longer than usual). Tail with:  bash autorun.sh logs"
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
