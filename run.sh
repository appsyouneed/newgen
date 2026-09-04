#!/usr/bin/env bash
# ============================================================
#  NewGen VPS Launcher  --  run this on the VPS
#
#  Usage:
#    bash run.sh              # start in vidgen mode (default)
#    bash run.sh picgen       # start in picgen mode (Qwen loads first)
#    bash run.sh -picgen      # same as above
#    bash run.sh stop         # kill it
#    bash run.sh status       # show PID
#    bash run.sh logs         # tail live log
#    bash run.sh restart      # stop then start
#    bash run.sh restart picgen  # restart in picgen mode
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
            echo "[run] Stopping PID $PID..."
            kill "$PID"
        fi
        rm -f "$PID_FILE"
    fi
    # Also catch anything missed (nohup, direct python3, venv python, etc.)
    pkill -f "app\.py" 2>/dev/null || true
    echo "[run] Stopped."
}

# $1 = optional mode flag ("picgen" / "-picgen" / "--picgen"), empty = vidgen
do_start() {
    local MODE_ARG=""
    local mode_display="vidgen (default)"
    local raw="${1:-}"
    local flag="${raw#-}"   # strip any leading dashes
    flag="${flag#-}"        # strip a second dash (handles --picgen too)
    if [[ "${flag,,}" == "picgen" ]]; then
        MODE_ARG="--picgen"
        mode_display="picgen"
    fi

    if is_running; then
        echo "[run] Already running (PID $(cat "$PID_FILE")). Run:  bash run.sh restart"
        exit 0
    fi
    [[ -f "$APP" ]] || { echo "[run] ERROR: $APP not found."; exit 1; }

    echo "[run] Starting app.py in $mode_display mode..."
    echo "[run] Log  -> $LOG"

    # nohup + setsid: process survives SSH disconnect and terminal close
    nohup setsid "$PYTHON" "$APP" $MODE_ARG > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"

    echo "[run] Waiting for startup..."
    for i in $(seq 1 30); do
        sleep 2
        if grep -q "Running on local URL" "$LOG" 2>/dev/null; then
            echo "[run] Started (PID $(cat "$PID_FILE")). Gradio is up."
            echo "[run] Confirm AutorunAPI:"
            grep "AutorunAPI\|Running on" "$LOG" | tail -5
            echo ""
            echo "[run] ── Live log (Ctrl+C to detach, app keeps running) ──"
            tail -n 80 -f "$LOG"
            return
        fi
        if ! is_running; then
            echo "[run] ERROR: process died. Last log:"
            tail -20 "$LOG"
            exit 1
        fi
    done
    echo "[run] Still starting (taking longer than usual)."
    echo ""
    echo "[run] ── Live log (Ctrl+C to detach, app keeps running) ──"
    tail -n 80 -f "$LOG"
}

# Parse command — first arg is the verb, second is optional mode
CMD="${1:-start}"
MODE="${2:-}"

# Allow mode as first arg with no verb (e.g. bash run.sh picgen)
case "${CMD,,}" in
    picgen|-picgen|--picgen)
        MODE="$CMD"
        CMD="start"
        ;;
esac

case "$CMD" in
    start)   do_start "$MODE" ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 2; do_start "$MODE" ;;
    status)
        if is_running; then
            echo "[run] Running (PID $(cat "$PID_FILE"))"
        else
            echo "[run] Not running."
        fi
        ;;
    logs)
        echo "[run] Tailing $LOG  (Ctrl+C to stop)..."
        tail -f "$LOG"
        ;;
    *)
        echo "Usage: bash run.sh [picgen] [start|stop|restart|status|logs]"
        echo "       bash run.sh picgen          # start in picgen mode"
        echo "       bash run.sh restart picgen  # restart in picgen mode"
        exit 1
        ;;
esac
