#!/usr/bin/env bash
# Kill any running app.py process, regardless of how it was launched.
# Also cleans up a stale PID file if the process is already gone.

PID_FILE="$(cd "$(dirname "$0")" && pwd)/app.pid"
KILLED=0

# 1. Kill by PID file if it exists
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[kill] Killing PID $PID (from app.pid)..."
        kill -9 "$PID" && KILLED=1
    else
        echo "[kill] PID $PID in app.pid is already gone — removing stale file."
    fi
    rm -f "$PID_FILE"
fi

# 2. Kill any remaining python processes running app.py (catches nohup/setsid
#    children that have a different parent PID than what app.pid recorded, or
#    instances started without autorun.sh).
while IFS= read -r PID; do
    echo "[kill] Killing PID $PID (app.py process)..."
    kill -9 "$PID" 2>/dev/null && KILLED=1
done < <(pgrep -f "python.*app\.py" 2>/dev/null)

if [[ $KILLED -eq 1 ]]; then
    echo "[kill] Done."
else
    echo "[kill] No running app.py found."
fi
