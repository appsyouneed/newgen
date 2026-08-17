#!/bin/bash
if [ "$EUID" -ne 0 ]; then exec sudo bash "$0" "$@"; fi
echo "=== Stopping newgen ==="
systemctl stop newgen 2>/dev/null || true
pkill -f "python3 /root/newgen/app.py" 2>/dev/null || true
lsof -ti:7860 | xargs kill -9 2>/dev/null || true
echo "Newgen stopped."
