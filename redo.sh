#!/bin/bash
# redo.sh — stop the running app and restart it cleanly via autorun.sh.
# Always uses the isolated app venv python (never bare python3 / system python).
# Usage: bash /root/newgen/redo.sh
cd /root/newgen
exec bash /root/newgen/autorun.sh restart
