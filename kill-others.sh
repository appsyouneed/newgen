#!/bin/bash
pkill -9 -f "python3 app.py" 2>/dev/null || true
pkill -9 -f "python app.py" 2>/dev/null || true
echo "Done."
