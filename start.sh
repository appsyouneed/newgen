#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$EUID" -ne 0 ]; then exec sudo bash "$0" "$@"; fi
cp "$SCRIPT_DIR/newgen.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable newgen.service
systemctl restart newgen.service
echo "✓ newgen service started!"
echo "Check status: systemctl status newgen"
echo "View logs: tail -f /root/newgen/newgen.log"
