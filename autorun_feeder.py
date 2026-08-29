#!/usr/bin/env python3
"""
autorun_feeder.py  —  run this on YOUR LOCAL MACHINE, not the VPS.

It watches a local folder, sorts the images, and pushes them one at a time
to the newgen app running on the VPS via an SSH tunnel.

SETUP (one-time):
  1. Open an SSH tunnel in a terminal and leave it running:
       ssh -L 7861:127.0.0.1:7861 root@YOUR_VPS_IP -N
  2. Run this script:
       python3 autorun_feeder.py --folder ~/my-autorun-images
  3. Click "Autorun" in the app, then click "▶ Start Push Autorun".
  4. This script will detect the app is ready and push images one by one.
     Each image is sent, processed, downloaded to your browser, then cleared.
     The next image is only sent after the VPS signals it is ready again.

OPTIONS:
  --folder PATH   Local folder containing images (default: ./autorun)
  --port PORT     Local tunnel port (default: 7861)
  --pause SECS    Extra wait between images beyond app readiness (default: 0)
  --once          Exit after all images are processed (default: loops/waits)
  --dry-run       Print what would be sent without actually sending anything

SUPPORTED FORMATS: .jpg  .jpeg  .png  .webp
Images are sent in case-insensitive alphabetical order, exactly once each.
"""

import argparse
import sys
import time
import urllib.request
import urllib.error
import json
import os
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
API_BASE  = "http://127.0.0.1:{port}/autorun"


def api_url(port, path):
    return f"http://127.0.0.1:{port}/autorun/{path}"


def get_status(port):
    try:
        with urllib.request.urlopen(api_url(port, "status"), timeout=5) as r:
            return json.loads(r.read())["state"]
    except urllib.error.URLError:
        return None  # tunnel not up / app not running


def push_image(port, filepath: Path) -> dict:
    import mimetypes
    mime = mimetypes.guess_type(str(filepath))[0] or "image/jpeg"
    boundary = "----NewgenFeeder"
    data = filepath.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filepath.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        api_url(port, "push"),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}
    except urllib.error.URLError as e:
        return {"error": str(e)}


def discover(folder: Path):
    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED
    ]
    return sorted(files, key=lambda p: p.name.lower())


def wait_for_state(port, target_state, timeout=300, poll=2.0):
    """Poll until state matches target or timeout. Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = get_status(port)
        if s is None:
            print("  [feeder] Cannot reach app — is the SSH tunnel up?")
        elif s == target_state:
            return True
        else:
            print(f"  [feeder] App state: {s!r}  (waiting for {target_state!r})")
        time.sleep(poll)
    return False


def main():
    parser = argparse.ArgumentParser(description="Newgen autorun local feeder")
    parser.add_argument("--folder", default="./autorun",
                        help="Local folder with images to process")
    parser.add_argument("--port", type=int, default=7861,
                        help="SSH tunnel local port (default 7861)")
    parser.add_argument("--pause", type=float, default=0,
                        help="Extra seconds to wait between images")
    parser.add_argument("--once", action="store_true",
                        help="Exit after all images are sent")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without sending anything")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[feeder] Folder not found: {folder}")
        sys.exit(1)

    images = discover(folder)
    if not images:
        print(f"[feeder] No supported images in {folder}")
        sys.exit(1)

    print(f"[feeder] Found {len(images)} image(s) in {folder}:")
    for i, f in enumerate(images, 1):
        print(f"  {i:3}. {f.name}")
    print()

    if args.dry_run:
        print("[feeder] Dry run — nothing sent.")
        return

    # Wait for tunnel + app to be up
    print("[feeder] Waiting for SSH tunnel and app to be reachable…")
    while get_status(args.port) is None:
        print("  [feeder] Not reachable yet — is the SSH tunnel running?")
        print(f"  Run:  ssh -L {args.port}:127.0.0.1:{args.port} root@YOUR_VPS_IP -N")
        time.sleep(5)
    print("[feeder] App reachable!")

    # Wait for user to click "Start Push Autorun" in the app (state becomes "ready")
    print('[feeder] Waiting for app to enter "ready" state…')
    print('         → Click "Autorun" in the app, then "▶ Start Push Autorun"')
    if not wait_for_state(args.port, "ready", timeout=600):
        print("[feeder] Timed out waiting for ready state. Exiting.")
        sys.exit(1)

    total = len(images)
    for idx, img_path in enumerate(images, start=1):
        print(f"\n[feeder] Pushing {idx}/{total}: {img_path.name}")

        result = push_image(args.port, img_path)
        if "error" in result:
            print(f"[feeder] ERROR pushing {img_path.name}: {result['error']}")
            print("[feeder] Stopping.")
            sys.exit(1)

        print(f"[feeder] Accepted by app. Waiting for generation to finish…")

        # Wait for app to go busy then back to ready (or done/idle for last image)
        # Give it a moment to transition to busy first
        time.sleep(2)

        if idx < total:
            # Wait for "ready" signal — means generation + download + cleanup done
            if not wait_for_state(args.port, "ready", timeout=600):
                print("[feeder] Timed out waiting for app to become ready after generation.")
                sys.exit(1)
            if args.pause > 0:
                print(f"[feeder] Extra pause: {args.pause}s")
                time.sleep(args.pause)
        else:
            # Last image — wait for idle (app finished, no more images needed)
            print("[feeder] Last image sent — waiting for app to finish…")
            wait_for_state(args.port, "idle", timeout=600)

    print(f"\n[feeder] ✓ All {total} image(s) processed successfully.")
    if not args.once:
        print("[feeder] Done. You can close this script.")


if __name__ == "__main__":
    main()
