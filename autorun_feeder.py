#!/usr/bin/env python3
"""Feed local images to NewGen Push Autorun through an SSH tunnel.

After each generation the VPS queues the finished .mp4 on GET /autorun/download.
This feeder fetches it, writes it to --download-dir (default: D:\\Apps\\newgen\\downloads),
and only then POSTs /autorun/ready so the VPS knows it can accept the next image.

State machine on the VPS (app.py):
  idle  ->  (user clicks Start Push Autorun)  ->  ready
  ready ->  (feeder POSTs /autorun/push)       ->  busy
  busy  ->  (generation finishes)              ->  done   (video available on /autorun/download)
  done  ->  (feeder POSTs /autorun/ready)      ->  ready  (clears done, accepts next push)

The feeder must:
  1. Wait for state == "ready"  (app is accepting images)
  2. POST /autorun/push         (sends image, state -> busy)
  3. Wait for state == "done"   (generation finished, video queued)
  4. GET  /autorun/download     (pull the .mp4 while state is still "done")
  5. POST /autorun/ready        (signal app to go back to "ready" for next image)
  6. Repeat for next image
"""
import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def endpoint(port, path):
    return f"http://127.0.0.1:{port}/autorun/{path}"


def request_json(port, path, method="GET", body=None, headers=None):
    request = urllib.request.Request(
        endpoint(port, path), data=body, method=method, headers=headers or {}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        return {"error": error.read().decode(errors="replace"), "http_status": error.code}
    except Exception as error:
        return {"error": str(error)}


def send_image(port, path):
    boundary = "----NewgenAutorunBoundary"
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return request_json(
        port,
        "push",
        method="POST",
        body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def download_video(port, download_dir, timeout=300):
    """
    Pull the finished .mp4 from GET /autorun/download and save it locally.

    The VPS queues the video in memory as soon as generation completes (state
    transitions to "done").  This function polls until the video is available
    (HTTP 200) or until `timeout` seconds elapse.

    IMPORTANT: this must be called BEFORE POST /autorun/ready.  The /ready
    signal clears the "done" state and moves the app back to "ready" so it
    can accept the next image.  Downloading after signalling ready creates a
    race where the video bytes may already be cleared.

    Returns the local Path on success, None on failure.
    """
    deadline = time.monotonic() + timeout
    last_print = 0.0
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(endpoint(port, "download"), method="GET")
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    time.sleep(2)
                    continue
                # Extract filename from Content-Disposition header
                cd = resp.headers.get("Content-Disposition", "")
                fname = None
                for part in cd.split(";"):
                    part = part.strip()
                    if part.startswith("filename="):
                        fname = part.split("=", 1)[1].strip('"')
                        break
                if not fname:
                    fname = f"vidgen_{int(time.time())}.mp4"

                dest = Path(download_dir) / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"  [feeder] Downloaded -> {dest} ({dest.stat().st_size // 1024} KB)")
                return dest

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Video not queued yet — generation may still be finalising
                now = time.monotonic()
                if now - last_print >= 10:
                    print("  [feeder] Waiting for video to be queued for download…")
                    last_print = now
                time.sleep(2)
                continue
            print(f"  [feeder] Download HTTP error {e.code}: {e.read().decode(errors='replace')}")
            return None
        except Exception as e:
            print(f"  [feeder] Download error: {e}")
            time.sleep(3)

    print("  [feeder] Timed out waiting for video download.")
    return None


def signal_ready(port):
    """POST /autorun/ready — tells the VPS to transition done -> ready."""
    return request_json(port, "ready", method="POST")


def get_state(port):
    result = request_json(port, "status")
    if "error" in result:
        return None  # tunnel/app unreachable
    return result.get("state")


def wait_for(port, expected, timeout=1800, label=None):
    """
    Poll /autorun/status until state == expected or timeout expires.

    Prints a progress line at most once every 15 seconds so the terminal
    isn't spammed during long generations, but still shows life signs.

    Returns True if reached, False on timeout.
    """
    deadline = time.monotonic() + timeout
    last_print = 0.0
    while time.monotonic() < deadline:
        current = get_state(port)
        if current == expected:
            return True
        now = time.monotonic()
        if now - last_print >= 15:
            if current is None:
                print("  [feeder] App/tunnel unreachable — retrying…")
            else:
                msg = label or f"waiting for state={expected!r}"
                print(f"  [feeder] state={current!r}  ({msg})")
            last_print = now
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description="NewGen Push Autorun feeder")
    parser.add_argument(
        "--folder",
        default="./autorun",
        help="Folder containing input images (default: ./autorun)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Local port the SSH tunnel forwards to (default: 7861)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0,
        help="Extra seconds to wait between images (default: 0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process only the first image then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List images that would be sent but do not send them",
    )
    parser.add_argument(
        "--download-dir",
        default=r"D:\Apps\newgen\downloads",
        help="Local folder where finished .mp4 files are saved",
    )
    parser.add_argument(
        "--gen-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for a single generation to complete (default: 1800)",
    )
    parser.add_argument(
        "--dl-timeout",
        type=int,
        default=300,
        help="Seconds to wait for the video download after generation (default: 300)",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[feeder] Folder not found: {folder}")
        return 1

    images = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED],
        key=lambda p: p.name.casefold(),
    )
    if not images:
        print(f"[feeder] No supported images ({', '.join(SUPPORTED)}) found in {folder}")
        return 1

    if args.once:
        images = images[:1]

    print(f"[feeder] Found {len(images)} image(s) in {folder}:")
    for i, image in enumerate(images, 1):
        print(f"  {i:3}. {image.name}")

    if args.dry_run:
        print("[feeder] Dry run — nothing sent.")
        return 0

    # ------------------------------------------------------------------ #
    #  Wait for the app/tunnel to be reachable                            #
    # ------------------------------------------------------------------ #
    print("[feeder] Waiting for app/tunnel to be reachable…")
    try:
        while get_state(args.port) is None:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[feeder] Interrupted — exiting.")
        return 1

    print("[feeder] App reachable.")
    print("[feeder] Waiting for state=ready — click [Start Push Autorun] in the app if you haven't yet.")

    try:
        if not wait_for(args.port, "ready", timeout=900,
                        label="waiting for user to click Start Push Autorun"):
            print("[feeder] Timed out waiting for initial ready state.")
            return 1

        for i, image in enumerate(images, 1):
            print(f"\n[feeder] ── Image {i}/{len(images)}: {image.name} ──")

            # ---- Push the image ----------------------------------------
            result = send_image(args.port, image)
            if "error" in result:
                print(f"[feeder] Push failed: {result['error']}")
                return 1
            print(f"[feeder] Accepted by VPS. Waiting for generation to finish…")

            # ---- Wait for generation to complete (state: busy -> done) ---
            if not wait_for(args.port, "done", timeout=args.gen_timeout,
                            label="generating video"):
                print("[feeder] Timed out waiting for generation to complete.")
                return 1
            print("[feeder] Generation done.")

            # ---- Download the finished video BEFORE signalling ready -----
            # (signalling ready clears the done state and the app may accept
            # the next push immediately, racing with our download attempt)
            print(f"[feeder] Downloading video to {args.download_dir}…")
            dest = download_video(args.port, args.download_dir, timeout=args.dl_timeout)
            if dest is None:
                print("[feeder] Warning: video download failed — still signalling ready to unblock the VPS.")

            # ---- Signal the VPS: ready for next image --------------------
            sig = signal_ready(args.port)
            if "error" in sig:
                print(f"[feeder] /autorun/ready signal failed: {sig['error']}")
                return 1
            print("[feeder] VPS signalled ready for next image.")

            if args.pause and i < len(images):
                print(f"[feeder] Pausing {args.pause}s before next image…")
                time.sleep(args.pause)

            # ---- Wait for VPS to confirm it's ready for the next push ----
            # (state transitions done -> ready after /autorun/ready is POSTed;
            # this wait ensures we don't push the next image before the VPS
            # has processed the ready signal and looped back)
            if i < len(images):
                if not wait_for(args.port, "ready", timeout=60,
                                label="waiting for VPS to confirm ready"):
                    print("[feeder] VPS didn't return to ready in time — proceeding anyway.")

    except KeyboardInterrupt:
        print("\n[feeder] Interrupted — cancelling push autorun on VPS…")
        request_json(args.port, "cancel", method="POST")
        return 1

    print(f"\n[feeder] ✓ All {len(images)} image(s) processed successfully.")
    print(f"[feeder] Videos saved to: {args.download_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
