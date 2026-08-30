#!/usr/bin/env python3
"""Feed local images to NewGen Push Autorun through an SSH tunnel.

After each generation the VPS queues the finished .mp4 on GET /autorun/download.
This feeder fetches it, writes it to --download-dir (default: D:\\Apps\\newgen\\downloads),
and only then POSTs /autorun/ready so the VPS knows it can accept the next image.
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
        return {"error": error.read().decode(errors="replace")}
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


def download_video(port, download_dir, timeout=120):
    """
    Pull the finished .mp4 from GET /autorun/download and save it locally.
    Polls until the endpoint has a video ready (VPS queues it after generation).
    Returns the local Path on success, None on failure.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                endpoint(port, "download"), method="GET"
            )
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
                # Not ready yet — generation still in progress
                print("  [feeder] Waiting for video to be ready for download…")
                time.sleep(3)
                continue
            print(f"  [feeder] Download HTTP error {e.code}: {e.read().decode(errors='replace')}")
            return None
        except Exception as e:
            print(f"  [feeder] Download error: {e}")
            time.sleep(3)
    print("  [feeder] Timed out waiting for video download.")
    return None


def signal_ready(port):
    """POST /autorun/ready to tell the VPS it may accept the next image."""
    return request_json(port, "ready", method="POST")


def state(port):
    result = request_json(port, "status")
    return result.get("state")


def wait_for(port, expected, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = state(port)
        if current == expected:
            return True
        if current is None:
            print("  [feeder] App/tunnel unavailable; retrying…")
        else:
            print(f"  [feeder] state={current!r}; waiting for {expected!r}")
        time.sleep(2)
    return False


def main():
    parser = argparse.ArgumentParser(description="NewGen Push Autorun feeder")
    parser.add_argument("--folder", default="./autorun")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--pause", type=float, default=0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--download-dir",
        default=r"D:\Apps\newgen\downloads",
        help="Local folder where finished .mp4 files are saved (default: D:\\Apps\\newgen\\downloads)",
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
        print(f"[feeder] No supported images in {folder}")
        return 1

    print(f"[feeder] Found {len(images)} image(s) in {folder}:")
    for i, image in enumerate(images, 1):
        print(f"  {i:3}. {image.name}")
    if args.dry_run:
        print("[feeder] Dry run — nothing sent.")
        return 0

    print("[feeder] Waiting for app/tunnel…")
    while state(args.port) is None:
        time.sleep(5)
    print("[feeder] App reachable. Click Start Push Autorun in the app.")
    if not wait_for(args.port, "ready"):
        print("[feeder] Timed out waiting for initial ready state.")
        return 1

    for i, image in enumerate(images, 1):
        print(f"\n[feeder] Pushing {i}/{len(images)}: {image.name}")
        result = send_image(args.port, image)
        if "error" in result:
            print(f"[feeder] Push failed: {result['error']}")
            return 1
        print("[feeder] Accepted; waiting for generation to finish…")

        # Wait until the VPS transitions to "done" (video queued, storage cleared).
        if not wait_for(args.port, "done"):
            print("[feeder] Timed out waiting for generation to complete.")
            return 1

        # Pull the finished video from the VPS to the local download folder.
        print(f"[feeder] Generation done — downloading video to {args.download_dir}")
        dest = download_video(args.port, args.download_dir)
        if dest is None:
            print("[feeder] Warning: could not download video — signalling ready anyway.")

        # Tell the VPS it can accept the next image.
        sig = signal_ready(args.port)
        if "error" in sig:
            print(f"[feeder] /autorun/ready signal failed: {sig['error']}")
            return 1
        print("[feeder] VPS signalled ready for next image.")

        if args.pause and i < len(images):
            time.sleep(args.pause)

    print(f"\n[feeder] ✓ All {len(images)} image(s) processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
