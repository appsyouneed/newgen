#!/bin/bash
set -e

echo "=== Newgen Setup (Video + Photo Generator) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

# ---------------------------------------------------------------------------
# CRITICAL: This setup script NEVER touches the system torch / torchvision /
# torchaudio / numpy stack that ships with your Ubuntu 22.04 VPS.
# Those versions (torch 2.8 dev+cu128, etc.) are already correct and must
# not be changed.  All Python-level dependencies that the app needs are
# installed into /root/newgen/.app-venv so they are completely isolated from
# both the OS-managed python3 packages and the system torch stack.
#
# The ONLY system packages installed here are apt-managed OS tools:
#   ffmpeg, wget, unzip, git, git-lfs, python3-pip, python3-venv
# Those are safe because they live in /usr/bin / /usr/lib, not in
# site-packages, and cannot conflict with Python packages.
# ---------------------------------------------------------------------------

PYTHON="python3"

echo "Installing system apt dependencies (safe — OS tools only, not Python packages)..."
if lsof /var/lib/dpkg/lock-frontend > /dev/null 2>&1; then
    lsof -t /var/lib/dpkg/lock-frontend | xargs kill -9 2>/dev/null || true
fi
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock
rm -f /var/lib/dpkg/updates/*
dpkg --configure -a || true
apt-get update
apt-get install -y --fix-missing ffmpeg wget unzip git python3-pip python3-venv git-lfs

echo "Creating directories..."
mkdir -p /root/newgen/tmp
mkdir -p /root/newgen/mmaudio
mkdir -p /root/newgen/wan22_distill
mkdir -p /root/.cache/huggingface
chmod 1777 /root/newgen/tmp

# ---------------------------------------------------------------------------
# App venv — isolated from system Python site-packages.
# We do NOT use --system-site-packages so nothing leaks in from the OS level.
# The venv gets its own pip, its own diffusers, gradio, etc.
# torch / torchvision / torchaudio are NOT reinstalled here; the venv
# inherits the system torch via PYTHONPATH override at app launch time
# (see autorun.sh / systemd service), keeping the multi-GB dev build intact.
# ---------------------------------------------------------------------------
APP_VENV="/root/newgen/.app-venv"

if [ ! -f "$APP_VENV/bin/python" ]; then
    echo "Creating isolated app venv at $APP_VENV ..."
    $PYTHON -m venv "$APP_VENV"
fi

APP_PY="$APP_VENV/bin/python"
APP_PIP="$APP_VENV/bin/pip"

echo "Upgrading pip inside venv..."
"$APP_PIP" install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# Detect the system torch location so the venv can import it without
# reinstalling it.  We find the real site-packages dir by asking the system
# python where torch lives, then prepend it to PYTHONPATH for all subsequent
# pip calls and for the running app.  This means pip inside the venv will
# treat torch/torchvision/torchaudio as "already satisfied" (via PYTHONPATH)
# and skip downloading them.
# ---------------------------------------------------------------------------
SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SYS_SITE" ] && [ -d "$SYS_SITE/torch" ]; then
    echo "System torch found at $SYS_SITE."
    echo "Symlinking ONLY torch/torchvision/torchaudio/torchao (and their"
    echo "private deps) into the venv's own site-packages, instead of"
    echo "exporting PYTHONPATH to the whole system dist-packages dir."
    echo "(A blanket PYTHONPATH export shadows EVERY same-named package in"
    echo "the venv with the system copy — e.g. it silently made the app"
    echo "import system diffusers 0.33.1 instead of the venv's own 0.37.1,"
    echo "even though pip had correctly installed 0.37.1 into the venv.)"
    VENV_SITE_PACKAGES="$APP_VENV/lib/$($APP_PY -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
    for pkg in torch torchgen torchvision torchaudio torchao \
               functorch caffe2 \
               nvidia triton; do
        if [ -e "$SYS_SITE/$pkg" ] && [ ! -e "$VENV_SITE_PACKAGES/$pkg" ]; then
            ln -s "$SYS_SITE/$pkg" "$VENV_SITE_PACKAGES/$pkg"
        fi
    done
    # dist-info / egg-info dirs so `pip show torch` and dependency
    # resolution inside the venv recognize torch as already satisfied.
    for distinfo in "$SYS_SITE"/torch-*.dist-info "$SYS_SITE"/torch-*.egg-info \
                     "$SYS_SITE"/torchvision-*.dist-info "$SYS_SITE"/torchaudio-*.dist-info \
                     "$SYS_SITE"/torchao-*.dist-info; do
        [ -e "$distinfo" ] || continue
        base="$(basename "$distinfo")"
        [ -e "$VENV_SITE_PACKAGES/$base" ] || ln -s "$distinfo" "$VENV_SITE_PACKAGES/$base"
    done
else
    echo "WARNING: Could not locate system torch in $SYS_SITE."
    echo "  Do NOT let setup.sh install torch — that would overwrite your dev build."
fi

echo "Installing Python application dependencies into isolated venv..."
# Requirements file has torch/torchvision/torchaudio commented out intentionally.
# With PYTHONPATH pointing at the system site-packages, pip will resolve them
# as already satisfied and skip download.
"$APP_PIP" install --quiet --no-cache-dir \
    -r "$SCRIPT_DIR/requirements.txt"

echo "Ensuring critical pinned packages inside venv..."
# These are pinned to exact versions the app requires. They live only in the
# venv — the system site-packages are never touched.
# IMPORTANT: torch / torchvision / torchaudio / torchao are NOT installed here.
# They come from the system Python stack via PYTHONPATH (see run_app.sh).
# Installing them here would pull in a mismatched torch build and overwrite
# the system torch 2.8 dev+cu128.
"$APP_PIP" install --quiet --no-cache-dir \
    Pillow \
    "transformers>=4.52.0,<5.0" \
    "huggingface-hub>=0.34.0,<1.0" \
    "numpy>=1.26,<2.1" \
    "diffusers>=0.34.0,<0.38.0" \
    "safetensors>=0.4.0" \
    accelerate

# ---------------------------------------------------------------------------
# CRITICAL: Evict any torch / torchvision / torchaudio / torchao that pip may
# have pulled in as transitive dependencies of the packages above (e.g.
# diffusers, accelerate, torchao all list torch as a dependency and pip will
# happily download a fresh copy if it doesn't see one on sys.path at install
# time).  We run this AFTER every pip install block so any stray copies are
# removed before the app starts.  The system copies remain available via
# PYTHONPATH at runtime.
# ---------------------------------------------------------------------------
echo "Evicting any venv-local torch/torchvision/torchaudio/torchao (must use system copies)..."
"$APP_PIP" uninstall -y torch torchvision torchaudio torchao 2>/dev/null || true
echo "Torch eviction done — system torch will be used via PYTHONPATH at runtime."

echo "Installing torchcodec into venv (required by torchaudio.save for Foley audio)..."
# torchaudio 2.6+ routes torchaudio.save() through TorchCodec; without it the
# HunyuanVideo-Foley step crashes and videos come out silent.
"$APP_PIP" install --quiet --no-cache-dir torchcodec 2>/dev/null || true

# ---------------------------------------------------------------------------
# Patch gradio/oauth.py inside the VENV for huggingface_hub >= 0.26
# huggingface_hub removed HfFolder in 0.26.0; gradio 4.43.0 still imports it
# at module level, crashing the entire process on startup.
# This patch lives only inside the venv's copy of gradio — the system is untouched.
# ---------------------------------------------------------------------------
echo "Patching venv gradio/oauth.py for huggingface_hub >= 0.26 compatibility..."
VENV_SITE="$APP_VENV/lib/$(ls $APP_VENV/lib/)/site-packages"
OAUTH_FILE="$VENV_SITE/gradio/oauth.py"
python3 - "$OAUTH_FILE" <<'PYPATCH'
import sys, pathlib

candidate = pathlib.Path(sys.argv[1])
if not candidate.exists():
    print(f"  ERROR: not found: {candidate}")
    sys.exit(1)

text = candidate.read_text()

# Remove any previously botched patch attempts, then apply cleanly.
# The original line we need to replace is exactly:
#   from huggingface_hub import HfFolder, whoami
# It may have been wrapped in broken try blocks already — strip all of that
# out and rewrite from the first import of fastapi.responses onward.

GOOD = """\
try:
    from huggingface_hub import HfFolder, whoami
except ImportError:
    from huggingface_hub import whoami
    try:
        from huggingface_hub import get_token as _get_token
    except ImportError:
        _get_token = lambda: None  # noqa: E731

    class HfFolder:  # noqa: N801
        @staticmethod
        def get_token():
            return _get_token()"""

# Find the block between 'from fastapi.responses import RedirectResponse'
# and 'from .utils import get_space' and replace it wholesale.
ANCHOR_START = "from fastapi.responses import RedirectResponse"
ANCHOR_END = "from .utils import get_space"

start = text.index(ANCHOR_START) + len(ANCHOR_START)
end = text.index(ANCHOR_END)

text = text[:start] + "\n" + GOOD + "\n\n" + text[end:]
candidate.write_text(text)
print(f"  Patched: {candidate}")
PYPATCH

echo "Patching venv gradio_client/utils.py for pydantic v2 bool-schema compatibility..."
GRADIO_CLIENT_FILE="$VENV_SITE/gradio_client/utils.py"
python3 - "$GRADIO_CLIENT_FILE" <<'PYPATCH'
import sys, pathlib

candidate = pathlib.Path(sys.argv[1])
if not candidate.exists():
    print(f"  ERROR: not found: {candidate}")
    sys.exit(1)

text = candidate.read_text()
if "if not isinstance(schema, dict):" in text:
    print(f"  Already patched: {candidate}")
    sys.exit(0)

OLD = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
NEW = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return "Any"\n    if schema == {}:'
if OLD in text:
    candidate.write_text(text.replace(OLD, NEW, 1))
    print(f"  Patched: {candidate}")
    sys.exit(0)

OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
NEW2 = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return "unknown"\n    if "const" in schema:'
if OLD2 in text:
    candidate.write_text(text.replace(OLD2, NEW2, 1))
    print(f"  Patched get_type fallback: {candidate}")
    sys.exit(0)

print(f"  ERROR: no matching pattern in {candidate}")
sys.exit(1)
PYPATCH

echo "Installing SageAttention into venv for accelerated inference..."
"$APP_PIP" install --quiet --no-cache-dir sageattention 2>/dev/null || {
    echo "Pre-built SageAttention not available for this GPU arch — building from source..."
    "$APP_PIP" install "git+https://github.com/thu-ml/SageAttention.git" --no-cache-dir 2>/dev/null || {
        echo "⚠️  SageAttention install failed — will fall back to standard SDPA at runtime."
    }
}

echo "Installing rembg into venv (BiRefNet background removal for Merge Photos)..."
"$APP_PIP" install --quiet --no-cache-dir "rembg[gpu]" 2>/dev/null || {
    echo "rembg[gpu] failed — trying CPU-only rembg..."
    "$APP_PIP" install --quiet --no-cache-dir rembg 2>/dev/null || {
        echo "⚠️  rembg install failed — Merge Photos will use original images without background removal."
    }
}

echo "Ensuring pyOpenSSL inside venv..."
"$APP_PY" -c "from OpenSSL import SSL" 2>/dev/null || \
    "$APP_PIP" install --quiet pyopenssl

echo "Setting up RIFE interpolation model..."
if [ ! -d "/root/newgen/train_log/model" ] || [ ! -f "/root/newgen/train_log/RIFE_HDv3.py" ]; then
    rm -rf /root/newgen/train_log /root/newgen/__MACOSX /root/newgen/RIFEv4.26_0921.zip
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git /tmp/rife
    mkdir -p /root/newgen/train_log
    cp -r /tmp/rife/model /root/newgen/train_log/
    wget -q -P /root/newgen https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip
    unzip -o /root/newgen/RIFEv4.26_0921.zip -d /root/newgen
    rm -rf /tmp/rife /root/newgen/__MACOSX
    echo "RIFE installed."
else
    echo "RIFE already installed, skipping."
fi

echo "Preparing Wan 2.2 distilled weights directory..."
mkdir -p /root/newgen/wan22_distill
echo "Note: the two merged 4-step BF16 experts (~28.6 GB each, ~57 GB total)"
echo "download automatically on first video generation."
echo "Plan for roughly 150-200 GB total across Wan, Qwen and MMAudio."

echo "Copying app files to /root/newgen/..."
if [ "$SCRIPT_DIR" != "/root/newgen" ]; then
    cp "$SCRIPT_DIR/app.py" /root/newgen/app.py
    cp "$SCRIPT_DIR/prompts.py" /root/newgen/prompts.py
    cp "$SCRIPT_DIR/requirements.txt" /root/newgen/requirements.txt
else
    echo "Already in /root/newgen — skipping file copy."
fi

# Copy qwenimage module if present
if [ -d "$SCRIPT_DIR/qwenimage" ] && [ "$SCRIPT_DIR/qwenimage" != "/root/newgen/qwenimage" ]; then
    cp -r "$SCRIPT_DIR/qwenimage" /root/newgen/qwenimage
fi

# Copy starters if present
if [ -d "$SCRIPT_DIR/starters" ] && [ "$SCRIPT_DIR/starters" != "/root/newgen/starters" ]; then
    cp -r "$SCRIPT_DIR/starters" /root/newgen/starters
fi

# ---------------------------------------------------------------------------
# Write the launch wrapper that ensures the app venv's python is used but
# the system torch is visible via PYTHONPATH.  This is what systemd / autorun
# should call instead of plain "python3 app.py".
# ---------------------------------------------------------------------------
cat > /root/newgen/run_app.sh << 'LAUNCHER'
#!/bin/bash
# Launch app.py using the isolated app venv.
# System torch is made visible via symlinks placed directly inside the
# venv's own site-packages by setup.sh (see the SYS_SITE block above) —
# NOT via a PYTHONPATH export. A PYTHONPATH pointing at the whole system
# dist-packages dir would shadow every other same-named package in the
# venv (this previously caused the app to silently load system diffusers
# 0.33.1 instead of the venv's correctly-installed 0.37.1). Do not
# reintroduce a PYTHONPATH export here.
APP_VENV="/root/newgen/.app-venv"
cd /root/newgen
exec "$APP_VENV/bin/python" app.py "$@"
LAUNCHER
chmod +x /root/newgen/run_app.sh

# Check if running in Docker (no systemd)
if [ ! -d /run/systemd/system ]; then
    echo "Docker environment — skipping systemd setup"
    echo ""
    echo "=== Docker Startup Instructions ==="
    echo "Run: /root/newgen/run_app.sh"
    echo "  (uses isolated venv; system torch preserved)"
    echo ""
    bash "$SCRIPT_DIR/autorun.sh"
    exit 0
fi

echo "Setting up systemd service..."
cp "$SCRIPT_DIR/newgen.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable newgen
systemctl start newgen

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: App now runs via /root/newgen/run_app.sh"
echo "  System torch 2.8 dev+cu128 is PRESERVED."
echo "  All app dependencies live in /root/newgen/.app-venv"
echo ""
echo "🎬 New Features Available:"
echo "   • Wan 2.2 I2V A14B merged 4-step distill (BF16, no LoRA)"
echo "   • Qwen relocates subjects, Wan animates them"
echo "   • 4-step MoE (high-noise then low-noise expert)"
echo "   • One model resident on GPU at a time"
echo "   • 720p native generation (1280x720)"
echo "   • Zero configuration interface"
echo ""
echo "Service commands:"
echo "  systemctl status newgen"
echo "  systemctl restart newgen"
echo "  tail -f /root/newgen/newgen.log"
echo ""
echo "App: http://0.0.0.0:7860"
echo "📋 Video Tab: Qwen relocate -> Wan 2.2 merged 4-step animate"
echo "🖼️  Image Tab: Unchanged (Qwen Image Edit)"

bash "$SCRIPT_DIR/autorun.sh"
