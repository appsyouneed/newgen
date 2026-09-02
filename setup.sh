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
    echo "System torch found at $SYS_SITE — venv will use it via PYTHONPATH."
    export PYTHONPATH="$SYS_SITE:${PYTHONPATH:-}"
else
    echo "WARNING: Could not locate system torch in $SYS_SITE."
    echo "  The app venv will attempt to use whatever torch is on PYTHONPATH at runtime."
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
"$APP_PIP" install --quiet --no-cache-dir \
    Pillow \
    "transformers>=4.52.0,<5.0" \
    "huggingface-hub>=0.34.0,<1.0" \
    "numpy>=1.26,<2.1" \
    "diffusers>=0.34.0,<0.38.0" \
    "safetensors>=0.4.0" \
    torchao \
    accelerate

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
"$APP_PY" - <<'PYPATCH'
import sys, pathlib, site

for sp in site.getsitepackages():
    candidate = pathlib.Path(sp, "gradio", "oauth.py")
    if candidate.exists():
        text = candidate.read_text()
        OLD = "from huggingface_hub import HfFolder, whoami"
        if OLD not in text:
            print(f"  Already patched or pattern not found: {candidate}")
            break
        NEW = """\
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
        candidate.write_text(text.replace(OLD, NEW, 1))
        print(f"  Patched: {candidate}")
        break
PYPATCH

echo "Patching venv gradio_client/utils.py for pydantic v2 bool-schema compatibility..."
"$APP_PY" - <<'PYPATCH'
import sys, pathlib, site

for sp in site.getsitepackages():
    candidate = pathlib.Path(sp, "gradio_client", "utils.py")
    if candidate.exists():
        text = candidate.read_text()
        if "if not isinstance(schema, dict):" in text:
            print(f"  Already patched: {candidate}")
            break
        OLD = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
        NEW = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return "Any"\n    if schema == {}:'
        if OLD in text:
            candidate.write_text(text.replace(OLD, NEW, 1))
            print(f"  Patched: {candidate}")
            break
        OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
        NEW2 = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return "unknown"\n    if "const" in schema:'
        if OLD2 in text:
            candidate.write_text(text.replace(OLD2, NEW2, 1))
            print(f"  Patched get_type fallback: {candidate}")
            break
        print(f"  ERROR: no matching pattern in {candidate}")
        break
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
# Launch app.py using the isolated app venv, with system torch on PYTHONPATH.
# This preserves the system torch 2.8 dev+cu128 while using venv's gradio/
# diffusers/transformers/etc.
APP_VENV="/root/newgen/.app-venv"
SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SYS_SITE" ] && [ -d "$SYS_SITE/torch" ]; then
    export PYTHONPATH="$SYS_SITE:${PYTHONPATH:-}"
fi
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
