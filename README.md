<p align="center">
  <h1 align="center">🎬 NewGen — AI Video & Photo Generator</h1>
  <p align="center">
    <em>Unrestricted NSFW-capable video and image generation powered by Wan 2.2 I2V Lightning + Qwen Image Edit</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Wan_2.2-I2V_Lightning-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Qwen-Image_Edit-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MMAudio-Sound_FX-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" />
</p>

---

## ✨ Features

🎬 **Video Generator (Vidgen)**
- Wan 2.2 I2V A14B — WAMU v2 Lightning merge (4-step distilled, NSFW-capable)
- Image-to-video with prompt-guided motion and scene control
- Segment chaining for videos up to 10 minutes
- Adaptive flow shift for optimal prompt following
- Preset prompt library for quick generation
- MMAudio integration for automatic sound generation

🖼️ **Photo Editor (Picgen)**
- Qwen Image Edit 2511 + Rapid AIO NSFW weights
- Multi-image input with custom edit instructions
- Preset prompt library for quick generation
- Starter images for quick reference loading (up to 4 slots)
- High-quality 4-step inference

⚡ **Performance Engine**
- Auto-detects GPU configuration and selects optimal execution mode
- SageAttention 2 integration (2-3× attention speedup)
- torch.compile with kernel fusion
- TF32 matmul acceleration
- Pipeline parallelism across multiple GPUs (accelerate balanced)
- Zero-swap concurrent mode on high-VRAM cards

🧠 **Smart GPU Modes**
- **Concurrent** — Both models GPU-resident, instant tab switching (≥48GB per GPU)
- **Stacked** — Models split across multiple GPUs via accelerate (multi-GPU, <48GB each)
- **Single** — Standard CPU offload + swap (1 GPU)

---

## 🚀 Quick Start

```bash
# Clone or copy files to /root/newgen/
cd /root/newgen

# Run setup (installs all dependencies, downloads models on first run)
bash setup.sh

# Start the app
python3 app.py
```

The app launches on `http://0.0.0.0:7860`

### Startup Flags

```bash
python3 app.py           # Default: vidgen tab loads first
python3 app.py -picgen   # Picgen tab loads first (single GPU mode only)
```

---

## 🖥️ System Requirements

- **OS:** Ubuntu 22.04 or 24.04
- **GPU:** NVIDIA with CUDA support (see GPU guide below)
- **RAM:** 64GB+ recommended (models stay in RAM for fast swapping)
- **Disk:** ~150-200GB for all model weights

---

## 📁 Project Structure

```
/root/newgen/
├── app.py              # Main application
├── prompts.py          # Preset prompt libraries
├── setup.sh            # One-click installer
├── clear.sh            # Clear generated files
├── requirements.txt    # Python dependencies
├── qwenimage/          # Custom Qwen pipeline module
├── starters/           # Quick-load reference images (start1.jpg - start4.jpg)
├── outputs/            # Generated images and videos
│   ├── images/
│   └── videos/
├── models/             # Downloaded model weights
├── train_log/          # RIFE interpolation model
└── tmp/                # Temporary/Gradio files
```

---

## 🧹 Maintenance

```bash
# Clear all generated files (keeps app intact)
bash clear.sh

# Or use the 🗑 Clear Storage button in the web UI
```

---

## 🛠️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Gradio Web UI                      │
├──────────────────────┬──────────────────────────────┤
│   🎬 Video Generator │   🖼️ Photo Editor             │
├──────────────────────┼──────────────────────────────┤
│   Wan 2.2 I2V A14B   │   Qwen Image Edit 2511       │
│   WAMU v2 Lightning   │   + Rapid AIO NSFW v23       │
│   (4-step distilled)  │   (4-step inference)         │
├──────────────────────┴──────────────────────────────┤
│              Inference Acceleration Layer             │
│   SageAttention │ torch.compile │ TF32 │ TeaCache*  │
├─────────────────────────────────────────────────────┤
│              GPU Mode Selection Engine                │
│   Concurrent │ Stacked (Pipeline Parallel) │ Single  │
├─────────────────────────────────────────────────────┤
│   RIFE Interpolation  │  MMAudio  │  OpenCV/FFmpeg   │
└─────────────────────────────────────────────────────┘
```
*TeaCache available but disabled for 4-step models (quality preservation)*

---

## 📝 Prompt Library (`prompts.py`)

The app includes an extensive preset prompt library organized by category for both tabs. Selecting any preset instantly populates the prompt field — no typing needed.

### 🎬 Video Presets (8 categories)

| Category | Description |
|---|---|
| **Solo** | Single subject motion and posing |
| **Couple (Unseen)** | Two subjects, one partially shown |
| **Couple (Seen)** | Two subjects, both fully visible |
| **Multiple** | Group scenes with motion |
| **Multi-Step** | Sequential action prompts |
| **Environment** | Background/setting changes |
| **Custom** | Miscellaneous creative prompts |
| **Multiple (Variants)** | Group scenes with different compositions |

### 🖼️ Photo Presets (7 categories)

| Category | Description |
|---|---|
| **Solo** | Single subject edits and poses |
| **Couple (Man Unseen)** | Paired scenes, one subject implied |
| **Couple (Man Seen)** | Paired scenes, both visible |
| **Multiple Women** | Group compositions, subjects only |
| **Multiple (Man Unseen)** | Group with implied additional subject |
| **Multiple (Man Seen)** | Group with all subjects visible |
| **Multi-Step** | Chained sequential edit instructions |

### Customization

Edit `prompts.py` to add your own presets. Each category is a Python dictionary — keys are display names, values are the prompt text:

```python
my_prompts_dict = {
    "Preset Name": "your detailed prompt text here",
    "Another Preset": "another prompt...",
}
```

---

## 🖼️ Starter Images (`starters/`)

The Picgen tab includes 10 quick-load buttons (**1** through **10**) that instantly load a reference image into the editor. This is useful for keeping frequently-used subjects (yourself, a partner, a character) one click away.

### Setup

Place images in the `starters/` folder with these names:

```
/root/newgen/starters/
├── start1.jpg    (or .png or .webp)
├── start2.jpg
├── start3.png
├── start4.webp
├── ...
└── start10.jpg
```

- Supports `.jpg`, `.png`, and `.webp` formats (one per slot)
- Any resolution works — automatically resized for the pipeline
- You can use any number of the 10 slots — empty slots are simply ignored
- Click any numbered button in the Picgen tab to instantly load that image as input

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEWGEN_FORCE_SINGLE_GPU` | `0` | Set to `1` to force single-GPU swap mode |

---

## 🎮 GPU Instance Selection Guide

Recommended GPUs and stacking configurations for SimplePod or similar cloud providers.

---

### 🟢 RTX 3090 (24GB)

> **Single Tab:** 2× stacked (48GB total)
> **Double Tab:** 4× stacked (96GB total)
> **Price for 1 tab (hourly):** ~$0.32/h
> **Price for 2 tabs (hourly):** ~$0.64/h
> **Est. Vidgen (3.5s video):** ~45-60s
> **Est. Picgen (1 image):** ~10-14s
> **Template(s):**
> - `simplepodai/ubuntu22.04-devel:cuda126`
> - `nvidia/cuda:12.4.1-devel-ubuntu22.04`

---

### 🟡 RTX 4090 (24GB)

> **Single Tab:** 2× stacked (48GB total)
> **Double Tab:** 4× stacked (96GB total)
> **Price for 1 tab (hourly):** ~$0.70/h
> **Price for 2 tabs (hourly):** ~$1.40/h
> **Est. Vidgen (3.5s video):** ~25-35s
> **Est. Picgen (1 image):** ~5-7s
> **Template(s):**
> - `simplepodai/ubuntu22.04-devel:cuda126`
> - `nvidia/cuda:12.4.1-devel-ubuntu22.04`

---

### 🔵 RTX 5090 (32GB)

> **Single Tab:** 2× stacked (64GB total)
> **Double Tab:** 3× stacked (96GB total)
> **Price for 1 tab (hourly):** ~$1.00/h
> **Price for 2 tabs (hourly):** ~$1.50/h
> **Est. Vidgen (3.5s video):** ~18-25s
> **Est. Picgen (1 image):** ~4-6s
> **Template(s):**
> - `simplepodai/ubuntu22.04-devel:cuda128`

---

### 🟣 RTX PRO 6000 MIG 2g.48gb (48GB)

> **Single Tab:** 1× (no stacking needed)
> **Double Tab:** 2× stacked (96GB total)
> **Price for 1 tab (hourly):** ~$0.79/h
> **Price for 2 tabs (hourly):** ~$1.58/h
> **Est. Vidgen (3.5s video):** ~15-22s
> **Est. Picgen (1 image):** ~4-6s
> **Template(s):**
> - `simplepodai/ubuntu22.04-devel:cuda128`

---

### ⚫ RTX PRO 6000 Blackwell (95GB)

> **Single Tab:** 1× (no stacking needed)
> **Double Tab:** 1× (both models fit on a single card!)
> **Price for 1 tab (hourly):** ~$1.00/h
> **Price for 2 tabs (hourly):** ~$1.00/h (same card, no extra cost)
> **Est. Vidgen (3.5s video):** ~10-14s
> **Est. Picgen (1 image):** ~2.5-3s
> **Template(s):**
> - `simplepodai/ubuntu22.04-devel:cuda128`

---

### 📝 GPU Notes

- App auto-detects GPU count, VRAM, and selects optimal mode automatically
- No manual configuration needed — just run `python3 app.py`
- SageAttention installs automatically via `setup.sh` for extra speed
- Stacked mode splits model layers across all GPUs (accelerate `balanced` device_map)
- Concurrent mode (≥48GB per GPU) pins one model per GPU — both tabs instant
- Generation times assume all optimizations active (SageAttention + torch.compile + TF32)
