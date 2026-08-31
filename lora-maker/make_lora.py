"""
make_lora.py — Wan 2.2 I2V Subject LoRA Trainer  (Gradio web UI)
Trains identity / body-part / object LoRAs (high + low noise) from photo folders.
Targets WAMU v2  WanImageToVideoPipeline with dual transformers:
  _high.safetensors  → transformer   (high-noise expert)
  _low.safetensors   → transformer_2 (low-noise expert)

Requirements:
  - Ubuntu 22.04+, Python 3.10+, Git, NVIDIA GPU with CUDA 12.8 already installed
  - gradio (installed via requirements.txt in the newgen stack)
  - No tkinter / display required — runs headless on SimplePod VPS

musubi-tuner is cloned automatically on first training run.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import threading
import queue
from pathlib import Path
import warnings

# Suppress deprecation warnings from gradio/starlette and other noisy libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr

# ─── Constants ────────────────────────────────────────────────────────────────

APP_TITLE   = "LoRA Maker — WAMU v2 / Wan 2.2 I2V Lightning"
APP_VERSION = "2.0"

TOOLS_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "LoRAMaker"
TUNER_DIR = TOOLS_DIR / "musubi-tuner"
CACHE_DIR = TOOLS_DIR / "model_cache"
VENV_DIR  = TOOLS_DIR / "venv"

WAN_MODEL_REPO = "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING"

SUBJECT_PRESETS = {
    "Person (me / identity)": {
        "trigger_base":     "ohwx_man",
        "aliases":          "add me, the man, dev, i, me",
        "description":      "a man's face and body",
        "caption_template": "{trigger}, a man looking at the camera, neutral expression, natural lighting",
    },
    "Person (woman)": {
        "trigger_base":     "ohwx_woman",
        "aliases":          "add her, the woman, she, her",
        "description":      "a woman's face and body",
        "caption_template": "{trigger}, a woman looking at the camera, neutral expression, natural lighting",
    },
    "Body part — penis": {
        "trigger_base":     "ohwx_penis",
        "aliases":          "the penis, penis, dick, cock",
        "description":      "explicit body part close-up",
        "caption_template": "{trigger}, realistic close-up photo",
    },
    "Body part — other": {
        "trigger_base":     "ohwx_bodypart",
        "aliases":          "",
        "description":      "body part close-up",
        "caption_template": "{trigger}, realistic close-up photo",
    },
    "Object / thing": {
        "trigger_base":     "ohwx_object",
        "aliases":          "",
        "description":      "a specific object",
        "caption_template": "{trigger}, a photo of the object, natural lighting",
    },
    "Custom": {
        "trigger_base":     "ohwx_custom",
        "aliases":          "",
        "description":      "custom subject",
        "caption_template": "{trigger}",
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def sanitize_trigger(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")

def parse_aliases(raw: str) -> list:
    return [a.strip() for a in raw.split(",") if a.strip()]

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts)

def venv_python() -> Path:
    return VENV_DIR / "bin" / "python"

def ensure_venv(log_q: queue.Queue):
    py = venv_python()
    if py.exists():
        return
    log_q.put("Creating Python virtual environment…")
    subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    log_q.put("✓ venv created")

def pip_install(packages: list, venv_py: Path, log_q: queue.Queue):
    log_q.put(f"pip install {' '.join(packages[:3])}{'…' if len(packages) > 3 else ''}")
    result = subprocess.run(
        [str(venv_py), "-m", "pip", "install", "--quiet", "--upgrade"] + packages,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install failed (exit {result.returncode}):\n{result.stderr[-600:]}"
        )

# ─── Training engine ──────────────────────────────────────────────────────────

class TrainingSession:
    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.log_q      = queue.Queue()
        self.progress_q = queue.Queue()
        self.done       = threading.Event()
        self.cancel     = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self.cancel.set()

    def _log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_q.put(f"[{stamp}]  {msg}")

    def _progress(self, pct: int, label: str = ""):
        self.progress_q.put((pct, label))

    def _run(self):
        try:
            self._phase_env()
            if self.cancel.is_set():
                self._log("⚠  Cancelled.")
                return
            all_entries: dict = {}
            all_files:   list = []
            subjects = self.cfg["subjects"]
            n = len(subjects)
            for idx, subj in enumerate(subjects):
                self._log(f"\n═══ Subject {idx+1}/{n}: {subj['trigger']} ═══")
                if self.cancel.is_set():
                    break
                img_dir, video_dir = self._phase_caption(subj)
                if self.cancel.is_set():
                    break
                out_high, out_low = self._phase_train(subj, img_dir, video_dir, idx, n)
                if self.cancel.is_set():
                    break
                entry, files = self._phase_export(subj, out_high, out_low)
                all_entries.update(entry)
                all_files.extend(files)
            self._finish(all_entries, all_files)
        except Exception as exc:
            import traceback
            self._log(f"✗  ERROR: {exc}")
            self._log(traceback.format_exc())
        finally:
            self.done.set()

    # ── Phase 1: environment ──────────────────────────────────────────────────

    def _phase_env(self):
        self._log("── Phase 1: Setting up environment ──")
        self._progress(2, "Environment")
        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ensure_venv(self.log_q)
        py = venv_python()
        # Install all packages needed by musubi-tuner (non-torch deps from pyproject.toml)
        # plus general utilities — doing this upfront avoids repeated ModuleNotFoundError
        # crashes when individual packages are missing from the venv.
        pip_install([
            # general utilities
            "pillow", "requests", "tqdm", "toml",
            # huggingface ecosystem
            "huggingface_hub", "accelerate", "diffusers", "transformers", "safetensors",
            # musubi-tuner required deps (from pyproject.toml dependencies[])
            "voluptuous",   # dataset/config_utils.py
            "easydict",     # wan/configs/wan_i2v_14B.py
            "einops",       # various wan modules
            "bitsandbytes", # 8-bit optimiser
            "ftfy",         # text normalisation (Wan2.1+)
            "sentencepiece", # FLUX / tokenizers
            "av",           # video frame reading
            "opencv-python", # image processing
        ], py, self.log_q)
        self._log("✓ Core packages installed")

        res = subprocess.run(
            [str(py), "-c",
             "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'NO_CUDA')"],
            capture_output=True, text=True,
        )
        cuda_ok = "CUDA_OK" in res.stdout

        if cuda_ok:
            self._log("✓ PyTorch + CUDA already available in venv — skipping install")
        else:
            sys_res = subprocess.run(
                ["python3", "-c",
                 "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'NO_CUDA')"],
                capture_output=True, text=True,
            )
            if "CUDA_OK" in sys_res.stdout:
                self._log("System PyTorch has CUDA — rebuilding venv with --system-site-packages…")
                shutil.rmtree(str(VENV_DIR), ignore_errors=True)
                subprocess.run(
                    [sys.executable, "-m", "venv", "--system-site-packages", str(VENV_DIR)],
                    check=True,
                )
                pip_install([
                    "pillow", "requests", "tqdm", "toml",
                    "huggingface_hub", "accelerate", "diffusers", "transformers", "safetensors",
                    "voluptuous", "easydict", "einops", "bitsandbytes",
                    "ftfy", "sentencepiece", "av", "opencv-python",
                ], venv_python(), self.log_q)
                self._log("✓ Venv now inherits system PyTorch + CUDA 12.8")
            else:
                self._log("Installing PyTorch cu128 wheels (first-time only)…")
                pip_install(
                    ["torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu128"],
                    py, self.log_q,
                )
                self._log("✓ PyTorch + CUDA 12.8 installed")

        # Determine musubi-tuner state:
        #   - .git present          → valid clone; try git pull
        #   - dir exists, no .git   → partial/broken clone from a previous failed run; wipe and reclone
        #   - dir absent            → fresh clone
        is_valid_clone = (TUNER_DIR / ".git").exists()
        is_broken_dir  = TUNER_DIR.exists() and not is_valid_clone

        if is_broken_dir:
            self._log("⚠  musubi-tuner directory exists but is not a git repo "
                      "(likely a partial clone from a previous failed run) — removing and recloning…")
            shutil.rmtree(str(TUNER_DIR), ignore_errors=True)

        if is_valid_clone:
            self._log("Updating musubi-tuner…")
            pull = subprocess.run(
                ["git", "-C", str(TUNER_DIR), "pull", "--ff-only"],
                capture_output=True, text=True,
            )
            if pull.returncode == 0:
                self._log("✓ musubi-tuner up to date")
            else:
                self._log(f"  git pull returned {pull.returncode} — continuing with existing clone")
        else:
            self._log("Cloning musubi-tuner…")
            clone = subprocess.run(
                ["git", "clone", "--depth=1",
                 "https://github.com/kohya-ss/musubi-tuner.git", str(TUNER_DIR)],
                capture_output=True, text=True,
            )
            if clone.returncode != 0:
                raise RuntimeError(
                    f"git clone failed (exit {clone.returncode}):\n{clone.stderr[-600:]}"
                )
            self._log("✓ musubi-tuner cloned")

        req_file     = TUNER_DIR / "requirements.txt"
        pyproject    = TUNER_DIR / "pyproject.toml"
        skip_prefixes = ("torch", "torchvision", "torchaudio")

        if pyproject.exists():
            # Modern musubi-tuner uses pyproject.toml — install the package itself
            # in editable mode so all its declared dependencies are resolved by pip.
            # We pass --no-deps for torch/torchvision (already in venv) via constraint.
            self._log("Installing musubi-tuner via pyproject.toml (editable, no-torch)…")
            constraint_file = TOOLS_DIR / "musubi_no_torch.txt"
            constraint_file.write_text(
                "torch\ntorchvision\ntorchaudio\n", encoding="utf-8"
            )
            result = subprocess.run(
                [str(venv_python()), "-m", "pip", "install", "--quiet",
                 "--constraint", str(constraint_file),
                 "-e", str(TUNER_DIR)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                # editable install may fail if hatchling isn't available yet; fall back
                # to just installing the deps listed explicitly above — they already cover
                # everything musubi-tuner needs for Wan2.x LoRA training.
                self._log(f"  (editable install returned {result.returncode} — "
                          "deps already installed above, continuing)")
            else:
                self._log("✓ musubi-tuner package installed")
        elif req_file.exists():
            self._log("Installing musubi-tuner requirements (torch lines filtered)…")
            filtered = TOOLS_DIR / "musubi_reqs_filtered.txt"
            lines = req_file.read_text(encoding="utf-8").splitlines()
            kept  = [l for l in lines
                     if not any(l.strip().lower().startswith(p) for p in skip_prefixes)]
            filtered.write_text("\n".join(kept), encoding="utf-8")
            result = subprocess.run(
                [str(venv_python()), "-m", "pip", "install", "--quiet", "-r", str(filtered)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pip install musubi-tuner requirements failed "
                    f"(exit {result.returncode}):\n{result.stderr[-600:]}"
                )
            self._log("✓ musubi-tuner requirements installed")
        else:
            self._log("  (no requirements.txt or pyproject.toml found — skipping)")

        self._progress(10, "Environment done")

    # ── Phase 2: caption ──────────────────────────────────────────────────────

    def _phase_caption(self, subj: dict) -> tuple:
        """
        Copy images and/or videos into the dataset staging directory, write
        caption .txt files alongside them, and return:
          (image_dataset_dir_or_None, video_dataset_dir_or_None)
        Both dirs live under TOOLS_DIR/dataset/<trigger>/.
        """
        self._log(f"── Phase 2: Captioning for '{subj['trigger']}' ──")
        self._progress(15, "Captioning")
        trigger  = subj["trigger"]
        template = subj.get("caption_template") or "{trigger}"
        if "{trigger}" not in template:
            template = "{trigger}, " + template

        img_exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        caption_variants = [
            template.format(trigger=trigger),
            template.format(trigger=trigger) + ", natural lighting",
            template.format(trigger=trigger) + ", sharp focus, realistic photo",
            template.format(trigger=trigger) + ", three-quarter view",
            template.format(trigger=trigger) + ", candid photo, ambient light",
            template.format(trigger=trigger) + ", close-up, soft lighting",
            template.format(trigger=trigger) + ", warm light, portrait",
            template.format(trigger=trigger) + ", side view, natural light",
        ]

        img_dir_out   = None
        video_dir_out = None

        # ── Images ────────────────────────────────────────────────────────────
        photo_folder = subj.get("folder", "")
        if photo_folder:
            photos = Path(photo_folder)
            if photos.exists():
                images = [f for f in photos.iterdir() if f.suffix.lower() in img_exts]
                if images:
                    img_dir_out = TOOLS_DIR / "dataset" / trigger / "images"
                    img_dir_out.mkdir(parents=True, exist_ok=True)
                    self._log(f"Captioning {len(images)} images…")
                    for i, img_path in enumerate(images):
                        dest = img_dir_out / img_path.name
                        shutil.copy2(img_path, dest)
                        dest.with_suffix(".txt").write_text(
                            caption_variants[i % len(caption_variants)], encoding="utf-8"
                        )
                        if (i + 1) % 5 == 0 or (i + 1) == len(images):
                            self._log(f"  Captioned {i+1}/{len(images)}")
                    self._log(f"✓ {len(images)} images captioned")

        # ── Videos ────────────────────────────────────────────────────────────
        video_folder = subj.get("video_folder", "")
        if video_folder:
            vfolder = Path(video_folder)
            if vfolder.exists():
                videos = [f for f in vfolder.iterdir() if f.suffix.lower() in video_exts]
                if videos:
                    video_dir_out = TOOLS_DIR / "dataset" / trigger / "videos"
                    video_dir_out.mkdir(parents=True, exist_ok=True)
                    self._log(f"Captioning {len(videos)} videos…")
                    for i, vid_path in enumerate(videos):
                        dest = video_dir_out / vid_path.name
                        shutil.copy2(vid_path, dest)
                        dest.with_suffix(".txt").write_text(
                            caption_variants[i % len(caption_variants)], encoding="utf-8"
                        )
                        if (i + 1) % 5 == 0 or (i + 1) == len(videos):
                            self._log(f"  Captioned {i+1}/{len(videos)}")
                    self._log(f"✓ {len(videos)} videos captioned")

        if img_dir_out is None and video_dir_out is None:
            raise RuntimeError(
                "No images or videos found. "
                "Check that the photo/video folders exist and contain supported files."
            )

        self._progress(25, "Captioning done")
        return img_dir_out, video_dir_out

    # ── Phase 3: train ────────────────────────────────────────────────────────
    #
    # musubi-tuner wan_train_network.py requires individual raw model files:
    #   --dit           low-noise DiT  (.safetensors)
    #   --dit_high_noise high-noise DiT (.safetensors)
    #   --vae           WanVAE          (.safetensors or .pth)
    #   --t5            T5 encoder      (.pth)
    #
    # These are NOT a diffusers pipeline dir. app.py uses diffusers format
    # (cached by HF Hub). We download the raw Comfy-Org repackaged files
    # separately into CACHE_DIR/wan_raw/.
    #
    # Training is a 3-step pipeline:
    #   1. wan_cache_latents.py          — encode images → latent .npz files
    #   2. wan_cache_text_encoder_outputs.py — encode captions → text .npz files
    #   3. wan_train_network.py          — train the LoRA (both transformers at once)
    #
    # One training run produces ONE LoRA file covering both transformers.
    # We then split it into _high and _low by convention so app.py can load
    # them onto the right transformer — both files are identical copies of
    # the same combined LoRA (app.py's load_loras_to_pipeline handles routing).

    # ── Model file helpers ────────────────────────────────────────────────────

    @staticmethod
    def _hf_download(py: Path, repo: str, filename: str, dest: Path,
                     log_q, label: str = "") -> Path:
        """Download a single file from HF Hub into dest/ and return its path."""
        out = dest / Path(filename).name
        if out.exists():
            return out
        label = label or filename
        log_q.put(f"  Downloading {label}…")
        script = (
            "from huggingface_hub import hf_hub_download\n"
            "import shutil, pathlib\n"
            f"src = hf_hub_download(repo_id={repo!r}, filename={filename!r})\n"
            f"dst = pathlib.Path({str(dest)!r}) / pathlib.Path({filename!r}).name\n"
            "dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "shutil.copy2(src, dst)\n"
            'print("HF_DL_DONE:" + str(dst))\n'
        )
        result = subprocess.run([str(py), "-c", script],
                                capture_output=True, text=True)
        if "HF_DL_DONE:" not in result.stdout:
            raise RuntimeError(
                f"Failed to download {filename} from {repo}:\n"
                f"{result.stderr[-600:]}"
            )
        return out

    def _ensure_wan_raw_models(self, py: Path) -> dict:
        """
        Ensure raw Wan2.2 model files exist in CACHE_DIR/wan_raw/ and return
        a dict with keys: dit_low, dit_high, vae, t5.

        Download strategy:
          DiT weights  — Comfy-Org/Wan_2.2_ComfyUI_Repackaged (fp16)
          VAE          — Comfy-Org/Wan_2.1_ComfyUI_repackaged  (Wan2.1 VAE works for 2.2 14B)
          T5           — Wan-AI/Wan2.1-I2V-14B-720P
        """
        raw_dir = CACHE_DIR / "wan_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # ── DiT weights (Wan2.2 I2V 14B, fp16) ───────────────────────────────
        DIT_REPO   = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
        DIT_HIGH   = "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
        DIT_LOW    = "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

        # ── VAE (Wan2.1 VAE — works for 2.2 14B, NOT 2.2 5B) ─────────────────
        VAE_REPO   = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        VAE_FILE   = "split_files/vae/wan_2.1_vae.safetensors"

        # ── T5 text encoder ───────────────────────────────────────────────────
        T5_REPO    = "Wan-AI/Wan2.1-I2V-14B-720P"
        T5_FILE    = "models_t5_umt5-xxl-enc-bf16.pth"

        self._log("Checking Wan2.2 raw model files…")

        dit_high_path = self._hf_download(py, DIT_REPO, DIT_HIGH, raw_dir,
                                           self.log_q, "DiT high-noise (Wan2.2 I2V 14B fp16)")
        if self.cancel.is_set():
            return {}

        dit_low_path  = self._hf_download(py, DIT_REPO, DIT_LOW,  raw_dir,
                                           self.log_q, "DiT low-noise  (Wan2.2 I2V 14B fp16)")
        if self.cancel.is_set():
            return {}

        vae_path      = self._hf_download(py, VAE_REPO, VAE_FILE, raw_dir,
                                           self.log_q, "VAE (Wan2.1 vae — compatible with 2.2 14B)")
        if self.cancel.is_set():
            return {}

        t5_path       = self._hf_download(py, T5_REPO,  T5_FILE,  raw_dir,
                                           self.log_q, "T5 text encoder (umt5-xxl bf16)")
        if self.cancel.is_set():
            return {}

        self._log("✓ All raw model files ready")
        return {
            "dit_high": dit_high_path,
            "dit_low":  dit_low_path,
            "vae":      vae_path,
            "t5":       t5_path,
        }

    def _phase_train(self, subj: dict, img_dir, video_dir, subj_idx: int, total_subjects: int):
        self._log(f"── Phase 3: Training LoRA for '{subj['trigger']}' ──")
        cfg     = self.cfg
        trigger = subj["trigger"]
        rank    = cfg["rank"]
        steps   = cfg["steps"]
        lr      = cfg["lr"]
        py      = venv_python()
        work_dir = TOOLS_DIR / "workdir" / trigger
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir = work_dir / "output"
        out_dir.mkdir(exist_ok=True)

        # Separate cache dirs — musubi-tuner requires a unique cache_directory per [[datasets]]
        img_cache_dir   = work_dir / "latent_cache_images"
        video_cache_dir = work_dir / "latent_cache_videos"
        if img_dir:
            img_cache_dir.mkdir(exist_ok=True)
        if video_dir:
            video_cache_dir.mkdir(exist_ok=True)

        # ── Dataset TOML ─────────────────────────────────────────────────────
        # One flat [[datasets]] block per source type (images / videos).
        # resolution=832 = 480p landscape (Wan2.2 I2V 480p standard width).
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        toml_lines = [
            "[general]",
            'caption_extension = ".txt"',
            "enable_bucket = true",
            "bucket_no_upscale = true",
            "",
        ]
        if img_dir:
            n_images    = count_images(img_dir)
            img_repeats = max(1, 200 // max(1, n_images))
            toml_lines += [
                "[[datasets]]",
                "resolution = 832",
                f'image_directory = "{str(img_dir).replace(chr(92), "/")}"',
                f"num_repeats = {img_repeats}",
                f'cache_directory = "{str(img_cache_dir).replace(chr(92), "/")}"',
                "",
            ]
        if video_dir:
            n_videos    = sum(1 for f in video_dir.iterdir() if f.suffix.lower() in video_exts)
            vid_repeats = max(1, 50 // max(1, n_videos))
            toml_lines += [
                "[[datasets]]",
                "resolution = 832",
                f'video_directory = "{str(video_dir).replace(chr(92), "/")}"',
                # [1, 25] = single-frame stills + ~1.5s motion clips (at 16fps)
                "target_frames = [1, 25]",
                'frame_extraction = "head"',
                f"num_repeats = {vid_repeats}",
                f'cache_directory = "{str(video_cache_dir).replace(chr(92), "/")}"',
                "",
            ]
        dataset_toml = work_dir / "dataset.toml"
        dataset_toml.write_text("\n".join(toml_lines), encoding="utf-8")

        # ── Raw model files ───────────────────────────────────────────────────
        models = self._ensure_wan_raw_models(py)
        if self.cancel.is_set() or not models:
            return work_dir / f"{trigger}.safetensors", work_dir / f"{trigger}.safetensors"

        base_span = 65 / max(1, total_subjects)
        subj_base = 25 + subj_idx * base_span

        # ── Step 1: cache latents ─────────────────────────────────────────────
        self._log("  Step 1/3: Caching latents…")
        self._progress(int(subj_base + base_span * 0.05), "Caching latents")
        self._run_subprocess(
            py,
            [
                str(py),
                str(TUNER_DIR / "src" / "musubi_tuner" / "wan_cache_latents.py"),
                "--dataset_config", str(dataset_toml),
                "--vae",            str(models["vae"]),
                "--i2v",
                "--vae_cache_cpu",
            ],
            label="cache_latents",
            progress_base=int(subj_base + base_span * 0.05),
            progress_span=int(base_span * 0.15),
        )
        if self.cancel.is_set():
            return work_dir / f"{trigger}.safetensors", work_dir / f"{trigger}.safetensors"

        # ── Step 2: cache text encoder outputs ────────────────────────────────
        self._log("  Step 2/3: Caching text encoder outputs…")
        self._progress(int(subj_base + base_span * 0.20), "Caching text encodings")
        self._run_subprocess(
            py,
            [
                str(py),
                str(TUNER_DIR / "src" / "musubi_tuner" / "wan_cache_text_encoder_outputs.py"),
                "--dataset_config", str(dataset_toml),
                "--t5",             str(models["t5"]),
                "--batch_size",     "4",
            ],
            label="cache_text",
            progress_base=int(subj_base + base_span * 0.20),
            progress_span=int(base_span * 0.10),
        )
        if self.cancel.is_set():
            return work_dir / f"{trigger}.safetensors", work_dir / f"{trigger}.safetensors"

        # ── Step 3: train ─────────────────────────────────────────────────────
        self._log(f"  Step 3/3: Training ({steps} steps, rank {rank}, lr {lr})…")
        self._log("  Training both transformers in one pass (--dit low + --dit_high_noise high)")
        self._progress(int(subj_base + base_span * 0.30), "Training")

        out_name = trigger  # musubi-tuner appends .safetensors
        combined = out_dir / f"{trigger}.safetensors"

        train_script = TUNER_DIR / "src" / "musubi_tuner" / "wan_train_network.py"
        if not train_script.exists():
            raise RuntimeError(
                f"Training script not found: {train_script}\n"
                "Expected musubi-tuner layout: src/musubi_tuner/wan_train_network.py\n"
                f"Contents of {TUNER_DIR}:\n"
                + "\n".join(str(p) for p in sorted(TUNER_DIR.rglob("*.py"))[:30])
            )

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "fp16",
            str(train_script),
            # ── model ──
            "--task",              "i2v-A14B",
            "--dit",               str(models["dit_low"]),
            "--dit_high_noise",    str(models["dit_high"]),
            "--fp8_base",
            "--sdpa",
            # ── dataset ──
            "--dataset_config",    str(dataset_toml),
            # ── LoRA ──
            "--network_module",    "networks.lora_wan",
            "--network_dim",       str(rank),
            "--network_alpha",     str(rank // 2),
            # ── optimiser ──
            "--optimizer_type",    "adamw8bit",
            "--learning_rate",     lr,
            "--lr_scheduler",      "cosine_with_restarts",
            "--max_train_steps",   str(steps),
            "--gradient_checkpointing",
            # With app.py stopped, full GPU VRAM is available.
            # offload_inactive_dit swaps the idle DiT to CPU between timestep regions —
            # much faster than blocks_to_swap since it only swaps once per step, not per block.
            # If you see OOM (e.g. app.py is running), replace with: "--blocks_to_swap", "20"
            "--offload_inactive_dit",
            # ── Wan2.2 I2V flow shift ──
            # fp16 DiT weights require mixed_precision=fp16 (bf16 is rejected by musubi-tuner)
            # flow_shift=3.0 for 480p I2V (official default); 5.0 for higher resolutions
            "--timestep_sampling",    "shift",
            "--discrete_flow_shift",  "3.0",
            "--mixed_precision",      "fp16",
            "--seed",              "42",
            "--save_every_n_steps", str(max(100, steps // 5)),
            # ── output ──
            "--output_dir",        str(out_dir),
            "--output_name",       out_name,
        ]

        self._run_subprocess(
            py, cmd, label="train",
            progress_base=int(subj_base + base_span * 0.30),
            progress_span=int(base_span * 0.65),
            cwd=str(TUNER_DIR),
        )

        # Locate the output file — musubi-tuner writes <output_dir>/<output_name>.safetensors
        # or with a step suffix for intermediate saves.  Find the most recent.
        candidates = sorted(
            out_dir.glob(f"{trigger}*.safetensors"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"Training finished but no .safetensors found in {out_dir}.\n"
                "Check the log for where musubi-tuner saved its output."
            )
        combined = candidates[0]
        self._log(f"✓ LoRA saved: {combined.name}")

        # app.py expects _high and _low files (one LoRA file covers both transformers,
        # so we expose the same file under both names — app.py loads each onto the
        # correct transformer via its load_loras_to_pipeline() routing logic).
        out_high = work_dir / f"{trigger}_high.safetensors"
        out_low  = work_dir / f"{trigger}_low.safetensors"
        shutil.copy2(combined, out_high)
        shutil.copy2(combined, out_low)
        self._log(f"  → {out_high.name}  (for pipe.transformer   — high-noise expert)")
        self._log(f"  → {out_low.name}   (for pipe.transformer_2 — low-noise expert)")
        return out_high, out_low

    def _run_subprocess(self, py, cmd, label="subprocess",
                        progress_base=25, progress_span=35, cwd=None):
        """Run cmd, stream every output line to the log, raise on non-zero exit."""
        self._log(f"  CMD: {' '.join(str(c) for c in cmd)}")
        env = {**os.environ}
        # Ensure the venv's site-packages are on PYTHONPATH for accelerate/scripts
        site_pkg = VENV_DIR / "lib"
        sp_dirs = list(site_pkg.glob("python*/site-packages")) if site_pkg.exists() else []
        py_paths = [str(d) for d in sp_dirs[:1]]
        # musubi-tuner is not pip-installed; its scripts import the musubi_tuner
        # package relative to the repo checkout, so put its src/ on PYTHONPATH.
        tuner_src = TUNER_DIR / "src"
        if tuner_src.exists():
            py_paths.append(str(tuner_src))
        if py_paths:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = ":".join(py_paths + ([existing] if existing else []))
        # Reduce CUDA memory fragmentation — critical when another process (app.py)
        # is already occupying most of GPU VRAM.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd or str(TUNER_DIR),
            env=env,
        )

        tail_lines: list[str] = []

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            self._log(f"  {line}")
            tail_lines.append(line)
            if len(tail_lines) > 40:
                tail_lines.pop(0)
            # Progress parsing for training steps
            if label == "train" and "step" in line.lower() and "/" in line:
                try:
                    for part in line.split():
                        if "/" in part and part.replace("/", "").isdigit():
                            cur, tot = part.split("/")
                            pct = progress_base + int(int(cur) / int(tot) * progress_span)
                            self._progress(pct, f"Training step {part}")
                            break
                except Exception:
                    pass
            if self.cancel.is_set():
                proc.terminate()
                return

        proc.wait()

        if proc.returncode not in (0, None) and not self.cancel.is_set():
            tail = "\n".join(tail_lines[-20:])
            raise RuntimeError(
                f"{label} subprocess exited with code {proc.returncode}.\n"
                f"Last output:\n{tail}"
            )

    # ── Phase 4: export ───────────────────────────────────────────────────────

    def _phase_export(self, subj: dict, out_high: Path, out_low: Path):
        self._log(f"── Phase 4: Exporting '{subj['trigger']}' ──")
        self._progress(92, "Exporting")
        trigger  = subj["trigger"]
        weight   = self.cfg["weight"]
        out_dir  = Path(self.cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        final_high = out_dir / f"{trigger}_high.safetensors"
        final_low  = out_dir / f"{trigger}_low.safetensors"
        shutil.copy2(out_high, final_high)
        shutil.copy2(out_low,  final_low)
        self._log(f"✓ Copied to {out_dir}")
        aliases = subj.get("aliases", [])
        entry = {
            trigger: {
                "display_name":           subj.get("description") or f"LoRA: {trigger}",
                "description":            subj.get("description") or "",
                "high_filename":          final_high.name,
                "low_filename":           final_low.name,
                "trigger_prompt":         trigger,
                "trigger_aliases":        aliases,
                "prompt_mode":            "append",
                "high_weight":            weight,
                "low_weight":             weight,
                "recommended_steps":      None,
                "recommended_flow_shift": None,
                "notes": (
                    f"Trigger: '{trigger}'. "
                    + (f"Aliases: {', '.join(aliases)}. " if aliases else "")
                    + "Reduce to 0.6–0.7 if it overpowers other LoRAs."
                ),
                "auto_enabled": False,
                "tags": ["subject", "personal"],
            }
        }
        return entry, [str(final_high), str(final_low)]

    # ── Finish ────────────────────────────────────────────────────────────────

    def _finish(self, all_entries: dict, all_files: list):
        self._progress(100, "Done!")
        json_str = json.dumps(all_entries, indent=2, ensure_ascii=False)
        self._log("")
        self._log("🎉  Training complete!")
        for f in all_files:
            self._log(f"    {f}")
        self._log("")
        self._log("HOW TO USE YOUR TRIGGER ALIASES:")
        for key, entry in all_entries.items():
            aliases = entry.get("trigger_aliases", [])
            if aliases:
                self._log(f"  '{key}': any of — {', '.join(aliases)}")
        self._log("  Add these to your prompts to auto-activate the LoRA.")
        self._log("")
        self._log("── loras.json entry ──")
        self._log(json_str)
        json_path = self.cfg.get("json_path")
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                existing.update(all_entries)
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(existing, fh, indent=2, ensure_ascii=False)
                self._log(f"✓ Auto-updated {json_path}")
            except Exception as e:
                self._log(f"⚠  Could not update loras.json: {e}")
        self.log_q.put(f"__JSON_READY__{json_str}")
        self.log_q.put(f"__FILES_READY__{json.dumps(all_files)}")


# ─── Global state ─────────────────────────────────────────────────────────────

_session = None   # type: TrainingSession | None


# ─── Gradio callbacks ─────────────────────────────────────────────────────────

def preset_defaults(preset_name: str):
    p = SUBJECT_PRESETS.get(preset_name, {})
    return (
        p.get("trigger_base", "ohwx_custom"),
        p.get("aliases", ""),
        p.get("description", ""),
        p.get("caption_template", "{trigger}"),
    )

def validate_subject(folder: str, trigger: str) -> str:
    if not folder:
        return "—"
    p = Path(folder)
    if not p.exists():
        return f"⚠  Folder not found: {folder}"
    n = count_images(p)
    if n < 1:
        return f"⚠  No images found in {folder}"
    t = sanitize_trigger(trigger)
    if len(t) < 3:
        return "⚠  Trigger word must be at least 3 characters."
    return f"✓  {n} images found. Trigger: '{t}'"

def validate_video_folder(video_folder: str) -> str:
    if not video_folder:
        return "—"
    p = Path(video_folder)
    if not p.exists():
        return f"⚠  Folder not found: {video_folder}"
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [f for f in p.iterdir() if f.suffix.lower() in video_exts]
    if not videos:
        return f"⚠  No video files found (mp4/mov/avi/mkv/webm)"
    return f"✓  {len(videos)} video(s) found"

def start_training(folder, video_folder, trigger, aliases, description, caption_template,
                   rank, steps, lr, weight, output_dir, json_path):
    global _session
    if _session and not _session.done.is_set():
        return "⚠  Training already running — cancel it first.", "", 0, "Already running"
    # At least one of photo folder or video folder must be provided
    has_photos = bool(folder and Path(folder).exists() and count_images(Path(folder)) > 0)
    has_videos = False
    if video_folder and Path(video_folder).exists():
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        has_videos = any(f.suffix.lower() in video_exts for f in Path(video_folder).iterdir())
    if not has_photos and not has_videos:
        return "⚠  Provide at least one photo or video folder with content.", "", 0, "Validation error"
    trig = sanitize_trigger(trigger)
    if len(trig) < 3:
        return "⚠  Trigger must be ≥ 3 characters.", "", 0, "Validation error"
    if not output_dir:
        return "⚠  Output directory is required.", "", 0, "Validation error"
    cfg = {
        "rank":       int(rank),
        "steps":      int(steps),
        "lr":         str(lr),
        "weight":     float(weight),
        "output_dir": output_dir,
        "json_path":  json_path if json_path else None,
        "subjects": [{
            "folder":           folder if has_photos else "",
            "video_folder":     video_folder if has_videos else "",
            "trigger":          trig,
            "aliases":          parse_aliases(aliases),
            "description":      description,
            "caption_template": caption_template or "{trigger}",
        }],
    }
    _session = TrainingSession(cfg)
    _session.start()
    return "✅  Training started — switch to Training Log tab.", "", 0, "Training…"

def cancel_training():
    global _session
    if _session and not _session.done.is_set():
        _session.stop()
        return "⚠  Cancel requested…"
    return "No training running."

def poll_log(current_log: str, current_pct: float, current_json: str):
    global _session
    if _session is None:
        return current_log, current_pct, "Idle", current_json
    new_lines  = []
    new_json   = current_json
    new_pct    = current_pct
    new_status = "Training…" if not _session.done.is_set() else "Done"
    while True:
        try:
            line = _session.log_q.get_nowait()
            if line.startswith("__JSON_READY__"):
                new_json = line[len("__JSON_READY__"):]
            elif line.startswith("__FILES_READY__"):
                pass
            else:
                new_lines.append(line)
        except queue.Empty:
            break
    while True:
        try:
            pct, label = _session.progress_q.get_nowait()
            new_pct    = pct
            new_status = label or new_status
        except queue.Empty:
            break
    if _session.done.is_set():
        new_status = "✅  Done!" if not _session.cancel.is_set() else "Cancelled"
    sep = "\n" if current_log and new_lines else ""
    new_log = current_log + sep + "\n".join(new_lines)
    return new_log, new_pct, new_status, new_json


# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_ui():
    default_output = str(Path.home() / "loras")
    preset_names   = list(SUBJECT_PRESETS.keys())

    # gr.Blocks with no theme= argument — theme goes in launch() for Gradio 6+
    with gr.Blocks(title=APP_TITLE) as demo:

        gr.Markdown(f"""
# 🎬 {APP_TITLE}
Trains personal identity / subject LoRAs for WAMU v2 (Wan 2.2 I2V Lightning).  
Produces `<trigger>_high.safetensors` + `<trigger>_low.safetensors` — drop in `loras/` and add JSON to `loras.json`.
        """)

        # ── Setup & Train tab ─────────────────────────────────────────────────
        with gr.Tab("⚙️  Setup & Train"):
            gr.Markdown("### Subject")
            preset_dd = gr.Dropdown(
                choices=preset_names, value=preset_names[0],
                label="Subject preset",
                info="Fills in sensible defaults — customise below.",
            )
            with gr.Row():
                folder_tb = gr.Textbox(
                    label="Photo folder (absolute server path)",
                    value="/root/devin",
                    placeholder="/root/photos/me",
                    info="JPG/PNG/WEBP/BMP. Leave blank if using videos only.",
                    scale=3,
                )
                folder_status = gr.Textbox(label="Folder status", interactive=False, scale=1)
            with gr.Row():
                video_folder_tb = gr.Textbox(
                    label="Video folder (optional — absolute server path)",
                    value="",
                    placeholder="/root/devin_clips",
                    info="MP4/MOV/AVI/MKV/WEBM. 480p recommended. Can be used alone or with photos.",
                    scale=3,
                )
                video_folder_status = gr.Textbox(label="Video status", interactive=False, scale=1)
            with gr.Row():
                trigger_tb = gr.Textbox(
                    label="Trigger word", value="my_self",
                    info="Rare unique token, no spaces. e.g. ohwx_alex",
                    scale=1,
                )
                aliases_tb = gr.Textbox(
                    label="Trigger aliases (comma-separated)",
                    value="add me, the man, dev, i, me, guy, man, male, devin, dude, the dude, the guy, meee, myself",
                    info="Natural phrases that auto-activate this LoRA.",
                    scale=2,
                )
            with gr.Row():
                desc_tb = gr.Textbox(
                    label="What's in the photos",
                    value="a man's face and body",
                    scale=1,
                )
                caption_tb = gr.Textbox(
                    label="Caption template  ({trigger} = placeholder)",
                    value="{trigger}, a man looking at the camera, neutral expression, natural lighting",
                    scale=2,
                )

            gr.Markdown("### LoRA Training Settings")
            with gr.Row():
                rank_dd = gr.Dropdown(
                    choices=[8, 16, 32, 64, 128], value=32,
                    label="LoRA Rank",
                    info="32 is a good default. Higher = bigger file.",
                    scale=1,
                )
                steps_sl = gr.Slider(
                    minimum=200, maximum=3000, value=1000, step=100,
                    label="Train Steps",
                    info="800–1500 for 20–60 photos.",
                    scale=2,
                )
                lr_dd = gr.Dropdown(
                    choices=["5e-5", "1e-4", "2e-4", "5e-4"], value="1e-4",
                    label="Learning Rate",
                    scale=1,
                )
                weight_sl = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                    label="LoRA Weight",
                    info="Written to loras.json. 0.85 default.",
                    scale=1,
                )

            gr.Markdown("### Output")
            with gr.Row():
                output_tb = gr.Textbox(
                    label="Save LoRAs to (absolute path)",
                    value=default_output,
                    scale=2,
                )
                json_tb = gr.Textbox(
                    label="loras.json path (optional — auto-add)",
                    value="/root/newgen/loras/loras.json",
                    placeholder="/root/newgen/loras/loras.json",
                    scale=2,
                )
            with gr.Row():
                train_btn  = gr.Button("▶  Start Training", variant="primary", scale=2)
                cancel_btn = gr.Button("■  Cancel",         variant="stop",    scale=1)
            train_status = gr.Textbox(label="Status", interactive=False)

        # ── Training Log tab ──────────────────────────────────────────────────
        with gr.Tab("📊  Training Log"):
            gr.Markdown("Auto-refreshes every 3 s. Hit **Refresh now** to force an update.")
            with gr.Row():
                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="Progress (%)", interactive=False, scale=3,
                )
                progress_status = gr.Textbox(
                    label="Phase", value="Idle", interactive=False, scale=1,
                )
            log_box = gr.Textbox(
                label="Log output",
                lines=30,
                max_lines=80,
                interactive=False,
            )
            refresh_btn = gr.Button("🔄  Refresh now")

        # ── Output tab ────────────────────────────────────────────────────────
        with gr.Tab("📦  Output / loras.json"):
            gr.Markdown(
                "Copy the JSON block below into your `loras.json`, "
                "or set the path in Setup to auto-add when training finishes."
            )
            json_out = gr.Textbox(
                label="loras.json entry (copy this)",
                lines=30,
                max_lines=80,
                interactive=False,
            )

        # ── Help tab ──────────────────────────────────────────────────────────
        with gr.Tab("📖  Help"):
            gr.Markdown(f"""
## Quick Start
1. Set **Photo folder** — absolute path to your images (e.g. `/root/devin`)
2. Optionally set **Video folder** — 480p clips (MP4/MOV/etc.), alone or combined with photos
3. Set a unique **Trigger word** (`ohwx_alex`, `sks_john`, …)
4. Set **Trigger aliases** — comma-separated phrases users type in prompts
5. Optionally set **loras.json path** to auto-add the entry on completion
6. Click **Start Training** — then switch to the **Training Log** tab
7. When done, copy the JSON from the **Output** tab into `loras.json`

---
## Training data tips
- **Photos**: 15–80 images; 30–50 is ideal. Mix lighting, angles, backgrounds.
- **Videos**: 480p recommended (832×480). 5–30 short clips (2–10s each) works well.
- You can mix both — the trainer creates separate dataset blocks for each.
- JPG / PNG / WEBP / BMP accepted for images; MP4 / MOV / AVI / MKV / WEBM for videos.

---
## Troubleshooting

| Error | Fix |
|-------|-----|
| CUDA out of memory | Reduce rank to 16 |
| wan_transformer_index not recognised | `cd ~/.local/share/LoRAMaker/musubi-tuner && git pull` |
| Training produces blanks | More varied photos, aim for 40+ |
| Face not appearing | Increase weight to 0.9 in loras.json |
| LoRA not in app | Check both _high and _low are in `loras/`; check loras.json filenames |

Model: `{WAN_MODEL_REPO}`  
Version: {APP_VERSION}
            """)

        # ── Hidden state ──────────────────────────────────────────────────────
        log_state  = gr.State("")
        json_state = gr.State("")

        # ── Event wiring ──────────────────────────────────────────────────────

        preset_dd.change(
            fn=preset_defaults,
            inputs=[preset_dd],
            outputs=[trigger_tb, aliases_tb, desc_tb, caption_tb],
        )
        folder_tb.change(
            fn=validate_subject, inputs=[folder_tb, trigger_tb], outputs=[folder_status]
        )
        trigger_tb.change(
            fn=validate_subject, inputs=[folder_tb, trigger_tb], outputs=[folder_status]
        )
        video_folder_tb.change(
            fn=validate_video_folder, inputs=[video_folder_tb], outputs=[video_folder_status]
        )
        train_btn.click(
            fn=start_training,
            inputs=[folder_tb, video_folder_tb, trigger_tb, aliases_tb, desc_tb, caption_tb,
                    rank_dd, steps_sl, lr_dd, weight_sl, output_tb, json_tb],
            outputs=[train_status, log_box, progress_bar, progress_status],
        )
        cancel_btn.click(fn=cancel_training, outputs=[train_status])

        # Manual refresh — updates outputs then syncs hidden state
        refresh_btn.click(
            fn=poll_log,
            inputs=[log_state, progress_bar, json_state],
            outputs=[log_box, progress_bar, progress_status, json_out],
        )
        log_box.change(fn=lambda v: v, inputs=[log_box], outputs=[log_state])
        json_out.change(fn=lambda v: v, inputs=[json_out], outputs=[json_state])

        # Auto-poll — gr.Timer is the Gradio 6+ replacement for every= on demo.load
        timer = gr.Timer(value=3, active=True)
        timer.tick(
            fn=poll_log,
            inputs=[log_state, progress_bar, json_state],
            outputs=[log_box, progress_bar, progress_status, json_out],
        )

    return demo


# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Maker — Gradio web UI")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       default=7862, type=int,
                        help="Port to bind (default: 7862)")
    parser.add_argument("--share",      action="store_true")
    parser.add_argument("--no-browser", action="store_true",
                        help="Headless — don't try to open a browser tab")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
    )
