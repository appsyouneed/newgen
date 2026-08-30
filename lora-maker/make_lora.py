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
    subprocess.run(
        [str(venv_py), "-m", "pip", "install", "--quiet", "--upgrade"] + packages,
        check=True, capture_output=True,
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
                dataset_dir = self._phase_caption(subj)
                if self.cancel.is_set():
                    break
                out_high, out_low = self._phase_train(subj, dataset_dir, idx, n)
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
        pip_install(["pillow", "requests", "tqdm", "toml", "huggingface_hub"], py, self.log_q)
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
                pip_install(["pillow", "requests", "tqdm", "toml", "huggingface_hub"],
                            venv_python(), self.log_q)
                self._log("✓ Venv now inherits system PyTorch + CUDA 12.8")
            else:
                self._log("Installing PyTorch cu128 wheels (first-time only)…")
                pip_install(
                    ["torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu128"],
                    py, self.log_q,
                )
                self._log("✓ PyTorch + CUDA 12.8 installed")

        if not (TUNER_DIR / "requirements.txt").exists():
            self._log("Cloning musubi-tuner…")
            subprocess.run(
                ["git", "clone", "--depth=1",
                 "https://github.com/kohya-ss/musubi-tuner.git", str(TUNER_DIR)],
                check=True, capture_output=True,
            )
            self._log("✓ musubi-tuner cloned")
        else:
            self._log("✓ musubi-tuner already present")

        req_file = TUNER_DIR / "requirements.txt"
        if req_file.exists():
            self._log("Installing musubi-tuner requirements (torch lines filtered)…")
            filtered = TOOLS_DIR / "musubi_reqs_filtered.txt"
            skip = ("torch", "torchvision", "torchaudio")
            lines = req_file.read_text(encoding="utf-8").splitlines()
            kept  = [l for l in lines if not any(l.strip().lower().startswith(p) for p in skip)]
            filtered.write_text("\n".join(kept), encoding="utf-8")
            subprocess.run(
                [str(venv_python()), "-m", "pip", "install", "--quiet", "-r", str(filtered)],
                check=True, capture_output=True,
            )
            self._log("✓ musubi-tuner requirements installed")

        self._progress(10, "Environment done")

    # ── Phase 2: caption ──────────────────────────────────────────────────────

    def _phase_caption(self, subj: dict) -> Path:
        self._log(f"── Phase 2: Captioning images for '{subj['trigger']}' ──")
        self._progress(15, "Captioning")
        trigger  = subj["trigger"]
        photos   = Path(subj["folder"])
        template = subj.get("caption_template") or "{trigger}"
        if "{trigger}" not in template:
            template = "{trigger}, " + template
        dataset_dir = TOOLS_DIR / "dataset" / trigger
        dataset_dir.mkdir(parents=True, exist_ok=True)
        exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [f for f in photos.iterdir() if f.suffix.lower() in exts]
        self._log(f"Captioning {len(images)} images…")
        variants = [
            template.format(trigger=trigger),
            template.format(trigger=trigger) + ", natural lighting",
            template.format(trigger=trigger) + ", sharp focus, realistic photo",
            template.format(trigger=trigger) + ", three-quarter view",
            template.format(trigger=trigger) + ", candid photo, ambient light",
            template.format(trigger=trigger) + ", close-up, soft lighting",
            template.format(trigger=trigger) + ", warm light, portrait",
            template.format(trigger=trigger) + ", side view, natural light",
        ]
        for i, img_path in enumerate(images):
            dest = dataset_dir / img_path.name
            shutil.copy2(img_path, dest)
            dest.with_suffix(".txt").write_text(variants[i % len(variants)], encoding="utf-8")
            if (i + 1) % 5 == 0 or (i + 1) == len(images):
                self._log(f"  Captioned {i+1}/{len(images)}")
        self._log(f"✓ {len(images)} images captioned")
        self._progress(25, "Captioning done")
        return dataset_dir

    # ── Phase 3: train ────────────────────────────────────────────────────────

    def _phase_train(self, subj: dict, dataset_dir: Path, subj_idx: int, total_subjects: int):
        self._log(f"── Phase 3: Training LoRA for '{subj['trigger']}' ──")
        cfg     = self.cfg
        trigger = subj["trigger"]
        rank    = cfg["rank"]
        steps   = cfg["steps"]
        lr      = cfg["lr"]
        py      = venv_python()
        work_dir = TOOLS_DIR / "workdir" / trigger
        work_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = work_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        repeats = max(1, 200 // max(1, count_images(dataset_dir)))
        dataset_config = {
            "general": {"caption_extension": ".txt", "shuffle_caption": False, "keep_tokens": 1},
            "datasets": [{
                "resolution": [512, 512],
                "enable_bucket": True,
                "bucket_no_upscale": True,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1024,
                "subsets": [{"image_dir": str(dataset_dir), "num_repeats": repeats}],
            }],
        }
        dataset_toml = work_dir / "dataset.toml"
        try:
            import toml
            dataset_toml.write_text(toml.dumps(dataset_config), encoding="utf-8")
        except ImportError:
            lines = [
                "[general]",
                'caption_extension = ".txt"',
                "shuffle_caption = false",
                "keep_tokens = 1",
                "",
                "[[datasets]]",
                "resolution = [512, 512]",
                "enable_bucket = true",
                "bucket_no_upscale = true",
                "min_bucket_reso = 256",
                "max_bucket_reso = 1024",
                "",
                "  [[datasets.subsets]]",
                f'  image_dir = "{str(dataset_dir).replace(chr(92), "/")}"',
                f"  num_repeats = {repeats}",
            ]
            dataset_toml.write_text("\n".join(lines), encoding="utf-8")

        base_span = 65 / max(1, total_subjects)
        subj_base = 25 + subj_idx * base_span

        self._log(f"Training HIGH-noise LoRA ({steps} steps, rank {rank}, lr {lr})…")
        out_high = work_dir / f"{trigger}_high.safetensors"
        self._run_musubi(py, dataset_toml, out_high, ckpt_dir, rank, steps, lr,
                         "high", progress_base=subj_base, progress_span=base_span * 0.55)
        if self.cancel.is_set():
            return out_high, out_high

        self._log(f"Training LOW-noise LoRA ({steps} steps, rank {rank}, lr {lr})…")
        out_low = work_dir / f"{trigger}_low.safetensors"
        self._run_musubi(py, dataset_toml, out_low, ckpt_dir, rank, steps, lr,
                         "low", progress_base=subj_base + base_span * 0.55,
                         progress_span=base_span * 0.45)
        return out_high, out_low

    def _run_musubi(self, py, dataset_toml, output_path, ckpt_dir,
                    rank, steps, lr, noise_mode, cfg=None,
                    progress_base=25, progress_span=35):
        train_script = TUNER_DIR / "train_wan_i2v.py"
        if not train_script.exists():
            for name in ("train_wan.py", "train.py"):
                alt = TUNER_DIR / name
                if alt.exists():
                    train_script = alt
                    break

        model_dir = CACHE_DIR / "wan_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        if not (model_dir / "model_index.json").exists():
            self._log("Downloading WAMU v2 base model (~57 GB, first time only)…")
            self._log("  If app.py already cached it, HF Hub will skip the download.")
            dl_script = (
                f'from huggingface_hub import snapshot_download\n'
                f'snapshot_download(repo_id="{WAN_MODEL_REPO}",'
                f'local_dir=r"{model_dir}")\n'
                f'print("DOWNLOAD_DONE")\n'
            )
            result = subprocess.run([str(py), "-c", dl_script], capture_output=True, text=True)
            if "DOWNLOAD_DONE" not in result.stdout and result.returncode != 0:
                raise RuntimeError(f"Model download failed:\n{result.stderr[-800:]}")
            self._log("✓ Base model ready")

        # WAMU v2 dual-transformer routing:
        #   high-noise → transformer   (index 0) → _high.safetensors → pipe.transformer
        #   low-noise  → transformer_2 (index 1) → _low.safetensors  → pipe.transformer_2
        transformer_index = 0 if noise_mode == "high" else 1

        cmd = [
            str(py), str(train_script),
            "--pretrained_model_name_or_path", str(model_dir),
            "--dataset_config",      str(dataset_toml),
            "--output_dir",          str(ckpt_dir),
            "--output_name",         output_path.stem,
            "--network_module",      "networks.lora",
            "--network_dim",         str(rank),
            "--network_alpha",       str(rank // 2),
            "--optimizer_type",      "AdamW8bit",
            "--learning_rate",       lr,
            "--lr_scheduler",        "cosine_with_restarts",
            "--max_train_steps",     str(steps),
            "--save_every_n_steps",  str(max(100, steps // 5)),
            "--save_last_n_steps",   "1",
            "--mixed_precision",     "bf16",
            "--gradient_checkpointing",
            "--noise_offset",        "0.0",
            "--train_noise_level",   noise_mode,
            "--wan_transformer_index", str(transformer_index),
            "--seed",                "42",
        ]

        self._log(f"  {train_script.name} --train_noise_level {noise_mode} "
                  f"--wan_transformer_index {transformer_index}")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(TUNER_DIR),
        )
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            if "step" in line.lower() and "/" in line:
                self._log(f"  {line}")
                try:
                    for part in line.split():
                        if "/" in part:
                            cur, tot = part.split("/")
                            pct = progress_base + int(int(cur) / int(tot) * progress_span)
                            self._progress(pct, f"{noise_mode}-noise step {part}")
                            break
                except Exception:
                    pass
            elif any(kw in line.lower()
                     for kw in ("loss", "epoch", "saved", "error", "warn", "cuda")):
                self._log(f"  {line}")
            if self.cancel.is_set():
                proc.terminate()
                return
        proc.wait()

        saved = list(ckpt_dir.glob(f"{output_path.stem}*.safetensors"))
        if saved:
            best = max(saved, key=lambda p: p.stat().st_mtime)
            shutil.copy2(best, output_path)
            self._log(f"✓ {noise_mode}-noise LoRA saved: {output_path.name}")
        elif output_path.exists():
            self._log(f"✓ {noise_mode}-noise LoRA: {output_path.name}")
        else:
            raise RuntimeError(
                f"Training finished but no output file for {noise_mode}-noise. "
                "Check logs above."
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
        return "⚠  No photo folder entered."
    p = Path(folder)
    if not p.exists():
        return f"⚠  Folder not found: {folder}"
    n = count_images(p)
    if n < 5:
        return f"⚠  Found only {n} image(s) — need at least 5."
    t = sanitize_trigger(trigger)
    if len(t) < 3:
        return "⚠  Trigger word must be at least 3 characters."
    return f"✓  {n} images found. Trigger: '{t}'"

def start_training(folder, trigger, aliases, description, caption_template,
                   rank, steps, lr, weight, output_dir, json_path):
    global _session
    if _session and not _session.done.is_set():
        return "⚠  Training already running — cancel it first.", "", 0, "Already running"
    if not folder:
        return "⚠  Photo folder is required.", "", 0, "Validation error"
    p = Path(folder)
    if not p.exists():
        return f"⚠  Folder not found: {folder}", "", 0, "Validation error"
    n = count_images(p)
    if n < 5:
        return f"⚠  Need at least 5 images; found {n}.", "", 0, "Validation error"
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
            "folder":           folder,
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
                    placeholder="/root/photos/me",
                    info="15–80 photos. JPG/PNG/WEBP/BMP.",
                    scale=3,
                )
                folder_status = gr.Textbox(label="Folder status", interactive=False, scale=1)
            with gr.Row():
                trigger_tb = gr.Textbox(
                    label="Trigger word", value="ohwx_man",
                    info="Rare unique token, no spaces. e.g. ohwx_alex",
                    scale=1,
                )
                aliases_tb = gr.Textbox(
                    label="Trigger aliases (comma-separated)",
                    value="add me, the man, dev, i, me",
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
1. Set **Photo folder** — absolute path on the server (e.g. `/root/photos/me`)
2. Set a unique **Trigger word** (`ohwx_alex`, `sks_john`, …)
3. Set **Trigger aliases** — comma-separated phrases users type in prompts
4. Optionally set **loras.json path** to auto-add the entry on completion
5. Click **Start Training** — then switch to the **Training Log** tab
6. When done, copy the JSON from the **Output** tab into `loras.json`

---
## Dual-transformer model (WAMU v2)

| File | Expert | app.py target |
|------|--------|--------------|
| `<trigger>_high.safetensors` | high-noise | `pipe.transformer` |
| `<trigger>_low.safetensors` | low-noise | `pipe.transformer_2` |

Both files required. `load_loras_to_pipeline()` routes them automatically.

---
## Photo tips
- 15–80 photos; 30–50 is ideal
- Mix lighting, angles, backgrounds, expressions
- JPG / PNG / WEBP / BMP accepted

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
        train_btn.click(
            fn=start_training,
            inputs=[folder_tb, trigger_tb, aliases_tb, desc_tb, caption_tb,
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
