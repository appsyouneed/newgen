LoRA Maker — WAMU v2 / Wan 2.2 I2V Lightning Subject Trainer
=============================================================

Creates a personal identity LoRA from your photos and loads it into
app.py's existing LoRA system via loras.json.


WHICH MODEL THIS TRAINS AGAINST
--------------------------------
WAMU v2 — TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING

This is the exact same model app.py uses (WAN_MODEL_REPO in app.py).
LoRAs trained here are a perfect match for the pipeline.

WAMU v2 is a dual-transformer WanImageToVideoPipeline:
  pipe.transformer   = high-noise expert
  pipe.transformer_2 = low-noise expert

Each trained LoRA produces two files that map to these:
  <trigger>_high.safetensors  → pipe.transformer
  <trigger>_low.safetensors   → pipe.transformer_2

app.py's load_loras_to_pipeline() loads them onto the correct expert
automatically. Nothing extra needed — drop them in loras/ and configure
loras.json.


QUICK START
-----------
1. python3 make_lora.py
2. Setup tab → Browse for your photo folder
3. Set a unique trigger word  (e.g. "ohwx_alex")
4. (Optional) Enable "Auto-caption with Florence-2"
5. Click "Start Training"
6. Copy the two output .safetensors files into your newgen/loras/ folder
7. Add the JSON block to your loras.json  (auto-done if you set the path)
8. Restart newgen — check the LoRA box and use your trigger word in prompts


REQUIREMENTS
------------
• Ubuntu 22.04
• Python 3.10+   (sudo apt install python3 python3-venv python3-tk)
• Git             (sudo apt install git)
• CUDA 12.8 already installed (system install on VPS — NOT reinstalled)
• ~60 GB free disk space for WAMU v2 weights + working files
  (if WAMU v2 is already cached at ~/.cache/huggingface, nothing new downloads)

PyTorch is NEVER overwritten. The trainer detects your existing CUDA 12.8
install and inherits it. See CUDA / PYTORCH HANDLING below.

Hardware on this VPS:
  RTX PRO 6000 Blackwell, 95 GB VRAM, CUDA 12.8, bf16 native
  Expected training time: ~25–45 min per subject (1500 steps, rank 32)

Step count guide:
  20–30 photos  → 1000–1200 steps
  30–50 photos  → 1200–1600 steps  (default: 1500)
  50–80 photos  → 1500–2000 steps


PHOTO TIPS — CRITICAL FOR GOOD IDENTITY RESULTS
-------------------------------------------------
Quantity:
  • 20 minimum (below this: unpredictable)
  • 40–70 is the sweet spot
  • 100+ has diminishing returns without more steps

Diversity (most important factor):
  • ANGLES: front, 3/4 view, side profile, slight up/down
  • DISTANCE: tight face crop, head+shoulders, waist up, full body
  • LIGHTING: daylight, indoor, overcast, harsh sun, backlit, evening
  • EXPRESSIONS: neutral, smile, laughing, serious, looking away
  • BACKGROUNDS: plain wall, outdoors, busy scene, dark, bright
  • CLOTHING: multiple different outfits (avoids learning clothes not face)

Quality rules:
  • DO: Natural unfiltered photos. Real photos beat AI-generated.
  • DO: Include some photos where you're not the only subject
  • DO: Include full-body shots even if you mainly want face
  • DO NOT: Beauty filters, heavy retouching, face-morphing apps
  • DO NOT: Photos with face partially hidden (hats, hands, hair across face)
  • DO NOT: Group photos where your face is tiny — crop yourself out first
  • DO NOT: Screenshots from video at low resolution (use 512×512 minimum)

Cropping:
  • Portrait face: head fills ~60-70% of frame height, don't cut forehead
  • Full body: feet to a few inches above head, background on all sides
  • Close-up: eyes to chin with forehead visible
  • Keep natural aspect ratios — the trainer handles mixed sizes

Formats: .jpg .jpeg .png .webp .bmp all accepted


TRIGGER WORD TIPS
-----------------
• Use something rare — NOT "me", "man", "the man", "guy"
• Good examples:  ohwx_alex   sks_john   jdoe_person   myname123
• This word goes in your prompts wherever you'd say "the man" or "me"
• Example prompt: "ohwx_alex walking along a beach at sunset"


TRIGGER ALIASES
---------------
Aliases are natural-language phrases that automatically activate your
LoRA — you don't have to type the trigger word in every prompt.

  Trigger word:  ohwx_alex
  Aliases:       add me, the man, dev, i, me

Separate aliases with commas. Edit the defaults in the Setup tab.
These are written into loras.json and picked up by app.py's alias
matching system at generation time.


LORAS.JSON ENTRY
----------------
After training, the Output tab shows the exact JSON block to paste into
your newgen/loras/loras.json. Or set the loras.json path in Setup and
it will be added automatically.

Example entry (matches app.py's discover_loras() format exactly):
  "ohwx_alex": {
    "display_name":    "Me (ohwx_alex)",
    "description":     "My face and body LoRA",
    "trigger_prompt":  "ohwx_alex",
    "trigger_aliases": ["add me", "the man", "dev", "i", "me"],
    "prompt_mode":     "append",
    "high_filename":   "ohwx_alex_high.safetensors",
    "low_filename":    "ohwx_alex_low.safetensors",
    "high_weight":     0.85,
    "low_weight":      0.85,
    "auto_enabled":    false
  }


MIXING WITH OTHER LORAS
-----------------------
Your identity LoRA stacks with other LoRAs via app.py's multi-LoRA system.
If it overpowers other LoRAs, reduce weight to 0.6–0.7 in loras.json.
If your face isn't appearing strongly enough, increase to 0.9.


DUAL-TRANSFORMER TRAINING
--------------------------
WAMU v2 has two transformer experts. This trainer runs musubi-tuner TWICE
per subject, once per expert:

  Pass 1: --train_noise_level high --wan_transformer_index 0
           → trains against pipe.transformer (high-noise expert)
           → output: <trigger>_high.safetensors

  Pass 2: --train_noise_level low  --wan_transformer_index 1
           → trains against pipe.transformer_2 (low-noise expert)
           → output: <trigger>_low.safetensors

Both files are required for full-quality results. Training runs
sequentially, not concurrently (one set of transformer weights at a time).


PRECISION
---------
Training uses --mixed_precision bf16 throughout. This matches the WAMU v2
distilled BF16 target weights exactly.

Previous versions used fp16 — this caused a precision mismatch between the
LoRA deltas (computed relative to fp16 distributions) and the BF16 pipeline
at inference time. bf16 is the correct setting.

CUDA / PYTORCH HANDLING
-----------------------
The env phase checks three things before touching PyTorch:

  1. Does the venv already have working CUDA torch? → done, skip.
  2. Does system Python have working CUDA torch? → recreate venv with
     --system-site-packages so the existing install is inherited.
  3. Only if both fail → installs cu128 wheels from pytorch.org.

musubi-tuner's requirements.txt has all torch/torchvision/torchaudio
lines filtered out before install for the same reason.

Your CUDA 12.8 environment is NEVER overwritten.


MODEL DOWNLOAD (FIRST RUN)
--------------------------
The first run downloads WAMU v2 (~57 GB) if it isn't already cached.
Hugging Face Hub caches models at ~/.cache/huggingface — if app.py has
already downloaded WAMU v2, the trainer will find it there and skip
the download entirely (snapshot_download reuses the cache).

After the first run, retraining is fast (only the LoRA weights are
retrained — the base model is never touched).


WHAT THE APP DOES AUTOMATICALLY
--------------------------------
Once you click Train:
  • Detects and reuses existing CUDA 12.8 / PyTorch (never overwrites)
  • Installs musubi-tuner and support packages (first run only)
  • (Optional) runs Florence-2 auto-captioning
  • Trains high-noise LoRA (transformer_index=0)
  • Trains low-noise LoRA (transformer_index=1)
  • Copies finished files to your output folder
  • Writes/updates loras.json with trigger + aliases

WHAT YOU DO IN THE GUI:
  • Pick the preset type (Person / Body part / Object / Custom)
  • Browse to your photo folder
  • Set the trigger word
  • Set aliases
  • Describe what's in the photos (for template captions)
  • Optionally enable Florence-2 auto-captioning
  • Click ＋ Add Another Subject for multiple subjects


FILES
-----
  make_lora.py           — the GUI app (run with: python3 make_lora.py)
  make_lora.sh           — convenience launcher with dependency checks
  README_LoRAMaker.txt   — this file

  ~/.local/share/LoRAMaker/            — tools and model cache
  ~/.local/share/LoRAMaker/venv/       — Python virtual environment
  (output folder)                      — finished LoRA .safetensors files


TROUBLESHOOTING
---------------
"No module named tkinter"    → sudo apt install python3-tk
"Git not found"              → sudo apt install git
"CUDA out of memory"         → Shouldn't happen on 95 GB; reduce rank to 16
                               if you hit it anyway
"No module named torch"      → Let first-run setup finish; then check:
                               python3 -c "import torch; print(torch.cuda.is_available())"
Training produces blanks     → Need more varied photos; try 40+ images with
                               better diversity (angles, lighting, backgrounds)
Face barely appears          → Increase LoRA weight to 0.9–0.95 in loras.json
Face appears then morphs     → Normal for video — the LoRA anchors identity at
                               generation start; this is a prompt issue, not training
Face overpowers scene        → Reduce weight to 0.7–0.75
Body wrong but face ok       → Add more full-body shots to training set
LoRA not showing in app.py   → Check loras.json entry matches discover_loras()
                               format; both _high and _low files must be in loras/
