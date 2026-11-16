#!/usr/bin/env python3
"""
Standalone script to pre-merge NSFW weights into Qwen model and save it.
Run this ONCE, then the app will load the merged model directly on subsequent runs.
Avoids all the post-load hanging that plagues the main app startup.
"""
import os
import sys
import gc
import torch
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Find the snapshot directory (HF cache structure)
QWEN_CACHE_BASE = os.path.join(SCRIPT_DIR, "models", "Qwen-Image-Edit-2511", "models--Qwen--Qwen-Image-Edit-2511")
# Get the first (and typically only) snapshot
snapshots_dir = os.path.join(QWEN_CACHE_BASE, "snapshots")
if os.path.exists(snapshots_dir):
    snapshots = os.listdir(snapshots_dir)
    if snapshots:
        BASE_MODEL_LOAD_PATH = os.path.join(snapshots_dir, snapshots[0])
    else:
        BASE_MODEL_LOAD_PATH = QWEN_CACHE_BASE
else:
    BASE_MODEL_LOAD_PATH = QWEN_CACHE_BASE
BASE_MODEL_SAVE_PATH = "/media/dev/504E0BF74E0BD49E/Qwen-Image-Edit-2511"  # Save TO here
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(SCRIPT_DIR, "models", "rapid-aio", "v19", "Qwen-Rapid-AIO-NSFW-v19.safetensors")
MERGED_MARKER = os.path.join(BASE_MODEL_SAVE_PATH, ".merged_nsfw")

print("[merge] Loading Qwen pipeline...")
try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Copy of _merge_nsfw_into_qwen from app.py (minimal version without app dependencies)
def _merge_nsfw_into_qwen(pipe, state_dict):
    """Merge NSFW checkpoint into the pipeline."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        print("[merge] No transformer found, skipping NSFW merge")
        return
    
    print("[merge] Merging transformer weights...")
    transformer.load_state_dict(state_dict.get("transformer", {}), strict=False)
    
    vae = getattr(pipe, "vae", None)
    if vae is not None and "vae" in state_dict:
        print("[merge] Merging VAE weights...")
        vae.load_state_dict(state_dict.get("vae", {}), strict=False)
    
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None and "text_encoder" in state_dict:
        print("[merge] Merging text encoder weights...")
        text_encoder.load_state_dict(state_dict.get("text_encoder", {}), strict=False)

print("[merge] Loading base model from existing directory...")
os.makedirs(BASE_MODEL_SAVE_PATH, exist_ok=True)

model_index_path = os.path.join(BASE_MODEL_LOAD_PATH, "model_index.json")
print(f"[merge] Loading from: {BASE_MODEL_LOAD_PATH}")
print(f"[merge] Model index exists: {os.path.exists(model_index_path)}")

pipe = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL_LOAD_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    device_map="cpu"
)

print("[merge] Pipeline loaded. Now merging NSFW weights...")

# Use existing NSFW weights
if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
    print("[merge] ERROR: NSFW weights not found at", NSFW_WEIGHTS_LOCAL_PATH)
    sys.exit(1)

print(f"[merge] Using existing NSFW weights: {NSFW_WEIGHTS_LOCAL_PATH}")

print("[merge] Loading NSFW weights from disk...")
nsfw_state = load_file(NSFW_WEIGHTS_LOCAL_PATH)

print("[merge] Merging NSFW into pipeline...")
_merge_nsfw_into_qwen(pipe, nsfw_state)

print("[merge] Cleaning up...")
gc.collect()
torch.cuda.empty_cache()

print("[merge] Enabling VAE tiling/slicing...")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

print(f"[merge] Saving merged model to {BASE_MODEL_SAVE_PATH}...")
pipe.save_pretrained(BASE_MODEL_SAVE_PATH, safe_serialization=True)

print(f"[merge] Marking model as merged...")
with open(MERGED_MARKER, "w") as f:
    f.write("merged")

print("\n" + "="*70)
print("[merge] DONE! Merged model saved to external drive.")
print("[merge] Next startup will load from external drive.")
print("="*70)
