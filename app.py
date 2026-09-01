import os
import shutil
import subprocess
import sys
import random
import math
import tempfile
import warnings
import logging
import time
import gc
import uuid
import threading
import json
import base64
import hashlib
import contextlib
import queue as _queue
import os
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress all noisy warnings before any other imports
warnings.filterwarnings("ignore")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# STARTUP MODE
#
# -vidgen (default) - Video Generator tab is shown first and, on a single-GPU
# -picgen - Photo Editor tab is shown first and, on a single-GPU box,
#   Qwen loads to GPU at startup instead (Wan stays on CPU until first use).
# Has no effect with two GPUs, since neither model is ever swapped there.
# ---------------------------------------------------------------------------

STARTUP_MODE = "vidgen"
for _arg in sys.argv[1:]:
    _flag = _arg.lstrip("-").lower()
    if _flag == "vidgen":
        STARTUP_MODE = "vidgen"
    elif _flag == "picgen":
        STARTUP_MODE = "picgen"

# Environment setup
os.makedirs("/root/newgen/tmp", exist_ok=True)
os.environ.update({
    "TMPDIR": "/root/newgen/tmp",
    "TEMP": "/root/newgen/tmp",
    "TMP": "/root/newgen/tmp",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "ABSL_MIN_LOG_LEVEL": "3",
    "GRPC_VERBOSITY": "ERROR",
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,backend:cudaMallocAsync",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    "HF_HOME": "/root/.cache/huggingface",
    "CUDA_LAUNCH_BLOCKING": "0",
    "OMP_NUM_THREADS": "8",
})

import cv2
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Use 'highest' to preserve strict dtype consistency  'high' causes bfloat16/float32
# mismatches in Qwen's text encoder during concurrent dual-GPU inference
# Do NOT enable cudnn.benchmark - Blackwell GPUs (GB202) fail on conv3d engine search
# Do NOT enable cudnn.benchmark  Blackwell GPUs (GB202) fail on conv3d engine search
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
# Do not fix thread counts - multi-GPU concurrent inference needs flexible threading
# Do not fix thread counts  multi-GPU concurrent inference needs flexible threading

# Suppress noisy library warnings after torch is imported
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from PIL import Image
from safetensors.torch import load_file

import gradio as gr
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.utils.export_utils import export_to_video

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# OUTPUTS
#
# Results are written here with descriptive, unique filenames. Gradio serves
# downloads using the on-disk filename, so naming the file properly is what
# makes the download arrive as e.g. picgen_20260729-143355_a1b2c3d4.png rather
# than an extensionless temp name.
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(SCRIPT_DIR) / "outputs"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
for _d in (IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def unique_output_path(kind: str, extension: str, index: int = None) -> Path:
    """
    Build a collision-free output path.

    Timestamp gives chronological ordering, the uuid fragment guarantees
    uniqueness across concurrent requests, and the extension is always present.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:8]
    suffix = f"_{index:02d}" if index is not None else ""
    ext = extension if extension.startswith(".") else f".{extension}"
    folder = IMAGE_OUTPUT_DIR if kind == "picgen" else VIDEO_OUTPUT_DIR
    return folder / f"{kind}_{stamp}_{token}{suffix}{ext}"


def prepare_download_file(file_path: str, suggested_name: str = None) -> str:
    """
    Prepare a file for download with proper naming.
    
    Creates a symlink or copy with a clean, descriptive name to avoid
    Chrome security warnings and ensure proper file extensions.
    """
    if not file_path or not os.path.exists(file_path):
        return file_path
    
    # If no suggested name, use the original filename
    if not suggested_name:
        suggested_name = os.path.basename(file_path)
    
    # Ensure proper extension
    _, ext = os.path.splitext(file_path)
    if not suggested_name.endswith(ext):
        suggested_name += ext
    
    # Return the original path - Gradio will handle serving it correctly
    return file_path

# Import picgen prompt dicts and handlers
from prompts import (
    solo_prompts_dict, couple_man_unseen_prompts_dict, couple_man_seen_prompts_dict, 
    multiple_women_prompts_dict, multiple_man_unseen_prompts_dict, multiple_man_seen_prompts_dict, multistep_prompts_dict,
    vid_solo_prompts_dict, vid_couple_prompts_dict, vid_multiple_prompts_dict, vid_multistep_prompts_dict,
    vid_environment_prompts_dict, vid_custom_prompts_dict, vid_multiple_man_unseen_prompts_dict, vid_multiple_man_seen_prompts_dict,
    update_solo_prompt, update_couple_man_unseen_prompt, update_couple_man_seen_prompt, 
    update_multiple_women_prompt, update_multiple_man_unseen_prompt, update_multiple_man_seen_prompt, update_multistep_prompt,
    update_vid_prompt1, update_vid_prompt2, update_vid_prompt3, update_vid_prompt4,
    update_vid_prompt5, update_vid_prompt6, update_vid_prompt7, update_vid_prompt8,
)

# ---------------------------------------------------------------------------
# FRAME EXTRACTION (vidgen)
# ---------------------------------------------------------------------------


def extract_frame(video_path, timestamp):
    """Extract frame from video, save as JPG, and return file path."""
    print(f"  [extract_frame] Starting extraction...")
    print(f"   - video_path: {video_path}")
    print(f"   - timestamp: {timestamp}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" [extract_frame] Failed to open video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   - Video FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Duration: {duration:.2f}s")
    
    frame_number = int(timestamp * fps)
    print(f"   - Target frame number: {frame_number}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f" [extract_frame] Frame read successfully")
        print(f"   - Frame shape: {frame.shape}")
        
        # Save frame as high-quality JPG file
        output_path = unique_output_path("extracted_frame", "jpg")
        print(f"   - Saving to: {output_path}")
        
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f" [extract_frame] Frame saved successfully\n")
        return str(output_path)
    else:
        print(f" [extract_frame] Failed to read frame at position {frame_number}\n")
        return None


# ---------------------------------------------------------------------------
# RIFE
# ---------------------------------------------------------------------------

if not os.path.exists("train_log/RIFE_HDv3.py"):
    print("Downloading RIFE Model...")
    if not os.path.exists("RIFEv4.26_0921.zip"):
        subprocess.run([
            "wget", "-q",
            "https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.26_0921.zip",
            "-O", "RIFEv4.26_0921.zip"
        ], check=True)
    subprocess.run(["unzip", "-n", "RIFEv4.26_0921.zip"], check=True)

sys.path.append(os.path.join(os.getcwd(), "train_log"))
from train_log.RIFE_HDv3 import Model

# ---------------------------------------------------------------------------
# DEVICE PLACEMENT
#
# One GPU  : Qwen and Wan share cuda:0 and are swapped in and out of VRAM.
# Two GPUs : Qwen pins to cuda:0, Wan to cuda:1. Nothing is ever swapped, and
#            both tabs can be used concurrently.
# Override with NEWGEN_FORCE_SINGLE_GPU=1 to force swapping on a multi-GPU box.
# ---------------------------------------------------------------------------

# FAST SWAPPING MODE - one model at a time, optimized swaps
FORCE_DUAL_RESIDENT = False
AGGRESSIVE_OPTIMIZATION = True

_gpu_count = torch.cuda.device_count()
if _gpu_count < 1:
    raise RuntimeError("No CUDA device visible  this app requires a GPU.")

DUAL_GPU = _gpu_count >= 2 and os.environ.get("NEWGEN_FORCE_SINGLE_GPU") != "1"
PIC_DEVICE = "cuda:0"
WAN_DEVICE = "cuda:1" if DUAL_GPU else "cuda:0"

if DUAL_GPU:
    print(f"Dual GPU: Qwen -> {PIC_DEVICE}, Wan -> {WAN_DEVICE}. Both load at startup, no swapping.")

# Separate queues in dual GPU mode so both tabs run concurrently
PIC_QUEUE_ID = "pic-gpu" if DUAL_GPU else "gpu"
WAN_QUEUE_ID = "wan-gpu" if DUAL_GPU else "gpu"

# Auxiliary models (RIFE interpolation, MMAudio) live alongside Qwen.
device = torch.device(PIC_DEVICE)

rife_model = Model()
rife_model.load_model("train_log", -1)
rife_model.eval()
rife_model.device()

# RIFE runs in float32 deliberately. Its warplayer caches the sampling grid in
# float32, so `grid + flow` promotes to float32 and grid_sample then rejects a
# half-precision input ("expected scalar type Half but found Float"). Casting
# flownet to .half() only works if interpolation is never actually invoked.
# Interpolation is a post-process outside the diffusion loop, so the cost of
# float32 here is minor.
rife_model.flownet = rife_model.flownet.float()


@torch.no_grad()
def interpolate_bits(frames_np, multiplier=2, scale=1.0):
    if isinstance(frames_np, list):
        T = len(frames_np)
        H, W, C = frames_np[0].shape
    else:
        T, H, W, C = frames_np.shape

    if multiplier < 2:
        if isinstance(frames_np, np.ndarray):
            return list(frames_np)
        return frames_np

    n_interp = multiplier - 1
    tmp = max(128, int(128 / scale))
    ph = ((H - 1) // tmp + 1) * tmp
    pw = ((W - 1) // tmp + 1) * tmp
    padding = (0, pw - W, 0, ph - H)

    def to_tensor(frame_np):
        t = torch.from_numpy(frame_np).to(device)
        if t.dtype != torch.float32:
            t = t.float()
        t = t.permute(2, 0, 1).unsqueeze(0)
        # float32 to match flownet  see the note at model load.
        return F.pad(t, padding)

    def from_tensor(tensor):
        t = tensor[0, :, :H, :W]
        t = t.permute(1, 2, 0)
        return t.float().cpu().numpy()

    def make_inference(I0, I1, n):
        if rife_model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(rife_model.inference(I0, I1, (i + 1) * 1. / (n + 1), scale))
            return res
        else:
            middle = rife_model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = make_inference(I0, middle, n=n // 2)
            second_half = make_inference(middle, I1, n=n // 2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    output_frames = []
    I1 = to_tensor(frames_np[0])
    total_steps = T - 1

    with tqdm(total=total_steps, desc="Interpolating", unit="frame") as pbar:
        for i in range(total_steps):
            I0 = I1
            output_frames.append(from_tensor(I0))
            I1 = to_tensor(frames_np[i + 1])
            mid_tensors = make_inference(I0, I1, n_interp)
            for mid in mid_tensors:
                output_frames.append(from_tensor(mid))
            if (i + 1) % 50 == 0:
                pbar.update(50)
        pbar.update(total_steps % 50)
        output_frames.append(from_tensor(I1))

    del I0, I1, mid_tensors
    return output_frames


# ---------------------------------------------------------------------------
# MMAUDIO
# ---------------------------------------------------------------------------

MMAUDIO_REPO = "cloud19/NSFW_MMaudio"
MMAUDIO_DIR = Path("/root/newgen/mmaudio")
MMAUDIO_DIR.mkdir(parents=True, exist_ok=True)

_MMAUDIO_FILES = [
    "weights/mmaudio_large_44k_v2.pth",
    "ext_weights/synchformer_state_dict.pth",
    "ext_weights/v1-44.pth",
    "nsfw_gold_8.5k_final.pth",
]


def _ensure_mmaudio_files():
    for repo_path in _MMAUDIO_FILES:
        local_path = MMAUDIO_DIR / repo_path
        if not local_path.exists():
            print(f"Downloading mmaudio/{repo_path}...")
            hf_hub_download(
                repo_id=MMAUDIO_REPO,
                filename=repo_path,
                local_dir=str(MMAUDIO_DIR),
                local_dir_use_symlinks=False,
            )


_ensure_mmaudio_files()

try:
    import mmaudio
    from mmaudio.eval_utils import generate, load_video, make_video
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import MMAudio, get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils
    from mmaudio.model.sequence_config import CONFIG_44K
    _MMAUDIO_AVAILABLE = True
except Exception as e:
    print(f"MMAudio import failed: {e}")
    _MMAUDIO_AVAILABLE = False

_mm_dtype = torch.bfloat16
_mm_nsfw_path = MMAUDIO_DIR / "nsfw_gold_8.5k_final.pth"
_mm_vae_path = MMAUDIO_DIR / "ext_weights/v1-44.pth"
_mm_sync_path = MMAUDIO_DIR / "ext_weights/synchformer_state_dict.pth"
_mm_net = None
_mm_feature_utils = None
_mm_seq_cfg = None


def _ensure_mmaudio_loaded():
    global _mm_net, _mm_feature_utils, _mm_seq_cfg
    if _mm_net is not None:
        return
    print("Loading MMAudio on demand...")
    seq_cfg = CONFIG_44K
    net: MMAudio = get_my_mmaudio("large_44k").to(device, _mm_dtype).eval()
    net.load_weights(torch.load(str(_mm_nsfw_path), map_location=device, weights_only=True))
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=str(_mm_vae_path),
        synchformer_ckpt=str(_mm_sync_path),
        enable_conditions=True,
        mode="44k",
        bigvgan_vocoder_ckpt=None,
        need_vae_encoder=False,
    ).to(device, _mm_dtype).eval()
    _mm_net, _mm_feature_utils, _mm_seq_cfg = net, feature_utils, seq_cfg
    print("MMAudio loaded.")


@torch.inference_mode()
def add_audio_to_video(video_path: str, audio_prompt: str, duration_sec: float) -> str:
    if not _MMAUDIO_AVAILABLE:
        return video_path
    try:
        _ensure_mmaudio_loaded()
        rng = torch.Generator(device=device)
        rng.seed()
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
        video_info = load_video(Path(video_path), duration_sec)
        clip_frames = video_info.clip_frames.unsqueeze(0)
        sync_frames = video_info.sync_frames.unsqueeze(0)
        _mm_seq_cfg.duration = video_info.duration_sec
        _mm_net.update_seq_lengths(
            _mm_seq_cfg.latent_seq_len,
            _mm_seq_cfg.clip_seq_len,
            _mm_seq_cfg.sync_seq_len,
        )
        audios = generate(
            clip_frames,
            sync_frames,
            [audio_prompt],
            negative_text=["music"],
            feature_utils=_mm_feature_utils,
            net=_mm_net,
            fm=fm,
            rng=rng,
            cfg_strength=4.5,
        )
        audio = audios.float().cpu()[0]
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        make_video(video_info, out_path, audio, sampling_rate=_mm_seq_cfg.sampling_rate)
        return out_path
    except Exception as e:
        print(f"MMAudio generation failed: {e}")
        return video_path


# ---------------------------------------------------------------------------
# WAN 2.2 I2V A14B  MERGED 4-STEP DISTILL (BF16, no LoRA)
#
# Weights: lightx2v/Wan2.2-Distill-Models
#   wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors  (28.6 GB)
#   wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors   (28.6 GB)
# These are fully merged checkpoints: the 4-step distillation is baked into
# the weights, so no speed LoRA is loaded or fused at any point.
#
# Wan 2.2 A14B is a MoE pair: the high-noise expert runs the early steps and
# the low-noise expert runs the late steps. In diffusers these map to
# `transformer` and `transformer_2` on WanImageToVideoPipeline.
# ---------------------------------------------------------------------------

MAX_SEED = np.iinfo(np.int32).max

# ---------------------------------------------------------------------------
# WAN 2.2 I2V MODEL  WAMU v2 LIGHTNING (NSFW)
#
# WAMU v2 is a Wan 2.2 I2V Lightning merge trained on explicit content.
# This is a complete diffusers pipeline (transformer + transformer_2 + vae +
# text_encoder + scheduler) with 4-step distillation already merged.
# No LoRAs. No separate expert loading.
# ---------------------------------------------------------------------------

WAN_MODEL_REPO = "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING"
print(f"Video model: WAMU v2  Wan 2.2 I2V Lightning merge (NSFW-capable)")

# ---------------------------------------------------------------------------
# LORA MANAGEMENT
# ---------------------------------------------------------------------------
LORA_DIR = Path(SCRIPT_DIR) / "loras"
LORA_DIR.mkdir(exist_ok=True)
LORA_CONFIG_FILE = LORA_DIR / "loras.json"

def load_lora_config():
    """Load LoRA configuration from JSON file."""
    if not LORA_CONFIG_FILE.exists():
        return {}
    try:
        with open(LORA_CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def save_lora_config(config):
    """Save LoRA configuration to JSON file."""
    try:
        with open(LORA_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        return False

def download_lora_file(url, filename, progress_callback=None):
    """Download a LoRA file from URL with progress display."""
    import requests
    
    output_path = LORA_DIR / filename
    if output_path.exists():
        print(f" {filename} already exists")
        return True, "File already exists"
    
    try:
        print(f"\n Downloading {filename}")
        print(f"   Source: {url}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        # Use tqdm for progress display (like model downloads)
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        chunk_size = len(chunk)
                        downloaded += chunk_size
                        pbar.update(chunk_size)
                        
                        # Also call callback if provided
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded / total_size)
        
        print(f" Downloaded: {filename}\n")
        return True, "Download complete"
    except Exception as e:
        print(f" Download failed: {e}\n")
        if output_path.exists():
            output_path.unlink()
        return False, str(e)

def check_lora_status(config):
    """Check which LoRAs are downloaded."""
    status = {}
    for lora_id, lora_info in config.items():
        high_exists = False
        low_exists = False
        
        if lora_info.get('high_filename'):
            high_path = LORA_DIR / lora_info['high_filename']
            high_exists = high_path.exists()
        
        if lora_info.get('low_filename'):
            low_path = LORA_DIR / lora_info['low_filename']
            low_exists = low_path.exists()
        
        status[lora_id] = {
            'high_exists': high_exists,
            'low_exists': low_exists,
            'high_downloadable': bool(lora_info.get('high_url')),
            'low_downloadable': bool(lora_info.get('low_url')),
        }
    
    return status

def discover_loras():
    """Auto-discover all .safetensors LoRA files and merge with config."""
    # Load config first
    config = load_lora_config()
    loras = {}
    
    # Add configured LoRAs
    for lora_id, lora_info in config.items():
        loras[lora_id] = {
            'display_name': lora_info.get('display_name', lora_id),
            'description': lora_info.get('description', ''),
            'high': str(LORA_DIR / lora_info['high_filename']) if lora_info.get('high_filename') and (LORA_DIR / lora_info['high_filename']).exists() else None,
            'low': str(LORA_DIR / lora_info['low_filename']) if lora_info.get('low_filename') and (LORA_DIR / lora_info['low_filename']).exists() else None,
            'trigger_prompt': lora_info.get('trigger_prompt'),
            'prompt_mode': lora_info.get('prompt_mode', 'append'),
            'example_prompts': lora_info.get('example_prompts', []),
            'high_weight': lora_info.get('high_weight', 1.0),
            'low_weight': lora_info.get('low_weight', 1.0),
            'recommended_steps': lora_info.get('recommended_steps'),
            'recommended_flow_shift': lora_info.get('recommended_flow_shift'),
            'notes': lora_info.get('notes', ''),
            'config': lora_info,  # Keep full config for downloads
        }
    
    # Also discover any unconfigured LoRAs in folder
    if LORA_DIR.exists():
        for lora_file in LORA_DIR.glob("*.safetensors"):
            filename = lora_file.name
            # Check if already in config
            already_configured = False
            for lora_info in config.values():
                if filename in [lora_info.get('high_filename'), lora_info.get('low_filename')]:
                    already_configured = True
                    break
            
            if not already_configured:
                # Parse high/low from filename
                filename_lower = filename.lower()
                if "_high" in filename_lower or "_high_" in filename_lower:
                    noise_type = "high"
                elif "_low" in filename_lower or "_low_" in filename_lower:
                    noise_type = "low"
                else:
                    noise_type = "unknown"
                
                # Extract base name
                base_name = filename.replace('.safetensors', '')
                for suffix in ["_high_noise", "_low_noise", "_high", "_low"]:
                    if suffix in filename_lower:
                        idx = filename_lower.index(suffix)
                        base_name = filename[:idx]
                        break
                
                if base_name not in loras:
                    loras[base_name] = {
                        'display_name': base_name,
                        'description': 'Unconfigured LoRA',
                        'high': None,
                        'low': None,
                        'trigger_prompt': None,
                        'prompt_mode': 'append',
                        'example_prompts': [],
                        'high_weight': 1.0,
                        'low_weight': 1.0,
                        'notes': 'Add to loras.json for full configuration',
                        'config': {},
                    }
                
                loras[base_name][noise_type] = str(lora_file)
    
    return loras

# Track active LoRAs (global state)
_active_loras = {}  # {base_name: {"high": path, "low": path}}

def _wan_flat_key_to_module_path(flat_key: str) -> str | None:  # unused — kept for reference
    """
    Convert a Wan-native flat LoRA key to a dotted transformer parameter path.

    Wan LoRA files encode the full module path in a flat underscore-separated
    prefix.  For example:
        lora_unet_blocks_0_self_attn_q   →   blocks.0.self_attn.q
        lora_unet_blocks_27_ffn_0        →   blocks.27.ffn.0
        lora_unet_blocks_3_cross_attn_k  →   blocks.3.cross_attn.k

    The mapping rules:
      - Strip leading "lora_unet_" (or "lora_" alone if no "unet_").
      - Replace "_blocks_N_" with ".blocks.N." (N is an integer).
      - The remainder maps component names: self_attn, cross_attn, ffn stay as-is
        but use dots; the trailing single-letter (q/k/v/o) stays as-is.
      - ffn layers: ffn_0 → ffn.0, ffn_2 → ffn.2

    Returns a dotted path string, or None if the key cannot be decoded.
    """
    import re

    # Strip the leading lora_unet_ or lora_ prefix used by the file format
    if flat_key.startswith("lora_unet_"):
        rest = flat_key[len("lora_unet_"):]
    elif flat_key.startswith("lora_"):
        rest = flat_key[len("lora_"):]
    else:
        return None

    # Must start with "blocks_<int>_"
    m = re.match(r'^blocks_(\d+)_(.+)$', rest)
    if not m:
        return None

    block_idx = m.group(1)
    layer_part = m.group(2)  # e.g. "self_attn_q", "cross_attn_k", "ffn_0"

    # Map layer_part to dotted sub-path
    # Attention layers: self_attn_q/k/v/o, cross_attn_q/k/v/o
    attn_m = re.match(r'^(self_attn|cross_attn)_([qkvo])$', layer_part)
    if attn_m:
        attn_type = attn_m.group(1)   # self_attn or cross_attn
        proj = attn_m.group(2)        # q / k / v / o
        return f"blocks.{block_idx}.{attn_type}.{proj}"

    # FFN layers: ffn_0, ffn_2, ffn_4 …
    ffn_m = re.match(r'^ffn_(\d+)$', layer_part)
    if ffn_m:
        ffn_idx = ffn_m.group(1)
        return f"blocks.{block_idx}.ffn.{ffn_idx}"

    # Unknown — return None so the caller can skip
    return None


def load_loras_to_pipeline(pipe, selected_loras):
    """
    Load selected LoRAs into the Wan 2.2 dual-transformer pipeline using PEFT native API.

    Wan 2.2 I2V has two transformer experts:
      pipe.transformer   — high-noise denoising expert (early steps)
      pipe.transformer_2 — low-noise denoising expert (late steps)

    Uses diffusers/PEFT's built-in load_lora_weights() which correctly handles
    all LoRA key formats automatically.
    """
    global _active_loras

    if pipe is None:
        return

    # Unload all existing LoRAs first
    try:
        if hasattr(pipe, 'unfuse_lora'):
            pipe.unfuse_lora()
    except Exception:
        pass
    try:
        if hasattr(pipe, 'unload_lora_weights'):
            pipe.unload_lora_weights()
    except Exception:
        pass

    _active_loras = {}

    if not selected_loras:
        print("[LoRA] No LoRAs selected — all unloaded.")
        return

    loaded_count = 0

    for base_name, lora_info in selected_loras.items():
        high_path   = lora_info.get("high")
        low_path    = lora_info.get("low")
        high_weight = float(lora_info.get("high_weight") or 1.0)
        low_weight  = float(lora_info.get("low_weight")  or 1.0)
        lora_ok     = False

        # High-noise LoRA → pipe.transformer
        if high_path and os.path.exists(high_path):
            try:
                pipe.load_lora_weights(high_path, adapter_name=f"{base_name}_high")
                pipe.set_adapters([f"{base_name}_high"], adapter_weights=[high_weight])
                print(f"[LoRA] transformer: '{base_name}_high' loaded (weight={high_weight:.3f}) <- {os.path.basename(high_path)}")
                lora_ok = True
            except Exception as e:
                print(f"[LoRA] ERROR: failed to load high-noise LoRA for '{base_name}': {e}")

        # Low-noise LoRA → pipe.transformer_2
        has_t2 = hasattr(pipe, 'transformer_2') and pipe.transformer_2 is not None
        if low_path and os.path.exists(low_path):
            if has_t2:
                try:
                    pipe.load_lora_weights(low_path, adapter_name=f"{base_name}_low")
                    pipe.set_adapters([f"{base_name}_low"], adapter_weights=[low_weight])
                    print(f"[LoRA] transformer_2: '{base_name}_low' loaded (weight={low_weight:.3f}) <- {os.path.basename(low_path)}")
                    lora_ok = True
                except Exception as e:
                    print(f"[LoRA] ERROR: failed to load low-noise LoRA for '{base_name}': {e}")
            else:
                print(f"[LoRA] WARNING: '{base_name}' has low-noise file but pipeline has no transformer_2")

        if lora_ok:
            _active_loras[base_name] = lora_info
            loaded_count += 1
        else:
            print(f"[LoRA] WARNING: '{base_name}' — no files loaded (high={high_path}, low={low_path})")

    if loaded_count:
        print(f"[LoRA] {loaded_count} LoRA(s) loaded and activated successfully.")
    else:
        print("[LoRA] WARNING: no LoRAs were successfully loaded.")


def apply_lora_prompt_modifications(base_prompt, selected_loras_info):
    """
    Apply trigger prompts from active LoRAs to the user's base prompt.

    Modes:
    - append:  add trigger to the end if not already present (deduplicates)
    - prepend: add trigger to the front if not already present (deduplicates)
    - replace: the trigger IS the prompt — replaces the base prompt entirely.
               Used for LoRAs (like deepthroat) that require a very specific
               prompt structure; the user should use the example prompt dropdown
               which loads the full prompt, but if they haven't done so we still
               inject the trigger so the LoRA isn't silently doing nothing.
    """
    modified_prompt = base_prompt.strip()

    for lora_info in selected_loras_info.values():
        trigger = (lora_info.get('trigger_prompt') or "").strip()
        if not trigger:
            continue

        mode = lora_info.get('prompt_mode', 'append')

        if mode == 'replace':
            # Only replace if the user hasn't already typed anything meaningful.
            # If the prompt is empty or just the default placeholder, use the trigger.
            # Otherwise prepend the trigger so it's present without destroying the user's work.
            if not modified_prompt:
                modified_prompt = trigger
            elif trigger.lower() not in modified_prompt.lower():
                modified_prompt = f"{trigger}, {modified_prompt}"

        elif mode == 'prepend':
            if trigger.lower() not in modified_prompt.lower():
                modified_prompt = f"{trigger}, {modified_prompt}"

        else:  # append (default) — also handles undocumented modes like 'natural'
            if trigger.lower() not in modified_prompt.lower():
                modified_prompt = f"{modified_prompt}, {trigger}"

    return modified_prompt.strip().strip(',').strip()

def check_lora_compatibility(selected_loras_info):
    """
    Check compatibility between selected LoRAs and return merged settings.
    Returns: (is_compatible, compatibility_message, merged_settings)
    """
    if not selected_loras_info:
        return True, "", {}
    
    lora_names = list(selected_loras_info.keys())
    
    # If only one LoRA, always compatible
    if len(lora_names) == 1:
        lora_info = list(selected_loras_info.values())[0]
        return True, f" Using {lora_info['display_name']}", {
            'recommended_steps': lora_info.get('recommended_steps'),
            'recommended_flow_shift': lora_info.get('recommended_flow_shift'),
            'high_weight': lora_info.get('high_weight', 1.0),
            'low_weight': lora_info.get('low_weight', 1.0),
        }
    
    # Multiple LoRAs - check compatibility
    recommended_steps = []
    recommended_flow_shifts = []
    high_weights = []
    low_weights = []
    warnings = []
    
    for lora_id, lora_info in selected_loras_info.items():
        if lora_info.get('recommended_steps'):
            recommended_steps.append(lora_info['recommended_steps'])
        if lora_info.get('recommended_flow_shift'):
            recommended_flow_shifts.append(lora_info['recommended_flow_shift'])
        high_weights.append(lora_info.get('high_weight', 1.0))
        low_weights.append(lora_info.get('low_weight', 1.0))
    
    # Check if settings are compatible
    steps_compatible = len(set(recommended_steps)) <= 1 if recommended_steps else True
    flow_compatible = len(set(recommended_flow_shifts)) <= 1 if recommended_flow_shifts else True
    
    # Generate compatibility message
    display_names = [info['display_name'] for info in selected_loras_info.values()]
    
    if not steps_compatible:
        warnings.append(f" Step conflict: {', '.join(map(str, set(recommended_steps)))} steps recommended")
    
    if not flow_compatible:
        warnings.append(f" Flow shift conflict: {', '.join(map(str, set(recommended_flow_shifts)))}")
    
    # Determine merged settings
    merged_steps = recommended_steps[0] if steps_compatible and recommended_steps else None
    merged_flow = recommended_flow_shifts[0] if flow_compatible and recommended_flow_shifts else None
    
    # Average weights when combining
    merged_high_weight = sum(high_weights) / len(high_weights)
    merged_low_weight = sum(low_weights) / len(low_weights)
    
    is_compatible = steps_compatible and flow_compatible
    
    if is_compatible:
        message = f" Compatible: {', '.join(display_names)}"
    else:
        message = f" Combining: {', '.join(display_names)}\n" + '\n'.join(warnings)
    
    merged_settings = {
        'recommended_steps': merged_steps,
        'recommended_flow_shift': merged_flow,
        'high_weight': merged_high_weight,
        'low_weight': merged_low_weight,
    }
    
    return is_compatible, message, merged_settings

def apply_lora_settings(selected_loras_info, user_steps, user_flow_shift, flow_shift_auto):
    """
    Apply LoRA-recommended settings when a LoRA is active.

    Rules:
    - recommended_steps: ALWAYS applied when a LoRA is active — overrides whatever
      the slider currently shows. The UI already sets the slider on checkbox change
      via update_lora_compatibility_and_steps, so this ensures the actual generation
      call uses the right value even if the slider drifted.
    - recommended_flow_shift: applied unless the user explicitly put the slider into
      manual mode (flow_shift_auto=False AND they moved it off the LoRA recommendation).
      When in auto mode the LoRA recommendation always wins.
    - high_weight / low_weight: used by load_loras_to_pipeline which reads them from
      the lora_info dict — no extra action needed here.
    """
    if not selected_loras_info:
        return user_steps, user_flow_shift, ""

    is_compatible, compat_msg, merged_settings = check_lora_compatibility(selected_loras_info)
    messages = [compat_msg] if compat_msg else []

    # ── Steps: always use the LoRA recommendation when one exists ────────────
    final_steps = user_steps
    recommended_steps = merged_settings.get('recommended_steps')
    if recommended_steps is not None:
        final_steps = recommended_steps
        messages.append(f"Steps set to {recommended_steps} (LoRA recommendation)")

    # ── Flow shift: use LoRA recommendation unless user is in manual override ─
    final_flow_shift = user_flow_shift
    recommended_flow = merged_settings.get('recommended_flow_shift')
    if recommended_flow is not None:
        if flow_shift_auto:
            # Auto mode: always use LoRA recommendation
            final_flow_shift = recommended_flow
            messages.append(f"Flow shift set to {recommended_flow} (LoRA recommendation, auto mode)")
        else:
            # Manual mode: use LoRA recommendation unless user explicitly changed it
            # "explicitly changed" means the slider value is not equal to any LoRA
            # recommendation AND not the global default — i.e. the user touched it.
            # We use the LoRA recommendation as the default baseline here.
            final_flow_shift = recommended_flow
            messages.append(f"Flow shift set to {recommended_flow} (LoRA recommendation)")

    if not is_compatible and len(selected_loras_info) > 1:
        messages.append("Note: multiple LoRAs with conflicting settings — using first recommendation.")

    return final_steps, final_flow_shift, "\n".join(messages)

# Discover LoRAs at startup
LORA_CONFIG = load_lora_config()
AVAILABLE_LORAS = discover_loras()
LORA_STATUS = check_lora_status(LORA_CONFIG)

for lora_id, info in AVAILABLE_LORAS.items():
    status = LORA_STATUS.get(lora_id, {})
    high_status = "OK" if info['high'] else ("DL" if status.get('high_downloadable') else "X")
    low_status = "OK" if info['low'] else ("DL" if status.get('low_downloadable') else "X")

# ---------------------------------------------------------------------------

# 16 fps is what Wan-AI's own diffusers example for I2V-A14B exports at.
# (The 24 fps figure in the Wan 2.2 docs refers to the TI2V-5B model.)
FIXED_FPS = 16

# WAMU v2 is distillation-merged, so guidance must stay at 1.0.
WAN_STEPS = 3  # Default fallback, actual steps come from UI slider
WAN_FLOW_SHIFT = 6.9
WAN_GUIDANCE = 1.0

MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 97       # 6s per segment (97 frames 16 fps = 6.06s)  keeps quality high
SEGMENT_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)   # ~6.1s per segment
MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = 600.0        # 10 minutes max via chaining

# Frame geometry. Area-based, matching the official sizing recipe.
# mod 16 == vae_scale_factor_spatial (8) * transformer patch_size (2).
AREA_720P = 1280 * 720
AREA_480P = 832 * 480
MULTIPLE_OF = 16

wan_pipe = None
_wan_loaded = False
_wan_scheduler_config = None

# Track current input image path(s) to exclude from clear_storage.
# NOTE: a single overwritten global is not safe once multiple images (reference
# + end frame + sequence slots) or concurrent generations are involved, so we
# also keep a thread-safe SET of every path that is currently "live" in an
# input widget. clear_storage() consults this set in addition to the legacy
# single-path variable below (kept for backwards compatibility).
_protected_image_paths = set()
_protected_paths_lock = threading.Lock()

# Paths that are *currently being used by an in-flight generation*.
# generate_video() (and sequence equivalents) add paths here at the START of
# a run and remove them when the run finishes or errors.  The widget-change
# tracker (_make_image_tracker) must not unprotect a path that is still in
# this set — otherwise replacing an input image mid-generation would
# immediately expose the file the running job is reading.
_generation_active_paths = set()
_generation_active_lock = threading.Lock()


def _generation_protect(path):
    """Pin a path as in-use by a running generation."""
    if not path:
        return
    try:
        p = str(path)
        with _generation_active_lock:
            _generation_active_paths.add(p)
        _protect_path(p)          # also in the general set
    except Exception:
        pass


def _generation_release(path):
    """Release a path from the generation-active set (generation done/error)."""
    if not path:
        return
    try:
        p = str(path)
        with _generation_active_lock:
            _generation_active_paths.discard(p)
        # Only unprotect from the general set if the widget is no longer
        # pointing at this file either.  We leave the general set alone here
        # and let the widget tracker clean it up on the next change event.
    except Exception:
        pass


def _is_generation_active(path) -> bool:
    """True if path is currently held by a running generation."""
    if not path:
        return False
    try:
        p = str(path)
        with _generation_active_lock:
            return p in _generation_active_paths
    except Exception:
        return False


# Additional set of protected *filenames* (basenames only).
# protect_current_inputs() populates this before each automatic clear so that
# even if the full path lookup misses (e.g. Gradio moved the file), anything
# whose filename matches the current first-frame or last-frame is never deleted.
_protected_image_filenames = set()
_protected_filenames_lock = threading.Lock()


def _protect_filename(path):
    """Register the basename of path as protected from clear_storage()."""
    if not path:
        return
    try:
        name = os.path.basename(str(path))
        if name:
            with _protected_filenames_lock:
                _protected_image_filenames.add(name)
    except Exception:
        pass


def _unprotect_filename(path):
    """Remove the basename of path from the protected-filenames set."""
    if not path:
        return
    try:
        name = os.path.basename(str(path))
        if name:
            with _protected_filenames_lock:
                _protected_image_filenames.discard(name)
    except Exception:
        pass


def _is_filename_protected(item_path) -> bool:
    """True if the item's basename is in the protected-filenames set."""
    try:
        name = os.path.basename(str(item_path))
        with _protected_filenames_lock:
            return name in _protected_image_filenames
    except Exception:
        return False


def _protect_path(path):
    """Mark a filesystem path as protected from clear_storage()."""
    if not path:
        return
    try:
        p = str(path)
    except Exception:
        return
    with _protected_paths_lock:
        _protected_image_paths.add(p)


def _unprotect_path(path):
    """Remove a path from the protected set (safe no-op if absent)."""
    if not path:
        return
    try:
        p = str(path)
    except Exception:
        return
    with _protected_paths_lock:
        _protected_image_paths.discard(p)


def _is_protected(item_path) -> bool:
    """True if item_path is, or contains, a currently-protected input image.

    Two layers of protection:
    1. Full-path set: _protected_image_paths (and legacy _current_input_image_path).
    2. Filename set: _protected_image_filenames — any item whose *basename*
       matches a currently-live first-frame or last-frame filename is skipped,
       even if the full path lookup misses (e.g. Gradio moved/renamed the file).
    """
    # Layer 2: filename-based protection (fast, checked first)
    if _is_filename_protected(item_path):
        return True

    # Layer 1: full-path protection
    item_path = Path(item_path)
    with _protected_paths_lock:
        protected_now = set(_protected_image_paths)
    if _current_input_image_path:
        protected_now.add(_current_input_image_path)
    for p in protected_now:
        if not p:
            continue
        protected_path = Path(p)
        if item_path == protected_path:
            return True
        if item_path.is_dir():
            try:
                protected_path.relative_to(item_path)
                return True
            except ValueError:
                pass
    return False



_current_input_image_path = None


def _set_flow_shift(pipe, flow_shift):
    """
    Rebuild the scheduler with a different flow shift.

    Flow shift controls how the noise schedule is warped, which materially
    affects motion amount on Wan. Always derived from the pristine config so
    repeated changes cannot compound. Safe to mutate because video jobs are
    serialised by the queue.
    """
    if flow_shift is None or _wan_scheduler_config is None:
        return

    cfg = dict(_wan_scheduler_config)
    key = "flow_shift" if "flow_shift" in cfg else "shift" if "shift" in cfg else None
    if key is None:
        return

    try:
        if abs(float(cfg.get(key) or 0.0) - float(flow_shift)) < 1e-6:
            current = getattr(pipe.scheduler.config, key, None)
            if current is not None and abs(float(current) - float(flow_shift)) < 1e-6:
                return
        cfg[key] = float(flow_shift)
        pipe.scheduler = type(pipe.scheduler).from_config(cfg)
    except Exception as e:
        print(f"Could not apply flow_shift={flow_shift} ({e})  keeping default.")


_wan_load_lock = threading.Lock()


def _load_wan(target_device="cpu"):
    """
    Build the Wan 2.2 I2V pipeline with both merged 4-step experts.

    Guarded by its own lock, separate from the VRAM swap lock, so the one-time
    ~57 GB download and model build cannot block the Photo Editor tab.
    """
    global wan_pipe, _wan_loaded
    if _wan_loaded and wan_pipe is not None:
        return wan_pipe

    with _wan_load_lock:
        if _wan_loaded and wan_pipe is not None:
            return wan_pipe
        return _build_wan_pipeline(target_device)


def _from_pretrained_cached(repo: str, **kwargs):
    """Try the local cache first, then fall back to downloading."""
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(
            repo, local_files_only=True, **kwargs
        )
        return pipe
    except Exception:
        return WanImageToVideoPipeline.from_pretrained(repo, **kwargs)


def _build_wan_pipeline(target_device="cpu"):
    global wan_pipe, _wan_loaded, _wan_scheduler_config

    if target_device == "cpu":
        # CPU load  standard path for single-GPU swap mode
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
        print(f" WAMU v2 loaded to CPU (ready for fast swapping)")
    else:
        # Load to CPU then move to target GPU.
        # device_map doesn't support specific device indices in this diffusers version.
        # The .to() call is a fast VRAM transfer  no computation, just memory copy.
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
        pipeline = pipeline.to(target_device)
        torch.cuda.synchronize(target_device)
        print(f" WAMU v2 loaded directly to {target_device} - Ready for video generation!")

    _wan_scheduler_config = dict(pipeline.scheduler.config)
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    wan_pipe = pipeline
    _wan_loaded = True
    return wan_pipe


# DUAL-RESIDENT mode - no swapping needed, no availability checks
# Both models are loaded to GPU at startup


default_video_prompt = (
    "Cinematic, slow sweeping camera movement, natural body motion, "
    "detailed skin texture, dramatic ambient lighting"
)
default_negative_prompt = (
    "static, frozen, blurry, low quality, distorted, deformed, "
    "extra limbs, watermark, text, logo"
)


def model_title():
    gpu_note = (
        f"Dual GPU: Qwen on {PIC_DEVICE}, Wan on {WAN_DEVICE} (no swapping)."
        if DUAL_GPU else
        "Single GPU: models swap on demand."
    )
    return (
        "##  WAMU v2  Wan 2.2 I2V Lightning (NSFW)\n"
        f"4-step distilled merge. No LoRAs. {gpu_note}"
    )


def _ensure_pil(image):
    """
    Normalize a Gradio image value to a PIL Image (or None).

    Optional `gr.Image` components can hand back `None`, `""`, or a file
    path depending on how they were last touched, instead of always giving
    a PIL image. Any falsy value (None or empty string) becomes None; a
    string path is opened; anything else is returned unchanged.
    """
    if not image:
        return None
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image


def resize_image_for_wan(image: Image.Image, resolution: str = "720p") -> Image.Image:
    """
    Fit an image to a target pixel area while preserving aspect ratio.

    This mirrors Wan-AI's own sizing recipe: pick an area budget, derive width
    and height from the image's aspect ratio, then round both to a multiple of
    16. For I2V the area matters, not fixed dimensions.
    """
    # Handle string paths / stray empty values from Gradio
    image = _ensure_pil(image)

    # Check cache first
    cached = _get_cached_resized(image, resolution)
    if cached is not None:
        return cached
    
    max_area = AREA_480P if resolution == "480p" else AREA_720P

    aspect = image.height / image.width
    height = int(round(np.sqrt(max_area * aspect))) // MULTIPLE_OF * MULTIPLE_OF
    width = int(round(np.sqrt(max_area / aspect))) // MULTIPLE_OF * MULTIPLE_OF

    # Ensure minimum dimensions and proper alignment for VAE
    height = max(MULTIPLE_OF * 8, height)  # Increased minimum for VAE compatibility
    width = max(MULTIPLE_OF * 8, width)   # Increased minimum for VAE compatibility
    
    # Ensure dimensions are compatible with Wan's VAE encoder
    # VAE encoder expects specific ratios, adjust if needed
    if height % 32 != 0:
        height = (height // 32 + 1) * 32
    if width % 32 != 0:
        width = (width // 32 + 1) * 32
    
    resized = image.resize((width, height), Image.LANCZOS)
    
    # Cache the result
    _cache_resized(image, resolution, resized)
    return resized


def resize_and_crop_to_match(target_image, reference_image):
    ref_width, ref_height = reference_image.size
    target_width, target_height = target_image.size
    scale = max(ref_width / target_width, ref_height / target_height)
    new_width, new_height = int(target_width * scale), int(target_height * scale)
    resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - ref_width) // 2, (new_height - ref_height) // 2
    return resized.crop((left, top, left + ref_width, top + ref_height))


def _remove_background_rembg(pil_image: Image.Image) -> Image.Image:
    """
    Remove background from a PIL image using rembg with birefnet-general model.
    Returns RGBA image. Falls back to original image converted to RGBA if rembg unavailable.
    """
    try:
        from rembg import remove, new_session
        session = new_session("birefnet-general")
        rgba = remove(pil_image.convert("RGBA"), session=session)
        return rgba
    except ImportError:
        print("rembg not installed — background removal skipped. Install with: pip install rembg[gpu]")
        return pil_image.convert("RGBA")
    except Exception as e:
        print(f"Background removal failed: {e} — using original image")
        return pil_image.convert("RGBA")


def merge_photos_fn(img_a, img_b) -> Image.Image | None:
    """
    Merge two photos side by side on a white background.
    Removes backgrounds using BiRefNet, centers both subjects vertically,
    outputs a 1280x720 landscape image ready for Wan 2.2.
    """
    if img_a is None or img_b is None:
        return None

    # Load both images
    pil_a = _ensure_pil(img_a)
    pil_b = _ensure_pil(img_b)

    # Remove backgrounds from both
    rgba_a = _remove_background_rembg(pil_a)
    rgba_b = _remove_background_rembg(pil_b)

    # Trim transparent padding to tight bounding box of each subject
    def trim_to_subject(rgba: Image.Image) -> Image.Image:
        bbox = rgba.getbbox()
        if bbox:
            return rgba.crop(bbox)
        return rgba

    rgba_a = trim_to_subject(rgba_a)
    rgba_b = trim_to_subject(rgba_b)

    # Target output size (Wan 2.2 native landscape)
    OUT_W, OUT_H = 1280, 720

    # Scale both subjects to the same height — use 85% of output height so
    # there is breathing room top and bottom, then center vertically
    TARGET_H = int(OUT_H * 0.85)

    def scale_to_height(rgba: Image.Image, h: int) -> Image.Image:
        aspect = rgba.width / rgba.height
        w = max(1, int(h * aspect))
        return rgba.resize((w, h), Image.LANCZOS)

    scaled_a = scale_to_height(rgba_a, TARGET_H)
    scaled_b = scale_to_height(rgba_b, TARGET_H)

    # Build white canvas
    canvas = Image.new("RGBA", (OUT_W, OUT_H), (255, 255, 255, 255))

    # Center both subjects horizontally in their respective halves,
    # center vertically on the canvas
    half_w = OUT_W // 2
    y_offset = (OUT_H - TARGET_H) // 2

    # Person A — right-aligned within left half (they face inward naturally)
    x_a = max(0, half_w - scaled_a.width - 10)
    canvas.paste(scaled_a, (x_a, y_offset), scaled_a)

    # Person B — left-aligned within right half
    x_b = half_w + 10
    canvas.paste(scaled_b, (x_b, y_offset), scaled_b)

    # Flatten to white RGB
    final = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))
    final.paste(canvas.convert("RGB"), mask=canvas.split()[3])

    return final


def get_num_frames(duration_seconds: float) -> int:
    """
    Frame count for a duration, snapped to the 4n+1 layout Wan's VAE requires
    and capped at MAX_FRAMES_MODEL (97 frames = ~6s) to stay within the quality
    window before prompt degradation occurs.
    """
    raw = int(round(float(duration_seconds) * FIXED_FPS))
    raw = int(np.clip(raw, MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))
    # snap down to nearest 4n+1
    frames = ((raw - 1) // 4) * 4 + 1
    return max(9, min(MAX_FRAMES_MODEL, frames))


# ---------------------------------------------------------------------------
# SCENE MODES
#
# Keep scene       no image editing at all. The photo goes straight to Wan, so
#                   background, bodies and framing stay pixel-identical.
# Replace scene    Qwen repaints the environment around the subjects first.
# Custom edit      Qwen applies your edit instruction verbatim, no template.
# ---------------------------------------------------------------------------

MODE_KEEP = "Keep original scene"
MODE_REPLACE = "Replace background / environment"
MODE_CUSTOM = "Custom edit instruction"
MODE_AUTORUN = "Autorun"
MODE_SEQUENCE = "Sequence"
MODE_CUSTOM_SEQ = "Custom edit sequence"
SCENE_MODES = [MODE_KEEP, MODE_REPLACE, MODE_CUSTOM, MODE_AUTORUN, MODE_SEQUENCE, MODE_CUSTOM_SEQ]
SEQUENCE_MAX_SLOTS = 10
CUSTOM_SEQ_MAX_SLOTS = 10

AUTORUN_DIR = Path(SCRIPT_DIR) / "autorun"
AUTORUN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# PUSH AUTORUN: local download destination (Windows path via SSH pipe).
# The SSH tunnel feeder will receive each finished .mp4 over stdin and write
# it to this folder on the local machine.  Set to None to disable SSH-pipe
# download and fall back to browser download only.
# ---------------------------------------------------------------------------
PUSH_LOCAL_DOWNLOAD_DIR = r"D:\Apps\newgen\downloads"


def discover_autorun_images():
    """
    Return a sorted list of image paths from the autorun folder.
    Raises gr.Error if the folder is missing or contains no supported images.
    """
    if not AUTORUN_DIR.exists():
        raise gr.Error(
            f"Autorun folder not found: {AUTORUN_DIR}  "
            "Create an 'autorun' directory next to app.py and add images."
        )
    files = [
        f for f in AUTORUN_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in AUTORUN_EXTENSIONS
    ]
    if not files:
        raise gr.Error(
            f"No supported images found in {AUTORUN_DIR}  "
            "Add .jpg, .jpeg, .png, or .webp files and try again."
        )
    return sorted(files, key=lambda p: p.name.lower())

RELOCATE_INSTRUCTION = (
    "Keep the people exactly as they are  identical faces, facial features, "
    "hair, body shape, proportions and skin tone. Do not alter their identity "
    "or their bodies. Replace the entire background and environment with: "
    "{prompt}. Relight the subjects to match the new environment."
)

# Cache for vidgen to speed up repeated generations
_vidgen_cache = {
    "resized_images": {},        # keyed by image hash + resolution
}
_vidgen_cache_lock = threading.Lock()
MAX_VIDGEN_CACHE = 10


def _hash_pil_image(img):
    """Fast hash of a PIL image."""
    if isinstance(img, str):
        # It's a file path, hash the path and file size
        hasher = hashlib.sha256()
        hasher.update(img.encode())
        try:
            hasher.update(str(os.path.getsize(img)).encode())
        except OSError:
            pass
        return hasher.hexdigest()
    hasher = hashlib.sha256()
    hasher.update(f"{img.size}".encode())
    img_array = np.array(img.resize((64, 64), Image.LANCZOS))
    hasher.update(img_array.tobytes())
    return hasher.hexdigest()


def _get_cached_resized(image, resolution):
    """Get cached resized image if available."""
    img_hash = _hash_pil_image(image)
    key = (img_hash, resolution)
    with _vidgen_cache_lock:
        return _vidgen_cache["resized_images"].get(key)


def _cache_resized(image, resolution, resized):
    """Cache resized image."""
    img_hash = _hash_pil_image(image)
    key = (img_hash, resolution)
    with _vidgen_cache_lock:
        cache = _vidgen_cache["resized_images"]
        if len(cache) >= MAX_VIDGEN_CACHE:
            cache.pop(next(iter(cache)))
        cache[key] = resized


def edit_reference_frame(
    image: Image.Image,
    mode: str,
    prompt: str,
    seed: int,
    steps: int,
    guidance: float,
) -> Image.Image:
    """
    Optional stage 1  Qwen Image Edit prepares the starting frame.

    Returns the image untouched in keep-scene mode, so nothing is repainted and
    the original background and bodies survive exactly as shot.
    """
    if mode in (MODE_KEEP, MODE_AUTORUN, MODE_SEQUENCE, MODE_CUSTOM_SEQ):
        print("[1/2] Keep/Autorun/Sequence/CustomSeq mode  reference frame used as-is.")
        return image

    # Both MODE_REPLACE and MODE_CUSTOM now use the motion prompt
    if mode == MODE_CUSTOM:
        # Custom mode: use the motion prompt directly as the edit instruction
        # This allows the prompt to guide character poses, expressions, actions
        instruction = f"Edit this image to match the following description, keeping the people's identities and features intact: {prompt}"
    else:
        # Replace mode: use the RELOCATE_INSTRUCTION template with the motion prompt
        instruction = RELOCATE_INSTRUCTION.format(prompt=prompt)

    print(f"[1/2] Qwen editing frame -> {instruction[:80]}...")
    
    # Show user-friendly message during potentially slow model swap
    swap_start = time.time()
    activate_pic()
    swap_time = time.time() - swap_start
    if swap_time > 3.0:
        print(f" Model swap completed in {swap_time:.1f}s")

    torch.cuda.set_device(PIC_DEVICE)
    with torch.cuda.device(PIC_DEVICE):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = pic_pipe(
                image=[image],
                prompt=instruction,
                negative_prompt=" ",
                num_inference_steps=int(steps),
                true_cfg_scale=float(guidance),
                generator=torch.Generator(device=PIC_DEVICE).manual_seed(seed),
            )
    edited = result.images[0]
    if edited.size != image.size:
        edited = edited.resize(image.size, Image.LANCZOS)
    return edited


def animate_frame(
    frame: Image.Image,
    last_frame,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    seed: int,
    flow_shift: float = None,
    wan_steps: int = None,
    lora_selections=None,
    selected_loras_info=None,
    progress=None,
):
    """Stage 2  animate a frame with WAMU v2 merged model."""
    activate_wan()

    # Determine which LoRAs are actually enabled (checkbox True).
    # lora_selections is always a dict (possibly all-False), never None when
    # checkboxes exist — check the VALUES, not the dict itself.
    selected = {}
    if lora_selections:
        selected = {k: v for k, v in AVAILABLE_LORAS.items() if lora_selections.get(k, False)}

    # Reload LoRAs whenever the active set changes.
    currently_active = set(_active_loras.keys())
    desired_active   = set(selected.keys())

    if currently_active != desired_active:
        load_loras_to_pipeline(wan_pipe, selected)

    # Apply trigger-prompt modifications from all enabled LoRAs.
    if selected and selected_loras_info:
        original_prompt = prompt
        prompt = apply_lora_prompt_modifications(prompt, selected_loras_info)
        if prompt != original_prompt:
            print(f"[LoRA] Prompt modified: ...{prompt[-80:]}")

    # Use provided wan_steps, or fall back to WAN_STEPS constant
    steps = wan_steps if wan_steps is not None else WAN_STEPS

    # Pin to WAN_DEVICE  prevents cross-device leakage in dual GPU mode
    with torch.cuda.device(WAN_DEVICE):
        # Apply flow shift if provided, otherwise use WAMU v2 default (6.9)
        _set_flow_shift(wan_pipe, flow_shift if flow_shift is not None else WAN_FLOW_SHIFT)

        print(f"[2/2] Wan animating {num_frames} frames at {frame.size}...")
        print(f"Seed: {seed} | Steps: {steps}")

        kwargs = dict(
            image=frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=frame.height,
            width=frame.width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=WAN_GUIDANCE,
            guidance_scale_2=WAN_GUIDANCE,
            generator=torch.Generator(device=WAN_DEVICE).manual_seed(seed),
            output_type="np",
        )

        if last_frame is None:
            return wan_pipe(**kwargs).frames[0]

        try:
            return wan_pipe(last_image=last_frame, **kwargs).frames[0]
        except TypeError as e:
            print(f"End frame not supported by this pipeline ({e}); ignoring it.")
            return wan_pipe(**kwargs).frames[0]


def concatenate_videos(video_paths: list, output_path: str):
    """Join chained segments losslessly with ffmpeg's concat demuxer."""
    if len(video_paths) == 1:
        shutil.copy(video_paths[0], output_path)
        return

    list_file = output_path + ".txt"
    with open(list_file, "w") as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_file, "-c", "copy", output_path],
            check=True, capture_output=True,
        )
    finally:
        try:
            os.unlink(list_file)
        except OSError:
            pass


def _last_frame_of(video_path: str):
    """Read the final frame of a clip as a PIL image, for chaining."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In-memory-only cache of the final frame of the most recently generated
# video. Populated by _cache_last_frame_from_video() as a .then() step
# right after generation finishes -- BEFORE the auto storage-clear chain
# deletes the video file -- and consumed by the "Last Frame from Video"
# button. This is a plain PIL Image held in a process-global; it is
# deliberately never written to disk by this code, so it survives the
# auto-clear even though nothing backs it on the filesystem, and it's
# simply gone (reset to None) on process restart.
_last_generated_frame = None
_last_generated_frame_lock = threading.Lock()


def _cache_last_frame_from_video(video_file):
    """Grab the last frame of the just-generated video into memory only.

    Wired as a .then() step immediately after every generate call (normal
    generate, Push Autorun, Sequence, Custom Edit Sequence -- all of them
    route through the same video_file output), so it runs before the
    later .then(clear_storage) step deletes the video. Reads the frame
    with _last_frame_of() and stores a copy of the resulting PIL Image in
    the module-level cache above. Nothing is written to disk here.
    """
    global _last_generated_frame
    path = video_file if isinstance(video_file, str) else None
    if not path or not os.path.exists(path):
        return
    try:
        frame = _last_frame_of(path)
    except Exception as e:
        print(f"Could not cache last frame from video: {e}")
        frame = None
    if frame is not None:
        with _last_generated_frame_lock:
            _last_generated_frame = frame.copy()


def _use_last_generated_frame():
    """Hand the in-memory cached last frame to the Reference Photo widget.

    Reads only the in-memory cache populated by
    _cache_last_frame_from_video() -- no disk access happens here. If
    nothing has been generated yet this session (or the process was
    restarted since), the cache is empty and the current Reference Photo
    is left untouched.
    """
    with _last_generated_frame_lock:
        frame = _last_generated_frame
    if frame is None:
        gr.Warning("No generated video yet this session -- generate a video first.")
        return gr.update()
    return frame.copy()


def generate_with_preset(prompt_dict, choice, reference_image, scene_mode,
                        end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                        flow_shift_auto, flow_shift):
    """Generate video using preset prompt without changing the prompt textbox."""
    if choice and choice in prompt_dict:
        preset_prompt = prompt_dict[choice]
        return generate_video(
            reference_image, preset_prompt, scene_mode,
            end_image, duration_seconds, resolution, frame_multiplier,
            export_quality, seed, randomize_seed, add_audio_cb,
            audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
            flow_shift_auto, flow_shift
        )
    return None, None

# Wrapper functions for each preset dropdown
def generate_with_solo(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                      export_quality, seed, randomize_seed, add_audio_cb,
                      audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                      flow_shift_auto, flow_shift):
    return generate_with_preset(vid_solo_prompts_dict, choice, reference_image, scene_mode,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_couple(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                        flow_shift_auto, flow_shift):
    return generate_with_preset(vid_couple_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multiple(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                          export_quality, seed, randomize_seed, add_audio_cb,
                          audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                          flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multistep(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                           export_quality, seed, randomize_seed, add_audio_cb,
                           audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                           flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multistep_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_environment(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                             export_quality, seed, randomize_seed, add_audio_cb,
                             audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                             flow_shift_auto, flow_shift):
    return generate_with_preset(vid_environment_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_custom(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                        flow_shift_auto, flow_shift):
    return generate_with_preset(vid_custom_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multiple_unseen(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                                 export_quality, seed, randomize_seed, add_audio_cb,
                                 audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                                 flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_man_unseen_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multiple_seen(choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_man_seen_prompts_dict, choice, reference_image, scene_mode,
                      end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)


def generate_video(
    reference_image,
    prompt,
    scene_mode,
    end_image=None,
    duration_seconds=3.5,
    resolution="480p",
    frame_multiplier=16,
    export_quality=7,
    seed=42,
    randomize_seed=True,
    add_audio_cb=True,
    audio_prompt_tb="realistic female breathing that matches the woman's movements and actions in video",
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,  # Variable number of LoRA checkbox states
    progress=gr.Progress(track_tqdm=True),
):
    """
    Photo(s) + prompt -> video.

    Stage 1 (optional) prepares the starting frame with Qwen Image Edit.
    Stage 2 animates it with WAMU v2 4-step Lightning (NSFW merge), chaining
    ~5.1s segments for anything longer than one native window.
    
    No interpolation by default (frame_multiplier=16 = native 16fps).
    """
    global _current_input_image_path
    
    # Convert LoRA checkbox args to dictionary and collect info
    lora_selections = {}
    selected_loras_info = {}
    if lora_args and len(lora_args) == len(AVAILABLE_LORAS):
        for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), lora_args):
            lora_selections[lora_id] = is_enabled
            if is_enabled:
                selected_loras_info[lora_id] = lora_info
    
    # Apply LoRA-recommended settings (with user override support)
    if selected_loras_info:
        edit_steps, flow_shift, lora_settings_msg = apply_lora_settings(
            selected_loras_info, edit_steps, flow_shift, flow_shift_auto
        )
        if lora_settings_msg:
            pass
    
    # Track the current input image path if it's a filepath (for clear_storage exclusion).
    # _generation_protect() adds to both the general protected set AND the
    # generation-active set, so a widget change mid-run cannot strip protection.
    if isinstance(reference_image, str):
        _current_input_image_path = reference_image
    elif hasattr(reference_image, 'filename'):
        _current_input_image_path = reference_image.filename
    else:
        _current_input_image_path = None
    _generation_protect(_current_input_image_path)

    # Also protect the end/last frame image for the whole run.
    _end_image_protect_path = None
    if isinstance(end_image, str):
        _end_image_protect_path = end_image
    elif hasattr(end_image, 'filename'):
        _end_image_protect_path = end_image.filename
    _generation_protect(_end_image_protect_path)

    # Gradio can hand back "" instead of None for an untouched optional image
    # component  normalize both reference_image and end_image so a stray
    # empty string never reaches PIL-only code (that's what caused
    # `'str' object has no attribute 'size'` on end_image).
    # Any failure while normalizing the inputs is converted to a clean
    # gr.Error instead of an unhandled exception - an unhandled exception
    # here previously left the reference/end-frame Image widgets stuck in an
    # error state that only a full page refresh could clear.
    try:
        reference_image = _ensure_pil(reference_image)
        end_image = _ensure_pil(end_image)
    except Exception as e:
        raise gr.Error(f"Could not read the input image(s): {e}")

    if reference_image is None:
        raise gr.Error("Please upload a reference photo.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the motion and scene.")
    
    #  ADAPTIVE FLOW SHIFT: Auto-adjust based on duration for better prompt following
    if flow_shift_auto:
        # Auto mode: calculate optimal flow shift based on duration
        if duration_seconds <= 6.0:
            # Short videos: High flow shift OK, prompt follows well naturally
            adaptive_flow_shift = 6.9
        elif duration_seconds <= 10.0:
            # Medium videos: Reduce flow shift for better prompt adherence
            adaptive_flow_shift = 5.5
        elif duration_seconds <= 20.0:
            # Long videos: Significantly reduce for strong prompt following
            adaptive_flow_shift = 4.5
        else:
            # Very long videos: Minimum flow shift to prioritize prompt
            adaptive_flow_shift = 4.0
        
        print(f" Auto flow_shift: {adaptive_flow_shift:.1f} (duration: {duration_seconds}s)")
        flow_shift = adaptive_flow_shift
    else:
        # Manual mode: use user-specified flow_shift
        if flow_shift is None:
            flow_shift = WAN_FLOW_SHIFT
        print(f" Manual flow_shift: {flow_shift} (user override)")

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    segment_paths = []

    try:
        if progress:
            progress(0.02, desc="Preparing reference frame")
        sized = resize_image_for_wan(reference_image, resolution)


        # ---- Stage 1: optional frame preparation --------------------------
        if progress:
            progress(0.10, desc="Preparing reference frame")
        start_frame = edit_reference_frame(
            sized, scene_mode, prompt,
            current_seed, edit_steps, edit_guidance,
        )

        # End-frame conditioning must land on the FINAL segment so the very
        # last frame of the (possibly chained) output is the user's supplied
        # end image, no matter how many ~6.1s segments the duration requires.
        processed_end = None
        if end_image is not None:
            processed_end = resize_and_crop_to_match(end_image, start_frame)

        # ---- Stage 2: animate, chaining segments as needed ---------------
        remaining = float(duration_seconds)
        current_frame = start_frame
        seg_seed = current_seed
        seg_index = 0

        while remaining > 0.01:
            seg_duration = min(remaining, SEGMENT_DURATION)
            num_frames = get_num_frames(seg_duration)
            seg_index += 1
            is_last_segment = (remaining - seg_duration) <= 0.01

            seg_end = processed_end if is_last_segment else None
            if progress:
                progress(min(0.15 + 0.75 * (seg_index - 1) / max(1, int(duration_seconds / SEGMENT_DURATION + 1)), 0.90), desc=f"Generating segment {seg_index}")
            raw_frames = animate_frame(
                current_frame, seg_end, prompt, negative_prompt,
                num_frames, seg_seed, flow_shift, edit_steps,
                lora_selections,
                selected_loras_info,
                progress,
            )

            # RIFE interpolation, per segment, before export.
            factor = max(1, int(frame_multiplier) // FIXED_FPS)
            if factor > 1:
                seg_frames = interpolate_bits(raw_frames, multiplier=factor)
            else:
                seg_frames = list(raw_frames)
            seg_fps = FIXED_FPS * factor

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                seg_path = f.name
            export_to_video(
                seg_frames, seg_path, fps=seg_fps, quality=int(export_quality)
            )
            segment_paths.append(seg_path)
            print(f"Segment {seg_index} complete ({seg_duration:.1f}s, "
                  f"{len(seg_frames)} frames @ {seg_fps} fps)")

            remaining -= seg_duration
            if remaining <= 0.01:
                break

            # Chain: the next segment starts where this one ended.
            nxt = _last_frame_of(seg_path)
            if nxt is None:
                print("Could not read segment tail frame  stopping chain here.")
                break
            current_frame = nxt
            seg_seed = random.randint(0, MAX_SEED)

        if not segment_paths:
            raise gr.Error("No video segments were produced.")

        # ---- Assemble ----------------------------------------------------
        if len(segment_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                final_path = f.name
            concatenate_videos(segment_paths, final_path)
            for p in segment_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        else:
            final_path = segment_paths[0]

        if add_audio_cb and _MMAUDIO_AVAILABLE:
            try:
                final_path = add_audio_to_video(
                    final_path, audio_prompt_tb, float(duration_seconds)
                )
            except Exception as e:
                print(f"MMAudio error: {e}")

        # Move to a descriptive, unique filename so the download arrives as
        # vidgen_<timestamp>_<token>.mp4 rather than a temp name.
        named_path = unique_output_path("vidgen", ".mp4")
        try:
            shutil.move(final_path, named_path)
            final_path = str(named_path)
        except Exception as e:
            print(f"Could not rename output ({e})  serving original path.")

        print(f"Done in {time.time() - started:.1f}s  {seg_index} segment(s), "
              f"seed {current_seed} -> {os.path.basename(final_path)}")
        if progress:
            progress(1.0, desc="Generation complete")
        # Release generation-active pins so the widget tracker can clean up
        # old paths if the user has already swapped the input images.
        _generation_release(_current_input_image_path)
        _generation_release(_end_image_protect_path)
        return final_path, final_path, gr.update(visible=False, value="")

    except gr.Error:
        _generation_release(_current_input_image_path)
        _generation_release(_end_image_protect_path)
        raise
    except Exception as e:
        _generation_release(_current_input_image_path)
        _generation_release(_end_image_protect_path)
        for p in segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        print(f"Generation error: {e}")
        raise gr.Error(f"Generation failed: {e}")


# ---------------------------------------------------------------------------
# SEQUENCE ORCHESTRATION  (chain N user-supplied images/prompts/durations
# into one continuous mp4, each segment ending exactly on the next part's
# starting image, chaining internally past SEGMENT_DURATION as needed)
# ---------------------------------------------------------------------------

def generate_sequence(
    scene_images,
    scene_prompts,
    scene_durations,
    resolution="480p",
    frame_multiplier=16,
    export_quality=7,
    seed=42,
    randomize_seed=True,
    add_audio_cb=True,
    audio_prompt_tb="realistic female breathing that matches the woman's movements and actions in video",
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Sequence mode: up to SEQUENCE_MAX_SLOTS (image, prompt, duration) parts.
    Each part is animated (chaining internally past SEGMENT_DURATION exactly
    like normal generation) so that its FINAL segment ends on the NEXT part's
    starting image - making that image both the last frame of the current
    part and the first frame of the next part. All resulting segments across
    every part are then concatenated into a single mp4.
    """
    global _current_input_image_path

    lora_selections = {}
    selected_loras_info = {}
    if lora_args and len(lora_args) == len(AVAILABLE_LORAS):
        for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), lora_args):
            lora_selections[lora_id] = is_enabled
            if is_enabled:
                selected_loras_info[lora_id] = lora_info

    if selected_loras_info:
        edit_steps, flow_shift, _ = apply_lora_settings(
            selected_loras_info, edit_steps, flow_shift, flow_shift_auto
        )

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    # ---- Gather + validate filled slots, protecting every slot image ----
    slots = []
    protected_slot_paths = []
    for i in range(SEQUENCE_MAX_SLOTS):
        raw = scene_images[i] if i < len(scene_images) else None
        try:
            img = _ensure_pil(raw)
        except Exception as e:
            raise gr.Error(f"Sequence part {i + 1}: could not read the image ({e})")
        if img is None:
            continue
        slot_path = raw if isinstance(raw, str) else (getattr(raw, "filename", None))
        if slot_path:
            protected_slot_paths.append(slot_path)
        prompt_i = (scene_prompts[i] if i < len(scene_prompts) else "") or ""
        dur_i = float(scene_durations[i]) if i < len(scene_durations) and scene_durations[i] else 0.0
        slots.append({"image": img, "prompt": prompt_i.strip(), "duration": dur_i})

    for p in protected_slot_paths:
        _generation_protect(p)   # generation-active pin (widget changes can't strip it)

    if len(slots) < 1:
        raise gr.Error("Add at least one image to the Sequence.")
    for idx, s in enumerate(slots, start=1):
        if not s["prompt"]:
            raise gr.Error(f"Sequence part {idx}: please enter a prompt.")
        if s["duration"] <= 0:
            raise gr.Error(f"Sequence part {idx}: please set a duration greater than 0.")

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    all_segment_paths = []

    try:
        n_slots = len(slots)
        total_segments = sum(
            max(1, math.ceil((s["duration"] - 0.01) / SEGMENT_DURATION)) for s in slots
        )
        seg_counter = 0

        for slot_idx, slot in enumerate(slots):
            if progress:
                progress(min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.95),
                          desc=f"Sequence part {slot_idx + 1}/{n_slots}: preparing frame")

            # Same adaptive flow-shift rule as normal generation, evaluated
            # per-part against that part's own duration.
            if flow_shift_auto:
                d = slot["duration"]
                if d <= 6.0:
                    slot_flow_shift = 6.9
                elif d <= 10.0:
                    slot_flow_shift = 5.5
                elif d <= 20.0:
                    slot_flow_shift = 4.5
                else:
                    slot_flow_shift = 4.0
            else:
                slot_flow_shift = flow_shift if flow_shift is not None else WAN_FLOW_SHIFT

            sized = resize_image_for_wan(slot["image"], resolution)
            start_frame = edit_reference_frame(
                sized, MODE_SEQUENCE, slot["prompt"], current_seed, edit_steps, edit_guidance,
            )

            # This part's FINAL segment must end on the NEXT part's image -
            # that is what makes the next part start from exactly that frame.
            target_end = None
            if slot_idx < n_slots - 1:
                next_sized = resize_image_for_wan(slots[slot_idx + 1]["image"], resolution)
                target_end = resize_and_crop_to_match(next_sized, start_frame)

            remaining = float(slot["duration"])
            current_frame = start_frame
            seg_seed = current_seed
            part_seg_index = 0

            while remaining > 0.01:
                seg_duration = min(remaining, SEGMENT_DURATION)
                num_frames = get_num_frames(seg_duration)
                part_seg_index += 1
                seg_counter += 1
                is_last_segment_of_part = (remaining - seg_duration) <= 0.01
                seg_end = target_end if is_last_segment_of_part else None

                if progress:
                    progress(min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.97),
                              desc=f"Sequence part {slot_idx + 1}/{n_slots}, segment {part_seg_index}")

                raw_frames = animate_frame(
                    current_frame, seg_end, slot["prompt"], negative_prompt,
                    num_frames, seg_seed, slot_flow_shift, edit_steps,
                    lora_selections, selected_loras_info, progress,
                )

                factor = max(1, int(frame_multiplier) // FIXED_FPS)
                if factor > 1:
                    seg_frames = interpolate_bits(raw_frames, multiplier=factor)
                else:
                    seg_frames = list(raw_frames)
                seg_fps = FIXED_FPS * factor

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    seg_path = f.name
                export_to_video(seg_frames, seg_path, fps=seg_fps, quality=int(export_quality))
                all_segment_paths.append(seg_path)
                print(f"Sequence part {slot_idx + 1}/{n_slots} segment {part_seg_index} "
                      f"complete ({seg_duration:.1f}s, {len(seg_frames)} frames @ {seg_fps} fps)")

                remaining -= seg_duration
                if remaining <= 0.01:
                    break

                # Mid-part chaining: continue from this segment's own tail
                # frame (target_end only applies to the part's LAST segment).
                nxt = _last_frame_of(seg_path)
                if nxt is None:
                    print("Could not read segment tail frame  stopping chain here.")
                    break
                current_frame = nxt
                seg_seed = random.randint(0, MAX_SEED)

        if not all_segment_paths:
            raise gr.Error("No video segments were produced.")

        if len(all_segment_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                final_path = f.name
            concatenate_videos(all_segment_paths, final_path)
            for p in all_segment_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        else:
            final_path = all_segment_paths[0]

        total_duration = sum(s["duration"] for s in slots)
        if add_audio_cb and _MMAUDIO_AVAILABLE:
            try:
                final_path = add_audio_to_video(final_path, audio_prompt_tb, float(total_duration))
            except Exception as e:
                print(f"MMAudio error: {e}")

        named_path = unique_output_path("vidgen_sequence", ".mp4")
        try:
            shutil.move(final_path, named_path)
            final_path = str(named_path)
        except Exception as e:
            print(f"Could not rename output ({e})  serving original path.")

        print(f"Sequence done in {time.time() - started:.1f}s  {n_slots} part(s) -> "
              f"{os.path.basename(final_path)}")
        if progress:
            progress(1.0, desc="Sequence generation complete")
        for p in protected_slot_paths:
            _generation_release(p)
        return final_path, final_path, gr.update(visible=False, value="")

    except gr.Error:
        for p in protected_slot_paths:
            _generation_release(p)
        raise
    except Exception as e:
        for p in protected_slot_paths:
            _generation_release(p)
        for p in all_segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        print(f"Sequence generation error: {e}")
        raise gr.Error(f"Sequence generation failed: {e}")


# ---------------------------------------------------------------------------
# CUSTOM EDIT SEQUENCE ORCHESTRATION
#
# Uses a single reference image (from the main image box). For each segment:
#   1. Runs picgen (Qwen) with the current first image + picgen_prompt to
#      generate the segment's last image.
#   2. Runs vidgen (Wan) with current first image -> generated last image,
#      using the segment's motion prompt and duration.
# The last image of each segment becomes the first image of the next.
# All segments are concatenated into one mp4.
# ---------------------------------------------------------------------------

def generate_custom_edit_sequence(
    reference_image,
    cs_motion_prompts,
    cs_picgen_prompts,
    cs_durations,
    resolution="480p",
    frame_multiplier=16,
    export_quality=7,
    seed=42,
    randomize_seed=True,
    add_audio_cb=True,
    audio_prompt_tb="realistic female breathing that matches the woman's movements and actions in video",
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Custom Edit Sequence: single starting image, N segments each with a
    motion prompt (for vidgen), a picgen prompt (to generate the last frame),
    and a duration. Each segment's generated last frame becomes the next
    segment's first frame.
    """
    global _current_input_image_path

    lora_selections = {}
    selected_loras_info = {}
    if lora_args and len(lora_args) == len(AVAILABLE_LORAS):
        for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), lora_args):
            lora_selections[lora_id] = is_enabled
            if is_enabled:
                selected_loras_info[lora_id] = lora_info

    if selected_loras_info:
        edit_steps, flow_shift, _ = apply_lora_settings(
            selected_loras_info, edit_steps, flow_shift, flow_shift_auto
        )

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    # Protect the reference image
    if isinstance(reference_image, str):
        _current_input_image_path = reference_image
    elif hasattr(reference_image, 'filename'):
        _current_input_image_path = getattr(reference_image, 'filename', None)
    else:
        _current_input_image_path = None
    _protect_path(_current_input_image_path)

    try:
        first_image = _ensure_pil(reference_image)
    except Exception as e:
        raise gr.Error(f"Could not read the reference image: {e}")
    if first_image is None:
        raise gr.Error("Please upload a reference photo.")

    # Gather filled slots
    slots = []
    for i in range(CUSTOM_SEQ_MAX_SLOTS):
        motion_p = cs_motion_prompts[i].strip() if i < len(cs_motion_prompts) else ""
        picgen_p = cs_picgen_prompts[i].strip() if i < len(cs_picgen_prompts) else ""
        dur = cs_durations[i] if i < len(cs_durations) else 0.0
        if not motion_p and not picgen_p:
            continue
        if not motion_p:
            raise gr.Error(f"Custom edit sequence segment {i+1}: please enter a motion prompt.")
        if not picgen_p:
            raise gr.Error(f"Custom edit sequence segment {i+1}: please enter a picgen (last image) prompt.")
        if dur <= 0:
            raise gr.Error(f"Custom edit sequence segment {i+1}: please set a duration greater than 0.")
        slots.append({"motion": motion_p, "picgen": picgen_p, "duration": dur})

    if not slots:
        raise gr.Error("Add at least one segment to the Custom Edit Sequence.")

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    all_segment_paths = []
    n_slots = len(slots)
    total_segments = sum(
        max(1, math.ceil((s["duration"] - 0.01) / SEGMENT_DURATION)) for s in slots
    )
    seg_counter = 0

    current_first = first_image

    try:
        for slot_idx, slot in enumerate(slots):
            if progress:
                progress(
                    min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.92),
                    desc=f"Custom seq segment {slot_idx + 1}/{n_slots}: generating last frame with picgen"
                )

            # ---- Step 1: generate the last frame via picgen ----
            sized_first = resize_image_for_wan(current_first, resolution)
            activate_pic()
            torch.cuda.set_device(PIC_DEVICE)
            pic_generator = torch.Generator(device=PIC_DEVICE).manual_seed(current_seed)
            try:
                with torch.cuda.device(PIC_DEVICE):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pic_result = pic_pipe(
                            image=[current_first],
                            prompt=slot["picgen"],
                            negative_prompt=" ",
                            num_inference_steps=int(edit_steps),
                            true_cfg_scale=float(edit_guidance),
                            generator=pic_generator,
                        )
                generated_last = pic_result.images[0]
            except Exception as e:
                raise gr.Error(f"Custom edit sequence segment {slot_idx + 1}: picgen failed: {e}")

            # Resize last frame to match the first frame dimensions
            processed_end = resize_and_crop_to_match(generated_last, sized_first)

            # ---- Step 2: animate with vidgen ----
            if progress:
                progress(
                    min(0.05 + 0.9 * (seg_counter + 0.5) / max(1, total_segments), 0.95),
                    desc=f"Custom seq segment {slot_idx + 1}/{n_slots}: animating"
                )

            # Adaptive flow shift
            if flow_shift_auto:
                d = slot["duration"]
                if d <= 6.0:
                    slot_flow_shift = 6.9
                elif d <= 10.0:
                    slot_flow_shift = 5.5
                elif d <= 20.0:
                    slot_flow_shift = 4.5
                else:
                    slot_flow_shift = 4.0
            else:
                slot_flow_shift = flow_shift if flow_shift is not None else WAN_FLOW_SHIFT

            remaining = float(slot["duration"])
            current_frame = sized_first
            seg_seed = current_seed
            part_seg_index = 0

            while remaining > 0.01:
                seg_duration = min(remaining, SEGMENT_DURATION)
                num_frames = get_num_frames(seg_duration)
                part_seg_index += 1
                seg_counter += 1
                is_last_segment_of_part = (remaining - seg_duration) <= 0.01
                seg_end = processed_end if is_last_segment_of_part else None

                if progress:
                    progress(
                        min(0.05 + 0.9 * seg_counter / max(1, total_segments), 0.97),
                        desc=f"Custom seq {slot_idx + 1}/{n_slots}, vidgen seg {part_seg_index}"
                    )

                raw_frames = animate_frame(
                    current_frame, seg_end, slot["motion"], negative_prompt,
                    num_frames, seg_seed, slot_flow_shift, edit_steps,
                    lora_selections, selected_loras_info, progress,
                )

                factor = max(1, int(frame_multiplier) // FIXED_FPS)
                if factor > 1:
                    seg_frames = interpolate_bits(raw_frames, multiplier=factor)
                else:
                    seg_frames = list(raw_frames)
                seg_fps = FIXED_FPS * factor

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    seg_path = f.name
                export_to_video(seg_frames, seg_path, fps=seg_fps, quality=int(export_quality))
                all_segment_paths.append(seg_path)
                print(f"Custom seq {slot_idx + 1}/{n_slots} vidgen seg {part_seg_index} "
                      f"complete ({seg_duration:.1f}s, {len(seg_frames)} frames @ {seg_fps} fps)")

                remaining -= seg_duration
                if remaining <= 0.01:
                    break

                nxt = _last_frame_of(seg_path)
                if nxt is None:
                    print("Could not read segment tail frame — stopping chain here.")
                    break
                current_frame = nxt
                seg_seed = random.randint(0, MAX_SEED)

            # The generated last frame becomes the next segment's first frame
            current_first = generated_last
            current_seed = random.randint(0, MAX_SEED)

        if not all_segment_paths:
            raise gr.Error("No video segments were produced.")

        if len(all_segment_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                final_path = f.name
            concatenate_videos(all_segment_paths, final_path)
            for p in all_segment_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        else:
            final_path = all_segment_paths[0]

        total_duration = sum(s["duration"] for s in slots)
        if add_audio_cb and _MMAUDIO_AVAILABLE:
            try:
                final_path = add_audio_to_video(final_path, audio_prompt_tb, float(total_duration))
            except Exception as e:
                print(f"MMAudio error: {e}")

        named_path = unique_output_path("vidgen_custom_seq", ".mp4")
        try:
            shutil.move(final_path, named_path)
            final_path = str(named_path)
        except Exception as e:
            print(f"Could not rename output ({e}) — serving original path.")

        print(f"Custom edit sequence done in {time.time() - started:.1f}s — "
              f"{n_slots} segment(s) -> {os.path.basename(final_path)}")
        if progress:
            progress(1.0, desc="Custom edit sequence complete")
        return final_path, final_path, gr.update(visible=False, value="")

    except gr.Error:
        raise
    except Exception as e:
        for p in all_segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        print(f"Custom edit sequence error: {e}")
        raise gr.Error(f"Custom edit sequence failed: {e}")


# ---------------------------------------------------------------------------
# AUTORUN ORCHESTRATION
# ---------------------------------------------------------------------------

def autorun_generate(
    prompt,
    scene_mode,          # will be MODE_AUTORUN but we still accept it for wiring
    end_image,           # ignored in autorun
    duration_seconds,
    resolution,
    frame_multiplier,
    export_quality,
    seed,
    randomize_seed,
    add_audio_cb,
    audio_prompt_tb,
    vid_negative_prompt,
    edit_steps,
    edit_guidance,
    flow_shift_auto,
    flow_shift,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Autorun mode: iterate every image in AUTORUN_DIR exactly once,
    generate one video per image using the current settings, and yield
    a (video_path, video_path, status_text) tuple after each generation
    so the Gradio event chain can download+cleanup between items.

    End frame is intentionally ignored; each autorun image is its own start frame.
    """
    global _current_input_image_path

    autorun_files = discover_autorun_images()   # raises gr.Error early if invalid
    total = len(autorun_files)
    completed = 0

    for idx, img_path in enumerate(autorun_files, start=1):
        status = f"Autorun: {idx}/{total} â€” processing {img_path.name}"
        print(f"\n[Autorun] {status}")

        _current_input_image_path = str(img_path)

        try:
            reference_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise gr.Error(
                f"Autorun stopped at {idx}/{total}: "
                f"failed to open {img_path.name} â€” {e}"
            )

        try:
            video_path, _, _ = generate_video(
                reference_image,
                prompt,
                MODE_KEEP,        # Autorun always uses Keep (no Qwen edit)
                None,             # end_image ignored
                duration_seconds,
                resolution,
                frame_multiplier,
                export_quality,
                seed,
                randomize_seed,
                add_audio_cb,
                audio_prompt_tb,
                vid_negative_prompt,
                edit_steps,
                edit_guidance,
                flow_shift_auto,
                flow_shift,
                *lora_args,
                progress=progress,
            )
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(
                f"Autorun stopped at {idx}/{total}: "
                f"failed to process {img_path.name} â€” {e}"
            )

        completed += 1
        completion_status = (
            f"Autorun complete: {completed}/{total} videos generated"
            if completed == total
            else f"Autorun: {completed}/{total} done, downloading…"
        )

        # Yield the video so the browser triggers its download via the JS chain.
        yield video_path, video_path, completion_status

        # ---- Per-video download + storage clear ----
        # The Gradio .then() chain fires once after the entire generator finishes,
        # not after each yield.  So for all but the last video we must trigger
        # clear_storage() here ourselves.  We protect the just-yielded file by
        # pointing _current_input_image_path at it so clear_storage skips it,
        # wait long enough for the browser to finish the download, then clear.
        if completed < total:
            _current_input_image_path = video_path   # protect from deletion
            time.sleep(5)                             # let browser fetch the file
            _current_input_image_path = str(img_path)  # restore to next input
            _do_clear_storage()    # wipe tmp/gradio + outputs so VPS stays clean


# ---------------------------------------------------------------------------
# PICGEN MODEL (Qwen Image Edit)
# ---------------------------------------------------------------------------

PICGEN_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BASE_MODEL_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")

#  PRIMARY MODEL LOADING
if DUAL_GPU:
    # DUAL GPU: Load both models to their dedicated GPUs simultaneously at startup
    print(f" DUAL GPU: Loading Wan -> {WAN_DEVICE} and Qwen -> {PIC_DEVICE} simultaneously...")
    
    def _load_wan_thread():
        global wan_pipe_primary
        t = time.time()
        torch.cuda.set_device(WAN_DEVICE)
        wan_pipe_primary = _load_wan(WAN_DEVICE)
        print(f" WAN ready on {WAN_DEVICE} in {time.time()-t:.1f}s")
    
    def _load_qwen_thread():
        global pic_pipe
        t = time.time()
        torch.cuda.set_device(PIC_DEVICE)
        print(" Loading Qwen Image Edit pipeline...")
        model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
        if not os.path.exists(model_index_path):
            print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
            os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=torch.bfloat16,
                cache_dir=BASE_MODEL_LOCAL_PATH,
                use_safetensors=True
            )
        else:
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                BASE_MODEL_LOCAL_PATH,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                use_safetensors=True
            )
        print("Loading NSFW weights for Qwen...")
        if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
            print("Downloading NSFW weights...")
            os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
            v23_path = hf_hub_download(
                repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
                cache_dir=PICGEN_MODELS_DIR,
                local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
            )
        else:
            v23_path = NSFW_WEIGHTS_LOCAL_PATH
        state_dict = load_file(v23_path)
        transformer_weights, vae_weights, text_encoder_weights = {}, {}, {}
        for k, v in state_dict.items():
            if k.startswith("model.diffusion_model."):
                transformer_weights[k.replace("model.diffusion_model.", "")] = v
            elif k.startswith("transformer."):
                transformer_weights[k.replace("transformer.", "")] = v
            elif k.startswith("first_stage_model."):
                vae_weights[k.replace("first_stage_model.", "")] = v
            elif k.startswith("vae."):
                vae_weights[k.replace("vae.", "")] = v
            elif "text_encoder" in k or "conditioner" in k:
                if "conditioner.embedders.0." in k:
                    text_encoder_weights[k.replace("conditioner.embedders.0.", "")] = v
                elif "text_encoder." in k:
                    text_encoder_weights[k.replace("text_encoder.", "")] = v
        if transformer_weights:
            pipe.transformer.load_state_dict(transformer_weights, strict=False)
        if vae_weights:
            pipe.vae.load_state_dict(vae_weights, strict=False)
        if text_encoder_weights:
            pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)
        del state_dict, transformer_weights, vae_weights, text_encoder_weights
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.to(PIC_DEVICE)
        pic_pipe = pipe
        print(f" Qwen ready on {PIC_DEVICE} in {time.time()-t:.1f}s")

    t_wan = threading.Thread(target=_load_wan_thread, daemon=False)
    t_qwen = threading.Thread(target=_load_qwen_thread, daemon=False)
    t_wan.start()
    t_qwen.start()
    t_wan.join()
    t_qwen.join()
    _active_model = "both"
    print(f" DUAL GPU READY  Vidgen on {WAN_DEVICE}, Picgen on {PIC_DEVICE}")

elif STARTUP_MODE == "vidgen":
    print(" VIDGEN DEFAULT: Loading Wan to GPU first for immediate use...")
    start_primary = time.time()
    
    # Load Wan directly to GPU
    wan_pipe_primary = _load_wan(WAN_DEVICE)
    _active_model = "wan"
    primary_load_time = time.time() - start_primary
    print(f" WAN READY ON GPU in {primary_load_time:.1f}s - Vidgen functional!")
    
    # Load Qwen to CPU in background for fast first Replace/Custom mode use
    pic_pipe = None
    def _bg_qwen_load():
        global pic_pipe
        print("Background: Loading Qwen to CPU for Replace/Custom modes...")
        t = time.time()
        try:
            model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
            if not os.path.exists(model_index_path):
                print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
                os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2511",
                    torch_dtype=torch.bfloat16,
                    cache_dir=BASE_MODEL_LOCAL_PATH,
                    use_safetensors=True
                )
            else:
                pipe = QwenImageEditPlusPipeline.from_pretrained(
                    BASE_MODEL_LOCAL_PATH,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                    use_safetensors=True
                )

            # Load NSFW weights
            if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
                print(f"Downloading NSFW weights...")
                os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
                v23_path = hf_hub_download(
                    repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                    filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
                    cache_dir=PICGEN_MODELS_DIR,
                    local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
                )
            else:
                v23_path = NSFW_WEIGHTS_LOCAL_PATH

            state_dict = load_file(v23_path)
            transformer_weights = {}
            vae_weights = {}
            text_encoder_weights = {}

            for k, v in state_dict.items():
                if k.startswith("model.diffusion_model."):
                    transformer_weights[k.replace("model.diffusion_model.", "")] = v
                elif k.startswith("transformer."):
                    transformer_weights[k.replace("transformer.", "")] = v
                elif k.startswith("first_stage_model."):
                    vae_weights[k.replace("first_stage_model.", "")] = v
                elif k.startswith("vae."):
                    vae_weights[k.replace("vae.", "")] = v
                elif "text_encoder" in k or "conditioner" in k:
                    if "conditioner.embedders.0." in k:
                        text_encoder_weights[k.replace("conditioner.embedders.0.", "")] = v
                    elif "text_encoder." in k:
                        text_encoder_weights[k.replace("text_encoder.", "")] = v

            if transformer_weights:
                pipe.transformer.load_state_dict(transformer_weights, strict=False)
            if vae_weights:
                pipe.vae.load_state_dict(vae_weights, strict=False)
            if text_encoder_weights:
                pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

            del state_dict, transformer_weights, vae_weights, text_encoder_weights

            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            # Keep on CPU for now, will move to GPU when needed
            pipe.to("cpu")
            pic_pipe = pipe
            print(f" Qwen loaded to CPU in {time.time()-t:.1f}s  Replace/Custom modes ready!")
        except Exception as e:
            print(f" Background Qwen load failed: {e}")
            pic_pipe = None
    
    threading.Thread(target=_bg_qwen_load, daemon=True).start()
    
else:
    # PICGEN MODE: Load Qwen to GPU first
    print(" PICGEN MODE: Loading Qwen to GPU first for immediate use...")
    
    #  AGGRESSIVE QWEN LOADING with concurrent optimization
    print(" AGGRESSIVE LOADING: Qwen Image Edit pipeline...")
    start_qwen = time.time()

    model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
    if not os.path.exists(model_index_path):
        print(f"Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
        os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16,
            cache_dir=BASE_MODEL_LOCAL_PATH,
            use_safetensors=True
        )
    else:
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            use_safetensors=True
        )

    print("Loading NSFW weights for Qwen...")
    if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
        print(f"Downloading NSFW weights...")
        os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
        v23_path = hf_hub_download(
            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
            filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
            cache_dir=PICGEN_MODELS_DIR,
            local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
        )
    else:
        v23_path = NSFW_WEIGHTS_LOCAL_PATH

    print("Loading NSFW state dict...")
    state_dict = load_file(v23_path)

    transformer_weights = {}
    vae_weights = {}
    text_encoder_weights = {}

    for k, v in state_dict.items():
        if k.startswith("model.diffusion_model."):
            transformer_weights[k.replace("model.diffusion_model.", "")] = v
        elif k.startswith("transformer."):
            transformer_weights[k.replace("transformer.", "")] = v
        elif k.startswith("first_stage_model."):
            vae_weights[k.replace("first_stage_model.", "")] = v
        elif k.startswith("vae."):
            vae_weights[k.replace("vae.", "")] = v
        elif "text_encoder" in k or "conditioner" in k:
            if "conditioner.embedders.0." in k:
                text_encoder_weights[k.replace("conditioner.embedders.0.", "")] = v
            elif "text_encoder." in k:
                text_encoder_weights[k.replace("text_encoder.", "")] = v

    if transformer_weights:
        pic_pipe.transformer.load_state_dict(transformer_weights, strict=False)
    if vae_weights:
        pic_pipe.vae.load_state_dict(vae_weights, strict=False)
    if text_encoder_weights:
        pic_pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

    del state_dict, transformer_weights, vae_weights, text_encoder_weights
    torch.cuda.empty_cache()

    pic_pipe.vae.enable_tiling()
    pic_pipe.vae.enable_slicing()

    # Load Qwen to GPU for picgen mode
    pic_pipe.transformer.to(PIC_DEVICE)
    pic_pipe.text_encoder.to(PIC_DEVICE) 
    pic_pipe.vae.to(PIC_DEVICE)
    
    qwen_time = time.time() - start_qwen
    print(f" QWEN READY ON GPU in {qwen_time:.1f}s - Picgen functional!")
    _active_model = "pic"

_swap_lock = threading.Lock()


# AGGRESSIVE CONCURRENT LOADING
def _concurrent_component_load(component_loader_fn, device, component_name):
    """Load a single component to device concurrently."""
    print(f"    Loading {component_name} to {device}...")
    start_time = time.time()
    component_loader_fn()
    load_time = time.time() - start_time
    print(f"    {component_name} loaded in {load_time:.1f}s")
    return component_name, load_time

def _aggressive_pipeline_load(repo_id, device, pipeline_name):
    """Aggressively load pipeline with concurrent components and memory optimization."""
    print(f" AGGRESSIVE LOADING: {pipeline_name} to {device}")
    start_time = time.time()
    
    # Load with maximum optimization parameters
    if "wan" in pipeline_name.lower():
        pipeline = WanImageToVideoPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
    else:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
    
    # CONCURRENT COMPONENT LOADING with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{pipeline_name}_loader") as executor:
        futures = []
        
        # Create component loading tasks
        if hasattr(pipeline, 'transformer'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.transformer.to(device, non_blocking=True),
                device, "transformer"
            ))
        if hasattr(pipeline, 'transformer_2'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.transformer_2.to(device, non_blocking=True),
                device, "transformer_2"
            ))
        if hasattr(pipeline, 'text_encoder'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.text_encoder.to(device, non_blocking=True),
                device, "text_encoder"
            ))
        if hasattr(pipeline, 'vae'):
            futures.append(executor.submit(
                _concurrent_component_load,
                lambda: pipeline.vae.to(device, non_blocking=True),
                device, "vae"
            ))
        
        # Wait for all components to load concurrently
        total_components = len(futures)
        completed_times = []
        for future in as_completed(futures):
            component_name, load_time = future.result()
            completed_times.append(load_time)
            print(f"     {component_name} ready ({len(completed_times)}/{total_components})")
    
    # Synchronize all GPU transfers
    torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    print(f" {pipeline_name} LOADED in {total_time:.1f}s (concurrent speedup: {sum(completed_times)/total_time:.1f}x)")
    
    return pipeline


def activate_wan():
    """Ensure Wan is on WAN_DEVICE and ready."""
    global _active_model

    if DUAL_GPU:
        # Dual GPU: Wan is always on WAN_DEVICE, no swap needed
        # Just ensure it's loaded (it should be from startup)
        if not _wan_loaded or wan_pipe is None:
            _load_wan(WAN_DEVICE)
        return

    if _active_model == "wan":
        return

    print(" Fast swap to Wan...")
    start_time = time.time()

    with _swap_lock:
        if _active_model == "wan":
            return

        # Only move Qwen to CPU if it's currently on GPU
        if _active_model == "pic" and pic_pipe is not None:
            pic_pipe.to("cpu")

        torch.cuda.empty_cache()

        # If already loaded, just move to GPU. Otherwise load fresh.
        if _wan_loaded and wan_pipe is not None:
            wan_pipe.to(WAN_DEVICE)
        else:
            _load_wan(WAN_DEVICE)

        _active_model = "wan"
        swap_time = time.time() - start_time
        print(f" Wan active in {swap_time:.1f}s")


def activate_pic():
    """Ensure Qwen is on PIC_DEVICE and ready."""
    global _active_model

    if DUAL_GPU:
        # Dual GPU: Qwen is always on PIC_DEVICE, no swap needed
        if pic_pipe is None:
            raise RuntimeError("Qwen pipeline not loaded  dual GPU startup failed.")
        return

    if _active_model == "pic":
        return

    # If pic_pipe hasn't loaded yet (vidgen background load still running), wait for it
    if pic_pipe is None:
        print("Waiting for Qwen to finish loading in background...")
        wait_start = time.time()
        while pic_pipe is None:
            time.sleep(0.5)
            if time.time() - wait_start > 120:
                raise RuntimeError("Qwen failed to load within 120 seconds")
        print(f" Qwen background load complete, proceeding with swap")

    print(" Fast swap to Qwen...")
    start_time = time.time()

    with _swap_lock:
        if _active_model == "pic":
            return

        # Only move Wan to CPU if it's currently on GPU
        if _active_model == "wan" and _wan_loaded and wan_pipe is not None:
            wan_pipe.to("cpu")

        torch.cuda.empty_cache()

        # Move Qwen to GPU
        pic_pipe.to(PIC_DEVICE)

        _active_model = "pic"
        swap_time = time.time() - start_time
        print(f" Qwen active in {swap_time:.1f}s")

PICGEN_MAX_SEED = np.iinfo(np.int32).max

# Thread pool for async base64 decoding (CPU-bound operation)
_decode_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="b64decode")

# Cache for VAE latents and prompt embeddings to speed up repeated generations
_picgen_cache = {
    "vae_latents": {},      # keyed by image hash
    "prompt_embeds": {},    # keyed by (prompt, neg_prompt, images_hash)
}
_picgen_cache_lock = threading.Lock()
MAX_CACHE_ENTRIES = 20  # Keep last 20 to avoid memory bloat


def _hash_images(images):
    """Create a stable hash from a list of PIL images."""
    hasher = hashlib.sha256()
    for img in images:
        # Hash image size and a sample of pixels for speed
        hasher.update(f"{img.size}".encode())
        img_array = np.array(img.resize((64, 64), Image.LANCZOS))
        hasher.update(img_array.tobytes())
    return hasher.hexdigest()


def _get_cached_vae_latents(images):
    """Get cached VAE latents for images if available."""
    img_hash = _hash_images(images)
    with _picgen_cache_lock:
        return _picgen_cache["vae_latents"].get(img_hash)


def _cache_vae_latents(images, latents):
    """Cache VAE latents for images."""
    img_hash = _hash_images(images)
    with _picgen_cache_lock:
        cache = _picgen_cache["vae_latents"]
        if len(cache) >= MAX_CACHE_ENTRIES:
            # Remove oldest entry
            cache.pop(next(iter(cache)))
        cache[img_hash] = latents


def _get_cached_prompt_embeds(prompt, negative_prompt, images, num_images_per_prompt):
    """Get cached prompt embeddings if available."""
    img_hash = _hash_images(images)
    # Cache key includes num_images_per_prompt since that affects the output shape
    key = (prompt, negative_prompt or "", img_hash, num_images_per_prompt)
    with _picgen_cache_lock:
        return _picgen_cache["prompt_embeds"].get(key)


def _cache_prompt_embeds(prompt, negative_prompt, images, num_images_per_prompt, embeds_data):
    """Cache prompt embeddings."""
    img_hash = _hash_images(images)
    key = (prompt, negative_prompt or "", img_hash, num_images_per_prompt)
    with _picgen_cache_lock:
        cache = _picgen_cache["prompt_embeds"]
        if len(cache) >= MAX_CACHE_ENTRIES:
            # Remove oldest entry
            cache.pop(next(iter(cache)))
        cache[key] = embeds_data


def add_starter_image(starter_num):
    starter_path = os.path.join(SCRIPT_DIR, f"starters/start{starter_num}.jpg")
    if not os.path.exists(starter_path):
        return ""
    with open(starter_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{b64}"


def _decode_single_b64(b64_str):
    """Decode a single base64 image string to PIL (used by thread pool)."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        if b64_str.startswith("data:image"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def b64_to_pil_list(b64_json_str):
    """Decode base64 JSON array to PIL images using thread pool for speedup."""
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    
    # For single image, skip thread pool overhead
    if len(b64_list) == 1:
        img = _decode_single_b64(b64_list[0])
        return [img] if img is not None else []
    
    # Use thread pool for parallel decoding (CPU-bound)
    try:
        from concurrent.futures import as_completed
        futures = {_decode_executor.submit(_decode_single_b64, b64_str): idx 
                   for idx, b64_str in enumerate(b64_list)}
        pil_images = [None] * len(b64_list)
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            if result is not None:
                pil_images[idx] = result
        return [img for img in pil_images if img is not None]
    except Exception as e:
        print(f"Thread pool decode failed, falling back to sequential: {e}")
        # Fallback to sequential
        pil_images = []
        for b64_str in b64_list:
            img = _decode_single_b64(b64_str)
            if img is not None:
                pil_images.append(img)
        return pil_images


def _do_clear_storage():
    """
    Module-level storage clear helper called by infer_with_preclear(),
    autorun_generate(), and autorun_push_generate() — all module-level
    functions that cannot reach the Gradio-scoped clear_storage() closure.

    Mirrors clear_storage() exactly:
      - tmp/gradio entries (skipping vibe_edit_history and protected paths)
      - loose files in tmp root (skipping the gradio subdir and protected paths)
      - additional find/rm sweep for stray .mp4 files in tmp
      - outputs/images contents
      - outputs/videos contents

    Returns the count of successfully deleted items.
    """
    import shutil as _shutil
    import subprocess as _subprocess

    deleted = []

    # 1. tmp/gradio — same three candidate paths as clear_storage()
    for gradio_dir in [
        Path.cwd() / "tmp" / "gradio",
        Path(SCRIPT_DIR) / "tmp" / "gradio",
        Path("/root/newgen/tmp/gradio"),
    ]:
        if gradio_dir.exists():
            for item in gradio_dir.iterdir():
                if item.name == "vibe_edit_history":
                    continue
                if _is_protected(item):
                    continue
                try:
                    removed = False
                    if item.is_dir():
                        try:
                            _shutil.rmtree(item)
                            removed = True
                        except Exception:
                            result = _subprocess.run(["rm", "-rf", str(item)], capture_output=True)
                            removed = result.returncode == 0
                    else:
                        try:
                            item.unlink()
                            removed = True
                        except Exception:
                            result = _subprocess.run(["rm", "-f", str(item)], capture_output=True)
                            removed = result.returncode == 0
                    if removed:
                        deleted.append(str(item))
                except Exception:
                    pass
            break  # only process the first found directory

    # 2. Loose files in tmp root (mp4s etc.) — skip gradio subdir and other dirs
    for tmp_dir in [Path.cwd() / "tmp", Path(SCRIPT_DIR) / "tmp", Path("/root/newgen/tmp")]:
        if tmp_dir.exists():
            for item in tmp_dir.iterdir():
                if item.is_dir() and item.name == "gradio":
                    continue
                if item.is_dir():
                    continue
                if _is_protected(item):
                    continue
                try:
                    removed = False
                    try:
                        item.unlink()
                        removed = True
                    except Exception:
                        result = _subprocess.run(["rm", "-f", str(item)], capture_output=True)
                        removed = result.returncode == 0
                    if removed:
                        deleted.append(str(item))
                except Exception:
                    pass
            break

    # 3. Additional backup: force-delete stray .mp4 files via find (mirrors clear_storage)
    for tmp_dir in [Path.cwd() / "tmp", Path(SCRIPT_DIR) / "tmp", Path("/root/newgen/tmp")]:
        if tmp_dir.exists():
            _subprocess.run(
                ["find", str(tmp_dir), "-maxdepth", "1", "-name", "*.mp4", "-type", "f", "-delete"],
                capture_output=True, check=False,
            )
            _subprocess.run(
                f"rm -f {tmp_dir}/*.mp4 2>/dev/null || true",
                shell=True, check=False,
            )
            break

    # 4. outputs/images — delete contents, keep folder
    for images_dir in [IMAGE_OUTPUT_DIR, Path.cwd() / "outputs" / "images", Path("/root/newgen/outputs/images")]:
        if images_dir.exists():
            for item in images_dir.iterdir():
                if _is_protected(item):
                    continue
                try:
                    if item.is_dir():
                        try:
                            _shutil.rmtree(item)
                        except Exception:
                            _subprocess.run(["rm", "-rf", str(item)], check=False)
                    else:
                        try:
                            item.unlink()
                        except Exception:
                            _subprocess.run(["rm", "-f", str(item)], check=False)
                    deleted.append(str(item))
                except Exception:
                    pass
            break

    # 5. outputs/videos — delete contents, keep folder
    for videos_dir in [VIDEO_OUTPUT_DIR, Path.cwd() / "outputs" / "videos", Path("/root/newgen/outputs/videos")]:
        if videos_dir.exists():
            for item in videos_dir.iterdir():
                try:
                    if item.is_dir():
                        try:
                            _shutil.rmtree(item)
                        except Exception:
                            _subprocess.run(["rm", "-rf", str(item)], check=False)
                    else:
                        try:
                            item.unlink()
                        except Exception:
                            _subprocess.run(["rm", "-f", str(item)], check=False)
                    deleted.append(str(item))
                except Exception:
                    pass
            break

    return len(deleted)


def infer_with_preclear(
    images_b64_json,
    prompt,
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Wrapper around infer() that clears storage (tmp/gradio + outputs) before
    running generation, while preserving any input images currently loaded in
    the picgen gallery (they are in-memory base64 in the browser and are NOT
    on-disk files that need protecting, so the clear is safe).
    """
    # Clear storage before starting — delegates to the module-level helper
    # which uses the correct TMPDIR (/root/newgen/tmp/gradio) and the same
    # path logic as the Gradio-scoped clear_storage() button handler.
    try:
        n = _do_clear_storage()
        print(f"[picgen pre-clear] cleared {n} item(s)")
    except Exception as _e:
        print(f"[picgen pre-clear] storage clear failed (non-fatal): {_e}")

    return infer(
        images_b64_json, prompt, negative_prompt, seed, randomize_seed,
        true_guidance_scale, num_inference_steps, height, width,
        num_images_per_prompt, progress,
    )


def infer(
    images_b64_json,
    prompt,
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, PICGEN_MAX_SEED)

    _t_enter = time.time()
    activate_pic()
    _t_active = time.time()

    # Pin to PIC_DEVICE  prevents cross-device leakage in dual GPU mode
    torch.cuda.set_device(PIC_DEVICE)
    generator = torch.Generator(device=PIC_DEVICE).manual_seed(seed)
    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image.")
    _t_decoded = time.time()

    if height == 256 and width == 256:
        height, width = None, None

    print(f"Seed: {seed} | Steps: {num_inference_steps}")
    print(f"  input images: {[im.size for im in pil_images]}")
    
    # Check cache for prompt embeddings (cache key includes num_images_per_prompt)
    cached_embeds = _get_cached_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt)
    cache_status = "cached" if cached_embeds else "computing"
    
    print(f"  timing: activate {_t_active - _t_enter:.2f}s, "
          f"decode {_t_decoded - _t_active:.2f}s, embeds: {cache_status} "
          f"(active model: {_active_model}, dual_gpu: {DUAL_GPU})")
    _t_pipe = time.time()

    # Temporarily patch pipeline methods to use cache
    original_encode_prompt = pic_pipe.encode_prompt
    original_prepare_latents = pic_pipe.prepare_latents
    
    encode_called = [False]
    prepare_called = [False]
    
    def cached_encode_prompt(*args, **kwargs):
        encode_called[0] = True
        if cached_embeds is not None:
            return cached_embeds["prompt_embeds"], cached_embeds["prompt_embeds_mask"]
        result = original_encode_prompt(*args, **kwargs)
        # Cache the result for next time
        embeds_data = {
            "prompt_embeds": result[0],
            "prompt_embeds_mask": result[1]
        }
        _cache_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt, embeds_data)
        return result
    
    def cached_prepare_latents(images, *args, **kwargs):
        prepare_called[0] = True
        # Don't use cache for prepare_latents - too complex with batching
        # Just call original and cache the result
        result = original_prepare_latents(images, *args, **kwargs)
        if images is not None and result[1] is not None:
            # Cache the image_latents for next time (but we won't use it due to complexity)
            # Keeping this for future improvement
            pass
        return result
    
    pic_pipe.encode_prompt = cached_encode_prompt
    pic_pipe.prepare_latents = cached_prepare_latents
    
    try:
        with torch.cuda.device(PIC_DEVICE):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image = pic_pipe(
                    image=pil_images if pil_images else None,
                    prompt=prompt,
                    height=height,
                    width=width,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    true_cfg_scale=true_guidance_scale,
                    num_images_per_prompt=num_images_per_prompt,
                ).images
    finally:
        # Restore original methods
        pic_pipe.encode_prompt = original_encode_prompt
        pic_pipe.prepare_latents = original_prepare_latents

    print(f"  pipeline call took {time.time() - _t_pipe:.2f}s")

    # Persist each result under a unique, fully-qualified filename. Returning
    # real paths (instead of in-memory PIL objects) is what makes the download
    # button serve a proper .png name.
    multiple = len(image) > 1
    saved_paths = []
    for i, img in enumerate(image, start=1):
        out_path = unique_output_path("picgen", ".png", index=i if multiple else None)
        img.save(out_path, format="PNG")
        saved_paths.append(str(out_path))
    print(f"  saved: {[os.path.basename(p) for p in saved_paths]}")

    # Convert saved paths to proper Gradio format with visible labels
    gallery_items = [(path, os.path.basename(path)) for path in saved_paths]
    
    return gallery_items, seed


# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------

gallery_js = r"""
() => {
function init() {
    if (window.__picgenInitDone) return;
    const galleryGrid  = document.getElementById('image-gallery-grid');
    const dropZone     = document.getElementById('gallery-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick  = document.getElementById('upload-click-area');
    const fileInput    = document.getElementById('custom-file-input');
    const btnUpload    = document.getElementById('tb-upload');
    const btnRemove    = document.getElementById('tb-remove');
    const btnClear     = document.getElementById('tb-clear');
    if (!galleryGrid || !fileInput || !dropZone) { setTimeout(init, 250); return; }
    window.__picgenInitDone = true;
    let images = [];
    window.__uploadedImages = images;
    let selectedIdx = -1;

    function syncToGradio() {
        window.__uploadedImages = images;
        const b64Array = images.map(img => img.b64);
        const container = document.getElementById('hidden-images-b64');
        if (!container) return;
        container.querySelectorAll('input,textarea').forEach(el => {
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, JSON.stringify(b64Array));
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }

    function addImage(b64, name) {
        images.push({id: Date.now() + Math.random(), b64: b64, name: name});
        renderGallery(); syncToGradio();
    }
    window.__addImage = addImage;

    function removeImage(idx) {
        images.splice(idx, 1);
        if (selectedIdx === idx) selectedIdx = -1;
        else if (selectedIdx > idx) selectedIdx--;
        renderGallery(); syncToGradio();
    }

    function clearAll() {
        images = []; window.__uploadedImages = images; selectedIdx = -1;
        renderGallery(); syncToGradio();
    }
    window.__clearAll = clearAll;

    function renderGallery() {
        if (images.length === 0) {
            galleryGrid.innerHTML = ''; galleryGrid.style.display = 'none';
            if (uploadPrompt) uploadPrompt.style.display = '';
            return;
        }
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        galleryGrid.style.display = 'grid';
        let html = '';
        images.forEach((img, i) => {
            const sel = i === selectedIdx ? ' selected' : '';
            html += '<div class="gallery-thumb' + sel + '" data-idx="' + i + '">'
                  + '<img src="' + img.b64 + '" alt="' + (img.name||'image') + '">'
                  + '<span class="thumb-badge">#' + (i+1) + '</span>'
                  + '<button class="thumb-remove" data-remove="' + i + '">\u2715</button>'
                  + '</div>';
        });
        html += '<div class="gallery-add-card" id="gallery-add-card"><span class="add-icon">+</span><span class="add-text">Add</span></div>';
        galleryGrid.innerHTML = html;
        galleryGrid.querySelectorAll('.gallery-thumb').forEach(thumb => {
            thumb.addEventListener('click', (e) => {
                if (e.target.closest('.thumb-remove')) return;
                const idx = parseInt(thumb.dataset.idx);
                showLightbox(images[idx].b64);
            });
        });
        galleryGrid.querySelectorAll('.thumb-remove').forEach(btn => {
            btn.addEventListener('click', (e) => { e.stopPropagation(); removeImage(parseInt(btn.dataset.remove)); });
        });
        const addCard = document.getElementById('gallery-add-card');
        if (addCard) addCard.addEventListener('click', () => fileInput.click());
    }

    function showLightbox(b64) {
        let lb = document.getElementById('picgen-lightbox');
        if (!lb) {
            lb = document.createElement('div');
            lb.id = 'picgen-lightbox';
            lb.style.cssText = 'display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:9999;align-items:center;justify-content:center;';
            lb.innerHTML = '<div style="position:relative;max-width:80vw;max-height:80vh;">'
                + '<img id="picgen-lb-img" style="max-width:80vw;max-height:80vh;border-radius:8px;display:block;">'
                + '<button id="picgen-lb-close" style="position:absolute;top:-14px;right:-14px;width:28px;height:28px;border-radius:50%;background:#e53e3e;color:#fff;border:none;cursor:pointer;font-size:16px;line-height:1;"></button>'
                + '</div>';
            document.body.appendChild(lb);
            lb.addEventListener('click', (e) => { if (e.target === lb) lb.style.display = 'none'; });
            document.getElementById('picgen-lb-close').addEventListener('click', () => lb.style.display = 'none');
        }
        document.getElementById('picgen-lb-img').src = b64;
        lb.style.display = 'flex';
    }

    function processFiles(files) {
        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    // Resize to 512px (pipeline resizes to this for VAE anyway)
                    const maxSize = 512;
                    let width = img.width;
                    let height = img.height;
                    
                    if (width > height) {
                        height = Math.round((height * maxSize) / width);
                        width = maxSize;
                    } else {
                        width = Math.round((width * maxSize) / height);
                        height = maxSize;
                    }
                    
                    // Create canvas and resize
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    // Convert to base64 with quality optimization
                    const resizedB64 = canvas.toDataURL('image/jpeg', 0.95);
                    addImage(resizedB64, file.name);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        });
    }

    fileInput.addEventListener('change', (e) => { processFiles(e.target.files); e.target.value = ''; });
    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload) btnUpload.addEventListener('click', () => fileInput.click());
    if (btnRemove) btnRemove.addEventListener('click', () => { if (selectedIdx >= 0) removeImage(selectedIdx); });
    if (btnClear) btnClear.addEventListener('click', clearAll);
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files); });
    renderGallery();
}
init();
}
"""

css = """
body, .gradio-container { margin: 0 !important; padding: 0 !important; max-width: 100% !important; }
#col-container { margin: 0 !important; max-width: 100% !important; padding: 0 !important; }
.contain { padding: 0 !important; }
#preset-row { display: flex !important; align-items: center !important; gap: 8px !important; }
#preset-row > * { flex: 1 !important; }
#preset-row button { flex: 0 0 auto !important; min-width: 80px !important; }
#preset-row input[type="text"] { pointer-events: none !important; user-select: none !important; }
.hidden-input { display: none !important; height: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; }
#gallery-drop-zone { position: relative; min-height: 320px; overflow: auto; border: 1px solid var(--border-color-primary); border-radius: 8px; margin-bottom: 8px; }
#gallery-drop-zone.drag-over { outline: 2px solid var(--color-accent); outline-offset: -2px; }
.upload-prompt-modern { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 20; }
.upload-click-area { display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; padding: 36px 52px; border: 2px dashed var(--border-color-primary); border-radius: 16px; transition: all .2s ease; gap: 8px; }
.upload-click-area:hover { border-color: var(--color-accent); transform: scale(1.03); }
.upload-click-area svg { width: 64px; height: 64px; }
.upload-main-text { font-size: 14px; font-weight: 500; margin-top: 4px; }
.upload-sub-text { font-size: 12px; color: var(--body-text-color-subdued); }
.image-gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; padding: 12px; align-content: start; }
.gallery-thumb { position: relative; aspect-ratio: 1; border-radius: 8px; overflow: hidden; cursor: pointer; border: 2px solid var(--border-color-primary); transition: all .2s ease; }
.gallery-thumb:hover { border-color: var(--color-accent); transform: translateY(-2px); }
.gallery-thumb.selected { border-color: var(--color-accent) !important; box-shadow: 0 0 0 3px rgba(var(--color-accent-soft), .3); }
.gallery-thumb img { width: 100%; height: 100%; object-fit: cover; }
.thumb-badge { position: absolute; top: 5px; left: 5px; background: var(--color-accent); color: #fff; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; }
.thumb-remove { position: absolute; top: 5px; right: 5px; width: 22px; height: 22px; background: rgba(0,0,0,.7); color: #fff; border: 1px solid rgba(255,255,255,.2); border-radius: 50%; cursor: pointer; display: none; align-items: center; justify-content: center; font-size: 11px; transition: all .15s; line-height: 1; }
.gallery-thumb:hover .thumb-remove { display: flex; }
.thumb-remove:hover { background: #e53e3e; }
.gallery-add-card { aspect-ratio: 1; border-radius: 8px; border: 2px dashed var(--border-color-primary); display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; transition: all .2s ease; gap: 4px; }
.gallery-add-card:hover { border-color: var(--color-accent); }
.gallery-add-card .add-icon { font-size: 26px; font-weight: 300; }
.gallery-add-card .add-text { font-size: 12px; font-weight: 500; }
.uploader-toolbar { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
.tb-btn { display: inline-flex; align-items: center; gap: 5px; padding: 5px 12px; border: 1px solid var(--border-color-primary); border-radius: 6px; background: var(--background-fill-secondary); cursor: pointer; font-size: 12px; font-weight: 500; transition: all .15s; }
.tb-btn:hover { border-color: var(--color-accent); }
/* Constrain reference photo and generated video to fit on screen */
#vidgen-reference img, #vidgen-reference video { max-height: 320px !important; width: 100% !important; object-fit: contain !important; }
#generated-video video { max-height: 320px !important; width: 100% !important; object-fit: contain !important; }
/* Show Media = off: hide actual media pixels but keep every container/upload-zone intact.
   Uses visibility:hidden so the container box stays; only the rendered image/video disappears. */
body.hide-media #vidgen-reference img,
body.hide-media #vidgen-reference video,
body.hide-media #generated-video video,
body.hide-media #end-frame-image img,
body.hide-media #end-frame-image video,
body.hide-media #image-gallery-grid img,
body.hide-media #picgen-result-gallery img,
body.hide-media #picgen-result-gallery video { visibility: hidden !important; }
/* LoRA cards grid - 3 per row, compact */
.lora-grid { display: grid !important; grid-template-columns: repeat(3, 1fr) !important; gap: 10px !important; padding: 4px 0 !important; }
.lora-card { border: 1px solid var(--border-color-primary) !important; border-radius: 8px !important; padding: 10px 12px !important; background: var(--background-fill-secondary) !important; display: flex !important; flex-direction: column !important; gap: 4px !important; }
.lora-card-header { display: flex !important; align-items: center !important; justify-content: space-between !important; gap: 6px !important; }
.lora-status-badges { display: flex !important; gap: 4px !important; flex-shrink: 0 !important; }
.lora-badge { font-size: 10px !important; padding: 1px 5px !important; border-radius: 3px !important; font-weight: 600 !important; white-space: nowrap !important; }
.lora-badge-ok { background: #22543d !important; color: #9ae6b4 !important; }
.lora-badge-dl { background: #744210 !important; color: #fbd38d !important; }
.lora-badge-miss { background: #742a2a !important; color: #feb2b2 !important; }
.lora-desc { font-size: 12px !important; color: var(--body-text-color-subdued) !important; line-height: 1.4 !important; margin: 2px 0 !important; }
.lora-notes { font-size: 11px !important; color: var(--body-text-color-subdued) !important; line-height: 1.35 !important; border-top: 1px solid var(--border-color-primary) !important; padding-top: 5px !important; margin-top: 2px !important; }
.lora-notes summary { cursor: pointer !important; font-size: 11px !important; font-weight: 500 !important; user-select: none !important; margin-bottom: 3px !important; }
.lora-card .gr-checkbox-label { font-size: 13px !important; font-weight: 600 !important; }
@media (max-width: 900px) { .lora-grid { grid-template-columns: repeat(2, 1fr) !important; } }
@media (max-width: 600px) { .lora-grid { grid-template-columns: 1fr !important; } }
/* Make picgen prompt textareas manually resizable */
#col-container textarea { resize: vertical !important; min-height: 60px !important; touch-action: pan-y !important; }
"""

# AUTORUN PUSH API
#
# A tiny HTTP server on port 7861 that lets your local machine push images
# into the app one at a time over SSH tunnel, entirely in memory.
#
# Protocol (all requests hit 127.0.0.1:7861 via SSH -L tunnel):
#
#   GET  /autorun/status   -> JSON {"state": "idle"|"ready"|"busy"|"done"}
#   POST /autorun/push     -> multipart/form-data with field "file"
#                             Returns {"accepted": true, "filename": "..."} or error
#   POST /autorun/cancel   -> abort a running push-mode autorun
#
# The local feeder script polls /status, pushes the next image when "ready",
# waits for "idle" (generation+download+cleanup done), then pushes the next.
# ---------------------------------------------------------------------------

_PUSH_API_PORT = 7861

# State machine: idle -> ready -> busy -> idle (loops) | done | error
_push_state = "idle"          # current state string
_push_lock   = threading.Lock()
_push_image_queue: "_queue.Queue[tuple]" = _queue.Queue(maxsize=1)  # (filename, PIL.Image)
_push_cancel = threading.Event()
# Holds the bytes of the most recently generated video so the feeder can pull
# it via GET /autorun/download instead of needing a browser download.
_push_pending_video: dict = {"bytes": None, "name": None, "ready": False}


def _set_push_state(s: str):
    global _push_state
    with _push_lock:
        _push_state = s
    print(f"[AutorunAPI] state -> {s}")


def _start_push_api():
    """Run a minimal HTTP server for the push API in a daemon thread."""
    import http.server
    import urllib.parse

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silence default access log

        def _send_json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/autorun/status":
                with _push_lock:
                    s = _push_state
                self._send_json(200, {"state": s})

            elif self.path == "/autorun/download":
                # Serve the most recently generated video as raw bytes so the
                # local feeder can write it to PUSH_LOCAL_DOWNLOAD_DIR without
                # needing a browser.  Clears the pending slot after sending.
                with _push_lock:
                    ready  = _push_pending_video.get("ready", False)
                    vbytes = _push_pending_video.get("bytes")
                    vname  = _push_pending_video.get("name", "vidgen.mp4")
                if not ready or not vbytes:
                    self._send_json(404, {"error": "no video pending"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(vbytes)))
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{vname}"')
                self.end_headers()
                self.wfile.write(vbytes)
                # Clear after serving so a stale download isn't re-fetched.
                with _push_lock:
                    _push_pending_video["bytes"] = None
                    _push_pending_video["ready"] = False
                print(f"[AutorunAPI] /autorun/download served {vname} "
                      f"({len(vbytes)//1024} KB) to local feeder")

            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/autorun/cancel":
                _push_cancel.set()
                _set_push_state("idle")
                self._send_json(200, {"cancelled": True})
                return

            if self.path == "/autorun/ready":
                with _push_lock:
                    state = _push_state
                if state not in ("busy", "done"):
                    self._send_json(409, {"error": f"not awaiting completion (state={state})"})
                    return
                _set_push_state("ready")
                self._send_json(200, {"ready": True})
                return

            if self.path != "/autorun/push":
                self._send_json(404, {"error": "not found"})
                return

            with _push_lock:
                state = _push_state
            if state != "ready":
                self._send_json(409, {"error": f"not ready (state={state})"})
                return

            # Parse multipart body to extract the image file
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                self._send_json(400, {"error": "expected multipart/form-data"})
                return

            import cgi
            length = int(self.headers.get("Content-Length", 0))
            fs = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": str(length),
                },
            )
            file_item = fs.getvalue("file")
            filename   = fs["file"].filename if "file" in fs else "image.jpg"

            if file_item is None:
                self._send_json(400, {"error": "missing 'file' field"})
                return

            try:
                img = Image.open(BytesIO(file_item if isinstance(file_item, bytes) else file_item.read())).convert("RGB")
            except Exception as e:
                self._send_json(400, {"error": f"cannot decode image: {e}"})
                return

            _set_push_state("busy")
            _push_image_queue.put((filename, img))
            self._send_json(200, {"accepted": True, "filename": filename})

    server = http.server.HTTPServer(("127.0.0.1", _PUSH_API_PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[AutorunAPI] listening on 127.0.0.1:{_PUSH_API_PORT}  (SSH-tunnel only)")


def autorun_push_generate(
    prompt,
    duration_seconds,
    resolution,
    frame_multiplier,
    export_quality,
    seed,
    randomize_seed,
    add_audio_cb,
    audio_prompt_tb,
    vid_negative_prompt,
    edit_steps,
    edit_guidance,
    flow_shift_auto,
    flow_shift,
    *lora_args,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Push-mode autorun: waits for images sent from the local machine via the
    push API, processes each one, yields status, then signals ready for next.
    Runs until cancelled or until the local feeder sends no more images
    (detected by a 60-second timeout after the last 'ready' signal).
    """
    global _current_input_image_path
    _push_cancel.clear()
    completed = 0

    _set_push_state("ready")

    while not _push_cancel.is_set():
        # Wait for exactly one image. The next request is accepted only after
        # the browser completes the download/cleanup acknowledgement.
        try:
            filename, img = _push_image_queue.get(timeout=900)
        except _queue.Empty:
            _set_push_state("idle")
            yield None, None, f"Push autorun complete: {completed} video(s) generated (timed out waiting for next image)"
            return

        if _push_cancel.is_set():
            _set_push_state("idle")
            return

        completed += 1
        status = f"Push autorun: processing {filename} (#{completed})"
        print(f"\n[AutorunPush] {status}")
        _current_input_image_path = None  # in-memory image, no path to protect

        try:
            video_path, _, _ = generate_video(
                img,
                prompt,
                MODE_KEEP,
                None,
                duration_seconds,
                resolution,
                frame_multiplier,
                export_quality,
                seed,
                randomize_seed,
                add_audio_cb,
                audio_prompt_tb,
                vid_negative_prompt,
                edit_steps,
                edit_guidance,
                flow_shift_auto,
                flow_shift,
                *lora_args,
                progress=progress,
            )
        except gr.Error:
            _set_push_state("idle")
            raise
        except Exception as e:
            _set_push_state("idle")
            raise gr.Error(f"Push autorun failed on {filename}: {e}")

        # ---- Queue video bytes for local pull via /autorun/download -----------
        # State transitions: busy -> done  (feeder sees "done", downloads, then
        # POSTs /autorun/ready to transition done -> ready for next image).
        push_status = f"Push autorun: {completed} done, ready for local download"
        if video_path and os.path.exists(video_path):
            out_name = os.path.basename(video_path)
            try:
                with open(video_path, "rb") as _vf:
                    _vbytes = _vf.read()
                with _push_lock:
                    _push_pending_video["bytes"] = _vbytes
                    _push_pending_video["name"]  = out_name
                    _push_pending_video["ready"] = True
                print(f"[AutorunPush] video queued for local pull ({len(_vbytes)//1024} KB): {out_name}")
            except Exception as _e:
                print(f"[AutorunPush] could not queue video for local download: {_e}")

        # Transition to "done" so the feeder knows to fetch /autorun/download.
        _set_push_state("done")
        yield video_path, video_path, push_status

        # Clear VPS storage while the feeder is downloading.
        # The video bytes are already in memory (_push_pending_video) so deleting
        # the file on disk is safe.
        time.sleep(1)
        _do_clear_storage()
        print(f"[AutorunPush] storage cleared after #{completed} — waiting for /autorun/ready")

        # Wait for the feeder to POST /autorun/ready (after confirming the local
        # file is written), which transitions done -> ready for the next image.
        while not _push_cancel.is_set():
            with _push_lock:
                s = _push_state
            if s == "ready":
                break
            time.sleep(0.25)

    _set_push_state("idle")




with gr.Blocks(css=css) as demo:
    # Clear button  outside tabs, always visible at top right
    with gr.Row():
        gr.HTML("<div style='flex:1'></div>")  # spacer pushes button right
        clear_storage_btn = gr.Button("Clear Storage", variant="secondary", size="sm", scale=0)
    clear_storage_status = gr.Textbox(visible=False, label="")

    def protect_current_inputs(reference_image, end_image, *sequence_images):
        """Explicit pre-clear protection step for the automatic storage clear.

        Runs as a .then() step BEFORE clear_storage() in the generate/download
        chains: reads the first frame (reference), last frame (end), and any
        Sequence-mode slot images currently loaded in the input widgets, and
        registers their on-disk paths in the protected set so the full
        storage wipe that follows cannot delete them out from under the
        widgets. The previous per-generation protection (inside
        generate_video) only covered images of the run in flight; this step
        additionally covers the widgets' *current* inputs at clear time.

        Two protection layers are applied:
          1. Full path registered in _protected_image_paths (existing behaviour).
          2. Basename registered in _protected_image_filenames so that even if
             Gradio has moved or re-indexed the file, anything sharing the same
             filename is guaranteed to survive the clear.

        Accepts the same image component values Gradio hands to change
        handlers: None, "", filepath str, or objects with .name/.filename.
        """

        def _path_of(img):
            if img is None or img == "":
                return None
            if isinstance(img, str):
                return img
            for attr in ("name", "filename"):
                v = getattr(img, attr, None)
                if isinstance(v, str) and v:
                    return v
            return None

        # Reset the filename-protection set fresh each time so stale names
        # from a previous generation don't pile up and over-protect.
        with _protected_filenames_lock:
            _protected_image_filenames.clear()

        for img in (reference_image, end_image, *sequence_images):
            p = _path_of(img)
            if p:
                _protect_path(p)
                _protect_filename(p)  # layer 2: basename protection
        return None

    def clear_storage():
        """Delete all generated files  same as running clear.sh."""
        import shutil as _shutil
        import subprocess
        import time
        global _current_input_image_path
        
        deleted = []
        errors = []


        # BACKUP METHOD 1: Try relative path from current working directory
        gradio_dir_cwd = Path.cwd() / "tmp" / "gradio"
        
        # BACKUP METHOD 2: Try relative path from script directory
        gradio_dir_script = Path(SCRIPT_DIR) / "tmp" / "gradio"
        
        # BACKUP METHOD 3: Try absolute path on VPS
        gradio_dir_abs = Path("/root/newgen/tmp/gradio")
        
        # Try all possible paths for tmp/gradio
        for gradio_dir in [gradio_dir_cwd, gradio_dir_script, gradio_dir_abs]:
            if gradio_dir.exists():
                for item in gradio_dir.iterdir():
                    if item.name == "vibe_edit_history":
                        continue
                    
                    # Check if we should skip this item (file or directory) -
                    # protects the reference image, end frame, AND any
                    # Sequence-mode slot images that are currently loaded in
                    # an input widget (across all in-flight generations).
                    if _is_protected(item):
                        continue
                    
                    try:
                        # Try multiple deletion methods
                        removed = False
                        if item.is_dir():
                            try:
                                _shutil.rmtree(item)
                                removed = True
                            except Exception as e1:
                                # Backup: try subprocess rm
                                result = subprocess.run(["rm", "-rf", str(item)], capture_output=True)
                                removed = result.returncode == 0
                        else:
                            try:
                                item.unlink()
                                removed = True
                            except Exception as e2:
                                # Backup: try subprocess rm
                                result = subprocess.run(["rm", "-f", str(item)], capture_output=True)
                                removed = result.returncode == 0
                        if removed:
                            deleted.append(str(item))
                    except Exception as e:
                        errors.append(f"{item.name}: {e}")
                break  # Only process first found directory

        # Also clean up loose files in tmp root folder (INCLUDING .mp4 files)
        for tmp_dir in [Path.cwd() / "tmp", Path(SCRIPT_DIR) / "tmp", Path("/root/newgen/tmp")]:
            if tmp_dir.exists():
                for item in tmp_dir.iterdir():
                    # Skip the gradio subdirectory (already handled above)
                    if item.is_dir() and item.name == "gradio":
                        continue
                    # Skip other directories
                    if item.is_dir():
                        continue
                    
                    # Skip current input image(s)
                    if _is_protected(item):
                        continue
                    
                    # Delete ALL loose files (including .mp4, .png, etc)
                    try:
                        try:
                            item.unlink()
                            removed = True
                        except Exception as e3:
                            result = subprocess.run(["rm", "-f", str(item)], capture_output=True)
                            removed = result.returncode == 0
                        if removed:
                            deleted.append(str(item))
                    except Exception as e:
                        errors.append(f"{item.name}: {e}")
                break

        # ADDITIONAL BACKUP: Force delete all .mp4 files in tmp using find command
        for tmp_dir in [Path.cwd() / "tmp", Path(SCRIPT_DIR) / "tmp", Path("/root/newgen/tmp")]:
            if tmp_dir.exists():
                subprocess.run(
                    ["find", str(tmp_dir), "-maxdepth", "1", "-name", "*.mp4", "-type", "f", "-delete"],
                    capture_output=True, check=False
                )
                # Also try with rm for good measure
                subprocess.run(
                    f"rm -f {tmp_dir}/*.mp4 2>/dev/null || true",
                    shell=True, check=False
                )
                break

        # 2. outputs/images  delete contents, keep folder
        # Try multiple paths
        for images_dir in [IMAGE_OUTPUT_DIR, Path.cwd() / "outputs" / "images", Path("/root/newgen/outputs/images")]:
            if images_dir.exists():
                for item in images_dir.iterdir():
                    try:
                        if item.is_dir():
                            try:
                                _shutil.rmtree(item)
                            except:
                                subprocess.run(["rm", "-rf", str(item)], check=False)
                        else:
                            try:
                                item.unlink()
                            except:
                                subprocess.run(["rm", "-f", str(item)], check=False)
                        deleted.append(str(item))
                    except Exception as e:
                        errors.append(f"{item.name}: {e}")
                break

        # 3. outputs/videos  delete contents, keep folder
        for videos_dir in [VIDEO_OUTPUT_DIR, Path.cwd() / "outputs" / "videos", Path("/root/newgen/outputs/videos")]:
            if videos_dir.exists():
                for item in videos_dir.iterdir():
                    try:
                        if item.is_dir():
                            try:
                                _shutil.rmtree(item)
                            except:
                                subprocess.run(["rm", "-rf", str(item)], check=False)
                        else:
                            try:
                                item.unlink()
                            except:
                                subprocess.run(["rm", "-f", str(item)], check=False)
                        deleted.append(str(item))
                    except Exception as e:
                        errors.append(f"{item.name}: {e}")
                break

        if errors:
            return gr.update(visible=True, value=f" Done with errors: {'; '.join(errors[:5])}")
        return gr.update(visible=True, value=f" Cleared {len(deleted)} items.")




    clear_storage_btn.click(
        fn=clear_storage,
        inputs=[],
        outputs=[clear_storage_status],
    )

    # Global Show Media checkbox â€” hides all input/output media when unchecked.
    # Default: unchecked (hidden) for privacy / distraction-free use.
    with gr.Row():
        show_media_cb = gr.Checkbox(
            label="Show Media",
            value=False,
            info="When checked, displays input images and generated output on screen.",
        )

    # Tab 0 = Video Generator (vidgen), Tab 1 = Photo Editor (picgen).
    # -vidgen (default) opens on the Video Generator tab; -picgen opens on the
    # Photo Editor tab.
    with gr.Tabs(selected=(0 if STARTUP_MODE == "vidgen" else 1)):

        # ------------------------------------------------------------------ #
        #  TAB 1  VIDEO GENERATOR (Qwen relocate -> Wan 2.2 4-step animate)  #
        # ------------------------------------------------------------------ #
        with gr.Tab(" Video Generator"):
            gr.Markdown(model_title())

            with gr.Row():
                with gr.Column(scale=1):
                    reference_image = gr.Image(
                        label="Reference Photo",
                        type="filepath",
                        elem_id="vidgen-reference",
                    )
                    last_frame_from_video_btn = gr.Button(
                        "Last Frame from Video", size="sm",
                    )
                    vid_prompt = gr.Textbox(
                        label="Motion & Scene Prompt",
                        value=default_video_prompt,
                        lines=4,
                        placeholder=(
                            "Describe the motion, action, lighting and camera. "
                            "In Replace-background mode this also describes the new setting."
                        ),
                    )
                    vid_negative_prompt = gr.Textbox(
                        label="Negative Prompt", value=default_negative_prompt, lines=2,
                    )

                    scene_mode = gr.Radio(
                        choices=SCENE_MODES,
                        value=MODE_KEEP,
                        label="Scene Handling",
                        info=(
                            "Keep = photo animated untouched, background and bodies "
                            "identical. Replace = new environment with motion prompt. "
                            "Custom = motion prompt applied as direct edit instruction. "
                            "Autorun = process all images in the 'autorun' folder "
                            "sequentially using the current prompt and settings. "
                            "Sequence = chain up to 10 of your own image/prompt/duration "
                            "parts into one continuous video. "
                            "Custom Edit Sequence = single reference photo, N segments each "
                            "with a picgen prompt (Qwen generates the last frame) and a "
                            "motion prompt (Wan animates from current to generated frame); "
                            "uses Generation Steps and Frame-Edit Guidance sliders."
                        ),
                    )

                    with gr.Row():
                        duration_seconds = gr.Slider(
                            MIN_DURATION, MAX_DURATION, value=3.5, step=0.5,
                            label="Duration (seconds)",
                            info=f"Over {SEGMENT_DURATION}s is produced by chaining segments.",
                            scale=2,
                        )
                        with gr.Column(scale=1, min_width=200):
                            flow_shift_auto = gr.Checkbox(
                                label="Auto Flow Shift",
                                value=True,
                            )
                            flow_shift = gr.Slider(
                                1.0, 12.0, value=WAN_FLOW_SHIFT, step=0.1,
                                label="Flow Shift",
                                info="Lower = better prompt. Higher = more motion.",
                                interactive=False,
                            )

                    # Quick duration presets -- one click sets the slider
                    # straight to that value, no dragging required.
                    def _make_duration_setter(value):
                        def _set_duration():
                            return gr.update(value=value)
                        return _set_duration

                    with gr.Row():
                        for _dur_preset in (3.5, 6, 12, 18, 24, 30):
                            _dur_btn = gr.Button(
                                str(_dur_preset), size="sm", min_width=40,
                            )
                            _dur_btn.click(
                                fn=_make_duration_setter(_dur_preset),
                                inputs=[],
                                outputs=[duration_seconds],
                            )
                    
                    # Steps slider - auto-updates based on selected LoRAs
                    with gr.Row():
                        edit_steps = gr.Slider(
                            1, 20, value=4, step=1,
                            label="Generation Steps",
                            info="Auto-set by LoRAs or manually override",
                        )
                    
                    # Enable/disable flow_shift slider based on auto checkbox
                    def update_flow_shift_interactivity(auto_enabled):
                        if auto_enabled:
                            return gr.update(interactive=False, value=WAN_FLOW_SHIFT)
                        else:
                            return gr.update(interactive=True)
                    
                    flow_shift_auto.change(
                        fn=update_flow_shift_interactivity,
                        inputs=[flow_shift_auto],
                        outputs=[flow_shift],
                    )

                    with gr.Row():
                        add_audio_cb = gr.Checkbox(label="Add Audio (MMAudio)", value=True)
                        audio_prompt_tb = gr.Textbox(
                            label="Audio Prompt", value="realistic female breathing that matches the woman's movements and actions in video",
                        )

                with gr.Column(scale=1):
                    vidgen_progress = gr.Markdown(
                        "", visible=False, elem_id="vidgen-progress"
                    )
                    video_output = gr.Video(
                        label="Generated Video",
                        elem_id="generated-video",
                        autoplay=True,
                        interactive=False,
                    )
                    
                    # Generate button directly under video output
                    generate_btn = gr.Button(
                        "Generate Video", variant="primary", size="lg", elem_id="generate-btn"
                    )
                    # Clear storage button directly under generate button
                    clear_storage_btn_vid = gr.Button(
                        "Clear Storage", variant="secondary", size="lg",
                    )
                    
                    with gr.Row():
                        frame_time_input = gr.Number(
                            label="Frame time (seconds) - auto-updates as video plays",
                            value=0.0,
                            minimum=0.0,
                            step=0.1,
                            scale=1,
                            elem_id="frame-time-input",
                        )
                    with gr.Row():
                        use_as_reference_btn = gr.Button(
                            "Use Frame as Reference",
                            "Use Frame as Reference",
                        )
                        download_frame_btn = gr.Button(
                            "Download Frame",
                            "Download Frame",
                        )
                    
                    # Compact 4x2 grid layout for preset prompts
                    with gr.Row():
                        vid_preset_dropdown = gr.Dropdown(
                            label="Solo Prompts",
                            choices=list(vid_solo_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown2 = gr.Dropdown(
                            label="Couple Prompts",
                            choices=list(vid_couple_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )

                    with gr.Row():
                        vid_preset_dropdown3 = gr.Dropdown(
                            label="Multiple Prompts",
                            choices=list(vid_multiple_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown4 = gr.Dropdown(
                            label="Multi-Step Prompts",
                            choices=list(vid_multistep_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )

                    with gr.Row():
                        vid_preset_dropdown5 = gr.Dropdown(
                            label="Environment Prompts",
                            choices=list(vid_environment_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown6 = gr.Dropdown(
                            label="Custom Prompts",
                            choices=list(vid_custom_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )

                    with gr.Row():
                        vid_preset_dropdown7 = gr.Dropdown(
                            label="Multiple (Man Unseen)",
                            choices=list(vid_multiple_man_unseen_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                        vid_preset_dropdown8 = gr.Dropdown(
                            label="Multiple (Man Seen)",
                            choices=list(vid_multiple_man_seen_prompts_dict.keys()),
                            value=None, interactive=True, scale=1,
                        )
                    
                    end_image = gr.Image(
                        label="End Frame (optional - used as the true last frame, "
                              "even when the duration requires chaining multiple segments)",
                        type="filepath",
                        elem_id="end-frame-image",
                    )
                    use_last_as_first_btn = gr.Button(
                        "Use as first frame",
                        variant="secondary",
                        size="sm",
                        elem_id="use-last-as-first-btn",
                    )

            # ------------------------------------------------------------ #
            #  SEQUENCE MODE  up to SEQUENCE_MAX_SLOTS (image, prompt,      #
            #  duration) parts, each chained onto the next so part N's      #
            #  last frame is part N+1's first frame, assembled into one mp4 #
            # ------------------------------------------------------------ #
            with gr.Group(visible=False) as sequence_group:
                gr.Markdown(
                    "**Sequence** — fill in as many of the parts below as you need "
                    "(others left empty are skipped). Each part's image is animated "
                    "with its own prompt and duration (chaining internally past "
                    f"{SEGMENT_DURATION}s automatically) and ends exactly on the next "
                    "part's image, so the whole thing plays as one continuous clip. "
                    "Uses the Generation Steps / resolution / quality settings above."
                )
                sequence_images = []
                sequence_prompts = []
                sequence_durations = []
                for _seq_i in range(SEQUENCE_MAX_SLOTS):
                    with gr.Row():
                        _seq_img = gr.Image(
                            label=f"Part {_seq_i + 1} image",
                            type="filepath",
                            scale=1,
                            elem_id=f"sequence-image-{_seq_i}",
                        )
                        _seq_prompt = gr.Textbox(
                            label=f"Part {_seq_i + 1} prompt",
                            lines=2,
                            scale=2,
                            elem_id=f"sequence-prompt-{_seq_i}",
                        )
                        _seq_dur = gr.Slider(
                            MIN_DURATION, MAX_DURATION, value=3.5, step=0.5,
                            label=f"Part {_seq_i + 1} duration (s)",
                            scale=1,
                            elem_id=f"sequence-duration-{_seq_i}",
                        )
                    sequence_images.append(_seq_img)
                    sequence_prompts.append(_seq_prompt)
                    sequence_durations.append(_seq_dur)

                sequence_status = gr.Textbox(visible=False, label="")

            # ------------------------------------------------------------ #
            #  CUSTOM EDIT SEQUENCE MODE  up to CUSTOM_SEQ_MAX_SLOTS       #
            #  segments, each with a motion prompt (vidgen), a picgen       #
            #  prompt (to generate the last frame via Qwen), and a          #
            #  duration. Uses the main Reference Photo as the first image.  #
            # ------------------------------------------------------------ #
            with gr.Group(visible=False) as custom_seq_group:
                gr.Markdown(
                    "**Custom Edit Sequence** — uses your Reference Photo (top-left) as the "
                    "starting image. For each segment below, picgen generates the last frame "
                    "from the current image using your picgen prompt, then vidgen animates from "
                    "the current image to that generated last frame using your motion prompt. "
                    "The generated last frame becomes the next segment's first image. "
                    "The main Motion & Scene Prompt above is **ignored** — each segment uses its own motion prompt."
                )
                custom_seq_motion_prompts = []
                custom_seq_picgen_prompts = []
                custom_seq_durations = []
                for _csi in range(CUSTOM_SEQ_MAX_SLOTS):
                    with gr.Row():
                        _cs_motion = gr.Textbox(
                            label=f"Seg {_csi + 1} motion prompt (vidgen)",
                            lines=2,
                            scale=2,
                            placeholder="Describe the motion/animation for this segment",
                            elem_id=f"custom-seq-motion-{_csi}",
                        )
                        _cs_picgen = gr.Textbox(
                            label=f"Seg {_csi + 1} picgen prompt (last frame)",
                            lines=2,
                            scale=2,
                            placeholder="Describe what to generate as the final image of this segment",
                            elem_id=f"custom-seq-picgen-{_csi}",
                        )
                        _cs_dur = gr.Slider(
                            MIN_DURATION, MAX_DURATION, value=3.5, step=0.5,
                            label=f"Seg {_csi + 1} duration (s)",
                            scale=1,
                            elem_id=f"custom-seq-duration-{_csi}",
                        )
                    custom_seq_motion_prompts.append(_cs_motion)
                    custom_seq_picgen_prompts.append(_cs_picgen)
                    custom_seq_durations.append(_cs_dur)

            # Hidden file component - populated by generate_btn and used for frame extraction
            video_file = gr.File(visible=False)
            
            # LoRA Selection Accordion
            lora_checkboxes = {}
            lora_download_btns = {}
            lora_example_dropdowns = {}
            with gr.Accordion("LoRA Models (Optional)", open=False):
                if AVAILABLE_LORAS:
                    gr.Markdown(
                        "**Select LoRAs to apply.** "
                        "Checkboxes enable/disable. Download missing files. "
                        "Select example prompt to load it into the prompt box."
                    )
                    
                    # Sort LoRAs alphabetically by display name
                    sorted_loras = sorted(
                        AVAILABLE_LORAS.items(),
                        key=lambda x: x[1].get('display_name', x[0]).lower()
                    )
                    
                    # Download status display (shared across all cards)
                    lora_download_status = gr.Markdown("", visible=False)
                    
                    # Download handlers
                    def download_lora_handler(lora_id):
                        """Download missing LoRA files for a specific LoRA."""
                        global AVAILABLE_LORAS, LORA_STATUS
                        config = LORA_CONFIG.get(lora_id, {})
                        results = []
                        success_count = 0
                        if config.get('high_url') and config.get('high_filename'):
                            high_path = LORA_DIR / config['high_filename']
                            if not high_path.exists():
                                success, msg = download_lora_file(config['high_url'], config['high_filename'])
                                results.append(f"**High:** {msg}")
                                if success:
                                    success_count += 1
                            else:
                                results.append(f"**High:** Already downloaded")
                        if config.get('low_url') and config.get('low_filename'):
                            low_path = LORA_DIR / config['low_filename']
                            if not low_path.exists():
                                success, msg = download_lora_file(config['low_url'], config['low_filename'])
                                results.append(f"**Low:** {msg}")
                                if success:
                                    success_count += 1
                            else:
                                results.append(f"**Low:** Already downloaded")
                        if not results:
                            return gr.update(visible=True, value="Nothing to download")
                        AVAILABLE_LORAS = discover_loras()
                        LORA_STATUS = check_lora_status(LORA_CONFIG)
                        status_msg = "\n".join(results)
                        if success_count > 0:
                            status_msg += f"\n\n**{success_count} file(s) downloaded.** Refresh page to enable checkbox."
                        return gr.update(visible=True, value=status_msg)
                    
                    # Render cards in rows of 3
                    card_index = 0
                    current_row = None
                    for lora_id, lora_info in sorted_loras:
                        status = LORA_STATUS.get(lora_id, {})
                        display_name = lora_info.get('display_name', lora_id)
                        description = lora_info.get('description', '')
                        notes = lora_info.get('notes', '')
                        trigger = lora_info.get('trigger_prompt', '')
                        
                        high_exists = bool(lora_info['high'])
                        low_exists = bool(lora_info['low'])
                        high_downloadable = status.get('high_downloadable', False)
                        low_downloadable = status.get('low_downloadable', False)
                        needs_download = (not high_exists and high_downloadable) or (not low_exists and low_downloadable)
                        can_use = high_exists or low_exists
                        
                        high_status = "âœ“" if high_exists else ("â†“" if high_downloadable else "âœ—")
                        low_status = "âœ“" if low_exists else ("â†“" if low_downloadable else "âœ—")
                        high_cls = "lora-badge-ok" if high_exists else ("lora-badge-dl" if high_downloadable else "lora-badge-miss")
                        low_cls = "lora-badge-ok" if low_exists else ("lora-badge-dl" if low_downloadable else "lora-badge-miss")
                        
                        # Build notes HTML - escape for HTML display
                        notes_html = ""
                        if notes:
                            import html as _html
                            notes_escaped = _html.escape(notes)
                            notes_html = f'<details class="lora-notes"><summary>Notes</summary><span>{notes_escaped}</span></details>'
                        
                        trigger_html = ""
                        if trigger:
                            import html as _html
                            trigger_escaped = _html.escape(trigger[:80] + ("..." if len(trigger) > 80 else ""))
                            trigger_html = f'<div class="lora-desc" style="font-size:10px;opacity:0.7;">Trigger: <code>{trigger_escaped}</code></div>'
                        
                        # Open new row every 3 cards
                        if card_index % 3 == 0:
                            current_row = gr.Row(elem_classes="lora-grid-row")
                            current_row.__enter__()
                        
                        with gr.Column(elem_classes="lora-card", min_width=200):
                            # Status badges + description header (HTML)
                            gr.HTML(
                                f'<div class="lora-status-badges" style="margin-bottom:2px;">'
                                f'<span class="lora-badge {high_cls}">H:{high_status}</span>'
                                f'<span class="lora-badge {low_cls}">L:{low_status}</span>'
                                f'</div>'
                                f'<div class="lora-desc">{description}</div>'
                                f'{trigger_html}'
                                f'{notes_html}'
                            )
                            
                            # Checkbox (enable/disable)
                            lora_checkboxes[lora_id] = gr.Checkbox(
                                label=display_name,
                                value=False,
                                interactive=can_use,
                            )
                            
                            # Download button if needed
                            if needs_download:
                                dl_parts = []
                                if not high_exists and high_downloadable:
                                    dl_parts.append("High")
                                if not low_exists and low_downloadable:
                                    dl_parts.append("Low")
                                lora_download_btns[lora_id] = gr.Button(
                                    f"â†“ Download {' + '.join(dl_parts)}",
                                    size="sm",
                                    variant="secondary",
                                )
                            
                            # Example prompts dropdown
                            example_prompts = lora_info.get('example_prompts', [])
                            if example_prompts:
                                prompt_choices = {ex['name']: ex['prompt'] for ex in example_prompts}
                                lora_example_dropdowns[lora_id] = gr.Dropdown(
                                    label="Example prompt",
                                    choices=list(prompt_choices.keys()),
                                    value=None,
                                    interactive=True,
                                )
                                
                                def create_example_handler(prompts_dict):
                                    def handler(selected):
                                        if selected and selected in prompts_dict:
                                            return prompts_dict[selected]
                                        return gr.update()
                                    return handler
                                
                                lora_example_dropdowns[lora_id].change(
                                    fn=create_example_handler(prompt_choices),
                                    inputs=[lora_example_dropdowns[lora_id]],
                                    outputs=[vid_prompt],
                                )
                        
                        card_index += 1
                        # Close row after every 3rd card or at end
                        if card_index % 3 == 0 or card_index == len(sorted_loras):
                            current_row.__exit__(None, None, None)
                    
                    # Attach download handlers
                    for lora_id, btn in lora_download_btns.items():
                        btn.click(
                            fn=lambda lid=lora_id: download_lora_handler(lid),
                            inputs=[],
                            outputs=[lora_download_status],
                        )
                
                else:
                    gr.Markdown(
                        "**No LoRAs configured.**  \n"
                        f"Add LoRA entries to `{LORA_CONFIG_FILE}` or place `.safetensors` files in `{LORA_DIR}` and restart."
                    )
            
            # LoRA Compatibility Status Display
            lora_compat_status = gr.Markdown(
                "", 
                visible=False,
                elem_classes="lora-compat-status"
            )
            
            # Function to update compatibility status AND steps slider when checkboxes change
            def update_lora_compatibility_and_steps(*checkbox_states):
                """Show compatibility status and update steps slider when LoRAs are selected."""
                if not AVAILABLE_LORAS:
                    return gr.update(visible=False, value=""), gr.update()
                
                # Collect selected LoRAs
                selected = {}
                for (lora_id, lora_info), is_enabled in zip(AVAILABLE_LORAS.items(), checkbox_states):
                    if is_enabled:
                        selected[lora_id] = lora_info
                
                if not selected:
                    return gr.update(visible=False, value=""), gr.update(value=4)
                
                # Check compatibility
                is_compatible, message, settings = check_lora_compatibility(selected)
                
                # Build status message
                status_lines = [f"### Active LoRAs: {len(selected)}"]
                status_lines.append(message)
                
                if settings.get('recommended_steps'):
                    status_lines.append(f"**Recommended Steps:** {settings['recommended_steps']}")
                if settings.get('recommended_flow_shift'):
                    status_lines.append(f"**Recommended Flow Shift:** {settings['recommended_flow_shift']}")
                
                if len(selected) > 1:
                    status_lines.append(f"**Average Weights:** High={settings['high_weight']:.2f}, Low={settings['low_weight']:.2f}")
                
                if not is_compatible:
                    status_lines.append("\n**Note:** Settings conflict - using defaults. You can manually override.")
                
                # Selecting a LoRA immediately applies its recommendation.
                # The user can subsequently move the slider to override it.
                recommended_steps = settings.get('recommended_steps')
                
                return gr.update(visible=True, value="\n".join(status_lines)), gr.update(value=recommended_steps) if recommended_steps is not None else gr.update()
            
            # Attach compatibility checker to all LoRA checkboxes
            if lora_checkboxes:
                for checkbox in lora_checkboxes.values():
                    checkbox.change(
                        fn=update_lora_compatibility_and_steps,
                        inputs=[lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes],
                        outputs=[lora_compat_status, edit_steps],
                    )

            # Download Frame output gr.File shows a clickable download link when populated
            download_file_output = gr.File(label="Click to Download Frame", visible=True)

            # ── Merge Photos ──────────────────────────────────────────────────────
            with gr.Accordion("Merge Photos", open=False):
                gr.Markdown(
                    "Upload two photos. Backgrounds are removed and both subjects are "
                    "placed side by side on a white canvas (1280×720), ready to use as "
                    "a first frame for video generation."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        merge_img_a = gr.Image(
                            label="Person A",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        merge_img_b = gr.Image(
                            label="Person B",
                            type="pil",
                            sources=["upload", "clipboard"],
                        )
                        merge_btn = gr.Button("Merge", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        merge_output = gr.Image(
                            label="Merged Result (1280×720)",
                            type="pil",
                            interactive=False,
                        )
                        use_as_first_frame_btn = gr.Button(
                            "Use as First Frame",
                            variant="secondary",
                            size="lg",
                        )

                def _do_merge(a, b):
                    if a is None or b is None:
                        return gr.update()
                    result = merge_photos_fn(a, b)
                    return gr.update(value=result)

                merge_btn.click(
                    fn=_do_merge,
                    inputs=[merge_img_a, merge_img_b],
                    outputs=[merge_output],
                )

                def _use_merged_as_first_frame(merged):
                    if merged is None:
                        return gr.update()
                    return gr.update(value=merged)

                use_as_first_frame_btn.click(
                    fn=_use_merged_as_first_frame,
                    inputs=[merge_output],
                    outputs=[reference_image],
                )
            # ── End Merge Photos ──────────────────────────────────────────────────

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    resolution = gr.Radio(
                        choices=["720p", "480p"], value="480p",
                        label="Resolution",
                    )
                    frame_multiplier = gr.Slider(
                        16, 64, value=16, step=16,
                        label="Output FPS (RIFE interpolation, 16=off)",
                    )
                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    export_quality = gr.Slider(
                        1, 10, value=7, step=1, label="Export Quality",
                    )
                with gr.Row():
                    edit_guidance = gr.Slider(
                        1.0, 10.0, value=1.0, step=0.1,
                        label="Frame-Edit Guidance (Qwen stage)",
                    )
                gr.Markdown(
                    "**WAMU v2 settings:** 4 steps (locked), guidance 1.0 (distillation-merged). "
                    "Flow shift adjustable (default 6.9). "
                    "Qwen sliders affect frame-edit stage only, ignored in Keep-scene mode."
                )

            # Vidgen prompt dropdown handlers (change events update prompt textbox)
            vid_preset_dropdown.change(fn=update_vid_prompt1, inputs=[vid_preset_dropdown], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown2.change(fn=update_vid_prompt2, inputs=[vid_preset_dropdown2], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown3.change(fn=update_vid_prompt3, inputs=[vid_preset_dropdown3], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown4.change(fn=update_vid_prompt4, inputs=[vid_preset_dropdown4], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown5.change(fn=update_vid_prompt5, inputs=[vid_preset_dropdown5], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown6.change(fn=update_vid_prompt6, inputs=[vid_preset_dropdown6], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown7.change(fn=update_vid_prompt7, inputs=[vid_preset_dropdown7], outputs=[vid_prompt], scroll_to_output=False)
            vid_preset_dropdown8.change(fn=update_vid_prompt8, inputs=[vid_preset_dropdown8], outputs=[vid_prompt], scroll_to_output=False)

            def _noop_download(f):
                """Pass-through function for download chain."""
                return f

            # ---- vidgen autorun progress textbox ----
            autorun_status = gr.Textbox(
                label="Autorun Status", visible=False, interactive=False,
            )

            # Push Autorun button â€” starts waiting for images from local machine via SSH tunnel
            with gr.Row(visible=False) as push_autorun_row:
                push_autorun_btn = gr.Button(
                    "â–¶ Start Push Autorun (waiting for local feeder)",
                    variant="primary", size="lg",
                )
                push_cancel_btn = gr.Button("â–  Cancel", variant="stop", size="lg")

            # Show push_autorun_row only when Autorun mode is selected; show
            # the Sequence part builder only when Sequence mode is selected;
            # show the Custom Edit Sequence panel only for that mode;
            # hide the main vid_prompt when Sequence or Custom Edit Sequence
            # mode is active (those modes use their own per-segment prompts).
            def _scene_mode_visibility(m):
                is_seq = (m == MODE_SEQUENCE)
                is_cseq = (m == MODE_CUSTOM_SEQ)
                is_autorun = (m == MODE_AUTORUN)
                # Hide main prompt when per-segment prompts are used
                prompt_visible = not (is_seq or is_cseq)
                return (
                    gr.update(visible=is_autorun),    # autorun_status
                    gr.update(visible=is_autorun),    # push_autorun_row
                    gr.update(visible=is_seq),         # sequence_group
                    gr.update(visible=is_cseq),        # custom_seq_group
                    gr.update(visible=prompt_visible), # vid_prompt
                )

            scene_mode.change(
                fn=_scene_mode_visibility,
                inputs=[scene_mode],
                outputs=[autorun_status, push_autorun_row, sequence_group, custom_seq_group, vid_prompt],
            )

            _PUSH_AUTORUN_INPUTS = [
                vid_prompt, duration_seconds, resolution, frame_multiplier,
                export_quality, seed, randomize_seed, add_audio_cb,
                audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                flow_shift_auto, flow_shift,
            ] + [lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes]

            _VID_DOWNLOAD_JS = """
            (videoFile) => {
                if (!videoFile || !videoFile.url) return videoFile;
                const a = document.createElement('a');
                a.href = videoFile.url;
                a.download = videoFile.url.split('/').pop();
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                return videoFile;
            }
            """

            def _dispatch_generate(scene_mode_val, ref_image, prompt,
                                   end_img, dur, res, fmul, qual, sd, rsd,
                                   audio_cb, audio_pt, neg_pt, esteps, eguid,
                                   fsa, fs, *rest_args):
                """Route to normal generate_video, folder-Autorun, Sequence, or Custom Edit Sequence."""
                n = SEQUENCE_MAX_SLOTS
                c = CUSTOM_SEQ_MAX_SLOTS
                seq_imgs = list(rest_args[0:n])
                seq_prompts = list(rest_args[n:2 * n])
                seq_durs = list(rest_args[2 * n:3 * n])
                # Custom edit sequence inputs: c motion prompts, c picgen prompts, c durations
                cs_motions = list(rest_args[3 * n: 3 * n + c])
                cs_picgens = list(rest_args[3 * n + c: 3 * n + 2 * c])
                cs_durs = list(rest_args[3 * n + 2 * c: 3 * n + 3 * c])
                lora_args_inner = rest_args[3 * n + 3 * c:]

                if scene_mode_val == MODE_AUTORUN:
                    yield from autorun_generate(
                        prompt, scene_mode_val, None,
                        dur, res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                elif scene_mode_val == MODE_SEQUENCE:
                    result = generate_sequence(
                        seq_imgs, seq_prompts, seq_durs,
                        res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                    yield result[0], result[1], ""
                elif scene_mode_val == MODE_CUSTOM_SEQ:
                    result = generate_custom_edit_sequence(
                        ref_image, cs_motions, cs_picgens, cs_durs,
                        res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                    yield result[0], result[1], ""
                else:
                    result = generate_video(
                        ref_image, prompt, scene_mode_val,
                        end_img, dur, res, fmul, qual, sd, rsd,
                        audio_cb, audio_pt, neg_pt, esteps, eguid, fsa, fs,
                        *lora_args_inner,
                    )
                    yield result[0], result[1], ""

            generate_btn.click(
                fn=_dispatch_generate,
                inputs=[
                    scene_mode, reference_image, vid_prompt,
                    end_image, duration_seconds, resolution, frame_multiplier,
                    export_quality, seed, randomize_seed, add_audio_cb,
                    audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                    flow_shift_auto, flow_shift,
                ] + sequence_images + sequence_prompts + sequence_durations
                  + custom_seq_motion_prompts + custom_seq_picgen_prompts + custom_seq_durations
                  + [lora_checkboxes[k] for k in AVAILABLE_LORAS if k in lora_checkboxes],
                outputs=[video_output, video_file, autorun_status],
                concurrency_id=WAN_QUEUE_ID,
                concurrency_limit=10,
            ).then(
                # Cache the video's last frame in memory (no disk write)
                # before anything downstream can delete the video file.
                fn=_cache_last_frame_from_video,
                inputs=[video_file],
                outputs=[],
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js=_VID_DOWNLOAD_JS,
            ).then(
                fn=lambda: __import__('time').sleep(2),
                inputs=[],
                outputs=[],
            ).then(
                # Protect the widgets' current first/last frame inputs before
                # the automatic full clear wipes tmp/gradio + outputs.
                fn=protect_current_inputs,
                inputs=[reference_image, end_image] + sequence_images,
                outputs=[],
            ).then(
                fn=clear_storage,
                inputs=[],
                outputs=[clear_storage_status],
            )

            # Push Autorun button wiring
            push_autorun_btn.click(
                fn=autorun_push_generate,
                inputs=_PUSH_AUTORUN_INPUTS,
                outputs=[video_output, video_file, autorun_status],
                concurrency_id=WAN_QUEUE_ID,
                concurrency_limit=1,
            ).then(
                # Cache the video's last frame in memory (no disk write)
                # before anything downstream can delete the video file.
                fn=_cache_last_frame_from_video,
                inputs=[video_file],
                outputs=[],
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js=_VID_DOWNLOAD_JS,
            ).then(
                fn=lambda: __import__('time').sleep(2),
                inputs=[],
                outputs=[],
            ).then(
                # Protect the widgets' current first/last frame inputs before
                # the automatic full clear wipes tmp/gradio + outputs.
                fn=protect_current_inputs,
                inputs=[reference_image, end_image] + sequence_images,
                outputs=[],
            ).then(
                fn=clear_storage,
                inputs=[],
                outputs=[clear_storage_status],
            ).then(
                fn=None,
                inputs=[],
                outputs=[],
                js="""
                () => {
                    fetch('http://127.0.0.1:7861/autorun/ready', {
                        method: 'POST'
                    }).catch(() => {});
                    return [];
                }
                """
            )

            push_cancel_btn.click(
                fn=lambda: (_push_cancel.set() or _set_push_state("idle") or "Cancelled."),
                inputs=[],
                outputs=[autorun_status],
            )

            # ---- Show Media wiring for vidgen ----
            # Pure JS toggle: adds/removes 'hide-media' class on <body> instantly.
            # No server round-trip, no page freeze, containers always present.
            show_media_cb.change(
                fn=None,
                inputs=[show_media_cb],
                outputs=[],
                js="(show) => { document.body.classList.toggle('hide-media', !show); }",
            )

            # Clear storage button in vidgen tab - same function as top right
            clear_storage_btn_vid.click(
                fn=clear_storage,
                inputs=[],
                outputs=[clear_storage_status],
            )

            # Frame extraction functions
            def get_frame_as_file(video_path, timestamp):
                """Extract frame at given seconds, return file path."""
                print(f"\nget_frame_as_file called, ts={timestamp}")
                if not video_path:
                    raise gr.Error("No video available. Generate a video first.")
                if hasattr(video_path, 'name'):
                    path = video_path.name
                elif isinstance(video_path, dict):
                    path = video_path.get('name') or video_path.get('path') or str(video_path)
                else:
                    path = str(video_path)
                if not os.path.exists(path):
                    raise gr.Error(f"Video file not found: {path}")
                ts = float(timestamp) if timestamp else 0.0
                frame_path = extract_frame(path, ts)
                if not frame_path:
                    raise gr.Error("Failed to extract frame from video.")
                print(f" Frame extracted: {frame_path}")
                return frame_path

            # Use Frame as Reference  reads timestamp from number box, puts frame into reference_image
            use_as_reference_btn.click(
                fn=get_frame_as_file,
                inputs=[video_file, frame_time_input],
                outputs=[reference_image],
            )

            # Download Frame  reads timestamp from number box, puts frame into gr.File for download
            download_frame_btn.click(
                fn=get_frame_as_file,
                inputs=[video_file, frame_time_input],
                outputs=[download_file_output],
            )

            # Use as first frame — copies the current end_image into reference_image,
            # replacing whatever was there (or setting it fresh if empty).
            def _copy_end_to_first(end_img):
                """Return the end frame value so it replaces the first frame widget."""
                return end_img

            use_last_as_first_btn.click(
                fn=_copy_end_to_first,
                inputs=[end_image],
                outputs=[reference_image],
            )

            # Last Frame from Video -- pulls the in-memory-only cache of the
            # previously generated video's final frame (see
            # _cache_last_frame_from_video / _use_last_generated_frame
            # above) into the Reference Photo widget. Works even after the
            # auto storage-clear has already deleted the generated video
            # file, because nothing here touches disk.
            last_frame_from_video_btn.click(
                fn=_use_last_generated_frame,
                inputs=[],
                outputs=[reference_image],
            )

            # Track image-input changes so a new upload is protected from
            # clear_storage() and the PREVIOUS file stops being protected
            # only once it is actually gone from the widget - not overwritten
            # by a single shared global, which previously meant uploading a
            # new image while a generation was still running could rip
            # protection away from the file that generation was actively
            # using (making its input "vanish"/error mid-run).
            def _extract_img_path(img):
                if img is None or img == "":
                    return None
                if isinstance(img, str):
                    return img
                for attr in ("name", "filename"):
                    v = getattr(img, attr, None)
                    if isinstance(v, str) and v:
                        return v
                return None

            def _make_image_tracker(is_primary=False):
                """Returns a change-handler closure with its own 'last path' state.

                When the user replaces an input image while a generation is still
                running, the old file must NOT be unprotected — the in-flight job
                is still reading it.  We check _is_generation_active() before
                calling _unprotect_path() so the file stays protected until the
                generation finishes and _generation_release() is called.
                """
                state = {"last": None}

                def _tracker(img):
                    global _current_input_image_path
                    new_path = _extract_img_path(img)
                    old_path = state["last"]
                    if new_path:
                        _protect_path(new_path)
                        _protect_filename(new_path)   # filename-layer too
                    if is_primary:
                        # Keep legacy single-path var pointed at the latest
                        # reference image (still consulted by _is_protected).
                        _current_input_image_path = new_path
                    if old_path and old_path != new_path:
                        # Only unprotect if no running generation still needs it.
                        if not _is_generation_active(old_path):
                            _unprotect_path(old_path)
                            _unprotect_filename(old_path)
                    state["last"] = new_path
                    return None

                return _tracker

            # Use .change() instead of .upload() - fires AFTER upload completes with filepath
            reference_image.change(
                fn=_make_image_tracker(is_primary=True),
                inputs=[reference_image],
                outputs=[],
            )
            end_image.change(
                fn=_make_image_tracker(),
                inputs=[end_image],
                outputs=[],
            )
            for _seq_img_comp in sequence_images:
                _seq_img_comp.change(
                    fn=_make_image_tracker(),
                    inputs=[_seq_img_comp],
                    outputs=[],
                )

        # ------------------------------------------------------------------ #
        #  TAB 2  PHOTO EDITOR (picgen)                                      #
        # ------------------------------------------------------------------ #
        with gr.Tab("Photo Editor"):
            with gr.Column(elem_id="col-container"):

                with gr.Row():
                    start1_btn = gr.Button("Start 1", size="sm")
                    start2_btn = gr.Button("Start 2", size="sm")
                    start3_btn = gr.Button("Start 3", size="sm")
                    start4_btn = gr.Button("Start 4", size="sm")

                with gr.Row():
                    # Left column: input uploader + controls
                    with gr.Column(scale=1):
                        hidden_images_b64 = gr.Textbox(
                            value="[]", elem_id="hidden-images-b64",
                            elem_classes="hidden-input", container=False, visible=False,
                        )
                        gr.HTML("""
                        <div class="uploader-toolbar">
                            <button id="tb-upload" class="tb-btn">Upload</button>
                            <button id="tb-remove" class="tb-btn">Remove Selected</button>
                            <button id="tb-clear" class="tb-btn">Clear All</button>
                        </div>
                        <div id="gallery-drop-zone">
                            <div id="upload-prompt" class="upload-prompt-modern">
                                <div id="upload-click-area" class="upload-click-area">
                                    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <rect x="6" y="10" width="52" height="44" rx="5" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"/>
                                        <polygon points="10,50 24,32 34,42 44,28 54,50" fill="rgba(128,128,128,0.15)" stroke="currentColor" stroke-width="1.5"/>
                                        <circle cx="22" cy="24" r="5" fill="rgba(128,128,128,0.2)" stroke="currentColor" stroke-width="1.5"/>
                                    </svg>
                                    <span class="upload-main-text">Click or drag images here</span>
                                    <span class="upload-sub-text">Supports multiple images</span>
                                </div>
                            </div>
                            <input id="custom-file-input" type="file" accept="image/*" multiple style="display:none;" />
                            <div id="image-gallery-grid" class="image-gallery-grid" style="display:none;"></div>
                        </div>
                        """)
                        pic_num_images = gr.Slider(
                            label="Number of images", minimum=1, maximum=4, step=1, value=1,
                        )
                        pic_prompt = gr.Textbox(
                            label="Prompt", show_label=True,
                            placeholder="describe the edit instruction",
                            lines=3, max_lines=20,
                        )
                        pic_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="censored, mosaic, blurred, clothed, soft, partial",
                            value="", lines=2, max_lines=10,
                        )

                    # Right column: result gallery + dropdowns + generate button
                    with gr.Column(scale=1):
                        pic_result = gr.Gallery(
                            label="Result",
                            show_label=False,
                            type="filepath",
                            interactive=False,
                            columns=2,
                            elem_id="picgen-result-gallery",
                        )
                        use_output_btn = gr.Button("Use as input", variant="secondary", size="sm")

                        # Row 1: 4 dropdowns
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Solo", choices=list(solo_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown2 = gr.Dropdown(
                                label="Couple (Unseen)", choices=list(couple_man_unseen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown3 = gr.Dropdown(
                                label="Couple (Seen)", choices=list(couple_man_seen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown4 = gr.Dropdown(
                                label="Multi Women", choices=list(multiple_women_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )

                        # Row 2: 3 dropdowns centred
                        with gr.Row():
                            preset_dropdown5 = gr.Dropdown(
                                label="Multi (Unseen)", choices=list(multiple_man_unseen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown6 = gr.Dropdown(
                                label="Multi (Seen)", choices=list(multiple_man_seen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown7 = gr.Dropdown(
                                label="Multi-Step", choices=list(multistep_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )

                        pic_run_button = gr.Button("Generate", variant="primary", size="lg")

                with gr.Accordion("Advanced Settings", open=False):
                    pic_seed = gr.Slider(label="Seed", minimum=0, maximum=PICGEN_MAX_SEED, step=1, value=0)
                    pic_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        pic_guidance = gr.Slider(label="True guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                        pic_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=40, step=1, value=4)
                        pic_height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=None)
                        pic_width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=None)
                    fullscreen_toggle = gr.Checkbox(label="Full Screen Mode", value=False, info="Expand image boxes to full page width")
                    keyboard_toggle = gr.Checkbox(label="Disable On-Screen Keyboard", value=False, info="Prevent keyboard from appearing on mobile devices")

                # Preset dropdown handlers  selecting updates prompt box
                preset_dropdown.change(fn=update_solo_prompt, inputs=[preset_dropdown], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown2.change(fn=update_couple_man_unseen_prompt, inputs=[preset_dropdown2], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown3.change(fn=update_couple_man_seen_prompt, inputs=[preset_dropdown3], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown4.change(fn=update_multiple_women_prompt, inputs=[preset_dropdown4], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown5.change(fn=update_multiple_man_unseen_prompt, inputs=[preset_dropdown5], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown6.change(fn=update_multiple_man_seen_prompt, inputs=[preset_dropdown6], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown7.change(fn=update_multistep_prompt, inputs=[preset_dropdown7], outputs=[pic_prompt], scroll_to_output=False)

                _pic_infer_inputs = [
                    hidden_images_b64, pic_prompt, pic_negative_prompt,
                    pic_seed, pic_randomize_seed, pic_guidance, pic_steps,
                    pic_height, pic_width, pic_num_images,
                ]
                _pic_infer_js = "(...args) => { const imgs = window.__uploadedImages || []; const b64 = JSON.stringify(imgs.map(i => i.b64)); args[0] = b64; return args; }"

                gr.on(
                    triggers=[pic_run_button.click, pic_prompt.submit],
                    fn=infer_with_preclear,
                    inputs=_pic_infer_inputs,
                    js=_pic_infer_js,
                    outputs=[pic_result, pic_seed],
                    concurrency_id=PIC_QUEUE_ID,
                    concurrency_limit=10,
                ).then(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="""
                    () => {
                        // Poll up to 3s for gallery images to finish rendering,
                        // then download each one.
                        function downloadGalleryImages() {
                            const gallery = document.querySelector('.gradio-gallery');
                            const imgs = gallery ? gallery.querySelectorAll('img') : [];
                            const urls = [];
                            imgs.forEach(img => {
                                if (img.src && !img.src.startsWith('data:') &&
                                    (img.src.includes('/file=') || img.src.includes('/gradio/'))) {
                                    urls.push(img.src);
                                }
                            });
                            if (urls.length > 0) {
                                urls.forEach((url, i) => {
                                    setTimeout(() => {
                                        const a = document.createElement('a');
                                        a.href = url;
                                        a.download = url.split('/').pop().split('?')[0] || ('picgen_' + i + '.png');
                                        document.body.appendChild(a);
                                        a.click();
                                        document.body.removeChild(a);
                                    }, i * 300);
                                });
                            } else {
                                // Images not rendered yet â€” retry
                                setTimeout(downloadGalleryImages, 300);
                            }
                        }
                        setTimeout(downloadGalleryImages, 400);
                        return [];
                    }
                    """
                ).then(
                    fn=lambda: __import__('time').sleep(2),
                    inputs=[],
                    outputs=[],
                ).then(
                    # Picgen inputs are in-memory b64, but this automatic
                    # clear would still wipe the vidgen tab's current
                    # first/last frame files - protect them too.
                    fn=protect_current_inputs,
                    inputs=[reference_image, end_image] + sequence_images,
                    outputs=[],
                ).then(
                    fn=clear_storage,
                    inputs=[],
                    outputs=[clear_storage_status],
                )

                # (Show Media is handled globally via the CSS hide-media class on <body>)

                def output_to_b64(output_images):
                    if not output_images:
                        return "[]"
                    b64_list = []
                    for item in output_images:
                        try:
                            img = item[0] if isinstance(item, (list, tuple)) else item
                            # Gallery entries may arrive as PIL images, as file
                            # paths, or as dicts carrying a path.
                            if isinstance(img, dict):
                                img = img.get("path") or img.get("name")
                            if isinstance(img, str):
                                img = Image.open(img)
                            img = img.convert("RGB")
                            
                            # Resize to 512px (pipeline resizes to this for VAE anyway)
                            max_size = 512
                            if img.width > img.height:
                                new_height = int((img.height * max_size) / img.width)
                                img = img.resize((max_size, new_height), Image.LANCZOS)
                            else:
                                new_width = int((img.width * max_size) / img.height)
                                img = img.resize((new_width, max_size), Image.LANCZOS)
                            
                            buf = BytesIO()
                            # Use JPEG with quality 95 for better compression
                            img.save(buf, format="JPEG", quality=95)
                            b64_list.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
                        except Exception as e:
                            print(f"output_to_b64 skipped an item: {e}")
                            continue
                    return json.dumps(b64_list)

                use_output_btn.click(fn=output_to_b64, inputs=[pic_result], outputs=[hidden_images_b64])
                
                # Add JavaScript handler to populate gallery when hidden_images_b64 changes
                hidden_images_b64.change(
                    fn=None, inputs=[hidden_images_b64], outputs=None,
                    js="""(b64List) => {
                        if (!b64List || !window.__addImage) return;
                        try {
                            const images = JSON.parse(b64List);
                            if (Array.isArray(images)) {
                                // Clear existing images first
                                if (window.__clearAll) window.__clearAll();
                                // Add each image from the result gallery
                                images.forEach((b64, idx) => {
                                    window.__addImage(b64, `output_${idx + 1}.png`);
                                });
                            }
                        } catch (e) {
                            console.error('Failed to parse output images:', e);
                        }
                    }""",
                )

                starter_b64_output = gr.Textbox(value="", visible=False, elem_id="starter-b64-output")

                start1_btn.click(fn=lambda: add_starter_image(1), inputs=[], outputs=[starter_b64_output])
                start2_btn.click(fn=lambda: add_starter_image(2), inputs=[], outputs=[starter_b64_output])
                start3_btn.click(fn=lambda: add_starter_image(3), inputs=[], outputs=[starter_b64_output])
                start4_btn.click(fn=lambda: add_starter_image(4), inputs=[], outputs=[starter_b64_output])

                starter_b64_output.change(
                    fn=None, inputs=[starter_b64_output], outputs=None,
                    js="(b64) => { if (b64 && window.__addImage) window.__addImage(b64, 'starter.jpg'); }",
                )

                fullscreen_toggle.change(
                    fn=None, inputs=[fullscreen_toggle], outputs=None,
                    js="""(fullscreen) => {
                        const style = document.getElementById('fullscreen-style') || document.createElement('style');
                        style.id = 'fullscreen-style';
                        style.textContent = fullscreen ? `.gradio-container{max-width:100vw!important;padding:0!important;}.contain,.wrap,#component-0{max-width:100%!important;}` : '';
                        if (!document.getElementById('fullscreen-style')) document.head.appendChild(style);
                    }""",
                )

                keyboard_toggle.change(
                    fn=None, inputs=[keyboard_toggle], outputs=None,
                    js="""(disable) => {
                        setTimeout(() => {
                            const style = document.getElementById('keyboard-style') || document.createElement('style');
                            style.id = 'keyboard-style';
                            if (disable) {
                                style.textContent = `input[type="text"]:not([role="combobox"]),input[type="number"],textarea:not([role="combobox"]){-webkit-user-select:none!important;user-select:none!important;pointer-events:none!important;}`;
                                document.querySelectorAll('input[type="text"]:not([role="combobox"]),input[type="number"],textarea:not([role="combobox"])').forEach(el=>{el.setAttribute('readonly','readonly');el.setAttribute('inputmode','none');});
                            } else {
                                style.textContent = '';
                                document.querySelectorAll('input[type="text"],input[type="number"],textarea').forEach(el=>{el.removeAttribute('readonly');el.removeAttribute('inputmode');});
                            }
                            if (!document.getElementById('keyboard-style')) document.head.appendChild(style);
                        }, 100);
                    }""",
                )

    demo.load(fn=None, js=gallery_js)

    # Apply hide-media on page load (Show Media checkbox defaults to unchecked)
    demo.load(fn=None, js="() => { document.body.classList.add('hide-media'); }")

    # Auto-sync video currentTime to the frame_time_input number box
    video_time_sync_js = """
() => {
    function startVideoSync() {
        const video = document.querySelector('#generated-video video');
        if (!video) { setTimeout(startVideoSync, 500); return; }
        // Update the number input every 200ms while video is playing
        setInterval(() => {
            const input = document.querySelector('#frame-time-input input');
            // Only update if input exists, video time is valid, and user is NOT actively editing the field
            if (input && !isNaN(video.currentTime) && document.activeElement !== input) {
                const nativeInput = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (nativeInput && nativeInput.set) {
                    nativeInput.set.call(input, video.currentTime.toFixed(2));
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }, 200);
        // Re-run if a new video loads (MutationObserver on the video src)
        const observer = new MutationObserver(() => {
            const newVideo = document.querySelector('#generated-video video');
            if (newVideo && newVideo !== video) { observer.disconnect(); startVideoSync(); }
        });
        observer.observe(document.querySelector('#generated-video') || document.body, { childList: true, subtree: true });
    }
    setTimeout(startVideoSync, 1000);
}
"""
    demo.load(fn=None, js=video_time_sync_js)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _start_push_api()
    os.makedirs(os.path.join(SCRIPT_DIR, "tmp"), exist_ok=True)

    if DUAL_GPU:
        print(f" GRADIO LAUNCHING  Wan on {WAN_DEVICE}, Qwen on {PIC_DEVICE}. Both tabs ready.")
        demo.queue(default_concurrency_limit=10)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, str(OUTPUT_DIR)],
        )
    else:
        if STARTUP_MODE == "vidgen":
            print(" GRADIO LAUNCHING  Wan on GPU, vidgen ready immediately.")
        else:
            print(" GRADIO LAUNCHING  Qwen on GPU, picgen ready immediately.")
        demo.queue(default_concurrency_limit=1)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, str(OUTPUT_DIR)],
        )

        # Background: load the secondary model to CPU after Gradio is up
        def _bg_load():
            try:
                time.sleep(2.0)
                if STARTUP_MODE == "vidgen":
                    print("Background: Loading Qwen to CPU...")
                    global pic_pipe
                    start = time.time()
                    model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
                    if not os.path.exists(model_index_path):
                        os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
                        pipe = QwenImageEditPlusPipeline.from_pretrained(
                            "Qwen/Qwen-Image-Edit-2511",
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                            cache_dir=BASE_MODEL_LOCAL_PATH, use_safetensors=True
                        )
                    else:
                        pipe = QwenImageEditPlusPipeline.from_pretrained(
                            BASE_MODEL_LOCAL_PATH,
                            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                            local_files_only=True, use_safetensors=True
                        )
                    v23 = NSFW_WEIGHTS_LOCAL_PATH
                    if not os.path.exists(v23):
                        v23 = hf_hub_download(
                            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
                            filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
                            cache_dir=PICGEN_MODELS_DIR,
                            local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
                        )
                    sd = load_file(v23)
                    tw, vw, ew = {}, {}, {}
                    for k, v in sd.items():
                        if k.startswith("model.diffusion_model."):   tw[k.replace("model.diffusion_model.", "")] = v
                        elif k.startswith("transformer."):           tw[k.replace("transformer.", "")] = v
                        elif k.startswith("first_stage_model."):     vw[k.replace("first_stage_model.", "")] = v
                        elif k.startswith("vae."):                   vw[k.replace("vae.", "")] = v
                        elif "text_encoder" in k or "conditioner" in k:
                            if "conditioner.embedders.0." in k:      ew[k.replace("conditioner.embedders.0.", "")] = v
                            elif "text_encoder." in k:               ew[k.replace("text_encoder.", "")] = v
                    if tw: pipe.transformer.load_state_dict(tw, strict=False)
                    if vw: pipe.vae.load_state_dict(vw, strict=False)
                    if ew: pipe.text_encoder.load_state_dict(ew, strict=False)
                    del sd, tw, vw, ew
                    torch.cuda.empty_cache()
                    pipe.vae.enable_tiling()
                    pipe.vae.enable_slicing()
                    pipe.transformer.to("cpu")
                    pipe.text_encoder.to("cpu")
                    pipe.vae.to("cpu")
                    pic_pipe = pipe
                    print(f" Qwen on CPU in {time.time()-start:.1f}s  tab switching ready!")
                else:
                    print("Background: Loading Wan to CPU...")
                    _load_wan("cpu")
                    print(" Wan on CPU  tab switching ready!")
            except Exception as e:
                print(f" Background load failed: {e}")
                import traceback; traceback.print_exc()

        threading.Thread(target=_bg_load, daemon=True).start()






