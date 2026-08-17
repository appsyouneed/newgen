import os
import shutil
import subprocess
import sys
import random
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
# -vidgen (default) — Video Generator tab is shown first and, on a single-GPU
#   box, Wan loads to GPU at startup (Qwen stays on CPU until first use).
# -picgen — Photo Editor tab is shown first and, on a single-GPU box,
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
torch.set_float32_matmul_precision('highest')
# Do NOT enable cudnn.benchmark — Blackwell GPUs (GB202) fail on conv3d engine search
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Do not fix thread counts — multi-GPU concurrent inference needs flexible threading

# Suppress noisy library warnings after torch is imported
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress torchao import warning from diffusers
import io as _io
import contextlib as _ctx

from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from PIL import Image
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# SAGEATTENTION — 2-3x faster attention kernels (optional, auto-detected)
# ---------------------------------------------------------------------------
_SAGE_ATTENTION_AVAILABLE = False
try:
    from sageattention import sageattn
    _SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    pass

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
    print(f"🎞️  [extract_frame] Starting extraction...")
    print(f"   - video_path: {video_path}")
    print(f"   - timestamp: {timestamp}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [extract_frame] Failed to open video file")
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
        print(f"✅ [extract_frame] Frame read successfully")
        print(f"   - Frame shape: {frame.shape}")
        
        # Save frame as high-quality JPG file
        output_path = unique_output_path("extracted_frame", "jpg")
        print(f"   - Saving to: {output_path}")
        
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"✅ [extract_frame] Frame saved successfully\n")
        return str(output_path)
    else:
        print(f"❌ [extract_frame] Failed to read frame at position {frame_number}\n")
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
# HARDWARE DETECTION & EXECUTION MODE SELECTION
#
# Automatically detects GPU configuration and selects optimal execution mode:
#   CONCURRENT  — Both models GPU-resident, no swapping (≥80GB total VRAM)
#   STACKED     — Multi-GPU, models swap but leverage all GPUs (multi-GPU, <80GB)
#   SINGLE      — One GPU, standard CPU offload + swap
#
# Override: NEWGEN_FORCE_SINGLE_GPU=1 forces single-GPU swap mode on any box.
# ---------------------------------------------------------------------------

_gpu_count = torch.cuda.device_count()
if _gpu_count < 1:
    raise RuntimeError("No CUDA device visible — this app requires a GPU.")

# Gather GPU info
_gpu_info = []
_total_vram_mb = 0
for i in range(_gpu_count):
    props = torch.cuda.get_device_properties(i)
    vram_mb = props.total_memory // (1024 * 1024)
    _gpu_info.append({
        "index": i,
        "name": props.name,
        "vram_mb": vram_mb,
        "compute": f"{props.major}.{props.minor}",
    })
    _total_vram_mb += vram_mb

# Minimum VRAM to hold both models simultaneously (~45GB WAN + ~35GB Qwen)
_CONCURRENT_THRESHOLD_MB = 80 * 1024  # 80 GB

# Minimum per-GPU VRAM to fit a full model without splitting
_PER_GPU_FULL_MODEL_MB = 48 * 1024  # 48 GB (WAN needs ~45GB active)

# Determine execution mode
_force_single = os.environ.get("NEWGEN_FORCE_SINGLE_GPU") == "1"
_max_single_gpu_vram = max(g["vram_mb"] for g in _gpu_info)

if _force_single or _gpu_count == 1:
    GPU_MODE = "single"
elif _gpu_count >= 2 and _max_single_gpu_vram >= _PER_GPU_FULL_MODEL_MB:
    # Each GPU can hold a full model on its own — pin one model per GPU
    GPU_MODE = "concurrent"
else:
    # Multi-GPU but individual cards too small for a full model — split via accelerate
    GPU_MODE = "stacked"

# Device assignments based on mode
if GPU_MODE == "concurrent":
    PIC_DEVICE = "cuda:0"
    WAN_DEVICE = "cuda:1" if _gpu_count >= 2 else "cuda:0"
    PIC_QUEUE_ID = "pic-gpu"
    WAN_QUEUE_ID = "wan-gpu"
elif GPU_MODE == "stacked":
    # Stacked: models split across all GPUs via accelerate balanced device_map
    PIC_DEVICE = "cuda:0"
    WAN_DEVICE = "cuda:0"  # Balanced mode handles device placement internally
    PIC_QUEUE_ID = "pic-gpu"
    WAN_QUEUE_ID = "wan-gpu"
else:  # single
    PIC_DEVICE = "cuda:0"
    WAN_DEVICE = "cuda:0"
    PIC_QUEUE_ID = "gpu"
    WAN_QUEUE_ID = "gpu"

# Legacy compatibility
DUAL_GPU = (GPU_MODE == "concurrent")

# Auxiliary models (RIFE interpolation, MMAudio) always on cuda:0
device = torch.device("cuda:0")

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
        # float32 to match flownet — see the note at model load.
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
# INFERENCE ACCELERATION UTILITIES
#
# These are applied to pipelines after loading to maximize generation speed.
# All optimizations are optional and degrade gracefully if unavailable.
# ---------------------------------------------------------------------------

_TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')
_TEACACHE_ENABLED = False  # Set True after pipeline load if successful
_SAGE_PATCHED = False  # Only patch once globally


def apply_sage_attention(pipe):
    """Replace SDPA attention with SageAttention for 2-3x kernel speedup.
    
    WARNING: This patches F.scaled_dot_product_attention GLOBALLY.
    Only call this if you're sure it won't break other models (e.g. Qwen).
    Currently DISABLED to prevent black images in picgen.
    """
    # DISABLED: Global SDPA patching breaks Qwen's text encoder, causing all-black images.
    # SageAttention only works safely with Wan's transformer architecture.
    # Until per-model attention patching is implemented, this stays disabled.
    return False


def apply_torch_compile(pipe, mode="reduce-overhead"):
    """Apply torch.compile to transformer blocks for kernel fusion speedup."""
    # DISABLED: torch.compile causes infinite recompilation on Wan's MoE architecture
    # (dual transformer switching mid-inference creates dynamic control flow that
    # triggers repeated retracing). Re-enable only for single-transformer models.
    return False


def apply_teacache(pipe, threshold=0.05):
    """Enable TeaCache for video diffusion — training-free timestep caching."""
    global _TEACACHE_ENABLED
    try:
        if hasattr(pipe, 'enable_teacache'):
            # Diffusers built-in TeaCache support
            pipe.enable_teacache(
                cache_interval=2,
                rel_l1_thresh=threshold,
            )
            _TEACACHE_ENABLED = True
            return True
    except Exception as e:
        print(f"  TeaCache enable failed: {e}")
    return False


def apply_all_optimizations(pipe, pipe_name="model", enable_compile=True, enable_teacache=False, teacache_thresh=0.05):
    """Apply all available inference accelerations to a pipeline."""
    results = {}
    
    # SageAttention
    sage_ok = apply_sage_attention(pipe)
    results["SageAttention"] = sage_ok
    
    # TeaCache (video models only)
    if enable_teacache:
        tea_ok = apply_teacache(pipe, threshold=teacache_thresh)
        results["TeaCache"] = tea_ok
    
    # torch.compile (applied last so it captures the optimized graph)
    if enable_compile:
        compile_ok = apply_torch_compile(pipe)
        results["torch.compile"] = compile_ok
    
    applied = [k for k, v in results.items() if v]
    skipped = [k for k, v in results.items() if not v]
    if applied:
        print(f"  ✅ {pipe_name}: {', '.join(applied)}")
    if skipped:
        print(f"  ⏭️  {pipe_name} skipped: {', '.join(skipped)}")
    return results


# ---------------------------------------------------------------------------
# WAN 2.2 I2V A14B — MERGED 4-STEP DISTILL (BF16, no LoRA)
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
# WAN 2.2 I2V MODEL — WAMU v2 LIGHTNING (NSFW)
#
# WAMU v2 is a Wan 2.2 I2V Lightning merge trained on explicit content.
# This is a complete diffusers pipeline (transformer + transformer_2 + vae +
# text_encoder + scheduler) with 4-step distillation already merged.
# No LoRAs. No separate expert loading.
# ---------------------------------------------------------------------------

WAN_MODEL_REPO = "TestOrganizationPleaseIgnore/WAMU_v2_WAN2.2_I2V_LIGHTNING"
print(f"Video model: WAMU v2 — Wan 2.2 I2V Lightning merge (NSFW-capable)")

# 16 fps is what Wan-AI's own diffusers example for I2V-A14B exports at.
# (The 24 fps figure in the Wan 2.2 docs refers to the TI2V-5B model.)
FIXED_FPS = 16

# WAMU v2 is distillation-merged, so guidance must stay at 1.0.
WAN_STEPS = 4
WAN_FLOW_SHIFT = 6.9
WAN_GUIDANCE = 1.0

MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 97       # 6s per segment (97 frames ÷ 16 fps = 6.06s) — keeps quality high
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
        print(f"Could not apply flow_shift={flow_shift} ({e}) — keeping default.")


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
        print(f"Loaded {repo} from local cache.")
        return pipe
    except Exception:
        print(f"Fetching {repo}...")
        return WanImageToVideoPipeline.from_pretrained(repo, **kwargs)


def _build_wan_pipeline(target_device="cpu"):
    global wan_pipe, _wan_loaded, _wan_scheduler_config

    if target_device == "cpu":
        # CPU load — standard path for single-GPU swap mode
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
        print(f"🎯 WAMU v2 loaded to CPU (ready for fast swapping)")
    elif target_device == "balanced":
        # STACKED MODE: Split model across all available GPUs using accelerate
        # This distributes transformer layers evenly across GPUs so the full model
        # is GPU-resident without needing any single card to hold it all.
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            use_safetensors=True
        )
        print(f"🎯 WAMU v2 loaded BALANCED across {_gpu_count} GPUs — pipeline parallelism active!")
    else:
        # Load to CPU then move to target GPU.
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=None,
            use_safetensors=True
        )
        pipeline = pipeline.to(target_device)
        torch.cuda.synchronize(target_device)
        print(f"🎯 WAMU v2 loaded directly to {target_device} - Ready for video generation!")

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
        "## 🎬 WAMU v2 — Wan 2.2 I2V Lightning (NSFW)\n"
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
# Keep scene      — no image editing at all. The photo goes straight to Wan, so
#                   background, bodies and framing stay pixel-identical.
# Replace scene   — Qwen repaints the environment around the subjects first.
# Custom edit     — Qwen applies your edit instruction verbatim, no template.
# ---------------------------------------------------------------------------

MODE_KEEP = "Keep original scene"
MODE_REPLACE = "Replace background / environment"
MODE_CUSTOM = "Custom edit instruction"
SCENE_MODES = [MODE_KEEP, MODE_REPLACE, MODE_CUSTOM]

RELOCATE_INSTRUCTION = (
    "Keep the people exactly as they are — identical faces, facial features, "
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
    edit_instruction: str,
    seed: int,
    steps: int,
    guidance: float,
) -> Image.Image:
    """
    Optional stage 1 — Qwen Image Edit prepares the starting frame.

    Returns the image untouched in keep-scene mode, so nothing is repainted and
    the original background and bodies survive exactly as shot.
    """
    if mode == MODE_KEEP:
        print("[1/2] Keep-scene mode — reference frame used as-is.")
        return image

    if mode == MODE_CUSTOM:
        instruction = (edit_instruction or "").strip()
        if not instruction:
            raise gr.Error(
                "Custom edit mode needs an edit instruction. Switch to "
                f"'{MODE_KEEP}' if you do not want the frame changed."
            )
    else:
        instruction = RELOCATE_INSTRUCTION.format(prompt=prompt)

    activate_pic()
    print(f"[1/2] Qwen editing frame -> {instruction[:80]}...")

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
):
    """Stage 2 — animate a frame with WAMU v2 merged model."""
    activate_wan()

    # Pin to WAN_DEVICE — prevents cross-device leakage in dual GPU mode
    with torch.cuda.device(WAN_DEVICE):
        # Apply flow shift if provided, otherwise use WAMU v2 default (6.9)
        _set_flow_shift(wan_pipe, flow_shift if flow_shift is not None else WAN_FLOW_SHIFT)

        print(f"[2/2] Wan animating {num_frames} frames at {frame.size}...")
        print(f"Prompt: {prompt!r} | Seed: {seed} | Steps: {WAN_STEPS}")

        kwargs = dict(
            image=frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=frame.height,
            width=frame.width,
            num_frames=num_frames,
            num_inference_steps=WAN_STEPS,
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
            print(f"End frame not supported by this pipeline ({e}) — ignoring it.")
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


def generate_with_preset(prompt_dict, choice, reference_image, scene_mode, edit_instruction,
                        end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                        flow_shift_auto, flow_shift):
    """Generate video using preset prompt without changing the prompt textbox."""
    if choice and choice in prompt_dict:
        preset_prompt = prompt_dict[choice]
        return generate_video(
            reference_image, preset_prompt, scene_mode, edit_instruction,
            end_image, duration_seconds, resolution, frame_multiplier,
            export_quality, seed, randomize_seed, add_audio_cb,
            audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
            flow_shift_auto, flow_shift
        )
    return None, None

# Wrapper functions for each preset dropdown
def generate_with_solo(choice, reference_image, scene_mode, edit_instruction,
                      end_image, duration_seconds, resolution, frame_multiplier,
                      export_quality, seed, randomize_seed, add_audio_cb,
                      audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                      flow_shift_auto, flow_shift):
    return generate_with_preset(vid_solo_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_couple(choice, reference_image, scene_mode, edit_instruction,
                        end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                        flow_shift_auto, flow_shift):
    return generate_with_preset(vid_couple_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multiple(choice, reference_image, scene_mode, edit_instruction,
                          end_image, duration_seconds, resolution, frame_multiplier,
                          export_quality, seed, randomize_seed, add_audio_cb,
                          audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                          flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multistep(choice, reference_image, scene_mode, edit_instruction,
                           end_image, duration_seconds, resolution, frame_multiplier,
                           export_quality, seed, randomize_seed, add_audio_cb,
                           audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                           flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multistep_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_environment(choice, reference_image, scene_mode, edit_instruction,
                             end_image, duration_seconds, resolution, frame_multiplier,
                             export_quality, seed, randomize_seed, add_audio_cb,
                             audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                             flow_shift_auto, flow_shift):
    return generate_with_preset(vid_environment_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_custom(choice, reference_image, scene_mode, edit_instruction,
                        end_image, duration_seconds, resolution, frame_multiplier,
                        export_quality, seed, randomize_seed, add_audio_cb,
                        audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                        flow_shift_auto, flow_shift):
    return generate_with_preset(vid_custom_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multiple_unseen(choice, reference_image, scene_mode, edit_instruction,
                                 end_image, duration_seconds, resolution, frame_multiplier,
                                 export_quality, seed, randomize_seed, add_audio_cb,
                                 audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                                 flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_man_unseen_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)

def generate_with_multiple_seen(choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift):
    return generate_with_preset(vid_multiple_man_seen_prompts_dict, choice, reference_image, scene_mode, edit_instruction,
                               end_image, duration_seconds, resolution, frame_multiplier,
                               export_quality, seed, randomize_seed, add_audio_cb,
                               audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                               flow_shift_auto, flow_shift)


def generate_video(
    reference_image,
    prompt,
    scene_mode,
    edit_instruction="",
    end_image=None,
    duration_seconds=3.5,
    resolution="480p",
    frame_multiplier=16,
    export_quality=7,
    seed=42,
    randomize_seed=True,
    add_audio_cb=False,
    audio_prompt_tb="natural ambient sound",
    negative_prompt=None,
    edit_steps=4,
    edit_guidance=1.0,
    flow_shift_auto=True,
    flow_shift=None,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Photo(s) + prompt -> video.

    Stage 1 (optional) prepares the starting frame with Qwen Image Edit.
    Stage 2 animates it with WAMU v2 4-step Lightning (NSFW merge), chaining
    ~5.1s segments for anything longer than one native window.
    
    No interpolation by default (frame_multiplier=16 = native 16fps).
    """
    # Gradio can hand back "" instead of None for an untouched optional image
    # component — normalize both reference_image and end_image so a stray
    # empty string never reaches PIL-only code (that's what caused
    # `'str' object has no attribute 'size'` on end_image).
    reference_image = _ensure_pil(reference_image)
    end_image = _ensure_pil(end_image)

    if reference_image is None:
        raise gr.Error("Please upload a reference photo.")
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the motion and scene.")
    
    # 🎯 ADAPTIVE FLOW SHIFT: Auto-adjust based on duration for better prompt following
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
        
        print(f"🎯 Auto flow_shift: {adaptive_flow_shift:.1f} (duration: {duration_seconds}s)")
        flow_shift = adaptive_flow_shift
    else:
        # Manual mode: use user-specified flow_shift
        if flow_shift is None:
            flow_shift = WAN_FLOW_SHIFT
        print(f"🎯 Manual flow_shift: {flow_shift} (user override)")

    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    segment_paths = []

    try:
        sized = resize_image_for_wan(reference_image, resolution)
        print(f"Resized image to {sized.size} for VAE compatibility")

        # ---- Stage 1: optional frame preparation --------------------------
        start_frame = edit_reference_frame(
            sized, scene_mode, prompt, edit_instruction,
            current_seed, edit_steps, edit_guidance,
        )

        # End-frame conditioning applies to the first segment only.
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

            seg_end = processed_end if seg_index == 1 else None
            raw_frames = animate_frame(
                current_frame, seg_end, prompt, negative_prompt,
                num_frames, seg_seed, flow_shift,
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
                print("Could not read segment tail frame — stopping chain here.")
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
                    final_path, audio_prompt, float(duration_seconds)
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
            print(f"Could not rename output ({e}) — serving original path.")

        print(f"Done in {time.time() - started:.1f}s — {seg_index} segment(s), "
              f"seed {current_seed} -> {os.path.basename(final_path)}")
        return final_path, final_path

    except gr.Error:
        raise
    except Exception as e:
        for p in segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        print(f"Generation error: {e}")
        raise gr.Error(f"Generation failed: {e}")


# ---------------------------------------------------------------------------
# PICGEN MODEL (Qwen Image Edit)
# ---------------------------------------------------------------------------

PICGEN_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BASE_MODEL_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")

# 🚀 PRIMARY MODEL LOADING
if GPU_MODE == "stacked":
    # STACKED MODE: Split both models across all GPUs using accelerate pipeline parallelism.
    # Both models stay GPU-resident (split across cards) — zero swap latency.
    print(f"🚀 STACKED MODE: Loading models balanced across {_gpu_count} GPUs...")
    start_stacked = time.time()
    
    # Load WAN with balanced device_map (splits transformer layers across GPUs)
    print("  Loading WAN balanced across GPUs...")
    _load_wan("balanced")
    
    # Load Qwen with balanced device_map
    print("  Loading Qwen balanced across GPUs...")
    model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
    if not os.path.exists(model_index_path):
        print(f"  Downloading Qwen base model to {BASE_MODEL_LOCAL_PATH}...")
        os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2511",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            use_safetensors=True
        )
    else:
        pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            use_safetensors=True
        )
    # Load NSFW weights
    if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
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
    tw, vw, ew = {}, {}, {}
    for k, v in state_dict.items():
        if k.startswith("model.diffusion_model."):   tw[k.replace("model.diffusion_model.", "")] = v
        elif k.startswith("transformer."):           tw[k.replace("transformer.", "")] = v
        elif k.startswith("first_stage_model."):     vw[k.replace("first_stage_model.", "")] = v
        elif k.startswith("vae."):                   vw[k.replace("vae.", "")] = v
        elif "text_encoder" in k or "conditioner" in k:
            if "conditioner.embedders.0." in k:      ew[k.replace("conditioner.embedders.0.", "")] = v
            elif "text_encoder." in k:               ew[k.replace("text_encoder.", "")] = v
    if tw: pic_pipe.transformer.load_state_dict(tw, strict=False)
    if vw: pic_pipe.vae.load_state_dict(vw, strict=False)
    if ew: pic_pipe.text_encoder.load_state_dict(ew, strict=False)
    del state_dict, tw, vw, ew
    pic_pipe.vae.enable_tiling()
    pic_pipe.vae.enable_slicing()
    print(f"  ✅ Qwen loaded balanced across GPUs")
    
    _active_model = "both"
    stacked_time = time.time() - start_stacked
    print(f"✅ STACKED MODE READY in {stacked_time:.1f}s — both models split across {_gpu_count} GPUs, no swap needed!")
    
    # Apply optimizations
    print("⚡ Applying inference optimizations...")
    apply_all_optimizations(wan_pipe, "WAN (vidgen)", enable_compile=True, enable_teacache=False)

elif DUAL_GPU:
    # DUAL GPU: Load both models to their dedicated GPUs simultaneously at startup
    print(f"🚀 DUAL GPU: Loading Wan → {WAN_DEVICE} and Qwen → {PIC_DEVICE} simultaneously...")
    
    def _load_wan_thread():
        global wan_pipe_primary
        t = time.time()
        torch.cuda.set_device(WAN_DEVICE)
        wan_pipe_primary = _load_wan(WAN_DEVICE)
        print(f"✅ WAN ready on {WAN_DEVICE} in {time.time()-t:.1f}s")
    
    def _load_qwen_thread():
        global pic_pipe
        t = time.time()
        torch.cuda.set_device(PIC_DEVICE)
        print("🚀 Loading Qwen Image Edit pipeline...")
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
        print(f"✅ Qwen ready on {PIC_DEVICE} in {time.time()-t:.1f}s")

    t_wan = threading.Thread(target=_load_wan_thread, daemon=False)
    t_qwen = threading.Thread(target=_load_qwen_thread, daemon=False)
    t_wan.start()
    t_qwen.start()
    t_wan.join()
    t_qwen.join()
    _active_model = "both"
    print(f"✅ DUAL GPU READY — Vidgen on {WAN_DEVICE}, Picgen on {PIC_DEVICE}")

    # Apply inference optimizations to both pipelines
    print("⚡ Applying inference optimizations...")
    _wan_opt = apply_all_optimizations(wan_pipe, "WAN (vidgen)", enable_compile=True, enable_teacache=False, teacache_thresh=0.05)

elif STARTUP_MODE == "vidgen":
    print("🚀 VIDGEN DEFAULT: Loading Wan to GPU first for immediate use...")
    start_primary = time.time()
    
    # Load Wan directly to GPU
    wan_pipe_primary = _load_wan(WAN_DEVICE)
    _active_model = "wan"
    primary_load_time = time.time() - start_primary
    print(f"✅ WAN READY ON GPU in {primary_load_time:.1f}s - Vidgen functional!")
    
    # Apply optimizations to WAN immediately
    print("⚡ Applying inference optimizations to WAN...")
    apply_all_optimizations(wan_pipe, "WAN (vidgen)", enable_compile=True, enable_teacache=False, teacache_thresh=0.05)
    
    # Define pic_pipe as None for now - will load in background
    pic_pipe = None
    
else:
    # PICGEN MODE: Load Qwen to GPU first
    print("🚀 PICGEN MODE: Loading Qwen to GPU first for immediate use...")
    
    # 🚀 AGGRESSIVE QWEN LOADING with concurrent optimization
    print("🚀 AGGRESSIVE LOADING: Qwen Image Edit pipeline...")
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
    print(f"✅ QWEN READY ON GPU in {qwen_time:.1f}s - Picgen functional!")
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
    print(f"🚀 AGGRESSIVE LOADING: {pipeline_name} to {device}")
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
            print(f"    ✅ {component_name} ready ({len(completed_times)}/{total_components})")
    
    # Synchronize all GPU transfers
    torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    print(f"🎯 {pipeline_name} LOADED in {total_time:.1f}s (concurrent speedup: {sum(completed_times)/total_time:.1f}x)")
    
    return pipeline


def activate_wan():
    """Ensure Wan is on WAN_DEVICE and ready."""
    global _active_model

    if DUAL_GPU or GPU_MODE == "stacked":
        # Both modes: Wan is always GPU-resident (dedicated or balanced), no swap
        if not _wan_loaded or wan_pipe is None:
            _load_wan(WAN_DEVICE if DUAL_GPU else "balanced")
        return

    if _active_model == "wan":
        return

    print("🚀 Fast swap to Wan...")
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
        print(f"🎯 Wan active in {swap_time:.1f}s")


def activate_pic():
    """Ensure Qwen is on PIC_DEVICE and ready."""
    global _active_model

    if DUAL_GPU or GPU_MODE == "stacked":
        # Both modes: Qwen is always GPU-resident (dedicated or balanced), no swap
        if pic_pipe is None:
            raise RuntimeError("Qwen pipeline not loaded.")
        return

    if _active_model == "pic":
        return

    # If pic_pipe hasn't loaded yet (vidgen background load still running), wait for it
    if pic_pipe is None:
        print("⏳ Waiting for Qwen to finish loading in background...")
        wait_start = time.time()
        while pic_pipe is None:
            time.sleep(0.5)
            if time.time() - wait_start > 120:
                raise RuntimeError("Qwen failed to load within 120 seconds")
        print(f"✅ Qwen background load complete, proceeding with swap")

    print("🚀 Fast swap to Qwen...")
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
        print(f"🎯 Qwen active in {swap_time:.1f}s")

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
    """Load a starter image (supports .jpg, .png, .webp)."""
    starters_dir = os.path.join(SCRIPT_DIR, "starters")
    for ext in ("jpg", "png", "webp"):
        starter_path = os.path.join(starters_dir, f"start{starter_num}.{ext}")
        if os.path.exists(starter_path):
            with open(starter_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode()
            mime = {"jpg": "jpeg", "png": "png", "webp": "webp"}[ext]
            return f"data:image/{mime};base64,{b64}"
    return ""


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

    # Pin to PIC_DEVICE — prevents cross-device leakage in dual GPU mode
    torch.cuda.set_device(PIC_DEVICE)
    generator = torch.Generator(device=PIC_DEVICE).manual_seed(seed)
    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image.")
    _t_decoded = time.time()

    if height == 256 and width == 256:
        height, width = None, None

    print(f"Prompt: '{prompt}' | Seed: {seed} | Steps: {num_inference_steps}")
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
                + '<button id="picgen-lb-close" style="position:absolute;top:-14px;right:-14px;width:28px;height:28px;border-radius:50%;background:#e53e3e;color:#fff;border:none;cursor:pointer;font-size:16px;line-height:1;">✕</button>'
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
/* Make picgen prompt textareas manually resizable */
#col-container textarea { resize: vertical !important; min-height: 60px !important; touch-action: pan-y !important; }
/* Starter image thumbnail grid */
.starter-grid { display: flex; flex-wrap: nowrap; gap: 4px; padding: 6px 0; justify-content: center; overflow-x: auto; }
.starter-thumb { position: relative; display: flex; flex-direction: column; align-items: center; cursor: pointer; border: 2px solid var(--border-color-primary); border-radius: 6px; overflow: visible; transition: all .15s; flex: 0 0 auto; width: calc(10% - 4px); min-width: 40px; max-width: 70px; }
.starter-thumb:hover { border-color: var(--color-accent); transform: translateY(-2px); box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
.starter-thumb img { width: 100%; aspect-ratio: 1; object-fit: contain; background: var(--background-fill-secondary); border-radius: 4px 4px 0 0; }
.starter-thumb span { font-size: 10px; font-weight: 600; padding: 2px 0; color: var(--body-text-color); }
/* Hover/long-press preview — 4x larger, positioned above the thumbnail */
.starter-thumb .starter-preview { display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); z-index: 9999; pointer-events: none; padding: 4px; background: var(--background-fill-primary); border: 2px solid var(--color-accent); border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.starter-thumb .starter-preview img { width: 240px; height: 240px; object-fit: contain; border-radius: 4px; }
.starter-thumb:hover .starter-preview { display: block; }
.starter-thumb.touch-active .starter-preview { display: block; }
"""

with gr.Blocks(css=css) as demo:
    # Clear button — outside tabs, always visible at top right
    with gr.Row():
        gr.HTML("<div style='flex:1'></div>")  # spacer pushes button right
        clear_storage_btn = gr.Button("🗑 Clear Storage", variant="secondary", size="sm", scale=0)
    clear_storage_status = gr.Textbox(visible=False, label="")

    def clear_storage():
        """Delete all generated files — same as running clear.sh."""
        import shutil as _shutil
        deleted = []
        errors = []

        # 1. tmp/gradio — delete everything except vibe_edit_history
        gradio_dir = Path(SCRIPT_DIR) / "tmp" / "gradio"
        if gradio_dir.exists():
            for item in gradio_dir.iterdir():
                if item.name == "vibe_edit_history":
                    continue
                try:
                    if item.is_dir():
                        _shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted.append(item.name)
                except Exception as e:
                    errors.append(f"{item.name}: {e}")

        # 2. outputs/images — delete contents, keep folder
        images_dir = IMAGE_OUTPUT_DIR
        if images_dir.exists():
            for item in images_dir.iterdir():
                try:
                    if item.is_dir():
                        _shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted.append(item.name)
                except Exception as e:
                    errors.append(f"{item.name}: {e}")

        # 3. outputs/videos — delete contents, keep folder
        videos_dir = VIDEO_OUTPUT_DIR
        if videos_dir.exists():
            for item in videos_dir.iterdir():
                try:
                    if item.is_dir():
                        _shutil.rmtree(item)
                    else:
                        item.unlink()
                    deleted.append(item.name)
                except Exception as e:
                    errors.append(f"{item.name}: {e}")

        if errors:
            return gr.update(visible=True, value=f"⚠️ Done with errors: {'; '.join(errors)}")
        return gr.update(visible=True, value=f"✅ Cleared {len(deleted)} items.")

    clear_storage_btn.click(
        fn=clear_storage,
        inputs=[],
        outputs=[clear_storage_status],
    )

    # Tab 0 = Video Generator (vidgen), Tab 1 = Photo Editor (picgen).
    # -vidgen (default) opens on the Video Generator tab; -picgen opens on the
    # Photo Editor tab.
    with gr.Tabs(selected=("vidgen-tab" if STARTUP_MODE == "vidgen" else "picgen-tab")):

        # ------------------------------------------------------------------ #
        #  TAB 1 — VIDEO GENERATOR (Qwen relocate -> Wan 2.2 4-step animate)  #
        # ------------------------------------------------------------------ #
        with gr.Tab("🎬 Video Generator", id="vidgen-tab"):
            gr.Markdown(model_title())

            with gr.Row():
                with gr.Column(scale=1):
                    reference_image = gr.Image(
                        label="Reference Photo",
                        type="filepath",
                        elem_id="vidgen-reference",
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
                            "identical. Replace = new environment, subjects preserved. "
                            "Custom = your own edit instruction."
                        ),
                    )
                    edit_instruction = gr.Textbox(
                        label="Custom Edit Instruction",
                        value="",
                        lines=2,
                        visible=False,
                        placeholder="Applied verbatim to the frame before animation.",
                    )
                    
                    # Show/hide edit_instruction based on scene_mode
                    scene_mode.change(
                        fn=lambda mode: gr.update(visible=(mode == MODE_CUSTOM)),
                        inputs=[scene_mode],
                        outputs=[edit_instruction],
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
                        add_audio_cb = gr.Checkbox(label="Add Audio (MMAudio)", value=False)
                        audio_prompt_tb = gr.Textbox(
                            label="Audio Prompt", value="natural ambient sound",
                        )

                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Generated Video",
                        elem_id="generated-video",
                        autoplay=True,
                        interactive=False,
                    )
                    with gr.Row():
                        frame_time_input = gr.Number(
                            label="Frame time (seconds) — auto-updates as video plays",
                            value=0.0,
                            minimum=0.0,
                            step=0.1,
                            scale=1,
                            elem_id="frame-time-input",
                        )
                    with gr.Row():
                        use_as_reference_btn = gr.Button(
                            "📌 Use Frame as Reference",
                            scale=1,
                        )
                        download_frame_btn = gr.Button(
                            "💾 Download Frame",
                            scale=1,
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
                    
                    # Centered Generate button
                    generate_btn = gr.Button(
                        "🎬 Generate Video", variant="primary", size="lg"
                    )
                    end_image = gr.Image(
                        label="End Frame (optional, first segment only)",
                        type="pil",
                    )
            # Hidden file component — populated by generate_btn and used for frame extraction
            video_file = gr.File(visible=False)
            # Download Frame output — gr.File shows a clickable download link when populated
            download_file_output = gr.File(label="⬇ Click to Download Frame", visible=True)

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
                    edit_steps = gr.Slider(
                        1, 20, value=4, step=1,
                        label="Frame-Edit Steps (Qwen stage)",
                    )
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

            generate_btn.click(
                fn=generate_video,
                inputs=[
                    reference_image, vid_prompt, scene_mode, edit_instruction,
                    end_image, duration_seconds, resolution, frame_multiplier,
                    export_quality, seed, randomize_seed, add_audio_cb,
                    audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                    flow_shift_auto, flow_shift,
                ],
                outputs=[video_output, video_file],
                concurrency_id=WAN_QUEUE_ID,
                concurrency_limit=10,  # Allow multiple in queue, processed sequentially
            )

            # Frame extraction functions
            def get_frame_as_file(video_path, timestamp):
                """Extract frame at given seconds, return file path."""
                print(f"\n🔍 get_frame_as_file called, ts={timestamp}")
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
                print(f"✅ Frame extracted: {frame_path}")
                return frame_path

            # Use Frame as Reference — reads timestamp from number box, puts frame into reference_image
            use_as_reference_btn.click(
                fn=get_frame_as_file,
                inputs=[video_file, frame_time_input],
                outputs=[reference_image],
            )

            # Download Frame — reads timestamp from number box, puts frame into gr.File for download
            download_frame_btn.click(
                fn=get_frame_as_file,
                inputs=[video_file, frame_time_input],
                outputs=[download_file_output],
            )

        # ------------------------------------------------------------------ #
        #  TAB 2 — PHOTO EDITOR (picgen)                                      #
        # ------------------------------------------------------------------ #
        with gr.Tab("🖼️ Photo Editor", id="picgen-tab"):
            with gr.Column(elem_id="col-container"):

                # Build starter image thumbnails HTML
                def _build_starter_thumbnails_html():
                    """Generate HTML for clickable starter image thumbnails with hover preview."""
                    starters_dir = os.path.join(SCRIPT_DIR, "starters")
                    html_parts = []
                    for i in range(1, 11):
                        img_b64 = None
                        for ext in ("jpg", "png", "webp"):
                            path = os.path.join(starters_dir, f"start{i}.{ext}")
                            if os.path.exists(path):
                                with open(path, "rb") as f:
                                    data = f.read()
                                mime = {"jpg": "jpeg", "png": "png", "webp": "webp"}[ext]
                                img_b64 = f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
                                break
                        if img_b64:
                            html_parts.append(
                                f'<div class="starter-thumb" onclick="window.__loadStarter({i})" title="Starter {i}"'
                                f' ontouchstart="window.__starterTouchStart(this)" ontouchend="window.__starterTouchEnd(this)"'
                                f' ontouchmove="window.__starterTouchEnd(this)">'
                                f'<div class="starter-preview"><img src="{img_b64}" /></div>'
                                f'<img src="{img_b64}" />'
                                f'<span>{i}</span></div>'
                            )
                    if not html_parts:
                        return ""
                    return (
                        '<div class="starter-grid">' + ''.join(html_parts) + '</div>'
                    )

                _starter_html = _build_starter_thumbnails_html()
                if _starter_html:
                    gr.HTML(_starter_html)

                # Hidden components for starter functionality
                starter_trigger = gr.Number(value=0, visible=False, elem_id="starter-trigger")

                with gr.Row():
                    # Left column: input uploader + controls
                    with gr.Column(scale=1):
                        hidden_images_b64 = gr.Textbox(
                            value="[]", elem_id="hidden-images-b64",
                            elem_classes="hidden-input", container=False, visible=False,
                        )
                        gr.HTML("""
                        <div class="uploader-toolbar">
                            <button id="tb-upload" class="tb-btn">⬆ Upload</button>
                            <button id="tb-remove" class="tb-btn">✕ Remove Selected</button>
                            <button id="tb-clear" class="tb-btn">🗑 Clear All</button>
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
                        )
                        use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm")

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

                        pic_run_button = gr.Button("🖼️ Generate", variant="primary", size="lg")

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

                # Preset dropdown handlers — selecting updates prompt box
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
                    fn=infer,
                    inputs=_pic_infer_inputs,
                    js=_pic_infer_js,
                    outputs=[pic_result, pic_seed],
                    concurrency_id=PIC_QUEUE_ID,
                    concurrency_limit=10,
                )

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

                # When starter_trigger changes (from JS click), load that starter image
                starter_trigger.change(
                    fn=lambda num: add_starter_image(int(num)) if num and int(num) > 0 else "",
                    inputs=[starter_trigger],
                    outputs=[starter_b64_output],
                )

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

    # Starter image click handler — updates the hidden number input to trigger Gradio event
    starter_click_js = """
() => {
    window.__loadStarter = function(num) {
        const input = document.querySelector('#starter-trigger input');
        if (input) {
            const nativeInput = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
            if (nativeInput && nativeInput.set) {
                nativeInput.set.call(input, '0');
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
                setTimeout(() => {
                    nativeInput.set.call(input, String(num));
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }, 50);
            }
        }
    };
    // Mobile long-press preview (2 seconds hold shows preview, release hides it)
    window.__starterTouchStart = function(el) {
        el._touchTimer = setTimeout(() => {
            el.classList.add('touch-active');
            el._previewing = true;
        }, 500);
    };
    window.__starterTouchEnd = function(el) {
        clearTimeout(el._touchTimer);
        el.classList.remove('touch-active');
        // Only fire click if not previewing (short tap)
        if (el._previewing) {
            el._previewing = false;
        }
    };
}
"""
    demo.load(fn=None, js=starter_click_js)

    # Auto-sync video currentTime to the frame_time_input number box
    video_time_sync_js = """
() => {
    function startVideoSync() {
        const video = document.querySelector('#generated-video video');
        if (!video) { setTimeout(startVideoSync, 500); return; }
        // Update the number input every 100ms while video is playing
        setInterval(() => {
            const input = document.querySelector('#frame-time-input input');
            if (input && !isNaN(video.currentTime)) {
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

if __name__ == "__main__":
    os.makedirs(os.path.join(SCRIPT_DIR, "tmp"), exist_ok=True)

    # ===== STARTUP DIAGNOSTICS BANNER =====
    mode_names = {
        "concurrent": "Dual-Tab Concurrent Mode (Both Models GPU-Resident)",
        "stacked": "Multi-GPU Pipeline Parallelism (Both Models Split Across GPUs)",
        "single": "Single-GPU Dynamic Mode (CPU Offload + Swap)",
    }
    print("\n" + "=" * 80)
    print("                    AI ENGINE INITIALIZATION & DIAGNOSTICS")
    print("=" * 80)
    gpu_summary = ", ".join([f"{g['name']} ({g['vram_mb']}MB)" for g in _gpu_info])
    print(f"GPUs: {_gpu_count}x — {gpu_summary}")
    print(f"Total VRAM: {_total_vram_mb // 1024} GB")
    print(f"Operational Mode: {mode_names.get(GPU_MODE, GPU_MODE)}")
    opt_status = []
    opt_status.append(f"SageAttention ({'Enabled' if _SAGE_ATTENTION_AVAILABLE else 'Not Installed'})")
    opt_status.append(f"TeaCache ({'Enabled' if _TEACACHE_ENABLED else 'Available'})")
    opt_status.append(f"torch.compile ({'Available' if _TORCH_COMPILE_AVAILABLE else 'Unavailable'})")
    print(f"Optimizations: {' | '.join(opt_status)}")
    print("-" * 80)
    print("              ESTIMATED GENERATION TIMES (CURRENT SETUP)")
    print("-" * 80)
    # Rough estimates based on hardware
    if _total_vram_mb >= 180000:  # 2x 95GB Blackwell
        print("  * Photo Editor (1 Image @ 4 Steps)       : ~3 - 5 seconds")
        print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~8 - 15 seconds")
    elif _total_vram_mb >= 80000:  # 1x 95GB or equivalent
        print("  * Photo Editor (1 Image @ 4 Steps)       : ~5 - 8 seconds")
        print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~15 - 25 seconds")
    elif _total_vram_mb >= 40000:  # 2x 24GB
        print("  * Photo Editor (1 Image @ 4 Steps)       : ~8 - 12 seconds")
        print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~30 - 50 seconds")
    else:  # single 24GB or less
        print("  * Photo Editor (1 Image @ 4 Steps)       : ~10 - 15 seconds")
        print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~45 - 80 seconds")
    print("=" * 80 + "\n")

    if DUAL_GPU or GPU_MODE == "stacked":
        mode_label = "CONCURRENT" if DUAL_GPU else "STACKED (pipeline parallel)"
        print(f"🚀 GRADIO LAUNCHING — {mode_label}. Both tabs ready.")
        demo.queue(default_concurrency_limit=10)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, str(OUTPUT_DIR)],
        )
    else:
        if STARTUP_MODE == "vidgen":
            print("🚀 GRADIO LAUNCHING — Wan on GPU, vidgen ready immediately.")
        else:
            print("🚀 GRADIO LAUNCHING — Qwen on GPU, picgen ready immediately.")
        demo.queue(default_concurrency_limit=1)

        # Start background loading BEFORE demo.launch (which blocks forever)
        def _bg_load():
            try:
                time.sleep(2.0)
                if STARTUP_MODE == "vidgen":
                    print("📦 Background: Loading Qwen to CPU...")
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
                    print(f"✅ Qwen on CPU in {time.time()-start:.1f}s — tab switching ready!")
                else:
                    print("📦 Background: Loading Wan to CPU...")
                    _load_wan("cpu")
                    # Apply optimizations to Wan
                    apply_sage_attention(wan_pipe)
                    print("✅ Wan on CPU — tab switching ready!")
            except Exception as e:
                print(f"❌ Background load failed: {e}")
                import traceback; traceback.print_exc()

        threading.Thread(target=_bg_load, daemon=True).start()

        # Launch Gradio AFTER starting background thread (demo.launch blocks forever)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, str(OUTPUT_DIR)],
        )