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
# 'highest' forces full-precision FP32 matmul kernels, which silently defeats
# the allow_tf32 flags set right below — those only take effect at 'high' or
# lower. The main denoising math runs in bf16 either way (unaffected), but any
# fp32 matmul in the pipeline (text-encoder bits, RIFE, misc ops) was paying
# the slow-path tax for no accuracy benefit. 'high' enables TF32 tensor-core
# matmul for fp32 inputs — the same numerics class already opted into via
# allow_tf32 two lines down, just no longer silently overridden here.
torch.set_float32_matmul_precision('high')
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

# The RIFE zip only contains train_log/ (flownet.pkl + .py wrappers).
# RIFE_HDv3.py imports from "model.warplayer" and "model.IFNet_HDv3" which are
# part of the Practical-RIFE repo's model/ directory. If model/ doesn't exist
# at cwd, clone just that folder from the repo.
if not os.path.exists("model") or not os.path.exists("model/warplayer.py"):
    print("Downloading RIFE model/ package (warplayer, IFNet)...")
    subprocess.run([
        "git", "clone", "--depth=1", "--filter=blob:none", "--sparse",
        "https://github.com/hzwer/Practical-RIFE.git", "_rife_tmp"
    ], check=True)
    subprocess.run(["git", "sparse-checkout", "set", "model"], check=True, cwd="_rife_tmp")
    # Move model/ to cwd and clean up
    if os.path.exists("_rife_tmp/model"):
        shutil.move("_rife_tmp/model", "model")
    shutil.rmtree("_rife_tmp", ignore_errors=True)

sys.path.append(os.path.join(os.getcwd(), "train_log"))
from train_log.RIFE_HDv3 import Model

# ---------------------------------------------------------------------------
# HARDWARE DETECTION & EXECUTION MODE SELECTION
#
# Automatically detects GPU configuration and selects optimal execution mode:
#   CONCURRENT  — Both models fully GPU-resident simultaneously, no swapping.
#                 Requires enough total VRAM for both models + activation headroom.
#   SINGLE      — One model on GPU at a time (spread across all available GPUs),
#                 the other stays on CPU. Swaps on tab switch.
#
# Model VRAM requirements (bfloat16):
#   WAN 2.2 I2V A14B : ~57 GB
#   Qwen Image Edit  : ~35 GB
#   Combined         : ~92 GB
#   Headroom needed  : ~15 GB (activations, KV cache, intermediates)
#   Safe concurrent  : ≥107 GB total VRAM
#
# Override: NEWGEN_FORCE_SINGLE_GPU=1 forces swap mode on any box.
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

# Concurrent (dual-resident) mode keeps BOTH models loaded on the GPU(s) at
# once, with no swapping. That only works when there are two separate
# physical GPUs — one per model — so each has its own fully uncontended VRAM.
#
# A single GPU, even a 95 GB Blackwell card, does NOT reliably fit both
# models resident at once. The optimistic math below (Wan ~57 GB + Qwen peak
# ~20-25 GB via enable_model_cpu_offload = ~82 GB, "fits" in 95 GB) does not
# hold in practice: Wan alone has been observed pinning ~94 GB by itself,
# leaving no room for Qwen and crashing with a CUDA OOM on the very first
# Photo Editor generation. So concurrent mode now requires >= 2 GPUs, full
# stop — a single card always uses swap mode (one model on GPU, the other
# parked on CPU, swapped in on tab switch / first use of that model).
_force_single = os.environ.get("NEWGEN_FORCE_SINGLE_GPU") == "1"

if _force_single or _gpu_count < 2:
    GPU_MODE = "single"
else:
    # Two or more GPUs: each model gets its own dedicated card.
    GPU_MODE = "concurrent"

# Legacy alias — stacked no longer exists, everything is either concurrent or single.
DUAL_GPU = (GPU_MODE == "concurrent")

# Device assignments based on mode
if GPU_MODE == "concurrent":
    # Always >= 2 GPUs here (see gating above) — each model gets its own card.
    PIC_DEVICE = "cuda:0"
    WAN_DEVICE = "cuda:1"
    PIC_QUEUE_ID = "pic-gpu"
    WAN_QUEUE_ID = "wan-gpu"
else:  # single — one model on GPU (all cards), other on CPU, swapped on demand
    PIC_DEVICE = "cuda:0"
    WAN_DEVICE = "cuda:0"
    PIC_QUEUE_ID = "gpu"
    WAN_QUEUE_ID = "gpu"

# Flag: True when both models share a single physical GPU in concurrent mode.
# Structurally always False now (concurrent mode requires >= 2 GPUs) — kept
# only so the single-card-concurrent code paths below remain valid dead code
# rather than needing to be ripped out, in case a future card genuinely has
# the headroom to justify reviving this path.
_SINGLE_CARD_CONCURRENT = (GPU_MODE == "concurrent" and _gpu_count == 1)

# Whether Qwen's generator/execution device should be treated as CUDA.
# True for: two-GPU concurrent (Qwen pinned), and single-card concurrent
# IF there's enough free VRAM after Wan loads to use enable_model_cpu_offload.
# Set to False (forcing a CPU generator, matching swap-mode's requirement)
# if single-card concurrent falls back to enable_sequential_cpu_offload for
# Qwen because Wan alone leaves too little headroom — see _load_qwen_thread.
_QWEN_CUDA_GENERATOR = DUAL_GPU

# Auxiliary models (RIFE interpolation, MMAudio) run on the last GPU.
# In single/swap mode cuda:0 is shared by both primary models, so putting
# RIFE there eats ~500 MB of headroom that Qwen's text_encoder needs. Using
# the last card keeps the swap device clean.
_rife_device_idx = _gpu_count - 1 if _gpu_count > 1 else 0
device = torch.device(f"cuda:{_rife_device_idx}")

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
    """
    Patch SageAttention into Wan's transformer attention modules directly.

    Previous approach patched F.scaled_dot_product_attention globally, which
    broke Qwen's text encoder (all-black images). This approach replaces only
    the forward methods on Wan's WanAttention blocks, leaving every other model
    completely untouched.

    SageAttention gives a true 2-3x speedup on attention compute — the dominant
    cost in the denoising loop — with numerically equivalent output (it's a
    flash-attention variant, not an approximation).
    """
    if not _SAGE_ATTENTION_AVAILABLE:
        return False
    try:
        from sageattention import sageattn
        patched = 0

        def _make_sage_forward(original_forward):
            """
            Wrap a module's forward so SDPA calls inside it use sageattn.
            We temporarily swap F.scaled_dot_product_attention on the torch.nn.functional
            module that the attention class already has a reference to, but only
            for the duration of that one forward call — thread-local would be
            ideal, but since generation is serialised by the queue, a brief
            monkeypatch+restore is safe and avoids class-level surgery.
            """
            _orig_sdpa = F.scaled_dot_product_attention

            def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                           is_causal=False, scale=None, enable_gqa=False, **kwargs):
                # Parameter names MUST match torch.nn.functional's real
                # signature (query/key/value) rather than shorthand (q/k/v).
                # Some call sites (e.g. diffusers' newer attention_dispatch
                # path used by Qwen) call this by keyword using the real
                # names — with mismatched names those keywords fall through
                # to **kwargs and query/key/value are left unfilled, raising
                # "missing 3 required positional arguments". Positional
                # callers (Wan) work under either naming, so matching the
                # real signature is strictly safer and fixes both.
                try:
                    return sageattn(query, key, value, attn_mask=attn_mask,
                                     is_causal=is_causal, sm_scale=scale)
                except Exception:
                    return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                                      dropout_p=dropout_p, is_causal=is_causal,
                                      scale=scale, enable_gqa=enable_gqa)

            def _forward_with_sage(*args, **kwargs):
                F.scaled_dot_product_attention = _sage_sdpa
                try:
                    return original_forward(*args, **kwargs)
                finally:
                    F.scaled_dot_product_attention = _orig_sdpa

            return _forward_with_sage

        # Patch every attention block inside the transformer(s).
        # WanImageToVideoPipeline has .transformer and .transformer_2 (MoE pair).
        transformers_to_patch = []
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            transformers_to_patch.append(pipe.transformer)
        if hasattr(pipe, 'transformer_2') and pipe.transformer_2 is not None:
            transformers_to_patch.append(pipe.transformer_2)

        for xfmr in transformers_to_patch:
            for module in xfmr.modules():
                # Target the named attention classes used by Wan's diffusers impl.
                cls_name = type(module).__name__
                if "Attention" in cls_name and hasattr(module, 'forward'):
                    module.forward = _make_sage_forward(module.forward)
                    patched += 1

        if patched:
            print(f"  SageAttention: patched {patched} attention modules (Wan transformers only)")
            return True
        return False
    except Exception as e:
        print(f"  SageAttention patch failed: {e}")
        return False


def apply_torch_compile(pipe, mode="default"):
    """
    Compile the pipeline's transformer forward pass(es) for kernel fusion.

    Previous attempt used the default mode on the whole pipeline object, which
    triggered infinite recompilation because Wan's MoE pair (transformer +
    transformer_2) alternates mid-inference — the graph tracer saw dynamic
    control flow and re-traced on every step.

    Fix: compile each transformer independently with dynamic=True (handles
    varying sequence lengths without retracing) and fullgraph=False (allows the
    Python-level MoE switching to remain outside the compiled region). The
    compiled kernels cover the heavy attention + MLP compute inside each
    transformer's forward, which is where the time actually goes.

    Mode is 'default', NOT 'reduce-overhead'. 'reduce-overhead' captures a
    CUDA Graph to cut Python/launch overhead — but CUDA Graphs need the
    allocator to support a live-allocation-pool check, and this process runs
    with PYTORCH_CUDA_ALLOC_CONF=...backend:cudaMallocAsync (chosen deliberately
    for the offload hooks' constant alloc/free churn as models stream on and
    off GPU). cudaMallocAsync does not implement that check at all, so
    'reduce-overhead' crashes hard the first time a compiled forward actually
    runs — not a flaky failure, a guaranteed one, and torch._dynamo's
    suppress_errors does NOT catch it (it's a runtime crash inside the
    compiled graph's execution, not a trace/compile-time error). 'default'
    still gets the real win — TorchInductor kernel fusion of the attention +
    MLP compute — it just skips the incompatible graph-capture step. Given the
    bottleneck here is GPU compute (large matmuls/attention), not CPU
    dispatch overhead, 'default' captures nearly all of the available speedup
    with none of the crash risk.

    First call triggers a ~30s JIT compilation; all subsequent calls are fast.
    """
    if not _TORCH_COMPILE_AVAILABLE:
        return False
    try:
        compiled = 0
        for attr in ('transformer', 'transformer_2'):
            xfmr = getattr(pipe, attr, None)
            if xfmr is None:
                continue
            # compile the forward method, not the module itself, to avoid
            # issues with accelerate hooks wrapping the module
            xfmr.forward = torch.compile(
                xfmr.forward,
                mode=mode,
                dynamic=True,       # handles varying S/B without retrace
                fullgraph=False,    # allows Python control flow at MoE switch
            )
            compiled += 1
        if compiled:
            print(f"  torch.compile: compiled {compiled} transformer(s) "
                  f"(mode={mode}, dynamic=True) — first run triggers JIT warmup")
            return True
        return False
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        return False


def apply_vae_fp16(pipe):
    """
    Cast the VAE to fp16 for decode.

    RTX 4070 Ti Super (Ada Lovelace) has the same fp16 tensor-core throughput
    as bf16 but with faster memory bandwidth on sub-16-bit ops. More importantly,
    diffusers' VAE tiling already splits frames into chunks, so precision
    differences are invisible in the output. bfloat16 is kept for the main
    denoising loop (better dynamic range for diffusion math); fp16 is only
    applied to the VAE encoder+decoder which are pure conv nets.
    """
    try:
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            pipe.vae = pipe.vae.to(dtype=torch.float16)
            print("  VAE → fp16 (faster decode on Ada, imperceptible quality delta)")
            return True
    except Exception as e:
        print(f"  VAE fp16 cast failed: {e}")
    return False


def apply_teacache(pipe, threshold=0.05):
    """
    Enable TeaCache for video diffusion — training-free timestep caching.

    TeaCache skips redundant transformer evaluations at timesteps where the
    residual L1 distance between consecutive hidden states falls below
    `threshold`. At 0.05 it typically skips ~40% of forward passes with
    negligible visual difference. Lower = fewer skips = slower but safer.
    """
    global _TEACACHE_ENABLED
    try:
        if hasattr(pipe, 'enable_teacache'):
            pipe.enable_teacache(
                cache_interval=2,
                rel_l1_thresh=threshold,
            )
            _TEACACHE_ENABLED = True
            return True
    except Exception as e:
        print(f"  TeaCache enable failed: {e}")
    return False


def apply_all_optimizations(pipe, pipe_name="model", enable_compile=True,
                            enable_teacache=False, teacache_thresh=0.05,
                            enable_sage=True, enable_vae_fp16=False):
    """Apply all available inference accelerations to a pipeline."""
    results = {}

    # SageAttention — per-module patch, safe for mixed-model setups
    if enable_sage:
        results["SageAttention"] = apply_sage_attention(pipe)

    # TeaCache — timestep skip for video transformers
    if enable_teacache:
        results["TeaCache"] = apply_teacache(pipe, threshold=teacache_thresh)

    # VAE fp16 — faster decode on Ada/Ampere, no visible quality change
    if enable_vae_fp16:
        results["VAE-fp16"] = apply_vae_fp16(pipe)

    # torch.compile — kernel fusion, applied last to capture optimized graph
    if enable_compile:
        results["torch.compile"] = apply_torch_compile(pipe)

    applied = [k for k, v in results.items() if v]
    skipped = [k for k, v in results.items() if not v]
    if applied:
        print(f"  ✅ {pipe_name}: {', '.join(applied)}")
    if skipped:
        print(f"  ⏭️  {pipe_name} skipped: {', '.join(skipped)}")
    return results


# ---------------------------------------------------------------------------
# VRAM-TIERED OFFLOAD STRATEGY
#
# In single-GPU "swap" mode, only ONE model is ever resident on the card at a
# time — the other is fully parked on CPU. That means whichever model is
# active gets the ENTIRE card's VRAM to itself, not a shared slice. Despite
# that, the previous code unconditionally called
# enable_sequential_cpu_offload() for every swap — the slowest of the three
# available strategies (it streams weights onto the GPU one individual layer
# at a time, for every single layer, on every single forward pass). That
# tier exists to survive tight-VRAM situations; it is massive overkill on any
# card with enough free memory to hold the model outright.
#
# This picks the fastest tier that will actually fit, fastest first:
#   "full"          — pipe.to(device): fully resident, zero offload overhead
#   "model_offload" — enable_model_cpu_offload: whole-submodule streaming
#                      (submodule moves to GPU only while executing, then
#                      back to CPU — peak VRAM = size of the largest single
#                      submodule, not the whole model)
#   "sequential"    — enable_sequential_cpu_offload: layer-by-layer (safest,
#                      slowest — reserved for genuinely tight VRAM)
#
# The winning tier is cached per model so repeat swaps skip the measurement
# and jump straight to it. If a cached tier ever OOMs anyway (fragmentation,
# another process, etc.) the call transparently falls back a tier, retries,
# and re-caches the safe result — generation never crashes because of this,
# it can only get faster than the old always-slowest default.
# ---------------------------------------------------------------------------

_offload_tier_cache = {}
_OFFLOAD_TIERS = ("full", "model_offload", "sequential")


def _apply_offload_tier(pipe, model_key, device_idx, full_weight_gb, submodule_gb, safety_gb=5):
    """Pick + apply the fastest offload strategy for `pipe` that fits in free VRAM."""
    device = f"cuda:{device_idx}"
    cached = _offload_tier_cache.get(model_key)

    if cached is None:
        torch.cuda.synchronize(device_idx)
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
        free_gb = free_bytes / (1024 ** 3)
        if free_gb >= full_weight_gb + safety_gb:
            candidates = ["full", "model_offload", "sequential"]
        elif free_gb >= submodule_gb + safety_gb:
            candidates = ["model_offload", "sequential"]
        else:
            candidates = ["sequential"]
        print(f"  offload tier probe ({model_key}): {free_gb:.1f} GB free on {device} "
              f"→ trying {candidates[0]}")
    else:
        start = _OFFLOAD_TIERS.index(cached)
        candidates = list(_OFFLOAD_TIERS[start:])

    last_err = None
    for tier in candidates:
        try:
            if tier == "full":
                pipe.to(device)
            elif tier == "model_offload":
                pipe.enable_model_cpu_offload(gpu_id=device_idx)
            else:
                pipe.enable_sequential_cpu_offload(gpu_id=device_idx)
            if tier != cached:
                _offload_tier_cache[model_key] = tier
                print(f"  ✅ {model_key}: offload tier = {tier}")
            return tier
        except torch.cuda.OutOfMemoryError as e:
            last_err = e
            print(f"  ⚠️  {model_key}: tier '{tier}' OOM'd, falling back...")
            torch.cuda.empty_cache()
            continue
    # "sequential" is always last in the chain and has the smallest footprint
    # of the three — if even that raises, something else is wrong.
    raise last_err


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
WAN_STEPS = 3
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

# ---------------------------------------------------------------------------
# WAN INFERENCE CACHES
#
# Two LRU-style caches that survive across calls in the same process:
#
#   _wan_text_cache  — T5 text embeddings keyed by (prompt, negative_prompt).
#                      Encoding a ~100-token prompt through T5-XXL takes ~0.3s
#                      and produces identical outputs for the same text, so
#                      caching is pure win. Invalidated on model swap because
#                      the text_encoder moves off GPU and may be reloaded.
#
#   _wan_image_cache — VAE image latents keyed by (image_hash, resolution).
#                      The VAE encode of the reference frame is identical for
#                      the same image every time. Saving it avoids a ~0.5s
#                      encode + GPU round-trip per generation.
#
# Both caches store GPU tensors (on WAN_DEVICE). They are cleared in
# activate_wan() only when a model swap actually happens, so they persist
# across repeated same-model calls.
# ---------------------------------------------------------------------------

_wan_cache_lock = threading.Lock()
_wan_text_cache: dict = {}   # (prompt, neg) -> (encoder_hidden_states, attention_mask)
_wan_image_cache: dict = {}  # (img_hash, res) -> image_latents tensor
_WAN_CACHE_MAX = 16


def _wan_hash_image(pil_img):
    """Fast perceptual hash of a PIL image for cache keying."""
    arr = np.array(pil_img.resize((64, 64), Image.LANCZOS))
    h = hashlib.sha256()
    h.update(f"{pil_img.size}".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _wan_cache_clear():
    """Drop all Wan-side caches (call after a model swap to free GPU tensors)."""
    with _wan_cache_lock:
        _wan_text_cache.clear()
        _wan_image_cache.clear()


def _wan_text_cache_get(prompt, negative_prompt):
    key = (prompt, negative_prompt or "")
    with _wan_cache_lock:
        return _wan_text_cache.get(key)


def _wan_text_cache_put(prompt, negative_prompt, value):
    key = (prompt, negative_prompt or "")
    with _wan_cache_lock:
        if len(_wan_text_cache) >= _WAN_CACHE_MAX:
            _wan_text_cache.pop(next(iter(_wan_text_cache)))
        _wan_text_cache[key] = value


def _wan_image_cache_get(pil_img, resolution):
    key = (_wan_hash_image(pil_img), resolution)
    with _wan_cache_lock:
        return _wan_image_cache.get(key)


def _wan_image_cache_put(pil_img, resolution, value):
    key = (_wan_hash_image(pil_img), resolution)
    with _wan_cache_lock:
        if len(_wan_image_cache) >= _WAN_CACHE_MAX:
            _wan_image_cache.pop(next(iter(_wan_image_cache)))
        _wan_image_cache[key] = value


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
        # No longer used — kept for safety, treated same as a direct GPU load
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory={i: 10 * 1024 for i in range(torch.cuda.device_count())},
            use_safetensors=True
        )
        print(f"🎯 WAMU v2 loaded BALANCED across {_gpu_count} GPUs (≤10 GB/card) — pipeline parallelism active!")
    else:
        # Load to CPU then spread across all GPUs via balanced device_map.
        # This is used when the model is the active one at startup (single/swap mode
        # with multiple GPUs). Each card contributes its share; total must fit.
        pipeline = WanImageToVideoPipeline.from_pretrained(
            WAN_MODEL_REPO,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            use_safetensors=True
        )
        torch.cuda.synchronize()
        print(f"🎯 WAMU v2 loaded across {torch.cuda.device_count()} GPUs (balanced) - Ready for video generation!")

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
MODE_TIMELINE = "Timeline (per-segment prompts)"
SCENE_MODES = [MODE_KEEP, MODE_REPLACE, MODE_CUSTOM, MODE_TIMELINE]

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
    additional_images: list = None,
) -> Image.Image:
    """
    Optional stage 1 — Qwen Image Edit prepares the starting frame.

    Returns the image untouched in keep-scene mode, so nothing is repainted and
    the original background and bodies survive exactly as shot.
    
    When additional_images are provided (Custom/Replace modes), all images are
    passed to Qwen as multi-reference input for better context.
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

    # Build image list — primary image first, then any additional references
    images_list = [image]
    if additional_images:
        for ref in additional_images:
            pil_ref = _ensure_pil(ref)
            if pil_ref is not None:
                images_list.append(pil_ref)
        if len(images_list) > 1:
            print(f"  Multi-reference: {len(images_list)} images provided to Qwen")

    torch.cuda.set_device(PIC_DEVICE)
    _pic_ctx = torch.cuda.device(PIC_DEVICE) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
    _pic_autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
    with _pic_ctx:
        with _pic_autocast:
            result = pic_pipe(
                image=images_list,
                prompt=instruction,
                negative_prompt=" ",
                num_inference_steps=int(steps),
                true_cfg_scale=float(guidance),
                generator=torch.Generator(device=PIC_DEVICE if _QWEN_CUDA_GENERATOR else "cpu").manual_seed(seed),
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
    """
    Stage 2 — animate a frame with WAMU v2 merged model.

    Caches T5 text embeddings and VAE image latents across calls so that
    repeated generations with the same prompt / reference image skip those
    encode steps entirely. On a 6×4070 Ti Super rig this saves ~0.4-0.8s
    per generation after the first warm call.
    """
    activate_wan()

    # In concurrent/dual mode pin computation to WAN_DEVICE.
    # In swap mode enable_sequential_cpu_offload manages device routing
    # internally — wrapping with torch.cuda.device() confuses it and causes
    # "Expected all tensors to be on the same device" crashes.
    _cuda_ctx = torch.cuda.device(WAN_DEVICE) if DUAL_GPU else contextlib.nullcontext()
    with _cuda_ctx:
        # Apply flow shift if provided, otherwise use WAMU v2 default (6.9)
        _set_flow_shift(wan_pipe, flow_shift if flow_shift is not None else WAN_FLOW_SHIFT)

        print(f"[2/2] Wan animating {num_frames} frames at {frame.size}...")
        print(f"Prompt: {prompt!r} | Seed: {seed} | Steps: {WAN_STEPS}")

        # ---- Text embedding cache ----------------------------------------
        # WanImageToVideoPipeline accepts prompt_embeds + negative_prompt_embeds
        # as direct kwargs, bypassing the T5 encoder entirely on cache hits.
        neg = negative_prompt or ""
        cached_text = _wan_text_cache_get(prompt, neg)
        text_kwargs = {}
        if cached_text is not None:
            pe, pe_mask, npe, npe_mask = cached_text
            text_kwargs = dict(
                prompt_embeds=pe,
                negative_prompt_embeds=npe,
                prompt_attention_mask=pe_mask,
                negative_prompt_attention_mask=npe_mask,
            )
            print("  text embeds: cache hit")
        else:
            print("  text embeds: computing (will cache)")

        # ---- Base kwargs ---------------------------------------------------
        # When enable_sequential_cpu_offload is active the pipeline's internal
        # device is CPU, so the generator must also be CPU — passing a CUDA
        # generator to a CPU pipeline raises "Cannot generate a cpu tensor from
        # a generator of type cuda". In concurrent/dual mode Wan is pinned to
        # WAN_DEVICE so a CUDA generator is correct there.
        _wan_gen_device = "cpu" if not DUAL_GPU else WAN_DEVICE
        base_kwargs = dict(
            image=frame,
            height=frame.height,
            width=frame.width,
            num_frames=num_frames,
            num_inference_steps=WAN_STEPS,
            guidance_scale=WAN_GUIDANCE,
            guidance_scale_2=WAN_GUIDANCE,
            generator=torch.Generator(device=_wan_gen_device).manual_seed(seed),
            output_type="np",
            **text_kwargs,
        )

        # Only pass prompt/negative_prompt when we don't have cached embeds
        if not text_kwargs:
            base_kwargs["prompt"] = prompt
            base_kwargs["negative_prompt"] = neg

        # ---- Run pipeline & capture text embeds on first call --------------
        if last_frame is None:
            result = wan_pipe(**base_kwargs)
        else:
            try:
                result = wan_pipe(last_image=last_frame, **base_kwargs)
            except TypeError as e:
                print(f"End frame not supported by this pipeline ({e}) — ignoring it.")
                result = wan_pipe(**base_kwargs)

        # Cache the text embeddings for next call (if we just computed them)
        if cached_text is None and hasattr(result, '_text_embeds_cache'):
            # Some diffusers versions expose this on the result object
            _wan_text_cache_put(prompt, neg, result._text_embeds_cache)
        elif cached_text is None:
            # Fallback: encode explicitly and cache for next time.
            # This runs only once per unique prompt; after that it's instant.
            try:
                with torch.no_grad():
                    _enc_out = wan_pipe.encode_prompt(
                        prompt=prompt,
                        negative_prompt=neg,
                        device=WAN_DEVICE if DUAL_GPU else "cpu",
                        do_classifier_free_guidance=False,
                    )
                    # encode_prompt returns varying shapes depending on version;
                    # unpack safely
                    if isinstance(_enc_out, (list, tuple)) and len(_enc_out) >= 4:
                        _wan_text_cache_put(prompt, neg, tuple(_enc_out[:4]))
                    elif isinstance(_enc_out, (list, tuple)) and len(_enc_out) == 2:
                        # (prompt_embeds, attention_mask) only — store with None placeholders
                        _wan_text_cache_put(prompt, neg, (_enc_out[0], _enc_out[1], None, None))
            except Exception as _cache_err:
                # Non-fatal — next call will just re-encode
                print(f"  text embed cache fill failed (non-fatal): {_cache_err}")

        return result.frames[0]


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
    additional_refs=None,
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
    # If Timeline mode, the user should use the Timeline Generate button instead
    if scene_mode == MODE_TIMELINE:
        raise gr.Error("Timeline mode selected — use the '🎬 Generate Timeline Video' button instead.")
    
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

        # Process additional reference images if provided
        additional_pil = None
        if additional_refs and scene_mode != MODE_KEEP:
            additional_pil = []
            for ref_file in additional_refs:
                ref_path = ref_file.name if hasattr(ref_file, 'name') else str(ref_file)
                pil_ref = _ensure_pil(ref_path)
                if pil_ref:
                    additional_pil.append(pil_ref)
            if additional_pil:
                print(f"  {len(additional_pil)} additional reference image(s) for Qwen edit")

        # ---- Stage 1: optional frame preparation --------------------------
        start_frame = edit_reference_frame(
            sized, scene_mode, prompt, edit_instruction,
            current_seed, edit_steps, edit_guidance,
            additional_images=additional_pil,
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
# TIMELINE VIDEO GENERATION
#
# Generates a multi-segment video where each segment has its own Qwen edit
# instruction. Each segment's starting frame is generated by editing the
# previous segment's last frame with the next prompt.
# ---------------------------------------------------------------------------

def generate_timeline_video(
    reference_image,
    segment_prompts_json,
    motion_prompt,
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
    additional_refs=None,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Timeline video generation — each segment gets its own Qwen edit instruction.
    
    Flow: reference_image → Qwen edit (prompt 1) → Wan animate → last frame
         → Qwen edit (prompt 2) → Wan animate → last frame → ... → final video
    """
    reference_image = _ensure_pil(reference_image)
    if reference_image is None:
        raise gr.Error("Please upload a reference photo.")
    
    # Parse segment prompts
    try:
        segment_prompts = json.loads(segment_prompts_json) if segment_prompts_json else []
    except (json.JSONDecodeError, TypeError):
        segment_prompts = []
    
    if not segment_prompts:
        raise gr.Error("No segment prompts provided. Add at least one segment prompt.")
    
    # Use motion prompt for animation if provided
    if not motion_prompt or not motion_prompt.strip():
        motion_prompt = default_video_prompt
    
    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt
    
    # Flow shift
    num_segments = len(segment_prompts)
    total_duration = num_segments * SEGMENT_DURATION
    
    if flow_shift_auto:
        if total_duration <= 6.0:
            flow_shift = 6.9
        elif total_duration <= 10.0:
            flow_shift = 5.5
        elif total_duration <= 20.0:
            flow_shift = 4.5
        else:
            flow_shift = 4.0
        print(f"🎯 Timeline auto flow_shift: {flow_shift:.1f} ({num_segments} segments, {total_duration:.1f}s)")
    else:
        if flow_shift is None:
            flow_shift = WAN_FLOW_SHIFT
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    segment_paths = []
    
    # Process additional reference images
    additional_pil = None
    if additional_refs:
        additional_pil = []
        for ref_file in additional_refs:
            ref_path = ref_file.name if hasattr(ref_file, 'name') else str(ref_file)
            pil_ref = _ensure_pil(ref_path)
            if pil_ref:
                additional_pil.append(pil_ref)

    try:
        sized = resize_image_for_wan(reference_image, resolution)
        current_frame = sized
        
        for seg_idx, seg_prompt in enumerate(segment_prompts):
            seg_num = seg_idx + 1
            seg_prompt = seg_prompt.strip()
            
            if not seg_prompt:
                print(f"⚠️  Segment {seg_num}: empty prompt, using previous frame as-is")
                start_frame = current_frame
            else:
                # Qwen edits the current frame with this segment's prompt
                print(f"🎬 Segment {seg_num}/{num_segments}: Qwen editing → {seg_prompt[:60]}...")
                
                # Build image list for Qwen
                images_for_qwen = [current_frame]
                if additional_pil:
                    images_for_qwen.extend(additional_pil)
                
                # Timeline needs stronger guidance to make visible changes between segments
                timeline_guidance = max(float(edit_guidance), 3.0)
                timeline_steps = max(int(edit_steps), 6)
                
                activate_pic()
                torch.cuda.set_device(PIC_DEVICE)
                _tl_ctx = torch.cuda.device(PIC_DEVICE) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
                _tl_ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
                with _tl_ctx:
                    with _tl_ac:
                        result = pic_pipe(
                            image=images_for_qwen,
                            prompt=seg_prompt,
                            negative_prompt=" ",
                            num_inference_steps=timeline_steps,
                            true_cfg_scale=timeline_guidance,
                            generator=torch.Generator(device=PIC_DEVICE if _QWEN_CUDA_GENERATOR else "cpu").manual_seed(current_seed + seg_idx),
                        )
                start_frame = result.images[0]
                if start_frame.size != current_frame.size:
                    start_frame = start_frame.resize(current_frame.size, Image.LANCZOS)
            
            # Animate this segment with Wan
            num_frames = get_num_frames(SEGMENT_DURATION)
            seg_seed = current_seed + seg_idx * 100
            
            raw_frames = animate_frame(
                start_frame, None, motion_prompt, negative_prompt,
                num_frames, seg_seed, flow_shift,
            )
            
            # RIFE interpolation
            factor = max(1, int(frame_multiplier) // FIXED_FPS)
            if factor > 1:
                seg_frames = interpolate_bits(raw_frames, multiplier=factor)
            else:
                seg_frames = list(raw_frames)
            seg_fps = FIXED_FPS * factor
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                seg_path = f.name
            export_to_video(seg_frames, seg_path, fps=seg_fps, quality=int(export_quality))
            segment_paths.append(seg_path)
            
            print(f"  ✅ Segment {seg_num} complete ({SEGMENT_DURATION:.1f}s, {len(seg_frames)} frames)")
            
            # Get last frame for next segment's Qwen edit
            nxt = _last_frame_of(seg_path)
            if nxt is not None:
                current_frame = nxt
            
            current_seed = random.randint(0, MAX_SEED)
        
        if not segment_paths:
            raise gr.Error("No video segments were produced.")
        
        # Assemble final video
        if len(segment_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                final_path = f.name
            concatenate_videos(segment_paths, final_path)
            for p in segment_paths:
                try: os.unlink(p)
                except OSError: pass
        else:
            final_path = segment_paths[0]
        
        # Optional audio
        if add_audio_cb and _MMAUDIO_AVAILABLE:
            try:
                final_path = add_audio_to_video(final_path, audio_prompt_tb, total_duration)
            except Exception as e:
                print(f"MMAudio error: {e}")
        
        # Rename to proper output path
        named_path = unique_output_path("vidgen_timeline", ".mp4")
        try:
            shutil.move(final_path, named_path)
            final_path = str(named_path)
        except Exception:
            pass
        
        print(f"🎬 Timeline done in {time.time() - started:.1f}s — {num_segments} segments")
        return final_path, final_path
    
    except gr.Error:
        raise
    except Exception as e:
        for p in segment_paths:
            try: os.unlink(p)
            except OSError: pass
        print(f"Timeline generation error: {e}")
        raise gr.Error(f"Timeline generation failed: {e}")


def _generate_timeline_with_durations(
    reference_image, prompts_json, durations_json, motion_prompt,
    resolution="480p", frame_multiplier=16, export_quality=7,
    seed=42, randomize_seed=True, add_audio_cb=False,
    audio_prompt_tb="natural ambient sound", negative_prompt=None,
    edit_steps=4, edit_guidance=1.0, flow_shift_auto=True,
    flow_shift=None, additional_refs=None,
):
    """
    Timeline video generation with the two-phase approach:
    
    Phase 1 — Generate keyframe images sequentially with Picgen (Qwen):
      Image 0 = user's reference photo
      Image 1 = Qwen edits Image 0 with Segment 1 prompt
      Image 2 = Qwen edits Image 1 with Segment 2 prompt
      ...
    
    Phase 2 — Generate video between each pair of keyframes with Vidgen (Wan):
      Video 1 = animate Image 0 → Image 1, motion = Segment 1 prompt
      Video 2 = animate Image 1 → Image 2, motion = Segment 2 prompt
      ...
    
    Phase 3 — Stitch all segment videos into one final video.
    """
    reference_image = _ensure_pil(reference_image)
    if reference_image is None:
        raise gr.Error("Please upload a reference photo.")
    
    segment_prompts = json.loads(prompts_json)
    segment_durations = json.loads(durations_json)
    
    if not segment_prompts:
        raise gr.Error("No segment prompts provided.")
    
    if not negative_prompt or not str(negative_prompt).strip():
        negative_prompt = default_negative_prompt
    
    num_segments = len(segment_prompts)
    total_duration = sum(segment_durations)
    
    # Flow shift
    if flow_shift_auto:
        if total_duration <= 6.0:
            flow_shift = 6.9
        elif total_duration <= 10.0:
            flow_shift = 5.5
        elif total_duration <= 20.0:
            flow_shift = 4.5
        else:
            flow_shift = 4.0
    else:
        if flow_shift is None:
            flow_shift = WAN_FLOW_SHIFT
    
    print(f"🎬 Timeline: {num_segments} segments, {total_duration:.1f}s total, flow_shift={flow_shift}")
    
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    started = time.time()
    
    # Process additional reference images
    additional_pil = None
    if additional_refs:
        additional_pil = []
        for ref_file in additional_refs:
            ref_path = ref_file.name if hasattr(ref_file, 'name') else str(ref_file)
            pil_ref = _ensure_pil(ref_path)
            if pil_ref:
                additional_pil.append(pil_ref)
    
    try:
        sized = resize_image_for_wan(reference_image, resolution)
        
        # Timeline guidance — strong enough to make visible changes
        timeline_guidance = max(float(edit_guidance), 3.0)
        timeline_steps = max(int(edit_steps), 6)
        
        # ==================================================================
        # PHASE 1: Generate all keyframe images with Picgen (Qwen)
        # ==================================================================
        print(f"📸 Phase 1: Generating {num_segments} keyframe images...")
        keyframes = [sized]  # Image 0 = user's input
        
        activate_pic()
        torch.cuda.set_device(PIC_DEVICE)
        
        for seg_idx in range(num_segments):
            seg_prompt = segment_prompts[seg_idx].strip()
            current_input = keyframes[-1]  # Always edit from the previous keyframe
            
            images_for_qwen = [current_input]
            if additional_pil:
                images_for_qwen.extend(additional_pil)
            
            print(f"  📸 Keyframe {seg_idx + 1}: {seg_prompt[:60]}...")
            
            _kf_ctx = torch.cuda.device(PIC_DEVICE) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
            _kf_ac = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
            with _kf_ctx:
                with _kf_ac:
                    result = pic_pipe(
                        image=images_for_qwen,
                        prompt=seg_prompt,
                        negative_prompt=" ",
                        num_inference_steps=timeline_steps,
                        true_cfg_scale=timeline_guidance,
                        generator=torch.Generator(device=PIC_DEVICE if _QWEN_CUDA_GENERATOR else "cpu").manual_seed(current_seed + seg_idx),
                    )
            new_keyframe = result.images[0]
            if new_keyframe.size != sized.size:
                new_keyframe = new_keyframe.resize(sized.size, Image.LANCZOS)
            
            keyframes.append(new_keyframe)
            print(f"  ✅ Keyframe {seg_idx + 1} generated")
        
        print(f"📸 Phase 1 complete: {len(keyframes)} keyframes (including input)")
        
        # ==================================================================
        # PHASE 2: Generate video between each pair of keyframes with Vidgen (Wan)
        # ==================================================================
        print(f"🎬 Phase 2: Generating {num_segments} video segments...")
        segment_paths = []
        
        for seg_idx in range(num_segments):
            seg_num = seg_idx + 1
            seg_prompt = segment_prompts[seg_idx].strip()
            seg_duration = float(segment_durations[seg_idx])
            seg_duration = max(0.5, min(6.0, seg_duration))
            seg_seed = current_seed + seg_idx * 100
            
            first_frame = keyframes[seg_idx]      # Start image
            last_frame = keyframes[seg_idx + 1]   # End image (target)
            
            print(f"  🎬 Seg {seg_num}/{num_segments} ({seg_duration}s): {seg_prompt[:50]}...")
            
            # Wan animates first_frame → last_frame with segment prompt as motion
            num_frames = get_num_frames(seg_duration)
            
            raw_frames = animate_frame(
                first_frame, last_frame, seg_prompt, negative_prompt,
                num_frames, seg_seed, flow_shift,
            )
            
            # RIFE interpolation
            factor = max(1, int(frame_multiplier) // FIXED_FPS)
            if factor > 1:
                seg_frames = interpolate_bits(raw_frames, multiplier=factor)
            else:
                seg_frames = list(raw_frames)
            seg_fps = FIXED_FPS * factor
            
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                seg_path = f.name
            export_to_video(seg_frames, seg_path, fps=seg_fps, quality=int(export_quality))
            segment_paths.append(seg_path)
            
            print(f"  ✅ Seg {seg_num} done ({seg_duration:.1f}s, {len(seg_frames)} frames)")
        
        # ==================================================================
        # PHASE 3: Stitch all videos together
        # ==================================================================
        print(f"🎬 Phase 3: Stitching {len(segment_paths)} segments...")
        
        if not segment_paths:
            raise gr.Error("No video segments were produced.")
        
        if len(segment_paths) > 1:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                final_path = f.name
            concatenate_videos(segment_paths, final_path)
            for p in segment_paths:
                try: os.unlink(p)
                except OSError: pass
        else:
            final_path = segment_paths[0]
        
        # Audio
        if add_audio_cb and _MMAUDIO_AVAILABLE:
            try:
                final_path = add_audio_to_video(final_path, audio_prompt_tb, total_duration)
            except Exception as e:
                print(f"MMAudio error: {e}")
        
        # Rename
        named_path = unique_output_path("vidgen_timeline", ".mp4")
        try:
            shutil.move(final_path, named_path)
            final_path = str(named_path)
        except Exception:
            pass
        
        print(f"🎬 Timeline done in {time.time() - started:.1f}s — {num_segments} segments, {total_duration:.1f}s")
        return final_path, final_path
    
    except gr.Error:
        raise
    except Exception as e:
        for p in segment_paths:
            try: os.unlink(p)
            except OSError: pass
        raise gr.Error(f"Timeline generation failed: {e}")


# ---------------------------------------------------------------------------
# PICGEN MODEL (Qwen Image Edit)
# ---------------------------------------------------------------------------

PICGEN_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BASE_MODEL_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")

# 🚀 PRIMARY MODEL LOADING


def _load_qwen_to_cpu():
    """
    Load Qwen to CPU with NSFW weights merged.
    Returns a fresh pipeline ready for enable_sequential_cpu_offload().
    """
    model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
    repo = BASE_MODEL_LOCAL_PATH if os.path.exists(model_index_path) else "Qwen/Qwen-Image-Edit-2511"
    kwargs = {"local_files_only": True} if os.path.exists(model_index_path) else {}

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        **kwargs,
    )

    v23 = NSFW_WEIGHTS_LOCAL_PATH
    if not os.path.exists(v23):
        os.makedirs(os.path.dirname(v23), exist_ok=True)
        v23 = hf_hub_download(
            repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
            filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
            cache_dir=PICGEN_MODELS_DIR,
            local_dir=os.path.join(PICGEN_MODELS_DIR, "rapid-aio"),
        )

    sd = load_file(v23, device="cpu")
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

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # NOTE: SageAttention + fp16 VAE decode were both briefly enabled here to
    # match Wan, but both were only ever validated against Wan's
    # model/hardware combo and produced all-black output on Qwen. Testing
    # them one at a time on real hardware to isolate which (if either) is
    # actually safe for Qwen — see TESTING NOTE below for current state.
    #
    # - SageAttention (enable_sage): apply_sage_attention() patches whatever
    #   pipe.transformer it's given — on Qwen that runs sageattn() against
    #   Qwen's Q/K/V tensor layout, which has never been verified
    #   (tensor_layout HND vs NHD, GQA handling). It also sits inside the
    #   denoising loop, so a subtle mismatch could degrade prompt adherence
    #   without an obvious crash. Left OFF until independently verified.
    #
    # - fp16 VAE decode (enable_vae_fp16): TESTING NOW. Runs only after
    #   denoising is done (pure latent-to-pixel conv net), so it cannot
    #   affect prompt adherence — it's decode-only. Failure mode is binary
    #   and obvious (all-black on overflow, fine otherwise), unlike Sage's
    #   silent-degradation risk. Turned ON here to test in isolation with
    #   Sage OFF — if generations come back correct, keep it; if black,
    #   revert this one flag back to False.
    #
    # torch.compile is deliberately OFF for Qwen. Wan pre-buckets every input
    # to one of a small set of fixed resolutions (resize_image_for_wan), so
    # torch.compile only ever compiles ~2 shapes, once. Qwen picgen takes
    # whatever dimensions the user uploads, unbucketed — every differently
    # sized photo is a new shape, forcing Dynamo to recompile (~20-30s) on
    # close to every generation instead of once. That's a net slowdown, not
    # a speedup, and it's what was showing up as the "Inductor Compilation"
    # progress-bar spam instead of clean diffusion steps.
    apply_all_optimizations(
        pipe, "Qwen (picgen)",
        enable_compile=False,
        enable_teacache=False,
        enable_sage=False,
        enable_vae_fp16=True,
    )
    return pipe
if DUAL_GPU:
    # CONCURRENT MODE — either two dedicated GPUs or one 95 GB Blackwell
    if _SINGLE_CARD_CONCURRENT:
        print(f"🚀 SINGLE-CARD CONCURRENT: Loading Wan fully to {WAN_DEVICE}, Qwen with model-offload...")
    else:
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
                low_cpu_mem_usage=True,
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

        if _SINGLE_CARD_CONCURRENT:
            # enable_model_cpu_offload: streams each top-level submodule (text_encoder,
            # transformer, vae) onto cuda:0 only while it is executing, then moves it
            # back to CPU immediately. Peak resident VRAM = largest single submodule
            # (~20 GB transformer) rather than all 35 GB at once. Wan's 57 GB stays
            # pinned at all times. Total peak: ~57 + ~20 = ~77 GB — fits 95 GB easily
            # IN THEORY. In practice the exact Wan checkpoint size varies (custom
            # merges, driver/CUDA-context overhead, etc.), so rather than trust that
            # math blindly we measure the real free VRAM left after Wan is pinned and
            # only use the faster whole-submodule offload if there's real headroom.
            # Otherwise fall back to enable_sequential_cpu_offload (layer-by-layer,
            # ~2-4 GB peak, slower but immune to this OOM) so the app stays usable
            # instead of crashing on the first Photo Editor generation.
            global _QWEN_CUDA_GENERATOR
            torch.cuda.synchronize(WAN_DEVICE)
            torch.cuda.empty_cache()
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            _free_gb = _free_bytes / (1024 ** 3)
            _QWEN_SUBMODULE_SAFETY_GB = 30  # largest Qwen submodule (~20-25 GB) + margin
            if _free_gb >= _QWEN_SUBMODULE_SAFETY_GB:
                pipe.enable_model_cpu_offload(gpu_id=0)
                _QWEN_CUDA_GENERATOR = True
                print(f"✅ Qwen ready with model-cpu-offload on cuda:0 in {time.time()-t:.1f}s "
                      f"({_free_gb:.1f} GB free after Wan)")
            else:
                pipe.enable_sequential_cpu_offload(gpu_id=0)
                _QWEN_CUDA_GENERATOR = False
                print(f"⚠️  Only {_free_gb:.1f} GB free after Wan (< {_QWEN_SUBMODULE_SAFETY_GB} GB safety "
                      f"margin) — Qwen falling back to sequential-cpu-offload (slower, OOM-safe) "
                      f"in {time.time()-t:.1f}s")
        else:
            # Two GPUs: pin Qwen fully to its dedicated card for maximum throughput
            pipe.to(PIC_DEVICE)
            print(f"✅ Qwen ready on {PIC_DEVICE} in {time.time()-t:.1f}s")

        # Same optimization stack as _load_qwen_to_cpu() — SageAttention OFF,
        # fp16 VAE decode ON for isolated testing; see the TESTING NOTE in
        # that function for why (decode-only, can't affect prompt adherence,
        # binary black/fine failure mode). torch.compile stays OFF for the
        # unbucketed-input-shape reason noted there as well. This pipe is
        # built inline here rather than via _load_qwen_to_cpu(), so it needs
        # its own call.
        apply_all_optimizations(
            pipe, "Qwen (picgen)",
            enable_compile=False,
            enable_teacache=False,
            enable_sage=False,
            enable_vae_fp16=True,
        )
        pic_pipe = pipe

    if _SINGLE_CARD_CONCURRENT:
        # On a single card, load Wan first (it pins 57 GB), then Qwen.
        # Loading both in parallel would cause a transient OOM during weight init.
        _load_wan_thread()
        _load_qwen_thread()
    else:
        t_wan = threading.Thread(target=_load_wan_thread, daemon=False)
        t_qwen = threading.Thread(target=_load_qwen_thread, daemon=False)
        t_wan.start()
        t_qwen.start()
        t_wan.join()
        t_qwen.join()

    _active_model = "both"
    if _SINGLE_CARD_CONCURRENT:
        print(f"✅ SINGLE-CARD CONCURRENT READY — Wan pinned, Qwen offloaded (both on cuda:0)")
    else:
        print(f"✅ DUAL GPU READY — Vidgen on {WAN_DEVICE}, Picgen on {PIC_DEVICE}")

    # Apply inference optimizations to both pipelines
    print("⚡ Applying inference optimizations...")
    _wan_opt = apply_all_optimizations(
        wan_pipe, "WAN (vidgen)",
        enable_compile=True,
        enable_teacache=True, teacache_thresh=0.05,
        enable_sage=True,
        enable_vae_fp16=True,
    )

elif STARTUP_MODE == "vidgen":
    print("🚀 VIDGEN DEFAULT: Loading Wan to GPU first for immediate use...")
    start_primary = time.time()
    _build_wan_pipeline("cpu")
    # Swap mode: Wan gets the WHOLE card while it's active (Qwen is fully
    # parked on CPU), so pick the fastest tier that actually fits instead of
    # always defaulting to the slowest layer-by-layer offload.
    _apply_offload_tier(wan_pipe, "wan", 0, full_weight_gb=60, submodule_gb=32)
    _active_model = "wan"
    print(f"✅ WAN READY in {time.time()-start_primary:.1f}s - Vidgen functional!")
    apply_all_optimizations(
        wan_pipe, "WAN (vidgen)",
        enable_compile=True,
        enable_teacache=True, teacache_thresh=0.05,
        enable_sage=True,
        enable_vae_fp16=True,
    )

    # Load Qwen to CPU in background so first tab-switch is fast
    pic_pipe = None
    def _bg_qwen_load():
        global pic_pipe
        print("📦 Background: Loading Qwen to CPU...")
        t = time.time()
        pic_pipe = _load_qwen_to_cpu()
        print(f"✅ Qwen on CPU in {time.time()-t:.1f}s — tab switching ready!")
    threading.Thread(target=_bg_qwen_load, daemon=True).start()

else:
    # PICGEN MODE
    print("🚀 PICGEN MODE: Loading Qwen to GPU first for immediate use...")
    start_qwen = time.time()
    pic_pipe = _load_qwen_to_cpu()
    # Swap mode: Qwen gets the WHOLE card while it's active (Wan is fully
    # parked on CPU) — same reasoning as the Wan branch above.
    _qwen_tier = _apply_offload_tier(pic_pipe, "pic", 0, full_weight_gb=42, submodule_gb=26)
    # Only "sequential" routes activations through CPU; "full" and
    # "model_offload" keep the active submodule resident on CUDA, so the
    # generator can (and for correctness, must) live on CUDA too.
    _QWEN_CUDA_GENERATOR = (_qwen_tier != "sequential")
    _active_model = "pic"
    print(f"✅ QWEN READY in {time.time()-start_qwen:.1f}s - Picgen functional!")

    # Load WAN to CPU in background so first tab-switch is fast
    def _bg_wan_load():
        print("📦 Background: Loading Wan to CPU...")
        t = time.time()
        _build_wan_pipeline("cpu")
        print(f"✅ Wan on CPU in {time.time()-t:.1f}s — tab switching ready!")
    threading.Thread(target=_bg_wan_load, daemon=True).start()

_swap_lock = threading.Lock()
# Set True once activate_wan() drops pic_pipe for the first time. Lets
# activate_pic() tell "still doing the one-time startup background load"
# apart from "was loaded, then evicted by a swap" — the latter needs a
# fresh rebuild, not another wait on a load that already finished.
_qwen_dropped = False


def activate_wan():
    """Ensure Wan is active on GPU."""
    global _active_model, pic_pipe, wan_pipe, _wan_loaded, _qwen_dropped

    if DUAL_GPU:
        # Concurrent mode — Wan is always GPU-resident (fully pinned).
        # On a single 95 GB card Qwen uses enable_model_cpu_offload so it
        # never permanently occupies Wan's VRAM; no swap is ever needed.
        if not _wan_loaded or wan_pipe is None:
            _load_wan(WAN_DEVICE)
        return

    if _active_model == "wan":
        return

    print("🔄 Swapping to Wan...")
    t = time.time()

    with _swap_lock:
        if _active_model == "wan":
            return

        # Drop Qwen from GPU: delete the offload-hooked pipeline and free memory.
        # enable_sequential_cpu_offload() hooks can't be undone — deletion is the
        # only clean path.
        if pic_pipe is not None:
            pic_pipe = None
            _qwen_dropped = True
            gc.collect()
            torch.cuda.empty_cache()

        # Load WAN to CPU (weights already cached — fast), then stream to GPU.
        if not _wan_loaded or wan_pipe is None:
            _build_wan_pipeline("cpu")
            # Apply optimizations to the freshly loaded pipeline.
            # (If the pipeline was already built and cached, optimizations were
            # applied at build time and survive in the CPU-side weights.)
            apply_all_optimizations(
                wan_pipe, "WAN (swap-in)",
                enable_compile=True,
                enable_teacache=True, teacache_thresh=0.05,
                enable_sage=True,
                enable_vae_fp16=True,
            )
        # Whole card is free for Wan right now (Qwen just got dropped above) —
        # use the fastest tier that fits rather than always the slowest one.
        _apply_offload_tier(wan_pipe, "wan", 0, full_weight_gb=60, submodule_gb=32)

        _active_model = "wan"
        print(f"🎯 Wan active in {time.time()-t:.1f}s")


def activate_pic():
    """Ensure Qwen is active on GPU."""
    global _active_model, pic_pipe, wan_pipe, _wan_loaded, _qwen_dropped, _QWEN_CUDA_GENERATOR

    if DUAL_GPU:
        # Concurrent mode — Qwen is always ready.
        # On a single 95 GB card its enable_model_cpu_offload hooks stream
        # submodules to GPU only during inference and free them immediately
        # after, so no manual management is needed here.
        if pic_pipe is None:
            raise RuntimeError("Qwen pipeline not loaded.")
        return

    if _active_model == "pic":
        return

    # pic_pipe is None either because the one-time startup background load
    # hasn't finished yet, or because a previous activate_wan() call dropped
    # it (it always deletes the offload-hooked pipeline outright — those
    # hooks can't be undone in place). Only the first case should wait;
    # the second needs a fresh rebuild, since nothing else will ever
    # repopulate pic_pipe on its own.
    if pic_pipe is None:
        if _qwen_dropped:
            print("🔁 Rebuilding Qwen (was dropped on a previous swap)...")
            rebuild_start = time.time()
            pic_pipe = _load_qwen_to_cpu()
            print(f"✅ Qwen rebuilt in {time.time()-rebuild_start:.1f}s")
        else:
            print("⏳ Waiting for Qwen background load...")
            wait_start = time.time()
            while pic_pipe is None:
                time.sleep(0.5)
                if time.time() - wait_start > 300:
                    raise RuntimeError("Qwen failed to load within 300 seconds")
            print("✅ Qwen background load complete")

    print("🔄 Swapping to Qwen...")
    t = time.time()

    with _swap_lock:
        if _active_model == "pic":
            return

        # Drop WAN from GPU: same pattern — delete and free.
        if wan_pipe is not None:
            wan_pipe = None
            _wan_loaded = False
            _wan_cache_clear()   # free GPU tensors held in text/image caches
            gc.collect()
            torch.cuda.empty_cache()

        # pic_pipe is already on CPU (loaded in startup, previous swap, or
        # just rebuilt above). Whole card is free for it right now (Wan just
        # got dropped) — use the fastest tier that fits.
        _qwen_tier = _apply_offload_tier(pic_pipe, "pic", 0, full_weight_gb=42, submodule_gb=26)
        _QWEN_CUDA_GENERATOR = (_qwen_tier != "sequential")
        _qwen_dropped = False

        _active_model = "pic"
        print(f"🎯 Qwen active in {time.time()-t:.1f}s")

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


STARTER_IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")


def _find_starter_image_path(starter_num):
    """Return (path, ext) for the first matching starter image file, or (None, None)."""
    starters_dir = os.path.join(SCRIPT_DIR, "starters")
    for ext in STARTER_IMAGE_EXTS:
        path = os.path.join(starters_dir, f"start{starter_num}.{ext}")
        if os.path.exists(path):
            return path, ext
    return None, None


def add_starter_image(starter_num):
    """Load a starter image (supports .jpg, .jpeg, .png, .webp) as a base64 data URI."""
    path, ext = _find_starter_image_path(starter_num)
    if not path:
        return ""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}[ext]
    return f"data:image/{mime};base64,{b64}"


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

    # Pin to PIC_DEVICE — prevents cross-device leakage in dual GPU mode.
    # Generator device must match Qwen's actual offload strategy, not just the
    # nominal GPU_MODE:
    # - enable_model_cpu_offload (whole submodules pinned during use): CUDA generator
    # - enable_sequential_cpu_offload (layer-by-layer, incl. single-card concurrent's
    #   low-VRAM fallback, and swap mode): activations route through CPU, so the
    #   generator must be CPU or diffusers raises a device mismatch.
    torch.cuda.set_device(PIC_DEVICE)
    _pic_gen_device = PIC_DEVICE if _QWEN_CUDA_GENERATOR else "cpu"
    generator = torch.Generator(device=_pic_gen_device).manual_seed(seed)
    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image.")
    _t_decoded = time.time()

    if height == 256 and width == 256:
        height, width = None, None

    print(f"Prompt: '{prompt}' | Seed: {seed} | Steps: {num_inference_steps}")
    print(f"  input images: {[im.size for im in pil_images]}")

    # ---- Prompt embedding cache -------------------------------------------
    # The previous approach monkey-patched pic_pipe.encode_prompt so it could
    # return cached values. The problem: accelerate's sequential offload hooks
    # fire at pre_forward time, BEFORE encode_prompt runs, moving the
    # text_encoder to GPU regardless of whether we'd return early. That's what
    # caused the 14.5 GB + 1.02 GB OOM.
    #
    # Fix: call encode_prompt once here (outside the pipeline), cache the
    # result tensor, and on subsequent calls pass prompt_embeds directly as a
    # pipeline kwarg — which causes the pipeline to skip the encode_prompt call
    # entirely, so the offload hook for text_encoder never fires.
    cached_embeds = _get_cached_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt)

    if cached_embeds is not None:
        print(f"  timing: activate {_t_active - _t_enter:.2f}s, "
              f"decode {_t_decoded - _t_active:.2f}s, embeds: cache hit "
              f"(active model: {_active_model}, dual_gpu: {DUAL_GPU}, qwen_cuda_gen: {_QWEN_CUDA_GENERATOR})")
        pic_kwargs = dict(
            image=pil_images,
            prompt_embeds=cached_embeds["prompt_embeds"],
            prompt_embeds_mask=cached_embeds["prompt_embeds_mask"],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )
    else:
        print(f"  timing: activate {_t_active - _t_enter:.2f}s, "
              f"decode {_t_decoded - _t_active:.2f}s, embeds: computing "
              f"(active model: {_active_model}, dual_gpu: {DUAL_GPU}, qwen_cuda_gen: {_QWEN_CUDA_GENERATOR})")
        pic_kwargs = dict(
            image=pil_images,
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )

    _t_pipe = time.time()

    _pic_ctx2 = torch.cuda.device(PIC_DEVICE) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
    _pic_autocast2 = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if _QWEN_CUDA_GENERATOR else contextlib.nullcontext()
    with _pic_ctx2:
        with _pic_autocast2:
            result = pic_pipe(**pic_kwargs)
            image = result.images

    # Cache the embeddings for next time if this was a fresh encode.
    # QwenImageEditPlusPipeline stores the last-computed embeds on the result
    # under various attribute names depending on diffusers version — try them.
    if cached_embeds is None:
        try:
            pe = getattr(result, 'prompt_embeds', None)
            pe_mask = getattr(result, 'prompt_embeds_mask', None)
            if pe is not None and pe_mask is not None:
                _cache_prompt_embeds(
                    prompt, negative_prompt, pil_images, num_images_per_prompt,
                    {"prompt_embeds": pe, "prompt_embeds_mask": pe_mask},
                )
            else:
                # Encode directly and cache; runs only once per unique prompt.
                with torch.no_grad(), torch.cuda.device(PIC_DEVICE):
                    enc_result = pic_pipe.encode_prompt(
                        prompt=prompt,
                        image=pil_images,
                        device=PIC_DEVICE if _QWEN_CUDA_GENERATOR else "cpu",
                        num_images_per_prompt=num_images_per_prompt,
                    )
                if isinstance(enc_result, (list, tuple)) and len(enc_result) >= 2:
                    _cache_prompt_embeds(
                        prompt, negative_prompt, pil_images, num_images_per_prompt,
                        {"prompt_embeds": enc_result[0], "prompt_embeds_mask": enc_result[1]},
                    )
        except Exception as _ce:
            # Non-fatal — cache miss just means next call re-encodes
            print(f"  embed cache fill skipped (non-fatal): {_ce}")

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
                        label="Reference Photo(s)",
                        type="filepath",
                        elem_id="vidgen-reference",
                    )
                    additional_refs = gr.File(
                        label="Additional Reference Images (for Custom/Replace mode)",
                        file_count="multiple",
                        file_types=["image"],
                        visible=False,
                    )
                    vid_prompt = gr.Textbox(
                        label="Motion & Scene Prompt",
                        value="",
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
                            "Keep = animated untouched. Replace = new environment. "
                            "Custom = your edit instruction. Timeline = per-segment prompts."
                        ),
                    )
                    edit_instruction = gr.Textbox(
                        label="Custom Edit Instruction",
                        value="",
                        lines=2,
                        visible=False,
                        placeholder="e.g. remove all clothing, keep everything else identical",
                    )
                    edit_instruction_info = gr.Markdown(
                        "**Custom Edit** tells Qwen how to modify the photo *before* animation. "
                        "Example: `remove all clothing` or `add a red dress`. "
                        "The Motion Prompt above separately tells WAN how to *animate* the result.",
                        visible=False,
                    )
                    
                    # Timeline mode UI — 10 segment slots with custom durations
                    timeline_section = gr.Column(visible=False)
                    with timeline_section:
                        gr.Markdown(
                            "**Timeline:** Each segment prompt is used as both the image edit instruction AND the video motion prompt. "
                            "Empty segments are skipped. Fill in order."
                        )
                        timeline_prompts = []
                        timeline_durations = []
                        for i in range(1, 11):
                            with gr.Row():
                                tb = gr.Textbox(
                                    label=f"Seg {i}",
                                    placeholder=f"What happens in segment {i} (leave empty to skip)",
                                    lines=1,
                                    scale=4,
                                )
                                dur = gr.Slider(
                                    0.5, 6.0, value=3.5, step=0.5,
                                    label="Sec",
                                    scale=1,
                                )
                            timeline_prompts.append(tb)
                            timeline_durations.append(dur)
                        timeline_generate_btn = gr.Button("🎬 Generate Timeline Video", variant="primary", size="lg")
                    
                    # Show/hide sections based on scene_mode
                    def _update_scene_visibility(mode):
                        is_timeline = (mode == MODE_TIMELINE)
                        return (
                            gr.update(visible=(mode == MODE_CUSTOM)),           # edit_instruction
                            gr.update(visible=(mode == MODE_CUSTOM)),           # edit_instruction_info
                            gr.update(visible=(mode != MODE_KEEP and not is_timeline)),  # additional_refs
                            gr.update(visible=is_timeline),                     # timeline_section
                            gr.update(visible=not is_timeline),                 # vid_prompt
                            gr.update(visible=not is_timeline),                 # vid_negative_prompt
                            gr.update(visible=not is_timeline),                 # duration_row
                        )

                    duration_row = gr.Row()
                    with duration_row:
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
                    
                    # Connect scene_mode change AFTER duration_row is defined
                    scene_mode.change(
                        fn=_update_scene_visibility,
                        inputs=[scene_mode],
                        outputs=[edit_instruction, edit_instruction_info, additional_refs, timeline_section, vid_prompt, vid_negative_prompt, duration_row],
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
                    generate_btn = gr.Button(
                        "🎬 Generate Video", variant="primary", size="lg"
                    )
                    vid_clear_storage_btn = gr.Button(
                        "🗑 Clear Storage", variant="secondary", size="sm"
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
                    
                    end_image = gr.Image(
                        label="End Frame (optional, first segment only)",
                        type="pil",
                    )
            # Hidden file component — populated by generate_btn and used for frame extraction
            video_file = gr.File(visible=False)

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

            def _noop_download(f):
                return f

            generate_btn.click(
                fn=generate_video,
                inputs=[
                    reference_image, vid_prompt, scene_mode, edit_instruction,
                    additional_refs, end_image, duration_seconds, resolution, frame_multiplier,
                    export_quality, seed, randomize_seed, add_audio_cb,
                    audio_prompt_tb, vid_negative_prompt, edit_steps, edit_guidance,
                    flow_shift_auto, flow_shift,
                ],
                outputs=[video_output, video_file],
                concurrency_id=WAN_QUEUE_ID,
                concurrency_limit=10,
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js="""(file) => {
                    if (file) {
                        const url = file.url || (typeof file === 'string' ? file : (file.path || file.name || ''));
                        if (url) {
                            const a = document.createElement('a');
                            a.href = file.url ? file.url : '/file=' + url;
                            a.download = file.orig_name || 'vidgen.mp4';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                        }
                    }
                    setTimeout(() => {
                        const clearBtn = document.querySelector('#clear-storage-btn button, button#clear-storage-btn');
                        if (clearBtn) {
                            clearBtn.click();
                        } else {
                            const allBtns = document.querySelectorAll('button');
                            for (const btn of allBtns) {
                                if (btn.textContent.includes('Clear Storage')) { btn.click(); break; }
                            }
                        }
                    }, 1500);
                    return [file];
                }""",
            )

            vid_clear_storage_btn.click(
                fn=clear_storage,
                inputs=[],
                outputs=[clear_storage_status],
            )

            # Timeline generate handler
            def _timeline_generate(
                ref_image,
                p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                d1, d2, d3, d4, d5, d6, d7, d8, d9, d10,
                resolution_val, frame_mult, exp_quality,
                seed_val, rand_seed, audio_cb, audio_prompt,
                neg_prompt, e_steps, e_guidance,
                fs_auto, fs_val, add_refs,
                progress=gr.Progress(track_tqdm=True),
            ):
                """Collect filled segments and generate timeline video."""
                all_prompts = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
                all_durations = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
                segments = []
                for i in range(10):
                    p = (all_prompts[i] or "").strip()
                    if p:
                        d = float(all_durations[i]) if all_durations[i] else 3.5
                        d = max(0.5, min(6.0, d))
                        segments.append({"prompt": p, "duration": d})
                if not segments:
                    raise gr.Error("Please enter at least one segment prompt.")
                prompts_json = json.dumps([s["prompt"] for s in segments])
                durations_json = json.dumps([s["duration"] for s in segments])
                return _generate_timeline_with_durations(
                    ref_image, prompts_json, durations_json, None,
                    resolution_val, frame_mult, exp_quality,
                    seed_val, rand_seed, audio_cb, audio_prompt,
                    neg_prompt, e_steps, e_guidance,
                    fs_auto, fs_val, add_refs,
                )

            timeline_generate_btn.click(
                fn=_timeline_generate,
                inputs=[
                    reference_image,
                    *timeline_prompts,
                    *timeline_durations,
                    resolution, frame_multiplier, export_quality,
                    seed, randomize_seed, add_audio_cb, audio_prompt_tb,
                    vid_negative_prompt, edit_steps, edit_guidance,
                    flow_shift_auto, flow_shift, additional_refs,
                ],
                outputs=[video_output, video_file],
                concurrency_id=WAN_QUEUE_ID,
                concurrency_limit=10,
            ).then(
                fn=_noop_download,
                inputs=[video_file],
                outputs=[video_file],
                js="""(file) => {
                    if (file) {
                        const url = file.url || (typeof file === 'string' ? file : (file.path || file.name || ''));
                        if (url) {
                            const a = document.createElement('a');
                            a.href = file.url ? file.url : '/file=' + url;
                            a.download = file.orig_name || 'vidgen_timeline.mp4';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                        }
                    }
                    setTimeout(() => {
                        const clearBtn = document.querySelector('#clear-storage-btn button, button#clear-storage-btn');
                        if (clearBtn) {
                            clearBtn.click();
                        } else {
                            const allBtns = document.querySelectorAll('button');
                            for (const btn of allBtns) {
                                if (btn.textContent.includes('Clear Storage')) { btn.click(); break; }
                            }
                        }
                    }, 1500);
                    return [file];
                }""",
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

            # Download Frame — extracts frame then triggers browser download via JS
            download_frame_output = gr.File(visible=False, elem_id="download-frame-file")
            
            download_frame_btn.click(
                fn=get_frame_as_file,
                inputs=[video_file, frame_time_input],
                outputs=[download_frame_output],
            ).then(
                fn=None,
                inputs=[download_frame_output],
                outputs=[],
                js="""(file) => {
                    if (file && file.url) {
                        const a = document.createElement('a');
                        a.href = file.url;
                        a.download = file.orig_name || 'frame.jpg';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    } else if (file) {
                        // Try Gradio file path format
                        const url = typeof file === 'string' ? file : file.path || file.name || file;
                        const a = document.createElement('a');
                        a.href = '/file=' + url;
                        a.download = 'frame.jpg';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }
                }""",
            )

        # ------------------------------------------------------------------ #
        #  TAB 2 — PHOTO EDITOR (picgen)                                      #
        # ------------------------------------------------------------------ #
        with gr.Tab("🖼️ Photo Editor", id="picgen-tab"):
            with gr.Column(elem_id="col-container"):

                # Starter image thumbnails — loaded dynamically on each page load
                # so images added after app startup appear without a restart.
                def _get_starter_thumbnails_html():
                    """
                    Build thumbnail HTML once at page construction. Images are embedded
                    as base64 data URIs (reliable — no /file= path/whitelisting issues,
                    no endpoint round-trip). Thumbnails are clickable — clicking one
                    fires the corresponding numbered button underneath it.
                    """
                    starters_dir = os.path.join(SCRIPT_DIR, "starters")
                    html_parts = []
                    found_any = False
                    for i in range(1, 11):
                        data_url = add_starter_image(i)
                        if data_url:
                            html_parts.append(
                                f'<div style="flex:1;text-align:center;min-width:0;cursor:pointer;" '
                                f'title="Starter {i}" '
                                f'onclick="(function(){{var btns=document.querySelectorAll(\'#starter-btn-row button\');if(btns[{i-1}])btns[{i-1}].click();}})()">'
                                f'<img src="{data_url}" style="width:100%;aspect-ratio:1;object-fit:cover;'
                                f'border:1px solid var(--border-color-primary);border-radius:4px;'
                                f'background:var(--background-fill-secondary);" />'
                                f'</div>'
                            )
                            found_any = True
                        else:
                            html_parts.append(
                                f'<div style="flex:1;min-width:0;text-align:center;'
                                f'font-size:10px;color:var(--body-text-color-subdued);">{i}</div>'
                            )
                    if not found_any:
                        return (f'<div style="color:var(--body-text-color-subdued);font-size:12px;padding:4px 0;">'
                                f'No starter images found in {starters_dir}</div>')
                    return '<div style="display:flex;gap:4px;padding:4px 0;">' + "".join(html_parts) + "</div>"

                starter_thumbnails_html = gr.HTML(_get_starter_thumbnails_html())

                # Real clickable buttons — aligned under each thumbnail
                with gr.Row(elem_id="starter-btn-row"):
                    starter_btns = [gr.Button(str(i), size="sm", min_width=30) for i in range(1, 11)]

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
                        use_as_next_seg_btn = gr.Button("📎 Use As Next Segment", variant="secondary", size="sm")

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

                # Each hidden button directly calls add_starter_image (like working backup)
                for i, btn in enumerate(starter_btns, start=1):
                    btn.click(fn=lambda num=i: add_starter_image(num), inputs=[], outputs=[starter_b64_output])

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

            # ==================================================================
            # PROGRESSION MODE — Timeline at bottom of picgen tab
            # ==================================================================
            with gr.Accordion("🎬 Progression Mode (Video Timeline)", open=False):
                gr.Markdown(
                    "Build a video timeline by adding images and prompts for each segment. "
                    "Use the editor above to generate images, then click **📎 Use As Next Segment** to add them here. "
                    "Each segment animates from its image to the next segment's image using the prompt as motion."
                )
                
                # 10 segment slots — each with image, prompt, duration, and move buttons
                prog_images = []
                prog_prompts = []
                prog_durations = []
                prog_up_btns = []
                prog_down_btns = []
                
                for i in range(1, 11):
                    with gr.Row():
                        with gr.Column(scale=0, min_width=50):
                            up_btn = gr.Button("↑", size="sm", min_width=30)
                            down_btn = gr.Button("↓", size="sm", min_width=30)
                        img = gr.Image(
                            label=f"Seg {i}",
                            type="filepath",
                            scale=1,
                            height=100,
                        )
                        prompt = gr.Textbox(
                            label=f"Motion/Action",
                            placeholder=f"What happens in segment {i}",
                            lines=2,
                            scale=2,
                        )
                        dur = gr.Slider(
                            0.5, 6.0, value=3.5, step=0.5,
                            label="Sec",
                            scale=0,
                            min_width=80,
                        )
                    prog_images.append(img)
                    prog_prompts.append(prompt)
                    prog_durations.append(dur)
                    prog_up_btns.append(up_btn)
                    prog_down_btns.append(down_btn)
                
                prog_generate_btn = gr.Button("🎬 Generate Progression Video", variant="primary", size="lg")
                
                # --- Reorder handlers (swap segments up/down) ---
                def _swap_segments(idx_to_move, direction, *all_values):
                    """Swap two adjacent segments. Returns all values reordered."""
                    # all_values = 10 images + 10 prompts + 10 durations = 30 values
                    images = list(all_values[:10])
                    prompts = list(all_values[10:20])
                    durations = list(all_values[20:30])
                    
                    idx = int(idx_to_move)
                    target = idx + (-1 if direction == "up" else 1)
                    
                    if 0 <= target <= 9:
                        images[idx], images[target] = images[target], images[idx]
                        prompts[idx], prompts[target] = prompts[target], prompts[idx]
                        durations[idx], durations[target] = durations[target], durations[idx]
                    
                    return images + prompts + durations
                
                all_prog_components = prog_images + prog_prompts + prog_durations
                
                for i in range(10):
                    prog_up_btns[i].click(
                        fn=lambda *vals, idx=i: _swap_segments(idx, "up", *vals),
                        inputs=all_prog_components,
                        outputs=all_prog_components,
                    )
                    prog_down_btns[i].click(
                        fn=lambda *vals, idx=i: _swap_segments(idx, "down", *vals),
                        inputs=all_prog_components,
                        outputs=all_prog_components,
                    )
                
                # --- "Use As Next Segment" handler ---
                def _add_to_next_segment(gallery_images, *current_images):
                    """Add the first result image to the next empty segment slot."""
                    if not gallery_images:
                        raise gr.Error("No images in result gallery.")
                    
                    # Get the first image from gallery
                    first_item = gallery_images[0]
                    if isinstance(first_item, (list, tuple)):
                        img_path = first_item[0]
                    elif isinstance(first_item, dict):
                        img_path = first_item.get("name") or first_item.get("path")
                    else:
                        img_path = first_item
                    
                    # Find next empty slot
                    images_list = list(current_images)
                    for idx in range(10):
                        if not images_list[idx]:
                            images_list[idx] = img_path
                            print(f"📎 Added image to segment {idx + 1}")
                            return images_list
                    
                    raise gr.Error("All 10 segment slots are full.")
                
                use_as_next_seg_btn.click(
                    fn=_add_to_next_segment,
                    inputs=[pic_result] + prog_images,
                    outputs=prog_images,
                )
                
                # --- Generate Progression Video handler ---
                def _generate_progression(
                    *all_inputs,
                    progress=gr.Progress(track_tqdm=True),
                ):
                    """
                    Generate video from progression timeline.
                    Each segment animates from its image to the next segment's image.
                    """
                    # Parse inputs: 10 images + 10 prompts + 10 durations + settings
                    images = list(all_inputs[:10])
                    prompts = list(all_inputs[10:20])
                    durations = list(all_inputs[20:30])
                    resolution_val, frame_mult, exp_quality = all_inputs[30], all_inputs[31], all_inputs[32]
                    seed_val, rand_seed = all_inputs[33], all_inputs[34]
                    audio_cb, audio_prompt = all_inputs[35], all_inputs[36]
                    neg_prompt = all_inputs[37]
                    fs_auto, fs_val = all_inputs[38], all_inputs[39]
                    
                    # Collect segments that have both an image and a prompt
                    segments = []
                    for i in range(10):
                        img = images[i]
                        prompt = (prompts[i] or "").strip()
                        if img and prompt:
                            dur = float(durations[i]) if durations[i] else 3.5
                            dur = max(0.5, min(6.0, dur))
                            segments.append({"image": img, "prompt": prompt, "duration": dur})
                    
                    if len(segments) < 1:
                        raise gr.Error("Need at least 1 segment with both an image and a prompt.")
                    
                    # Build the video: animate from each segment's image to the next
                    if not neg_prompt or not str(neg_prompt).strip():
                        neg_prompt = default_negative_prompt
                    
                    total_duration = sum(s["duration"] for s in segments)
                    
                    # Flow shift
                    if fs_auto:
                        if total_duration <= 6.0: flow_shift = 6.9
                        elif total_duration <= 10.0: flow_shift = 5.5
                        elif total_duration <= 20.0: flow_shift = 4.5
                        else: flow_shift = 4.0
                    else:
                        flow_shift = float(fs_val) if fs_val else WAN_FLOW_SHIFT
                    
                    current_seed = random.randint(0, MAX_SEED) if rand_seed else int(seed_val)
                    started = time.time()
                    segment_paths = []
                    
                    print(f"🎬 Progression: {len(segments)} segments, {total_duration:.1f}s total")
                    
                    try:
                        for seg_idx in range(len(segments)):
                            seg = segments[seg_idx]
                            seg_num = seg_idx + 1
                            
                            # Load and resize the start frame
                            start_img = _ensure_pil(seg["image"])
                            if start_img is None:
                                raise gr.Error(f"Segment {seg_num}: could not load image.")
                            start_frame = resize_image_for_wan(start_img, resolution_val)
                            
                            # End frame = next segment's image (if exists)
                            end_frame = None
                            if seg_idx + 1 < len(segments):
                                end_img = _ensure_pil(segments[seg_idx + 1]["image"])
                                if end_img:
                                    end_frame = resize_and_crop_to_match(end_img, start_frame)
                            
                            # Animate
                            num_frames = get_num_frames(seg["duration"])
                            seg_seed = current_seed + seg_idx * 100
                            
                            print(f"  🎬 Seg {seg_num} ({seg['duration']}s): {seg['prompt'][:50]}...")
                            
                            raw_frames = animate_frame(
                                start_frame, end_frame, seg["prompt"], neg_prompt,
                                num_frames, seg_seed, flow_shift,
                            )
                            
                            # RIFE
                            factor = max(1, int(frame_mult) // FIXED_FPS)
                            if factor > 1:
                                seg_frames = interpolate_bits(raw_frames, multiplier=factor)
                            else:
                                seg_frames = list(raw_frames)
                            seg_fps = FIXED_FPS * factor
                            
                            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                                seg_path = f.name
                            export_to_video(seg_frames, seg_path, fps=seg_fps, quality=int(exp_quality))
                            segment_paths.append(seg_path)
                            print(f"  ✅ Seg {seg_num} done")
                        
                        # Stitch
                        if len(segment_paths) > 1:
                            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                                final_path = f.name
                            concatenate_videos(segment_paths, final_path)
                            for p in segment_paths:
                                try: os.unlink(p)
                                except OSError: pass
                        else:
                            final_path = segment_paths[0]
                        
                        # Audio
                        if audio_cb and _MMAUDIO_AVAILABLE:
                            try:
                                final_path = add_audio_to_video(final_path, audio_prompt, total_duration)
                            except Exception as e:
                                print(f"MMAudio error: {e}")
                        
                        # Rename
                        named_path = unique_output_path("vidgen_progression", ".mp4")
                        try:
                            shutil.move(final_path, named_path)
                            final_path = str(named_path)
                        except Exception:
                            pass
                        
                        print(f"🎬 Progression done in {time.time() - started:.1f}s — {len(segments)} segments")
                        return final_path
                    
                    except gr.Error:
                        raise
                    except Exception as e:
                        for p in segment_paths:
                            try: os.unlink(p)
                            except OSError: pass
                        raise gr.Error(f"Progression generation failed: {e}")
                
                # Video output for progression mode
                prog_video_output = gr.Video(label="Progression Video", interactive=False)
                
                prog_generate_btn.click(
                    fn=_generate_progression,
                    inputs=all_prog_components + [
                        resolution, frame_multiplier, export_quality,
                        seed, randomize_seed, add_audio_cb, audio_prompt_tb,
                        vid_negative_prompt, flow_shift_auto, flow_shift,
                    ],
                    outputs=[prog_video_output],
                    concurrency_id=WAN_QUEUE_ID,
                    concurrency_limit=10,
                )

    demo.load(fn=None, js=gallery_js)

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
        "concurrent": "Dual-GPU Concurrent Mode (Both Models GPU-Resident, no swapping)",
        "single": f"Swap Mode ({_gpu_count}-GPU — active model streamed to GPU, idle model parked on CPU)",
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
    # Rough estimates based on hardware AND mode — concurrent mode (>=2 GPUs,
    # both models pinned, no swap penalty) is meaningfully faster than swap
    # mode (one model streamed layer-by-layer via CPU offload at a time).
    if GPU_MODE == "concurrent":
        if _total_vram_mb >= 180000:  # 2x 95GB Blackwell
            print("  * Photo Editor (1 Image @ 4 Steps)       : ~3 - 5 seconds")
            print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~8 - 15 seconds")
        else:  # 2 smaller GPUs, each model still fully pinned to its own card
            print("  * Photo Editor (1 Image @ 4 Steps)       : ~5 - 8 seconds")
            print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~15 - 25 seconds")
    else:  # single — swap mode, one model active at a time via sequential offload
        if _total_vram_mb >= 80000:  # 1x 95GB Blackwell — plenty of headroom, still layer-streamed
            print("  * Photo Editor (1 Image @ 4 Steps)       : ~8 - 14 seconds")
            print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~20 - 35 seconds")
            print("    (Only the active tab's model is on GPU — the other is parked on CPU)")
        elif _total_vram_mb >= 40000:  # 2x 24GB
            print("  * Photo Editor (1 Image @ 4 Steps)       : ~8 - 12 seconds")
            print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~30 - 50 seconds")
        else:  # single 24GB or less
            print("  * Photo Editor (1 Image @ 4 Steps)       : ~10 - 15 seconds")
            print("  * Video Generator (3.5s Clip @ 16 FPS)   : ~45 - 80 seconds")
    print("=" * 80 + "\n")

    if DUAL_GPU:
        print(f"🚀 GRADIO LAUNCHING — DUAL-GPU CONCURRENT. Both tabs GPU-resident, no swapping.")
        # Each model queue allows 1 concurrent job (GPU is serialized per model).
        # Both queues run independently — Picgen and Vidgen do not block each other.
        demo.queue(default_concurrency_limit=1)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, str(OUTPUT_DIR), os.path.join(SCRIPT_DIR, "starters")],
        )
    else:
        if STARTUP_MODE == "vidgen":
            print("🚀 GRADIO LAUNCHING — Wan on GPU, vidgen ready immediately.")
        else:
            print("🚀 GRADIO LAUNCHING — Qwen on GPU, picgen ready immediately.")
        demo.queue(default_concurrency_limit=1)

        # Launch Gradio (blocks forever — background threads already started above)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            allowed_paths=[SCRIPT_DIR, str(OUTPUT_DIR), os.path.join(SCRIPT_DIR, "starters")],
        )