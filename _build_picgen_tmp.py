# -*- coding: utf-8 -*-
"""TEMP build script - deleted after use. Assembles picgen/app.py from app.py."""
import io, os, re

SRC = "app.py"
OUT = os.path.join("picgen", "app.py")
with io.open(SRC, "r", encoding="utf-8") as fh:
    _lines = fh.readlines()

def seg(a, b, dedent=False):
    text = "".join(_lines[a - 1 : b])
    if dedent:
        text = "\n".join((ln[4:] if ln.startswith("    ") else ln) for ln in text.split("\n"))
    return text

G1 = '''# ---------------------------------------------------------------------------
# STARTUP MODE
#
# This is the standalone PICGEN build (Qwen Image Edit photo app). The newgen
# app's combined vidgen/picgen startup-mode parser is intentionally NOT used
# here: there is no vidgen tab, no Wan pipeline, no MMAudio, no RIFE, and no
# lip-sync in this app -- only the Photo Editor.
# ---------------------------------------------------------------------------
STARTUP_MODE = "picgen"

'''

GPU_TAIL = '''GPU_PROFILE = _detect_gpu_profile(0)

GPU_VRAM_GB   = GPU_PROFILE["vram_gb"]
GPU_CC        = GPU_PROFILE["cc"]
GPU_NAME      = GPU_PROFILE["name"]
GPU_HIGH_VRAM = GPU_PROFILE["high_vram"]

# Single device serves the whole picgen app (Qwen pipeline only).
PIC_DEVICE = "cuda:0"
PIC_QUEUE_ID = "gpu"
device = torch.device(PIC_DEVICE)

'''

SMART_RESIDENCY = '''# Single-GPU smart residency (picgen build):
#   >=40 GB cards (RTX PRO 6000 Blackwell etc.): the full fp8 pipeline
#     (transformer ~19 GB + text encoder ~15.5 GB + VAE ~0.2 GB) fits
#     co-resident -- keep everything on the GPU, no offload hooks, fastest.
#   24-32 GB cards (RTX 3090 / RTX 4090 24 GB, RTX 5090 32 GB): diffusers
#     model-CPU-offload streams components on demand; the fp8 transformer
#     still stays resident for the whole denoise loop, so per-step speed
#     is unaffected and peak VRAM stays far below the limit.
if _device_total_vram_gb(PIC_DEVICE) >= FULL_RESIDENCY_VRAM_GB:
    _safe_move_to_device(pic_pipe, PIC_DEVICE)
    _offload_state["pic"] = False
    print(f"    [pic] full residency on {PIC_DEVICE} "
          f"({_device_total_vram_gb(PIC_DEVICE):.0f} GB card) -- no offload.")
else:
    _enable_pic_offload(pic_pipe)
'''

ACTIVATE_PIC = '''
def _activate_model(target):
    """Ensure the Qwen (picgen) pipeline is the single GPU-resident model.

    (picgen build) There is only one model in this process, so activation
    reduces to: place the Qwen pipeline on PIC_DEVICE via the shared offload
    logic -- full residency on >=40 GB cards, model-CPU-offload below that.
    The RLock still serialises activation against inference and cleanup so
    behaviour matches the newgen app exactly.
    """
    global _active_model
    with _gpu_op_lock:
        if target != "pic":
            return
        if _active_model == "pic" and pic_pipe is not None:
            return
        if pic_pipe is None:
            raise RuntimeError("Qwen pipeline has not finished loading")
        _enable_offload(pic_pipe, PIC_DEVICE, "pic")
        _active_model = "pic"


def activate_pic():
    """Ensure Qwen alone is fully resident and ready."""
    started = time.time()
    _activate_model("pic")
    if time.time() - started > 3:
        print(f" Qwen active in {time.time() - started:.1f}s")

'''

CLEANUP_GLUE = '''
def _full_gpu_cleanup(offload_pipelines=True):
    """Release caches and, optionally, all pipeline VRAM. (picgen build)"""
    global _active_model
    with _gpu_op_lock:
        _clear_picgen_cache()
        if offload_pipelines:
            if pic_pipe is not None:
                try:
                    _disable_offload(pic_pipe, "pic")
                except Exception as exc:
                    print(f"Cleanup warning (Qwen): {exc}")
            _active_model = None
        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


def _do_clear_storage():
    """
    Module-level storage clear helper called by infer_with_preclear() and the
    automatic post-generation clear. (picgen build: no vidgen player/sequence
    files to handle.)

    'Clearing storage' means:
      1. Releasing all transient _media_store entries (picgen images) so their
         RAM is freed.
      2. Cleaning up Gradio's session upload dir (tmp/gradio/) for uploaded
         input-image temp files, respecting the protection system so files
         still needed by a running generation are kept.

    Returns the count of released/deleted items.
    """
    import shutil as _shutil

    count = 0

    import glob as _glob
    _tmp = os.path.join(SCRIPT_DIR, "tmp", "gradio")
    import time as _time
    _now = _time.time()
    for _pat in (
        _tmp + "/picgen_*.png",
    ):
        for _f in _glob.glob(_pat):
            try:
                import os as _oss
                if _now - _oss.path.getmtime(_f) > 60:
                    _oss.unlink(_f); count += 1
            except Exception:
                pass
    _media_store_release_prefix("picgen_")

    for gradio_dir in [
        Path(SCRIPT_DIR) / "tmp" / "gradio",
    ]:
        if gradio_dir.exists():
            for item in gradio_dir.iterdir():
                if item.name == "vibe_edit_history":
                    continue
                if _is_protected(item):
                    continue
                try:
                    if item.is_dir():
                        _shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    count += 1
                except Exception:
                    pass
            break

    return count

'''

TABS_GLUE = '''    with gr.Tabs():
'''

PROTECT_NEW = '''    def protect_current_inputs():
        """Explicit pre-clear protection step for the automatic storage clear.

        (picgen build) The picgen tab keeps its inputs in RAM (base64 inside
        the hidden textbox), so there are no on-disk input files to protect
        before the automatic storage clear. Kept as a chain step so the
        generate wiring stays identical to the newgen app.
        """
        return None
'''

FASTAPI_IMPORTS = '''
from fastapi.responses import Response as _FastAPIResponse
from fastapi import Request as _FastAPIRequest

'''

LAUNCH = '''
if __name__ == "__main__":
    print(
        f" GRADIO LAUNCHING  Qwen on {PIC_DEVICE} "
        f"({GPU_NAME}, {GPU_VRAM_GB:.0f} GB) -- picgen ready immediately."
    )
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[SCRIPT_DIR, os.path.join(SCRIPT_DIR, "tmp", "gradio")],
    )
'''

PROMPTS_IMPORT = '''from prompts import (
    solo_prompts_dict, couple_man_unseen_prompts_dict, couple_man_seen_prompts_dict,
    multiple_women_prompts_dict, multiple_man_unseen_prompts_dict, multiple_man_seen_prompts_dict, multistep_prompts_dict,
    update_solo_prompt, update_couple_man_unseen_prompt, update_couple_man_seen_prompt,
    update_multiple_women_prompt, update_multiple_man_unseen_prompt, update_multiple_man_seen_prompt, update_multistep_prompt,
)
'''

parts = []
parts.append(seg(1, 56))                     # header, imports, logging filters
parts.append(G1)                             # STARTUP_MODE = "picgen"
parts.append(seg(121, 139))                  # gpu op lock, SCRIPT_DIR, makedirs
parts.append(seg(140, 482))                  # early gradio patches, crypto, version self-heals
parts.append(seg(485, 544))                  # diffusers cache reset + env vars
parts.append(seg(545, 571))                  # cv2/numpy/torch/PIL imports
parts.append("\nimport gradio as gr\n")
parts.append(seg(740, 747))                  # Qwen pipeline import + sys.path
parts.append(seg(748, 852))                  # media store, print(), crypto helpers
parts.append(seg(883, 890))                  # _media_name
parts.append(PROMPTS_IMPORT)                 # picgen-only prompt dicts + updaters
parts.append(seg(1019, 1089))                # _detect_gpu_profile
parts.append(GPU_TAIL)
parts.append(seg(5916, 5918))                # PICGEN_MODELS_DIR etc.
parts.append(seg(5920, 6146))                # safe-move, offload-to-cpu, fp8, probe
parts.append(seg(6149, 6174))                # offload banner + _offload_state + FULL_RESIDENCY_VRAM_GB
parts.append(seg(6175, 6281))                # _device_total_vram_gb .. _pic_offload_enabled
parts.append(seg(6473, 6565, dedent=True))   # picgen startup load (dedented from else-branch)
parts.append(SMART_RESIDENCY)
parts.append(seg(6567, 6570, dedent=True))   # load timing + _active_model = "pic"
parts.append(ACTIVATE_PIC)
parts.append(seg(6719, 6939))                # caches, starters, b64 decode
parts.append(CLEANUP_GLUE)
parts.append(seg(7043, 7252))                # infer_with_preclear + infer
parts.append(seg(3549, 3690))                # path/filename protection helpers
parts.append(seg(3694, 3706))                # current input/merge path vars
parts.append(seg(7253, 7542))                # gallery_js
parts.append(seg(7543, 7696))                # css
parts.append(seg(7942, 8013))                # Blocks + top bar + clear storage + show media
parts.append(TABS_GLUE)
parts.append(seg(9446, 9892))                # the picgen tab, verbatim
parts.append(seg(9893, 9993))                # gallery load + download intercept + starter note
parts.append(seg(10026, 10670))              # encryption init js + outpaint popup js
parts.append(FASTAPI_IMPORTS)
parts.append(seg(10790, 10883))              # /starters /media /logs/stream routes
parts.append(LAUNCH)

doc = "\n".join(p if p.endswith("\n") else p + "\n" for p in parts)

# --- protect_current_inputs: replace vidgen signature/body with picgen no-op
m = re.search(
    r"    def protect_current_inputs\(reference_image, end_image, merge_a, merge_b\):.*?\n        return None\n",
    doc, flags=re.S)
if not m:
    raise SystemExit("protect_current_inputs regex failed")
doc = doc[:m.start()] + PROTECT_NEW + doc[m.end():]

rep = [
    # RAM paths: newgen -> picgen
    ('os.makedirs("/dev/shm/newgen", exist_ok=True)',
     'os.makedirs("/dev/shm/picgen", exist_ok=True)'),
    ('"TMPDIR": "/dev/shm/newgen"', '"TMPDIR": "/dev/shm/picgen"'),
    ('"TEMP": "/dev/shm/newgen"', '"TEMP": "/dev/shm/picgen"'),
    ('"TMP": "/dev/shm/newgen"', '"TMP": "/dev/shm/picgen"'),
    # diffusers version-check message: no Wan in this build
    ('f"  WanImageToVideoPipeline will fail to import.\\n"',
     'f"  The Qwen pipeline import may fail.\\n"'),
    # offload state: pic only
    ('_offload_state = {"pic": False, "wan": False}',
     '_offload_state = {"pic": False}'),
    # infer timing print: no dual_gpu in this build
    ('f"(active model: {_active_model}, dual_gpu: {DUAL_GPU})")',
     'f"(active model: {_active_model})")'),
    # startup print: residency mode is now conditional
    ('print(f" QWEN READY (model-CPU-offload) in {qwen_time:.1f}s - Picgen functional!")',
     'print(f" QWEN READY in {qwen_time:.1f}s - Picgen functional!")'),
    # pin CUDA device right after the PICGEN MODE banner (top level now)
    ('print(" PICGEN MODE: Loading Qwen to GPU first for immediate use...")\n',
     'print(" PICGEN MODE: Loading Qwen to GPU first for immediate use...")\n'
     'torch.cuda.set_device(PIC_DEVICE)\n'),
    # remove the vidgen player-key block from the protection vars
    ('\n# Tracks the _media_store key for the video currently loaded in the player.\n'
     '# This key uses the "vidgen_player_" prefix which _do_clear_storage deliberately\n'
     '# does NOT release, so the video stays playable in the player across auto-clears.\n'
     '# Reset to None when a new generation starts (which calls _do_clear_storage as\n'
     '# a pre-clear) or when the user explicitly clicks Clear Storage.\n'
     '_current_player_media_key = None\n', '\n'),
]
for old, new in rep:
    if old not in doc:
        raise SystemExit("MISSING PATTERN:\n" + old[:200])
    doc = doc.replace(old, new, 1)

# generate chains: no vidgen widgets to protect in picgen (2 occurrences)
doc = doc.replace(
    "                    inputs=[reference_image, end_image, merge_img_a, merge_img_b],",
    "                    inputs=[],",
)

with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(doc)

print("wrote", OUT, len(doc.splitlines()), "lines")
