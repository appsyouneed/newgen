[run] Waiting for Gradio startup...
[run] ERROR: process died. Last log:
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/diffusers/models/modeling_utils.py", line 1451, in to
    return super().to(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1367, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 927, in _apply
    module._apply(fn)
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 927, in _apply
    module._apply(fn)
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 927, in _apply
    module._apply(fn)
  [Previous line repeated 2 more times]
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 954, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/root/newgen/.app-venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1353, in convert
    return t.to(
           ^^^^^
torch.OutOfMemoryError: Allocation on device
root@sophisticated-rabbit:~/newgen#

ok then we must ensure:

1. i can load either tab by itself into gpu fully (must work for 5090 and other gpus that have more or same amount of vram as well 
2. i can use bash run.sh -picgen or in general load picgen mode when starting
3. no possibility of cuda oom errors 
4. no reduction of speed, quality or prompt adherence
5. both tabs must be able to load fully into gpu but only 1 at a time whatever i choose
6. i must be able to still switch between them

give updated files now with the guaranteed fixes applied




The OOM is still happening inside `_safe_move_to_device` when moving the transformer — the component is too large to fit with RIFE already resident. Let me see the full log to find exactly which component triggered it, then check what RIFE takes up and whether there's a way to temporarily move it to CPU during the load.




Let me read the uploaded file first.Let me read the critical middle sections:Now let me find the core `activate_wan`, `activate_pic`, and `_safe_move_to_device` functions:Now I have a complete picture of the problem. Let me look at the RIFE loading and picgen startup path:Now I have the complete picture. Here is the root cause and the fix:

**Root cause:** RIFE is loaded unconditionally at line 1095–1100 onto `device = torch.device(PIC_DEVICE)` — **always CUDA**, regardless of `STARTUP_MODE`. In picgen mode, RIFE sits on GPU consuming ~0.5–1 GB. Then when `_safe_move_to_device` tries to move the Qwen transformer (~16 GB) to the same GPU, there isn't enough headroom. The fix is: **RIFE must live on CPU and only move to GPU when actually interpolating, then move back**, or more precisely for the single-GPU case — RIFE must offload to CPU before any large model swap and restore after.

Also: `interpolate_bits` always sends tensors to `device` (hard-coded to `PIC_DEVICE`), not `WAN_DEVICE`, which is another bug.Good — the RIFE `Model.device()` call moves flownet to `cuda:0` by default. Now I have everything I need to write the fix. Let me produce the corrected file sections:Now I have enough to write the complete patched `app.py`. The file is 10,585 lines so I'll copy it and apply the targeted patches: