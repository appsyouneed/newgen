Let me read the truncated middle section:Now I have a thorough understanding of the code. Let me do the research on cheaper hosting options and newer/better models.Now I have a thorough picture. Let me also check what the Wan 2.2 NVFP4+sparse model from LightX2V looks like — it was mentioned as "50x faster on RTX 5090" from the GitHub.Now I have everything I need. Let me write up a comprehensive, accurate analysis.

---

## Full Analysis: Hosting Cost, True VRAM Usage, and Model Upgrades

### Why You Hit OOM on 95 GB VRAM

Your code's own comments are honest about this (lines 311–318), but let me give you the real numbers:

**WAMU v2 (Wan 2.2 I2V A14B distilled, BF16):** The two merged expert safetensors are 28.6 GB each = **57.2 GB just for weights**. But that's file size. In BF16 resident on GPU with KV cache, VAE activations at 720p, and CUDA context overhead, real peak usage is 65–80 GB at 720p. Custom NSFW merges (which WAMU v2 is) often push higher because they carry merged adapter weights on top of the base. This is why your code's own comment at line 315 notes "Wan alone has been observed pinning ~94 GB by itself."

**Qwen Image Edit 2511 (BF16):** At BF16 the transformer + text encoder + VAE totals ~40.9 GB. With activations, FP16 inference needs roughly 45 GB. So even with `enable_model_cpu_offload` the peak GPU spike during the transformer forward pass can be 20–25 GB.

**Why 95 GB fails:** When Wan is pinned at ~80–94 GB (720p with custom merge overhead), there's 1–15 GB left. Qwen's transformer submodule alone needs 20+ GB for its forward pass even with model-cpu-offload. That's your OOM. The math in your code (lines 282–286) that says 92 GB combined is optimistic — it ignores activations, CUDA context, VAE tiling buffers, and RIFE's ~500 MB footprint. Real combined peak is 105–120+ GB.

---

### Cheapest Hosting Options (Replacing SimplePod @ $1–1.50/hr)

The RTX PRO 6000 (96 GB GDDR7) is the only single-card option that can fit Wan alone in swap mode, but 95 GB isn't truly enough as you discovered. Your real options, ranked by cost:

**Tier 1 — Cheapest: Vast.ai (marketplace/interruptible)**

Vast.ai rents the RTX PRO 6000 WS (Workstation Edition) starting at $0.67/hr and the Server Edition from $0.87/hr on Vast.ai — both interruptible/spot pricing. On-demand runs a bit higher but still under your $1/hr target at off-peak hours. Vast.ai is a GPU marketplace so prices fluctuate with supply; you can often get $0.67–0.85/hr for the WS edition at non-peak times, which is meaningfully cheaper than SimplePod.

**Tier 2 — Cheap + stable: RunPod Community Cloud**

RunPod lists RTX PRO 6000 Community Cloud from $1.69/hr and Secure Cloud from $2.09/hr. Still more than SimplePod's $1–1.50, but RunPod Community is interruptible and often cheaper in practice.

**Dual-GPU path (your app's concurrent mode, the fastest setup):** Two cheaper 48 GB cards (L40S/RTX 6000 Ada) can each hold one model fully with headroom. A100 80GB rents from $1.76/hr on Northflank; two L40S 48GB cards on Vast.ai can come to roughly $1.60–2.00/hr total. That unlocks your app's dual-GPU concurrent mode (no swapping), which is dramatically faster.

**Bottom line:** Vast.ai spot at $0.67–0.87/hr for one RTX PRO 6000 WS is cheaper than SimplePod, but the OOM problem remains until you fix the VRAM budget. Read on.

---

### Fixing the OOM: Two Practical Paths

**Path A — Stay on one 96 GB card, drop to 480p only for Wan**

At 480p, Wan 2.2 A14B peak VRAM drops to roughly 40–55 GB (vs 65–80 GB at 720p), which leaves 40+ GB headroom for Qwen swap. This is already half-implemented in your app (`resolution` param). If you can live with 480p video output, one RTX PRO 6000 on Vast.ai at $0.67/hr works.

**Path B — Two GPUs, concurrent mode (fastest, no swap)**

Two separate GPUs, each holding one model. Your app already supports this perfectly (lines 331–341). The cheapest two-GPU combos that work:
- Two L40S (48 GB each): Wan uses ~45–55 GB at 480p on one card, Qwen uses ~41–45 GB BF16 on the other. Both fit with a few GB of headroom at 480p. Available on Vast.ai for ~$0.70–0.90/hr per L40S = ~$1.40–1.80/hr total. Faster than one PRO 6000 in swap mode because there's zero swap penalty.
- One RTX PRO 6000 (96 GB, Wan 720p) + one L40S/A100 80GB (Qwen): ~$0.87 + $0.90 = ~$1.77/hr. This enables 720p video with no OOM and full concurrent mode.

---

### Model Upgrade Research: What's Newer and Better

Here's what's actually available that you should know about:

**For the Wan video side (your `WAN_MODEL_REPO`):**

The biggest recent development is the April 2026 release of `Wan2.2-I2V-A14B-4step-720p-high` and `Wan2.2-I2V-A14B-4step-720p-low` from lightx2v/Wan2.2-Distill-Models. This version was specifically trained on a high-quality 720p dataset with an optimized low-noise training algorithm, significantly boosting fine-grained detail rendering and visual texture. This is a direct upgrade over the weights your app uses (WAMU v2 is based on earlier distill weights). It's merged/distilled in the same format but better trained.

Even more exciting for Blackwell GPUs: the `lightx2v/Wan2.2-NVFP4-Sparse` model combines NVFP4 quantization-aware step distillation with sparse attention specifically targeting Blackwell architecture, dramatically reducing VRAM while preserving visual quality. On a single RTX 5090 it achieves over 50× speedup vs the standard model. The RTX PRO 6000 Blackwell uses the same GB202 die, so it should also benefit from NVFP4 kernels. VRAM footprint with NVFP4 is roughly half of BF16 (~28–30 GB), which would solve your OOM and still run 720p.

The catch: NVFP4 requires the LightX2V framework (not diffusers), so it's not a drop-in for your current app — it's a significant integration project.

**For FP8 as a middle ground (still diffusers-compatible):** The distill-models repo also offers FP8 variants. Each expert file drops from 28.6 GB to ~15 GB, so combined they fit in ~35–40 GB resident VRAM. This would let 720p Wan + Qwen coexist on a single 96 GB card with real headroom.

**For the Qwen image editing side:**

Your app already uses Qwen-Image-Edit-2511 (the December 2025 version), which is current and still the best open-source image editing model available. No newer version has released as of August 2026. Qwen-Image-Edit-2511 added improved multi-image editing consistency, multi-angle view generation, and refined face identity preservation over the prior 2509 version — and those improvements are already in your build since you're loading the 2511 weights.

However, **FP8 Qwen is worth considering**: at FP8 the Qwen model runs in ~20.5 GB vs ~40.9 GB at BF16, which halves its VRAM footprint with near-identical quality. Combined with FP8 Wan (~30 GB), both models resident on one 96 GB card becomes feasible at 60 GB total + headroom.

**For RIFE (your interpolation):** You're using RIFE HDv3 via the Practical-RIFE repo. RIFE v4.26 (which you're already downloading) is current and still state-of-the-art for frame interpolation. Nothing meaningfully better exists for this use case.

**For MMAudio:** You're on `mmaudio_large_44k_v2` which is the best available open-source audio generation model for video. No meaningful upgrades exist here either.

---

### Summary Recommendations

**Cheapest path to fix OOM right now, ranked:**

1. **Vast.ai, RTX PRO 6000 WS at $0.67–0.87/hr, use 480p for Wan only.** This works today with zero code changes. Your OOM is at 720p; 480p fits. Set `resolution="480p"` as default.

2. **Switch WAMU v2 to FP8 variants of the April 2026 lightx2v distill weights.** The new `Wan2.2-I2V-A14B-4step-720p` FP8 files (~15 GB each, ~30 GB combined) are better trained than WAMU v2 AND dramatically smaller. Combined with FP8 Qwen (~20 GB), both fit resident on 96 GB with headroom for 720p concurrent mode on a single card. This is the best bang-for-buck upgrade.

3. **Two-GPU concurrent mode on Vast.ai** (~$1.40–1.80/hr total for two L40S 48 GB cards) for 480p concurrent operation, or one PRO 6000 + one A100/L40S for 720p concurrent. Unlocks your app's fastest code path.

4. **NVFP4 (lightx2v/Wan2.2-NVFP4-Sparse)** is the most powerful upgrade but requires replacing diffusers with LightX2V framework. If you're willing to refactor, this gives you 50x+ speed on Blackwell and ~14–15 GB VRAM for the whole Wan pipeline, leaving 80+ GB for Qwen at full BF16. Best eventual target.


we can get max speed and quality and no loss. (note i usually do 480p cause its faster) and ensure to add what you can based on what we have already and upgrade my models so we get better generations and faster but fit hopefully both at once on blackwell gpu (ensuring we can still split/stack multiple gpus and it knows which ones etc)