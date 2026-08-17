GPU Instance Selection Guide

---

GPU: RTX 3090 (24GB)
Single Tab: 2× stacked (48GB total)
Double Tab: 4× stacked (96GB total)
Price for 1 tab (hourly): ~$0.32/h
Price for 2 tabs (hourly): ~$0.64/h
Est. Vidgen (3.5s video): ~45-60s
Est. Picgen (1 image): ~10-14s
Template(s):
  simplepodai/ubuntu22.04-devel:cuda126
  nvidia/cuda:12.4.1-devel-ubuntu22.04

GPU: RTX 4090 (24GB)
Single Tab: 2× stacked (48GB total)
Double Tab: 4× stacked (96GB total)
Price for 1 tab (hourly): ~$0.70/h
Price for 2 tabs (hourly): ~$1.40/h
Est. Vidgen (3.5s video): ~25-35s
Est. Picgen (1 image): ~5-7s
Template(s):
  simplepodai/ubuntu22.04-devel:cuda126
  nvidia/cuda:12.4.1-devel-ubuntu22.04

GPU: RTX 5090 (32GB)
Single Tab: 2× stacked (64GB total)
Double Tab: 3× stacked (96GB total)
Price for 1 tab (hourly): ~$1.00/h
Price for 2 tabs (hourly): ~$1.50/h
Est. Vidgen (3.5s video): ~18-25s
Est. Picgen (1 image): ~4-6s
Template(s):
  simplepodai/ubuntu22.04-devel:cuda128

GPU: RTX PRO 6000 MIG 2g.48gb (48GB)
Single Tab: 1× (48GB, no stacking needed)
Double Tab: 2× stacked (96GB total)
Price for 1 tab (hourly): ~$0.79/h
Price for 2 tabs (hourly): ~$1.58/h
Est. Vidgen (3.5s video): ~15-22s
Est. Picgen (1 image): ~4-6s
Template(s):
  simplepodai/ubuntu22.04-devel:cuda128

GPU: RTX PRO 6000 Blackwell (95GB)
Single Tab: 1× (95GB, no stacking needed)
Double Tab: 1× (95GB — both models fit on a single card)
Price for 1 tab (hourly): ~$1.00/h
Price for 2 tabs (hourly): ~$1.00/h (same card, no extra cost)
Est. Vidgen (3.5s video): ~10-14s
Est. Picgen (1 image): ~2.5-3s
Template(s):
  simplepodai/ubuntu22.04-devel:cuda128

---

Notes:
- App auto-detects GPU count, VRAM, and selects optimal mode automatically
- No manual configuration needed — just run python3 app.py
- SageAttention installs automatically via setup.sh for extra speed
- Stacked mode splits model layers across all GPUs (accelerate balanced device_map)
- Concurrent mode (≥48GB per GPU) pins one model per GPU — both tabs instant
- Generation times assume all optimizations active (SageAttention + torch.compile + TF32)
