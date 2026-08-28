LoRA Models Directory
=====================

This directory contains LoRA (Low-Rank Adaptation) models for the Wan 2.2 I2V video generator.

How It Works:
-------------
1. Add any .safetensors LoRA files to this folder
2. The app will automatically discover them on startup
3. They will appear as checkboxes in the Video Generator tab under "🎨 LoRA Models (Optional)"
4. Check the ones you want to use before generating a video

LoRA Naming Convention:
-----------------------
- Files with "_high" or "_high_noise" in the name = High-noise expert (structure/motion)
- Files with "_low" or "_low_noise" in the name = Low-noise expert (details/refinement)
- Both high and low versions are used together for best results

Example:
--------
dance_lora_high_noise.safetensors  → Applied to high-noise expert
dance_lora_low_noise.safetensors   → Applied to low-noise expert

The app groups LoRAs by their base name (everything before _high or _low).

Where to Get LoRAs:
-------------------
- Civitai: https://civitai.com/
- Search for "Wan 2.2 I2V" LoRAs
- Download both high and low versions for complete effect

Notes:
------
- LoRAs are disabled by default - you must check the box to enable them
- Multiple LoRAs can be enabled at once
- Changes take effect on the next video generation
- No code changes needed - just add files and restart the app
