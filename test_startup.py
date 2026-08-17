#!/usr/bin/env python3
"""
Test script to verify startup mode configuration without loading models.
"""
import sys

# Simulate the startup mode logic
STARTUP_MODE = "vidgen"  # New default
for _arg in sys.argv[1:]:
    _flag = _arg.lstrip("-").lower()
    if _flag == "vidgen":
        STARTUP_MODE = "vidgen"
    elif _flag == "picgen":
        STARTUP_MODE = "picgen"

# Simulate GPU detection
DUAL_GPU = False  # Most common case

print(f"🧪 Testing startup configuration:")
print(f"   STARTUP_MODE: {STARTUP_MODE}")
print(f"   DUAL_GPU: {DUAL_GPU}")

# Test model placement logic
if DUAL_GPU or STARTUP_MODE == "picgen":
    qwen_location = "GPU"
    initial_active_model = "pic"
else:
    qwen_location = "CPU"
    initial_active_model = "cpu"

print(f"   Qwen initial location: {qwen_location}")
print(f"   Initial active model: {initial_active_model}")

# Test tab selection (0 = vidgen, 1 = picgen)
selected_tab = 0 if STARTUP_MODE == "vidgen" else 1
tab_name = "Video Generator" if selected_tab == 0 else "Photo Editor"
print(f"   Selected tab: {selected_tab} ({tab_name})")

# Test background loading logic
if not DUAL_GPU and STARTUP_MODE == "vidgen":
    wan_startup_action = "Load Wan directly to GPU"
    final_active_model = "wan"
else:
    wan_startup_action = "Load Wan to CPU only"
    final_active_model = initial_active_model

print(f"   Background action: {wan_startup_action}")
print(f"   Final active model: {final_active_model}")

print("\n✅ Configuration test results:")
print(f"   - Vidgen is default: {'✅' if STARTUP_MODE == 'vidgen' else '❌'}")
print(f"   - Video tab shown first: {'✅' if selected_tab == 0 else '❌'}")
print(f"   - Wan loads to GPU first: {'✅' if final_active_model == 'wan' else '❌'}")
print(f"   - Qwen starts on CPU: {'✅' if qwen_location == 'CPU' else '❌'}")