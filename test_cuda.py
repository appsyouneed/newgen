#!/usr/bin/env python3
"""Quick CUDA diagnostics"""
import torch
import sys

print("=" * 60)
print("CUDA Diagnostics")
print("=" * 60)

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Try to create a tensor on GPU
        print("\nTesting GPU tensor creation...")
        x = torch.randn(100, 100).cuda()
        print("✓ Successfully created tensor on GPU")
        
        # Try a simple operation
        y = x @ x
        print("✓ Successfully performed GPU computation")
        
        print("\n✅ GPU is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ CUDA not available")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
