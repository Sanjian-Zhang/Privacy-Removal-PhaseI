#!/usr/bin/env python
import sys
print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTesting dp2 import...")
try:
    import dp2
    print("✓ Deep Privacy 2 imported successfully!")
    print(f"dp2 module location: {dp2.__file__}")
except Exception as e:
    print(f"✗ Import error: {e}")
    
print("\nTesting other key dependencies...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except Exception as e:
    print(f"✗ PyTorch import error: {e}")
    
try:
    import torchvision
    print(f"✓ Torchvision version: {torchvision.__version__}")
except Exception as e:
    print(f"✗ Torchvision import error: {e}")
