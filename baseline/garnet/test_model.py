#!/usr/bin/env python3
"""
Simple GaRNet model testing script
"""

import os
import sys
sys.path.append('./CODE')

import torch
import numpy as np
from model import GaRNet

def test_model():
    print("Testing GaRNet model...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = GaRNet(in_channels=4)
        model = model.to(device)
        print("✓ Model created successfully")
        
        dummy_input = torch.randn(1, 4, 512, 512).to(device)
        print("✓ Dummy input created")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✓ Model forward pass successful")
            print(f"  Output type: {type(output)}")
            if isinstance(output, (list, tuple)):
                print(f"  Number of outputs: {len(output)}")
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        print(f"  Output {i} shape: {out.shape}")
            else:
                print(f"  Output shape: {output.shape}")
        
        print("✓ Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_model()
