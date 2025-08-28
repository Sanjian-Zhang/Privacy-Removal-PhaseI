#!/usr/bin/env python3
"""
GaRNet complete usage example
This script demonstrates how to use GaRNet for text removal
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

sys.path.append('./CODE')
from model import GaRNet

def setup_garnet():
    """Initialize GaRNet model"""
    print("Setting up GaRNet...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = GaRNet(in_channels=3)
    model_path = "./WEIGHTS/GaRNet/saved_model.pth"
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        print("✗ Model file not found. Please run download_model.py first")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    return model, device

def run_example():
    """Run example"""
    print("\n" + "="*50)
    print("GaRNet Scene Text Removal Example")
    print("="*50)
    
    example_img = "./DATA/EXAMPLE/IMG/img_302.jpg"
    example_txt = "./DATA/EXAMPLE/TXT/img_302.txt"
    output_path = "./example_output.jpg"
    
    if not os.path.exists(example_img):
        print("✗ Example image not found")
        return False
    
    if not os.path.exists(example_txt):
        print("✗ Example text file not found")
        return False
    
    print(f"✓ Input image: {example_img}")
    print(f"✓ Text coordinates: {example_txt}")
    print(f"✓ Output will be saved to: {output_path}")
    
    model, device = setup_garnet()
    if model is None:
        return False
    
    print("\nRunning inference...")
    try:
        import subprocess
        result = subprocess.run([
            "python", "./CODE/inference.py", 
            "--gpu" if device == "cuda" else "",
            "--result_path", "./example_results",
            "--image_path", "./DATA/EXAMPLE/IMG",
            "--box_path", "./DATA/EXAMPLE/TXT"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✓ Inference completed successfully")
            if os.path.exists("./example_results/img_302.jpg"):
                print(f"✓ Output image saved to: ./example_results/img_302.jpg")
                return True
            else:
                print("✗ Output image not found")
                return False
        else:
            print(f"✗ Inference failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return False

def print_usage_info():
    """Print usage information"""
    print("\n" + "="*50)
    print("GaRNet Usage Information")
    print("="*50)
    print("\n1. Model Architecture:")
    print("   - GaRNet is a scene text removal model")
    print("   - Uses Gated Attention mechanism")
    print("   - Supports GPU acceleration")
    
    print("\n2. File Structure:")
    print("   ├── CODE/              # Model and inference code")
    print("   ├── DATA/EXAMPLE/      # Example data")
    print("   ├── WEIGHTS/           # Pre-trained models")
    print("   └── garnet_env/        # Python virtual environment")
    
    print("\n3. Main Functions:")
    print("   - inference.py: Image inference")
    print("   - eval.py: Model evaluation")
    print("   - model.py: Model definition")
    
    print("\n4. Input Format:")
    print("   - Images: JPG/PNG format")
    print("   - Text boxes: TXT file, each line contains coordinates of 4 points")
    
    print("\n5. Example Commands:")
    print("   cd CODE")
    print("   python inference.py --gpu --image_path ../DATA/EXAMPLE/IMG")

if __name__ == "__main__":
    print_usage_info()
    
    success = run_example()
    
    if success:
        print(f"\n✓ GaRNet example ran successfully!")
        print("You can now:")
        print("1. View the output result image")
        print("2. Test with your own images")
        print("3. Adjust parameters to optimize results")
    else:
        print(f"\n✗ Example failed, please check error messages")
        
    print(f"\nVirtual environment location: {os.path.abspath('./garnet_env')}")
    print("Activate virtual environment: source garnet_env/bin/activate")
