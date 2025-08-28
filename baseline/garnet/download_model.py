#!/usr/bin/env python3
"""
Script to download GaRNet pre-trained model
"""

import os
from huggingface_hub import hf_hub_download

def download_garnet_model():
    print("Downloading GaRNet pre-trained model from Hugging Face...")
    
    weights_dir = "WEIGHTS/GaRNet"
    os.makedirs(weights_dir, exist_ok=True)
    
    try:
        model_path = hf_hub_download(
            repo_id="naverpapago/garnet",
            filename="saved_model.pth",
            local_dir=weights_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Model downloaded successfully to: {model_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_garnet_model()
    if success:
        print("Model download completed! You can now run inference.")
    else:
        print("Model download failed. Please check your internet connection.")
