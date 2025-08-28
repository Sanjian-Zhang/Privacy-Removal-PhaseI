#!/usr/bin/env python3
"""
Garnet Text Anonymizer - Process Train Dataset
Processes images from Train dataset with text detection and anonymization
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import torchvision
from model import GaRNet


def detect_text_regions_simple(image):
    """Simple text detection using image processing techniques"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply morphological operations to find text-like regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask for potential text regions
    text_mask = np.zeros(gray.shape, dtype=np.uint8)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter based on aspect ratio and size (typical for text)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        # Heuristics for text detection
        if (1.5 < aspect_ratio < 8.0 and 100 < area < 50000 and 
            10 < w < 500 and 10 < h < 100):
            cv2.rectangle(text_mask, (x, y), (x + w, y + h), 255, -1)
    
    return text_mask.astype(np.float32) / 255.0


def process_image_with_garnet(image_path, model, device, input_size=512):
    """Process a single image with Garnet"""
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  Error: Could not load image {image_path}")
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image_rgb.shape
    print(f"  Image dimensions: {W}x{H}")
    
    # Detect text regions
    box_mask = detect_text_regions_simple(image_rgb)
    
    # Check if there are any text regions to process
    text_pixels = np.sum(box_mask)
    if text_pixels == 0:
        print(f"  No text regions detected, copying original image")
        return image  # Return original image if no text regions
    
    print(f"  Text area detected: {int(text_pixels)} pixels")
    
    # Resize mask to input size
    box_mask_resized = cv2.resize(box_mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    box_mask_resized = np.expand_dims(box_mask_resized, axis=0).astype(np.float32)
    
    # Preprocess image
    image_resized = cv2.resize(image_rgb, (input_size, input_size)).transpose(2, 0, 1).astype(np.float32)
    image_normalized = image_resized / 127.5 - 1
    
    # Convert to tensors
    image_tensor = torch.FloatTensor(image_normalized)
    mask_tensor = torch.FloatTensor(box_mask_resized)
    
    # Combine image and mask
    input_tensor = torch.cat([image_tensor, mask_tensor], dim=0).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        try:
            print(f"  Running Garnet inference...")
            _, _, _, _, result, _, _, _ = model(input_tensor)
            
            # Apply result only to masked regions
            result_processed = (1 - mask_tensor) * image_tensor + mask_tensor * result.cpu()
            
            # Convert back to image
            output_image = (torch.clamp(result_processed[0] + 1, 0, 2) * 127.5).cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            output_image = cv2.resize(output_image, (W, H))
            
            print(f"  ✓ Text anonymization completed")
            return output_image
            
        except Exception as e:
            print(f"  ✗ Error during inference: {e}")
            return image  # Return original image on error


def main():
    input_dir = "/input"
    output_dir = "/output"
    
    print("Starting Garnet Text Anonymizer (Train Dataset Mode)...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Garnet model...")
    try:
        model = GaRNet(3)
        model_path = "/workspace/garnet_model.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Error: Model file {model_path} not found!")
            sys.exit(1)
            
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print("No image files found in input directory!")
        sys.exit(1)
    
    image_files.sort()
    print(f"Found {len(image_files)} image files to process")
    
    # Process images in batches to avoid memory issues
    batch_size = 10
    processed_count = 0
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        for image_path in batch_files:
            filename = os.path.basename(image_path)
            
            # Process the image
            result_image = process_image_with_garnet(image_path, model, device)
            
            if result_image is not None:
                # Save result
                output_path = os.path.join(output_dir, f"garnet_{filename}")
                success = cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    print(f"  ✓ Saved: garnet_{filename}")
                    processed_count += 1
                else:
                    print(f"  ✗ Failed to save: {filename}")
            else:
                print(f"  ✗ Failed to process: {filename}")
    
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count}/{len(image_files)} images")
    
    # Show summary of output files
    output_files = glob.glob(os.path.join(output_dir, "*"))
    if output_files:
        print(f"\nGenerated {len(output_files)} output files")
        print("Sample output files:")
        for f in sorted(output_files)[:5]:  # Show first 5 files
            print(f"  {os.path.basename(f)}")
        if len(output_files) > 5:
            print(f"  ... and {len(output_files) - 5} more files")


if __name__ == "__main__":
    main()
