#!/usr/bin/env python3
"""
Batch face anonymization processing script
"""
import os
import torch
from pathlib import Path
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
import face_alignment
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline
from utils.anonymize_faces_in_image import anonymize_faces_in_image

def setup_models():
    """Initialize models"""
    print("Loading models...")
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True
    )
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
    scheduler = DDPMScheduler.from_pretrained(
        sd_model_id, subfolder="scheduler", use_safetensors=True
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        clip_model_id, use_safetensors=True
    )
    image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Enable memory efficient attention
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        pipe.enable_xformers_memory_efficient_attention()
    
    # Enable model CPU offload to save GPU memory
    if hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    
    print(f"Models loaded to {device}")

    generator = torch.manual_seed(1)
    
    # Setup face detector - choose device based on availability
    fa_device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd", device=fa_device
    )
    
    return pipe, generator, fa

def process_images_batch(input_dir, output_dir, pipe, generator, fa, batch_size=4):
    """Batch image processing - optimized version"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Get all image files
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_formats]
    
    print(f"Found {len(image_files)} images to process, batch size: {batch_size}")
    
    # Process in batches
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}")
        
        for i, image_file in enumerate(batch_files):
            global_idx = batch_start + i + 1
            print(f"  Processing image {global_idx}/{len(image_files)}: {image_file.name}")
            
            try:
                # Load original image
                original_image = load_image(str(image_file))
                
                # Perform face anonymization processing
                anon_image = anonymize_faces_in_image(
                    image=original_image,
                    face_alignment=fa,
                    pipe=pipe,
                    generator=generator,
                    face_image_size=512,
                    num_inference_steps=25,  # Use original recommended 25 steps
                    guidance_scale=4.0,      # Use original recommended 4.0
                    anonymization_degree=1.25,
                )
                
                # Save processed image
                output_file = output_path / f"anon_{image_file.name}"
                anon_image.save(str(output_file))
                print(f"  Saved: {output_file}")
                
            except Exception as e:
                print(f"  Error processing {image_file.name}: {str(e)}")
                continue
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  Batch {batch_start//batch_size + 1} completed, GPU cache cleared")

def main():
    # Set input and output directories
    input_dir = "/input"
    output_dir = "/output"
    
    print("Starting model initialization...")
    pipe, generator, fa = setup_models()
    
    print("Starting batch image processing...")
    # Optimized batch size for RTX 6000 Ada (48GB) memory capacity
    # Current usage: ~15GB/image, safe batch size: 3-4 images
    batch_size = 4  # Safe batch size for 48GB GPU
    process_images_batch(input_dir, output_dir, pipe, generator, fa, batch_size)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()
