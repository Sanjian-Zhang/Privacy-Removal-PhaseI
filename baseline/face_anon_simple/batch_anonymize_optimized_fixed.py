#!/usr/bin/env python3
"""
Face Anonymization Made Simple - Optimized batch anonymization processing script
Improve processing speed through reduced inference steps, half-precision, and intelligent handling of large images
"""

import os
import argparse
import glob
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import time

try:
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from diffusers import AutoencoderKL, DDPMScheduler
    from diffusers.utils import load_image
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure required dependencies are installed: pip install transformers diffusers")
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)

def setup_pipeline(use_fp16=True, enable_attention_slicing=True):
    """Setup optimized face anonymization processing pipeline"""
    print("🔧 Setting up optimized processing pipeline...")
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    # Use half precision when loading models
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    
    print("  - Loading UNet model...")
    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - Loading ReferenceNet model...")
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - Loading Conditioning ReferenceNet model...")
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        sd_model_id, subfolder="vae", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - Loading Scheduler...")
    scheduler = DDPMScheduler.from_pretrained(
        sd_model_id, subfolder="scheduler", use_safetensors=True
    )
    
    print("  - Loading CLIP feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
    
    print("  - Loading CLIP image encoder...")
    image_encoder = CLIPVisionModel.from_pretrained(
        clip_model_id, use_safetensors=True, torch_dtype=torch_dtype
    )

    # Create pipeline
    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # Enable memory optimization
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    
    # Enable memory efficient attention (if available)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  ✅ Enabled xformers memory efficient attention")
    except:
        print("  ⚠️ xformers not available, skipping memory optimization")
    
    # Enable model offloading to save VRAM
    try:
        pipe.enable_model_cpu_offload()
        print("  ✅ Enabled model CPU offload")
    except:
        print("  ⚠️ Unable to enable model CPU offload")
    
    print("✅ Optimized pipeline setup complete")
    return pipe

def preprocess_image(image_path, max_pixels=1920*1080):
    """Preprocess image, handling large images to avoid memory issues"""
    try:
        # Load image
        original_image = load_image(image_path)
        
        # Ensure correct image mode
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Smart size handling: preserve original size, but scale down very large images to avoid VRAM shortage
        w, h = original_image.size
        original_w, original_h = w, h
        
        # Calculate pixel count, scale if exceeds threshold
        current_pixels = w * h
        
        if current_pixels > max_pixels:
            # Calculate scaling factor
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            print(f"  Image too large, temporarily scaling: {w}x{h} → {new_w}x{new_h} (scale factor: {scale_factor:.3f})")
        else:
            new_w, new_h = w, h
        
        # Ensure dimensions are multiples of 8 (required by diffusion model)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        # If adjusted dimensions are 0, set minimum values
        if new_w == 0:
            new_w = 8
        if new_h == 0:
            new_h = 8
            
        return original_image, (original_w, original_h), (new_w, new_h)
    except Exception as e:
        print(f"❌ Image preprocessing failed: {str(e)}")
        return None, None, None

def anonymize_image_fast(pipe, image_path, output_path, 
                        anonymization_degree=1.25, 
                        num_inference_steps=25,
                        guidance_scale=4.0):
    """Fast anonymization of single image"""
    try:
        # Preprocess image
        original_image, original_size, processing_size = preprocess_image(image_path)
        if original_image is None:
            return False, 0
            
        original_w, original_h = original_size
        new_w, new_h = processing_size
        
        print(f"  Processing size: {new_w}x{new_h}")
        
        # Generate anonymized image
        generator = torch.manual_seed(42)
        
        start_time = time.time()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        anon_image = pipe(
            source_image=original_image,
            conditioning_image=original_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=new_w,
            height=new_h,
        ).images[0]
        
        processing_time = time.time() - start_time
        
        # Resize back to original size (if needed)
        if (new_w, new_h) != (original_w, original_h):
            print(f"  Resize back to original size: {original_w}x{original_h}")
            anon_image = anon_image.resize((original_w, original_h), Image.Resampling.LANCZOS)
        
        # Save result - use high quality settings to maintain original quality
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If original is PNG, save as PNG for lossless; otherwise use high quality JPEG
        if image_path.lower().endswith('.png'):
            anon_image.save(output_path.replace('.jpg', '.png'), format='PNG')
        else:
            anon_image.save(output_path, format='JPEG', quality=98, optimize=False)
        
        return True, processing_time
    except Exception as e:
        print(f"❌ Processing {image_path} failed: {str(e)}")
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, 0

def process_batch_optimized(args):
    """Optimized batch image processing"""
    # Get all input files
    input_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = glob.glob(input_pattern, recursive=True)
    
    if not image_files:
        print(f"❌ No image files found in {input_pattern}")
        return
    
    print(f"📁 Found {len(image_files)} image files")
    
    # Setup pipeline
    pipe = setup_pipeline(use_fp16=args.use_fp16)
    
    # Statistics
    success_count = 0
    total_time = 0
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        # Generate output path
        rel_path = os.path.relpath(image_file, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        
        # Check if should skip existing files
        if os.path.exists(output_path) and not args.overwrite:
            print(f"⏭️ Skipping existing file: {output_path}")
            continue
        
        # Process image
        success, proc_time = anonymize_image_fast(
            pipe, image_file, output_path,
            anonymization_degree=args.anonymization_degree,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
        
        if success:
            success_count += 1
            total_time += proc_time
            avg_time = total_time / success_count
            print(f"✅ Processing complete: {rel_path} ({proc_time:.2f}s, average: {avg_time:.2f}s)")
        else:
            print(f"❌ Processing failed: {rel_path}")
    
    print("=" * 60)
    print(f"🎉 Batch processing complete!")
    print(f"  - Successfully processed: {success_count}/{len(image_files)} files")
    print(f"  - Total time: {total_time:.2f}s")
    if success_count > 0:
        print(f"  - Average per image: {total_time/success_count:.2f}s")
    print(f"  - Output directory: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Face Anonymization Made Simple - Optimized batch processing tool")
    parser.add_argument("--input_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output image directory")
    parser.add_argument("--pattern", default="**/*.jpg", help="Image file matching pattern (default: '**/*.jpg')")
    parser.add_argument("--anonymization_degree", type=float, default=1.25, help="Anonymization degree (default: 1.25)")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Inference steps (default: 25, original 200)")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale (default: 4.0)")
    parser.add_argument("--use_fp16", action="store_true", help="Use half precision to save VRAM")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    print("🚀 Face Anonymization Made Simple - Optimized batch processing tool")
    print("=" * 60)
    print(f"⚙️ Configuration:")
    print(f"  - Input directory: {args.input_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Matching pattern: {args.pattern}")
    print(f"  - Anonymization degree: {args.anonymization_degree}")
    print(f"  - Inference steps: {args.num_inference_steps} (original: 200)")
    print(f"  - Guidance scale: {args.guidance_scale}")
    print(f"  - Use half precision: {args.use_fp16}")
    print(f"  - Overwrite files: {args.overwrite}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU device: {torch.cuda.get_device_name()}")
        print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 60)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start processing
    process_batch_optimized(args)

if __name__ == "__main__":
    main()
