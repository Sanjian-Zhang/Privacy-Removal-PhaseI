#!/usr/bin/env python3
"""
Face Anonymization Made Simple - Batch anonymization processing script
For batch processing anonymization of large quantities of images
"""

import os
import argparse
import glob
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)

def setup_pipeline():
    """Setup face anonymization processing pipeline"""
    print("🔧 Setting up processing pipeline...")
    
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
    pipe = pipe.to("cuda")
    
    return pipe

def anonymize_image(pipe, image_path, output_path, anonymization_degree=1.25):
    """Anonymize a single image"""
    try:
        original_image = load_image(image_path)
        original_width, original_height = original_image.size
        
        generator = torch.manual_seed(42)
        anon_image = pipe(
            source_image=original_image,
            conditioning_image=original_image,
            num_inference_steps=100,
            guidance_scale=4.0,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=original_width,
            height=original_height,
        ).images[0]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anon_image.save(output_path)
        return True
    except Exception as e:
        print(f"❌ Failed to process {image_path}: {str(e)}")
        return False

def process_batch(args):
    """Batch process images"""
    input_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = glob.glob(input_pattern)
    
    if not image_files:
        print(f"⚠️ No matching image files found: {input_pattern}")
        return
    
    print(f"🔍 Found {len(image_files)} image files to process")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pipe = setup_pipeline()
    
    success_count = 0
    failed_count = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        rel_path = os.path.relpath(image_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        
        if anonymize_image(pipe, image_path, output_path, args.anonymization_degree):
            success_count += 1
        else:
            failed_count += 1
    
    print("\n📊 Processing statistics:")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Total: {len(image_files)}")
    print(f"🎉 Processing completed! Results saved in: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Face Anonymization Made Simple - Batch Processing Tool")
    parser.add_argument("--input_dir", required=True, help="Input image directory")
    parser.add_argument("--output_dir", required=True, help="Output image directory")
    parser.add_argument("--pattern", default="*.jpg", help="Image file matching pattern (default: '*.jpg')")
    parser.add_argument("--anonymization_degree", type=float, default=1.25, help="Anonymization degree (default: 1.25)")
    
    args = parser.parse_args()
    
    print("🎭 Face Anonymization Made Simple - Batch Processing Tool")
    print("=" * 50)
    print(f"⚙️ Configuration:")
    print(f"  - Input directory: {args.input_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Matching pattern: {args.pattern}")
    print(f"  - Anonymization degree: {args.anonymization_degree}")
    print("=" * 50)
    
    try:
        process_batch(args)
    except KeyboardInterrupt:
        print("\n⚠️ User interrupted, exiting...")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
