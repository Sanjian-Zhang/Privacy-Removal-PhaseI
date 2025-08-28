#!/usr/bin/env python3
"""
Face Anonymization Made Simple - Single image processing script (using GPU 1)
"""

import os
import torch
import gc
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

def clear_cuda_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def setup_pipeline():
    """Setup face anonymization processing pipeline - using GPU 1"""
    print("🔧 Setting up processing pipeline (using GPU 1)...")
    
    device = "cuda:1"
    
    clear_cuda_memory()
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    print("📥 Loading models...")
    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True, torch_dtype=torch.float16
    )
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=torch.float16
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True, torch_dtype=torch.float16)
    scheduler = DDPMScheduler.from_pretrained(
        sd_model_id, subfolder="scheduler", use_safetensors=True
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        clip_model_id, use_safetensors=True
    )
    image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True, torch_dtype=torch.float16)

    print("🔗 Creating processing pipeline...")
    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        pipe.enable_xformers_memory_efficient_attention()
    
    if hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    
    pipe = pipe.to(device, torch_dtype=torch.float16)
    
    return pipe

def anonymize_single_image(input_path, output_path, anonymization_degree=1.25):
    """Process single image"""
    try:
        print(f"📸 Processing image: {input_path}")
        
        pipe = setup_pipeline()
        
        original_image = load_image(input_path)
        print(f"🖼️ Image size: {original_image.size}")
        
        target_size = 256
        width, height = original_image.size
        if width > target_size or height > target_size:
            if width > height:
                new_width = target_size
                new_height = int(height * target_size / width)
            else:
                new_height = target_size
                new_width = int(width * target_size / height)
        else:
            new_width, new_height = width, height
        
        print(f"🔄 Resized dimensions: {new_width}x{new_height}")
        
        generator = torch.manual_seed(42)
        
        with torch.cuda.amp.autocast():
            anon_image = pipe(
                source_image=original_image,
                conditioning_image=original_image,
                num_inference_steps=20,
                guidance_scale=4.0,
                generator=generator,
                anonymization_degree=anonymization_degree,
                width=new_width,
                height=new_height,
            ).images[0]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anon_image.save(output_path)
        
        print(f"✅ Processing completed: {output_path}")
        
        del pipe, anon_image, original_image
        clear_cuda_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        clear_cuda_memory()
        return False

def main():
    input_path = "/home/zhiqics/sanjian/dataset/test_images/images/frame_00032.jpg"
    output_path = "/home/zhiqics/sanjian/dataset/test_images/anon/frame_00032_anon.jpg"
    
    print("🎭 Face Anonymization Made Simple - Single Image Processing (GPU 1)")
    print("=" * 50)
    print(f"📁 Input file: {input_path}")
    print(f"📁 Output file: {output_path}")
    print("=" * 50)
    
    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        return
    
    success = anonymize_single_image(input_path, output_path)
    
    if success:
        print("🎉 Processing completed successfully!")
    else:
        print("❌ Processing failed!")

if __name__ == "__main__":
    main()
