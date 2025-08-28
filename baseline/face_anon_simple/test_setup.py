#!/usr/bin/env python3
"""
Test face_anon_simple project setup
"""

import torch
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

def test_setup():
    print("🚀 Starting face_anon_simple project setup test...")
    
    # Check CUDA availability
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    
    try:
        print("📦 Loading models...")
        face_model_id = "hkung/face-anon-simple"
        clip_model_id = "openai/clip-vit-large-patch14"
        sd_model_id = "stabilityai/stable-diffusion-2-1"

        print("  - Loading UNet model...")
        unet = UNet2DConditionModel.from_pretrained(
            face_model_id, subfolder="unet", use_safetensors=True
        )
        
        print("  - Loading ReferenceNet model...")
        referencenet = ReferenceNetModel.from_pretrained(
            face_model_id, subfolder="referencenet", use_safetensors=True
        )
        
        conditioning_referencenet = ReferenceNetModel.from_pretrained(
            face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
        )
        
        print("  - Loading VAE model...")
        vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
        
        print("  - Loading scheduler...")
        scheduler = DDPMScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler", use_safetensors=True
        )
        
        print("  - Loading CLIP model...")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            clip_model_id, use_safetensors=True
        )
        image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

        print("🔧 Creating processing pipeline...")
        pipe = StableDiffusionReferenceNetPipeline(
            unet=unet,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            vae=vae,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            scheduler=scheduler,
        )
        
        print("🚚 Moving models to GPU...")
        pipe = pipe.to("cuda")
        
        print("🖼️ Testing image loading...")
        original_image = load_image("my_dataset/test/14795.png")
        print(f"  - Image size: {original_image.size}")
        
        print("✅ All components loaded successfully!")
        print("🎉 face_anon_simple project setup completed, ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_setup()
    if success:
        print("\n🎊 Project setup test successful!")
    else:
        print("\n💥 Project setup test failed!")
