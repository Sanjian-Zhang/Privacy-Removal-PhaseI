#!/usr/bin/env python3
"""
测试face_anon_simple项目设置
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
    print("🚀 开始测试face_anon_simple项目设置...")
    
    # 检查CUDA可用性
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    try:
        print("📦 加载模型...")
        face_model_id = "hkung/face-anon-simple"
        clip_model_id = "openai/clip-vit-large-patch14"
        sd_model_id = "stabilityai/stable-diffusion-2-1"

        print("  - 加载UNet模型...")
        unet = UNet2DConditionModel.from_pretrained(
            face_model_id, subfolder="unet", use_safetensors=True
        )
        
        print("  - 加载ReferenceNet模型...")
        referencenet = ReferenceNetModel.from_pretrained(
            face_model_id, subfolder="referencenet", use_safetensors=True
        )
        
        conditioning_referencenet = ReferenceNetModel.from_pretrained(
            face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
        )
        
        print("  - 加载VAE模型...")
        vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
        
        print("  - 加载调度器...")
        scheduler = DDPMScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler", use_safetensors=True
        )
        
        print("  - 加载CLIP模型...")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            clip_model_id, use_safetensors=True
        )
        image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

        print("🔧 创建处理管道...")
        pipe = StableDiffusionReferenceNetPipeline(
            unet=unet,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            vae=vae,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            scheduler=scheduler,
        )
        
        print("🚚 将模型移动到GPU...")
        pipe = pipe.to("cuda")
        
        print("🖼️ 测试图像加载...")
        original_image = load_image("my_dataset/test/14795.png")
        print(f"  - 图像尺寸: {original_image.size}")
        
        print("✅ 所有组件加载成功！")
        print("🎉 face_anon_simple项目设置完成，可以正常使用!")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_setup()
    if success:
        print("\n🎊 项目设置测试成功!")
    else:
        print("\n💥 项目设置测试失败!")
