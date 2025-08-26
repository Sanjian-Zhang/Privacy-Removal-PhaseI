#!/usr/bin/env python3
"""
Face Anonymization Made Simple - 单张图片处理脚本 (使用GPU 1)
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
    """清理 CUDA 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def setup_pipeline():
    """设置人脸匿名化处理管道 - 使用GPU 1"""
    print("🔧 设置处理管道 (使用 GPU 1)...")
    
    # 设置使用 GPU 1
    device = "cuda:1"
    
    # 清理内存
    clear_cuda_memory()
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    # 加载模型
    print("📥 加载模型...")
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

    # 创建管道
    print("🔗 创建处理管道...")
    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    
    # 启用内存高效注意力
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        pipe.enable_xformers_memory_efficient_attention()
    
    # 启用内存高效VAE
    if hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    
    pipe = pipe.to(device, torch_dtype=torch.float16)
    
    return pipe

def anonymize_single_image(input_path, output_path, anonymization_degree=1.25):
    """处理单张图像"""
    try:
        print(f"📸 处理图像: {input_path}")
        
        # 设置管道
        pipe = setup_pipeline()
        
        # 加载图像
        original_image = load_image(input_path)
        print(f"🖼️ 图像尺寸: {original_image.size}")
        
        # 使用更小的尺寸以节省内存
        target_size = 256  # 进一步减小尺寸
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
        
        print(f"🔄 调整后尺寸: {new_width}x{new_height}")
        
        # 生成匿名化图像
        generator = torch.manual_seed(42)
        
        # 使用更少的推理步数
        with torch.cuda.amp.autocast():
            anon_image = pipe(
                source_image=original_image,
                conditioning_image=original_image,
                num_inference_steps=20,  # 大幅减少步数
                guidance_scale=4.0,
                generator=generator,
                anonymization_degree=anonymization_degree,
                width=new_width,
                height=new_height,
            ).images[0]
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anon_image.save(output_path)
        
        print(f"✅ 处理完成: {output_path}")
        
        # 清理内存
        del pipe, anon_image, original_image
        clear_cuda_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
        clear_cuda_memory()
        return False

def main():
    # 处理第一张图片
    input_path = "/home/zhiqics/sanjian/dataset/test_images/images/frame_00032.jpg"
    output_path = "/home/zhiqics/sanjian/dataset/test_images/anon/frame_00032_anon.jpg"
    
    print("🎭 Face Anonymization Made Simple - 单图处理 (GPU 1)")
    print("=" * 50)
    print(f"📁 输入文件: {input_path}")
    print(f"📁 输出文件: {output_path}")
    print("=" * 50)
    
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    success = anonymize_single_image(input_path, output_path)
    
    if success:
        print("🎉 处理成功完成!")
    else:
        print("❌ 处理失败!")

if __name__ == "__main__":
    main()
