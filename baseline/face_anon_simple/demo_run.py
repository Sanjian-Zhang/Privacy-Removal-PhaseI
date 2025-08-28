#!/usr/bin/env python3
"""
face_anon_simple人脸匿名化演示
"""

import torch
import os
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
    """设置人脸匿名化处理管道"""
    print("🔧 设置处理管道...")
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    # 加载模型
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

    # 创建管道
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

def anonymize_single_face(pipe, image_path, output_path, anonymization_degree=1.25):
    """对单个对齐的人脸进行匿名化处理"""
    print(f"🎭 处理图像: {image_path}")
    
    # 加载图像
    original_image = load_image(image_path)
    original_width, original_height = original_image.size
    print(f"  - 原始图像尺寸: {original_image.size}")
    
    # 生成匿名化图像
    generator = torch.manual_seed(1)
    anon_image = pipe(
        source_image=original_image,
        conditioning_image=original_image,
        num_inference_steps=200,
        guidance_scale=4.0,
        generator=generator,
        anonymization_degree=anonymization_degree,
        width=original_width,
        height=original_height,
    ).images[0]
    
    # 保存结果
    anon_image.save(output_path)
    print(f"  - 匿名化图像已保存至: {output_path}")
    
    return original_image, anon_image

def face_swap(pipe, source_path, target_path, output_path):
    """在两张图像之间进行人脸交换"""
    print(f"🔄 人脸交换: {source_path} -> {target_path}")
    
    # 加载图像
    source_image = load_image(source_path)
    target_image = load_image(target_path)
    target_width, target_height = target_image.size
    
    # 生成交换结果
    generator = torch.manual_seed(1)
    swap_image = pipe(
        source_image=source_image,
        conditioning_image=target_image,
        num_inference_steps=200,
        guidance_scale=4.0,
        generator=generator,
        anonymization_degree=0.0,
        width=target_width,
        height=target_height,
    ).images[0]
    
    # 保存结果
    swap_image.save(output_path)
    print(f"  - 交换结果已保存至: {output_path}")
    
    return source_image, target_image, swap_image

def main():
    print("🎭 Face Anonymization Made Simple - 演示程序")
    print("=" * 50)
    
    # 创建输出目录
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 设置管道
        pipe = setup_pipeline()
        
        # 测试1: 单人脸匿名化
        print("\n📝 测试1: 单人脸匿名化")
        original, anon = anonymize_single_face(
            pipe, 
            "my_dataset/test/14795.png", 
            f"{output_dir}/anon_14795.png"
        )
        
        # 测试2: 人脸交换
        print("\n📝 测试2: 人脸交换")
        source, target, swapped = face_swap(
            pipe,
            "my_dataset/test/00482.png",
            "my_dataset/test/14795.png", 
            f"{output_dir}/swap_result.png"
        )
        
        print("\n✅ 演示完成！")
        print(f"📁 结果已保存在 '{output_dir}' 目录中")
        print("🎉 face_anon_simple项目运行成功!")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
