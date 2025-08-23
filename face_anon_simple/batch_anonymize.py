#!/usr/bin/env python3
"""
Face Anonymization Made Simple - 批量匿名化处理脚本
用于批量处理大量图像的匿名化
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

def anonymize_image(pipe, image_path, output_path, anonymization_degree=1.25):
    """对单个图像进行匿名化处理"""
    try:
        # 加载图像
        original_image = load_image(image_path)
        original_width, original_height = original_image.size
        
        # 生成匿名化图像
        generator = torch.manual_seed(42)  # 使用固定种子以保持一致性
        anon_image = pipe(
            source_image=original_image,
            conditioning_image=original_image,
            num_inference_steps=100,  # 减少步数以提高批处理速度
            guidance_scale=4.0,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=original_width,
            height=original_height,
        ).images[0]
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anon_image.save(output_path)
        return True
    except Exception as e:
        print(f"❌ 处理 {image_path} 失败: {str(e)}")
        return False

def process_batch(args):
    """批量处理图像"""
    # 获取所有输入文件
    input_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = glob.glob(input_pattern)
    
    if not image_files:
        print(f"⚠️ 未找到匹配的图像文件: {input_pattern}")
        return
    
    print(f"🔍 找到 {len(image_files)} 个图像文件进行处理")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置处理管道
    pipe = setup_pipeline()
    
    # 批量处理图像
    success_count = 0
    failed_count = 0
    
    for image_path in tqdm(image_files, desc="处理图像"):
        # 构建输出路径
        rel_path = os.path.relpath(image_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        
        # 处理图像
        if anonymize_image(pipe, image_path, output_path, args.anonymization_degree):
            success_count += 1
        else:
            failed_count += 1
    
    # 打印统计信息
    print("\n📊 处理统计:")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {failed_count}")
    print(f"  - 总计: {len(image_files)}")
    print(f"🎉 处理完成! 结果保存在: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Face Anonymization Made Simple - 批量处理工具")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出图像目录")
    parser.add_argument("--pattern", default="*.jpg", help="图像文件匹配模式 (默认: '*.jpg')")
    parser.add_argument("--anonymization_degree", type=float, default=1.25, help="匿名化程度 (默认: 1.25)")
    
    args = parser.parse_args()
    
    print("🎭 Face Anonymization Made Simple - 批量处理工具")
    print("=" * 50)
    print(f"⚙️ 配置:")
    print(f"  - 输入目录: {args.input_dir}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 匹配模式: {args.pattern}")
    print(f"  - 匿名化程度: {args.anonymization_degree}")
    print("=" * 50)
    
    try:
        process_batch(args)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，正在退出...")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
