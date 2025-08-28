#!/usr/bin/env python3
"""
Face Anonymization Made Simple - Lossless batch anonymization processing script
Maintain original image dimensions and highest quality
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
import time

def setup_pipeline(use_fp16=False, enable_attention_slicing=True):
    """Setup high quality processing pipeline - lossless mode uses fp32"""
    print("🔧 Setting up high quality processing pipeline (lossless mode)...")
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    # Use fp32 to ensure highest quality
    torch_dtype = torch.float32
    
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
    
    print("  - 加载Scheduler...")
    scheduler = DDPMScheduler.from_pretrained(
        sd_model_id, subfolder="scheduler", use_safetensors=True
    )
    
    print("  - 加载CLIP特征提取器...")
    feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
    
    print("  - 加载CLIP图像编码器...")
    image_encoder = CLIPVisionModel.from_pretrained(
        clip_model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    
    # 创建管道
    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
    )
    
    # 启用内存优化
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    
    # 启用更多内存优化
    try:
        pipe.enable_model_cpu_offload()
        print("  ✅ 启用模型CPU卸载")
    except:
        print("  ⚠️ 无法启用模型CPU卸载")
    
    try:
        pipe.enable_sequential_cpu_offload()
        print("  ✅ 启用顺序CPU卸载")
    except:
        print("  ⚠️ 无法启用顺序CPU卸载")
    
    # 移至GPU
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ 使用GPU处理")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        print("  ✅ 清理GPU缓存")
    else:
        device = "cpu"
        print("⚠️ 未检测到CUDA，使用CPU处理")
    
    return pipe

def anonymize_image_lossless(pipe, image_path, output_path, 
                        anonymization_degree=1.25, 
                        num_inference_steps=50,  # 提高推理步数确保质量
                        guidance_scale=4.0):
    """无损匿名化单个图像"""
    try:
        # 加载图像
        original_image = load_image(image_path)
        
        # 保持原始图片尺寸，但对于超大图片进行智能缩放以避免显存不足
        w, h = original_image.size
        original_w, original_h = w, h
        
        # 计算像素数量，如果超过阈值则缩放
        max_pixels = 1920 * 1080  # 2M像素阈值，适合大多数GPU
        current_pixels = w * h
        
        if current_pixels > max_pixels:
            # 计算缩放比例
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            print(f"  图片过大，临时缩放: {w}x{h} → {new_w}x{new_h} (缩放比例: {scale_factor:.3f})")
        else:
            new_w, new_h = w, h
        
        # 确保尺寸是8的倍数（扩散模型要求）
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        
        # 如果调整后的尺寸为0，设置最小值
        if new_w == 0:
            new_w = 8
        if new_h == 0:
            new_h = 8
        
        print(f"  处理尺寸: {new_w}x{new_h}")
        
        # 生成匿名化图像
        generator = torch.manual_seed(42)
        
        start_time = time.time()
        anon_image = pipe(
            source_image=original_image,
            conditioning_image=original_image,
            num_inference_steps=num_inference_steps,  # 使用更多推理步数
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=new_w,
            height=new_h,
        ).images[0]
        
        processing_time = time.time() - start_time
        
        # 调整回原始尺寸（使用高质量重采样）
        if (new_w, new_h) != (original_w, original_h):
            print(f"  调整回原始尺寸: {original_w}x{original_h}")
            anon_image = anon_image.resize((original_w, original_h), Image.Resampling.LANCZOS)
        
        # 保存结果 - 无损保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 根据原始格式选择保存方式
        original_ext = os.path.splitext(image_path)[1].lower()
        if original_ext in ['.png', '.tiff', '.tif']:
            # 无损格式
            output_path_lossless = os.path.splitext(output_path)[0] + original_ext
            anon_image.save(output_path_lossless, format='PNG' if original_ext == '.png' else 'TIFF')
            print(f"  保存为无损格式: {output_path_lossless}")
        else:
            # JPEG格式使用最高质量
            anon_image.save(output_path, format='JPEG', quality=100, optimize=False, subsampling=0)
            print(f"  保存为高质量JPEG: {output_path}")
        
        return True, processing_time
    except Exception as e:
        print(f"❌ 处理 {image_path} 失败: {str(e)}")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, 0

def process_batch_lossless(args):
    """无损批量处理图像"""
    # 获取所有输入文件
    input_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = glob.glob(input_pattern, recursive=True)
    
    if not image_files:
        print(f"❌ 在 {input_pattern} 中未找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 个图像文件")
    
    # 设置管道
    pipe = setup_pipeline(use_fp16=False)  # 无损模式不使用fp16
    
    # 统计信息
    success_count = 0
    total_time = 0
    
    # 处理每个图像
    for image_file in tqdm(image_files, desc="处理图像"):
        # 生成输出路径
        rel_path = os.path.relpath(image_file, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)
        
        # 检查是否跳过已存在的文件
        if os.path.exists(output_path) and not args.overwrite:
            print(f"⏭️ 跳过已存在文件: {output_path}")
            continue
        
        # 处理图像
        success, proc_time = anonymize_image_lossless(
            pipe, image_file, output_path,
            anonymization_degree=args.anonymization_degree,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        )
        
        if success:
            success_count += 1
            total_time += proc_time
            avg_time = total_time / success_count
            print(f"✅ 处理完成: {rel_path} ({proc_time:.2f}s, 平均: {avg_time:.2f}s)")
        else:
            print(f"❌ 处理失败: {rel_path}")
    
    print("=" * 60)
    print(f"🎉 批量处理完成!")
    print(f"  - 成功处理: {success_count}/{len(image_files)} 个文件")
    print(f"  - 总用时: {total_time:.2f}s")
    if success_count > 0:
        print(f"  - 平均每张: {total_time/success_count:.2f}s")
    print(f"  - 输出目录: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Face Anonymization Made Simple - 无损批量处理工具")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出图像目录")
    parser.add_argument("--pattern", default="**/*.jpg", help="图像文件匹配模式 (默认: '**/*.jpg')")
    parser.add_argument("--anonymization_degree", type=float, default=1.25, help="匿名化程度 (默认: 1.25)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数 (默认: 50, 高质量模式)")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="引导缩放 (默认: 4.0)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    
    args = parser.parse_args()
    
    print("🚀 Face Anonymization Made Simple - 无损批量处理工具")
    print("=" * 60)
    print(f"⚙️ 配置:")
    print(f"  - 输入目录: {args.input_dir}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 匹配模式: {args.pattern}")
    print(f"  - 匿名化程度: {args.anonymization_degree}")
    print(f"  - 推理步数: {args.num_inference_steps} (高质量模式)")
    print(f"  - 引导缩放: {args.guidance_scale}")
    print(f"  - 质量模式: 无损模式 (FP32)")
    print(f"  - 覆盖文件: {args.overwrite}")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU设备: {torch.cuda.get_device_name()}")
        print(f"  - GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始处理
    process_batch_lossless(args)

if __name__ == "__main__":
    main()
