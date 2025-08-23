#!/usr/bin/env python3
"""
Face Anonymization Made Simple - 优化版批量匿名化处理脚本
通过减少推理步数、使用半精度等方式提升处理速度，并智能处理大尺寸图片
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
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所需依赖: pip install transformers diffusers")
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)

def setup_pipeline(use_fp16=True, enable_attention_slicing=True):
    """设置优化的人脸匿名化处理管道"""
    print("🔧 设置优化处理管道...")
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    # 加载模型时使用半精度
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    
    print("  - 加载UNet模型...")
    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - 加载ReferenceNet模型...")
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - 加载Conditioning ReferenceNet模型...")
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True, torch_dtype=torch_dtype
    )
    
    print("  - 加载VAE...")
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
        clip_model_id, use_safetensors=True, torch_dtype=torch_dtype
    )

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
    
    # 移动到GPU
    pipe = pipe.to("cuda")
    
    # 启用内存优化
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    
    # 启用内存高效注意力（如果可用）
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  ✅ 启用xformers内存高效注意力")
    except:
        print("  ⚠️ xformers不可用，跳过内存优化")
    
    # 启用模型卸载以节省显存
    try:
        pipe.enable_model_cpu_offload()
        print("  ✅ 启用模型CPU卸载")
    except:
        print("  ⚠️ 无法启用模型CPU卸载")
    
    print("✅ 优化管道设置完成")
    return pipe

def preprocess_image(image_path, max_pixels=1920*1080):
    """预处理图像，处理大尺寸图片以避免内存问题"""
    try:
        # 加载图像
        original_image = load_image(image_path)
        
        # 确保图像模式正确
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # 智能尺寸处理：保持原始尺寸，但对超大图片进行缩放避免显存不足
        w, h = original_image.size
        original_w, original_h = w, h
        
        # 计算像素数量，如果超过阈值则缩放
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
            
        return original_image, (original_w, original_h), (new_w, new_h)
    except Exception as e:
        print(f"❌ 预处理图像失败: {str(e)}")
        return None, None, None

def anonymize_image_fast(pipe, image_path, output_path, 
                        anonymization_degree=1.25, 
                        num_inference_steps=25,
                        guidance_scale=4.0):
    """快速匿名化单个图像"""
    try:
        # 预处理图像
        original_image, original_size, processing_size = preprocess_image(image_path)
        if original_image is None:
            return False, 0
            
        original_w, original_h = original_size
        new_w, new_h = processing_size
        
        print(f"  处理尺寸: {new_w}x{new_h}")
        
        # 生成匿名化图像
        generator = torch.manual_seed(42)
        
        start_time = time.time()
        
        # 清理GPU缓存
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
        
        # 调整回原始尺寸（如果需要）
        if (new_w, new_h) != (original_w, original_h):
            print(f"  调整回原始尺寸: {original_w}x{original_h}")
            anon_image = anon_image.resize((original_w, original_h), Image.Resampling.LANCZOS)
        
        # 保存结果 - 使用高质量设置保持原始质量
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 如果原图是PNG，保存为PNG保持无损；否则使用高质量JPEG
        if image_path.lower().endswith('.png'):
            anon_image.save(output_path.replace('.jpg', '.png'), format='PNG')
        else:
            anon_image.save(output_path, format='JPEG', quality=98, optimize=False)
        
        return True, processing_time
    except Exception as e:
        print(f"❌ 处理 {image_path} 失败: {str(e)}")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, 0

def process_batch_optimized(args):
    """优化版批量处理图像"""
    # 获取所有输入文件
    input_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = glob.glob(input_pattern, recursive=True)
    
    if not image_files:
        print(f"❌ 在 {input_pattern} 中未找到图像文件")
        return
    
    print(f"📁 找到 {len(image_files)} 个图像文件")
    
    # 设置管道
    pipe = setup_pipeline(use_fp16=args.use_fp16)
    
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
    parser = argparse.ArgumentParser(description="Face Anonymization Made Simple - 优化批量处理工具")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出图像目录")
    parser.add_argument("--pattern", default="**/*.jpg", help="图像文件匹配模式 (默认: '**/*.jpg')")
    parser.add_argument("--anonymization_degree", type=float, default=1.25, help="匿名化程度 (默认: 1.25)")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="推理步数 (默认: 25, 原始200)")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="引导缩放 (默认: 4.0)")
    parser.add_argument("--use_fp16", action="store_true", help="使用半精度浮点数以节省显存")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    
    args = parser.parse_args()
    
    print("🚀 Face Anonymization Made Simple - 优化批量处理工具")
    print("=" * 60)
    print(f"⚙️ 配置:")
    print(f"  - 输入目录: {args.input_dir}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 匹配模式: {args.pattern}")
    print(f"  - 匿名化程度: {args.anonymization_degree}")
    print(f"  - 推理步数: {args.num_inference_steps} (原始: 200)")
    print(f"  - 引导缩放: {args.guidance_scale}")
    print(f"  - 使用半精度: {args.use_fp16}")
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
    process_batch_optimized(args)

if __name__ == "__main__":
    main()
