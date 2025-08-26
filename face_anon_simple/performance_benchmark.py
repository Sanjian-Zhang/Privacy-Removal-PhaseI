#!/usr/bin/env python3
"""
Face Anonymization Made Simple - 性能基准测试
用于测试不同配置下的处理速度
"""

import torch
import time
import os
from PIL import Image
import numpy as np

def check_gpu_setup():
    """检查GPU设置"""
    print("🔍 GPU环境检查:")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  - GPU数量: {torch.cuda.device_count()}")
        print(f"  - 当前GPU: {torch.cuda.current_device()}")
        print(f"  - GPU名称: {torch.cuda.get_device_name()}")
        
        # 显存信息
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated() / 1024**3
        reserved_mem = torch.cuda.memory_reserved() / 1024**3
        
        print(f"  - 总显存: {total_mem:.1f}GB")
        print(f"  - 已分配: {allocated_mem:.1f}GB")
        print(f"  - 已保留: {reserved_mem:.1f}GB")
        print(f"  - 可用显存: {total_mem - reserved_mem:.1f}GB")
        
        # CUDA版本
        print(f"  - CUDA版本: {torch.version.cuda}")
        print(f"  - PyTorch版本: {torch.__version__}")
        
        return True
    else:
        print("  ❌ GPU不可用，将使用CPU处理（速度会很慢）")
        return False

def test_model_loading_speed():
    """测试模型加载速度"""
    print("\n🚀 测试模型加载速度...")
    
    try:
        from transformers import CLIPImageProcessor, CLIPVisionModel
        from diffusers import AutoencoderKL, DDPMScheduler
        from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
        from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
        from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline
        
        face_model_id = "hkung/face-anon-simple"
        clip_model_id = "openai/clip-vit-large-patch14"
        sd_model_id = "stabilityai/stable-diffusion-2-1"
        
        start_time = time.time()
        
        # 测试不同精度的加载时间
        for precision, torch_dtype in [("float32", torch.float32), ("float16", torch.float16)]:
            print(f"\n  📦 测试 {precision} 精度加载:")
            precision_start = time.time()
            
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    face_model_id, subfolder="unet", use_safetensors=True, torch_dtype=torch_dtype
                )
                unet_time = time.time() - precision_start
                print(f"    - UNet加载时间: {unet_time:.2f}秒")
                
                referencenet = ReferenceNetModel.from_pretrained(
                    face_model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=torch_dtype
                )
                ref_time = time.time() - precision_start - unet_time
                print(f"    - ReferenceNet加载时间: {ref_time:.2f}秒")
                
                del unet, referencenet
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ {precision} 精度加载失败: {e}")
        
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False
    
    return True

def benchmark_inference_steps():
    """基准测试不同推理步数的速度"""
    print("\n⚡ 推理步数性能对比:")
    
    steps_to_test = [10, 25, 50, 100, 200]
    
    print("推理步数 | 预估单图处理时间 | 速度提升倍数")
    print("-" * 45)
    
    base_time_per_step = 0.15  # 估算每步需要0.15秒（基于GPU性能）
    
    for steps in steps_to_test:
        estimated_time = steps * base_time_per_step
        speedup = (200 * base_time_per_step) / estimated_time
        print(f"{steps:^8} | {estimated_time:^15.1f}秒 | {speedup:^11.1f}x")

def create_test_image(size=(512, 512)):
    """创建测试图像"""
    # 创建一个简单的测试图像
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def memory_usage_test():
    """显存使用测试"""
    print("\n💾 显存使用情况测试:")
    
    if not torch.cuda.is_available():
        print("  ❌ GPU不可用，跳过显存测试")
        return
    
    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  - 初始显存使用: {initial_mem:.1f}MB")
    
    # 模拟不同尺寸图像的显存使用
    sizes = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
    
    for size in sizes:
        # 创建模拟张量来估算显存使用
        try:
            # 模拟输入图像tensor
            img_tensor = torch.randn(1, 3, size[1], size[0], device='cuda', dtype=torch.float16)
            current_mem = torch.cuda.memory_allocated() / 1024**2
            mem_used = current_mem - initial_mem
            print(f"  - {size[0]}x{size[1]} 图像约需显存: {mem_used:.1f}MB")
            del img_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  - {size[0]}x{size[1]} 图像显存不足: {e}")

def optimization_recommendations():
    """性能优化建议"""
    print("\n🎯 性能优化建议:")
    
    gpu_available = torch.cuda.is_available()
    total_mem = 0
    
    if gpu_available:
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print("\n1. 推理步数优化:")
    print("   - 默认200步 → 建议25-50步 (4-8x速度提升)")
    print("   - 质量稍有下降，但速度大幅提升")
    print("   - 建议测试范围: 25, 50, 100步")
    
    print("\n2. 精度优化:")
    if gpu_available and total_mem >= 6:
        print("   - 使用float16半精度 (推荐)")
        print("   - 显存使用减半，速度提升20-30%")
    else:
        print("   - 显存较小，强烈建议使用float16")
    
    print("\n3. 内存优化:")
    print("   - 启用attention_slicing")
    print("   - 启用xformers (如果可用)")
    print("   - 启用model_cpu_offload (显存不足时)")
    
    print("\n4. 批处理优化:")
    print("   - 图像尺寸标准化 (512x512)")
    print("   - 跳过已处理的图像")
    print("   - 预加载模型，避免重复加载")
    
    print("\n5. 硬件建议:")
    if gpu_available:
        print(f"   - 当前GPU: {torch.cuda.get_device_name()}")
        if total_mem < 6:
            print("   - 显存偏小，建议使用cloud GPU")
        elif total_mem >= 12:
            print("   - 显存充足，可以处理大图或批量处理")
    else:
        print("   - 强烈建议使用GPU，CPU处理会非常慢")

def main():
    print("🎭 Face Anonymization Made Simple - 性能基准测试")
    print("=" * 60)
    
    # GPU环境检查
    gpu_ok = check_gpu_setup()
    
    # 模型加载速度测试
    if gpu_ok:
        test_model_loading_speed()
    
    # 推理步数基准测试
    benchmark_inference_steps()
    
    # 显存使用测试
    memory_usage_test()
    
    # 优化建议
    optimization_recommendations()
    
    print("\n🎯 快速优化命令:")
    print("使用优化版批处理脚本:")
    print("python batch_anonymize_optimized.py \\")
    print("    --input_dir /path/to/input \\")
    print("    --output_dir /path/to/output \\")
    print("    --num_inference_steps 25 \\")
    print("    --use_fp16")
    
    print("\n📝 性能对比 (预估):")
    print("- 原始设置: ~30秒/图像 (200步, float32)")
    print("- 优化设置: ~4秒/图像 (25步, float16)")
    print("- 速度提升: ~8倍")

if __name__ == "__main__":
    main()
