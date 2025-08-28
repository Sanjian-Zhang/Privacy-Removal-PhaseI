#!/usr/bin/env python3
"""
Face Anonymization Made Simple - Performance Benchmark
For testing processing speed under different configurations
"""

import torch
import time
import os
from PIL import Image
import numpy as np

def check_gpu_setup():
    """Check GPU setup"""
    print("🔍 GPU Environment Check:")
    print(f"  - CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - Current GPU: {torch.cuda.current_device()}")
        print(f"  - GPU Name: {torch.cuda.get_device_name()}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_mem = torch.cuda.memory_allocated() / 1024**3
        reserved_mem = torch.cuda.memory_reserved() / 1024**3
        
        print(f"  - Total VRAM: {total_mem:.1f}GB")
        print(f"  - Allocated: {allocated_mem:.1f}GB")
        print(f"  - Reserved: {reserved_mem:.1f}GB")
        print(f"  - Available VRAM: {total_mem - reserved_mem:.1f}GB")
        
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - PyTorch Version: {torch.__version__}")
        
        return True
    else:
        print("  ❌ GPU not available, will use CPU (very slow)")
        return False

def test_model_loading_speed():
    """Test model loading speed"""
    print("\n🚀 Testing model loading speed...")
    
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
        
        for precision, torch_dtype in [("float32", torch.float32), ("float16", torch.float16)]:
            print(f"\n  📦 Testing {precision} precision loading:")
            precision_start = time.time()
            
            try:
                unet = UNet2DConditionModel.from_pretrained(
                    face_model_id, subfolder="unet", use_safetensors=True, torch_dtype=torch_dtype
                )
                unet_time = time.time() - precision_start
                print(f"    - UNet loading time: {unet_time:.2f}s")
                
                referencenet = ReferenceNetModel.from_pretrained(
                    face_model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=torch_dtype
                )
                ref_time = time.time() - precision_start - unet_time
                print(f"    - ReferenceNet loading time: {ref_time:.2f}s")
                
                del unet, referencenet
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ❌ {precision} precision loading failed: {e}")
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    
    return True

def benchmark_inference_steps():
    """Benchmark speed for different inference steps"""
    print("\n⚡ Inference steps performance comparison:")
    
    steps_to_test = [10, 25, 50, 100, 200]
    
    print("Steps    | Est. Time per Image | Speed Multiplier")
    print("-" * 45)
    
    base_time_per_step = 0.15
    
    for steps in steps_to_test:
        estimated_time = steps * base_time_per_step
        speedup = (200 * base_time_per_step) / estimated_time
        print(f"{steps:^8} | {estimated_time:^15.1f}s | {speedup:^11.1f}x")

def create_test_image(size=(512, 512)):
    """Create test image"""
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(img_array)

def memory_usage_test():
    """VRAM usage test"""
    print("\n💾 VRAM usage test:")
    
    if not torch.cuda.is_available():
        print("  ❌ GPU not available, skipping VRAM test")
        return
    
    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  - Initial VRAM usage: {initial_mem:.1f}MB")
    
    sizes = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
    
    for size in sizes:
        try:
            img_tensor = torch.randn(1, 3, size[1], size[0], device='cuda', dtype=torch.float16)
            current_mem = torch.cuda.memory_allocated() / 1024**2
            mem_used = current_mem - initial_mem
            print(f"  - {size[0]}x{size[1]} image approx. VRAM: {mem_used:.1f}MB")
            del img_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  - {size[0]}x{size[1]} image insufficient VRAM: {e}")

def optimization_recommendations():
    """Performance optimization recommendations"""
    print("\n🎯 Performance optimization recommendations:")
    
    gpu_available = torch.cuda.is_available()
    total_mem = 0
    
    if gpu_available:
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print("\n1. Inference steps optimization:")
    print("   - Default 200 steps → Recommended 25-50 steps (4-8x speed up)")
    print("   - Slight quality reduction, but significant speed improvement")
    print("   - Recommended test range: 25, 50, 100 steps")
    
    print("\n2. Precision optimization:")
    if gpu_available and total_mem >= 6:
        print("   - Use float16 half precision (recommended)")
        print("   - VRAM usage halved, 20-30% speed improvement")
    else:
        print("   - Low VRAM, strongly recommend using float16")
    
    print("\n3. Memory optimization:")
    print("   - Enable attention_slicing")
    print("   - Enable xformers (if available)")
    print("   - Enable model_cpu_offload (when VRAM insufficient)")
    
    print("\n4. Batch processing optimization:")
    print("   - Standardize image size (512x512)")
    print("   - Skip already processed images")
    print("   - Preload models, avoid repeated loading")
    
    print("\n5. Hardware recommendations:")
    if gpu_available:
        print(f"   - Current GPU: {torch.cuda.get_device_name()}")
        if total_mem < 6:
            print("   - Low VRAM, recommend cloud GPU")
        elif total_mem >= 12:
            print("   - Sufficient VRAM, can handle large images or batch processing")
    else:
        print("   - Strongly recommend using GPU, CPU processing will be very slow")

def main():
    print("🎭 Face Anonymization Made Simple - Performance Benchmark")
    print("=" * 60)
    
    gpu_ok = check_gpu_setup()
    
    if gpu_ok:
        test_model_loading_speed()
    
    benchmark_inference_steps()
    
    memory_usage_test()
    
    optimization_recommendations()
    
    print("\n🎯 Quick optimization command:")
    print("Use optimized batch processing script:")
    print("python batch_anonymize_optimized.py \\")
    print("    --input_dir /path/to/input \\")
    print("    --output_dir /path/to/output \\")
    print("    --num_inference_steps 25 \\")
    print("    --use_fp16")
    
    print("\n📝 Performance comparison (estimated):")
    print("- Original settings: ~30s/image (200 steps, float32)")
    print("- Optimized settings: ~4s/image (25 steps, float16)")
    print("- Speed improvement: ~8x")

if __name__ == "__main__":
    main()
