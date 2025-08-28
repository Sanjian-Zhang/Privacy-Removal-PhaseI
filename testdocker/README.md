# Privacy Removal Phase 1 - Docker Hub Distribution

## 🐳 Available Docker Images

**Docker Hub Repository**: [sanjinn/privacy_removal_phase1](https://hub.docker.com/r/sanjinn/privacy_removal_phase1)

### Available Tags:
- `sanjinn/privacy_removal_phase1:deepprivacy2` - **DeepPrivacy2 Latest (Recommended)** - Face & Full-body anonymization
- `sanjinn/privacy_removal_phase1:face-anon-v3` - Face anonymization v3.0
- `sanjinn/privacy_removal_phase1:garnet` - Text anonymization (CPU optimized)
- `sanjinn/privacy_removal_phase1:dashcam_anonymizer` - Dashcam face & license plate blurring
- `sanjinn/privacy_removal_phase1:deepprivacy` - DeepPrivacy GAN-based face anonymization

## 🚀 Quick Start Guide

### 1. DeepPrivacy2 (Latest - Recommended)

#### Face Anonymization
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/face.py \
  -i /input/image.jpg --output_path /output/result.jpg
```

#### Full-body Anonymization
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/FB_cse.py \
  -i /input/image.jpg --output_path /output/result.jpg --visualize
```

#### Batch Processing
```bash
mkdir -p input output
cp /your/images/* input/

docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  bash -c "find /input -name '*.jpg' -o -name '*.png' | while read img; do python3 anonymize.py configs/anonymizers/face.py -i \"\$img\" --output_path /output/\$(basename \"\$img\"); done"
```

### 2. Text Anonymization (GARNET)
```bash
docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:garnet
```

### 3. Face Anonymization v3.0
```bash
docker run --runtime=nvidia --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:face-anon-v3
```

### 4. Dashcam Processing
```bash
docker run --runtime=nvidia --gpus '"device=1"' \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

### 5. DeepPrivacy GAN (4K Support)
```bash
docker run --rm \
  -v /path/to/input:/workspace/input \
  -v /path/to/output:/workspace/output \
  -v /path/to/deepprivacy/source:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && ./batch_process.sh 2"
```

## 🔄 Complete Privacy Pipeline

### Option 1: Recommended (DeepPrivacy2 + Text + Dashcam)
```bash
# Step 1: Face anonymization with DeepPrivacy2 (latest)
docker run --rm \
  -v /input:/input -v /temp1:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/face.py \
  -i /input --output_path /output

# Step 2: Text anonymization
docker run --rm \
  -v /temp1:/input -v /temp2:/output \
  sanjinn/privacy_removal_phase1:garnet

# Step 3: License plate anonymization (if needed)
docker run --runtime=nvidia --gpus '"device=1"' \
  -v /temp2:/input -v /final_output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

### Option 2: Complete Anonymization (Full-body + Text + Dashcam)
```bash
# Step 1: Full-body anonymization with DeepPrivacy2
docker run --rm \
  -v /input:/input -v /temp1:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/FB_cse.py \
  -i /input --output_path /output --visualize

# Step 2: Text anonymization
docker run --rm \
  -v /temp1:/input -v /temp2:/output \
  sanjinn/privacy_removal_phase1:garnet

# Step 3: License plate anonymization
docker run --runtime=nvidia --gpus '"device=1"' \
  -v /temp2:/input -v /final_output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

## ⚙️ System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB recommended)
- **Disk Space**: 20GB available
- **OS**: Linux (Ubuntu 22.04+ recommended)

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: 6GB+ VRAM
- **NVIDIA Container Toolkit**: Installed
- **NVIDIA Drivers**: 450.80.02+

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
nvidia-smi
sudo systemctl restart docker
```

### Memory Issues
```bash
# Use smaller batches
docker run --rm -e BATCH_SIZE=1 [rest of command]

# Or process single images
docker run --rm \
  -v /input/single.jpg:/input/single.jpg \
  -v /output:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/face.py \
  -i /input/single.jpg --output_path /output/result.jpg
```

## 📊 Comparison Table

| Image | Target | GPU Required | Speed | Quality | Unique Feature |
|-------|--------|--------------|-------|---------|----------------|
| `deepprivacy2` | Face + Full-body | Optional | Fast | **State-of-the-Art** | Face + Full-body modes |
| `garnet` | Text regions | No (CPU) | Moderate | Very High | CPU-only operation |
| `face-anon-v3` | Human faces | Yes (12GB+) | Fast | High | Stable version |
| `dashcam_anonymizer` | Faces + License plates | Yes (6GB+) | Very Fast | High | License plate detection |
| `deepprivacy` | Human faces (GAN) | Optional | Medium | Highest | 4K support + GAN generation |

---

**Note**: DeepPrivacy2 (`:deepprivacy2`) is the latest and most advanced solution, supporting both face and full-body anonymization with state-of-the-art quality.

````

### Available Models
| Model | Quality | Speed | Parameters | Recommended Use |
|-------|---------|-------|------------|-----------------|
| `deep_privacy_v1` | Good | Fast | 46.92M | Quick processing |
| `fdf128_rcnn512` | **Best** | Medium | 47.39M | **Recommended for 4K** |
| `fdf128_retinanet512` | High | Medium | 49.84M | Balanced quality/speed |
| `fdf128_retinanet256` | Medium | Fast | 12.704M | Speed optimized |
| `fdf128_retinanet128` | Basic | Fastest | 3.17M | Low-end hardware |

### Memory Optimization Features
- **Intelligent Batching**: Automatically processes images in small batches to prevent memory overflow
- **Configurable Batch Size**: Adjust batch size based on available system memory
- **Progress Tracking**: Real-time progress updates for large datasets
- **Resume Capability**: Automatically skips already processed images for interrupted sessions

### Configuration & Performance

#### System Requirements
- **Minimum RAM**: 8GB (16GB recommended for 4K images)
- **Disk Space**: 20GB available space
- **GPU** (Optional): NVIDIA GPU with 6GB+ VRAM for acceleration
- **CPU**: Multi-core processor (8+ cores recommended)

#### Processing Performance
- **4K Images (CPU)**: ~1-2 minutes per image
- **4K Images (GPU)**: ~10-30 seconds per image
- **HD Images (CPU)**: ~30-60 seconds per image
- **HD Images (GPU)**: ~5-15 seconds per image

### Batch Processing Configuration

```bash
# Small batch size for 4K images (recommended)
./batch_process.sh 2

# Medium batch size for HD images
./batch_process.sh 5

# Large batch size for smaller images
./batch_process.sh 10
```

### Advanced Usage Examples

#### High-Quality 4K Processing
```bash
# Create working directories
mkdir -p 4k_input 4k_output

# Copy your 4K images
cp /your/4k/images/* 4k_input/

# Process with recommended settings for 4K
sudo docker run --rm \
  -v $(pwd)/4k_input:/workspace/input \
  -v $(pwd)/4k_output:/workspace/output \
  -v /path/to/deepprivacy:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && ./batch_process.sh 2"

# Check results
ls 4k_output/
# Output files: anonymized_frame_001.jpg, anonymized_frame_001_detected_left_anonymized_right.jpg
```

#### GPU Accelerated Processing (if available)
```bash
# Use GPU acceleration for faster processing
sudo docker run --gpus all --rm \
  -v $(pwd)/input:/workspace/input \
  -v $(pwd)/output:/workspace/output \
  -v /path/to/deepprivacy:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && python anonymize.py -s /workspace/input -t /workspace/output -m fdf128_rcnn512"
```

#### Custom Model Selection
```bash
# Use fastest model for quick processing
sudo docker run --rm \
  -v $(pwd)/input:/workspace/input \
  -v $(pwd)/output:/workspace/output \
  -v /path/to/deepprivacy:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && python anonymize.py -s /workspace/input -t /workspace/output -m fdf128_retinanet128"
```

### Output Files
For each input image, DeepPrivacy generates:
1. **Main anonymized image**: `anonymized_[original_name].jpg`
2. **Comparison image**: `anonymized_[original_name]_detected_left_anonymized_right.jpg`
   - Left side: Original with face detection boxes
   - Right side: Anonymized result

### Technical Details
- **Base Environment**: NVIDIA PyTorch 22.08 container
- **Deep Learning Framework**: PyTorch 1.13.0a0
- **Face Detection**: DSFD + Mask R-CNN for keypoint detection
- **Face Generation**: Generative Adversarial Network (GAN)
- **Image Processing**: OpenCV 4.5.5.64
- **NumPy Compatibility**: Fixed for latest NumPy versions

### Success Metrics
- ✅ **394/394** 4K images successfully processed in testing
- ✅ **Memory optimization** prevents out-of-memory errors
- ✅ **High-quality results** with realistic face generation
- ✅ **Production ready** for large-scale deployment
- ✅ **Batch processing** with automatic resume capability

### Troubleshooting

#### Memory Issues
```bash
# Reduce batch size for very large images
./batch_process.sh 1

# Monitor memory usage
docker stats
```

#### Performance Optimization
```bash
# Use faster model for speed
python anonymize.py -s input.jpg -t output.jpg -m fdf128_retinanet128

# Use best model for quality
python anonymize.py -s input.jpg -t output.jpg -m fdf128_rcnn512
```

## �🚗 Dashcam Anonymizer

### Features
- **Face Detection & Blurring**: Automatically detects and blurs human faces
- **License Plate Detection & Blurring**: Automatically detects and blurs license plates
- **4K Resolution Support**: Optimized for 3840x2160 resolution images
- **Multiple Format Support**: Supports JPG, PNG, BMP, TIFF formats
- **YOLOv8 Based**: Uses state-of-the-art YOLOv8 object detection model

### Quick Start for Dashcam Processing

```bash
# Pull the dashcam anonymizer image
sudo docker pull sanjinn/privacy_removal_phase1:dashcam_anonymizer

# Run on your dashcam images
sudo docker run --runtime=nvidia --gpus '"device=1"' \
  -v /path/to/your/dashcam/images:/input \
  -v /path/to/output/directory:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

### Configuration Parameters
- **Detection Confidence**: 0.1 (detects more objects, may include false positives)
- **Blur Radius**: 31 pixels (adjustable for stronger/weaker blur)
- **Supported Resolutions**: Any resolution (optimized for 4K)
- **GPU Memory**: Minimum 6GB VRAM recommended

### Processing Performance
- **4K Images**: ~2-5 seconds per image (depending on GPU)
- **HD Images**: ~1-2 seconds per image
- **Batch Processing**: Processes all images in input directory

### Example Output
Input: `dashcam_image.jpg` → Output: `dashcam_image_blurred.jpg`
- Faces automatically blurred with Gaussian blur
- License plates automatically blurred with Gaussian blur
- Original image quality preserved

## Building and Publishing to Docker Hub

### Push Script Usage

```bash
# Run the push script to upload to Docker Hub
./push_to_dockerhub.sh
```

This will push the `face-anon-v3` tag to Docker Hub.

### Manual Push Commands

```bash
# Tag the local image
sudo docker tag face-anon-processor:v3 sanjinn/privacy_removal_phase1:face-anon-v3

# Push to Docker Hub
sudo docker push sanjinn/privacy_removal_phase1:face-anon-v3
```

## Image Information

### Image Size: ~16GB
- Base: NVIDIA CUDA 12.1.0 runtime
- Models: Pre-downloaded anonymization models
- Dependencies: Python environment with all packages

### GPU Requirements:
- NVIDIA GPU with CUDA support
- Minimum 12GB VRAM
- NVIDIA Container Toolkit installed

---

## Publishing Checklist

### Face Anonymization (v3.0)
- [x] Docker Hub account created
- [x] Repository `sanjinn/privacy_removal_phase1` created on Docker Hub
- [x] Image built and tested locally
- [x] Tag `face-anon-v3` pushed to Docker Hub
- [x] Image tested by pulling from Docker Hub

### Dashcam Anonymizer
- [x] Docker Hub account created
- [x] Repository `sanjinn/privacy_removal_phase1` created on Docker Hub
- [x] Image built and tested locally
- [x] Tag `dashcam_anonymizer` pushed to Docker Hub
- [x] Image tested by pulling from Docker Hub
- [x] Documentation updated

### GARNET Text Anonymization (✅ **COMPLETED**)
- [x] Docker Hub account created
- [x] Repository `sanjinn/privacy_removal_phase1` created on Docker Hub
- [x] Image built and tested locally
- [x] Tag `garnet` pushed to Docker Hub
- [x] Documentation updated
- [x] 394/394 test images successfully processed
- [x] CPU optimization completed
- [x] High quality output verified (95% JPEG quality)
- [x] Production deployment ready

### DeepPrivacy2 Latest State-of-the-Art (✅ **NEWEST - 2024 RELEASE**)
- [x] Docker Hub account created
- [x] Repository `sanjinn/privacy_removal_phase1` created on Docker Hub
- [x] Image built and tested locally
- [x] Tag `deepprivacy2` pushed to Docker Hub
- [x] Documentation updated
- [x] 394/394 test images successfully processed
- [x] Face and full-body anonymization modes implemented
- [x] CPU/GPU auto-detection optimization completed
- [x] Batch processing with progress tracking implemented
- [x] Production deployment ready

### Available Images Status
- `sanjinn/privacy_removal_phase1:face-anon-v3` - ✅ Available
- `sanjinn/privacy_removal_phase1:dashcam_anonymizer` - ✅ Available on Docker Hub
- `sanjinn/privacy_removal_phase1:garnet` - ✅ Available on Docker Hub
- `sanjinn/privacy_removal_phase1:deepprivacy` - ✅ Available on Docker Hub
- `sanjinn/privacy_removal_phase1:deepprivacy2` - ✅ **Available on Docker Hub** (Latest State-of-the-Art)

## 📊 Complete Solution Summary

| Feature | Face Anonymization | Dashcam Anonymizer | GARNET Text | DeepPrivacy GAN | **DeepPrivacy2** |
|---------|-------------------|-------------------|-------------|-----------------|------------------|
| **Target** | Human faces | Faces + License plates | Text regions | Human faces (GAN) | **Faces + Full-body** |
| **Method** | Deep learning blur | YOLOv8 + Gaussian blur | Attention-based inpainting | Generative Adversarial Network | **Latest SOTA Diffusion** |
| **GPU Required** | Yes (12GB+) | Yes (6GB+) | No (CPU optimized) | Optional (6GB+ for acceleration) | **Optional (Auto-detect)** |
| **Quality** | High | High | Very High (95% JPEG) | Highest (Realistic generation) | **State-of-the-Art** |
| **Speed** | Fast | Very Fast | Moderate | Medium (CPU), Fast (GPU) | **Fast (Optimized)** |
| **4K Support** | Yes | Yes | Yes | Optimized for 4K | **Yes** |
| **Memory Management** | Standard | Standard | Standard | Intelligent batching | **Advanced Optimization** |
| **Docker Tag** | `:face-anon-v3` | `:dashcam_anonymizer` | `:garnet` | `:deepprivacy` | **`:deepprivacy2`** |
| **Release Year** | 2023 | 2023 | 2023 | 2022 | **2024** |
| **Unique Features** | - | License plate detection | CPU-only operation | Realistic face generation | **Face + Full-body modes** |

### Complete Privacy Pipeline
```bash
# Option 1: Standard face anonymization pipeline
# 1. Face anonymization
sudo docker run --runtime=nvidia --gpus all \
  -v /input:/input -v /temp1:/output \
  sanjinn/privacy_removal_phase1:face-anon-v3

# 2. Text anonymization  
sudo docker run --rm \
  -v /temp1:/input -v /temp2:/output \
  sanjinn/privacy_removal_phase1:garnet

# 3. License plate anonymization (if dashcam footage)
sudo docker run --runtime=nvidia --gpus '"device=1"' \
  -v /temp2:/input -v /final_output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

```bash
# Option 2: High-quality GAN-based pipeline (recommended for 4K)
# 1. DeepPrivacy GAN face anonymization (highest quality)
sudo docker run --rm \
  -v /input:/workspace/input -v /temp1:/workspace/output \
  -v /deepprivacy/source:/workspace/deepprivacy \
  sanjinn/privacy_removal_phase1:deepprivacy \
  bash -c "cd /workspace/deepprivacy && ./batch_process.sh 2"

# 2. Text anonymization
sudo docker run --rm \
  -v /temp1:/input -v /temp2:/output \
  sanjinn/privacy_removal_phase1:garnet

# 3. License plate anonymization (if needed)
sudo docker run --runtime=nvidia --gpus '"device=1"' \
  -v /temp2:/input -v /final_output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

```bash
# Option 3: Latest state-of-the-art pipeline (✅ **RECOMMENDED - DeepPrivacy2**)
# 1. DeepPrivacy2 face anonymization (2024 SOTA)
docker run --rm \
  -v /input:/input -v /temp1:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/face.py \
  -i /input --output_path /output

# 2. Text anonymization
docker run --rm \
  -v /temp1:/input -v /temp2:/output \
  sanjinn/privacy_removal_phase1:garnet

# 3. License plate anonymization (if needed)
docker run --runtime=nvidia --gpus '"device=1"' \
  -v /temp2:/input -v /final_output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```

```bash
# Option 4: Complete anonymization pipeline (Face + Full-body + Text + License plates)
# 1. DeepPrivacy2 full-body anonymization (most comprehensive)
docker run --rm \
  -v /input:/input -v /temp1:/output \
  sanjinn/privacy_removal_phase1:deepprivacy2 \
  python3 anonymize.py configs/anonymizers/FB_cse.py \
  -i /input --output_path /output --visualize

# 2. Text anonymization
docker run --rm \
  -v /temp1:/input -v /temp2:/output \
  sanjinn/privacy_removal_phase1:garnet

# 3. License plate anonymization (final step)
docker run --runtime=nvidia --gpus '"device=1"' \
  -v /temp2:/input -v /final_output:/output \
  sanjinn/privacy_removal_phase1:dashcam_anonymizer
```
