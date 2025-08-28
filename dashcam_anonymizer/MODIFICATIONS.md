# Dashcam Anonymizer Project Modifications

## Project Overview
This document records all modifications made to the original [dashcam_anonymizer](https://github.com/varungupta31/dashcam_anonymizer) project to adapt it to our specific requirements and environment.

## Modification List

### 1. Dockerization

#### New Files Added
- `Dockerfile` - Container configuration file
- `requirements.txt` - Python dependency management
- `run_dashcam_anonymizer.sh` - Execution script

#### Dockerfile Features
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```
- Based on official PyTorch CUDA image
- Automatic installation of system dependencies (OpenCV-related libraries)
- Automatic YOLO model download
- GPU acceleration support

### 2. Configuration File Modifications (`configs/img_blur.yaml`)

| Parameter | Original Value | Modified Value | Reason for Change |
|-----------|----------------|----------------|-------------------|
| `images_path` | `'images/'` | `'/input/'` | Adapt to Docker container mount path |
| `gpu_avail` | `False` | `True` | Enable GPU acceleration |
| `img_format` | `.png` | `.jpg` | Match actual data format |
| `img_width` | `960` | `3840` | Adapt to 4K resolution images |
| `img_height` | `540` | `2160` | Adapt to 4K resolution images |
| `output_folder` | `blurred_images` | `/output` | Adapt to Docker container mount path |
| `detection_conf_thresh` | `0.1` | `0.1` | **Unchanged** |
| `blur_radius` | `31` | `31` | **Unchanged** |

### 3. Code Functionality Enhancement (`blur_images.py`)

#### Improved Image Format Support
```python
# Original code only supported single format specified in config
images = sorted(glob.glob(config['images_path']+"/*"+config["img_format"]))

# Modified to support multiple image formats
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
all_images = []
for fmt in supported_formats:
    all_images.extend(glob.glob(config['images_path'] + "/*" + fmt))
    all_images.extend(glob.glob(config['images_path'] + "/*" + fmt.upper()))
```

#### Enhanced Error Handling
- Added image count checking and notifications
- Improved file matching logic
- Added detailed processing progress output
- Optimized exception handling mechanisms

#### File Matching Optimization
```python
# Smart matching of corresponding image files, supporting multiple formats
base_name = os.path.splitext(txt_file)[0]
image_file = None
for fmt in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']:
    potential_image = base_name + fmt
    image_path = os.path.join(image_folder, potential_image)
    if os.path.exists(image_path):
        image_file = potential_image
        break
```

### 4. Environment Configuration Optimization

#### Python Dependency Version Locking
- Locked PyTorch version to 2.0.1 (compatible with CUDA 11.7)
- Optimized OpenCV version to 4.8.0.76
- Unified NumPy version to 1.25.1

#### System Dependency Handling
- Automatic installation of OpenCV required system libraries
- Optimized Docker layer caching
- Reduced image size

### 5. Execution Method Improvements

#### Original Execution Method
```bash
conda create --name dashanon python=3.11
conda activate dashanon
pip install -r requirements.txt
python blur_images.py --config configs/img_blur.yaml
```

#### Docker Execution Method
```bash
# Build image
sudo docker build -t dashcam_anonymizer .

# Run processing
sudo docker run --runtime=nvidia --gpus '"device=1"' \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  dashcam_anonymizer
```

### 6. Directory Structure Adaptation

#### Input/Output Path Mapping
- Input directory: `/home/zhiqics/sanjian/test_dataset/images/Train` → `/input`
- Output directory: `/home/zhiqics/sanjian/output/dashcam_results` → `/output`

### 7. Performance Optimization

#### GPU Configuration
- Enabled NVIDIA GPU support
- Specified use of GPU device 1
- Optimized CUDA memory usage

#### Processing Optimization
- Support for batch processing multiple format images
- Improved memory management
- Optimized file I/O operations

## Unchanged Core Functionality

The following core parameters and functionality remain consistent with the original project:
- Detection confidence threshold (0.1)
- Blur intensity (31 pixel radius)
- YOLO model (best.pt)
- Core detection and blurring algorithms
- Output file naming conventions

## Usage Instructions

1. **Build Docker Image**
   ```bash
   cd /home/zhiqics/sanjian/testdocker/dashcam_anonymizer
   sudo docker build -t dashcam_anonymizer .
   ```

2. **Run Processing**
   ```bash
   sudo docker run --runtime=nvidia --gpus '"device=1"' \
     -v /home/zhiqics/sanjian/test_dataset/images/Train:/input \
     -v /home/zhiqics/sanjian/output/dashcam_results:/output \
     dashcam_anonymizer
   ```

3. **Or Use Convenience Script**
   ```bash
   /home/zhiqics/sanjian/testdocker/run_dashcam_anonymizer.sh
   ```

## Compatibility Notes

- ✅ Compatible with original project's core functionality
- ✅ Supports original project's model files
- ✅ Maintains original project's detection accuracy
- ✅ Output format consistent with original project
- ✅ Works with different resolution images

## Technology Stack

- **Containerization**: Docker + NVIDIA Container Runtime
- **Deep Learning**: PyTorch 2.0.1 + CUDA 11.7
- **Computer Vision**: OpenCV 4.8.0 + Ultralytics YOLOv8
- **Runtime Environment**: Ubuntu + NVIDIA GPU

---

*Created: August 25, 2025*  
*Project Path: `/home/zhiqics/sanjian/testdocker/dashcam_anonymizer`*
