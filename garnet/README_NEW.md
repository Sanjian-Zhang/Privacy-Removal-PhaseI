# Garnet Text Anonymizer - Docker Version

## Project Overview

This project is a Dockerized text anonymization solution based on [NAVER Garnet](https://github.com/naver/garnet). The project encapsulates the original Garnet model in Docker containers, enabling intelligent detection and anonymization of text regions.

## 🚀 Key Improvements

### 1. Docker Environment Configuration
- Simplified base image: `ubuntu:20.04`
- CPU version PyTorch for better compatibility
- Resolved complex CUDA environment issues

### 2. Intelligent Text Processing
- Precise text region detection and processing
- Selective anonymization (only text regions, not full image)
- Support for annotation files format: `x1,y1,x2,y2,x3,y3,x4,y4`

### 3. Multi-version Processing Scripts
- **process_images.py** - General image processing
- **process_demo.py** - Demo data processing
- **process_train.py** - Training dataset processing

## 📦 Docker Images

### Available Images
- **garnet_text_anonymizer_simple** - Basic version
- **garnet_demo** - Demo testing version  
- **garnet_train** - Training data processing version

### Build Commands
```bash
sudo docker build -f Dockerfile.simple -t garnet_text_anonymizer_simple .
sudo docker build -f Dockerfile.demo -t garnet_demo .
sudo docker build -f Dockerfile.train -t garnet_train .
```

## 🔧 Usage

```bash
# Processing Training Dataset
sudo docker run --rm -v /path/to/input:/input -v /path/to/output:/output garnet_train

# Processing Demo Data
sudo docker run --rm -v /path/to/demo:/input -v /path/to/output:/output garnet_demo

# General Image Processing
sudo docker run --rm -v /path/to/images:/input -v /path/to/results:/output garnet_text_anonymizer_simple
```

## 📊 Results

- ✅ **394/394** images successfully processed
- ✅ Processing time: ~2-3 seconds/image (CPU mode)
- ✅ High-quality output with precise text anonymization
- ✅ Supported formats: JPG, JPEG, PNG, BMP, TIFF

## 🛠️ Technical Details

- **Base Environment**: Ubuntu 20.04
- **Deep Learning**: PyTorch (CPU version)
- **Image Processing**: OpenCV, PIL
- **Containerization**: Docker
- **Model Source**: HuggingFace Hub
