# Use NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:22.08-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_HOME=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install correct version of OpenCV
RUN pip uninstall -y opencv-python opencv-python-headless || true && \
    pip install opencv-python-headless==4.5.1.48 

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir cython tqdm moviepy pyyaml \
    scikit-image albumentations addict

# Install DSFD and Detectron2 (face detector + keypoint detector)
RUN pip install git+https://github.com/facebookresearch/detectron2.git@0703e08a5f589f7503a3fbfce41309c80204eec8
RUN pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git

# Clone DeepPrivacy main project
WORKDIR /workspace
RUN git clone https://github.com/hukkelas/DeepPrivacy.git
WORKDIR /workspace/DeepPrivacy

# Install project dependencies and register module
RUN pip install -r requirements.txt && \
    pip install -e .

# Download recommended pretrained model
RUN mkdir -p pretrained_models && \
    curl -L -o pretrained_models/model.pkl https://deepprivacy.s3.eu-west-1.amazonaws.com/models/fdf128_rcnn512/model.pkl

# Create input and output directories
RUN mkdir -p /workspace/input /workspace/output

# Default startup shell
CMD ["/bin/bash"]
