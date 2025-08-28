FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libglib2.0-dev libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
WORKDIR /root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# 初始化 Conda 环境 + 使用 conda-forge 避开 TOS
RUN conda init bash && \
    conda config --set always_yes true && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# 设置工作目录并复制项目文件
WORKDIR /workspace
COPY . /workspace/face_anon_simple/
WORKDIR /workspace/face_anon_simple

# 使用 bash shell 激活并安装依赖
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda env create -f environment.yml && \
    echo "conda activate face-anon-simple" >> ~/.bashrc

# 设置执行权限
RUN chmod +x /workspace/face_anon_simple/run_processing.sh

# 设置启动命令
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate face-anon-simple && python process_images.py"]
