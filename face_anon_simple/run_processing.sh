#!/bin/bash

# 激活 conda 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate face-anon-simple

# 运行处理脚本
cd /workspace/face_anon_simple
python process_images.py
