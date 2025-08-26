# GaRNet - 场景文本去除

GaRNet是一个基于深度学习的场景文本去除模型，使用Gated Attention机制。

## 环境设置

1. **克隆项目**
   ```bash
   git clone https://github.com/naver/garnet.git
   cd garnet
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv garnet_env
   source garnet_env/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install --upgrade pip setuptools
   pip install numpy opencv-python torch torchvision ptflops scikit-image huggingface_hub
   ```

4. **下载预训练模型**
   ```bash
   python download_model.py
   ```

## 快速开始

### 运行示例
```bash
python run_example.py
```

### 基本推理
```bash
cd CODE
python inference.py --gpu \
    --image_path ../DATA/EXAMPLE/IMG \
    --box_path ../DATA/EXAMPLE/TXT \
    --result_path ./results
```

### 测试模型
```bash
python test_model.py
```

## 文件结构

```
garnet/
├── CODE/                    # 核心代码
│   ├── model.py            # GaRNet模型定义
│   ├── inference.py        # 推理脚本
│   ├── eval.py            # 评估脚本
│   └── utils.py           # 工具函数
├── DATA/                   # 数据目录
│   └── EXAMPLE/           # 示例数据
│       ├── IMG/           # 输入图像
│       └── TXT/           # 文本框坐标
├── WEIGHTS/               # 预训练模型
│   └── GaRNet/
│       └── saved_model.pth
├── garnet_env/            # Python虚拟环境
├── download_model.py      # 模型下载脚本
├── test_model.py         # 模型测试脚本
└── run_example.py        # 完整示例脚本
```

## 输入格式

### 图像文件
- 支持格式: JPG, PNG
- 建议尺寸: 512x512 (模型会自动调整)

### 文本框坐标文件 (.txt)
每行包含一个文本框的4个顶点坐标:
```
x1,y1,x2,y2,x3,y3,x4,y4
```

## 模型参数

- **输入通道**: 3 (RGB图像)
- **推理尺寸**: 512x512
- **设备**: 支持CPU和GPU

## 主要功能

1. **文本去除**: 从图像中去除指定的文本区域
2. **注意力可视化**: 可选的注意力图可视化
3. **批量处理**: 支持批量图像处理

## 使用提示

1. 确保输入图像清晰，文本区域明确
2. 文本框坐标要准确标注
3. 使用GPU可显著提升处理速度
4. 对于复杂背景，可能需要调整参数

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用CPU模式: 去掉--gpu参数

2. **模型加载失败**
   - 重新下载模型: `python download_model.py`
   - 检查模型文件路径

3. **依赖安装失败**
   - 升级pip: `pip install --upgrade pip`
   - 分别安装各个包

## 性能提示

- 使用GPU可将推理时间从几秒降低到毫秒级
- 较大图像会自动缩放到512x512进行处理
- 批量处理可提高整体效率

## 引用

如果使用此项目，请引用原论文:
```
@inproceedings{lee2022surprisingly,
  title={The Surprisingly Straightforward Scene Text Removal Method with Gated Attention and Region of Interest Generation: A Comprehensive Prominent Model Analysis},
  author={Lee, Hyeonsu and Choi, Chankyu},
  booktitle={European Conference on Computer Vision},
  pages={457--472},
  year={2022},
  organization={Springer}
}
```

## 项目状态

✅ 环境配置完成
✅ 模型下载成功  
✅ 推理测试通过
✅ 示例运行成功
