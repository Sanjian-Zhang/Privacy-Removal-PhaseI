# 如何处理自己的图片 - GaRNet 使用指南

## 必需文件

要使用GaRNet处理你自己的图片，你需要提供以下两种文件：

### 1. 图片文件 (.jpg)
- **格式**: JPG/JPEG
- **位置**: 放在一个文件夹中（例如：`./my_images/`）
- **命名**: 任意名称（例如：`image1.jpg`, `photo.jpg`）

### 2. 文本区域标注文件 (.txt)
- **格式**: 纯文本文件
- **位置**: 放在另一个文件夹中（例如：`./my_boxes/`）
- **命名**: 必须与对应图片文件同名（例如：`image1.txt`, `photo.txt`）
- **内容格式**: 每行包含一个文本区域的四个角点坐标

## 文本标注文件格式详解

### 坐标格式
每行表示一个文本区域，包含8个数字（4个角点的x,y坐标）：
```
x1,y1,x2,y2,x3,y3,x4,y4
```

### 角点顺序
坐标按照顺时针或逆时针顺序排列，通常为：
- (x1,y1): 左上角
- (x2,y2): 右上角  
- (x3,y3): 右下角
- (x4,y4): 左下角

### 示例文件内容
```
# 文件: image1.txt
136,122,384,118,384,178,136,178
121,225,401,225,401,275,121,275
186,290,335,290,335,339,186,339
```

这表示图片中有3个文本区域需要被移除。

## 如何获取文本区域坐标

### 方法1: 手动标注
使用图像标注工具（如labelImg、VIA等）手动框选文本区域并导出坐标。

### 方法2: 自动检测（推荐）
使用文本检测工具自动检测：

#### 使用CRAFT文本检测
```python
# 示例代码：使用CRAFT检测文本区域
import cv2
import numpy as np
# ... CRAFT相关导入

def detect_text_regions(image_path):
    # 加载图片
    image = cv2.imread(image_path)
    
    # 使用CRAFT检测文本
    # ... CRAFT检测代码
    
    # 返回检测到的文本框坐标
    return text_boxes

def save_boxes_to_txt(boxes, output_path):
    with open(output_path, 'w') as f:
        for box in boxes:
            # 将box坐标转换为所需格式
            line = ','.join([str(int(coord)) for coord in box.flatten()])
            f.write(line + '\\n')
```

#### 使用PaddleOCR检测
```python
from paddleocr import PaddleOCR

def detect_with_paddleocr(image_path, output_txt_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(image_path, cls=True)
    
    with open(output_txt_path, 'w') as f:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                box = line[0]  # 获取边界框坐标
                # 转换为所需格式：x1,y1,x2,y2,x3,y3,x4,y4
                coords = []
                for point in box:
                    coords.extend([int(point[0]), int(point[1])])
                f.write(','.join(map(str, coords)) + '\\n')
```

## 文件组织结构

```
your_project/
├── images/          # 存放图片文件
│   ├── image1.jpg
│   ├── image2.jpg
│   └── photo.jpg
├── boxes/           # 存放对应的坐标文件
│   ├── image1.txt
│   ├── image2.txt
│   └── photo.txt
└── results/         # 输出结果文件夹
```

## 运行处理命令

### 基本命令
```bash
cd /home/zhiqics/sanjian/baseline/garnet
source garnet_env/bin/activate

python CODE/inference.py \\
    --image_path ./your_images \\
    --box_path ./your_boxes \\
    --result_path ./your_results \\
    --input_size 512 \\
    --model_path ./WEIGHTS/GaRNet/saved_model.pth \\
    --gpu
```

### 参数说明
- `--image_path`: 图片文件夹路径
- `--box_path`: 坐标文件夹路径  
- `--result_path`: 结果输出路径
- `--input_size`: 输入图片大小（默认512）
- `--model_path`: 模型权重文件路径
- `--gpu`: 使用GPU加速（可选）
- `--attention_vis`: 可视化注意力图（可选）

## 创建自动化脚本

我为你创建了一个便捷脚本来处理图片：

### 使用PaddleOCR自动检测并处理
```bash
python process_my_images.py \\
    --input_dir ./my_images \\
    --output_dir ./processed_results
```

### 手动提供坐标文件
```bash
python process_my_images.py \\
    --input_dir ./my_images \\
    --box_dir ./my_boxes \\
    --output_dir ./processed_results
```

## 注意事项

1. **文件名对应**: 图片文件和坐标文件必须同名（除了扩展名）
2. **坐标精度**: 坐标应该是整数像素值
3. **图片格式**: 支持JPG格式，其他格式可能需要转换
4. **坐标范围**: 确保坐标在图片范围内
5. **空文件**: 如果图片中没有文本，创建空的.txt文件

## 故障排除

### 常见问题
1. **"No such file"错误**: 检查文件路径和文件名是否正确
2. **坐标格式错误**: 确保每行有8个用逗号分隔的数字
3. **内存不足**: 降低batch_size或图片分辨率
4. **GPU错误**: 去掉--gpu参数使用CPU模式

### 验证数据格式
运行以下命令检查数据格式：
```bash
python validate_data.py --image_dir ./your_images --box_dir ./your_boxes
```
