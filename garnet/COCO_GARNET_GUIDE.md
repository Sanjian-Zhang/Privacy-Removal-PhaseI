# COCO标注标签与Garnet模型结合使用指南

## 概述

本项目成功实现了基于COCO标注文件的图片处理流程，结合Garnet模型进行文本去除等任务。

## 数据集信息

### COCO标注文件结构
- **位置**: `/home/zhiqics/sanjian/dataset/annotations/instances_Train.json`
- **图片目录**: `/home/zhiqics/sanjian/dataset/images/Train/`
- **总图片数**: 394张
- **总标注数**: 3524个

### 类别定义
| ID | 名称 | 描述 | 标注数量 |
|----|------|------|----------|
| 1  | face | 人脸 | 2530 |
| 2  | license_plate | 车牌 | 143 |
| 3  | text | 文本 | 851 |

## 处理脚本

### 1. 完整处理脚本 (`process_coco_dataset.py`)

**功能**：
- 自动加载COCO标注文件
- 支持多种处理模式：可视化、文本去除、面部匿名化
- 结合Garnet模型进行文本去除

**使用方法**：
```bash
cd /home/zhiqics/sanjian/baseline/garnet
python process_coco_dataset.py --text_removal --visualize --max_images 10
```

**参数说明**：
- `--text_removal`: 启用Garnet文本去除
- `--visualize`: 生成可视化结果
- `--max_images N`: 处理前N张图片

### 2. 无损处理脚本 (`process_coco_dataset_lossless.py`)

**特点**：
- PNG格式输出，保证图像质量无损
- 适合需要高质量结果的场景

**使用方法**：
```bash
python process_coco_dataset_lossless.py --text_removal --visualize --max_images 5
```

### 3. 演示脚本 (`demo_coco_garnet.py`)

**功能**：
- 处理单张图片的完整演示
- 展示COCO标注解析和Garnet输入准备流程

**使用方法**：
```bash
python demo_coco_garnet.py --image_name frame_00030.jpg
```

## 处理流程

### 1. 标注数据加载
```python
# 加载COCO标注文件
categories, images, annotations_by_image = load_coco_data(annotation_file)

# 类别映射
{1: 'face', 2: 'license_plate', 3: 'text'}
```

### 2. 文本区域提取
```python
# 筛选文本类别标注 (category_id=3)
text_annotations = filter_text_annotations(annotations, text_category_id=3)

# 转换为Garnet输入格式
# bbox格式：[x, y, width, height] -> [x1,y1,x2,y1,x2,y2,x1,y2]
```

### 3. Garnet模型处理
```python
# 创建输入文件
create_garnet_input(image, text_annotations, txt_path, img_path)

# Garnet处理命令
python garnet_inference.py --input_image image.jpg --input_txt regions.txt --output_dir output/
```

## 处理结果

### 成功处理示例
- **处理图片数**: 10张
- **检测文本区域**: 7个
- **生成文件**: 50个处理结果

### 输出目录结构
```
/home/zhiqics/sanjian/dataset/images/Processed/
├── visualized/          # 可视化标注结果
├── garnet_input_txt/     # Garnet文本区域定义文件
├── garnet_input_img/     # Garnet输入图片
└── garnet_output/        # Garnet处理结果
```

### 无损处理结果
```
/home/zhiqics/sanjian/dataset/images/ProcessedLossless/
├── visualized/          # PNG格式可视化结果 (394个文件)
├── garnet_input_txt/     # 文本区域定义 (394个文件)  
├── garnet_input_img/     # 输入图片 (394个文件)
└── garnet_output/        # 高质量处理结果
```

## 关键技术点

### 1. COCO标注解析
- 处理大型JSON文件（3524个标注）
- 建立图片-标注索引关系
- 类别映射和筛选

### 2. 坐标转换
```python
# COCO bbox格式: [x, y, width, height]
bbox = ann['bbox']
x, y, w, h = bbox

# 转换为Garnet格式: [x1,y1,x2,y1,x2,y2,x1,y2]
x1, y1 = int(x), int(y)
x2, y2 = int(x + w), int(y + h)
garnet_format = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2}"
```

### 3. 可视化标注
- 不同类别使用不同颜色
- 绘制边界框和标签
- 支持多类别同时显示

## 使用建议

### 1. 批量处理
```bash
# 处理所有图片（无限制）
python process_coco_dataset.py --text_removal --visualize

# 处理指定数量
python process_coco_dataset.py --text_removal --visualize --max_images 100
```

### 2. 高质量需求
```bash
# 使用无损版本
python process_coco_dataset_lossless.py --text_removal --visualize
```

### 3. 单图片调试
```bash
# 使用演示脚本
python demo_coco_garnet.py --image_name your_image.jpg
```

## 扩展功能

脚本支持以下扩展：
- 面部匿名化处理
- 车牌检测和处理
- 自定义类别筛选
- 批量输出格式控制

## 总结

通过结合COCO标注标签和Garnet模型，我们成功实现了：

1. **自动化处理流程**: 从标注解析到模型输入准备的完整自动化
2. **多类别支持**: 人脸、车牌、文本的统一处理框架
3. **质量控制**: 提供有损和无损两种处理模式
4. **可视化验证**: 生成标注可视化结果便于验证
5. **灵活配置**: 支持多种处理选项和参数配置

这个处理流程为基于COCO标注的图像处理任务提供了完整的解决方案。
