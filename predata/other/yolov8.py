import os
import cv2
import shutil
from pathlib import Path
import numpy as np

# 尝试导入不同的人脸检测库
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: ultralytics 未安装，将使用 OpenCV 的人脸检测")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class FaceImageClassifier:
    def __init__(self, detection_method="yolo", min_face_size=50, confidence_threshold=0.5, model_path=None):
        """
        初始化人脸检测图片分类器
        
        Args:
            detection_method: 检测方法 ("opencv", "face_recognition", "mediapipe", "yolo")
            min_face_size: 最小人脸尺寸，用于过滤远方的小脸（像素）
            confidence_threshold: 检测置信度阈值
            model_path: YOLOv8模型路径
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.detection_method = detection_method.lower()
        self.model_path = model_path
        
        # 初始化不同的检测器
        self._init_detectors()
        
    def _init_detectors(self):
        """初始化各种检测器"""
        
        # OpenCV Haar Cascade (最稳定的方法)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            print("✓ OpenCV 人脸检测器初始化成功")
        except Exception as e:
            print(f"OpenCV 初始化失败: {e}")
            
        # MediaPipe Face Detection
        if MEDIAPIPE_AVAILABLE and self.detection_method == "mediapipe":
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=self.confidence_threshold
            )
            print("✓ MediaPipe 人脸检测器初始化成功")
            
        # YOLOv8 (如果指定使用)
        if YOLO_AVAILABLE and self.detection_method == "yolo":
            try:
                # 检查 CUDA 兼容性
                import torch
                cuda_available = torch.cuda.is_available()
                
                if cuda_available:
                    try:
                        # 测试基本 CUDA 操作
                        test_tensor = torch.tensor([1.0]).cuda()
                        del test_tensor
                        print("✓ CUDA 测试通过")
                    except Exception as cuda_error:
                        print(f"⚠️  CUDA 不兼容，将强制使用 CPU: {cuda_error}")
                        cuda_available = False
                        # 设置环境变量强制使用 CPU
                        os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                if self.model_path and os.path.exists(self.model_path):
                    self.yolo_model = YOLO(self.model_path)
                    print(f"✓ YOLOv8 人脸检测器初始化成功，使用模型: {self.model_path}")
                else:
                    # 尝试其他可能的模型路径
                    possible_models = ['yolov8n-face.pt', 'yolov8s-face.pt', 'yolov8m-face.pt', 'yolov8n.pt']
                    model_loaded = False
                    
                    for model_name in possible_models:
                        try:
                            self.yolo_model = YOLO(model_name)
                            print(f"✓ YOLOv8 检测器初始化成功，使用模型: {model_name}")
                            model_loaded = True
                            break
                        except:
                            continue
                    
                    if not model_loaded:
                        print("❌ YOLOv8 模型加载失败，将使用 OpenCV")
                        self.detection_method = "opencv"
                        return
                
                # 强制将模型设置为适当的设备
                device = 'cpu' if not cuda_available else 'cuda'
                if hasattr(self.yolo_model, 'to'):
                    self.yolo_model = self.yolo_model.to(device)
                    print(f"✓ YOLOv8 模型设置为 {device.upper()} 模式")
                        
            except Exception as e:
                print(f"YOLOv8 初始化失败: {e}，将使用 OpenCV")
                self.detection_method = "opencv"
    
    def detect_faces_opencv(self, image_path):
        """使用 OpenCV 检测人脸"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0, []
                
            # 转为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            valid_faces = []
            
            # 使用正面人脸检测器
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 使用侧面人脸检测器
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 合并检测结果
            all_faces = list(faces) + list(profile_faces)
            
            # 过滤重叠的检测框
            for (x, y, w, h) in all_faces:
                # 检查是否与已有的人脸重叠
                is_duplicate = False
                for existing_face in valid_faces:
                    ex, ey, ew, eh = existing_face['bbox']
                    # 计算重叠度
                    overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                    overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.3 * min(w * h, ew * eh):  # 30% 重叠认为是同一个人脸
                        is_duplicate = True
                        break
                
                if not is_duplicate and w >= self.min_face_size and h >= self.min_face_size:
                    # 过滤边缘检测（可能是误检）
                    if x > width * 0.02 and y > height * 0.02:
                        valid_faces.append({
                            'bbox': (x, y, x + w, y + h),
                            'confidence': 0.8,  # OpenCV 不提供置信度，给个估值
                            'width': w,
                            'height': h
                        })
            
            return len(valid_faces), valid_faces
            
        except Exception as e:
            print(f"OpenCV 检测出错 {image_path}: {str(e)}")
            return 0, []
    
    def detect_faces_face_recognition(self, image_path):
        """使用 face_recognition 库检测人脸"""
        try:
            # 加载图片
            image = face_recognition.load_image_file(image_path)
            height, width = image.shape[:2]
            
            # 检测人脸位置
            face_locations = face_recognition.face_locations(image)
            
            valid_faces = []
            for (top, right, bottom, left) in face_locations:
                face_width = right - left
                face_height = bottom - top
                
                if face_width >= self.min_face_size and face_height >= self.min_face_size:
                    valid_faces.append({
                        'bbox': (left, top, right, bottom),
                        'confidence': 0.9,
                        'width': face_width,
                        'height': face_height
                    })
            
            return len(valid_faces), valid_faces
            
        except Exception as e:
            print(f"face_recognition 检测出错 {image_path}: {str(e)}")
            return 0, []
    
    def detect_faces_mediapipe(self, image_path):
        """使用 MediaPipe 检测人脸"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0, []
                
            height, width = image.shape[:2]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.face_detection.process(rgb_image)
            valid_faces = []
            
            if results.detections:
                for detection in results.detections:
                    confidence = detection.score[0]
                    
                    if confidence >= self.confidence_threshold:
                        bbox = detection.location_data.relative_bounding_box
                        
                        # 转换为绝对坐标
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        
                        if w >= self.min_face_size and h >= self.min_face_size:
                            valid_faces.append({
                                'bbox': (x, y, x + w, y + h),
                                'confidence': confidence,
                                'width': w,
                                'height': h
                            })
            
            return len(valid_faces), valid_faces
            
        except Exception as e:
            print(f"MediaPipe 检测出错 {image_path}: {str(e)}")
            return 0, []
    
    def detect_faces_yolo(self, image_path):
        """使用 YOLOv8 检测人脸，包含 CUDA 错误处理"""
        try:
            # 首先尝试使用 GPU
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 检查 CUDA 是否真正可用
            if device == 'cuda':
                try:
                    # 测试 CUDA 操作
                    test_tensor = torch.tensor([1.0]).cuda()
                    del test_tensor
                except Exception as cuda_test_error:
                    print(f"CUDA 测试失败，切换到 CPU: {cuda_test_error}")
                    device = 'cpu'
            
            # 强制设置模型设备
            if hasattr(self.yolo_model, 'to'):
                self.yolo_model = self.yolo_model.to(device)
            
            # 运行推理，明确指定设备
            results = self.yolo_model(image_path, conf=self.confidence_threshold, device=device)
            
            if not results:
                return 0, []
            
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            valid_faces = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        if box_width >= self.min_face_size and box_height >= self.min_face_size:
                            valid_faces.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence,
                                'width': box_width,
                                'height': box_height
                            })
            
            return len(valid_faces), valid_faces
            
        except Exception as e:
            error_msg = str(e)
            print(f"YOLO 检测出错 {image_path}: {error_msg}")
            
            # 如果是 CUDA 相关错误，尝试强制使用 CPU
            if "CUDA" in error_msg.upper() or "kernel" in error_msg.lower():
                print("检测到 CUDA 错误，尝试使用 CPU 模式...")
                try:
                    # 重新初始化模型到 CPU
                    if hasattr(self, 'yolo_model'):
                        self.yolo_model = self.yolo_model.to('cpu')
                    
                    # 强制使用 CPU 进行推理
                    results = self.yolo_model(image_path, conf=self.confidence_threshold, device='cpu')
                    
                    if not results:
                        return 0, []
                    
                    img = cv2.imread(image_path)
                    valid_faces = []
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                
                                box_width = x2 - x1
                                box_height = y2 - y1
                                
                                if box_width >= self.min_face_size and box_height >= self.min_face_size:
                                    valid_faces.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': confidence,
                                        'width': box_width,
                                        'height': box_height
                                    })
                    
                    print(f"CPU 模式成功检测到 {len(valid_faces)} 张人脸")
                    return len(valid_faces), valid_faces
                    
                except Exception as cpu_error:
                    print(f"CPU 模式也失败: {cpu_error}")
                    return 0, []
            
            return 0, []
    
    def detect_faces(self, image_path):
        """
        主要的人脸检测函数，会尝试多种方法
        """
        # 首先尝试指定的方法
        if self.detection_method == "opencv":
            face_count, detections = self.detect_faces_opencv(image_path)
        elif self.detection_method == "face_recognition" and FACE_RECOGNITION_AVAILABLE:
            face_count, detections = self.detect_faces_face_recognition(image_path)
        elif self.detection_method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            face_count, detections = self.detect_faces_mediapipe(image_path)
        elif self.detection_method == "yolo" and YOLO_AVAILABLE:
            face_count, detections = self.detect_faces_yolo(image_path)
        else:
            # 默认使用 OpenCV
            face_count, detections = self.detect_faces_opencv(image_path)
        
        # 如果第一种方法没检测到，尝试其他方法
        if face_count == 0:
            print(f"  尝试备用检测方法...")
            
            # 尝试 OpenCV（如果不是主方法）
            if self.detection_method != "opencv":
                face_count, detections = self.detect_faces_opencv(image_path)
                if face_count > 0:
                    print(f"  OpenCV 检测到 {face_count} 张人脸")
                    return face_count, detections
            
            # 尝试 face_recognition（如果可用且不是主方法）
            if FACE_RECOGNITION_AVAILABLE and self.detection_method != "face_recognition":
                face_count, detections = self.detect_faces_face_recognition(image_path)
                if face_count > 0:
                    print(f"  face_recognition 检测到 {face_count} 张人脸")
                    return face_count, detections
        
        return face_count, detections
    
    def test_image(self, image_path):
        """测试单张图片的检测效果"""
        print(f"\n测试图片: {image_path}")
        
        if not os.path.exists(image_path):
            print("图片文件不存在！")
            return
        
        # 尝试所有可用的方法
        methods = [
            ("OpenCV", self.detect_faces_opencv),
            ("face_recognition", self.detect_faces_face_recognition if FACE_RECOGNITION_AVAILABLE else None),
            ("MediaPipe", self.detect_faces_mediapipe if MEDIAPIPE_AVAILABLE else None),
            ("YOLOv8", self.detect_faces_yolo if YOLO_AVAILABLE else None)
        ]
        
        for method_name, method_func in methods:
            if method_func is None:
                continue
                
            try:
                face_count, detections = method_func(image_path)
                print(f"{method_name:15}: {face_count} 张人脸")
                
                if face_count > 0 and detections:
                    for i, face in enumerate(detections):
                        bbox = face['bbox']
                        conf = face['confidence']
                        size = f"{face['width']:.0f}x{face['height']:.0f}"
                        print(f"  人脸 {i+1}: 位置{bbox}, 置信度{conf:.2f}, 尺寸{size}")
                        
            except Exception as e:
                print(f"{method_name:15}: 检测失败 - {str(e)}")
    
    def classify_and_save_images(self, input_dir, output_dir):
        """分类并保存图片"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        high_quality_dir = output_path / "high_quality" / "more_than_3_faces"
        low_quality_dir = output_path / "low_quality" / "3_or_less_faces"
        no_face_dir = output_path / "no_faces"
        
        high_quality_dir.mkdir(parents=True, exist_ok=True)
        low_quality_dir.mkdir(parents=True, exist_ok=True)
        no_face_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 统计信息
        stats = {
            'total': 0,
            'high_quality': 0,
            'low_quality': 0,
            'no_faces': 0,
            'errors': 0
        }
        
        print(f"使用检测方法: {self.detection_method.upper()}")
        print("开始处理图片...\n")
        
        # 遍历输入目录中的所有图片
        for image_file in input_path.rglob('*'):
            if image_file.suffix.lower() in image_extensions:
                stats['total'] += 1
                
                try:
                    print(f"处理 [{stats['total']}]: {image_file.name}")
                    
                    # 检测人脸
                    face_count, detections = self.detect_faces(str(image_file))
                    
                    # 根据人脸数量分类
                    if face_count > 3:
                        dest_dir = high_quality_dir
                        stats['high_quality'] += 1
                        print(f"  ✓ 高质量图片，检测到 {face_count} 张人脸")
                        
                    elif face_count > 0:
                        dest_dir = low_quality_dir
                        stats['low_quality'] += 1
                        print(f"  → 普通图片，检测到 {face_count} 张人脸")
                        
                    else:
                        dest_dir = no_face_dir
                        stats['no_faces'] += 1
                        print(f"  ✗ 无人脸图片")
                    
                    # 复制图片（不压缩）
                    dest_file = dest_dir / image_file.name
                    
                    # 处理文件名冲突
                    counter = 1
                    original_dest = dest_file
                    while dest_file.exists():
                        name_stem = original_dest.stem
                        suffix = original_dest.suffix
                        dest_file = dest_dir / f"{name_stem}_{counter}{suffix}"
                        counter += 1
                    
                    # 移动文件（保持原始质量）
                    shutil.move(str(image_file), str(dest_file))
                    
                except Exception as e:
                    print(f"  ✗ 处理出错: {str(e)}")
                    stats['errors'] += 1
        
        # 打印统计结果
        print("\n" + "="*60)
        print("处理完成！统计结果：")
        print(f"总图片数: {stats['total']}")
        print(f"高质量图片 (>3张人脸): {stats['high_quality']}")
        print(f"普通图片 (1-3张人脸): {stats['low_quality']}")
        print(f"无人脸图片: {stats['no_faces']}")
        print(f"处理出错: {stats['errors']}")
        print(f"\n图片已保存到: {output_path}")


def main():
    """主函数"""    
    # 配置参数
    input_directory = "/home/zhiqics/sanjian/predata/output_frames14"
    output_directory = "/home/zhiqics/sanjian/predata/clasical_14"
    model_path = "/home/zhiqics/sanjian/test_dlip/models/yolov8n-face.pt"
    
    # 选择检测方法: "opencv", "face_recognition", "mediapipe", "yolo"
    detection_method = "yolo"    # 使用你的 YOLOv8 人脸模型
    min_face_size = 30           # 降低最小人脸尺寸阈值
    confidence_threshold = 0.3   # 降低置信度阈值
    
    # 创建分类器实例
    classifier = FaceImageClassifier(
        detection_method=detection_method,
        min_face_size=min_face_size,
        confidence_threshold=confidence_threshold
    )
    
    # 检查输入目录
    if not os.path.exists(input_directory):
        print(f"输入目录不存在: {input_directory}")
        print("请创建输入目录并放入图片文件")
        
        # 创建测试目录和提示
        os.makedirs(input_directory, exist_ok=True)
        print(f"已创建目录: {input_directory}")
        return
    
    # 检查是否有图片文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in Path(input_directory).rglob('*') 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"在 {input_directory} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 测试第一张图片（调试用）
    if image_files:
        print("=" * 60)
        print("测试第一张图片的检测效果：")
        classifier.test_image(str(image_files[0]))
        print("=" * 60)
        
        # 询问是否继续
        response = input("\n是否继续处理所有图片？(y/n): ").lower()
        if response != 'y':
            return
    
    # 执行分类
    classifier.classify_and_save_images(input_directory, output_directory)


if __name__ == "__main__":
    main()


# 安装说明和解决方案
"""
如果检测不到人脸，请按以下步骤排查：

1. 安装依赖包：
   # 基础版本 (只用 OpenCV)
   pip install opencv-python
   
   # 完整版本 (推荐)
   pip install opencv-python face-recognition mediapipe ultralytics

2. 常见问题解决：
   
   问题1: OpenCV 检测不到人脸
   解决: 降低 min_face_size 和 confidence_threshold 参数
   
   问题2: 图片太大或太小
   解决: 脚本会自动处理不同尺寸的图片
   
   问题3: 人脸角度问题
   解决: 脚本同时使用正面和侧面检测器
   
   问题4: 光线问题
   解决: OpenCV 对光线变化有一定适应性

3. 测试建议：
   - 先用几张明确有人脸的图片测试
   - 查看测试输出，了解各种方法的效果
   - 根据结果选择最适合的检测方法

4. 参数调优：
   - min_face_size: 30-80 (像素)
   - confidence_threshold: 0.3-0.7
   - detection_method: "opencv" (最稳定)

5. 如果仍然检测不到：
   - 检查图片是否损坏
   - 尝试不同的检测方法
   - 人工检查图片中人脸是否清晰可见
"""