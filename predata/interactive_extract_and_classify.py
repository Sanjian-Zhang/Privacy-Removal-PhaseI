#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import gc
import re
import time
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple


import importlib.util
import numpy as np
import cv2
# ========= 分类器动态加载与调用 =========
def load_classifier_module(module_path: Path):
    """把 face_plate_classifier.py 当模块加载（不需要改你原文件）"""
    if not module_path.exists():
        print(f"❌ 分类器模块不存在: {module_path}"); sys.exit(1)
    spec = importlib.util.spec_from_file_location("face_plate_classifier", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_classifier(mod, frames_dir: Path, plate_model_path: Path):
    """
    关键点：在实例化 FacePlateClassifier 之前，直接重写它模块级全局常量，
    解决你原脚本里 INPUT_DIR/OUTPUT_BASE_DIR/模型路径写死的问题。
    """
    # 1) 改写输入/输出与模型路径
    mod.INPUT_DIR = str(frames_dir)
    mod.OUTPUT_BASE_DIR = str(frames_dir)
    mod.PLATE_MODEL_PATH = str(plate_model_path)

    # 2) 同步目录常量（它们在原模块里是基于 OUTPUT_BASE_DIR 定义的）
    mod.QUALIFIED_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "qualified")
    mod.INSUFFICIENT_SCORE_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "insufficient_score")
    mod.NO_CONTENT_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "no_content")
    mod.ANALYSIS_DIR = os.path.join(mod.OUTPUT_BASE_DIR, "analysis")
    for d in [mod.QUALIFIED_DIR, mod.INSUFFICIENT_SCORE_DIR, mod.NO_CONTENT_DIR, mod.ANALYSIS_DIR]:
        os.makedirs(d, exist_ok=True)

    # 3) 运行
    clf = mod.FacePlateClassifier()
    clf.run()

# ========= 通用工具 =========
def which_or_die(bin_name: str):
    path = shutil.which(bin_name)
    if not path:
        print(f"❌ 未找到依赖：{bin_name}（请先安装）")
        sys.exit(1)
    return path

def run(cmd, timeout=None, check=False, capture=False):
    """统一执行命令；对 ffmpeg/ffprobe 降噪但保留 stats。"""
    if isinstance(cmd, (list, tuple)) and cmd:
        head = str(cmd[0])
        if head.endswith("ffmpeg") or head.endswith("ffprobe") or head in ("ffmpeg", "ffprobe"):
            if "-hide_banner" not in cmd:
                cmd.insert(1, "-hide_banner")
            if "-loglevel" not in cmd:
                cmd.insert(1, "-loglevel"); cmd.insert(2, "error")
            if (head.endswith("ffmpeg") or head == "ffmpeg") and "-stats" not in cmd:
                cmd.insert(3, "-stats")
    print("🚀", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, timeout=timeout, check=check, text=True, capture_output=capture)

def atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno == 18:  # cross-device
            shutil.copy2(src, dst); src.unlink()
        else:
            raise

# ========= ffmpeg 兼容兜底（H.264/MP4） =========
def ffprobe_json(path: Path):
    which_or_die("ffprobe")
    res = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=format_name:stream=codec_name,avg_frame_rate,nb_frames",
        "-of", "json", str(path)
    ], capture=True)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout or "{}")
    except Exception:
        return None

def _transcode_to_h264(path: Path):
    which_or_die("ffmpeg")
    out = path.with_suffix(".h264.mp4")
    res = run([
        "ffmpeg", "-i", str(path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-y", str(out)
    ])
    if res.returncode != 0 or (not out.exists()) or out.stat().st_size == 0:
        print("❌ 转码失败"); sys.exit(1)
    atomic_move(out, path)
    print("🧩 已转码为 H.264 MP4")

def ensure_ffmpeg_compatible(path: Path):
    info = ffprobe_json(path)
    if not info or "streams" not in info or not info["streams"]:
        print("⚠️ ffprobe 失败，直接转码到 H.264…")
        _transcode_to_h264(path); return
    fmt = (info.get("format", {}) or {}).get("format_name", "") or ""
    vcodec = (info["streams"][0] or {}).get("codec_name", "")
    is_mp4 = "mp4" in fmt
    is_h264 = (vcodec == "h264")
    if is_mp4 and is_h264:
        print("✅ 已兼容：MP4 + H.264"); return
    if (not is_mp4) and is_h264:
        fixed = path.with_suffix(".repack.mp4")
        res = run(["ffmpeg", "-i", str(path), "-c", "copy", "-movflags", "+faststart", "-y", str(fixed)])
        if res.returncode == 0 and fixed.exists() and fixed.stat().st_size > 0:
            atomic_move(fixed, path)
            print("🔁 已重封装为 MP4（无重压缩）")
            return ensure_ffmpeg_compatible(path)
    _transcode_to_h264(path)

# ========= 下载（两阶段：优先 MP4+M4A，其次任意音频→AAC） =========
def download_video(url: str, out_dir: Path, video_number: str) -> Path:
    which_or_die("yt-dlp"); which_or_die("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_"))
    stem = f"downloaded_video{video_number}"
    out_tmpl = str(tmp_dir / (stem + ".%(ext)s"))
    out_final = out_dir / f"{stem}.mp4"

    attempts = [
        ("bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]", None, "优先 MP4+M4A"),
        ("bv*+ba/b", "ffmpeg:-c:v copy -c:a aac -b:a 192k -movflags +faststart", "兜底：任意音频→AAC")
    ]
    for i, (fmt, pp, desc) in enumerate(attempts, 1):
        print(f"🎯 尝试 {i}/{len(attempts)}：{desc}")
        cmd = [
            "yt-dlp", "-N", "8",
            "--no-part", "--continue", "--restrict-filenames",
            "--merge-output-format", "mp4",
            "-o", out_tmpl, "-f", fmt, url
        ]
        if pp:
            cmd += ["--postprocessor-args", pp]
        res = run(cmd)
        if res.returncode == 0:
            cand = None
            for f in tmp_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".mp4", ".mkv", ".mov"):
                    if ".f" in f.name or f.name.endswith(".ytdl"):  # 过滤分轨临时文件
                        continue
                    cand = f
            if cand:
                atomic_move(cand, out_final)
                print(f"✅ 下载完成：{out_final}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                ensure_ffmpeg_compatible(out_final)
                return out_final
        print("⚠️ 本次方案失败，切换下一种…")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("❌ 视频下载失败"); sys.exit(1)

# ========= 抽帧 =========
def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float, jpg_q: int, video_number: str) -> bool:
    which_or_die("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"video{video_number}_frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vsync", "vfr",
        "-vf", f"fps={fps}",
        "-q:v", str(jpg_q),
        "-vcodec", "mjpeg",
        "-y", str(pattern)
    ]
    res = run(cmd)
    if res.returncode == 0:
        print(f"✅ 抽帧完成：{output_dir}")
        return True
    else:
        print("❌ 抽帧失败")
        return False

# ========= 分类依赖检查 =========
def check_py_deps() -> bool:
    missing = []
    try:
        import torch  # noqa
    except Exception:
        missing.append("torch")
    try:
        from retinaface import RetinaFace  # noqa
    except Exception:
        missing.append("retina-face")
    try:
        from ultralytics import YOLO  # noqa
    except Exception:
        missing.append("ultralytics")
    try:
        import easyocr  # noqa
    except Exception:
        missing.append("easyocr")

    if missing:
        print("❌ 缺少依赖库：", ", ".join(missing))
        print("请安装：")
        for dep in missing:
            print(f"  pip install {dep}")
        return False
    return True

# ========= 正脸/车牌/OCR 评分分类器 =========
class FacePlateClassifier:
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self,
                 input_dir: str,
                 output_base: str,
                 plate_model_path: str,
                 score_threshold: int = 5,
                 clear_face_score: int = 2,
                 clear_plate_score: int = 2,
                 text_score: int = 2,
                 text_fields_threshold: int = 10,
                 yaw_thresh: float = 35.0,
                 min_face_conf: float = 0.8,
                 min_plate_conf: float = 0.5,
                 min_face_size: int = 60,
                 min_plate_size: int = 50,
                 min_face_clarity: float = 30.0,
                 face_area_px2: int = 3600,
                 face_dist_ratio: float = 0.3,
                 min_text_conf: float = 0.5,
                 min_text_len: int = 3,
                 use_gpu: bool = True,
                 gpu_id: int = 0,
                 enable_torch_opt: bool = True,
                 gc_frequency: int = 100,
                 progress_update_frequency: int = 50):
        self.cfg = locals().copy()  # 保存配置
        self.cfg.pop('self')

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("Classifier")
        self.start_time = time.time()
        self.stats = {'qualified': 0, 'insufficient_score': 0, 'no_content': 0, 'failed': 0}
        self.analysis_results: List[Dict] = []

        self.device = self._setup_gpu()
        self._create_output_dirs()
        self._initialize_models()
        self._initialize_ocr()

    # ----- init helpers -----
    def _setup_gpu(self) -> str:
        import torch
        try:
            if self.cfg["use_gpu"] and torch.cuda.is_available():
                cnt = torch.cuda.device_count()
                if self.cfg["gpu_id"] < cnt:
                    dev = f'cuda:{self.cfg["gpu_id"]}'
                    name = torch.cuda.get_device_name(self.cfg["gpu_id"])
                    mem = torch.cuda.get_device_properties(self.cfg["gpu_id"]).total_memory / 1024**3
                    self.logger.info(f"🚀 GPU: {name} (id {self.cfg['gpu_id']})，显存 {mem:.1f} GB")
                    torch.cuda.empty_cache()
                    if self.cfg["enable_torch_opt"]:
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                        self.logger.info("⚡ 已启用 cuDNN benchmark")
                    return dev
                else:
                    self.logger.warning(f"⚠️ 指定 GPU_ID {self.cfg['gpu_id']} 超范围（共 {cnt} 块）")
                    return 'cpu'
            else:
                if not torch.cuda.is_available():
                    self.logger.info("💻 未检测到 CUDA，使用 CPU")
                else:
                    self.logger.info("💻 已禁用 GPU，使用 CPU")
                return 'cpu'
        except Exception as e:
            self.logger.error(f"❌ GPU 设置失败: {e}")
            return 'cpu'

    def _create_output_dirs(self):
        base = Path(self.cfg["output_base"])
        self.qual_dir = base / "qualified"
        self.insuf_dir = base / "insufficient_score"
        self.empty_dir = base / "no_content"
        self.analysis_dir = base / "analysis"
        for d in [self.qual_dir, self.insuf_dir, self.empty_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 创建目录: {d}")

    def _initialize_models(self):
        from retinaface import RetinaFace
        from ultralytics import YOLO

        # RetinaFace 简测
        self.logger.info("🔍 初始化 RetinaFace…")
        _ = RetinaFace.detect_faces(np.ones((100, 100, 3), dtype=np.uint8) * 128)
        self.retina = RetinaFace
        self.logger.info("✅ RetinaFace 就绪")

        # YOLO 车牌
        plate_path = self.cfg["plate_model_path"]
        if not Path(plate_path).exists():
            raise FileNotFoundError(f"车牌模型不存在: {plate_path}")
        self.logger.info("🚗 初始化 YOLO 车牌模型…")
        self.plate_model = YOLO(plate_path)
        self.logger.info("✅ YOLO 车牌模型就绪")

    def _initialize_ocr(self):
        import easyocr
        gpu_flag = 'cuda' in self.device
        self.logger.info("📝 初始化 EasyOCR…")
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=gpu_flag)
        self.logger.info(f"✅ EasyOCR 就绪（{'GPU' if gpu_flag else 'CPU'}）")

    # ----- metrics helpers -----
    def _calc_face_clarity(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return 0.0
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mean = np.mean(np.sqrt(gx**2 + gy**2))
            return float(lap_var + 0.1 * grad_mean)
        except Exception:
            return 0.0

    def _calc_yaw(self, landmarks: Dict) -> float:
        try:
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            eye_center = (left_eye + right_eye) / 2
            eye_w = np.linalg.norm(right_eye - left_eye)
            if eye_w < 10:
                return 90.0
            yaw = abs((nose[0] - eye_center[0]) / eye_w) * 60.0
            return float(yaw)
        except Exception:
            return 90.0

    # ----- detectors -----
    def _faces(self, image_path: str) -> Tuple[int, List[Dict]]:
        dets = self.retina.detect_faces(image_path)
        if not isinstance(dets, dict) or len(dets) == 0:
            return 0, []
        img = cv2.imread(image_path)
        if img is None:
            return 0, []
        H, W = img.shape[:2]
        img_area = W * H

        out = []
        for _, fd in dets.items():
            conf = float(fd.get('score', 0.0))
            if conf < self.cfg["min_face_conf"]:
                continue
            if 'facial_area' not in fd or 'landmarks' not in fd:
                continue
            x1, y1, x2, y2 = fd['facial_area']
            fw, fh = x2 - x1, y2 - y1
            if min(fw, fh) < self.cfg["min_face_size"]:
                continue

            clarity = self._calc_face_clarity(img, (x1, y1, x2, y2))
            area = fw * fh
            dist_ratio = area / img_area
            is_clear = clarity >= self.cfg["min_face_clarity"]
            is_close = dist_ratio >= self.cfg["face_dist_ratio"]
            is_big = area >= self.cfg["face_area_px2"]
            if not (is_clear and (is_close or is_big)):
                continue

            yaw = self._calc_yaw(fd['landmarks'])
            is_frontal = yaw <= self.cfg["yaw_thresh"]
            if is_frontal:
                out.append({
                    "confidence": conf,
                    "yaw_angle": yaw,
                    "is_frontal": True,
                    "facial_area": [int(x1), int(y1), int(x2), int(y2)],
                    "face_size": [int(fw), int(fh)],
                    "quality_info": {
                        "clarity_score": clarity,
                        "distance_score": dist_ratio,
                        "face_area": int(area),
                        "is_clear": is_clear, "is_close": is_close, "is_large_enough": is_big
                    }
                })
        return len(out), out

    def _plates(self, image_path: str) -> Tuple[int, List[Dict]]:
        rs = self.plate_model(image_path, verbose=False)
        if not rs or len(rs) == 0:
            return 0, []
        r0 = rs[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return 0, []
        out = []
        for box in r0.boxes:
            conf = float(box.conf[0])
            if conf < self.cfg["min_plate_conf"]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            if min(w, h) < self.cfg["min_plate_size"]:
                continue
            out.append({"confidence": conf,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "plate_size": [float(w), float(h)]})
        return len(out), out

    def _texts(self, image_path: str) -> Tuple[int, List[Dict]]:
        img = cv2.imread(image_path)
        if img is None:
            return 0, []
        rs = self.ocr_reader.readtext(img)
        if not rs:
            return 0, []
        out = []
        for bbox, text, conf in rs:
            conf = float(conf) if conf is not None else 0.0
            if conf < self.cfg["min_text_conf"]:
                continue
            t = (text or "").strip()
            if len(t) < self.cfg["min_text_len"]:
                continue
            if t.replace(' ', '').replace('.', '').replace('-', '').replace('_', ''):
                out.append({"text": t, "confidence": conf, "bbox": bbox})
        return len(out), out

    def classify_image(self, image_path: str) -> Tuple[str, Dict]:
        try:
            fname = os.path.basename(image_path)
            n_face, face_det = self._faces(image_path)
            n_plate, plate_det = self._plates(image_path)
            n_text, text_det = self._texts(image_path)

            score = 0
            details = []
            if n_face > 0:
                s = n_face * self.cfg["clear_face_score"]; score += s
                details.append(f"清晰正脸 {n_face} 张 × {self.cfg['clear_face_score']} = {s} 分")
            if n_plate > 0:
                s = n_plate * self.cfg["clear_plate_score"]; score += s
                details.append(f"清晰车牌 {n_plate} 张 × {self.cfg['clear_plate_score']} = {s} 分")
            if n_text >= self.cfg["text_fields_threshold"]:
                s = self.cfg["text_score"]; score += s
                details.append(f"可识别文字 {n_text} 个字段(>={self.cfg['text_fields_threshold']}) = {s} 分")
            elif n_text > 0:
                details.append(f"可识别文字 {n_text} 个字段(<{self.cfg['text_fields_threshold']}) = 0 分")

            meets = score > self.cfg["score_threshold"]
            if meets:
                category = "qualified"; reason = f'总分 {score} 分 > {self.cfg["score_threshold"]} 分，符合要求'
            else:
                if score == 0:
                    category = "no_content"; reason = f'总分 {score} 分，无任何有效内容'
                else:
                    category = "insufficient_score"; reason = f'总分 {score} 分 ≤ {self.cfg["score_threshold"]} 分，不符合要求'

            analysis = {
                "filename": fname,
                "frontal_faces": n_face,
                "license_plates": n_plate,
                "text_count": n_text,
                "total_score": score,
                "score_details": details,
                "meets_requirements": meets,
                "score_threshold": self.cfg["score_threshold"],
                "face_details": face_det,
                "plate_details": plate_det,
                "text_details": text_det,
                "category": category,
                "reason": reason,
                "timestamp": time.time()
            }
            return category, analysis
        except Exception as e:
            return "failed", {"filename": os.path.basename(image_path), "error": str(e)}

    def _move_to(self, image_path: str, category: str) -> bool:
        target_dir = {"qualified": self.qual_dir, "insufficient_score": self.insuf_dir, "no_content": self.empty_dir}.get(category)
        if target_dir is None:
            return False
        name = os.path.basename(image_path)
        dst = target_dir / name
        c = 1
        while dst.exists():
            stem, ext = os.path.splitext(name)
            dst = target_dir / f"{stem}_{c}{ext}"
            c += 1
        try:
            shutil.move(image_path, dst)
            return True
        except Exception as e:
            self.logger.error(f"❌ 移动失败 {image_path} -> {dst}: {e}")
            return False

    def _image_files(self) -> List[str]:
        p = Path(self.cfg["input_dir"])
        if not p.exists():
            self.logger.error(f"❌ 输入目录不存在: {p}")
            return []
        files = []
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS:
                files.append(str(f))
        self.logger.info(f"📊 找到 {len(files)} 个图像文件")
        return files

    def _save_results(self):
        analysis_file = self.analysis_dir / "classification_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)

        summary_file = self.analysis_dir / "classification_summary.json"
        total = len(self.analysis_results)
        summary = {
            "total_processed": total,
            "statistics": self.stats.copy(),
            "processing_time": time.time() - self.start_time,
            "scoring_system": {
                "clear_face_score": self.cfg["clear_face_score"],
                "clear_plate_score": self.cfg["clear_plate_score"],
                "text_recognition_score": self.cfg["text_score"],
                "text_fields_threshold": self.cfg["text_fields_threshold"],
                "score_threshold": self.cfg["score_threshold"]
            },
            "configuration": {
                "yaw_angle_threshold": self.cfg["yaw_thresh"],
                "min_face_confidence": self.cfg["min_face_conf"],
                "min_plate_confidence": self.cfg["min_plate_conf"],
                "min_text_confidence": self.cfg["min_text_conf"]
            },
            "timestamp": time.time()
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 分析结果: {analysis_file}")
        self.logger.info(f"📊 统计摘要: {summary_file}")

    def _print_final(self):
        dt = time.time() - self.start_time
        total = sum(self.stats.values())
        self.logger.info("="*80)
        self.logger.info("🎉 分类完成（新计分系统）")
        self.logger.info(f"✅ 合格: {self.stats['qualified']} | ❌ 不足: {self.stats['insufficient_score']} | ⭕ 无内容: {self.stats['no_content']} | 💥 失败: {self.stats['failed']}")
        self.logger.info(f"📊 总处理: {total}  | ⏰ 用时: {dt:.1f}s")
        if total:
            self.logger.info(f"🚀 速度: {total/dt:.1f} 张/秒")
        for name, d in [("符合条件", self.qual_dir), ("分数不够", self.insuf_dir), ("无任何内容", self.empty_dir)]:
            if d.exists():
                c = len([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS])
                self.logger.info(f"  📁 {name}: {c}")
        self.logger.info("="*80)

    def run(self):
        from tqdm import tqdm
        import torch

        self.logger.info("🚀 启动分类器…")
        self.logger.info(f"📁 输入: {self.cfg['input_dir']} / 输出基准: {self.cfg['output_base']} / 设备: {self.device}")

        files = self._image_files()
        if not files:
            self.logger.warning("❌ 未找到图像"); return

        with tqdm(total=len(files), desc="分类进度", ncols=120,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for i, imgp in enumerate(files):
                cat, ana = self.classify_image(imgp)
                if cat != "failed":
                    if self._move_to(imgp, cat):
                        self.stats[cat] += 1
                    else:
                        self.stats["failed"] += 1
                        ana["move_failed"] = True
                else:
                    self.stats["failed"] += 1
                self.analysis_results.append(ana)
                pbar.update(1)

                if i % self.cfg["progress_update_frequency"] == 0 and i > 0:
                    s = f"✅{self.stats['qualified']} ❌{self.stats['insufficient_score']} ⭕{self.stats['no_content']}"
                    pbar.set_description(f"分类进度 ({s})")
                if i % self.cfg["gc_frequency"] == 0 and i > 0 and 'cuda' in self.device:
                    torch.cuda.empty_cache(); gc.collect()

        self._save_results()
        self._print_final()

# ========= 交互式入口 =========
def main():
    print("===== 🎬 交互式下载→抽帧→分类 工具 =====")
    url = input("请输入视频链接或本地文件路径: ").strip()
    if not url:
        print("❌ 必须输入链接或本地路径"); sys.exit(1)

    video_number = input("请输入视频编号（用于命名，如 38）: ").strip()
    if not video_number or not re.fullmatch(r"\d+", video_number):
        print("❌ 视频编号必须是数字"); sys.exit(1)

    fps_in = input("每秒抽帧数（默认=3，直接回车使用默认）: ").strip()
    fps = float(fps_in) if fps_in else 3.0

    jpg_in = input("JPG质量（默认=1，越小越清晰，2-31；直接回车=1）: ").strip()
    jpg_q = int(jpg_in) if jpg_in else 1

    # 用你的路径布局
    base_dir = Path("/home/zhiqics/sanjian/predata")
    video_dir = base_dir / "videos"
    frames_dir = base_dir / f"output_frames{video_number}"

    # 1) 获取视频
    downloaded = False
    if url.startswith("http"):
        video_path = download_video(url, video_dir, video_number)
        downloaded = True
    else:
        video_path = Path(url)
        if not video_path.exists():
            print(f"❌ 本地视频不存在：{video_path}"); sys.exit(1)
        ensure_ffmpeg_compatible(video_path)

    # 2) 抽帧
    ok = extract_frames_ffmpeg(
        video_path=video_path,
        output_dir=frames_dir,
        fps=fps,
        jpg_q=jpg_q,
        video_number=video_number
    )
    if not ok:
        sys.exit(1)

    # 3) 成功后删除“本次下载”的视频文件
    if downloaded:
        try:
            video_path.unlink(missing_ok=True)
            print(f"🧹 已删除下载视频：{video_path}")
            try:
                if video_dir.exists() and not any(video_dir.iterdir()):
                    video_dir.rmdir()
                    print(f"🧼 已清理空目录：{video_dir}")
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ 删除下载视频失败：{e}")

    # 4) 继续执行：正脸/车牌/OCR 分类
    print("\n===== 🧠 开始图像分类（正脸/车牌/OCR） =====")
    default_model = str(base_dir / "models/license_plate_detector.pt")
    model_path_in = input(f"请输入车牌检测模型路径（回车默认: {default_model}）: ").strip()
    plate_model = model_path_in if model_path_in else default_model

    use_gpu_in = input("是否启用GPU? (Y/n，默认Y): ").strip().lower()
    use_gpu = False if use_gpu_in == "n" else True
    gpu_id_in = input("GPU设备ID（默认=0）: ").strip()
    gpu_id = int(gpu_id_in) if gpu_id_in else 0

    # 依赖检查
    if not check_py_deps():
        sys.exit(1)

    # 优先尝试动态加载 face_plate_classifier.py
    classifier_py = Path(__file__).parent / "face_plate_classifier.py"
    try:
        if classifier_py.exists():
            print(f"🧩 动态加载分类器模块: {classifier_py}")
            mod = load_classifier_module(classifier_py)
            run_classifier(mod, frames_dir, Path(plate_model))
        else:
            print("⚠️ 未找到 face_plate_classifier.py，使用内置 FacePlateClassifier 类")
            clf = FacePlateClassifier(
                input_dir=str(frames_dir),
                output_base=str(frames_dir),
                plate_model_path=plate_model,
                score_threshold=5,
                clear_face_score=2,
                clear_plate_score=2,
                text_score=2,
                text_fields_threshold=10,
                yaw_thresh=35.0,
                min_face_conf=0.8,
                min_plate_conf=0.5,
                min_face_size=60,
                min_plate_size=50,
                min_face_clarity=30.0,
                face_area_px2=3600,
                face_dist_ratio=0.3,
                min_text_conf=0.5,
                min_text_len=3,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
                enable_torch_opt=True,
                gc_frequency=100,
                progress_update_frequency=50
            )
            clf.run()
    except KeyboardInterrupt:
        print("⚡ 用户中断")
    except Exception as e:
        print(f"❌ 分类流程异常：{e}")
        import traceback; print(traceback.format_exc())

if __name__ == "__main__":
    main()
