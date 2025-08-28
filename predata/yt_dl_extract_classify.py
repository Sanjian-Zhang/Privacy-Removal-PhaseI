#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil
import argparse
import subprocess
import tempfile
import re
import time
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

# ================= 基础工具（下载 & ffmpeg 兼容） =================

def which_or_die(bin_name: str):
    path = shutil.which(bin_name)
    if not path:
        print(f"❌ 未找到依赖：{bin_name}（请先安装）")
        sys.exit(1)
    return path

def run(cmd, timeout=None, check=False, capture=False):
    """
    - 对 ffmpeg/ffprobe 降噪但保留 stats
    - 对其他（如 yt-dlp）保留默认进度输出
    """
    if isinstance(cmd, (tuple, list)) and cmd:
        head = str(cmd[0])
        if head.endswith("ffmpeg") or head.endswith("ffprobe") or head == "ffmpeg" or head == "ffprobe":
            if "-hide_banner" not in cmd:
                cmd.insert(1, "-hide_banner")
            if "-loglevel" not in cmd:
                cmd.insert(1, "-loglevel")
                cmd.insert(2, "error")
            if (head.endswith("ffmpeg") or head == "ffmpeg") and "-stats" not in cmd:
                cmd.insert(3, "-stats")
    print("🚀", " ".join(str(x) for x in cmd))
    return subprocess.run(
        cmd, timeout=timeout, check=check, text=True, capture_output=capture
    )

def atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)  # 原子移动
    except OSError as e:
        if e.errno == 18:  # Invalid cross-device link
            shutil.copy2(src, dst)
            src.unlink()
        else:
            raise

def ffprobe_json(path: Path):
    which_or_die("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=format_name:stream=codec_name,codec_tag_string,avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json", str(path)
    ]
    res = run(cmd, capture=True)
    if res.returncode != 0:
        return None
    try:
        return json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        return None

def _transcode_to_h264(path: Path) -> dict:
    which_or_die("ffmpeg")
    out = path.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", str(out)
    ]
    res = run(cmd)
    if res.returncode != 0 or (not out.exists()) or out.stat().st_size == 0:
        print("❌ 转码失败")
        sys.exit(1)
    atomic_move(out, path)
    print("🧩 已转码为 H.264 MP4")
    info = ffprobe_json(path) or {}
    v = (info.get("streams") or [{}])[0]
    return {
        "ok": True,
        "fps": (v or {}).get("avg_frame_rate", "0/0"),
        "nb_frames": (v or {}).get("nb_frames", "0")
    }

def ensure_ffmpeg_compatible(path: Path) -> dict:
    """
    确保容器=mp4 且 编码=h264；
    - 容器非 mp4 且已是 h264：仅重封装
    - 编码非 h264：转码为 h264
    """
    info = ffprobe_json(path)
    if not info or "streams" not in info or not info["streams"]:
        print("⚠️ ffprobe 读取失败，直接转码为 H.264 MP4…")
        return _transcode_to_h264(path)

    fmt = (info.get("format", {}) or {}).get("format_name", "")
    v = info["streams"][0]
    vcodec = v.get("codec_name", "")
    nb_frames = v.get("nb_frames", "0")
    avg_fps = v.get("avg_frame_rate", "0/0")

    is_mp4 = "mp4" in (fmt or "")
    is_h264 = (vcodec == "h264")

    if is_mp4 and is_h264:
        print(f"✅ 已兼容：mp4 + h264，帧率={avg_fps}，总帧≈{nb_frames}")
        return {"ok": True, "fps": avg_fps, "nb_frames": nb_frames}

    if (not is_mp4) and is_h264:
        which_or_die("ffmpeg")
        fixed = path.with_suffix(".repack.mp4")
        cmd = ["ffmpeg", "-i", str(path), "-c", "copy", "-movflags", "+faststart", "-y", str(fixed)]
        res = run(cmd)
        if res.returncode == 0 and fixed.exists() and fixed.stat().st_size > 0:
            atomic_move(fixed, path)
            print("🔁 已重封装为 mp4（无重压缩）")
            return ensure_ffmpeg_compatible(path)

    return _transcode_to_h264(path)

PREFER_H264_FMT = ["-S", "codec:avc1,res,fps,ext:mp4"]

def download_youtube(url: str, out_path: Path, cookies: Optional[Path],
                     with_audio: bool, fmt_height: int, timeout: int, retries: int):
    which_or_die("yt-dlp"); which_or_die("ffmpeg"); which_or_die("ffprobe")
    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_"))
    tmp_file = tmp_dir / (out_path.name + ".part.mp4")

    base_cmd = [
        "yt-dlp", "-N", "8",
        "--no-part", "--continue", "--no-warnings", "--restrict-filenames",
        "--no-write-subs", "--no-write-auto-subs", "--no-embed-subs",
        "--no-write-info-json", "--no-write-description", "--no-write-thumbnail",
        "-o", str(tmp_file),
        *PREFER_H264_FMT,
        url,
    ]
    if cookies:
        base_cmd += ["--cookies", str(cookies)]

    if with_audio:
        base_cmd += ["--merge-output-format", "mp4",
                     "-f", f"bv*[height<={fmt_height}][fps<=60]+ba/b[height<={fmt_height}]"]
    else:
        base_cmd += ["--no-audio", "-f", f"bv*[height<={fmt_height}][fps<=60][ext=mp4]"]

    last_err = None
    for i in range(1, retries+1):
        print(f"🎯 正在下载（第 {i}/{retries} 次重试）：{url}")
        if i > 1 and tmp_file.exists():
            tmp_file.unlink()
        try:
            res = run(base_cmd, timeout=timeout)
            if res.returncode == 0 and tmp_file.exists() and tmp_file.stat().st_size > 0:
                atomic_move(tmp_file, out_path)
                print(f"✅ 下载完成：{out_path}（{out_path.stat().st_size/(1024*1024):.1f} MB）")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                ensure_ffmpeg_compatible(out_path)
                return
            else:
                last_err = f"返回码 {res.returncode}"
        except subprocess.TimeoutExpired:
            last_err = "下载超时"
        except Exception as e:
            last_err = f"异常：{e}"
        print(f"⚠️ 本次失败：{last_err}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("❌ 下载失败，已用尽重试次数"); sys.exit(1)

def extract_frames_ffmpeg(video_path: Path, output_dir: Path, fps: float,
                          start: Optional[float], duration: Optional[float],
                          scale_width: Optional[int], img_format: str,
                          png_level: int, jpg_q: int, threads: int):
    which_or_die("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 文件名前缀：video{编号}_frame_000001.jpg
    video_name = video_path.stem
    m = re.search(r'(\d+)', video_name)
    video_prefix = f"video{m.group(1)}" if m else video_name
    pattern = output_dir / f"{video_prefix}_frame_%06d.{img_format.lower()}"

    vf_chain = [f"fps={fps}"]
    ss_args, t_args = [], []
    if start is not None:
        ss_args = ["-ss", str(start)]
    if duration is not None:
        t_args = ["-t", str(duration)]
    if scale_width:
        vf_chain.append(f"scale={scale_width}:-2:flags=bicubic")

    cmd = ["ffmpeg"] + ss_args + [
        "-i", str(video_path), "-vsync", "vfr",
        "-threads", str(threads if threads is not None else 0),
        "-map_metadata", "-1", "-an"
    ] + t_args + ["-vf", ",".join(vf_chain), "-y"]

    fmt = img_format.lower()
    if fmt == "png":
        cmd += ["-compression_level", str(png_level)]
    elif fmt in ("jpg", "jpeg"):
        cmd += ["-q:v", str(jpg_q), "-vcodec", "mjpeg"]
    else:
        print("❌ 仅支持 png / jpg"); sys.exit(1)

    cmd += [str(pattern)]
    res = run(cmd)
    if res.returncode == 0:
        print(f"✅ 抽帧完成：{output_dir}")
    else:
        print("❌ 抽帧失败"); sys.exit(1)

# ================= 依赖检查（分类器用） =================

def check_py_deps():
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
        print("建议安装：")
        for dep in missing:
            print(f"  pip install {dep}")
        return False
    return True

# ================= 正脸/车牌/OCR 评分分类器 =================

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

        self.cfg = {
            "INPUT_DIR": input_dir,
            "OUTPUT_BASE_DIR": output_base,
            "PLATE_MODEL_PATH": plate_model_path,
            "SCORE_THRESHOLD": score_threshold,
            "CLEAR_FACE_SCORE": clear_face_score,
            "CLEAR_PLATE_SCORE": clear_plate_score,
            "TEXT_RECOGNITION_SCORE": text_score,
            "TEXT_FIELDS_THRESHOLD": text_fields_threshold,
            "YAW_ANGLE_THRESHOLD": yaw_thresh,
            "MIN_FACE_CONFIDENCE": min_face_conf,
            "MIN_PLATE_CONFIDENCE": min_plate_conf,
            "MIN_FACE_SIZE": min_face_size,
            "MIN_PLATE_SIZE": min_plate_size,
            "MIN_FACE_CLARITY_SCORE": min_face_clarity,
            "FACE_AREA_THRESHOLD": face_area_px2,
            "MAX_FACE_DISTANCE_RATIO": face_dist_ratio,
            "MIN_TEXT_CONFIDENCE": min_text_conf,
            "MIN_TEXT_LENGTH": min_text_len,
            "USE_GPU": use_gpu,
            "GPU_DEVICE_ID": gpu_id,
            "ENABLE_TORCH_OPTIMIZATION": enable_torch_opt,
            "GC_FREQUENCY": gc_frequency,
            "PROGRESS_UPDATE_FREQUENCY": progress_update_frequency,
        }

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("Classifier")

        self.start_time = time.time()
        self.stats = {'qualified': 0, 'insufficient_score': 0, 'no_content': 0, 'failed': 0}
        self.analysis_results: List[Dict] = []

        # 设备 & 目录 & 模型
        self.device = self._setup_gpu()
        self._create_output_dirs()
        self._initialize_models()
        self._initialize_ocr()

        self.logger.info("🚀 正脸/车牌/OCR 分类器初始化完成（新计分系统）")

    # ----- init helpers -----

    def _setup_gpu(self) -> str:
        import torch
        try:
            if self.cfg["USE_GPU"] and torch.cuda.is_available():
                cnt = torch.cuda.device_count()
                if self.cfg["GPU_DEVICE_ID"] < cnt:
                    dev = f'cuda:{self.cfg["GPU_DEVICE_ID"]}'
                    name = torch.cuda.get_device_name(self.cfg["GPU_DEVICE_ID"])
                    mem = torch.cuda.get_device_properties(self.cfg["GPU_DEVICE_ID"]).total_memory / 1024**3
                    self.logger.info(f"🚀 GPU: {name} (id {self.cfg['GPU_DEVICE_ID']})，显存 {mem:.1f} GB")
                    torch.cuda.empty_cache()
                    if self.cfg["ENABLE_TORCH_OPTIMIZATION"]:
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                        self.logger.info("⚡ 已启用 cuDNN benchmark")
                    return dev
                else:
                    self.logger.warning(f"⚠️ 指定的 GPU_ID {self.cfg['GPU_DEVICE_ID']} 超范围（共 {cnt} 块）")
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
        base = Path(self.cfg["OUTPUT_BASE_DIR"])
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

        # 先 sanity check RetinaFace
        self.logger.info("🔍 初始化 RetinaFace…")
        _ = RetinaFace.detect_faces(np.ones((100, 100, 3), dtype=np.uint8) * 128)
        self.retina = RetinaFace
        self.logger.info("✅ RetinaFace 就绪")

        # 车牌 YOLO
        plate_path = self.cfg["PLATE_MODEL_PATH"]
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
            if conf < self.cfg["MIN_FACE_CONFIDENCE"]:
                continue
            if 'facial_area' not in fd or 'landmarks' not in fd:
                continue
            x1, y1, x2, y2 = fd['facial_area']
            fw, fh = x2 - x1, y2 - y1
            if min(fw, fh) < self.cfg["MIN_FACE_SIZE"]:
                continue

            clarity = self._calc_face_clarity(img, (x1, y1, x2, y2))
            area = fw * fh
            dist_ratio = area / img_area
            is_clear = clarity >= self.cfg["MIN_FACE_CLARITY_SCORE"]
            is_close = dist_ratio >= self.cfg["MAX_FACE_DISTANCE_RATIO"]
            is_big = area >= self.cfg["FACE_AREA_THRESHOLD"]
            if not (is_clear and (is_close or is_big)):
                continue

            yaw = self._calc_yaw(fd['landmarks'])
            is_frontal = yaw <= self.cfg["YAW_ANGLE_THRESHOLD"]
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
        # ultralytics YOLO 自动用 GPU（若可用）
        rs = self.plate_model(image_path, verbose=False)
        if not rs or len(rs) == 0:
            return 0, []
        r0 = rs[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return 0, []
        out = []
        for box in r0.boxes:
            conf = float(box.conf[0])
            if conf < self.cfg["MIN_PLATE_CONFIDENCE"]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w, h = x2 - x1, y2 - y1
            if min(w, h) < self.cfg["MIN_PLATE_SIZE"]:
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
            if conf < self.cfg["MIN_TEXT_CONFIDENCE"]:
                continue
            t = (text or "").strip()
            if len(t) < self.cfg["MIN_TEXT_LENGTH"]:
                continue
            # 粗略去符号后是否还留字符
            if t.replace(' ', '').replace('.', '').replace('-', '').replace('_', ''):
                out.append({"text": t, "confidence": conf, "bbox": bbox})
        return len(out), out

    # ----- classify one image -----

    def classify_image(self, image_path: str) -> Tuple[str, Dict]:
        try:
            fname = os.path.basename(image_path)
            n_face, face_det = self._faces(image_path)
            n_plate, plate_det = self._plates(image_path)
            n_text, text_det = self._texts(image_path)

            score = 0
            details = []
            if n_face > 0:
                s = n_face * self.cfg["CLEAR_FACE_SCORE"]
                score += s
                details.append(f"清晰正脸 {n_face} 张 × {self.cfg['CLEAR_FACE_SCORE']} = {s} 分")

            if n_plate > 0:
                s = n_plate * self.cfg["CLEAR_PLATE_SCORE"]
                score += s
                details.append(f"清晰车牌 {n_plate} 张 × {self.cfg['CLEAR_PLATE_SCORE']} = {s} 分")

            if n_text >= self.cfg["TEXT_FIELDS_THRESHOLD"]:
                s = self.cfg["TEXT_RECOGNITION_SCORE"]
                score += s
                details.append(f"可识别文字 {n_text} 个字段(>={self.cfg['TEXT_FIELDS_THRESHOLD']}) = {s} 分")
            elif n_text > 0:
                details.append(f"可识别文字 {n_text} 个字段(<{self.cfg['TEXT_FIELDS_THRESHOLD']}) = 0 分")

            meets = score > self.cfg["SCORE_THRESHOLD"]
            if meets:
                category = "qualified"
                reason = f'总分 {score} 分 > {self.cfg["SCORE_THRESHOLD"]} 分，符合要求'
            else:
                if score == 0:
                    category = "no_content"; reason = f'总分 {score} 分，无任何有效内容'
                else:
                    category = "insufficient_score"; reason = f'总分 {score} 分 ≤ {self.cfg["SCORE_THRESHOLD"]} 分，不符合要求'

            analysis = {
                "filename": fname,
                "frontal_faces": n_face,
                "license_plates": n_plate,
                "text_count": n_text,
                "total_score": score,
                "score_details": details,
                "meets_requirements": meets,
                "score_threshold": self.cfg["SCORE_THRESHOLD"],
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

    # ----- IO helpers -----

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
        p = Path(self.cfg["INPUT_DIR"])
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
                "clear_face_score": self.cfg["CLEAR_FACE_SCORE"],
                "clear_plate_score": self.cfg["CLEAR_PLATE_SCORE"],
                "text_recognition_score": self.cfg["TEXT_RECOGNITION_SCORE"],
                "text_fields_threshold": self.cfg["TEXT_FIELDS_THRESHOLD"],
                "score_threshold": self.cfg["SCORE_THRESHOLD"]
            },
            "configuration": {
                "yaw_angle_threshold": self.cfg["YAW_ANGLE_THRESHOLD"],
                "min_face_confidence": self.cfg["MIN_FACE_CONFIDENCE"],
                "min_plate_confidence": self.cfg["MIN_PLATE_CONFIDENCE"],
                "min_text_confidence": self.cfg["MIN_TEXT_CONFIDENCE"]
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

    # ----- main run -----

    def run(self):
        from tqdm import tqdm
        import torch

        self.logger.info("🚀 启动分类器…")
        self.logger.info(f"📁 输入: {self.cfg['INPUT_DIR']} / 输出基准: {self.cfg['OUTPUT_BASE_DIR']} / 设备: {self.device}")

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

                if i % self.cfg["PROGRESS_UPDATE_FREQUENCY"] == 0 and i > 0:
                    s = f"✅{self.stats['qualified']} ❌{self.stats['insufficient_score']} ⭕{self.stats['no_content']}"
                    pbar.set_description(f"分类进度 ({s})")
                if i % self.cfg["GC_FREQUENCY"] == 0 and i > 0 and 'cuda' in self.device:
                    torch.cuda.empty_cache()
                    gc.collect()

        self._save_results()
        self._print_final()

# ================= CLI =================

def main():
    parser = argparse.ArgumentParser(
        description="YouTube 稳定下载 + FFmpeg 抽帧 + 正脸/车牌/OCR 评分分类 一体化工具"
    )
    # 下载
    parser.add_argument("--url", type=str, default="", help="YouTube URL（留空表示不下载）")
    parser.add_argument("--out_video", type=str, default="./downloaded_video.mp4", help="下载输出路径")
    parser.add_argument("--cookies", type=str, default="cookies.txt", help="cookies 文件路径（可选）")
    parser.add_argument("--with_audio", action="store_true", help="下载合并音频（默认不带音频）")
    parser.add_argument("--height", type=int, default=2160, help="最高分辨率高度限制")
    parser.add_argument("--dl_timeout", type=int, default=1200, help="下载超时时间（秒）")
    parser.add_argument("--dl_retries", type=int, default=2, help="下载重试次数")
    parser.add_argument("--ensure_only", action="store_true", help="只做兼容性修复后退出（调试用）")

    # 抽帧
    parser.add_argument("--video_path", type=str, default="", help="已有本地视频路径（优先生效）")
    parser.add_argument("--out_frames", type=str, default="", help="抽帧输出目录；留空跳过抽帧")
    parser.add_argument("--fps", type=float, default=3.0, help="每秒抽取帧数")
    parser.add_argument("--start", type=float, default=None, help="抽帧起始秒")
    parser.add_argument("--duration", type=float, default=None, help="抽帧时长秒")
    parser.add_argument("--scale_width", type=int, default=None, help="缩放宽度（如 3840 表示 4K）")
    parser.add_argument("--format", choices=["png", "jpg"], default="jpg", help="输出格式")
    parser.add_argument("--png_level", type=int, default=4, help="PNG 压缩级别 0-9")
    parser.add_argument("--jpg_q", type=int, default=1, help="JPG 质量（2-31，越小越清晰）")
    parser.add_argument("--threads", type=int, default=0, help="ffmpeg 线程数（0=自动）")

    # 分类器
    parser.add_argument("--classify", action="store_true", help="启用对帧目录的正脸/车牌/OCR 评分分类")
    parser.add_argument("--frames_dir_for_classify", type=str, default="", help="分类输入目录（通常是抽帧输出目录）")
    parser.add_argument("--output_base", type=str, default="", help="分类输出基目录（默认为输入目录）")
    parser.add_argument("--plate_model", type=str, default="", help="YOLO 车牌检测模型权重路径（*.pt）")

    # 评分阈值与开关
    parser.add_argument("--score_threshold", type=int, default=5)
    parser.add_argument("--clear_face_score", type=int, default=2)
    parser.add_argument("--clear_plate_score", type=int, default=2)
    parser.add_argument("--text_score", type=int, default=2)
    parser.add_argument("--text_fields_threshold", type=int, default=10)

    parser.add_argument("--yaw_thresh", type=float, default=35.0)
    parser.add_argument("--min_face_conf", type=float, default=0.8)
    parser.add_argument("--min_plate_conf", type=float, default=0.5)
    parser.add_argument("--min_face_size", type=int, default=60)
    parser.add_argument("--min_plate_size", type=int, default=50)
    parser.add_argument("--min_face_clarity", type=float, default=30.0)
    parser.add_argument("--face_area_px2", type=int, default=3600)
    parser.add_argument("--face_dist_ratio", type=float, default=0.3)
    parser.add_argument("--min_text_conf", type=float, default=0.5)
    parser.add_argument("--min_text_len", type=int, default=3)

    parser.add_argument("--use_gpu", action="store_true", help="分类器启用 GPU")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--no_torch_opt", action="store_true", help="禁用 cuDNN benchmark")

    parser.add_argument("--gc_frequency", type=int, default=100)
    parser.add_argument("--progress_update_frequency", type=int, default=50)

    args = parser.parse_args()

    # 1) 确定输入视频
    video_path: Optional[Path] = None
    if args.video_path:
        vp = Path(args.video_path)
        if not vp.exists():
            print(f"❌ 本地视频不存在：{vp}"); sys.exit(1)
        ensure_ffmpeg_compatible(vp)
        video_path = vp
    elif args.url:
        out_video = Path(args.out_video)
        cookies = Path(args.cookies) if args.cookies and Path(args.cookies).exists() else None
        download_youtube(args.url, out_video, cookies, args.with_audio, args.height, args.dl_timeout, args.dl_retries)
        video_path = out_video
    else:
        print("ℹ️ 未提供 --video_path 或 --url，跳过下载与抽帧。")

    if args.ensure_only:
        print("ℹ️ 已完成兼容性修复，按 --ensure_only 要求退出。"); sys.exit(0)

    # 2) 抽帧
    if video_path and args.out_frames:
        extract_frames_ffmpeg(
            video_path=video_path,
            output_dir=Path(args.out_frames),
            fps=args.fps,
            start=args.start,
            duration=args.duration,
            scale_width=args.scale_width,
            img_format=args.format,
            png_level=args.png_level,
            jpg_q=args.jpg_q,
            threads=args.threads
        )

    # 3) 分类
    if args.classify:
        frames_dir = args.frames_dir_for_classify or args.out_frames
        if not frames_dir:
            print("❌ 启用 --classify 时必须提供 --frames_dir_for_classify 或先用 --out_frames 抽帧"); sys.exit(1)
        output_base = args.output_base or frames_dir
        if not args.plate_model:
            print("❌ 分类器需要提供车牌模型 --plate_model=*.pt"); sys.exit(1)

        # 依赖检查
        if not check_py_deps():
            sys.exit(1)

        classifier = FacePlateClassifier(
            input_dir=frames_dir,
            output_base=output_base,
            plate_model_path=args.plate_model,
            score_threshold=args.score_threshold,
            clear_face_score=args.clear_face_score,
            clear_plate_score=args.clear_plate_score,
            text_score=args.text_score,
            text_fields_threshold=args.text_fields_threshold,
            yaw_thresh=args.yaw_thresh,
            min_face_conf=args.min_face_conf,
            min_plate_conf=args.min_plate_conf,
            min_face_size=args.min_face_size,
            min_plate_size=args.min_plate_size,
            min_face_clarity=args.min_face_clarity,
            face_area_px2=args.face_area_px2,
            face_dist_ratio=args.face_dist_ratio,
            min_text_conf=args.min_text_conf,
            min_text_len=args.min_text_len,
            use_gpu=args.use_gpu,
            gpu_id=args.gpu_id,
            enable_torch_opt=not args.no_torch_opt,
            gc_frequency=args.gc_frequency,
            progress_update_frequency=args.progress_update_frequency
        )
        try:
            classifier.run()
        except KeyboardInterrupt:
            print("⚡ 用户中断")
    else:
        print("ℹ️ 未启用 --classify，流程结束。")

if __name__ == "__main__":
    main()
