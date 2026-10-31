"""recon 公共工具：路径、视频解码、帧读取。

安全约定：视频是不可信数据，只交给 ffmpeg/OpenCV 解码，不执行任何内容；
本目录脚本一律用 `tools/.venv/bin/python -I` 运行。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "assets/video/DeepSeek-chan.mp4"
VIDEO_URI = "assets/video/DeepSeek-chan.mp4"
MODEL_DIR = ROOT / "assets/models/deepseek-chan/DeepSeek"
DATA = ROOT / ".data/recon"
FRAMES = DATA / "frames"          # 全分辨率 JPEG，frames/000000.jpg = 第 0 帧
SMALL = DATA / "small.npy"        # (N, SH, SW, 3) uint8 RGB 低分辨率缓存，用于切点检测
THUMBS = DATA / "thumbs"
SHOTS_JSON = ROOT / "scenes/deepseek-chan/shots.json"
SCHEMA = ROOT / "packages/scene/schema/shots.v1.json"
ANNOT = Path(__file__).resolve().parent / "annotations.json"   # 人工标注（受版本控制）

W, H, FPS = 1280, 720, 60
SW, SH = 320, 180


def probe() -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,nb_frames",
         "-of", "json", str(VIDEO)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": int(num) / int(den), "frameCount": int(s["nb_read_packets"])}


def sha256(path: Path = VIDEO) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_iter(width: int = W, height: int = H):
    """逐帧解码为 RGB uint8（按解码顺序 = 显示顺序，passthrough 不丢帧不补帧）。"""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(VIDEO), "-map", "0:v:0",
           "-fps_mode", "passthrough", "-vf", f"scale={width}:{height}:flags=area",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = width * height * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        p.stdout.close()
        p.wait()


def frame_path(i: int) -> Path:
    return FRAMES / f"{i:06d}.jpg"


def read_frame(i: int) -> np.ndarray:
    """读取第 i 帧（全分辨率 RGB）。"""
    import cv2
    im = cv2.imread(str(frame_path(i)), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(frame_path(i))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def load_small() -> np.ndarray:
    return np.load(SMALL, mmap_mode="r")


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)
