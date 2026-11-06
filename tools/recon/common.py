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


# ---------------------------------------------------------------- head 框 ← 眼睛锚点（2026-11-06 用户决定：全片统一）
# 输出到 shots.json 的 head 框一律由两眼推出（build.py unify_heads），人工 head 框只用于标定比例和核对。
# 几何：在眼线坐标系（x 轴 = 左眼→右眼，倾角 θ）里，头部是 kw·d × kh·d 的矩形，中心 = 两眼中点 + (ox·d, oy·d)；
# 输出它旋转 θ 后的轴对齐外接框（d = 眼距）。不裁到画面内（出画的头也给完整大小，头高 = 视在大小）。
# 标定（B3，2026-11-06）：28 个实例 72 个“人工 head 框 + 人工/细化眼”的锚点，去掉钢盆（key10）、Q 版、贴画面边缘被裁的框，
# 每个实例取中位数后再取实例间中位数：kw 3.25、kh 2.82、ox 0.01、oy −0.25（取整到下面的值）。
# 残差：H 相对误差 p25/p50/p75 = −9%/+2%/+13%（p5/p95 = −31%/+53%），实例中位误差 p10~p90 = −18%~+15%。
HEAD_KW, HEAD_KH, HEAD_OX, HEAD_OY = 3.25, 2.8, 0.0, -0.25

# 锚点阶段（manual.py VH3/BODY3 等）用的旧比例：head 宽 2.78d、高 3.0d，两眼在框高 55% 处。
# 这些框只作为跟踪区域/细化合理性检查的初值；build.py 输出前按这个比例反推 (两眼中点, 眼距)，再换成上面的统一比例。
ANCHOR_HEAD_KW, ANCHOR_HEAD_KH, ANCHOR_HEAD_EYE_Y = 2.78, 3.0, 0.55


def head_from_mid(mid, d, theta=0.0):
    """两眼中点 mid、眼距 d、眼线倾角 theta（弧度，图像坐标，顺时针为正）→ head [x, y, w, h]。"""
    import math
    c, s = math.cos(theta), math.sin(theta)
    cx = mid[0] + d * (c * HEAD_OX - s * HEAD_OY)
    cy = mid[1] + d * (s * HEAD_OX + c * HEAD_OY)
    w = d * (HEAD_KW * abs(c) + HEAD_KH * abs(s))
    h = d * (HEAD_KW * abs(s) + HEAD_KH * abs(c))
    return [cx - w / 2, cy - h / 2, w, h]


def head_from_eyes(eyes, rotation_hint=None):
    """两眼 [[x, y], [x, y]] → head [x, y, w, h]。眼线方向取 左→右（按 x 排序后）；
    rotation_hint（度）与眼线角度相差超过 90° 时（头部倒转超过 90°）把方向翻转 180°。"""
    import math
    (ax, ay), (bx, by) = sorted(eyes, key=lambda p: p[0])
    th = math.atan2(by - ay, bx - ax)
    if rotation_hint is not None:
        diff = (math.degrees(th) - rotation_hint + 180) % 360 - 180
        if abs(diff) > 90:
            th += math.pi
    return head_from_mid(((ax + bx) / 2, (ay + by) / 2), math.hypot(bx - ax, by - ay), th)


def anchor_head_eyes(head):
    """锚点阶段由眼睛（或画面外的眼睛估计）推出的 head 框 → (两眼中点, 眼距)；不是这个比例的框返回 None。"""
    x, y, w, h = head
    if h <= 0 or abs(w / h - ANCHOR_HEAD_KW / ANCHOR_HEAD_KH) > 0.004:
        return None
    return (x + w / 2, y + ANCHOR_HEAD_EYE_Y * h), h / ANCHOR_HEAD_KH
