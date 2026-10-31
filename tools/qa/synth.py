"""上限实验：用已知真值的渲染合成一段「参考视频」。

  tools/.venv/bin/python -I tools/qa/synth.py truth-scene --result .data/fit/s24/result.json --out .data/fit/synth/truth_scene.json
  pnpm render --scene .data/fit/synth/truth_scene.json --frames 2368-2441 --out .data/fit/synth/truth
  tools/.venv/bin/python -I tools/qa/synth.py encode --renders .data/fit/synth/truth --frames 2368-2441 \
      --out .data/fit/synth/s24_synth.mp4

退化方式模仿参考视频：
  1. 角色合成到一张缓慢滚动的白底浅紫圆点背景（与 s24 背景同类）；
  2. 整幅降到 0.6 倍（INTER_AREA）再放大回 1280×720（INTER_CUBIC），再做 σ=0.5 的高斯模糊——参考视频里角色明显
     比我们的渲染软、像从小画面放大（第一轮发现）；
  3. x264 编码（High、yuv420p、BT.709 TV range），码率 2.1 Mbps（原视频 ffprobe 为 2.07 Mbps），preset medium。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H, FPS = 1280, 720, 60


def background(f: int) -> np.ndarray:
    """白底浅紫圆点，每帧向左上滚动 1px（确定性）"""
    img = np.full((H, W, 3), (246, 246, 252), np.uint8)
    sp, r = 64, 15
    ox, oy = (-f) % sp, (-f // 2) % sp
    for j, y in enumerate(range(-sp + oy, H + sp, sp)):
        for x in range(-sp + ox + (sp // 2 if j % 2 else 0), W + sp, sp):
            cv2.circle(img, (int(x), int(y)), r, (124, 132, 222), -1, cv2.LINE_AA)
    return img


def cmd_truth_scene(args) -> None:
    res = json.loads(Path(args.result).read_text(encoding="utf-8"))
    scene = {
        "format": "charaforge.scene", "version": 1,
        "meta": {"title": f"上限实验真值（{res['shot']}）", "description": "tools/qa/synth.py 生成；参数取自该镜头第二轮拟合结果"},
        "timebase": {"fps": 60}, "duration": 3868,
        "composition": {"width": W, "height": H, "background": None, "antialias": True},
        "assets": {"models": [{"id": "dsc", "uri": "models/deepseek-chan/DeepSeek/大肥鱼.model3.json"}]},
        "layers": res["layers"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(scene), encoding="utf-8")
    print(f"truth scene → {args.out}")


def cmd_encode(args) -> None:
    a, b = (int(x) for x in args.frames.split("-"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
         "-c:v", "libx264", "-profile:v", "high", "-preset", "medium", "-b:v", "2100k", "-maxrate", "3000k", "-bufsize", "4200k",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(out)],
        stdin=subprocess.PIPE,
    )
    for f in range(a, b + 1):
        im = np.asarray(Image.open(Path(args.renders) / f"frame_{f:06d}.png").convert("RGBA"), dtype=np.float64)
        al = im[..., 3:] / 255.0
        comp = im[..., :3] * al + background(f).astype(np.float64) * (1 - al)
        small = cv2.resize(comp, (int(W * args.down), int(H * args.down)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (0, 0), args.sigma)
        enc.stdin.write(np.clip(np.round(up), 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg 失败")
    print(f"synthetic reference → {out}（{b - a + 1} 帧，第 0 帧 = 场景帧 {a}）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("truth-scene")
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("encode")
    p.add_argument("--renders", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--down", type=float, default=0.6)
    p.add_argument("--sigma", type=float, default=0.5)
    args = ap.parse_args()
    cmd_truth_scene(args) if args.cmd == "truth-scene" else cmd_encode(args)


if __name__ == "__main__":
    main()
