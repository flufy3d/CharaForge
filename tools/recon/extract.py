"""第 1 步：一次性解码整支视频。

产出：
  .data/recon/frames/NNNNNN.jpg  每一帧的全分辨率 JPEG（帧号 0 起，供人工查看/标注/跟踪）
  .data/recon/small.npy          320x180 RGB 低分辨率缓存（切点检测用）
  .data/recon/video.json         ffprobe 信息 + sha256
用法：tools/.venv/bin/python -I tools/recon/extract.py [--force]
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import DATA, FRAMES, SMALL, SH, SW, decode_iter, probe, sha256  # noqa: E402


def main() -> None:
    force = "--force" in sys.argv
    DATA.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    info = probe()
    info["sha256"] = sha256()
    n = info["frameCount"]
    if SMALL.exists() and not force and len(list(FRAMES.glob("*.jpg"))) == n:
        print("已存在，跳过（--force 重新解码）")
    else:
        small = np.lib.format.open_memmap(SMALL, mode="w+", dtype=np.uint8, shape=(n, SH, SW, 3))
        i = 0
        for fr in decode_iter():
            if i >= n:
                raise RuntimeError("解码帧数多于 ffprobe 包数")
            cv2.imwrite(str(FRAMES / f"{i:06d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            small[i] = cv2.resize(fr, (SW, SH), interpolation=cv2.INTER_AREA)
            i += 1
            if i % 500 == 0:
                print(i, "/", n, flush=True)
        small.flush()
        if i != n:
            raise RuntimeError(f"解码帧数 {i} ≠ {n}")
    (DATA / "video.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
