"""拼图工具：把若干帧排成网格，每格烧录帧号（可附加标注框/点）。

命令行（人工复核用）：
  sheet.py range A B STEP out.jpg [--cols N] [--tw W]     A..B（左闭右开）每 STEP 帧一格
  sheet.py frames f1,f2,... out.jpg [--cols N] [--tw W]   指定帧
  sheet.py crop f1,f2,... x,y,w,h out.jpg [--cols N] [--tw W]  只看某区域（放大看细节）
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import H, W, read_frame  # noqa: E402


def label(img: np.ndarray, text: str, scale: float = 0.5) -> None:
    th = max(1, int(round(scale * 2)))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    cv2.rectangle(img, (0, 0), (tw + 6, tht + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (3, tht + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), th, cv2.LINE_AA)


def grid(frames: list[int], cols: int = 6, tw: int = 320, crop=None, labels=None,
         overlay=None) -> np.ndarray:
    """返回 BGR 网格图。overlay(frame, img_rgb_fullres) 可在缩放前画标注。"""
    tiles = []
    for k, f in enumerate(frames):
        im = read_frame(f)
        if overlay is not None:
            im = im.copy()
            overlay(f, im)
        if crop is not None:
            x, y, w, h = crop
            im = im[y:y + h, x:x + w]
        th = int(round(im.shape[0] * tw / im.shape[1]))
        t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
        label(t, labels[k] if labels else str(f), 0.45 if tw < 300 else 0.6)
        tiles.append(t)
    th = tiles[0].shape[0]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (th + 2), cols * (tw + 2), 3), 40, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * (th + 2):r * (th + 2) + th, c * (tw + 2):c * (tw + 2) + tw] = t
    return out


def save(img: np.ndarray, path: str | Path, q: int = 88) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def main(argv: list[str]) -> None:
    cols, tw = 6, 320
    if "--cols" in argv:
        i = argv.index("--cols"); cols = int(argv[i + 1]); del argv[i:i + 2]
    if "--tw" in argv:
        i = argv.index("--tw"); tw = int(argv[i + 1]); del argv[i:i + 2]
    mode = argv[0]
    crop = None
    if mode == "range":
        a, b, s = int(argv[1]), int(argv[2]), int(argv[3])
        frames, out = list(range(a, b, s)), argv[4]
    elif mode == "frames":
        frames, out = [int(x) for x in argv[1].split(",")], argv[2]
    elif mode == "crop":
        frames = [int(x) for x in argv[1].split(",")]
        crop = tuple(int(v) for v in argv[2].split(","))
        out = argv[3]
    else:
        raise SystemExit(__doc__)
    save(grid(frames, cols, tw, crop), out)
    print(out, len(frames))


if __name__ == "__main__":
    _ = (W, H)
    main(sys.argv[1:])
