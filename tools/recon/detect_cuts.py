"""第 2 步：自动检测切点候选。

对相邻帧 (i-1, i) 计算：
  mad    低分辨率 RGB 平均绝对差（0~255）
  hist   HSV 直方图 Bhattacharyya 距离（0~1）
  ecr    边缘变化率（Edge Change Ratio，0~1）
  lum    第 i 帧平均亮度（判断闪白/闪黑）
  white  第 i 帧近白像素占比（白色剪影/闪白转场）
候选规则：mad 或 hist 相对局部中位数的突跳（峰值且明显高于前后各 W 帧的中位数），
再加亮度剧变。自动结果只是候选，最终切点由人工逐个看图（cut_sheets.py 生成前后各 2 帧的条带）确认，
写入 annotations.json。

产出 .data/recon/metrics.npz、.data/recon/cut_candidates.json
用法：tools/.venv/bin/python -I tools/recon/detect_cuts.py
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import DATA, load_small  # noqa: E402


def compute_metrics(small: np.ndarray) -> dict[str, np.ndarray]:
    n = small.shape[0]
    mad = np.zeros(n, np.float32)
    hist = np.zeros(n, np.float32)
    ecr = np.zeros(n, np.float32)
    lum = np.zeros(n, np.float32)
    white = np.zeros(n, np.float32)
    prev_h = prev_e = prev = None
    k = np.ones((3, 3), np.uint8)
    for i in range(n):
        f = np.asarray(small[i])
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h)
        e = cv2.Canny(g, 60, 150) > 0
        lum[i] = g.mean()
        white[i] = float((g > 235).mean())
        if prev is not None:
            mad[i] = np.abs(f.astype(np.int16) - prev.astype(np.int16)).mean()
            hist[i] = cv2.compareHist(prev_h, h, cv2.HISTCMP_BHATTACHARYYA)
            ed = cv2.dilate(e.astype(np.uint8), k) > 0
            ped = cv2.dilate(prev_e.astype(np.uint8), k) > 0
            ein = (e & ~ped).sum() / max(e.sum(), 1)
            eout = (prev_e & ~ed).sum() / max(prev_e.sum(), 1)
            ecr[i] = max(ein, eout)
        prev, prev_h, prev_e = f, h, e
    return {"mad": mad, "hist": hist, "ecr": ecr, "lum": lum, "white": white}


def candidates(m: dict[str, np.ndarray], win: int = 8) -> list[dict]:
    n = len(m["mad"])
    out = []
    for i in range(1, n):
        lo, hi = max(1, i - win), min(n, i + win + 1)
        nb = np.r_[m["mad"][lo:i], m["mad"][i + 1:hi]]
        nbh = np.r_[m["hist"][lo:i], m["hist"][i + 1:hi]]
        med, medh = float(np.median(nb)), float(np.median(nbh))
        r_mad = (m["mad"][i] - med) / (med + 2.0)
        r_h = (m["hist"][i] - medh) / (medh + 0.03)
        peak = m["mad"][i] >= m["mad"][max(1, i - 1)] and m["mad"][i] >= m["mad"][min(n - 1, i + 1)]
        dl = abs(float(m["lum"][i] - m["lum"][i - 1]))
        score = max(r_mad, r_h / 3)
        if (peak and (r_mad > 2.5 or r_h > 6) and m["mad"][i] > 6) or dl > 25:
            out.append({"frame": i, "mad": round(float(m["mad"][i]), 2), "hist": round(float(m["hist"][i]), 3),
                        "ecr": round(float(m["ecr"][i]), 3), "dlum": round(dl, 1), "score": round(float(score), 2)})
    return out


def main() -> None:
    small = load_small()
    m = compute_metrics(small)
    np.savez(DATA / "metrics.npz", **m)
    c = candidates(m)
    (DATA / "cut_candidates.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    print(len(c), "个候选")
    for x in c:
        print(x)


if __name__ == "__main__" and "--flow" not in sys.argv:
    main()


def flow_residual(small: np.ndarray) -> np.ndarray:
    """运动补偿残差：用 Farneback 光流把 i-1 帧 warp 到 i 帧后的平均绝对误差。
    快速运动（推拉、平移）残差低，真正换画面残差高。用来补漏 mad/hist 被运动掩盖的硬切。"""
    n = small.shape[0]
    res = np.zeros(n, np.float32)
    prev = None
    for i in range(n):
        g = cv2.cvtColor(np.asarray(small[i]), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(g, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            yy, xx = np.mgrid[0:90, 0:160].astype(np.float32)
            w = cv2.remap(prev, xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            res[i] = np.abs(w.astype(np.float32) - g).mean()
        prev = g
    return res


if __name__ == "__main__" and "--flow" in sys.argv:
    r = flow_residual(load_small())
    np.save(DATA / "flow_residual.npy", r)
