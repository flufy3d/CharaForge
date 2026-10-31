"""拟合内部用的 SSIM@360 打分（与 tools/qa/metric.py 的主指标同一定义；正式指标仍只来自 metric.py）。

  tools/.venv/bin/python -I tools/fit/score360.py <manifest.json>   （manifest 格式同 flow.py）
输出：JSON {帧号: SSIM@360}，遮罩 < 1% 画面的帧为 null。
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

W, H = 1280, 720
DISK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def one(it):
    cv2.setNumThreads(1)
    ours = np.fromfile(it["ours"], np.uint8).reshape(H, W, 4).astype(np.float64)
    ref = np.asarray(np.memmap(it["ref"], np.uint8, "r")[it["refIndex"] * W * H * 3:(it["refIndex"] + 1) * W * H * 3]).reshape(H, W, 3).astype(np.float64)
    a = ours[..., 3] / 255.0
    comp = ours[..., :3] + (1 - a[..., None]) * ref  # 预乘
    mask = cv2.dilate((a > 0.1).astype(np.uint8), DISK) > 0
    for x, y, w, h in it.get("excl", []):
        x0, y0 = max(0, int(np.ceil(x - 0.5))), max(0, int(np.ceil(y - 0.5)))
        x1, y1 = min(W, int(np.ceil(x + w - 0.5))), min(H, int(np.ceil(y + h - 0.5)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = False
    if mask.sum() < 0.01 * W * H:
        return it["frame"], None
    c = cv2.resize(comp, (640, 360), interpolation=cv2.INTER_AREA)
    r = cv2.resize(ref, (640, 360), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
    s = sum(structural_similarity(c[..., k], r[..., k], gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                  data_range=255, K1=0.01, K2=0.03, full=True)[1] for k in range(3)) / 3
    return it["frame"], float(s[m].mean())


def main() -> None:
    man = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    with Pool(min(8, max(1, len(man["frames"])))) as pool:
        res = pool.map(one, man["frames"])
    print(json.dumps({str(f): v for f, v in res}))


if __name__ == "__main__":
    main()
