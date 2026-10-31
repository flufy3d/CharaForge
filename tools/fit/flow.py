"""稠密光流（拟合用的对应关系）：我们的渲染（按 alpha 合成到参考帧上）→ 参考帧。

  tools/.venv/bin/python -I tools/fit/flow.py <manifest.json>

manifest：{"frames": [{"frame": f, "ours": ".../ours_f.rgba", "ref": ".../a_b.rgb", "refIndex": i,
                       "excl": [[x,y,w,h], ...]}], "out": "<dir>"}
  ours：1280×720 预乘 RGBA8 原始数据；ref：RefFrames 的缓存文件（RGB24 连续帧），取第 refIndex 帧。
输出：<out>/flow_<f>.f32，形状 (360, 640, 3) float32：dx, dy（全分辨率像素，ours 像素 → 参考帧对应位置的位移）、
有效标记（前后向一致、在我们的角色遮罩内、不在排除区）。采样点为全分辨率的偶数像素 (2i, 2j)。

方法：OpenCV DIS（PRESET_MEDIUM，带变分细化），灰度图先做 σ=1 高斯平滑（两边软硬程度不同）；
前后向一致性：|F(x) + B(x + F(x))| < 1.5 + 0.05·|F(x)|。
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720


def one(args) -> None:
    it, out = args
    cv2.setNumThreads(1)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    refs: dict[str, np.memmap] = {}
    if True:
        f = it["frame"]
        ours = np.fromfile(it["ours"], np.uint8).reshape(H, W, 4).astype(np.float32)
        if it["ref"] not in refs:
            refs[it["ref"]] = np.memmap(it["ref"], np.uint8, "r")
        ref = np.asarray(refs[it["ref"]][it["refIndex"] * W * H * 3:(it["refIndex"] + 1) * W * H * 3]).reshape(H, W, 3).astype(np.float32)
        a = ours[..., 3:] / 255.0
        comp = ours[..., :3] + (1 - a) * ref
        g1 = cv2.GaussianBlur(cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        g2 = cv2.GaussianBlur(cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
        fw = dis.calc(g1, g2, None)
        bw = dis.calc(g2, g1, None)
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        mx, my = xs + fw[..., 0], ys + fw[..., 1]
        bwx = cv2.remap(bw[..., 0], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e4)
        bwy = cv2.remap(bw[..., 1], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e4)
        err = np.hypot(fw[..., 0] + bwx, fw[..., 1] + bwy)
        valid = (err < 1.5 + 0.05 * np.hypot(fw[..., 0], fw[..., 1])) & (a[..., 0] > 0.5)
        for x, y, w, h in it.get("excl", []):
            x0, y0 = max(0, int(np.ceil(x - 0.5))), max(0, int(np.ceil(y - 0.5)))
            x1, y1 = min(W, int(np.ceil(x + w - 0.5))), min(H, int(np.ceil(y + h - 0.5)))
            if x1 > x0 and y1 > y0:
                valid[y0:y1, x0:x1] = False
        o = np.stack([fw[::2, ::2, 0], fw[::2, ::2, 1], valid[::2, ::2].astype(np.float32)], -1).astype(np.float32)
        o.tofile(out / f"flow_{f}.f32")


def main() -> None:
    man = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(man["out"])
    out.mkdir(parents=True, exist_ok=True)
    # 逐帧独立：多进程并行（每进程单线程 OpenCV）
    with Pool(min(8, max(1, len(man["frames"])))) as pool:
        pool.map(one, [(it, out) for it in man["frames"]])
    print(f"flow: {len(man['frames'])} 帧 → {out}")


if __name__ == "__main__":
    main()
