"""A5：torch 版 SSIM（tools/gpu/fit/loss.py）与 tools/qa/metric.py 对拍。

    tools/.venv/bin/python -I tools/gpu/tests/test_a5_loss.py [渲染目录（pnpm render 输出，含 s24 的帧）]
  1) ssim_map 与 metric.ssim_map 在任意图像上逐像素一致（随机图 + 边界）；
  2) 用 runtime 渲染的 s24 帧走 metric.frame_metric 的完整流程（合成到参考帧、全分辨率遮罩、INTER_AREA 降采样），
     把同样的降采样输入交给 torch 版，SSIM@360 应一致到 1e-6。
缺省渲染目录 .data/qa/render/s24_base；不存在时只跑第 1 项。
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools/qa"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import metric  # noqa: E402
from fit import loss  # noqa: E402

RENDERS = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else REPO / ".data/qa/render/s24_base"


class TestSsim(unittest.TestCase):
    def test_map_random(self):
        rng = np.random.default_rng(0)
        for H, W in ((37, 53), (90, 160), (11, 13)):
            x = rng.uniform(0, 255, (H, W, 3))
            y = np.clip(x + rng.normal(0, 30, (H, W, 3)), 0, 255)
            a = metric.ssim_map(x, y)
            b = loss.ssim_map(torch.tensor(x)[None], torch.tensor(y)[None])[0].numpy()
            self.assertLess(float(np.abs(a - b).max()), 1e-9, f"{H}x{W}")

    @unittest.skipUnless(RENDERS.exists(), "没有 s24 渲染")
    def test_metric_s24(self):
        shot, _ = (lambda s: (next(x for x in s["shots"] if x["id"] == "s24"), s))(metric.load_shots())
        frames = [2380, 2400, 2420, 2441]
        refs = metric.decode_ref(2368, 2442)
        worst = 0.0
        for f in frames:
            rgb, alpha = metric.load_render(RENDERS / f"frame_{f:06d}.png")
            ref = refs[f - 2368]
            want = metric.frame_metric(rgb, alpha, ref, shot, f)["ssim360"]
            comp = rgb * alpha[..., None] + ref.astype(np.float64) * (1 - alpha[..., None])
            mask = cv2.dilate((alpha > 0.1).astype(np.uint8), metric.DISK) > 0
            mask &= ~metric.exclusion_mask(shot, f)
            c360 = cv2.resize(comp, (640, 360), interpolation=cv2.INTER_AREA)
            r360 = cv2.resize(ref.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA)
            m360 = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
            S = loss.ssim_map(torch.tensor(c360)[None], torch.tensor(r360)[None])[0].numpy()
            got = float(S[m360].mean())
            worst = max(worst, abs(got - want))
            self.assertLess(abs(got - want), 1e-6, f"帧 {f}: torch {got} vs metric {want}")
        print(f"s24 SSIM@360 torch vs metric.py：最大差 {worst:.2e}")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
