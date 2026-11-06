"""A3：l2d.raster / l2d.composite 的单元测试与对拍回归。

  tools/.venv/bin/python -I -m unittest discover -s tools/gpu/tests -p "test_a3_*.py" -v

纯函数测试不依赖数据；对拍测试需要 a3_dump 的输出（.data/gpu/frames/main，见 tools/gpu/docs/raster.md），没有就跳过。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/gpu"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from l2d import composite as C  # noqa: E402
from l2d.raster import _mip_down_u8, _premultiply_u8, layout_bounds  # noqa: E402

DUMP = REPO / ".data/gpu/frames/main"
HAS_CUDA = torch.cuda.is_available()


class TestPure(unittest.TestCase):
    def test_premultiply(self):
        px = np.array([[[255, 128, 0, 128], [10, 20, 30, 0], [200, 100, 50, 255]]], dtype=np.uint8)
        out = _premultiply_u8(px)
        self.assertEqual(out[0, 0].tolist(), [128, 64, 0, 128])  # round(255·128/255)=128, round(128·128/255)=64
        self.assertEqual(out[0, 1].tolist(), [0, 0, 0, 0])
        self.assertEqual(out[0, 2].tolist(), [200, 100, 50, 255])

    def test_mip_down(self):
        t = torch.tensor([[[0, 0, 0, 0], [1, 2, 3, 255]], [[2, 2, 2, 0], [3, 2, 1, 255]]], dtype=torch.uint8)
        out = _mip_down_u8(t)
        self.assertEqual(out.shape, (1, 1, 4))
        self.assertEqual(out[0, 0].tolist(), [2, 2, 2, 128])  # (6+2)//4 = 2；(510+2)//4 = 128

    def test_layout_bounds(self):
        # 4 个上下文、1 张缓冲：RGBA 各一个，整张
        L = layout_bounds(4, 1)
        self.assertEqual([l["channel"] for l in L], [0, 1, 2, 3])
        self.assertTrue(all(l["bounds"] == (0.0, 0.0, 1.0, 1.0) for l in L))
        # 5 个：R 通道左右两半，其余整张
        L = layout_bounds(5, 1)
        self.assertEqual([l["channel"] for l in L], [0, 0, 1, 2, 3])
        self.assertEqual(L[0]["bounds"], (0.0, 0.0, 0.5, 1.0))
        self.assertEqual(L[1]["bounds"], (0.5, 0.0, 0.5, 1.0))
        # 49 个、2 张缓冲（本模型）：每张 25 → R 7 个（3×3 分格），GBA 各 6 个；第二张少 1 个
        L = layout_bounds(49, 2)
        self.assertEqual(sum(1 for l in L if l["buffer"] == 0), 25)
        self.assertEqual(sum(1 for l in L if l["buffer"] == 1), 24)
        self.assertEqual(L[6]["bounds"], (0.0, 2 / 3, 1 / 3, 1 / 3))

    def test_filters_identity(self):
        img = torch.rand(1, 16, 20, 4)
        img = torch.cat([img[..., :3] * img[..., 3:], img[..., 3:]], -1)
        q = C.quantize8(img)
        self.assertTrue(torch.equal(C.blur(q, 0.0), q))
        self.assertTrue(torch.equal(C.resample(q, 1.0), q))
        out = C.color(q)
        self.assertLessEqual(float((out - q).abs().max()), 1.5 / 255)

    def test_blur_mass(self):
        img = torch.zeros(1, 21, 21, 4)
        img[0, 10, 10] = 1.0
        out = C.blur(img, 2.0, quantize=False)
        self.assertAlmostEqual(float(out[..., 3].sum()), 1.0, places=5)
        # 边缘：缓冲外视为透明，按全部权重归一化 → 质量损失
        img2 = torch.zeros(1, 21, 21, 4)
        img2[0, 0, 0] = 1.0
        self.assertLess(float(C.blur(img2, 2.0, quantize=False)[..., 3].sum()), 0.5)

    def test_clip_coverage(self):
        cov = C.clip_coverage({"x": 10, "y": 10, "width": 20, "height": 10, "radius": 0, "feather": 0}, 1, 40, 40, torch.zeros(1))
        self.assertEqual(float(cov[0, 15, 15, 0]), 1.0)
        self.assertEqual(float(cov[0, 5, 5, 0]), 0.0)
        self.assertAlmostEqual(float(cov[0, 15, 9, 0]), 0.0, places=6)  # 像素中心 9.5，边界 10：d = 0.5 → 0
        circ = C.clip_coverage({"x": 0, "y": 0, "width": 40, "height": 40, "radius": 20, "feather": 4}, 1, 40, 40, torch.zeros(1))
        self.assertAlmostEqual(float(circ[0, 20, 20, 0]), 1.0)
        self.assertEqual(float(circ[0, 0, 0, 0]), 0.0)

    def test_blend(self):
        dst = torch.tensor([[[[0.2, 0.4, 0.6, 0.8]]]])
        src = torch.tensor([[[[0.5, 0.0, 0.25, 0.5]]]])
        n = C.blend(dst, src, "normal")[0, 0, 0]
        self.assertTrue(torch.allclose(n, torch.tensor([0.6, 0.2, 0.55, 0.9])))
        s = C.blend(dst, src, "screen")[0, 0, 0]
        self.assertTrue(torch.allclose(s, torch.tensor([0.5 + 0.2 * 0.5, 0.4, 0.25 + 0.6 * 0.75, 0.9])))
        m = C.blend(dst, src, "multiply")[0, 0, 0]
        self.assertTrue(torch.allclose(m, torch.tensor([0.5 * 0.2 + 0.1, 0.2, 0.25 * 0.6 + 0.3, 0.9])))


@unittest.skipUnless(HAS_CUDA and (DUMP / "manifest.json").exists(), "需要 CUDA 与 .data/gpu/frames/main（a3_dump）")
class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from l2d.dumpio import Dump
        from l2d.raster import load_rasterizer

        cls.dump = Dump(DUMP)
        cls.R = load_rasterizer(cls.dump, "dsc")

    def test_clip_contexts_match_framework(self):
        from l2d.raster import LayerInput, clip_matrices

        st = self.dump.model("dsc")
        g = self.R.geom
        self.assertEqual([c["masks"] for c in st["clipContexts"]], g.ctx_masks)
        self.assertEqual([c["clipped"] for c in st["clipContexts"]], g.ctx_clipped)
        L = self.dump.frame(2400)["layers"][0]
        ref = [(c["channel"], c["bufferIndex"]) for c in L["clipContexts"]]
        self.assertEqual(ref, [(l["channel"], l["buffer"]) for l in g.layouts])
        inp = LayerInput.from_dump([L])
        M = clip_matrices(g, g.to_units(inp.positions))[0].cpu().numpy()
        R = np.array([[[c["matrixForDraw"][0], c["matrixForDraw"][4], c["matrixForDraw"][12]],
                       [c["matrixForDraw"][1], c["matrixForDraw"][5], c["matrixForDraw"][13]]] for c in L["clipContexts"]])
        np.testing.assert_allclose(M, R, rtol=1e-5, atol=1e-5)

    def test_frame_parity(self):
        from l2d.raster import LayerInput

        f = 2400
        fr = self.dump.frame(f)
        layers = []
        with torch.no_grad():
            for L in fr["layers"]:
                if L["visible"]:
                    layers.append((self.R.render(LayerInput.from_dump([L]), fr["width"], fr["height"], aa=False), L["draw"]))
            comp = C.composite(layers, fr["height"], fr["width"], None, B=1)
        ours = torch.floor(comp[0].clamp(0, 1) * 255 + 0.5).cpu().numpy().astype(np.int32)
        ref = self.dump.ref_premul(f).astype(np.int32)
        d = np.abs(ours - ref)
        self.assertLess(d.mean(), 0.3)
        self.assertLess(int((d.max(-1) > 10).sum()), 50)

    def test_gradient_sign(self):
        from l2d.raster import LayerInput

        L = self.dump.frame(2400)["layers"][0]
        base = LayerInput.from_dump([L])
        base.to_screen = base.to_screen * 0.5
        g = self.R.geom
        i = int(np.argmax(np.diff(g.vo) * ((L["arrays"]["opacities"] > 0.5) & (L["arrays"]["dynamicFlags"] & 1 > 0))))
        v = torch.zeros_like(base.positions)
        v[0, g.vo[i]:g.vo[i + 1], 0] = 1.0

        def run(pos, aa):
            x = LayerInput(pos, base.opacities, base.multiply, base.screen, base.render_orders, base.dyn_flags, base.to_screen)
            return self.R.render(x, 640, 360, aa=aa, quantize=False)

        with torch.no_grad():
            T = run(base.positions + 3.0 * v, False)
        pos = base.positions.clone().requires_grad_(True)
        ((run(pos, True) - T) ** 2).mean().backward()
        ad = float((pos.grad * v).sum())
        with torch.no_grad():
            fd = (float(((run(base.positions + v, False) - T) ** 2).mean()) - float(((run(base.positions - v, False) - T) ** 2).mean())) / 2
        self.assertLess(ad, 0)  # 朝目标方向移动，损失下降
        self.assertEqual(math.copysign(1, ad), math.copysign(1, fd))
        self.assertLess(abs(ad / fd - 1), 0.5)


if __name__ == "__main__":
    unittest.main()
