"""
A1 变形器测试（快速版；完整对拍用 tools/gpu/scripts/a1_compare.py）。

    tools/.venv/bin/python -I tools/gpu/tests/test_a1_deform.py      （也可用 pytest）

- warp / rotation 变换的自定义反向：torch.autograd.gradcheck（float64，点覆盖网格内、带状区、远处，含三角形模式）；
- 与 Core 对拍（需要先跑 a1_make_refs.py 生成 .data/gpu/ref/；没有就跳过）：每组取 48 个样本，float32 / float64 都要过门槛；
- 整个前向对参数的梯度：float64 中心差分 vs autograd。
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1_core import MOC, REPO, load_ref  # noqa: E402
from l2d import moc3  # noqa: E402
from l2d.deform import L2D, _RotApply, _WarpApply  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
REF = REPO / '.data/gpu/ref'


def _sub(ref, n):
    idx = np.linspace(0, ref['params'].shape[0] - 1, n).astype(int)
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == ref['params'].shape[0] else v) for k, v in ref.items()}


class TestDeform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = moc3.load(MOC)

    def test_warp_gradcheck(self):
        torch.manual_seed(0)
        B = 2
        # 两个 warp：3×2 与 2×4 网格（列×行），随机扰动；第二个 isQuadSource = 0
        cols = torch.tensor([3, 2], device=DEV)
        rows = torch.tensor([2, 4], device=DEV)
        goff = torch.tensor([0, 12], device=DEV)
        quad = torch.tensor([True, False], device=DEV)
        G = 12 + 15
        grid = []
        for C, R in ((3, 2), (2, 4)):
            uu, vv = torch.meshgrid(torch.linspace(0, 1, C + 1), torch.linspace(0, 1, R + 1), indexing='xy')
            grid.append(torch.stack([uu + 0.3 * vv, vv + 0.2 * uu ** 2], -1).reshape(-1, 2))
        grid = (torch.cat(grid)[None].repeat(B, 1, 1) + 0.05 * torch.randn(B, G, 2)).double().to(DEV).requires_grad_(True)
        # 仿射外推项（独立参数，检查对它们的梯度）
        c = torch.randn(B, 2, 2, dtype=torch.float64, device=DEV, requires_grad=True)
        du = torch.randn(B, 2, 2, dtype=torch.float64, device=DEV, requires_grad=True)
        dv = torch.randn(B, 2, 2, dtype=torch.float64, device=DEV, requires_grad=True)
        # 点：网格内、带状区（四边与四角）、远处；避开格线（不可导）
        pts = torch.tensor([[0.13, 0.27], [0.71, 0.88], [0.52, 0.41], [-0.7, 0.37], [1.6, 0.21], [0.43, -1.3], [0.29, 2.2],
                            [-1.1, -0.6], [2.4, 2.7], [-2.6, 0.4], [3.6, -3.3], [0.9, 4.1]], dtype=torch.float64)
        N = pts.shape[0]
        p = (pts[None].repeat(B, 1, 1) + 0.01 * torch.randn(B, N, 2, dtype=torch.float64)).to(DEV).requires_grad_(True)
        pid = torch.tensor([0, 1] * (N // 2), device=DEV)
        meta = (pid, cols[pid], rows[pid], goff[pid], quad[pid])
        for chunk in (5, 1024):
            f = lambda p, grid, c, du, dv: _WarpApply.apply(p, grid, c, du, dv, *meta, chunk)  # noqa: E731
            self.assertTrue(torch.autograd.gradcheck(f, (p, grid, c, du, dv), eps=1e-6, atol=1e-7))

    def test_rot_gradcheck(self):
        torch.manual_seed(1)
        B, nR, N = 2, 3, 7
        d = dict(dtype=torch.float64, device=DEV)
        p = torch.randn(B, N, 2, **d).mul(50).requires_grad_(True)
        o = torch.randn(B, nR, 2, **d).requires_grad_(True)
        ang = torch.randn(B, nR, **d)
        cs = torch.cos(ang).requires_grad_(True)
        sn = torch.sin(ang).requires_grad_(True)
        sc = torch.rand(B, nR, **d).add(0.1).requires_grad_(True)
        rx = torch.tensor([[1.0, -1.0, 1.0]] * B, **d)
        ry = torch.tensor([[1.0, 1.0, -1.0]] * B, **d)
        pid = torch.randint(0, nR, (N,), device=DEV)
        f = lambda p, o, cs, sn, sc: _RotApply.apply(p, o, cs, sn, sc, rx, ry, pid)  # noqa: E731
        self.assertTrue(torch.autograd.gradcheck(f, (p, o, cs, sn, sc), eps=1e-6, atol=1e-7))

    def test_against_core(self):
        names = [n for n in ('random', 'special', 'scene', 'partop') if (REF / f'{n}.json').exists()]
        if not names:
            self.skipTest('没有 .data/gpu/ref/：先运行 a1_make_refs.py')
        from a1_compare import compare, evaluate
        for dt in (torch.float64, torch.float32):
            model = L2D(self.m, device=DEV, dtype=dt)
            for n in names:
                ref = _sub(load_ref(REF / n), 48)
                r = compare(model, ref, evaluate(model, ref, 16))
                with self.subTest(ref=n, dtype=str(dt)):
                    self.assertLess(r['vertex_px']['max'], 0.1)
                    self.assertLess(r['opacity']['max'], 1e-4)
                    self.assertLess(r['multiply']['max'], 1e-4)
                    self.assertLess(r['screen']['max'], 1e-4)
                    self.assertEqual(r['renderOrder_mismatch_samples'], 0)
                    self.assertEqual(r['drawOrder_mismatch'], 0)
                    self.assertEqual(r['visible_mismatch'], 0)
                    self.assertEqual(r['vertexUpdated_mismatch'], 0)

    def test_param_gradient(self):
        if not (REF / 'scene.json').exists():
            self.skipTest('没有 .data/gpu/ref/scene：先运行 a1_make_refs.py scene')
        from a1_grad_bench import grad_check
        r = grad_check(8, seed=3)
        self.assertLess(r['max_rel_err'], 1e-4)

    def test_float32_close_to_float64(self):
        P = torch.as_tensor(self.m['parameters']['defaultValues'][None]).repeat(3, 1)
        P[1] = torch.as_tensor(self.m['parameters']['minValues'])
        P[2] = torch.as_tensor(self.m['parameters']['maxValues'])
        with torch.no_grad():
            a = L2D(self.m, device=DEV, dtype=torch.float64)(P)
            b = L2D(self.m, device=DEV, dtype=torch.float32)(P)
        self.assertTrue(torch.equal(a.render_order, b.render_order))
        self.assertTrue(torch.equal(a.visible, b.visible))
        vis = a.visible.repeat_interleave(torch.as_tensor(np.diff(L2D(self.m, device=DEV).vertex_offsets), device=a.visible.device), dim=1)
        self.assertLess(float(((a.verts - b.verts.double()).norm(dim=-1) * vis).max()), 0.1)


if __name__ == '__main__':
    unittest.main()
