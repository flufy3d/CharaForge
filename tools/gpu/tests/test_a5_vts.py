"""A5：torch 版 VTS 前向（tools/gpu/fit/vts.py）与 TS 版（tools/fit/src/vts.ts）对拍。

    tools/.venv/bin/python -I tools/gpu/tests/test_a5_vts.py
需要 node（--experimental-strip-types）。随机输入（部分输入缺省 = 中性值）、两种 decouple、三个平滑系数 c。
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPO = HERE.parents[2]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from l2d import moc3  # noqa: E402
from fit.vts import Vts  # noqa: E402

VTUBE = REPO / "assets/models/deepseek-chan/DeepSeek/大肥鱼.vtube.json"


def model_defaults() -> dict:
    m = moc3.load(str(REPO / "assets/models/deepseek-chan/DeepSeek/大肥鱼.moc3"))
    p = m.groups["parameters"]
    return {k: float(v) for k, v in zip(p["ids"], p["defaultValues"])}


def ts_forward(q: dict, decouple: bool) -> dict:
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "in.json", Path(d) / "out.json"
        a.write_text(json.dumps(q))
        env = {**os.environ, "FIT_MODEL": json.dumps({"decouple": decouple})}
        subprocess.run(["node", "--experimental-strip-types", "--disable-warning=ExperimentalWarning",
                        str(REPO / "tools/gpu/scripts/a5_vts_ref.ts"), str(a), str(b)], check=True, env=env, cwd=REPO)
        return json.loads(b.read_text())


class TestVts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.defaults = model_defaults()

    def check(self, decouple: bool, c: float, seed: int):
        rng = np.random.default_rng(seed)
        v = Vts(VTUBE, self.defaults, decouple=decouple)
        n, f0 = 97, 2368
        # 一部分输入给随机序列（含突变），其余缺省（中性值）
        names = [k for k in v.inputs if rng.random() < 0.7]
        inputs = {k: (rng.normal(size=n).cumsum() * 3 + rng.normal(size=n)).tolist() for k in names}
        q = {"defaults": self.defaults, "inputs": inputs, "n": n, "f0": f0, "smoothC": c, "breathPeriod": 3.3, "breathPhase": 0.7}
        ref = ts_forward(q, decouple)
        self.assertEqual(ref["inputs"], v.inputs)
        for k, x in ref["neutral"].items():
            self.assertAlmostEqual(x, v.neutral[k], places=12, msg=k)
        out = v.forward({k: torch.tensor(x, dtype=torch.float64) for k, x in inputs.items()}, n, f0, c,
                        torch.tensor(3.3, dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64), device="cpu")
        self.assertEqual(set(ref["outputs"]), set(out))
        worst = 0.0
        for k, y in ref["outputs"].items():
            e = float(np.abs(np.asarray(y) - out[k].numpy()).max() / (1 + np.abs(y).max()))
            worst = max(worst, e)
            self.assertLess(e, 1e-10, msg=f"{k} decouple={decouple} c={c}")
        return worst

    def test_against_ts(self):
        worst = 0.0
        for decouple in (False, True):
            for c, seed in ((0.15, 1), (0.0, 2), (0.6, 3)):
                worst = max(worst, self.check(decouple, c, seed))
        print(f"VTS torch vs TS：最大相对误差 {worst:.2e}")

    def test_grad(self):
        """平滑矩阵求导：对输入序列的梯度 = Kᵀ·g（与解析一致）。"""
        v = Vts(VTUBE, self.defaults)
        n = 20
        x = torch.randn(n, dtype=torch.float64, requires_grad=True)
        y = v.forward({"FaceAngleZ": x}, n, 0, 0.15, torch.tensor(3.5, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), device="cpu")["AngleZ"]
        self.assertTrue(torch.autograd.gradcheck(lambda t: v.forward({"FaceAngleZ": t}, n, 0, 0.15, 3.5, 0.0, device="cpu")["AngleZ"], (x,)))
        del y


if __name__ == "__main__":
    unittest.main()
