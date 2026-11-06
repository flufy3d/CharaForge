"""A7：梯度对照。用 a3_gradperf.py 的梯度检查（自动求导方向导数 vs 中心差分）分别跑原实现与快速路径，
另外直接比较两条路径在同一输入上的 autograd 结果（无边缘梯度时应一致到浮点舍入；边缘梯度项本身逐次运行就有波动，单独报原实现自身两次的差）。

    tools/.venv/bin/python -I tools/gpu/scripts/a7_gradcheck.py [--dump .data/gpu/frames/s24b] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/gpu"))
sys.path.insert(1, str(REPO / "tools/gpu/scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import a3_gradperf as G  # noqa: E402
from l2d import raster  # noqa: E402
from l2d.dumpio import Dump  # noqa: E402
from l2d.raster import LayerInput, load_rasterizer  # noqa: E402


def direct(R, L, out):
    """同一输入：原实现 ×2 与快速路径的 autograd 结果。"""
    W, H, s = 640, 360, 0.5
    base = G.scaled(LayerInput.from_dump([L]), s)
    torch.manual_seed(0)
    with torch.no_grad():
        T = R.render(base, W, H, quantize=False, fast=False)
        T = T + 0.05 * torch.randn_like(T)
    names = ["positions", "to_screen", "opacities", "multiply", "screen"]
    rows = []
    for aa, aal, q in ((False, None, False), (True, None, False), (True, 8, True)):
        res = {}
        for tag, fast in (("orig", False), ("orig2", False), ("fast", True)):
            xs = [getattr(base, k).clone().requires_grad_(True) for k in names]
            x = LayerInput(xs[0], xs[2], xs[3], xs[4], base.render_orders, base.dyn_flags, xs[1])
            loss = ((R.render(x, W, H, aa=aa, aa_layers=aal, quantize=q, fast=fast) - T) ** 2).mean()
            loss.backward()
            res[tag] = [t.grad.detach().double().flatten() for t in xs]
        for i, nm in enumerate(names):
            a, b, c = res["orig"][i], res["fast"][i], res["orig2"][i]
            n = max(float(a.norm()), 1e-30)
            rows.append({"aa": aa, "aa_layers": aal, "quantize": q, "input": nm, "relErrFastVsOrig": float((a - b).norm()) / n,
                         "relErrOrigVsOrig": float((a - c).norm()) / n, "cosFastVsOrig": float((a * b).sum()) / max(n * float(b.norm()), 1e-30)})
            print(json.dumps(rows[-1]), flush=True)
    out["direct"] = rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(REPO / ".data/gpu/frames/s24b"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4.0e9 / torch.cuda.get_device_properties(0).total_memory))
    d = Dump(a.dump)
    R = load_rasterizer(d, "dsc")
    layers = [next(L for L in d.frame(f)["layers"] if L["visible"]) for f in d.frames]
    L = layers[len(layers) // 2]
    out = {}
    for tag, fast in (("orig", False), ("fast", True)):
        raster.FAST_DEFAULT = fast
        o = {}
        G.grad_check(R, L, o)
        out[tag] = {k: o[k] for k in ("vertex", "smooth", "layerTranslateX")}
        v = o["vertex"]
        sm = o["smooth"]
        summ = {}
        for kind in ("opacity", "multiply"):
            ad = np.array([r["ad"] for r in sm if r["kind"] == kind])
            fd = np.array([r["fd"] for r in sm if r["kind"] == kind])
            summ[kind] = {"signAgree": float((np.sign(ad) == np.sign(fd)).mean()), "corr": float(np.corrcoef(ad, fd)[0, 1]),
                          "ratioMedian": float(np.median(ad / fd))}
        out[tag]["smoothSummary"] = summ
        print(tag, json.dumps({"vertex": {k: v[k] for k in v}, "smooth": summ, "layerTranslateX": o["layerTranslateX"]}, ensure_ascii=False), flush=True)
    raster.FAST_DEFAULT = False
    direct(R, L, out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
