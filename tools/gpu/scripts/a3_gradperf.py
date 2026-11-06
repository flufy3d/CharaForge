"""A3 梯度检查与性能测量。

  tools/.venv/bin/python -I tools/gpu/scripts/a3_gradperf.py [--dump .data/gpu/frames/s24b] [--out .data/gpu/a3_gradperf.json]

梯度检查（s24 一帧，640×360 合成分辨率，前向仍是无抗锯齿的 runtime 等价渲染）：
  目标图 T = 把若干 drawable 整体平移 Δ（2 px）后的渲染；损失 L(x) = mean((I(x) − T)²)。
  对每个被测 drawable 的平移方向 v，比较自动求导的方向导数 ∇L·v 与中心差分 [L(x+εv) − L(x−εv)]/2ε（ε = 0.5、1 px），
  报告符号一致率与相关系数；另外对 drawable 不透明度、乘算色、图层平移做同样比较（这些是光滑量，差分应很接近）。
性能：
  单帧前向 + 反向（640×360、1280×720），批量 32 帧（640×360，按显存分块）耗时与显存峰值。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/gpu"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from l2d import composite as C  # noqa: E402
from l2d.dumpio import Dump  # noqa: E402
from l2d.raster import LayerInput, load_rasterizer  # noqa: E402


def scaled(inp: LayerInput, s: float) -> LayerInput:
    """合成分辨率缩放 s：to_screen 整体乘 s。"""
    return LayerInput(inp.positions, inp.opacities, inp.multiply, inp.screen, inp.render_orders, inp.dyn_flags, inp.to_screen * s)


def pipeline(R, inp, W, H, draw, aa=True, aa_layers=None):
    buf = R.render(inp, W, H, aa=aa, quantize=False, aa_layers=aa_layers)
    return C.composite([(buf, draw)], H, W, None, quantize=False, B=buf.shape[0])


def drawable_translation(R, idx: list[int], d: tuple[float, float], V: int) -> torch.Tensor:
    """把一组 drawable 的全部顶点平移 d（画布像素）的位移场 [V, 2]。"""
    g = R.geom
    v = torch.zeros((V, 2), device="cuda")
    for i in idx:
        v[g.vo[i]:g.vo[i + 1]] = torch.tensor(d, device="cuda")
    return v


def grad_check(R, L: dict, out: dict) -> None:
    W, H, s = 640, 360, 0.5
    base = scaled(LayerInput.from_dump([L]), s)
    draw = {**L["draw"], "filters": []}  # 只看光栅化本身（滤镜对位置的梯度是线性卷积，单独无须检查）
    g = R.geom
    a = L["arrays"]
    vis = np.nonzero(((a["dynamicFlags"] & 1) > 0) & (a["opacities"] > 0.2))[0]
    # 屏幕上面积足够大的 drawable
    P = base.positions[0]
    ts = base.to_screen[0].float()
    S = (P @ ts[:, :2].T + ts[:, 2]).cpu().numpy()
    big = []
    for i in vis:
        sl = S[g.vo[i]:g.vo[i + 1]]
        if len(sl) and (sl.max(0) - sl.min(0)).prod() > 400 and (sl.min(0) > 0).all() and (sl.max(0) < [W, H]).all():
            big.append(int(i))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(big, size=min(24, len(big)), replace=False).tolist())
    shift = 4.0  # 画布像素（≈ 2 px 屏幕位移 × 1/s... to_screen ≈ 0.44·0.5）
    Vn = base.positions.shape[1]
    with torch.no_grad():
        T = pipeline(R, LayerInput(base.positions + drawable_translation(R, pick, (shift, -shift), Vn)[None], *[getattr(base, k) for k in ("opacities", "multiply", "screen", "render_orders", "dyn_flags", "to_screen")]), W, H, draw, aa=False)

    def loss_at(pos, aa=False, **kw):
        inp = LayerInput(pos, kw.get("op", base.opacities), kw.get("mul", base.multiply), base.screen, base.render_orders, base.dyn_flags, kw.get("ts", base.to_screen))
        I = pipeline(R, inp, W, H, draw, aa=aa)
        return ((I - T) ** 2).mean()

    pos = base.positions.clone().requires_grad_(True)
    t0 = time.time()
    Lx = loss_at(pos, aa=True)
    Lx.backward()
    grad = pos.grad[0]
    out["gradTimeS"] = round(time.time() - t0, 3)
    rows = []
    for i in pick:
        for dvec in ((1.0, 0.0), (0.0, 1.0)):
            v = drawable_translation(R, [i], dvec, Vn)
            ad = float((grad * v).sum())
            fd = {}
            with torch.no_grad():
                for eps in (0.5, 1.0, 2.0):
                    lp = float(loss_at(base.positions + eps * v[None]))
                    lm = float(loss_at(base.positions - eps * v[None]))
                    fd[eps] = (lp - lm) / (2 * eps)
            rows.append({"drawable": i, "dir": dvec, "ad": ad, "fd": fd})
    ad = np.array([r["ad"] for r in rows])
    res = {}
    for eps in (0.5, 1.0, 2.0):
        fd = np.array([r["fd"][eps] for r in rows])
        sig = np.abs(fd) > 1e-3 * np.abs(fd).max()
        res[f"eps{eps}"] = {"signAgree": float((np.sign(ad[sig]) == np.sign(fd[sig])).mean()), "n": int(sig.sum()),
                            "corr": float(np.corrcoef(ad, fd)[0, 1]), "ratioMedian": float(np.median(ad[sig] / fd[sig]))}
    # 整组平移（目标方向）：−∇L·Δ 应 > 0
    vt = drawable_translation(R, pick, (shift, -shift), Vn)
    res["towardTarget"] = {"adDirectional": float((grad * vt).sum()), "expect": "< 0（朝目标移动使损失下降）"}
    out["vertex"] = res
    out["vertexRows"] = rows

    # 光滑量：不透明度、乘算色、图层平移
    op = base.opacities.clone().requires_grad_(True)
    mul = base.multiply.clone().requires_grad_(True)
    tx = torch.zeros((), dtype=torch.float64, device="cuda", requires_grad=True)
    ts2 = base.to_screen.clone()

    def ts_of(t):
        m = ts2.clone()
        m[:, 0, 2] = m[:, 0, 2] + t
        return m

    Ls = loss_at(base.positions, aa=True, op=op, mul=mul, ts=ts_of(tx))
    Ls.backward()
    smooth = []
    with torch.no_grad():
        for i in pick[:8]:
            e = 0.02
            dop = torch.zeros_like(base.opacities)
            dop[0, i] = e
            fdv = (float(loss_at(base.positions, op=base.opacities + dop)) - float(loss_at(base.positions, op=base.opacities - dop))) / (2 * e)
            smooth.append({"kind": "opacity", "drawable": i, "ad": float(op.grad[0, i]), "fd": fdv})
            dm = torch.zeros_like(base.multiply)
            dm[0, i, :3] = e
            fdv = (float(loss_at(base.positions, mul=base.multiply + dm)) - float(loss_at(base.positions, mul=base.multiply - dm))) / (2 * e)
            smooth.append({"kind": "multiply", "drawable": i, "ad": float(mul.grad[0, i, :3].sum()), "fd": fdv})
        fdl = [(float(loss_at(base.positions, ts=ts_of(torch.tensor(e, dtype=torch.float64, device="cuda")))) - float(loss_at(base.positions, ts=ts_of(torch.tensor(-e, dtype=torch.float64, device="cuda"))))) / (2 * e) for e in (0.5, 1.0)]
    out["smooth"] = smooth
    out["layerTranslateX"] = {"ad": float(tx.grad), "fd_eps0.5": fdl[0], "fd_eps1": fdl[1]}


def bench(R, layers: list[dict], W: int, H: int, s: float, B: int, chunk: int, aa=True, reps=3, aa_layers=None) -> dict:
    """B 帧（分块 chunk 帧一次）前向 + 反向（对顶点）的耗时与显存峰值。"""
    draw = layers[0]["draw"]
    inps = [scaled(LayerInput.from_dump(layers[i:i + chunk]), s) for i in range(0, B, chunk)]
    times = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for r in range(reps + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        for inp in inps:
            pos = inp.positions.clone().requires_grad_(True)
            x = LayerInput(pos, inp.opacities, inp.multiply, inp.screen, inp.render_orders, inp.dyn_flags, inp.to_screen)
            I = pipeline(R, x, W, H, draw, aa=aa, aa_layers=aa_layers)
            I.square().mean().backward()
        torch.cuda.synchronize()
        if r > 0:
            times.append(time.time() - t0)
    fw = []
    for r in range(reps + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for inp in inps:
                pipeline(R, inp, W, H, draw, aa=False)
        torch.cuda.synchronize()
        if r > 0:
            fw.append(time.time() - t0)
    return {"W": W, "H": H, "frames": B, "chunk": chunk, "aa": aa if aa_layers is None else f"front{aa_layers}", "fwdBwdS": round(float(np.median(times)), 4),
            "fwdOnlyS": round(float(np.median(fw)), 4), "peakGB": round(torch.cuda.max_memory_allocated() / 1e9, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(REPO / ".data/gpu/frames/s24b"))
    ap.add_argument("--out", default=str(REPO / ".data/gpu/a3_gradperf.json"))
    ap.add_argument("--skip-grad", action="store_true")
    ap.add_argument("--perf-only", action="store_true", help="同 --skip-grad")
    args = ap.parse_args()
    # 本代理的显存上限 4 GB：超出时抛 OOM（记录为 OOM），不影响其他进程
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4.0e9 / torch.cuda.get_device_properties(0).total_memory))
    d = Dump(args.dump)
    R = load_rasterizer(d, "dsc")
    out: dict = {"textureGB": round(torch.cuda.memory_allocated() / 1e9, 3)}
    frames = d.frames
    layers = [next(L for L in d.frame(f)["layers"] if L["visible"]) for f in frames]
    if not args.skip_grad and not args.perf_only:
        grad_check(R, layers[len(layers) // 2], out)
        print(json.dumps({k: out[k] for k in ("vertex", "layerTranslateX")}, indent=1, ensure_ascii=False), flush=True)
        print(json.dumps(out["smooth"], indent=1), flush=True)
    perf = []
    for (W, H, s, B, chunk, aa) in [(640, 360, 0.5, 1, 1, True), (1280, 720, 1.0, 1, 1, True), (640, 360, 0.5, 1, 1, False), (1280, 720, 1.0, 1, 1, False),
                                     (1280, 720, 1.0, 1, 1, 8),
                                     (640, 360, 0.5, 32, 1, True), (640, 360, 0.5, 32, 4, True), (640, 360, 0.5, 32, 8, 8), (640, 360, 0.5, 32, 8, False)]:
        if B > len(layers):
            continue
        try:
            torch.cuda.empty_cache()
            r = bench(R, layers, W, H, s, B, chunk, aa is not False, aa_layers=aa if aa is not True and aa is not False else None)
        except torch.cuda.OutOfMemoryError:
            r = {"W": W, "H": H, "frames": B, "chunk": chunk, "aa": aa, "error": "OOM（> 4 GB 上限）"}
        perf.append(r)
        print(r, flush=True)
    out["perf"] = perf
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
