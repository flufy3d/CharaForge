"""A1：梯度检查（float64 中心差分 vs autograd）与速度 / 显存测量。

    tools/.venv/bin/python -I tools/gpu/scripts/a1_grad_bench.py grad  [--n 24]
    tools/.venv/bin/python -I tools/gpu/scripts/a1_grad_bench.py bench [--dtype float32] [--batches 1,64,512]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1_core import MOC, REPO, load_ref  # noqa: E402
from l2d import moc3  # noqa: E402
from l2d.deform import KEY_RANGE_EPS, L2D  # noqa: E402


def grad_check(n: int, seed: int = 0):
    m = moc3.load(MOC)
    model = L2D(m, dtype=torch.float64)
    ref = load_ref(REPO / '.data/gpu/ref/scene')
    rng = np.random.default_rng(seed)
    P0 = torch.as_tensor(ref['params'][rng.integers(ref['params'].shape[0], size=2)], dtype=torch.float64)
    parts = torch.as_tensor(ref['parts'][:2], dtype=torch.float64)
    lo, hi = model._pmin[0].cpu(), model._pmax[0].cpu()
    # 固定的随机投影：损失 = Σ w_v · 顶点 + Σ w_o · 不透明度 + 颜色
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        o = model(P0, parts)
    Wv = torch.randn(o.verts.shape, generator=g, dtype=torch.float64, ).to(o.verts.device)
    Wo = torch.randn(o.opacity.shape, generator=g, dtype=torch.float64, ).to(o.verts.device)
    Wm = torch.randn(o.multiply.shape, generator=g, dtype=torch.float64, ).to(o.verts.device)

    def loss(P):
        o = model(P, parts)
        return (o.verts * Wv).sum() * 1e-3 + (o.opacity * Wo).sum() + (o.multiply * Wm).sum() + (o.screen * Wm).sum()

    # 选参数：被 keyform 用到、当前值离 key 和端点都足够远（避开分段点 / 吸附区），按梯度大小取前 n 个
    P = P0.clone().requires_grad_(True)
    L = loss(P)
    (gr,) = torch.autograd.grad(L, P)
    gr = gr.cpu()
    used = set(m.binding_param.tolist()) | set(m.blendshape_binding_param.tolist())
    cand = []
    for b in range(P0.shape[0]):
        for p in used:
            v = float(P0[b, p])
            keys = m.parameter_keys(p)
            span = float(hi[p] - lo[p])
            if span <= 0 or abs(gr[b, p]) == 0:
                continue
            d = min([abs(v - float(k)) for k in keys] + [v - float(lo[p]), float(hi[p]) - v])
            if d > 5 * KEY_RANGE_EPS + 1e-3:
                cand.append((float(abs(gr[b, p])), b, p))
    cand.sort(reverse=True)
    pick = cand[:n // 2] + cand[len(cand) // 2:len(cand) // 2 + n - n // 2]  # 大梯度 + 中等梯度各一半
    rows = []
    for _, b, p in pick:
        h = 1e-4 * max(1.0, float(hi[p] - lo[p]) / 2)
        with torch.no_grad():
            Pp = P0.clone(); Pp[b, p] += h
            Pm = P0.clone(); Pm[b, p] -= h
            fd = (float(loss(Pp)) - float(loss(Pm))) / (2 * h)
        ad = float(gr[b, p])
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-12)
        rows.append({'param': m['parameters']['ids'][p], 'sample': b, 'value': float(P0[b, p]), 'autograd': ad, 'fd': fd, 'rel_err': rel})
    worst = max(r['rel_err'] for r in rows)
    for r in rows:
        print(f"{r['param']:24s} v={r['value']:+9.4f} autograd={r['autograd']:+.6e} 差分={r['fd']:+.6e} 相对误差={r['rel_err']:.2e}")
    print(f'参数 {len(rows)} 个，最大相对误差 {worst:.2e}')
    return {'rows': rows, 'max_rel_err': worst}


def bench(dtype: str, batches: list[int], reps: int = 5):
    dt = getattr(torch, dtype)
    m = moc3.load(MOC)
    model = L2D(m, dtype=dt)
    ref = load_ref(REPO / '.data/gpu/ref/scene')
    allp = torch.as_tensor(ref['params'])
    out = []
    for B in batches:
        idx = torch.arange(B) % allp.shape[0]
        P = allp[idx].to('cuda', dt)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        # 前向（no_grad）
        with torch.no_grad():
            model(P)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                o = model(P)
            torch.cuda.synchronize()
            fwd = (time.perf_counter() - t) / reps
        mem_fwd = torch.cuda.max_memory_allocated() - base
        del o
        torch.cuda.reset_peak_memory_stats()
        # 前向 + 反向
        Pg = P.clone().requires_grad_(True)
        o = model(Pg)
        (o.verts.sum() + o.opacity.sum()).backward()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            Pg.grad = None
            o = model(Pg)
            (o.verts.sum() + o.opacity.sum()).backward()
        torch.cuda.synchronize()
        fb = (time.perf_counter() - t) / reps
        mem_fb = torch.cuda.max_memory_allocated() - base
        del o, Pg
        torch.cuda.empty_cache()
        r = {'B': B, 'dtype': dtype, 'forward_ms': fwd * 1e3, 'fwd_bwd_ms': fb * 1e3, 'forward_ms_per_sample': fwd * 1e3 / B,
             'peak_mem_fwd_MB': mem_fwd / 2**20, 'peak_mem_fwd_bwd_MB': mem_fb / 2**20}
        print(json.dumps(r, ensure_ascii=False))
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['grad', 'bench'])
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--dtype', default='float32')
    ap.add_argument('--batches', default='1,64,512')
    a = ap.parse_args(argv)
    if a.what == 'grad':
        r = grad_check(a.n)
        return 0 if r['max_rel_err'] < 1e-4 else 1
    bench(a.dtype, [int(x) for x in a.batches.split(',')])
    return 0


if __name__ == '__main__':
    sys.exit(main())
