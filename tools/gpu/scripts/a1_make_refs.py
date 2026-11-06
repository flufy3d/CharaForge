"""A1：生成对拍用的参数集并调用 Core dump，写到 .data/gpu/ref/。

    tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py random     # 300 组均匀随机
    tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py special    # 落在 key 上 / 端点 / 略超范围 / 混合
    tools/.venv/bin/python -I tools/gpu/scripts/a1_make_refs.py scene      # scenes/deepseek-chan 实际求值的帧（需先跑 a1_scene_params.ts）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from a1_core import MOC, REPO, run_core  # noqa: E402
from l2d import moc3  # noqa: E402

OUT = REPO / '.data/gpu/ref'


def gen_random(m: moc3.Moc3, n: int, rng) -> np.ndarray:
    p = m['parameters']
    lo, hi = p['minValues'].astype(np.float64), p['maxValues'].astype(np.float64)
    return (lo + (hi - lo) * rng.random((n, len(lo)))).astype(np.float32)


def gen_special(m: moc3.Moc3, n_each: int, rng) -> np.ndarray:
    p = m['parameters']
    lo, hi = p['minValues'].astype(np.float64), p['maxValues'].astype(np.float64)
    P = len(lo)
    keys = [m.parameter_keys(i).astype(np.float64) for i in range(P)]
    # 各参数所有 blend shape / 约束里出现的 key 也算「key」
    span = np.maximum(hi - lo, 1e-6)
    out = []

    def at_key(i):
        k = keys[i]
        return rng.choice(k) if len(k) else lo[i] + (hi[i] - lo[i]) * rng.random()

    for _ in range(n_each):  # 每个参数都恰好落在某个 key 上
        out.append([at_key(i) for i in range(P)])
    for _ in range(n_each):  # 每个参数在端点
        out.append(np.where(rng.random(P) < 0.5, lo, hi))
    for _ in range(n_each):  # 略超出范围（1% 或 0.0005 绝对量，测夹取和 key 范围容差）
        d = np.where(rng.random(P) < 0.5, 0.01 * span, 5e-4)
        out.append(np.where(rng.random(P) < 0.5, lo - d, hi + d))
    for _ in range(n_each):  # 混合：key 附近 ±1e-4、key 上、均匀、超范围
        r = rng.random(P)
        row = []
        for i in range(P):
            if r[i] < 0.25:
                row.append(at_key(i) + rng.choice([-1e-4, 1e-4, -2e-3, 2e-3]))
            elif r[i] < 0.5:
                row.append(at_key(i))
            elif r[i] < 0.85:
                row.append(lo[i] + (hi[i] - lo[i]) * rng.random())
            else:
                row.append(rng.choice([lo[i] - 0.05 * span[i], hi[i] + 0.05 * span[i]]))
        out.append(row)
    # 默认值
    out.append(p['defaultValues'].astype(np.float64))
    return np.asarray(out, np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['random', 'special', 'scene', 'partop'])
    ap.add_argument('-n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args(argv)
    m = moc3.load(MOC)
    rng = np.random.default_rng(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = None
    if a.kind == 'random':
        P = gen_random(m, a.n, rng)
    elif a.kind == 'special':
        P = gen_special(m, max(1, a.n // 4), rng)
    elif a.kind == 'partop':
        # 随机参数 + 随机部件不透明度（测部件不透明度级联）
        P = gen_random(m, a.n, rng)
        parts = np.where(rng.random((a.n, m.counts['parts'])) < 0.7, 1.0, rng.random((a.n, m.counts['parts']))).astype(np.float32)
    else:
        src = OUT / 'scene_frames.json'
        if not src.exists():
            raise SystemExit(f'缺少 {src}：先运行 node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types '
                             'tools/gpu/scripts/a1_scene_params.ts')
        meta = json.loads(src.read_text())
        n, NP, NQ = meta['n'], meta['P'], meta['Q']
        raw = np.fromfile(OUT / 'scene_frames.bin', '<f4')
        if raw.size != n * (NP + NQ) or NP != m.counts['parameters'] or NQ != m.counts['parts']:
            raise SystemExit('scene_frames 尺寸不符')
        P, parts = raw[:n * NP].reshape(n, NP), raw[n * NP:].reshape(n, NQ)
    r = run_core(P, parts=parts, out=OUT / a.kind)
    print(f'{a.kind}: {P.shape[0]} 组，Core {r["meta"]["msPerUpdate"]:.2f} ms/次 → {OUT / a.kind}.bin')


if __name__ == '__main__':
    main()
