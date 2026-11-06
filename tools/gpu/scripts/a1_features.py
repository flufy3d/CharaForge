"""A1：本模型没用到的特性，用「只在内存里改 moc 字节」的方式造出来，和 Core 对拍（验证实现的通用性）。

    tools/.venv/bin/python -I tools/gpu/scripts/a1_features.py

检查项：
  quad0     根 warp 改成 isQuadSource = 0（网格内三角形插值），网格改成非仿射；
  reflect   旋转变形器 keyform 打开 isReflectX / isReflectY；
  colors    变形器（warp、rotation）keyform 的乘算色 / 屏幕色改成非默认值（颜色沿变形器链级联）；
  warpout   根 warp 改成非仿射网格（网格外的带状区与仿射外插）。
每项用 12 组随机参数，比较顶点、不透明度、颜色、render order。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1_compare import compare, evaluate  # noqa: E402
from a1_core import MOC, run_core  # noqa: E402
from a1_make_refs import gen_random  # noqa: E402
from a1_probe import apply_patches  # noqa: E402
from l2d import moc3  # noqa: E402
from l2d.deform import L2D  # noqa: E402


def f32(*v):
    return struct.pack(f'<{len(v)}f', *v)


def i32(*v):
    return struct.pack(f'<{len(v)}i', *v)


def warp_grid_patches(m, wi, rng):
    g = m.groups
    W, wk = g['warpDeformers'], g['warpDeformerKeyforms']
    R, C = int(W['rows'][wi]), int(W['columns'][wi])
    out = []
    b, n = int(W['keyformSourcesBeginIndices'][wi]), int(W['keyformSourcesCounts'][wi])
    for k in range(b, b + n):
        uu, vv = np.meshgrid(np.linspace(0, 1, C + 1), np.linspace(0, 1, R + 1))
        grid = np.stack([uu * 1.6 - 0.8 + 0.3 * uu * vv + 0.02 * rng.standard_normal(uu.shape),
                         vv * 2.2 - 1.0 + 0.2 * uu ** 2 + 0.02 * rng.standard_normal(uu.shape)], -1).astype(np.float32)
        pb = int(wk['keyformPositionSourcesBeginIndices'][k])
        out.append((m.offsets['keyformPositions.xys'] + 4 * pb, grid.tobytes()))
    return out


def feature_patches(m: moc3.Moc3, name: str, rng):
    g = m.groups
    W, R = g['warpDeformers'], g['rotationDeformers']
    if name == 'warpout':
        return warp_grid_patches(m, 0, rng)
    if name == 'quad0':
        return warp_grid_patches(m, 0, rng) + [(m.offsets['warpDeformers.isQuadSource'] + 4 * w, i32(0)) for w in range(0, len(W['rows']), 3)]
    if name == 'reflect':
        rk = g['rotationDeformerKeyforms']
        out = []
        for r in range(0, len(R['baseAngles']), 2):
            b, n = int(R['keyformSourcesBeginIndices'][r]), int(R['keyformSourcesCounts'][r])
            fx, fy = (1, 0) if r % 4 == 0 else (0, 1)
            for k in range(b, b + n):
                out += [(m.offsets['rotationDeformerKeyforms.isReflectX'] + 4 * k, i32(fx)),
                        (m.offsets['rotationDeformerKeyforms.isReflectY'] + 4 * k, i32(fy))]
        return out
    if name == 'colors':
        out = []
        for grp, og in (('warpDeformerKeyforms', 'warpDeformers'), ('rotationDeformerKeyforms', 'rotationDeformers')):
            kf = g[grp]
            # 只改普通 keyform（blend shape 的 keyform 存的是增量，模型里颜色增量都是 0）
            normal = np.concatenate([np.arange(b, b + n) for b, n in zip(g[og]['keyformSourcesBeginIndices'], g[og]['keyformSourcesCounts'])])
            for k in normal[::5]:
                k = int(k)
                ci = int(kf['keyformMultiplyColorSourcesIndices'][k])
                si = int(kf['keyformScreenColorSourcesIndices'][k])
                mc = rng.uniform(0.3, 1.0, 3)
                sc = rng.uniform(0.0, 0.4, 3)
                for ch, v in zip('rgb', mc):
                    out.append((m.offsets[f'keyformColorsMultiply.{ch}'] + 4 * ci, f32(v)))
                for ch, v in zip('rgb', sc):
                    out.append((m.offsets[f'keyformColorsScreen.{ch}'] + 4 * si, f32(v)))
        return out
    raise ValueError(name)


def main():
    buf = Path(MOC).read_bytes()
    m = moc3.parse(buf)
    rng = np.random.default_rng(7)
    P = gen_random(m, 12, rng)
    ok = True
    for name in ('warpout', 'quad0', 'reflect', 'colors'):
        patches = feature_patches(m, name, rng)
        ref = run_core(P, patches=patches)
        m2 = moc3.parse(apply_patches(buf, patches))
        model = L2D(m2, dtype=torch.float64)
        r = compare(model, ref, evaluate(model, ref), worst=2)
        good = (r['vertex_px']['max'] < 0.1 and r['opacity']['max'] < 1e-4 and r['multiply']['max'] < 1e-4 and r['screen']['max'] < 1e-4
                and r['renderOrder_mismatch_samples'] == 0 and r['visible_mismatch'] == 0)
        ok &= good
        print(f'{name:8s} {"OK " if good else "BAD"} 顶点 max {r["vertex_px"]["max"]:.4f}px  不透明度 {r["opacity"]["max"]:.2e}  '
              f'乘算 {r["multiply"]["max"]:.2e}  屏幕 {r["screen"]["max"]:.2e}  render order 不一致 {r["renderOrder_mismatch_samples"]}  '
              f'可见性不一致 {r["visible_mismatch"]}  最差 {r["worst_drawables"][0]["id"]}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
