"""A1 对拍：用 tools/gpu/l2d/deform.py 求值参考参数集，与 Core 的 dump（a1_core_dump.mjs）逐项比较。

用法：
    tools/.venv/bin/python -I tools/gpu/scripts/a1_compare.py .data/gpu/ref/random [.data/gpu/ref/scene ...] \
        [--dtype float64] [--batch 64] [--worst 10] [--json out.json]
门槛：顶点最大误差 < 0.1px，不透明度 < 1e-4，render order 完全一致，乘算 / 屏幕色 < 1e-4。
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
import torch  # noqa: E402

from a1_core import MOC, load_ref  # noqa: E402
from l2d import moc3  # noqa: E402
from l2d.deform import L2D  # noqa: E402


def evaluate(model: L2D, ref: dict, batch: int = 64):
    """对 ref 的全部样本求值，返回 numpy 结果（与 ref 同形状）。"""
    P = torch.as_tensor(ref['params'])
    parts = torch.as_tensor(ref['parts']) if 'parts' in ref else None
    outs = {k: [] for k in ('verts', 'opacity', 'drawOrder', 'renderOrder', 'multiply', 'screen', 'visible', 'vertexUpdated')}
    with torch.no_grad():
        for i in range(0, P.shape[0], batch):
            o = model(P[i:i + batch], None if parts is None else parts[i:i + batch])
            outs['verts'].append(o.verts.double().cpu().numpy())
            outs['opacity'].append(o.opacity.double().cpu().numpy())
            outs['drawOrder'].append(o.draw_order.cpu().numpy())
            outs['renderOrder'].append(o.render_order.cpu().numpy())
            outs['multiply'].append(o.multiply.double().cpu().numpy())
            outs['screen'].append(o.screen.double().cpu().numpy())
            outs['visible'].append(o.visible.cpu().numpy())
            outs['vertexUpdated'].append(o.vertex_updated.cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def compare(model: L2D, ref: dict, got: dict, worst: int = 10) -> dict:
    vo = model.vertex_offsets
    D = model.D
    ids = model.drawable_ids
    err = np.linalg.norm(got['verts'] - ref['verts'], axis=-1)        # [N, V]（欧氏距离，像素）
    # Core 对不可见的 drawable 不更新顶点（新建模型时为 0），只比较 Core 可见的 drawable
    vmask = np.repeat(ref['visible'], np.diff(vo), axis=1)
    err = np.where(vmask, err, 0.0)
    per_d = np.stack([err[:, vo[d]:vo[d + 1]].max(1) if vo[d + 1] > vo[d] else np.zeros(err.shape[0]) for d in range(D)], 1)  # [N, D]
    ev = err[vmask]
    op_err = np.abs(got['opacity'] - ref['opacity'])
    mul_err = np.abs(got['multiply'] - ref['multiply'][..., :3]).max(-1)
    scr_err = np.abs(got['screen'] - ref['screen'][..., :3]).max(-1)
    ro_bad = got['renderOrder'] != ref['renderOrder']
    do_bad = got['drawOrder'] != ref['drawOrder']
    vis_bad = got['visible'] != ref['visible']
    vup_bad = got['vertexUpdated'] != ((ref['dynFlags'] & 32) != 0)   # bit5 = VertexPositionsDidChange
    r = {
        'samples': int(err.shape[0]),
        'visible_drawables': int(ref['visible'].sum()), 'compared_vertices': int(ev.size),
        'vertex_px': {'max': float(ev.max()), 'p99': float(np.percentile(ev, 99)), 'p999': float(np.percentile(ev, 99.9)),
                      'mean': float(ev.mean())},
        'opacity': {'max': float(op_err.max()), 'mean': float(op_err.mean())},
        'multiply': {'max': float(mul_err.max())},
        'screen': {'max': float(scr_err.max())},
        'renderOrder_mismatch_samples': int(ro_bad.any(1).sum()),
        'drawOrder_mismatch': int(do_bad.sum()),
        'visible_mismatch': int(vis_bad.sum()),
        'vertexUpdated_mismatch': int(vup_bad.sum()),
    }
    dmax = per_d.max(0)
    order = np.argsort(-dmax)[:worst]
    r['worst_drawables'] = [{'d': int(d), 'id': ids[d], 'max_px': float(dmax[d]), 'sample': int(per_d[:, d].argmax()),
                             'opacity_max': float(op_err[:, d].max())} for d in order]
    od = np.argsort(-op_err.max(0))[:worst]
    r['worst_opacity'] = [{'d': int(d), 'id': ids[d], 'max': float(op_err[:, d].max()), 'sample': int(op_err[:, d].argmax())}
                          for d in od if op_err[:, d].max() > 0]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('refs', nargs='+')
    ap.add_argument('--dtype', default='float64', choices=['float32', 'float64'])
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--worst', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    dt = getattr(torch, a.dtype)
    model = L2D(moc3.load(MOC), device=a.device, dtype=dt)
    report = {}
    ok = True
    for rp in a.refs:
        ref = load_ref(rp)
        got = evaluate(model, ref, a.batch)
        r = compare(model, ref, got, a.worst)
        r['pass'] = bool(r['vertex_px']['max'] < 0.1 and r['opacity']['max'] < 1e-4 and r['renderOrder_mismatch_samples'] == 0
                         and r['multiply']['max'] < 1e-4 and r['screen']['max'] < 1e-4 and r['drawOrder_mismatch'] == 0
                         and r['visible_mismatch'] == 0 and r['vertexUpdated_mismatch'] == 0)
        ok &= r['pass']
        report[rp] = r
        print(f'== {rp}（{a.dtype}）')
        print(json.dumps(r, ensure_ascii=False, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
