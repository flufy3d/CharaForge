"""
A2 物理前向测试（快速版；完整对拍用 tools/gpu/scripts/a2_compare.py）。

    tools/.venv/bin/python -I tools/gpu/tests/test_a2_physics.py      （也可用 pytest）

需要先运行 a2_ref_dump.ts 生成 .data/gpu/a2/ref/。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig, atan2, normalize  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PHYS = os.path.join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json')
REF = os.path.join(REPO, '.data/gpu/a2/ref')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def _load(name):
    meta = json.load(open(os.path.join(REF, f'{name}.json')))
    T, P = meta['T'], meta['P']
    xin = np.fromfile(os.path.join(REF, f'{name}.in.f32'), dtype='<f4').reshape(T, P)
    xout = np.fromfile(os.path.join(REF, f'{name}.out.f32'), dtype='<f4').reshape(T, P)
    return meta, xin, xout


def _range(rig):
    tab = json.load(open(os.path.join(REF, 'model_params.json')))
    r = np.array(tab['max'], dtype=np.float64) - np.array(tab['min'], dtype=np.float64)
    r[r == 0] = 1
    return r


def _run(rig, meta, x):
    """x：[T, B, P]；按参考序列的事件（stab / step + dt）推进。"""
    st = rig.initial_state(x.shape[1])
    outs = []
    with torch.no_grad():
        for t, ev in enumerate(meta['events']):
            if ev['kind'] == 'stab':
                st, o = rig.stabilize(st, x[t])
            else:
                st, o = rig.step(st, x[t], ev['dt'])
            outs.append(o)
    return torch.stack(outs)


def test_safe_primitives():
    z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64, requires_grad=True)
    a = atan2(y, z)
    assert torch.allclose(a, torch.atan2(y.detach(), z.detach()))
    a.sum().backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(y.grad).all() and float(z.grad[0]) == 0.0
    x = torch.tensor([0.0, 3.0], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.0, 4.0], dtype=torch.float64, requires_grad=True)
    nx, ny = normalize(x, w)
    assert torch.isnan(nx[0]) and float(nx[1].detach()) == 0.6 and float(ny[1].detach()) == 0.8   # 长度 0：前向 NaN（同 JS 0/0）
    (nx[1] + ny[1]).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


def test_forward_f64():
    rig = PhysicsRig.from_files(PHYS, os.path.join(REF, 'model_params.json'), device=DEV, dtype=torch.float64)
    rng = _range(rig)
    oi = np.array(rig.output_params)
    for name in ['s24_main', 'stab_walk', 'walk_dt144']:
        meta, xin, xout = _load(name)
        out = _run(rig, meta, torch.tensor(xin, dtype=torch.float64, device=DEV).unsqueeze(1))[:, 0].cpu().numpy()
        err = (np.abs(out[:, oi] - xout[:, oi]) / rng[oi]).max()
        same = np.array_equal(np.delete(out, oi, 1), np.delete(xout, oi, 1).astype(np.float64))
        print(f'  {name}: 最大误差 {err:.2e}，非物理参数原样 {same}')
        assert err < 1e-5 and same


def test_batch_independent():
    """批内样本互不影响：B=3（三条不同序列）与逐条单独运行逐位相同。"""
    rig = PhysicsRig.from_files(PHYS, os.path.join(REF, 'model_params.json'), device=DEV, dtype=torch.float64)
    seqs = [_load(n) for n in ['step', 'sweep', 'walk']]
    T = 200
    meta = {'events': [{'kind': 'step', 'dt': 1 / 60}] * T}
    x = torch.stack([torch.tensor(s[1][:T], dtype=torch.float64, device=DEV) for s in seqs], 1)
    both = _run(rig, meta, x)
    for b in range(3):
        one = _run(rig, meta, x[:, b:b + 1])
        assert torch.equal(both[:, b], one[:, 0]), b


if __name__ == '__main__':
    for f in [test_safe_primitives, test_forward_f64, test_batch_independent]:
        f()
        print(f'{f.__name__}: OK')
