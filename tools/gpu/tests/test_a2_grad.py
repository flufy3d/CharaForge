"""
A2 物理梯度检查：float64 下，对若干输入参数在若干帧上的扰动做中心差分，与 autograd（跨 30～60 帧的 BPTT）比较。

    tools/.venv/bin/python -I tools/gpu/tests/test_a2_grad.py        （也可用 pytest）

做法：取参考序列 s30_full（真实场景输入）的前 t0 帧把物理推进到某个状态（不求导），
再在其后 T = 60 帧的窗口上求 loss = Σ_t Σ_r w[t, r] · out[t, r]（r 为全部物理输出参数，w 固定随机）。
  - 单点：扰动输入参数 p 在窗口内第 k 帧的值，loss 只统计 k + 30 帧之后的输出（梯度至少穿过 30 帧）；
  - 方向导数：对窗口内全部帧、全部物理输入参数取随机方向 v，比较 ⟨∇loss, v⟩ 与差分
    （停在范围端点上的输入单独检查：方向指向范围内、单侧差分）；
  - stabilize：以窗口第 0 帧输入做 stabilization 后推进，对第 0 帧输入求导（梯度同时穿过 stabilization 与 60 帧）。
中心差分步长 h = 1e-6 × 参数范围；同时用 h/4 再算一次，两者不一致说明扰动跨过了不可导点（吸附阈值、±π 折回、夹取），
这种情况单独报告、不计入判定。门槛：相对误差 < 1e-4。比较时关闭 float32 舍入模拟（emulate_f32=False），
否则差分会被 float32 量化淹没；autograd 对舍入本来就是直通。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PHYS = os.path.join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json')
REF = os.path.join(REPO, '.data/gpu/a2/ref')
GATE = 1e-4
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup(seq='s30_full', t0=120, T=60):
    tab = json.load(open(os.path.join(REF, 'model_params.json')))
    meta = json.load(open(os.path.join(REF, f'{seq}.json')))
    xin = np.fromfile(os.path.join(REF, f'{seq}.in.f32'), dtype='<f4').reshape(meta['T'], meta['P'])
    rig = PhysicsRig.from_files(PHYS, os.path.join(REF, 'model_params.json'), device=DEV, dtype=torch.float64, emulate_f32=False)
    x = torch.tensor(xin, dtype=torch.float64, device=DEV).unsqueeze(1)  # [T, 1, P]
    st = rig.initial_state(1)
    with torch.no_grad():
        for t in range(t0):
            st, _ = rig.step(st, x[t])
    rng = torch.tensor(tab['max'], dtype=torch.float64) - torch.tensor(tab['min'], dtype=torch.float64)
    return rig, st, x[t0:t0 + T].clone(), tab, rng


def make_loss(rig, st0, T, tmin=0, stab=False, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(T, 1, len(rig.output_params), device=DEV, dtype=torch.float64, generator=g)
    w[:tmin] = 0
    oi = rig.output_param_index

    def loss(x):
        st = st0
        if stab:
            st, _ = rig.stabilize(st, x[0])
        _, out = rig.rollout(x, state=st)
        return (out[:, :, oi] * w).sum()
    return loss


def check(name, ag, f, h):
    """f(ε) = loss(x + ε·v)；返回 (通过, 说明)。"""
    with torch.no_grad():
        d1 = (f(h) - f(-h)) / (2 * h)
        d2 = (f(h / 4) - f(-h / 4)) / (h / 2)
    d1, d2 = float(d1), float(d2)
    scale = max(abs(ag), abs(d1), 1e-12)
    rel = abs(ag - d1) / scale
    smooth = abs(d1 - d2) / max(abs(d1), abs(d2), 1e-12) < 1e-5
    ok = rel < GATE
    tag = 'OK' if ok else ('跨不可导点' if not smooth else 'FAIL')
    print(f'  {name:42s} autograd {ag:+.6e}  差分 {d1:+.6e}  相对误差 {rel:.2e}  {tag}')
    return ok or not smooth, smooth


def run_all():
    rig, st0, x0, tab, rng = setup()
    T = x0.shape[0]
    ids = tab['ids']
    results = []
    print(f'设备 {DEV}；窗口 T={T}，起点 s30_full 第 120 帧（场景帧 {2600 + 120}）')
    print('单点扰动（loss 只统计扰动帧 +30 之后）：')
    for pname in ['AngleX', 'AngleZ', 'PositionY', 'ParamBreath', 'LeanInput', 'ParamEyeLOpen', 'ParamBrowLY']:
        p = ids.index(pname)
        for k in [0, 15, 29]:
            loss = make_loss(rig, st0, T, tmin=k + 30, seed=k + p)
            x = x0.clone().requires_grad_(True)
            loss(x).backward()
            ag = float(x.grad[k, 0, p])
            assert torch.isfinite(x.grad).all(), '梯度出现非有限值'
            h = 1e-6 * float(rng[p])

            def f(e, p=p, k=k, loss=loss):
                xx = x0.clone()
                xx[k, 0, p] += e
                return loss(xx)
            results.append(check(f'{pname} @k={k}（BPTT {T - k - 1}～{30} 帧）', ag, f, h))
    # 很多输入恰好停在参数范围端点（min / max，例如 JawOpen、Clothes1 = min）。端点处归一化有夹取拐点：
    # autograd 给的是「向范围内」的单侧导数（参数本来就不能越过范围，这是优化器需要的那一侧），
    # 中心差分则是两侧平均。所以方向导数分两种：内部条目用中心差分；端点条目让方向指向范围内，用单侧差分（Richardson 外推）。
    inp = torch.tensor(rig.input_params, device=DEV)
    lo = torch.tensor(tab['min'], dtype=torch.float64, device=DEV)
    hi = torch.tensor(tab['max'], dtype=torch.float64, device=DEV)
    at_lo, at_hi = x0 == lo, x0 == hi
    sel = torch.zeros_like(x0, dtype=torch.bool)
    sel[:, :, inp] = True
    print('方向导数（窗口内全部帧、全部不在端点上的物理输入参数，中心差分）：')
    for seed in range(3):
        loss = make_loss(rig, st0, T, seed=100 + seed)
        g = torch.Generator(device=DEV).manual_seed(seed)
        v = torch.randn(x0.shape, device=DEV, dtype=torch.float64, generator=g) * rng.to(DEV)
        v = torch.where(sel & ~at_lo & ~at_hi, v, 0.0)
        x = x0.clone().requires_grad_(True)
        loss(x).backward()
        ag = float((x.grad * v).sum())
        results.append(check(f'随机方向（内部）#{seed}', ag, lambda e, loss=loss, v=v: loss(x0 + e * v), 1e-6))
    print('方向导数（只动端点上的输入，方向指向范围内，单侧差分）：')
    for seed in range(2):
        loss = make_loss(rig, st0, T, seed=200 + seed)
        g = torch.Generator(device=DEV).manual_seed(10 + seed)
        a = torch.rand(x0.shape, device=DEV, dtype=torch.float64, generator=g) * rng.to(DEV)
        v = torch.where(sel & at_lo, a, torch.where(sel & at_hi, -a, 0.0))
        x = x0.clone().requires_grad_(True)
        loss(x).backward()
        ag = float((x.grad * v).sum())
        with torch.no_grad():
            f0 = loss(x0)
            D = lambda h: (loss(x0 + h * v) - f0) / h
            h = 1e-6
            d = float((4 * D(h / 2) - D(h)) / 3)
            d2 = float((4 * D(h / 8) - D(h / 4)) / 3)
        rel = abs(ag - d) / max(abs(ag), abs(d), 1e-12)
        smooth = abs(d - d2) / max(abs(d), abs(d2), 1e-12) < 1e-4
        ok = rel < GATE
        print(f'  {"端点向内 #" + str(seed):42s} autograd {ag:+.6e}  差分 {d:+.6e}  相对误差 {rel:.2e}  '
              f'{"OK" if ok else ("跨不可导点" if not smooth else "FAIL")}')
        results.append((ok or not smooth, smooth))
    print('stabilize + 60 帧（对第 0 帧输入求导）：')
    for pname in ['AngleX', 'PositionX', 'ParamEyeROpen']:
        p = ids.index(pname)
        loss = make_loss(rig, st0, T, stab=True, seed=7 + p)
        x = x0.clone().requires_grad_(True)
        loss(x).backward()
        ag = float(x.grad[0, 0, p])
        h = 1e-6 * float(rng[p])

        def f(e, p=p, loss=loss):
            xx = x0.clone()
            xx[0, 0, p] += e
            return loss(xx)
        results.append(check(f'stabilize {pname} @k=0', ag, f, h))
    n_ok = sum(1 for ok, _ in results if ok)
    n_ns = sum(1 for _, s in results if not s)
    print(f'{n_ok}/{len(results)} 通过（其中跨不可导点、不计入 {n_ns} 项）')
    return all(ok for ok, _ in results)


def test_grad():
    assert run_all()


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
