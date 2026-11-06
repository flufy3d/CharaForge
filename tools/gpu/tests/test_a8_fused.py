"""
A8：fused_rollout（Triton 融合核）与原路径（eager 的 rollout / step）的一致性、梯度、确定性。

    tools/.venv/bin/python -I tools/gpu/tests/test_a8_fused.py [--skip-a2]      （也可用 pytest）

检查（float64 rig，输入 s30_full 第 100 帧起加扰动，B = 3）：
  1. 前向：输出与全部状态（粒子位置 / 速度、lastGravity、rig 输出、InputCaches、remain）与 eager 逐位相同；
  2. 梯度：对输入参数、以及对初始状态的每个分量（loss 里也含最终状态），与 eager autograd 的相对误差 ≤ 1e-9；
  3. 确定性：同一输入跑两次，前向与梯度逐位相同；
  4. 拟合器口径（预热 90 帧 no_grad → 24 帧一窗、窗间 detach，全部前向完再一起反向），以及 200 帧不截断；
  5. 其他调度：dt = 1/50、1/144（每帧 0～2 个子步）、全新状态（in_cache 为 None）、resolved=False、emulate_f32=False；
  6. test_a2_grad 的全部 29 项（autograd 对中心差分）改走 fused_rollout 再跑一遍（--skip-a2 跳过）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig, PhysicsState  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PHYS = os.path.join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json')
REF_DIR = os.path.join(REPO, '.data/gpu/a2/ref')
DEV = 'cuda'
FIELDS = ('px', 'py', 'vx', 'vy', 'gx', 'gy')


def make_rig(**kw):
    return PhysicsRig.from_files(PHYS, os.path.join(REF_DIR, 'model_params.json'), device=DEV, dtype=torch.float64, **kw)


def inputs(rig, T, B=3, seed=0):
    meta = json.load(open(os.path.join(REF_DIR, 's30_full.json')))
    xin = np.fromfile(os.path.join(REF_DIR, 's30_full.in.f32'), dtype='<f4').reshape(meta['T'], meta['P'])
    xin = np.concatenate([xin] * ((100 + T) // len(xin) + 1))
    base = torch.tensor(xin[100:100 + T + 1], dtype=torch.float64, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = base.unsqueeze(1).repeat(1, B, 1) + 0.01 * torch.randn(T + 1, B, base.shape[1], device=DEV, dtype=torch.float64, generator=g)
    w = torch.randn(T, B, len(rig.output_params), device=DEV, dtype=torch.float64, generator=g)
    return x, w


def leaf_state(st: PhysicsState) -> PhysicsState:
    c = lambda t: t.detach().clone().requires_grad_(True)
    return PhysicsState(*(tuple(c(t) for t in getattr(st, f)) for f in FIELDS), c(st.cur), c(st.prev),
                        None if st.in_cache is None else c(st.in_cache), st.remain, st.resolved)


def state_tensors(st):
    return [t for f in FIELDS for t in getattr(st, f)] + [st.cur, st.prev] + ([st.in_cache] if st.in_cache is not None else [])


def run(rig, fused, x, w, st0, dt=1 / 60, window=None, detach=True, state_grad=True):
    """返回 (输出, 最终状态, 对 x 的梯度, 对初始状态各分量的梯度)。"""
    roll = (lambda s, st: rig.fused_rollout(s, st, dt)) if fused else (lambda s, st: rig.rollout(s, state=st, dt=dt))
    xx = x.clone().requires_grad_(True)
    s0 = leaf_state(st0) if state_grad else st0
    st = s0
    outs = []
    W = window or x.shape[0]
    for a in range(0, x.shape[0], W):
        st, o = roll(xx[a:a + W], st)
        outs.append(o)
        if detach and window:
            st = st.detach()
    out = torch.cat(outs)
    loss = (out[:, :, rig.output_param_index] * w[:out.shape[0]]).sum()
    if not (detach and window):   # 最终状态也进 loss，检查对状态的伴随
        for k, t in enumerate(state_tensors(st)):
            loss = loss + (t * (0.1 + 0.01 * k)).sum()
    loss.backward()
    gs = [t.grad for t in state_tensors(s0)] if state_grad else []
    return out.detach(), st, xx.grad, gs


def rel(a, b):
    if a is None or b is None:
        return 0.0 if (a is None or float(a.abs().max()) == 0) and (b is None or float(b.abs().max()) == 0) else float('inf')
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def check(tag, rig, x, w, st0, **kw):
    o0, s0, g0, gs0 = run(rig, False, x, w, st0, **kw)
    o1, s1, g1, gs1 = run(rig, True, x, w, st0, **kw)
    o2, _, g2, gs2 = run(rig, True, x, w, st0, **kw)
    fwd = float((o0 - o1).abs().max())
    stt = max(float((a - b).abs().max()) for a, b in zip(state_tensors(s0), state_tensors(s1)))
    gx = rel(g1, g0)
    gst = max([rel(a, b) for a, b in zip(gs1, gs0)], default=0.0)
    det = bool(torch.equal(o1, o2) and torch.equal(g1, g2) and all(
        (a is None and b is None) or torch.equal(a, b) for a, b in zip(gs1, gs2)))
    ok = fwd == 0.0 and stt == 0.0 and s0.remain == s1.remain and s0.resolved == s1.resolved and gx <= 1e-9 and gst <= 1e-9 and det
    print(f'  {tag:40s} 前向差 {fwd:.1e}  状态差 {stt:.1e}  梯度相对差：输入 {gx:.1e} / 状态 {gst:.1e}  两次逐位相同 {det}  '
          f'{"OK" if ok else "FAIL"}', flush=True)
    return ok


def warm(rig, x, n=90, dt=1 / 60):
    with torch.no_grad():
        st, _ = rig.rollout(x[:1].expand(n, -1, -1).contiguous(), dt=dt)
    return st


def run_all(skip_a2=False):
    res = []
    rig = make_rig()
    x, w = inputs(rig, 200)
    st = warm(rig, x)
    print(f'fused_rollout vs eager rollout（B={x.shape[1]}；前向要求逐位相同，梯度相对误差 ≤ 1e-9）')
    res.append(check('拟合器口径：24 帧一窗、窗间 detach，T=72', rig, x[1:73], w, st, window=24, state_grad=False))
    res.append(check('窗口 7（末窗不满），T=30', rig, x[1:31], w, st, window=7, state_grad=False))
    res.append(check('200 帧不截断，含对初始状态的梯度', rig, x[1:201], w, st))
    res.append(check('dt = 1/50，含状态梯度', rig, x[1:49], w, warm(rig, x, dt=1 / 50), dt=1 / 50))
    res.append(check('dt = 1/144，含状态梯度', rig, x[1:49], w, warm(rig, x, dt=1 / 144), dt=1 / 144))
    res.append(check('全新状态（in_cache 为 None），dt = 1/144', rig, x[1:49], w, rig.initial_state(x.shape[1]), dt=1 / 144))
    res.append(check('resolved=False，dt = 1/144', rig, x[1:25], w, rig.initial_state(x.shape[1], resolved=False), dt=1 / 144))
    rig2 = make_rig(emulate_f32=False)
    res.append(check('emulate_f32=False，含状态梯度', rig2, x[1:49], w, warm(rig2, x)))
    # 不求导时的前向（不写磁带）
    with torch.no_grad():
        a = rig.rollout(x[1:61], state=st)
        b = rig.fused_rollout(x[1:61], st)
    d = float((a[1] - b[1]).abs().max())
    ok = d == 0.0 and all(torch.equal(p, q) for p, q in zip(state_tensors(a[0]), state_tensors(b[0])))
    print(f'  {"no_grad 前向（不记磁带）":40s} 前向差 {d:.1e}  {"OK" if ok else "FAIL"}')
    res.append(ok)
    # float32 输入输出（f32io）
    x32 = x[1:25].float()
    with torch.no_grad():
        a = rig.rollout(x32, state=st)[1]
        b = rig.fused_rollout(x32, st)[1]
    ok = a.dtype == b.dtype == torch.float32 and torch.equal(a, b)
    print(f'  {"float32 输入输出":40s} 前向逐位相同 {ok}  {"OK" if ok else "FAIL"}')
    res.append(ok)
    # auto_fused：rollout() 自动走 fused_rollout
    rig.auto_fused = True
    try:
        with torch.no_grad():
            a = rig.rollout(x[1:25], state=st)[1]
    finally:
        rig.auto_fused = False
    with torch.no_grad():
        b = rig.fused_rollout(x[1:25], st)[1]
    ok = torch.equal(a, b)
    print(f'  {"auto_fused：rollout 走融合核、结果相同":40s} {"OK" if ok else "FAIL"}')
    res.append(ok)
    print(f'{sum(res)}/{len(res)} 通过')
    if not skip_a2:
        print('\ntest_a2_grad（autograd 对中心差分）改走 fused_rollout：')
        import test_a2_grad as A
        orig = PhysicsRig.rollout
        PhysicsRig.rollout = lambda self, s, state=None, dt=1 / 60, stabilize_first=False, checkpoint=False: \
            self.fused_rollout(s, state if not stabilize_first else self.stabilize(state, s[0])[0], dt)
        try:
            res.append(A.run_all())
        finally:
            PhysicsRig.rollout = orig
    return all(res)


def test_fused():
    assert run_all()


if __name__ == '__main__':
    sys.exit(0 if run_all(skip_a2='--skip-a2' in sys.argv) else 1)
