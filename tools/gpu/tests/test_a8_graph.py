"""
A8：graphed_rollout（CUDA graph 快速路径）与原路径 rollout 的一致性、确定性。

    tools/.venv/bin/python -I tools/gpu/tests/test_a8_graph.py [--compile]      （也可用 pytest，缺省不编译）

检查（float64 rig，输入 s30_full 加扰动，B = 3）：
  1. 拟合器口径（预热 90 帧 no_grad → 24 帧一窗、窗间 detach，全部窗口前向完再一起反向）：
     前向逐位相同；对输入的梯度相对误差 ≤ 1e-9（实际应逐位相同）；
  2. 不截断（状态带梯度穿过窗口）：同上，梯度也穿过上一窗的图；
  3. 确定性：同一输入跑两次，前向与梯度逐位相同（graphed 与原路径各自都要求）；
  4. 其他调度：dt = 1/50、1/144（每帧 0～2 个子步）、全新状态（in_cache 为 None、resolved=False）；
  5. 被覆盖的图实例再反向会报错（不会悄悄给出错误梯度）；auto_graph 开关。
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
DEV = 'cuda'
COMPILE = '--compile' in sys.argv


def make(T=72, B=3, seed=0):
    rig = PhysicsRig.from_files(PHYS, os.path.join(REF, 'model_params.json'), device=DEV, dtype=torch.float64)
    if COMPILE:
        rig.compile()
    meta = json.load(open(os.path.join(REF, 's30_full.json')))
    xin = np.fromfile(os.path.join(REF, 's30_full.in.f32'), dtype='<f4').reshape(meta['T'], meta['P'])
    base = torch.tensor(xin[100:100 + T + 1], dtype=torch.float64, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = base.unsqueeze(1).repeat(1, B, 1) + 0.01 * torch.randn(T + 1, B, base.shape[1], device=DEV, dtype=torch.float64, generator=g)
    w = torch.randn(T, B, len(rig.output_params), device=DEV, dtype=torch.float64, generator=g)
    return rig, x, w


def fit_like(rig, x, w, graphed, window=24, detach=True, dt=1 / 60, warm=90):
    """拟合器 physics_batch 的口径。返回 (输出, 梯度, 最终状态)。"""
    roll = rig.graphed_rollout if graphed else (lambda s, st, dt: rig.rollout(s, state=st, dt=dt))
    B = x.shape[1]
    st = rig.initial_state(B)
    if warm:
        with torch.no_grad():
            st, _ = roll(x[:1].expand(warm, -1, -1).contiguous(), st, dt)
        st = st.detach()
    xx = x[1:].clone().requires_grad_(True)
    outs = []
    for a in range(0, xx.shape[0], window):
        st, o = roll(xx[a:a + window], st, dt)
        outs.append(o)
        if detach:
            st = st.detach()
    out = torch.cat(outs)
    loss = (out[:, :, rig.output_param_index] * w).sum() + (st.cur * 0.1).sum() * (0 if detach else 1)
    loss.backward()
    return out.detach(), xx.grad, st


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def check_pair(tag, rig, x, w, **kw):
    o0, g0, s0 = fit_like(rig, x, w, False, **kw)
    o1, g1, s1 = fit_like(rig, x, w, True, **kw)
    o2, g2, s2 = fit_like(rig, x, w, True, **kw)
    o3, g3, _ = fit_like(rig, x, w, False, **kw)
    fwd = float((o0 - o1).abs().max())
    stt = max(float((a - b).abs().max()) for a, b in zip([*s0.px, s0.cur, s0.in_cache], [*s1.px, s1.cur, s1.in_cache]))
    gr = rel(g1, g0)
    det_g = bool(torch.equal(o1, o2) and torch.equal(g1, g2))
    det_o = bool(torch.equal(o0, o3) and torch.equal(g0, g3))
    ok = fwd <= 1e-12 and stt <= 1e-12 and gr <= 1e-9 and det_g and det_o and s0.remain == s1.remain
    print(f'  {tag:34s} 前向最大差 {fwd:.1e}  状态差 {stt:.1e}  梯度相对差 {gr:.1e}  '
          f'graphed 两次逐位相同 {det_g}  原路径两次逐位相同 {det_o}  {"OK" if ok else "FAIL"}', flush=True)
    # 每个检查换一套调度 / 窗口，旧的图实例不再用：释放（未编译时每张图有上万个节点，主机内存占用大）
    rig.graph_clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def run_all():
    rig, x, w = make()
    res = []
    print(f'graphed_rollout vs rollout（{"compile" if COMPILE else "eager"}，B={x.shape[1]}，T={x.shape[0] - 1}）')
    res.append(check_pair('24 帧一窗、窗间 detach（拟合器）', rig, x, w))
    res.append(check_pair('不截断（梯度穿过各窗）', rig, x, w, detach=False))
    res.append(check_pair('窗口 7（末窗不满）', rig, x, w, window=7))
    res.append(check_pair('dt = 1/50', rig, x, w, dt=1 / 50))
    res.append(check_pair('dt = 1/144', rig, x, w, dt=1 / 144))
    res.append(check_pair('全新状态（无预热，dt = 1/144）', rig, x, w, dt=1 / 144, warm=0))
    # resolved=False + 无预热：第一帧没有子步时 interpolate 不写参数
    st = rig.initial_state(x.shape[1], resolved=False)
    with torch.no_grad():
        a = rig.rollout(x[:10], state=st, dt=1 / 144)
        b = rig.graphed_rollout(x[:10], st, 1 / 144)
    d = float((a[1] - b[1]).abs().max())
    ok = d == 0.0 and a[0].resolved == b[0].resolved and a[0].remain == b[0].remain
    print(f'  {"resolved=False，dt = 1/144":34s} 前向最大差 {d:.1e}  {"OK" if ok else "FAIL"}')
    res.append(ok)
    # 被覆盖后再反向：同一实例在反向前被复用的情况由实例池避免；这里强制制造（retain_graph 后复用再反向）
    rig.graph_max_instances = 1
    rig.graph_clear()
    xx = x[1:25].clone().requires_grad_(True)
    st0 = rig.initial_state(x.shape[1])
    _, o = rig.graphed_rollout(xx, st0)
    o.sum().backward(retain_graph=True)
    _, o2 = rig.graphed_rollout(xx, st0)   # 复用同一实例（上一次的反向已跑完）
    try:
        o.sum().backward()
        ok = False
    except RuntimeError as e:
        ok = '覆盖' in str(e)
    o2.sum().backward()
    print(f'  {"实例被复用后再反向会报错":34s} {"OK" if ok else "FAIL"}')
    res.append(ok)
    # 实例池满、没有空闲实例时退回普通 rollout（结果相同）
    rig.graph_clear()
    xx = x[1:49].clone().requires_grad_(True)
    st = st0
    outs = []
    for a in range(0, 48, 24):
        st, o = rig.graphed_rollout(xx[a:a + 24], st)
        outs.append(o)
        st = st.detach()
    fb = rig.graph_stats['fallbacks']
    torch.cat(outs).sum().backward()
    g_fb = xx.grad.clone()
    rig.graph_max_instances = 64
    xx.grad = None
    o_ref = torch.cat([rig.rollout(xx[0:24], state=st0)[1], rig.rollout(xx[24:48], state=rig.rollout(xx[0:24].detach(), state=st0)[0])[1]])
    o_ref.sum().backward()
    ok = fb >= 1 and rel(g_fb, xx.grad) <= 1e-9
    print(f'  {"实例池满时退回普通路径":34s} fallbacks={fb} 梯度相对差 {rel(g_fb, xx.grad):.1e} {"OK" if ok else "FAIL"}')
    res.append(ok)
    # auto_graph
    rig.auto_graph = True
    with torch.no_grad():
        r0 = rig.graph_stats['replays']
        _, oa = rig.rollout(x[:24])
    rig.auto_graph = False
    with torch.no_grad():
        _, ob = rig.rollout(x[:24])
    ok = rig.graph_stats['replays'] == r0 + 1 and torch.equal(oa, ob)
    print(f'  {"auto_graph：rollout 走图、结果相同":34s} {"OK" if ok else "FAIL"}')
    res.append(ok)
    print(f'{sum(res)}/{len(res)} 通过；图统计 {rig.graph_stats}')
    return all(res)


def test_graphed():
    assert run_all()


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
