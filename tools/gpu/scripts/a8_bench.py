"""
A8 物理提速：耗时 / 显存测量与 profile。

    tools/.venv/bin/python -I tools/gpu/scripts/a8_bench.py bench [--B 1,4,16] [--T 24,200] [--mode compile,graphed] [--reps 7]
    tools/.venv/bin/python -I tools/gpu/scripts/a8_bench.py profile [--B 1] [--T 24] [--mode compile]

口径同拟合器（fit/traj.py 的 physics / physics_batch）：状态先在 no_grad 下预热 90 帧并 detach，
然后 T 帧按 24 帧一窗（--window）截断 BPTT，前向 + 反向。输入取参考序列 s30_full 的参数（循环补齐），
每个样本加不同的小扰动；loss = 随机权重 × 物理输出参数之和，对输入求梯度。
mode：
  eager    逐帧 rig.step（未编译）
  compile  rig.compile() 后的 rig.rollout（拟合器现在的用法）
  graphed  rig.compile() 后的 rig.graphed_rollout（A8，CUDA graph，前向 + 反向）
  fused    rig.fused_rollout（A8，Triton 融合核：整段前向一个核、反向一个核；不需要 compile）
计时为 reps 次的中位数（另报最小 / 最大），首次调用（编译 / 捕获）不计入；显存为该次调用的峰值增量。
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if '--l2d' in sys.argv:   # 指向旧版 l2d 包的上一级目录，测改动前
    sys.path.insert(0, os.path.expanduser(sys.argv[sys.argv.index('--l2d') + 1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PHYS = os.path.join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json')
REF = os.path.join(REPO, '.data/gpu/a2/ref')
WARMUP = 90


def gpu_util():
    exe = 'nvidia-smi'
    for p in ('/usr/lib/wsl/lib/nvidia-smi', '/usr/bin/nvidia-smi'):
        if os.path.exists(p):
            exe = p
            break
    try:
        r = subprocess.run([exe, '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5)
        u, m = r.stdout.strip().split(',')
        return f'{int(u)}% / {int(m)} MiB'
    except Exception:
        return '?'


def make_rig(dtype=torch.float64):
    return PhysicsRig.from_files(PHYS, os.path.join(REF, 'model_params.json'), device='cuda', dtype=dtype)


def make_inputs(rig, T, B, seed=0):
    meta = json.load(open(os.path.join(REF, 's30_full.json')))
    xin = np.fromfile(os.path.join(REF, 's30_full.in.f32'), dtype='<f4').reshape(meta['T'], meta['P'])
    xin = np.concatenate([xin] * ((T + 1) // len(xin) + 1))[:T + 1]
    base = torch.tensor(xin, dtype=torch.float64, device='cuda')
    g = torch.Generator(device='cuda').manual_seed(seed)
    x0 = base.unsqueeze(1).repeat(1, B, 1) + 0.01 * torch.randn(T + 1, B, base.shape[1], device='cuda', dtype=torch.float64, generator=g)
    w = torch.randn(T, B, len(rig.output_params), device='cuda', dtype=torch.float64, generator=g)
    return x0, w


def warm_state(rig, x0, mode):
    """拟合器口径的预热：no_grad 推进 WARMUP 帧（输入取第 0 帧），detach。"""
    B = x0.shape[1]
    seq = x0[:1].expand(WARMUP, -1, -1).contiguous()
    with torch.no_grad():
        if mode in ('graphed', 'fused'):
            st, _ = getattr(rig, mode + '_rollout')(seq, rig.initial_state(B))
        else:
            st, _ = rig.rollout(seq)
    return st.detach()


def fwd_bwd(rig, st0, x, w, mode, window):
    """T 帧前向（每 window 帧 detach）+ 反向。返回 (前向秒, 反向秒, 输出, 梯度)。"""
    outp = rig.output_param_index
    T = x.shape[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    st = st0
    outs = []
    for a in range(0, T, window):
        seg = x[a:a + window]
        if mode in ('graphed', 'fused'):
            st, o = getattr(rig, mode + '_rollout')(seg, st)
        elif mode == 'eager':
            os_ = []
            for t in range(seg.shape[0]):
                st, oo = rig.step(st, seg[t])
                os_.append(oo)
            o = torch.stack(os_)
        else:
            st, o = rig.rollout(seg, state=st)
        outs.append(o)
        st = st.detach()
    out = torch.cat(outs, 0)
    loss = (out[:, :, outp] * w).sum()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, out.detach(), x.grad


def cmd_bench(args):
    rig = make_rig()
    modes = args.mode.split(',')
    if any(m in ('compile', 'graphed') for m in modes):
        rig.compile()
    rows = []
    for T in [int(t) for t in args.T.split(',')]:
        for B in [int(b) for b in args.B.split(',')]:
            x0, w = make_inputs(rig, T, B)
            for mode in modes:
                st0 = warm_state(rig, x0, mode)
                times, peak = [], 0
                util0 = gpu_util()
                for rep in range(args.reps + 1):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    m0 = torch.cuda.memory_allocated()
                    x = x0[1:].clone().requires_grad_(True)
                    f, b, out, g = fwd_bwd(rig, st0, x, w, mode, args.window)
                    pk = torch.cuda.max_memory_allocated() - m0
                    if rep:
                        times.append((f, b))
                        peak = max(peak, pk)
                    del x, out, g
                tf = np.array([t[0] for t in times]); tb = np.array([t[1] for t in times]); tt = tf + tb
                util1 = gpu_util()
                r = dict(T=T, B=B, mode=mode, fwd=float(np.median(tf)), bwd=float(np.median(tb)), tot=float(np.median(tt)),
                         tot_min=float(tt.min()), tot_max=float(tt.max()), peak_mib=peak / 2**20, gpu=f'{util0} → {util1}',
                         reserved_mib=torch.cuda.memory_reserved() / 2**20)
                rows.append(r)
                print(f'T={T:3d} B={B:2d} {mode:8s}: 前向 {r["fwd"]*1e3:7.1f} ms  反向 {r["bwd"]*1e3:7.1f} ms  合计 {r["tot"]*1e3:7.1f} ms '
                      f'[{r["tot_min"]*1e3:.1f}～{r["tot_max"]*1e3:.1f}]  峰值 {r["peak_mib"]:.0f} MiB  reserved {r["reserved_mib"]:.0f} MiB  GPU {r["gpu"]}',
                      flush=True)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)


def cmd_profile(args):
    from torch.profiler import profile, ProfilerActivity
    rig = make_rig()
    mode = args.mode.split(',')[0]
    if mode in ('compile', 'graphed'):
        rig.compile()
    T, B = int(args.T.split(',')[0]), int(args.B.split(',')[0])
    x0, w = make_inputs(rig, T, B)
    st0 = warm_state(rig, x0, mode)
    for _ in range(3):
        x = x0[1:].clone().requires_grad_(True)
        fwd_bwd(rig, st0, x, w, mode, args.window)
    x = x0[1:].clone().requires_grad_(True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        f, b, _, _ = fwd_bwd(rig, st0, x, w, mode, args.window)
    ev = prof.events()
    kern = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]
    gpu_us = sum(e.device_time for e in kern) if kern and hasattr(kern[0], 'device_time') else sum(e.cuda_time for e in kern)
    cpu_names = {}
    for e in ev:
        if e.device_type == torch.autograd.DeviceType.CPU:
            cpu_names[e.name] = cpu_names.get(e.name, 0) + 1
    pick = lambda *ks: {k: v for k, v in cpu_names.items() if any(s in k for s in ks)}
    print(f'mode={mode} T={T} B={B}: 前向 {f*1e3:.1f} ms 反向 {b*1e3:.1f} ms（profile 下）')
    print(f'  GPU 内核 {len(kern)} 个（每帧 {len(kern)/T:.1f}），GPU 时间合计 {gpu_us/1e3:.2f} ms（每帧 {gpu_us/T:.1f} µs）')
    print('  运行时调用：', pick('cudaLaunchKernel', 'cuLaunchKernel', 'cudaGraphLaunch', 'Synchronize', 'cudaMemcpy', 'cudaStreamWait'))
    by = {}
    for e in kern:
        k = e.name[:60]
        d = by.setdefault(k, [0, 0.0])
        d[0] += 1
        d[1] += e.device_time if hasattr(e, 'device_time') else e.cuda_time
    print('  内核（按个数）:')
    for k, (n, t) in sorted(by.items(), key=lambda x: -x[1][0])[:25]:
        print(f'    {n:6d}  {t/1e3:8.2f} ms  {k}')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['bench', 'profile'])
    ap.add_argument('--B', default='1,4,16')
    ap.add_argument('--T', default='24,200')
    ap.add_argument('--mode', default='compile,graphed,fused')
    ap.add_argument('--reps', type=int, default=7)
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument('--json', default='')
    ap.add_argument('--l2d', default='', help='旧版 l2d 包的上一级目录（测改动前）')
    args = ap.parse_args()
    import torch._dynamo.config as dcfg
    dcfg.recompile_limit = 64   # 旧版 compile() 没放宽：多个 B 时第 9 份编译起静默退回 eager，测「改动前」也要排除这一点
    torch.cuda.set_per_process_memory_fraction(3 * 2**30 / torch.cuda.get_device_properties(0).total_memory)  # 与 A5 共用 GPU：≤ 3 GB
    {'bench': cmd_bench, 'profile': cmd_profile}[args.cmd](args)


if __name__ == '__main__':
    main()
