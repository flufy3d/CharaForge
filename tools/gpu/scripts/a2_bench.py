"""
A2 物理速度：T 帧前向 + 整段反向（BPTT）的耗时与显存。

    tools/.venv/bin/python -I tools/gpu/scripts/a2_bench.py [--B 1,64] [--T 300] [--dtype f64] [--mode eager,compile,compile+ckpt]

输入取参考序列 s30_full 的参数（循环补到 T 帧），批次内每个样本加不同的小扰动。
loss = 所有物理输出参数的加权和；反向对整段输入参数求梯度。
mode：eager = 逐帧调用 rig.step；compile = rig.compile() 之后（torch.compile 融合单帧前向与反向，首次调用含编译时间，不计入）；
      +ckpt = rollout(checkpoint=True) 逐帧激活检查点。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PHYS = os.path.join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json')
REF = os.path.join(REPO, '.data/gpu/a2/ref')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', default='1,64')
    ap.add_argument('--T', type=int, default=300)
    ap.add_argument('--dtype', default='f64')
    ap.add_argument('--mode', default='eager,compile')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--split', default='0', help='split_by_length：1、0 或 1,0（两种都测）')
    args = ap.parse_args()
    dev = 'cuda'
    dtype = torch.float64 if args.dtype == 'f64' else torch.float32
    rigs = {s: PhysicsRig.from_files(PHYS, os.path.join(REF, 'model_params.json'), device=dev, dtype=dtype,
                                    split_by_length=s == '1') for s in args.split.split(',')}
    rig = next(iter(rigs.values()))
    import json
    meta = json.load(open(os.path.join(REF, 's30_full.json')))
    xin = np.fromfile(os.path.join(REF, 's30_full.in.f32'), dtype='<f4').reshape(meta['T'], meta['P'])
    xin = np.concatenate([xin] * (args.T // len(xin) + 1))[:args.T]
    outp = rig.output_param_index
    for (split, rig), mode in [(r, m) for r in rigs.items() for m in args.mode.split(',')]:
        for B in [int(b) for b in args.B.split(',')]:
            base = torch.tensor(xin, dtype=dtype, device=dev)
            g = torch.Generator(device=dev).manual_seed(0)
            x0 = base.unsqueeze(1).repeat(1, B, 1) + 0.01 * torch.randn(args.T, B, base.shape[1], device=dev, dtype=dtype, generator=g)
            w = torch.randn(args.T, B, len(rig.output_params), device=dev, dtype=dtype, generator=g)
            if 'compile' in mode and rig._frame_c is None:
                rig.compile()
            times = []
            for rep in range(args.reps + 1):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_allocated()
                x = x0.clone().requires_grad_(True)
                t0 = time.perf_counter()
                _, out = rig.rollout(x, dt=1 / 60, checkpoint='ckpt' in mode)
                loss = (out[:, :, outp] * w).sum()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                peak = torch.cuda.max_memory_allocated() - m0
                bad = int((~torch.isfinite(x.grad)).sum())
                if rep:
                    times.append((t1 - t0, t2 - t1, peak))
                del x, out, loss
            f = np.median([t[0] for t in times]); b = np.median([t[1] for t in times]); pk = max(t[2] for t in times)
            print(f'{mode:12s} split={split} {args.dtype} B={B:3d} T={args.T}: 前向 {f:.3f}s  反向 {b:.3f}s  合计 {f + b:.3f}s  '
                  f'峰值显存 {pk / 2**20:.0f} MiB  非有限梯度 {bad}')


if __name__ == '__main__':
    main()
