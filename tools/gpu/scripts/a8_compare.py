"""
A8 对拍：用 a2_ref_dump.ts 的 11 个参考序列驱动 fused_rollout（Triton 融合核）或 graphed_rollout（CUDA graph），
与 runtime 参考比较（口径同 a2_compare.py），并与同一 rig 的逐帧 step（原路径）逐位比较。

    tools/.venv/bin/python -I tools/gpu/scripts/a8_compare.py [--path fused|graphed] [--dtype f64,f32io] [--compile] [--window 24]

序列按事件切段：stabilization 事件走 rig.stabilize；相邻、dt 相同的 step 事件每 window 帧一段交给 新路径
（jitter 每帧 dt 不同，于是每段 1 帧——仍然走图，只是每段都要新建图，较慢）。
报告：与参考的最大误差（相对参数范围，门槛同 a2_compare）、与原路径的最大差（应为 0）、rig 输出的最大差。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig  # noqa: E402
from a2_compare import GATE, PHYS, REPO, SEQS, load  # noqa: E402


def run_plain(rig, meta, xin, dtype, device):
    st = rig.initial_state(1)
    outs, rigs = [], []
    with torch.no_grad():
        for t, ev in enumerate(meta['events']):
            p = torch.tensor(xin[t:t + 1], dtype=dtype, device=device)
            st, o = rig.stabilize(st, p) if ev['kind'] == 'stab' else rig.step(st, p, ev['dt'])
            outs.append(o[0])
            rigs.append(st.cur[0])
    return torch.stack(outs).double().cpu().numpy(), torch.stack(rigs).double().cpu().numpy()


def run_graphed(rig, meta, xin, dtype, device, window, path):
    st = rig.initial_state(1)
    outs, rigs = [], []
    ev = meta['events']
    x = torch.tensor(xin, dtype=dtype, device=device).unsqueeze(1)  # [T, 1, P]
    t = 0
    with torch.no_grad():
        while t < len(ev):
            if ev[t]['kind'] == 'stab':
                st, o = rig.stabilize(st, x[t])
                outs.append(o[0])
                rigs.append(st.cur[0])
                t += 1
                continue
            dt = ev[t]['dt']
            e = t
            while e < len(ev) and e - t < window and ev[e]['kind'] != 'stab' and ev[e]['dt'] == dt:
                e += 1
            st, o = getattr(rig, path + "_rollout")(x[t:e], st, dt)
            outs += list(o[:, 0])
            # rig 输出（_currentRigOutputs）只比段末的状态；--window 1 时逐帧都比
            rigs += [None] * (e - t - 1) + [st.cur[0]]
            t = e
    return torch.stack(outs).double().cpu().numpy(), rigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=os.path.join(REPO, '.data/gpu/a2/ref'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='f64,f32io')
    ap.add_argument('--seqs', default=','.join(SEQS))
    ap.add_argument('--window', type=int, default=24)
    ap.add_argument("--compile", action="store_true", help="原路径与 graphed 用 compile 版（fused 不受影响，始终与 eager 逐位比较）")
    ap.add_argument("--path", default="fused", choices=["fused", "graphed"])
    args = ap.parse_args()
    tab = json.load(open(os.path.join(args.ref, 'model_params.json')))
    rng = np.array(tab['max'], dtype=np.float64) - np.array(tab['min'], dtype=np.float64)
    rng[rng == 0] = 1.0
    ok_all = True
    for dname in args.dtype.split(','):
        dtype = torch.float64 if dname == 'f64' else torch.float32
        rdt = torch.float32 if dname == 'f32' else torch.float64
        rig = PhysicsRig.from_files(PHYS, os.path.join(args.ref, 'model_params.json'), device=args.device, dtype=rdt)
        if args.compile and args.path == "graphed":
            rig.compile()
        outp = np.array(rig.output_params)
        print(f'== {dname}{"（compile）" if args.compile else ""} {args.path}_rollout，window={args.window}（门槛 {GATE[dname]:g}；'
              f'「原路径差」= 与同一 rig 逐帧 step 的最大绝对差，应为 0）')
        for name in args.seqs.split(','):
            meta, xin, xout, xrig = load(args.ref, name)
            t0 = time.time()
            ours, rigs = run_graphed(rig, meta, xin, dtype, args.device, args.window, args.path)
            el = time.time() - t0
            plain, prig = run_plain(rig, meta, xin, dtype, args.device)
            err = np.abs(ours[:, outp] - xout[:, outp].astype(np.float64)) / rng[outp]
            others = np.setdiff1d(np.arange(xin.shape[1]), outp)
            err_o = np.abs(ours[:, others] - xout[:, others]).max() if len(others) else 0.0
            T = err.shape[0]
            t, j = np.unravel_index(np.nanargmax(err), err.shape) if not np.isnan(err).all() else (0, 0)
            e = float(err[t, j]) if not np.isnan(err).any() else float('nan')
            q = [float(np.nanmax(err[a:b])) for a, b in zip(np.linspace(0, T, 5).astype(int)[:-1], np.linspace(0, T, 5).astype(int)[1:])]
            dpl = float(np.abs(ours - plain).max())
            ri = [k for k, r in enumerate(rigs) if r is not None]
            erig = float(np.abs(torch.stack([rigs[k] for k in ri]).double().cpu().numpy() - xrig[ri]).max())
            drig = float(np.abs(torch.stack([rigs[k] for k in ri]).double().cpu().numpy() - prig[ri]).max())
            ok = e < GATE[dname] and err_o == 0.0 and dpl == 0.0 and drig == 0.0
            ok_all &= ok
            fr = meta.get('frames', [0])[0] + int(t) if meta['kind'] == 'layer' else int(t)
            print(f'  {name:11s} T={T:4d} 最大误差 {e:.3e} @帧 {fr} 参数 {tab["ids"][outp[j]]:22s} '
                  f'四段 [{", ".join(f"{x:.1e}" for x in q)}] rig {erig:.1e} 其他参数 {err_o:g} '
                  f'原路径差 {dpl:g}（rig {drig:g}） {"OK" if ok else "FAIL"} ({el:.1f}s)', flush=True)
        print(f'  图：{rig.graph_stats}')
    print('全部通过' if ok_all else '有未通过项')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
