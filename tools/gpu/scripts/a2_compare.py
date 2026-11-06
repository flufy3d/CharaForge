"""
A2 物理对拍：用 a2_ref_dump.ts 的参考序列驱动 l2d/physics.py，逐帧比较物理之后的参数。

    tools/.venv/bin/python -I tools/gpu/scripts/a2_compare.py [--ref .data/gpu/a2/ref] [--device cuda] [--dtype f64,f32io,f32]

误差 = |ours − ref| / (参数 max − min)，在所有物理输出参数上取最大；同时报告所在帧、参数，
以及按时间四等分的分段最大误差（检查漂移），和 rig 输出（_currentRigOutputs，弧度/长度）的最大绝对误差。
门槛：float64 < 1e-5，float32 < 1e-3。
  f64   ：rig 与输入输出都是 float64（Float32Array 处做 float32 舍入）
  f32io ：输入输出 float32，rig 内部 float64
  f32   ：全部 float32
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from l2d.physics import PhysicsRig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PHYS = os.path.join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json')
GATE = {'f64': 1e-5, 'f32': 1e-3, 'f32io': 1e-3}
SEQS = ['s24_main', 's15_left', 's30_full', 's06_main', 'step', 'sweep', 'walk', 'stab_walk', 'walk_dt50', 'walk_dt144', 'jitter']


def load(ref, name):
    meta = json.load(open(os.path.join(ref, f'{name}.json')))
    T, P, NO = meta['T'], meta['P'], meta['NO']
    rd = lambda ext, dt, n: np.fromfile(os.path.join(ref, f'{name}.{ext}'), dtype=dt).reshape(T, n)
    return meta, rd('in.f32', '<f4', P), rd('out.f32', '<f4', P), rd('rig.f64', '<f8', NO)


def run(rig, meta, xin, dtype, device):
    st = rig.initial_state(1)
    outs, rigs = [], []
    with torch.no_grad():
        for t, ev in enumerate(meta['events']):
            p = torch.tensor(xin[t:t + 1], dtype=dtype, device=device)
            if ev['kind'] == 'stab':
                st, o = rig.stabilize(st, p)
            else:
                st, o = rig.step(st, p, ev['dt'])
            outs.append(o[0])
            rigs.append(st.cur[0])
    return torch.stack(outs).double().cpu().numpy(), torch.stack(rigs).double().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=os.path.join(REPO, '.data/gpu/a2/ref'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='f64,f32io,f32')
    ap.add_argument('--seqs', default=','.join(SEQS))
    ap.add_argument('--compile', action='store_true', help='用 rig.compile()（torch.compile）版本对拍')
    args = ap.parse_args()
    tab = json.load(open(os.path.join(args.ref, 'model_params.json')))
    rng = np.array(tab['max'], dtype=np.float64) - np.array(tab['min'], dtype=np.float64)
    rng[rng == 0] = 1.0
    ok_all = True
    for dname in args.dtype.split(','):
        dtype = torch.float64 if dname == 'f64' else torch.float32       # 输入输出张量的类型
        rdt = torch.float32 if dname == 'f32' else torch.float64         # f32io：float32 输入输出、float64 计算
        rig = PhysicsRig.from_files(PHYS, os.path.join(args.ref, 'model_params.json'), device=args.device, dtype=rdt)
        if args.compile:
            rig.compile()
        outp = np.array(rig.output_params)
        if dname == args.dtype.split(',')[0]:
            print(f'setting {rig.S} 个，分 {len(rig.stages)} 层 {len(rig.levels)} 组；物理输出参数 {len(outp)} 个，输入参数 {len(rig.input_params)} 个')
        print(f'== {dname}{"（compile）" if args.compile else ""}（门槛 {GATE[dname]:g}，误差相对参数范围）')
        for name in args.seqs.split(','):
            meta, xin, xout, xrig = load(args.ref, name)
            t0 = time.time()
            ours, orig = run(rig, meta, xin, dtype, args.device)
            el = time.time() - t0
            err = np.abs(ours[:, outp] - xout[:, outp].astype(np.float64)) / rng[outp]
            # 非物理参数必须原样不动
            others = np.setdiff1d(np.arange(xin.shape[1]), outp)
            err_o = np.abs(ours[:, others] - xout[:, others]).max() if len(others) else 0.0
            T = err.shape[0]
            t, j = np.unravel_index(np.nanargmax(err), err.shape) if not np.isnan(err).all() else (0, 0)
            e = float(err[t, j]) if not np.isnan(err).any() else float('nan')
            q = [float(np.nanmax(err[a:b])) for a, b in zip(np.linspace(0, T, 5).astype(int)[:-1], np.linspace(0, T, 5).astype(int)[1:])]
            erig = float(np.abs(orig - xrig).max())
            ok = e < GATE[dname] and err_o == 0.0
            ok_all &= ok
            fr = meta.get('frames', [0])[0] + int(t) if meta['kind'] == 'layer' else int(t)
            print(f'  {name:11s} T={T:4d} 最大误差 {e:.3e} @帧 {fr} 参数 {tab["ids"][outp[j]]:22s} '
                  f'四段 [{", ".join(f"{x:.1e}" for x in q)}] rig {erig:.1e} 其他参数 {err_o:g} '
                  f'{"OK" if ok else "FAIL"} ({el:.1f}s)')
    print('全部通过' if ok_all else '有未通过项')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
