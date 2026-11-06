"""A7：变形器（A1）与光栅化（A3）的耗时 / 显存测量，同一口径测改动前后。

    tools/.venv/bin/python -I tools/gpu/scripts/a7_bench.py deform [--l2d DIR] [--batches 1,64,512] [--reps 9] [--fast]
    tools/.venv/bin/python -I tools/gpu/scripts/a7_bench.py raster [--l2d DIR] [--reps 7] [--fast] [--chunks 1,4,8,16,32]
    tools/.venv/bin/python -I tools/gpu/scripts/a7_bench.py profile-deform|profile-raster [--l2d DIR] [--fast] [--out trace.json]

--l2d：从哪个目录导入 l2d 包（缺省本仓库 tools/gpu；测改动前的版本时指向保存的旧副本的上一级目录）。
--fast：使用 A7 新增的可选快速路径（deform：L2D.graphed() 的 CUDA graph；raster：render(..., fast=True)）。
每项取 reps 次的中位数；同时打印当时 nvidia-smi 的 GPU 占用（WSL 下进程列表不可见，只看利用率）。
进程显存限在 4 GB（set_per_process_memory_fraction），超出记 OOM。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def gpu_util() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def setup(l2d_root: str | None):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(l2d_root) if l2d_root else REPO / "tools/gpu"))
    sys.path.insert(1, str(REPO / "tools/gpu/scripts"))
    import torch
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4.0e9 / torch.cuda.get_device_properties(0).total_memory))
    import l2d
    print("l2d from", Path(l2d.__file__).parent, "| GPU util/mem:", gpu_util(), flush=True)


def med(xs):
    import numpy as np
    return float(np.median(xs))


# ----------------------------------------------------------------------------- deform

def deform_bench(a):
    import numpy as np
    import torch
    from a1_core import MOC, load_ref
    from l2d import moc3
    from l2d.deform import L2D
    import l2d.deform as Dm
    if a.triton is not None:
        Dm.USE_TRITON = bool(a.triton)
    print("deform USE_TRITON =", getattr(Dm, "USE_TRITON", None), flush=True)
    m = moc3.load(MOC)
    model = L2D(m, dtype=getattr(torch, a.dtype))
    allp = torch.as_tensor(load_ref(REPO / ".data/gpu/ref/scene")["params"])
    out = []
    def clear():
        if hasattr(model, "_graphs"):
            model._graphs.clear()
        torch.cuda.empty_cache()

    def measure(P, chunk, grad):
        """B 个样本分成每块 chunk 个（显存不够时），返回 (各次耗时, 显存峰值)。"""
        B = P.shape[0]
        clear()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn = model.graphed(chunk, grad=grad) if a.fast else model
        parts = [P[i:i + chunk] for i in range(0, B, chunk)]

        def step():
            if grad:
                for x in parts:
                    xg = x.clone().requires_grad_(True)
                    o = fn(xg)
                    (o.verts.sum() + o.opacity.sum()).backward()
            else:
                with torch.no_grad():
                    for x in parts:
                        fn(x)
        for _ in range(2):
            step()
        ts = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        return ts, torch.cuda.max_memory_allocated() - base

    for B in [int(x) for x in a.batches.split(",")]:
        P = allp[torch.arange(B) % allp.shape[0]].to("cuda", model.dtype)
        r = {"B": B, "fast": a.fast, "triton": getattr(Dm, "USE_TRITON", None), "dtype": a.dtype}
        for grad, key in ((False, "fwd"), (True, "fwd_bwd")):
            chunk = B
            while True:
                try:
                    ts, mem = measure(P, chunk, grad)
                    break
                except torch.cuda.OutOfMemoryError:
                    clear()
                    if chunk == 1:
                        raise
                    chunk //= 2  # 4 GB 放不下：分块（结果与整批相同），记下块大小
            r.update({f"{key}_ms": round(med(ts) * 1e3, 2), f"{key}_ms_min": round(min(ts) * 1e3, 2), f"peak_{key}_MB": round(mem / 2**20)})
            if chunk != B:
                r[f"{key}_chunk"] = chunk
        r["gpu"] = gpu_util()
        clear()
        print(json.dumps(r), flush=True)
        out.append(r)
    return out


# ----------------------------------------------------------------------------- raster

def raster_setup(dump):
    from l2d.dumpio import Dump
    from l2d.raster import load_rasterizer
    d = Dump(dump)
    R = load_rasterizer(d, "dsc", assets_root=REPO / "assets")
    layers = [next(L for L in d.frame(f)["layers"] if L["visible"]) for f in d.frames]
    return R, layers


def scaled_inp(layers, s):
    from l2d.raster import LayerInput
    inp = LayerInput.from_dump(layers)
    return LayerInput(inp.positions, inp.opacities, inp.multiply, inp.screen, inp.render_orders, inp.dyn_flags, inp.to_screen * s)


def pipeline(R, inp, W, H, draw, aa, fast, aa_layers=None):
    C_ck = fast
    from l2d import composite as C
    kw = {"fast": True} if fast else {}
    buf = R.render(inp, W, H, aa=aa, quantize=False, aa_layers=aa_layers, **kw)
    ck = {"checkpoint_filters": True} if C_ck else {}  # 快速路径同时打开滤镜分帧检查点（旧版 composite 没有这个参数）
    return C.composite([(buf, draw)], H, W, None, quantize=False, B=buf.shape[0], **ck)


def run_frames(R, inps, W, H, draw, grad, aa, fast, aa_layers=None):
    import torch
    from l2d.raster import LayerInput
    for inp in inps:
        if grad:
            pos = inp.positions.clone().requires_grad_(True)
            x = LayerInput(pos, inp.opacities, inp.multiply, inp.screen, inp.render_orders, inp.dyn_flags, inp.to_screen)
            I = pipeline(R, x, W, H, draw, aa, fast, aa_layers)
            I.square().mean().backward()
        else:
            with torch.no_grad():
                pipeline(R, inp, W, H, draw, False, fast)


def timed(fn, reps):
    import torch
    fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return ts


def raster_bench(a):
    import torch
    R, layers = raster_setup(a.dump)
    tex = torch.cuda.memory_allocated()
    draw = layers[0]["draw"]
    W, H, s = 640, 360, 0.5
    out = {"textureGB": round(tex / 1e9, 3), "rows": []}
    mid = len(layers) // 2
    # 单帧：前向（不求导）、前向 + 反向（边缘梯度全开 / 关）
    for aa in (True, False):
        inps = [scaled_inp([layers[mid]], s)]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tb = timed(lambda: run_frames(R, inps, W, H, draw, True, aa, a.fast), a.reps)
        pk = torch.cuda.max_memory_allocated()
        tf = timed(lambda: run_frames(R, inps, W, H, draw, False, aa, a.fast), a.reps) if aa else None
        r = {"case": "1 frame", "aa": aa, "fwd_s": round(med(tf), 4) if tf else None, "fwd_bwd_s": round(med(tb), 4),
             "fwd_bwd_min_s": round(min(tb), 4), "peakGB": round(pk / 1e9, 3), "gpu": gpu_util()}
        print(json.dumps(r), flush=True)
        out["rows"].append(r)
    # 32 帧，按 chunk 分块
    for chunk in [int(x) for x in a.chunks.split(",")]:
        for aa in ((True, False) if a.both_aa else (True,)):
            inps = [scaled_inp(layers[i:i + chunk], s) for i in range(0, 32, chunk)]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                tb = timed(lambda: run_frames(R, inps, W, H, draw, True, aa, a.fast), max(1, a.reps // 2))
                pk = torch.cuda.max_memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                tf = timed(lambda: run_frames(R, inps, W, H, draw, False, aa, a.fast), max(1, a.reps // 2))
                pkf = torch.cuda.max_memory_allocated()
                r = {"case": f"32 frames chunk {chunk}", "aa": aa, "fwd_s": round(med(tf), 3), "fwd_bwd_s": round(med(tb), 3),
                     "peakGB": round(pk / 1e9, 3), "peakFwdGB": round(pkf / 1e9, 3), "gpu": gpu_util()}
            except torch.cuda.OutOfMemoryError:
                r = {"case": f"32 frames chunk {chunk}", "aa": aa, "error": "OOM (>4GB)"}
            del inps
            torch.cuda.empty_cache()
            print(json.dumps(r), flush=True)
            out["rows"].append(r)
    return out


# ----------------------------------------------------------------------------- profile

def profile(a, what):
    import torch
    from torch.profiler import ProfilerActivity, profile as tprof
    if what == "deform":
        from a1_core import MOC, load_ref
        from l2d import moc3
        from l2d.deform import L2D
        model = L2D(moc3.load(MOC), dtype=torch.float32)
        allp = torch.as_tensor(load_ref(REPO / ".data/gpu/ref/scene")["params"])
        P = allp[torch.arange(a.B) % allp.shape[0]].to("cuda").requires_grad_(True)
        fn = model.graphed(a.B, grad=True) if a.fast else model

        def step():
            P.grad = None
            o = fn(P)
            (o.verts.sum() + o.opacity.sum()).backward()
    else:
        R, layers = raster_setup(a.dump)
        inps = [scaled_inp(layers[i:i + a.B], 0.5) for i in range(0, a.B, a.B)]
        draw = layers[0]["draw"]

        def step():
            run_frames(R, inps, 640, 360, draw, not a.no_grad, True, a.fast)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    n_kernels = sum(e.count for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA) if False else None
    print(ka.table(sort_by="self_cuda_time_total", row_limit=a.rows))
    print(ka.table(sort_by="self_cpu_time_total", row_limit=a.rows))
    ev = prof.events()
    names = {}
    nk = 0
    gpu_us = 0.0
    for e in ev:
        if e.device_type == torch.autograd.DeviceType.CUDA:
            nk += 1
            gpu_us += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    syncs = sum(e.count for e in ka if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaEventSynchronize"))
    launches = sum(e.count for e in ka if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cudaGraphLaunch"))
    print(json.dumps({"gpu_kernels": nk, "gpu_kernel_ms": round(gpu_us / 1e3, 2), "launch_calls": launches,
                      "sync_or_memcpy_calls": syncs, "detail": {e.key: e.count for e in ka if e.key.startswith("cuda") or e.key.startswith("cu")}}))
    if a.out:
        prof.export_chrome_trace(a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["deform", "raster", "profile-deform", "profile-raster"])
    ap.add_argument("--l2d", default=None)
    ap.add_argument("--batches", default="1,64,512")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--triton", type=int, default=None, help="deform：1/0 强制打开/关闭 Triton 融合核（缺省用模块默认值）")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dump", default=str(REPO / ".data/gpu/frames/s24b"))
    ap.add_argument("--chunks", default="1,4,8,16,32")
    ap.add_argument("--both-aa", action="store_true")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--no-grad", action="store_true")
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    setup(a.l2d)
    if a.what == "deform":
        r = deform_bench(a)
    elif a.what == "raster":
        r = raster_bench(a)
    else:
        profile(a, a.what.split("-")[1])
        return
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
