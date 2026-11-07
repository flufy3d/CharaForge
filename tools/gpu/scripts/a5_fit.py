"""A5：GPU 整段轨迹联合优化 → scene.json 图层（按拟合单元：continues 链合并的镜头、一个或多个图层）。

    tools/.venv/bin/python -I tools/gpu/scripts/a5_fit.py --shot s41 [--out .data/gpu/fit/<单元>] [--quick] [--seed 0]

冷启动：初值只来自 shots.json（眼睛锚点、cadence、效果、开关、clip）与 tools/fit 的规划（plans.ts，经 a5_plan.ts）；
不读现有 scene.json 的拟合结果。产出：<out>/layers.json（单元的全部图层）、<out>/fit.json（耗时、显存、曲线、常量、开关估计）、
<out>/fit.log、<out>/preview/*.jpg。合并进全片场景用 tools/gpu/scripts/a5_merge.ts。
"""
from __future__ import annotations

import os

# 确定性（须在 import torch 之前）：cuBLAS 的工作区配置；见 fit.md「可复现」
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fit.export import build_layer  # noqa: E402
from fit.optimize import Fitter, Stage  # noqa: E402
from fit.plan import finalize_nodes, make_batch, make_refs, make_unit  # noqa: E402
from fit.rig import REPO, VTUBE, Rig  # noqa: E402
from fit.traj import Traj  # noqa: E402
from fit.vts import Vts  # noqa: E402


def stages(quick: bool) -> list[Stage]:
    q = (lambda n: max(2, n // 4)) if quick else (lambda n: n)
    lo = dict(W=320, H=180, chunk=24)
    hi = dict(W=640, H=360, chunk=16)
    return [
        # 1) 光流对应（320×180）：变换 + 面捕输入 + 前倾；外层渲染算光流目标，内层只跑物理 + 变形
        Stage("F1-光流粗", **lo, outer=q(10), inner=8, lr={"tf": 3.0, "inputs": 0.3, "lean": 1.0}, flow_c=12.0, w_anchor=0.05),
        # 2) 光流对应（640×360）：加 postPhysics 偏移
        Stage("F2-光流细", **hi, outer=q(14), inner=8, lr={"tf": 1.5, "inputs": 0.15, "post": 0.2, "lean": 0.5}, flow_c=5.0, w_anchor=0.03),
        # 3) 光流对应（640×360）：尺度收紧到 3px
        Stage("F3-光流精", **hi, outer=q(10), inner=8, lr={"tf": 0.8, "inputs": 0.08, "post": 0.12, "lean": 0.3}, flow_c=3.0, w_anchor=0.02),
    ]


def save_preview(fitter: Fitter, path: Path, frames: list[int]) -> None:
    """预览：参考 | 渲染合成到参考（640×360）。"""
    with torch.no_grad():
        comp, _, _ = fitter.render(frames, 0.5, 640, 360)
        ref, _ = fitter.refs.at(640, 360, frames)
        ref = torch.tensor(ref, device=fitter.dev)
        c = comp[..., :3] * 255 + ref * (1 - comp[..., 3:4])
        img = torch.cat([torch.cat([r, x], 1) for r, x in zip(ref, c)], 0)
    Image.fromarray(img.clamp(0, 255).byte().cpu().numpy()).save(path, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="s24")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="迭代数 ÷4（调试）")
    ap.add_argument("--decouple", type=int, default=1, help="身体由独立输入 BodyX/Y/Z 驱动（vts.ts 的 MODELX.decouple）")
    ap.add_argument("--smooth-c", type=float, default=0.15)
    ap.add_argument("--mem-gb", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", type=int, default=0,
                    help="torch.use_deterministic_algorithms（内层慢约 3 倍；A3 光栅化本身不是逐位确定的，开了也不能逐位复现，见 fit.md）")
    ap.add_argument("--fast", type=int, default=1, help="A7 光栅化快速路径 + 滤镜检查点")
    ap.add_argument("--graph", type=int, default=1, help="A7 变形器 CUDA graph")
    ap.add_argument("--deform64", type=int, default=1, help="变形器 float64（逐位可复现）")
    ap.add_argument("--switch-after", default="F1",
                    help="在哪个阶段之后估计开关切换帧（缺省 F1：此时还没有逐节点的 postPhysics 偏移，姿态不会去迁就标注的切换时刻）")
    ap.add_argument("--max-frames", type=int, default=40, help="每次外层最多渲染的显示帧数（多于它时按子集轮换）")
    ap.add_argument("--ssim-iters", type=int, default=12, help="光流阶段之后 SSIM@360 精修的迭代数（0 关闭）")
    ap.add_argument("--lean", type=float, default=None, help="诊断：所有前倾可拟合的图层固定用这个初值（跳过多假设）")
    ap.add_argument("--no-lean-trial", action="store_true", help="前倾只做光度比较、不跑两支短试拟合")
    ap.add_argument("--no-register", action="store_true", help="诊断：不做小实例粗配准（Fitter.register_small）")
    args = ap.parse_args()

    # 可复现：固定种子 + 确定性算子（不支持确定性实现的算子只告警，见 fit.md）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    import cv2
    cv2.setRNGSeed(args.seed)

    t0 = time.time()
    # A7 的提速（tools/gpu/docs/perf.md）：光栅化快速路径、滤镜检查点、变形器 CUDA graph（批大小分桶）；
    # 变形器用 float64：float32 的稀疏矩阵乘不是逐次确定的，float64 下整个拟合逐位可复现（见 fit.md「可复现」）
    from l2d import composite as _C, raster as _R
    _R.FAST_DEFAULT = bool(args.fast)
    _C.CHECKPOINT = bool(args.fast)
    rig = Rig(phys_compile=False, deform_dtype=torch.float64 if args.deform64 else torch.float32)
    rig.use_graph = bool(args.graph)
    # --shot 可以是逗号分隔的多个镜头：各自所在的拟合单元摞成一批（多单元批处理），结果按单元分别落盘到 <out>/<单元>/
    units = []
    for x in args.shot.split(","):
        uu = make_unit(x)
        if uu.unit not in [v.unit for v in units]:
            units.append(uu)
    u = make_batch(units)
    out = Path(args.out or REPO / f".data/gpu/fit/{u.unit}")
    (out / "preview").mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit.log", "w", encoding="utf-8")

    def log(s: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    T: dict[str, float] = {}
    refs = make_refs(u)
    finalize_nodes(u, refs)
    vts = Vts(VTUBE, {k: float(v) for k, v in zip(rig.param_ids, rig.pdef)}, decouple=bool(args.decouple))
    trajs = [Traj(rig, lp, vts, smooth_c=args.smooth_c) for lp in u.layers]
    fitter = Fitter(trajs, u, refs, log)
    T["装载（模型、贴图、参考帧、规划）"] = time.time() - t0
    log(f"单元 {u.unit}（{'+'.join(u.shot_ids)}）[{u.start}, {u.end})，渲染帧 {len(fitter.F_all)}")
    for lp, t in zip(u.layers, trajs):
        log(f"  {lp.id} [{lp.fin}, {lp.fout}) z {lp.z}：节点 {len(lp.nodes)}，关键帧 {len(lp.keys)}，锚点 {len(lp.anchors)}，frameHold {lp.holds}，"
            f"开关 {[(s['param'], s['est'], s['fade']) for s in lp.switches]}，前倾 {'拟合' if t.lean_free else lp.fixed.get('lean')}，"
            f"clothes2 {t.clothes2}，clip {'有' if lp.clip else '无'}，景深 {'有' if t.has_dof else '无'}，翻牌 {'有' if lp.scale_x else '无'}，颜色 {lp.color_filter}")
    for k in u.skipped:
        log(f"  跳过 {k['id']}：{k['reason']}")

    t0 = time.time()
    with torch.no_grad():  # 物理编译（torch.compile，首次调用触发；FX 缓存热时约 4 s）
        from fit.traj import physics_batch
        physics_batch(trajs)
    fitter.reanchor()
    T["初值 + 物理编译"] = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    S = stages(args.quick)
    for st in S:
        st.max_frames = args.max_frames * len(units)  # 每个单元每次外层的帧数与单单元时相同
    previews = [fitter.F_all[i] for i in np.linspace(0, len(fitter.F_all) - 1, 5).astype(int)]
    save_preview(fitter, out / "preview/0_init.jpg", previews)
    t0 = time.time()
    if not args.no_register:
        fitter.register_small()
    T["小实例粗配准"] = time.time() - t0
    t0 = time.time()
    trial = Stage(**{**S[0].__dict__, "name": "前倾试拟合"})
    if args.lean is not None:
        for t in trajs:
            if t.lean_free:
                with torch.no_grad():
                    t.consts["lean"].fill_(args.lean)
        fitter.reanchor()
    else:
        fitter.search_lean(trial, trial_outer=0 if args.no_lean_trial else (2 if args.quick else 4))
    T["前倾多假设"] = time.time() - t0
    switches = []
    for k, st in enumerate(S):
        t0 = time.time()
        fitter.run_flow_stage(st)
        T[st.name] = time.time() - t0
        save_preview(fitter, out / f"preview/{k + 1}_{st.name}.jpg", previews)
        if st.name.startswith(args.switch_after) and any(t.plan.switches for t in trajs):
            t0 = time.time()
            switches = fitter.estimate_switches()
            T["开关切换帧估计"] = time.time() - t0
    t0 = time.time()
    if args.ssim_iters > 0:
        t0 = time.time()
        sst = Stage("S-SSIM精修", 640, 360, {"tf": 0.3, "inputs": 0.04, "post": 0.06, "lean": 0.1}, chunk=8, w_anchor=0.02,
                    max_frames=24 * len(units), lr_final=0.3)
        fitter.run_ssim_stage(sst, args.ssim_iters * min(3, len(fitter.subsets(sst))))
        T["S-SSIM精修"] = time.time() - t0
    for uu in (u.units or [u]):
        sub = Fitter([t for t in trajs if t.plan.unit == uu.unit], uu, refs, log) if u.units else fitter
        apr = sub.fit_appearance()
        log(f"外观 {uu.unit}：resample {apr['resample']}，SSIM@360 {apr['ssim']:.4f}")
    T["A-外观常量"] = time.time() - t0
    save_preview(fitter, out / "preview/9_final.jpg", previews)
    peak = torch.cuda.max_memory_allocated() / 2**30
    t0 = time.time()
    for uu in (u.units or [u]):
        od = (out / uu.unit) if u.units else out
        od.mkdir(parents=True, exist_ok=True)
        tr = [t for t in trajs if t.plan.unit == uu.unit]
        (od / "layers.json").write_text(json.dumps([build_layer(t) for t in tr], ensure_ascii=False))
    T["导出"] = time.time() - t0
    total = sum(T.values())
    log("耗时：" + "，".join(f"{k} {v:.1f}s" for k, v in T.items()) + f"；合计 {total:.1f}s；显存峰值 {peak:.2f} GB")
    log("前向细分：" + "，".join(f"{k} {v:.1f}s" for k, v in fitter.timing.items()))
    for uu in (u.units or [u]):
        od = (out / uu.unit) if u.units else out
        tr = [t for t in trajs if t.plan.unit == uu.unit]
        (od / "fit.json").write_text(json.dumps({
            "unit": uu.unit, "shots": uu.shot_ids, "range": [uu.start, uu.end], "batch": [v.unit for v in units],
            "timing": T, "totalSec": total, "forwardTiming": fitter.timing, "peakGB": peak,
            "renderFrames": len([f for f in fitter.F_all if uu.start <= f < uu.end]), "seed": args.seed,
            "switches": [x for x in switches if any(x["layer"] == t.plan.id for t in tr)],
            "layers": {t.plan.id: {"resample": t.resample, "consts": {k: (v.tolist() if v.dim() else float(v)) for k, v in t.consts.items()},
                                   "nodes": len(t.nodes), "holds": t.plan.holds, "smallK": t.small_k,
                                   # 头部角度输入（逐节点，诊断姿态漂移用）、体型开关的尺度增益（ln）
                                   "headInputs": {k: [round(float(v), 2) for v in t.inputs[:, i]] for i, k in enumerate(t.inputs_fit) if k.startswith("FaceAngle")},
                                   "swGain": [round(float(v), 4) for v in t.sw_gain],
                                   "clothes2": t.clothes2,  # 体型常量（小实例的粗配准可能改掉规划值）
                                   } for t in tr},
            "history": fitter.history if not u.units else [], "decouple": bool(args.decouple), "smoothC": args.smooth_c,
        }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
