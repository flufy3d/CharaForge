"""A5：GPU 整段轨迹联合优化 → scene.json 图层。

    tools/.venv/bin/python -I tools/gpu/scripts/a5_fit.py --shot s24 [--out .data/gpu/fit/s24] [--quick]

冷启动：初值只来自 shots.json（眼睛锚点、cadence、效果）与镜头规划；不读现有 scene.json 的拟合结果。
产出：<out>/layer.json（图层）、<out>/fit.json（耗时、显存、各阶段曲线、最终常量）、<out>/preview/*.jpg；
合并进全片场景用 tools/gpu/scripts/a5_merge.ts。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fit.export import build_layer  # noqa: E402
from fit.optimize import Fitter, Stage  # noqa: E402
from fit.plan import Refs, finalize_nodes, make_plan  # noqa: E402
from fit.rig import REPO, VTUBE, Rig  # noqa: E402
from fit.traj import Traj  # noqa: E402
from fit.vts import Vts  # noqa: E402


def stages(quick: bool) -> list[Stage]:
    q = (lambda n: max(2, n // 4)) if quick else (lambda n: n)
    lo = dict(W=320, H=180, chunk=12)
    hi = dict(W=640, H=360, chunk=4)
    return [
        # 1) 光流对应（320×180）：变换 + 面捕输入 + 前倾；外层渲染算光流目标，内层只跑物理 + 变形
        Stage("F1-光流粗", 0, **lo, outer=q(10), inner=8,
              lr={"tf": 3.0, "inputs": 0.3, "lean": 1.0},
              w_flow=1.0, flow_c=12.0, w_anchor=0.05),
        # 2) 光流对应（640×360）：加 postPhysics 偏移（次级物理分组、姿态输出）
        Stage("F2-光流细", 0, **hi, outer=q(14), inner=8,
              lr={"tf": 1.5, "inputs": 0.15, "post": 0.2, "lean": 0.5},
              w_flow=1.0, flow_c=5.0, w_anchor=0.03),
        # 3) 光流对应（640×360）：尺度收紧到 3px
        Stage("F3-光流精", 0, **hi, outer=q(10), inner=8,
              lr={"tf": 0.8, "inputs": 0.08, "post": 0.12, "lean": 0.3},
              w_flow=1.0, flow_c=3.0, w_anchor=0.02),
    ]


def synth_frames(rig, plan, scene_path: str, real: np.ndarray) -> np.ndarray:
    from fit.render import compose_one, raster_layer, to_screen
    from fit.tracks import LayerEval, hold_frame
    L = next(l for l in json.loads(Path(scene_path).read_text(encoding="utf-8"))["layers"] if l["id"] == plan.id)
    ev = LayerEval(rig, L)
    gs = [hold_frame(L, f) for f in plan.frames]
    P = ev.params_at(sorted(set(gs)))
    idx = {g: i for i, g in enumerate(sorted(set(gs)))}
    out = real.copy()
    for k, (f, g) in enumerate(zip(plan.frames, gs)):
        dr = ev.draw_at(g)
        with torch.no_grad():
            buf, _ = raster_layer(rig, P[idx[g]][None], to_screen(dr["transform"])[None].float(), 1280, 720, aa=False)
            img = compose_one(buf, dr)[0]
        c = img[..., :3] * 255 + torch.tensor(real[k], device=img.device).float() * (1 - img[..., 3:4])
        out[k] = c.round().clamp(0, 255).byte().cpu().numpy()
    return out


def save_preview(fitter: Fitter, path: Path, gs: list[int]) -> None:
    """预览：参考 | 渲染合成到参考（640×360）。"""
    t = fitter.t
    st = Stage("preview", 0, 640, 360, {})
    ref, _ = fitter.ref_at(640, 360)
    with torch.no_grad():
        after = t.physics(t.pre_params())
        P = t.final_params(after, gs)
        TS = t.to_screen(t.tf_at(gs), 0.5)
        img, _ = t.render(P, TS, 0.5, 640, 360, aa=False)
    rows = []
    for j, g in enumerate(gs):
        r = ref[g - t.plan.fin]
        c = img[j, ..., :3] * 255 + r * (1 - img[j, ..., 3:4])
        rows.append(torch.cat([r, c], 1))
    Image.fromarray(torch.cat(rows, 0).clamp(0, 255).byte().cpu().numpy()).save(path, quality=88)
    del st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="s24")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="迭代数 ÷4（调试）")
    ap.add_argument("--decouple", type=int, default=1, help="身体由独立输入 BodyX/Y/Z 驱动（vts.ts 的 MODELX.decouple）")
    ap.add_argument("--smooth-c", type=float, default=0.15)
    ap.add_argument("--mem-gb", type=float, default=12.0)
    ap.add_argument("--synth", default=None, help="诊断：合成基准——用该场景里本镜头的图层经 GPU 前向渲染、合成到原视频背景上，代替参考帧")
    args = ap.parse_args()
    out = Path(args.out or REPO / f".data/gpu/fit/{args.shot}")
    (out / "preview").mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit.log", "w", encoding="utf-8")

    def log(s: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    T: dict[str, float] = {}
    t0 = time.time()
    rig = Rig(phys_compile=True)
    plan = make_plan(args.shot)
    refs = Refs(plan)
    if args.synth:
        refs.full = synth_frames(rig, plan, args.synth, refs.full)
        log(f"合成基准：参考帧换成 {args.synth} 的 {plan.id} 经 GPU 前向的渲染")
    finalize_nodes(plan, refs.full)  # 画面重复检测 + cadence → 节点与 frameHold
    vts = Vts(VTUBE, {k: float(v) for k, v in zip(rig.param_ids, rig.pdef)}, decouple=bool(args.decouple))
    traj = Traj(rig, plan, vts, smooth_c=args.smooth_c)
    fitter = Fitter(traj, refs, log)
    T["装载（模型、贴图、参考帧）"] = time.time() - t0
    log(f"{plan.id}: [{plan.fin}, {plan.fout})，节点 {len(plan.nodes)}，变换关键帧 {len(plan.keys)}，锚点 {len(plan.anchors)}，"
        f"frameHold {plan.holds}，输入 {traj.inputs_fit}，post 组 {len(traj.post_groups)}")

    t0 = time.time()
    fitter.init_from_anchors()
    # 物理编译（torch.compile，首次调用触发；FX 缓存热时约 45 s）
    with torch.no_grad():
        traj.physics(traj.pre_params())
    T["初值 + 物理编译"] = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    S = stages(args.quick)
    t0 = time.time()
    lean = fitter.search_lean(values=(0.0, -10.0, -20.0, -25.0, -30.0), st=Stage("前倾搜索", 0, 320, 180, {}, w_ssim=0.3, w_blur=1.0, blur_sig=(16.0, 8.0), chunk=12, node_stride=3))
    log(f"前倾初值 {lean}")
    T["前倾多假设"] = time.time() - t0
    previews = [plan.nodes[i] for i in np.linspace(0, len(plan.nodes) - 1, 5).astype(int)]
    save_preview(fitter, out / "preview/0_init.jpg", previews)
    for k, st in enumerate(S):
        t0 = time.time()
        if st.outer > 0:
            fitter.run_flow_stage(st, st.outer, st.inner)
        else:
            fitter.run_stage(st)
        T[st.name] = time.time() - t0
        save_preview(fitter, out / f"preview/{k + 1}_{st.name}.jpg", previews)
    # 外观常量（1280×720，缓存图层缓冲，只对滤镜求导）
    t0 = time.time()
    ap = fitter.fit_appearance()
    log(f"外观：resample {ap['resample']}，SSIM@360 {ap['ssim']:.4f}")
    T["A-外观常量"] = time.time() - t0
    save_preview(fitter, out / "preview/9_final.jpg", previews)
    peak = torch.cuda.max_memory_allocated() / 2**30
    # 导出
    t0 = time.time()
    layer = build_layer(traj)
    (out / "layer.json").write_text(json.dumps(layer, ensure_ascii=False))
    T["导出"] = time.time() - t0
    total = sum(T.values())
    log("耗时：" + "，".join(f"{k} {v:.1f}s" for k, v in T.items()) + f"；合计 {total:.1f}s；显存峰值 {peak:.2f} GB")
    log("前向细分：" + "，".join(f"{k} {v:.1f}s" for k, v in fitter.timing.items()))
    (out / "fit.json").write_text(json.dumps({
        "shot": args.shot, "timing": T, "totalSec": total, "forwardTiming": fitter.timing, "peakGB": peak,
        "resample": traj.resample, "consts": {k: (v.tolist() if v.dim() else float(v)) for k, v in traj.consts.items()},
        "history": fitter.history, "decouple": bool(args.decouple), "smoothC": args.smooth_c,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
