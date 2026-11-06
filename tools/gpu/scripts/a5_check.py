"""A5：导出一致性对拍——GPU 前向按场景图层（tracks.py 的规范求值）渲染 1280×720，与 runtime（pnpm render）的同一帧比较。

    tools/.venv/bin/python -I tools/gpu/scripts/a5_check.py --scene .data/gpu/fit/s24/scene.json --layer s24.main \
        --renders .data/gpu/fit/s24/render --frames 2370,2391,2400,2413,2425,2441 [--json out.json]

口径同 a3_compare：两边都按 alpha 合成到参考视频同一帧，遮罩 = dilate(α_runtime > 0.1, 半径 3)（不扣排除区，比较的是两个渲染器），
RGB 高斯 σ = 1.5 的 SSIM@360（INTER_AREA 降采样）；另报预乘 RGBA8 的平均 / 最大像素误差，以及两边各自对参考帧的 metric 口径 SSIM@360。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fit.rig import REPO, Rig  # noqa: E402
from fit.render import compose_one, raster_layer, to_screen  # noqa: E402
from fit.tracks import LayerEval, hold_frame  # noqa: E402

sys.path.insert(0, str(REPO / "tools/qa"))
import metric  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--layer", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    L = next(l for l in scene["layers"] if l["id"] == args.layer)
    act = [l for l in scene["layers"] if l["in"] <= int(args.frames.split(",")[0]) < l["out"]]
    if len(act) != 1:
        raise SystemExit(f"只支持单图层帧（该帧有 {len(act)} 个图层）")
    shot = next(s for s in metric.load_shots()["shots"] if s["start"] <= L["in"] < s["end"])
    rig = Rig()
    ev = LayerEval(rig, L)
    frames = [int(x) for x in args.frames.split(",")]
    gs = [hold_frame(L, f) for f in frames]
    P = ev.params_at(gs)
    rows = []
    for f, g, p in zip(frames, gs, P):
        dr = ev.draw_at(g)
        with torch.no_grad():
            buf, _ = raster_layer(rig, p[None], to_screen(dr["transform"])[None].float(), 1280, 720, aa=False)
            img = compose_one(buf, dr)[0]
        ours = torch.floor(img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()  # 预乘 RGBA8
        rgb, alpha = metric.load_render(Path(args.renders) / f"frame_{f:06d}.png")
        ref = metric.decode_ref(f, f + 1)[0].astype(np.float64)
        rt_pm = np.floor(np.concatenate([rgb * alpha[..., None], alpha[..., None] * 255], -1) + 0.5)
        a_o = ours[..., 3] / 255.0
        comp_o = ours[..., :3].astype(np.float64) + ref * (1 - a_o[..., None])
        comp_r = rgb * alpha[..., None] + ref * (1 - alpha[..., None])
        mask = cv2.dilate((alpha > 0.1).astype(np.uint8), metric.DISK) > 0
        m360 = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
        S = metric.ssim_map(cv2.resize(comp_o, (640, 360), interpolation=cv2.INTER_AREA), cv2.resize(comp_r, (640, 360), interpolation=cv2.INTER_AREA))
        err = np.abs(ours.astype(np.float64) - rt_pm)
        s_gpu = metric.frame_metric(ours[..., :3] / np.maximum(a_o[..., None], 1e-9) * (a_o[..., None] > 0), a_o, ref.astype(np.uint8), shot, f)["ssim360"]
        s_rt = metric.frame_metric(rgb, alpha, ref.astype(np.uint8), shot, f)["ssim360"]
        r = {"frame": f, "g": g, "ssim360_gpu_vs_runtime": float(S[m360].mean()), "meanErr": float(err.mean()), "maxErr": float(err.max()),
             "metric_gpu": s_gpu, "metric_runtime": s_rt}
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    worst = min(r["ssim360_gpu_vs_runtime"] for r in rows)
    print(f"GPU 前向 vs runtime：SSIM@360 最小 {worst:.5f}，均值 {np.mean([r['ssim360_gpu_vs_runtime'] for r in rows]):.5f}"
          f"（门槛 > 0.99：{'通过' if worst > 0.99 else '不通过'}）")
    if args.json:
        Path(args.json).write_text(json.dumps({"frames": rows, "min": worst}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
