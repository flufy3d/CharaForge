"""A3 对拍：用 l2d.raster + l2d.composite 重渲 a3_dump 的帧，与 runtime 的输出逐帧比较。

  tools/.venv/bin/python -I tools/gpu/scripts/a3_compare.py --dump .data/gpu/frames/main [--dump ...] \
      --out .data/gpu/a3_compare.json [--save-diff ~/cf-scratch/a3/diff]

指标（每帧）：
  ssim360 / ssim720  口径同 tools/qa/metric.py：两者都按 alpha 合成到参考视频同一帧上（预乘：C = rgb + ref·(1 − α)），
                     角色遮罩 M = dilate(α_runtime > 0.1, 半径 3 圆盘)；SSIM@360 = INTER_AREA 降到 640×360、
                     RGB 三通道高斯（σ = 1.5）SSIM 图平均后在降采样遮罩（> 0.5）上取均值；SSIM@720 不降采样。
                     （不扣除 shots.json 的排除区：这里比的是两个渲染器，不是和原片比。）
  maxErr / meanErr   预乘 RGBA8 的逐通道绝对误差（0–255）：全画面最大值、全画面均值、遮罩内均值
  layer*             有 --layers 数据时：每个图层的 Cubism 绘制结果、滤镜之后，各自与 runtime 的缓冲比较
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/gpu"))
sys.path.insert(0, str(REPO / "tools/qa"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import metric  # noqa: E402  tools/qa/metric.py
from l2d import composite as C  # noqa: E402
from l2d.dumpio import Dump  # noqa: E402
from l2d.raster import LayerInput, load_rasterizer  # noqa: E402

# 主对拍集的帧类别（scenes/deepseek-chan/scene.json；挑帧依据见 tools/gpu/docs/raster.md）
CATEGORIES = {
    "s24 单实例": [2370, 2385, 2400, 2415, 2430, 2441],
    "blur/景深": [1520, 1712, 2140, 2306, 2650, 2700, 3490],
    "resample": [800, 1000, 1750, 3000, 3120, 3700],
    "clip": [1360, 1380, 1398, 1450, 520, 600, 640, 300, 1100, 2500, 2620, 2660],
    "多实例": [760, 900, 1300, 1500, 1880],
    "不透明度渐变": [1356, 1362, 1730, 1760, 1780],
    "scaleX": [1229, 1232, 1236],
}
VARIANT_CATEGORIES = {
    "flipX + 相机": [2380, 2420],
    "blend screen": [800, 900],
    "blend add": [1850, 1890],
    "blend multiply": [720, 770],
    "motionBlur": [1530, 1560],
    "负 scaleX + 非等比": [3110, 3130],
    "resample bicubic": [1000, 1040],
}


def decode_frames(frames: list[int]) -> dict[int, np.ndarray]:
    """一次 ffmpeg 解码多个帧（颜色转换与 metric.py / refvideo.ts 一致）。"""
    fs = sorted(set(frames))
    sel = "+".join(f"eq(n\\,{f})" for f in fs)
    vf = (f"select='{sel}',"
          "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(metric.VIDEO), "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, metric.H, metric.W, 3)
    assert len(arr) == len(fs), f"解码帧数 {len(arr)} ≠ {len(fs)}"
    return {f: arr[i] for i, f in enumerate(fs)}


def to_u8(x: torch.Tensor) -> np.ndarray:
    return torch.floor(x.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()


def frame_metrics(ours: np.ndarray, ref: np.ndarray, video: np.ndarray) -> dict:
    """ours/ref：预乘 RGBA uint8 [H,W,4]；video：RGB uint8。"""
    a_r = ref[..., 3].astype(np.float64) / 255
    a_o = ours[..., 3].astype(np.float64) / 255
    bg = video.astype(np.float64)
    c_r = ref[..., :3].astype(np.float64) + bg * (1 - a_r[..., None])
    c_o = ours[..., :3].astype(np.float64) + bg * (1 - a_o[..., None])
    mask = cv2.dilate((a_r > 0.1).astype(np.uint8), metric.DISK) > 0
    d = np.abs(ours.astype(np.int32) - ref.astype(np.int32))
    out = {"maskArea": int(mask.sum()), "maxErr": int(d.max()), "meanErr": float(d.mean()),
           "meanErrMask": float(d[mask].mean()) if mask.any() else None,
           "px>2": int((d.max(-1) > 2).sum()), "px>10": int((d.max(-1) > 10).sum())}
    if not mask.any():
        out["ssim360"] = out["ssim720"] = None
        return out
    c360 = cv2.resize(c_o, (640, 360), interpolation=cv2.INTER_AREA)
    r360 = cv2.resize(c_r, (640, 360), interpolation=cv2.INTER_AREA)
    m360 = cv2.resize(mask.astype(np.float64), (640, 360), interpolation=cv2.INTER_AREA) > 0.5
    out["ssim360"] = float(metric.ssim_map(c360, r360)[m360].mean()) if m360.any() else None
    out["ssim720"] = float(metric.ssim_map(c_o, c_r)[mask].mean())
    return out


def render_frame(dump: Dump, fr: dict, rasters: dict, quantize=True):
    W, H = fr["width"], fr["height"]
    layers = []
    per_layer = {}
    for L in fr["layers"]:
        if not L["visible"]:
            continue
        if L["model"] not in rasters:
            rasters[L["model"]] = load_rasterizer(dump, L["model"])
        R = rasters[L["model"]]
        with torch.no_grad():
            buf = R.render(LayerInput.from_dump([L]), W, H, aa=False, quantize=quantize)
        layers.append((buf, L["draw"]))
        per_layer[L["id"]] = buf
    with torch.no_grad():
        comp = C.composite(layers, H, W, dump.manifest["composition"].get("background"), quantize=quantize, B=1)
    return comp, per_layer, layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--out", default=str(REPO / ".data/gpu/a3_compare.json"))
    ap.add_argument("--frames", help="只比这些帧（逗号分隔）")
    ap.add_argument("--save-diff", help="把 ours|runtime|差异×8 的拼图存到该目录")
    args = ap.parse_args()

    report = {"frames": []}
    rasters: dict = {}
    for droot in args.dump:
        dump = Dump(droot)
        rasters.clear()
        torch.cuda.empty_cache()
        frames = dump.frames if not args.frames else [f for f in map(int, args.frames.split(",")) if f in dump.frames]
        cats = VARIANT_CATEGORIES if Path(droot).name == "variants" else CATEGORIES
        video = decode_frames(frames)
        for f in frames:
            t0 = time.time()
            fr = dump.frame(f)
            comp, per_layer, layers = render_frame(dump, fr, rasters)
            torch.cuda.synchronize()
            dt = time.time() - t0
            ours = to_u8(comp[0])
            ref = dump.ref_premul(f)
            m = frame_metrics(ours, ref, video[f])
            m.update({"dump": Path(droot).name, "frame": f, "layers": [L["id"] for L in fr["layers"] if L["visible"]],
                      "category": [k for k, v in cats.items() if f in v], "seconds": round(dt, 3)})
            # 分段：图层缓冲（Cubism 绘制）与滤镜之后
            seg = {}
            for L in fr["layers"]:
                if not L["visible"]:
                    continue
                bufs = dump.layer_buffers(f, L["id"])
                if bufs is None:
                    continue
                o = to_u8(per_layer[L["id"]][0])
                d1 = np.abs(o.astype(np.int32) - bufs[0].astype(np.int32))
                with torch.no_grad():
                    src = C.apply_filters(C.quantize8(per_layer[L["id"]]), L["draw"]["filters"])
                    # 滤镜单独对拍：输入用 runtime 的图层缓冲，排除光栅化误差
                    rt_in = torch.tensor(bufs[0], device="cuda", dtype=torch.float32)[None] / 255
                    src_rt = C.apply_filters(rt_in, L["draw"]["filters"])
                d2 = np.abs(to_u8(src[0]).astype(np.int32) - bufs[1].astype(np.int32))
                d3 = np.abs(to_u8(src_rt[0]).astype(np.int32) - bufs[1].astype(np.int32))
                seg[L["id"]] = {"raster": {"max": int(d1.max()), "mean": float(d1.mean()), "px>10": int((d1.max(-1) > 10).sum())},
                                "filtered": {"max": int(d2.max()), "mean": float(d2.mean())},
                                "filtersOnly": {"max": int(d3.max()), "mean": float(d3.mean())}}
            m["segments"] = seg
            report["frames"].append(m)
            print(f"{Path(droot).name} f{f} {m['category']} ssim360={m['ssim360']:.5f} ssim720={m['ssim720']:.5f} "
                  f"max={m['maxErr']} mean={m['meanErr']:.3f} px>10={m['px>10']} ({dt:.2f}s) "
                  + " ".join(f"{k}:r{v['raster']['mean']:.3f}/{v['raster']['max']} f{v['filtersOnly']['max']}" for k, v in seg.items()), flush=True)
            if args.save_diff:
                out_dir = Path(args.save_diff)
                out_dir.mkdir(parents=True, exist_ok=True)
                d = np.clip(np.abs(ours.astype(np.int32) - ref.astype(np.int32)) * 8, 0, 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / f"{Path(droot).name}_{f}.png"), np.concatenate([ours, ref, d], 1)[..., [2, 1, 0]])
    # 汇总
    rows = report["frames"]
    allcats = list(CATEGORIES) + list(VARIANT_CATEGORIES)
    summ = {}
    for cat in allcats:
        sub = [r for r in rows if cat in r["category"] and r["ssim360"] is not None]
        if not sub:
            continue
        summ[cat] = {"frames": len(sub),
                     "ssim360Min": min(r["ssim360"] for r in sub), "ssim360Mean": float(np.mean([r["ssim360"] for r in sub])),
                     "ssim720Min": min(r["ssim720"] for r in sub), "ssim720Mean": float(np.mean([r["ssim720"] for r in sub])),
                     "maxErr": max(r["maxErr"] for r in sub), "meanErr": float(np.mean([r["meanErr"] for r in sub]))}
    valid = [r for r in rows if r["ssim360"] is not None]
    worst = min(valid, key=lambda r: r["ssim360"])
    report["summary"] = {"categories": summ, "frames": len(valid),
                         "ssim360Min": worst["ssim360"], "worst": {"dump": worst["dump"], "frame": worst["frame"]},
                         "ssim360Mean": float(np.mean([r["ssim360"] for r in valid])),
                         "pass099": all(r["ssim360"] > 0.99 for r in valid)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
