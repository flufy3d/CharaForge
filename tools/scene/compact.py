"""把 scene.json 里逐帧烘焙（baked）的曲线压成稀疏关键帧（keys），并顺手做时间平滑。

做法：
  1. postPhysics 参数轨：先 5 点中值去单帧尖峰，再做 σ=1.5 帧的高斯低通（两端 3 帧渐隐，
     保持镜头入/出帧不变）；这些是拟合器逐帧精修出来的偏移，闪烁主要来自这里。
  2. pre 参数轨：只做「孤立单帧异常」替换，不动眨眼/口型这类正常快速动作。
  3. 变换、clip、opacity、模糊：不额外平滑，避免动到镜头运动。
  4. 所有曲线用 RDP（垂距）抽成稀疏 keys；常量参数只留 1 个 key。
     如果某条曲线 keys 反而比 baked 大（短曲线 / 高频曲线），就保留 baked —— 逐条取较小表示。
  5. 数值按类别保留 2～5 位小数，进一步压缩文本体积。

用法（在仓库根目录）：
  python tools/scene/compact.py scenes/deepseek-chan/scene.json --out scenes/deepseek-chan/scene.compact.json
  python tools/scene/compact.py scenes/deepseek-chan/scene.json --in-place

默认参数就是给 scenes/deepseek-chan 调的；别的场景可以先 --out 到新文件再对比。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# 每类曲线的抽帧容差（绝对误差，不是相对值）与保留小数位。
# 变换/裁切单位是屏幕像素，参数是 Cubism 参数单位（角度/0..1 比例等）。
TOL = {
    ("param", "postPhysics"): 0.010,
    ("param", "pre"): 0.005,
    ("transform", "x"): 0.20,
    ("transform", "y"): 0.20,
    ("transform", "scale"): 0.0010,
    ("transform", "scaleX"): 0.0010,
    ("transform", "scaleY"): 0.0010,
    ("transform", "rotation"): 0.020,
    ("clip", "x"): 0.50,
    ("clip", "y"): 0.50,
    ("clip", "width"): 0.50,
    ("clip", "height"): 0.50,
    ("clip", "radius"): 0.50,
    ("clip", "feather"): 0.50,
    ("opacity", "o"): 0.002,
    ("filter", "blur"): 0.020,
}
ND = {
    ("param", "postPhysics"): 4,
    ("param", "pre"): 4,
    ("transform", "x"): 3,
    ("transform", "y"): 3,
    ("transform", "scale"): 5,
    ("transform", "scaleX"): 5,
    ("transform", "scaleY"): 5,
    ("transform", "rotation"): 3,
    ("clip", "x"): 2,
    ("clip", "y"): 2,
    ("clip", "width"): 2,
    ("clip", "height"): 2,
    ("clip", "radius"): 2,
    ("clip", "feather"): 2,
    ("opacity", "o"): 4,
    ("filter", "blur"): 3,
}


def jbytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def tol_for(typ: str, stage: str | None, axis: str) -> float:
    return TOL.get((typ, stage), TOL.get((typ, axis), 0.01))


def nd_for(typ: str, stage: str | None, axis: str) -> int:
    return ND.get((typ, stage), ND.get((typ, axis), 4))


def quant(x: float, nd: int) -> float:
    y = round(float(x), nd)
    return 0.0 if y == 0 else y


def median5(values: list[float]) -> list[float]:
    """5 点中值（两端 1 帧用 3 点中值，首末帧不动）。"""
    n = len(values)
    out = list(values)
    if n < 3:
        return out
    for i in range(1, n - 1):
        if i == 1 or i == n - 2:
            window = sorted((values[i - 1], values[i], values[i + 1]))
            out[i] = window[1]
        else:
            window = sorted(values[i - 2 : i + 3])
            out[i] = window[2]
    return out


def despike_isolated(values: list[float], floor: float, rel: float) -> tuple[list[float], int]:
    """只替换「涨上去又立刻回来」的单帧孤立点，不碰单调的快速变化。"""
    n = len(values)
    out = list(values)
    if n < 5:
        return out, 0
    span = max(values) - min(values)
    threshold = max(floor, rel * span)
    changed = 0
    for i in range(1, n - 1):
        prev = values[i] - values[i - 1]
        nxt = values[i + 1] - values[i]
        if prev * nxt >= 0:
            continue
        returned = abs(values[i + 1] - values[i - 1])
        if returned > 0.5 * (abs(prev) + abs(nxt)):
            continue
        mid = sorted((values[i - 1], values[i], values[i + 1]))[1]
        if abs(values[i] - mid) > threshold:
            out[i] = mid
            changed += 1
    return out, changed


def gaussian_smooth(values: list[float], sigma: float, edge: int = 3) -> list[float]:
    """高斯低通；两端 edge 帧内把平滑量渐隐为 0，保证入/出帧值不变。"""
    n = len(values)
    if n < 3 or sigma <= 0:
        return list(values)
    radius = int(math.ceil(3 * sigma))
    kernel = [math.exp(-(k * k) / (2 * sigma * sigma)) for k in range(-radius, radius + 1)]
    total = sum(kernel)
    kernel = [k / total for k in kernel]
    padded = [values[0]] * radius + list(values) + [values[-1]] * radius
    smoothed = []
    for i in range(n):
        acc = 0.0
        base = i  # padded 的下标 i 对应原序列 i（前补 radius）
        for j, weight in enumerate(kernel):
            acc += padded[base + j] * weight
        smoothed.append(acc)
    if edge > 0 and n > 2 * edge:
        for i in range(n):
            w = min(1.0, i / edge, (n - 1 - i) / edge)
            smoothed[i] = values[i] + (smoothed[i] - values[i]) * w
    return smoothed


def rdp_keep(values: list[float], tol: float) -> list[int]:
    """RDP 垂距抽帧：返回需要保留的下标（含首末）。values 已按整数帧等距排列。"""
    n = len(values)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        va, vb = values[a], values[b]
        span = b - a
        best = -1.0
        best_i = -1
        for i in range(a + 1, b):
            pred = va + (vb - va) * ((i - a) / span)
            err = abs(values[i] - pred)
            if err > best:
                best = err
                best_i = i
        if best > tol:
            keep.add(best_i)
            stack.append((a, best_i))
            stack.append((best_i, b))
    return sorted(keep)


def keys_repr(values: list[float], start: int, tol: float, nd: int) -> dict:
    quantized = [quant(v, nd) for v in values]
    n = len(quantized)
    if n == 0:
        return {"keys": [{"t": start, "v": 0}]}
    if n == 1 or max(quantized) - min(quantized) <= tol:
        return {"keys": [{"t": start, "v": quant(quantized[0], nd)}]}
    keep = rdp_keep(quantized, tol)
    return {"keys": [{"t": start + i, "v": quantized[i]} for i in keep]}


def baked_repr(values: list[float], start: int, nd: int) -> dict:
    return {"baked": {"start": start, "values": [quant(v, nd) for v in values]}}


def eval_curve(curve: object, start: int, count: int) -> list[float]:
    if isinstance(curve, (int, float)):
        return [float(curve)] * count
    assert isinstance(curve, dict)
    if "baked" in curve:
        b = curve["baked"]
        vals = b["values"]
        return [float(vals[min(max(i, 0), len(vals) - 1)]) for i in range(count)]
    keys = curve["keys"]
    out: list[float] = []
    j = 0
    for i in range(count):
        t = start + i
        while j + 1 < len(keys) and keys[j + 1]["t"] <= t:
            j += 1
        if t <= keys[0]["t"]:
            out.append(float(keys[0]["v"]))
            continue
        if t >= keys[-1]["t"]:
            out.append(float(keys[-1]["v"]))
            continue
        a, b = keys[j], keys[j + 1]
        if b["t"] == a["t"]:
            out.append(float(a["v"]))
        else:
            w = (t - a["t"]) / (b["t"] - a["t"])
            out.append(float(a["v"]) + (float(b["v"]) - float(a["v"])) * w)
    return out


def walk_baked(layer: dict):
    """遍历图层里的所有 baked 曲线：(容器, 键, 类型, stage, axis, start, values)。

    只处理数值曲线；向量曲线（如 color.multiply）原样保留。
    """
    def numeric(values: object) -> bool:
        return isinstance(values, list) and (not values or isinstance(values[0], (int, float)))

    for track in layer.get("params") or []:
        curve = track.get("curve")
        if isinstance(curve, dict) and "baked" in curve:
            b = curve["baked"]
            if numeric(b["values"]):
                yield track, "curve", "param", track.get("stage", "pre"), track.get("id", ""), b["start"], b["values"]
    for prop in ("transform", "clip"):
        obj = layer.get(prop)
        if not isinstance(obj, dict):
            continue
        for axis, anim in obj.items():
            if isinstance(anim, dict) and "baked" in anim:
                b = anim["baked"]
                if numeric(b["values"]):
                    yield obj, axis, prop, None, axis, b["start"], b["values"]
    opacity = layer.get("opacity")
    if isinstance(opacity, dict) and "baked" in opacity:
        b = opacity["baked"]
        if numeric(b["values"]):
            yield layer, "opacity", "opacity", None, "o", b["start"], b["values"]
    for flt in layer.get("filters") or []:
        if not isinstance(flt, dict):
            continue
        for axis in ("sigma", "factor", "brightness", "contrast", "saturation"):
            anim = flt.get(axis)
            if isinstance(anim, dict) and "baked" in anim:
                b = anim["baked"]
                if numeric(b["values"]):
                    yield flt, axis, "filter", flt.get("type"), axis, b["start"], b["values"]


def compact_scene(
    scene: dict,
    *,
    sigma: float,
    median: bool,
    despike_pre: bool,
    tolerance_scale: float,
    hybrid: bool,
    dedupe_provenance: bool,
) -> tuple[dict, dict]:
    stats: dict[tuple[str, str | None], dict[str, float]] = {}
    smooth_dev_max = 0.0
    smooth_changed = 0
    pre_despiked = 0
    provenance_dropped = 0
    provenance_bytes = 0

    for layer in scene.get("layers") or []:
        if dedupe_provenance:
            seen: set[str] = set()
            for track in layer.get("params") or []:
                prov = track.get("provenance")
                if prov is None:
                    continue
                key = json.dumps(prov, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if key in seen:
                    provenance_bytes += jbytes(prov)
                    provenance_dropped += 1
                    del track["provenance"]
                else:
                    seen.add(key)

    for layer in scene.get("layers") or []:
        for container, key, typ, stage, axis, start, values in list(walk_baked(layer)):
            original = [float(v) for v in values]
            vals = list(original)

            if typ == "param" and stage == "postPhysics":
                if median:
                    med = median5(vals)
                    smooth_changed += sum(1 for a, b in zip(med, vals) if a != b)
                    vals = med
                if sigma > 0:
                    vals = gaussian_smooth(vals, sigma)
            elif typ == "param" and stage == "pre" and despike_pre:
                vals, changed = despike_isolated(vals, floor=0.03, rel=0.02)
                pre_despiked += changed

            tol = tol_for(typ, stage, axis) * tolerance_scale
            nd = nd_for(typ, stage, axis)
            curve = keys_repr(vals, int(start), tol, nd)
            if hybrid and len(vals) >= 4:
                alt = baked_repr(vals, int(start), nd)
                if jbytes(alt) < jbytes(curve):
                    curve = alt

            # 非参数属性允许常量；参数轨必须是 CurveNumber。
            if typ != "param" and (not vals or max(vals) - min(vals) <= tol):
                curve = quant(sum(vals) / len(vals), nd) if vals else 0.0

            container[key] = curve

            reconstructed = eval_curve(curve, int(start), len(original))
            errors = [abs(a - b) for a, b in zip(reconstructed, original)]
            simplify_errors = [abs(a - b) for a, b in zip(reconstructed, vals)]
            entry = stats.setdefault(
                (typ, stage),
                {
                    "curves": 0,
                    "values": 0,
                    "keys": 0,
                    "baked": 0,
                    "simp_max": 0.0,
                    "simp_sq": 0.0,
                    "smooth_max": 0.0,
                },
            )
            entry["curves"] += 1
            entry["values"] += len(original)
            if isinstance(curve, dict) and "keys" in curve:
                entry["keys"] += len(curve["keys"])
            elif isinstance(curve, dict) and "baked" in curve:
                entry["baked"] += 1
            entry["simp_max"] = max(entry["simp_max"], max(simplify_errors) if simplify_errors else 0.0)
            entry["simp_sq"] += sum(e * e for e in simplify_errors)

            if typ == "param" and stage == "postPhysics" and sigma > 0:
                smooth_max = max((abs(a - b) for a, b in zip(vals, original)), default=0.0)
                entry["smooth_max"] = max(entry["smooth_max"], smooth_max)
                smooth_dev_max = max(smooth_dev_max, smooth_max)
            elif typ == "param" and stage == "pre" and despike_pre:
                entry["smooth_max"] = max(entry["smooth_max"], max(errors) if errors else 0.0)

    summary = {
        "stats": stats,
        "smooth_dev_max": smooth_dev_max,
        "smooth_changed": smooth_changed,
        "pre_despiked": pre_despiked,
        "provenance_dropped": provenance_dropped,
        "provenance_bytes": provenance_bytes,
    }
    return scene, summary


def count_curve_kinds(scene: dict) -> tuple[int, int]:
    """统计 keys / baked 曲线数量，用于提示「输入已经被压过」。"""
    keys = baked = 0
    for layer in scene.get("layers") or []:
        curves = [track.get("curve") for track in layer.get("params") or []]
        for prop in ("transform", "clip"):
            obj = layer.get(prop)
            if isinstance(obj, dict):
                curves.extend(obj.values())
        curves.append(layer.get("opacity"))
        for flt in layer.get("filters") or []:
            if isinstance(flt, dict):
                curves.extend(flt.values())
        for curve in curves:
            if isinstance(curve, dict):
                if "keys" in curve:
                    keys += 1
                elif "baked" in curve:
                    baked += 1
    return keys, baked


def print_report(source: Path, raw_in: int, raw_out: int, summary: dict) -> None:
    print(f"输入 {source}  {raw_in / 1e6:.2f} MB")
    print(f"输出 {raw_out / 1e6:.2f} MB（{100 * raw_out / raw_in:.0f}%）")
    print(f"postPhysics 平滑：最大改动 {summary['smooth_dev_max']:.3f}，中值去尖峰 {summary['smooth_changed']} 个点；"
          f"pre 孤立点替换 {summary['pre_despiked']} 个")
    if summary["provenance_dropped"]:
        print(f"重复 provenance：去掉 {summary['provenance_dropped']} 份（约 {summary['provenance_bytes'] / 1e6:.2f} MB，"
              f"每图层每类保留 1 份）")
    print(f"{'类别':<22}{'曲线':>7}{'原始值':>9}{'keys':>9}{'baked':>7}{'平滑Δmax':>10}{'抽帧Δmax':>10}{'抽帧RMS':>9}")
    for (typ, stage), s in sorted(summary["stats"].items()):
        name = f"{typ}/{stage}" if stage else typ
        rms = math.sqrt(s["simp_sq"] / max(s["values"], 1))
        print(f"{name:<22}{s['curves']:>7}{s['values']:>9}{s['keys']:>9}{s['baked']:>7}"
              f"{s['smooth_max']:>10.4f}{s['simp_max']:>10.4f}{rms:>9.5f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把 scene.json 的 baked 曲线压成稀疏 keys，并平滑 postPhysics 抖动")
    parser.add_argument("scene", help="输入 scene.json")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", help="输出路径")
    target.add_argument("--in-place", action="store_true", help="原子替换输入文件")
    parser.add_argument("--sigma", type=float, default=1.5, help="postPhysics 高斯平滑 σ（帧，0 关闭；缺省 1.5）")
    parser.add_argument("--no-median", action="store_true", help="关闭 postPhysics 5 点中值去尖峰")
    parser.add_argument("--no-despike-pre", action="store_true", help="关闭 pre 参数孤立点替换")
    parser.add_argument("--tolerance-scale", type=float, default=1.0, help="抽帧容差倍率（缺省 1.0）")
    parser.add_argument("--no-hybrid", action="store_true", help="不保留 baked；所有曲线都输出 keys")
    parser.add_argument("--keep-param-provenance", action="store_true", help="保留每条参数轨的 provenance（默认每图层每类只留 1 份）")
    parser.add_argument("--force", action="store_true", help="输入已经是 keys 压缩版时仍继续（只处理剩余 baked 曲线）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.scene)
    text = source.read_text(encoding="utf-8")
    scene = json.loads(text)

    keys_count, baked_count = count_curve_kinds(scene)
    if keys_count > baked_count and not args.force:
        print(
            f"输入里已有 {keys_count} 条 keys、{baked_count} 条 baked，看起来已经是压缩后的场景。\n"
            "本次只处理 baked 曲线，重复跑会二次平滑。要改平滑/容差，请先恢复原始（烘焙）场景再跑；\n"
            "确实要继续就加 --force。",
            file=sys.stderr,
        )
        return 2

    scene, summary = compact_scene(
        scene,
        sigma=args.sigma,
        median=not args.no_median,
        despike_pre=not args.no_despike_pre,
        tolerance_scale=args.tolerance_scale,
        hybrid=not args.no_hybrid,
        dedupe_provenance=not args.keep_param_provenance,
    )

    out_text = json.dumps(scene, ensure_ascii=False, separators=(",", ":"))
    out_path = source if args.in_place else Path(args.out)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(out_text, encoding="utf-8")
    os.replace(tmp, out_path)

    if not args.quiet:
        print_report(source, len(text.encode("utf-8")), len(out_text.encode("utf-8")), summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
