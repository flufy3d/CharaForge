"""全片质检报告：逐帧 CSV、逐镜头表、全片汇总与最差镜头排行、静态 HTML、全片差异视频。

指标与帧分类全部调用 tools/qa/metric.py（唯一实现），本文件只做汇总和呈现。用法见 tools/qa/README.md。

  tools/.venv/bin/python -I tools/qa/report.py --render --shots s15,s24,s43
  tools/.venv/bin/python -I tools/qa/report.py --renders .data/qa/report/renders --layout heat

输出（--out，缺省 .data/qa/report/）：
  frames.csv   逐帧：帧号、镜头、是否拟合、类别、判定依据、SSIM@360、SSIM@720、覆盖率……
  shots.csv    逐镜头：各类帧数、有效帧数、均值、最小值、5% 分位、≥0.7 比例、不达标帧数、状态
  report.json  机器可读（每镜头 summary 与 metric.py score 的 summary 同结构，另有 perFrame、overall、ranking）
  index.html   静态报告（无外部依赖；SSIM 曲线为内联 SVG，缩略图在 thumbs/）
  diff_<layout>.mp4  差异视频（60fps，带原音频；--no-video 跳过）
"""

from __future__ import annotations

import argparse
import csv
import heapq
import html
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metric  # noqa: E402  （指标唯一实现）

REPO = metric.REPO
W, H, FPS = metric.W, metric.H, metric.FPS
SCENE = REPO / "scenes/deepseek-chan/scene.json"
OUT = REPO / ".data/qa/report"
THR = metric.THRESHOLD_360
CLS_ZH = {"normal": "普通", "visual": "目测", "effect": "特效"}
LAYOUTS = ("sbs", "overlay", "heat", "quad")
FONT_PATHS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/usr/share/fonts/truetype/arphic/uming.ttc"]


def log(*a) -> None:
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------- 输入 ----------------

def layer_spans(scene: dict) -> list[tuple[int, int]]:
    """模型图层的 [in, out) 区间（缺省为整片）。用 continues 串起来的跨镜头实例会合并成一个图层，
    所以不能按图层 id 前缀判断镜头归属，只能按区间。"""
    dur = int(scene.get("duration", metric.load_shots()["shots"][-1]["end"]))
    return [(int(l.get("in", 0)), int(l.get("out", dur))) for l in scene.get("layers", []) if l.get("kind") == "model"]


def covered_frames(shot: dict, lspans: list[tuple[int, int]]) -> list[int]:
    """镜头 [start, end) 与所有模型图层区间之并的交集（升序帧号）。非空即视为「已拟合」。"""
    fs: set[int] = set()
    for a, b in lspans:
        fs.update(range(max(a, shot["start"]), min(b, shot["end"])))
    return sorted(fs)


def frame_ranges(frames: list[int]) -> list[tuple[int, int]]:
    """升序帧号 → 闭区间列表。"""
    out: list[tuple[int, int]] = []
    for f in frames:
        if out and out[-1][1] == f - 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def spans(shots: list[dict]) -> list[list[dict]]:
    """把按时间排好的镜头分成首尾相接的连续段（每段一个解码管道，避免每个镜头从头解码）。"""
    out: list[list[dict]] = []
    for s in shots:
        if out and out[-1][-1]["end"] == s["start"]:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def ref_stream(a: int, b: int):
    """流式解码参考帧 [a, b)，颜色转换与 metric.decode_ref 完全相同（同一个 ffmpeg_vf）。"""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(metric.VIDEO), "-vf", metric.ffmpeg_vf(a, b),
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H * 3
    try:
        for f in range(a, b):
            buf = p.stdout.read(n)
            if len(buf) != n:
                raise RuntimeError(f"参考帧 {f} 解码失败（视频提前结束）")
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.wait()


def run_render(scene_path: Path, frames: list[int], renders: Path) -> None:
    """用 pnpm render（与播放器同一份代码）渲染这些帧（镜头与图层区间的交集）。"""
    spec = ",".join(f"{a}-{b}" if b > a else f"{a}" for a, b in frame_ranges(sorted(set(frames))))
    renders.mkdir(parents=True, exist_ok=True)
    log(f"pnpm render {spec} → {renders}")
    subprocess.run(["pnpm", "-s", "render", "--scene", str(scene_path), "--frames", spec, "--out", str(renders)],
                   cwd=REPO, check=True)


# ---------------- 画面 ----------------

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def font(size: int):
    if size not in _FONT_CACHE:
        for p in FONT_PATHS:
            if Path(p).exists():
                _FONT_CACHE[size] = ImageFont.truetype(p, size)
                break
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def burn(img: np.ndarray, lines: list[str], x=10, y=8, size=24, color=(255, 255, 255)) -> None:
    """左上角烧录多行文字（只把文字所在的小块转成 PIL，省时间）。"""
    fnt = font(size)
    lh = int(size * 1.3)
    wmax = max(int(fnt.getlength(t)) for t in lines) + 24
    hh = lh * len(lines) + 12
    x1, y1 = min(img.shape[1], x + wmax), min(img.shape[0], y + hh)
    patch = Image.fromarray(np.ascontiguousarray(img[y:y1, x:x1]))
    d = ImageDraw.Draw(patch)
    for i, t in enumerate(lines):
        d.text((6, 4 + i * lh), t, font=fnt, fill=color, stroke_width=3, stroke_fill=(0, 0, 0))
    img[y:y1, x:x1] = np.asarray(patch)


def center_text(img: np.ndarray, text: str, size=72) -> None:
    fnt = font(size)
    tw = int(fnt.getlength(text))
    burn(img, [text], x=max(0, (img.shape[1] - tw) // 2 - 12), y=img.shape[0] // 2 - size // 2 - 10, size=size,
         color=(255, 220, 120))


def diff_img(comp: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = np.clip(np.abs(comp - ref.astype(np.float64)).mean(-1) * 3, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)[..., ::-1].copy()


def heat_overlay(ref: np.ndarray, smap, mask, excl) -> np.ndarray:
    """原视频灰度压暗作底，遮罩内叠 SSIM 热力图（metric.heat 的配色），排除区蓝灰。"""
    g = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, -1).astype(np.float64) * 0.45
    hm = metric.heat(smap, mask).astype(np.float64)
    out = g.copy()
    out[mask] = g[mask] * 0.35 + hm[mask] * 0.65
    out[excl] = out[excl] * 0.4 + np.array([60, 60, 160]) * 0.6
    return np.clip(out, 0, 255).astype(np.uint8)


def compose(layout: str, ref: np.ndarray, fr: dict) -> np.ndarray:
    """按布局拼一帧差异视频。fr：{'info', 'comp', 'rgb', 'alpha', 'smap', 'mask', 'excl'}（未拟合时只有 info）。"""
    fitted = "comp" in fr
    if layout == "sbs":
        right = np.clip(fr["comp"], 0, 255).astype(np.uint8) if fitted else (ref * 0.35).astype(np.uint8)
        img = np.concatenate([ref, right], 1)
        if not fitted:
            sub = img[:, W:]
            center_text(sub, "未拟合")
            img[:, W:] = sub
        return np.ascontiguousarray(img)
    if layout == "overlay":
        if not fitted:
            return ref.copy()
        a = fr["alpha"][..., None] * 0.5  # 渲染以 50% 不透明度叠在原视频上（错位会出现重影）
        return np.clip(fr["rgb"] * a + ref.astype(np.float64) * (1 - a), 0, 255).astype(np.uint8)
    if layout == "heat":
        if not fitted:
            return ref.copy()
        return heat_overlay(ref, fr["smap"], fr["mask"], fr["excl"])
    if layout == "quad":
        if fitted:
            comp = np.clip(fr["comp"], 0, 255).astype(np.uint8)
            dif = diff_img(fr["comp"], ref)
            hm = metric.heat(fr["smap"], fr["mask"])
            hm[fr["excl"]] = (hm[fr["excl"]] * 0.4 + np.array([60, 60, 160]) * 0.6).astype(np.uint8)
        else:
            comp = dif = hm = (ref * 0.35).astype(np.uint8)
        ps = [cv2.resize(p, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for p in (ref, comp, dif, hm)]
        img = np.ascontiguousarray(np.concatenate([np.concatenate(ps[:2], 1), np.concatenate(ps[2:], 1)], 0))
        if not fitted:
            sub = img[:, W // 2:]
            center_text(sub, "未拟合", size=56)
            img[:, W // 2:] = sub
        return img
    raise ValueError(layout)


def frame_label(info: dict) -> list[str]:
    head = f"f={info['frame']}  {info['shot']}  [{CLS_ZH[info['class']]}]"
    if not info["fitted"]:
        return [head, "未拟合"]
    s, s7 = info.get("ssim360"), info.get("ssim720")
    if s is None:
        return [head, "SSIM 无（遮罩 <1%）"]
    mark = "" if info["class"] != "normal" else ("  ✓" if s >= THR else "  ✗<0.7")
    return [head, f"SSIM@360={s:.3f}  @720={s7:.3f}{mark}"]


def triptych(ref: np.ndarray, comp: np.ndarray, info: dict) -> bytes:
    """最差帧缩略图：原视频 | 渲染（合成到原视频背景）| 差异图，每格 640×360，JPEG。"""
    ps = [ref, np.clip(comp, 0, 255).astype(np.uint8), diff_img(comp, ref)]
    ps = [cv2.resize(p, (640, 360), interpolation=cv2.INTER_AREA) for p in ps]
    img = np.ascontiguousarray(np.concatenate(ps, 1))
    burn(img, [f"f={info['frame']} [{CLS_ZH[info['class']]}] SSIM@360={info['ssim360']:.3f}"], size=18)
    for i, t in enumerate(("原视频", "渲染", "差异")):
        burn(img, [t], x=640 * i + 520, y=320, size=18)
    bio = io.BytesIO()
    Image.fromarray(img).save(bio, "JPEG", quality=85)
    return bio.getvalue()


# ---------------- 视频编码 ----------------

def open_encoder(path: Path, size: tuple[int, int], segs: list[tuple[int, int]], crf: int, preset: str):
    """rawvideo → libx264；音频取原 wav 的相同区段（多段时 atrim + concat），保证音画同步。"""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
           "-i", str(metric.AUDIO)]
    parts = [f"[1:a]atrim=start={a / FPS:.6f}:end={b / FPS:.6f},asetpts=PTS-STARTPTS[a{i}]" for i, (a, b) in enumerate(segs)]
    fc = ";".join(parts) + ";" + "".join(f"[a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=0:a=1[aout]"
    cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ---------------- 统计 ----------------

def class_row(summ: dict, cls: str) -> dict:
    c = summ[cls]
    s = c["ssim360"]
    return {"frames": c["frames"], "valid": s.get("valid", 0), "mean": s.get("mean"), "min": s.get("min"),
            "p5": s.get("p5"), "fracGE0.7": s.get("fracGE0.7"),
            "below0.7": len(s.get("framesBelow0.7", [])) if s.get("valid") else 0,
            "worstFrame": s.get("worstFrame")}


def shot_status(fitted: bool, summ: dict | None) -> str:
    if not fitted:
        return "未拟合"
    n = summ["normal"]["ssim360"]
    if n.get("valid", 0) == 0:
        return "无有效普通帧"
    return "达标" if summ["normalPass"] else "不达标"


def ranking(shots_out: dict) -> dict:
    """最差镜头排行：普通帧按（不达标帧数降序，均值升序）；目测帧、特效帧各自按均值升序。"""
    rk = {}
    for cls in metric.CLASSES:
        rows = []
        for sid, v in shots_out.items():
            if not v["fitted"]:
                continue
            r = class_row(v["summary"], cls)
            if r["valid"]:
                rows.append({"shot": sid, **r})
        if cls == "normal":
            rows.sort(key=lambda r: (-r["below0.7"], r["mean"]))
        else:
            rows.sort(key=lambda r: r["mean"])
        rk[cls] = rows
    return rk


# ---------------- HTML ----------------

CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;--grid:#ecebe7;
--normal:#2a78d6;--visual:#eb6834;--effect:#1baf7a;--good:#0ca30c;--bad:#d03b3b;--warn:#b07800;--chip:#f3f2ef}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--panel:#232322;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e86;--line:#383835;--grid:#2e2e2c;--normal:#3987e5;--visual:#d95926;--effect:#199e70;--good:#2fbf2f;--bad:#e66767;
--warn:#e0a020;--chip:#2c2c2a}}
:root[data-theme="dark"]{--bg:#1a1a19;--panel:#232322;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#383835;--grid:#2e2e2c;
--normal:#3987e5;--visual:#d95926;--effect:#199e70;--good:#2fbf2f;--bad:#e66767;--warn:#e0a020;--chip:#2c2c2a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Noto Sans CJK SC",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}
h3{font-size:15px;margin:0}.sub{color:var(--ink2);overflow-wrap:anywhere}code{overflow-wrap:anywhere}a{color:var(--normal)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.tile .k{color:var(--ink2);font-size:12px}.tile .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;background:var(--chip);position:sticky;top:0}
td:first-child,th:first-child{text-align:left}tr.unfit td{color:var(--muted)}
th.g{text-align:center;border-left:1px solid var(--line)}td.gl{border-left:1px solid var(--line)}
.st{font-weight:600}.st.pass{color:var(--good)}.st.fail{color:var(--bad)}.st.na{color:var(--warn)}.st.unfit{color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;color:var(--ink2);font-size:12px;margin:4px 0}
.shot{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin:14px 0}
.shot svg{width:100%;height:auto;display:block}.thumbs{display:grid;grid-template-columns:1fr;gap:8px;margin-top:10px}
.thumbs figure{margin:0}.thumbs img{width:100%;border-radius:4px;display:block}
.thumbs figcaption{font-size:12px;color:var(--ink2)}.why{font-family:ui-monospace,monospace;font-size:11px;color:var(--ink2)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:14px}
"""


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def pct(v):
    return "—" if v is None else f"{v * 100:.0f}%"


def status_cls(st: str) -> str:
    return {"达标": "pass", "不达标": "fail", "无有效普通帧": "na", "未拟合": "unfit"}[st]


def svg_curve(sid: str, a: int, b: int, rows: list[dict]) -> str:
    """SSIM@360 逐帧曲线（点按类别着色，灰线为 SSIM@720，虚线为 0.7 门槛）。悬停看帧号与数值（<title>）。"""
    Wd, Hd, L, R, T, B = 1000, 220, 44, 12, 10, 28
    pw, ph = Wd - L - R, Hd - T - B
    x = lambda f: L + (f - a) / max(1, b - 1 - a) * pw  # noqa: E731
    y = lambda v: T + (1 - v) * ph  # noqa: E731
    g = [f'<svg viewBox="0 0 {Wd} {Hd}" role="img" aria-label="{sid} 逐帧 SSIM@360">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        g.append(f'<line x1="{L}" x2="{Wd - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
                 f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--ink2)">{v:.2f}</text>')
    g.append(f'<line x1="{L}" x2="{Wd - R}" y1="{y(THR):.1f}" y2="{y(THR):.1f}" stroke="var(--bad)" stroke-dasharray="5 4" stroke-width="1.2"/>'
)
    step = max(1, round((b - a) / 8 / 10) * 10)
    for f in range(a - a % step + step if a % step else a, b, step):
        g.append(f'<text x="{x(f):.1f}" y="{Hd - 8}" text-anchor="middle" font-size="11" fill="var(--ink2)">{f}</text>')
    # 类别色带（底部 4px），没有 SSIM 的帧也能看出类别
    for r in rows:
        g.append(f'<rect x="{x(r["frame"]) - pw / max(1, b - a) / 2:.1f}" y="{T + ph + 2}" width="{pw / max(1, b - a) + 0.3:.2f}" height="4" fill="var(--{r["class"]})"/>')
    pts = [(x(r["frame"]), y(r["ssim720"])) for r in rows if r.get("ssim720") is not None]
    if pts:
        g.append('<polyline fill="none" stroke="var(--muted)" stroke-width="1" opacity="0.7" points="'
                 + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts) + '"/>')
    for r in rows:
        if r.get("ssim360") is None:
            continue
        tip = f'f={r["frame"]} {CLS_ZH[r["class"]]} SSIM@360={r["ssim360"]:.3f} @720={r["ssim720"]:.3f}'
        g.append(f'<circle cx="{x(r["frame"]):.1f}" cy="{y(r["ssim360"]):.1f}" r="2.6" fill="var(--{r["class"]})">'
                 f'<title>{html.escape(tip)}</title></circle>')
    g.append("</svg>")
    return "".join(g)


def write_html(out: Path, rep: dict, video_name: str | None) -> None:
    shots = rep["shots"]
    ov = rep["overall"]
    fitted = [s for s, v in shots.items() if v["fitted"]]
    sts = [v["status"] for v in shots.values()]
    E = html.escape
    h = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>CharaForge 质检报告</title><style>{CSS}</style></head><body><main>',
         '<h1>CharaForge 全片质检报告</h1>',
         f'<div class="sub">生成于 {E(rep["generated"])} · 场景 <code>{E(rep["scene"])}</code> · 渲染 <code>{E(rep["renders"])}</code> · '
         f'指标定义 docs/pipeline.md（tools/qa/metric.py）</div>',
         '<div class="sub">文件：<a href="frames.csv">frames.csv</a> · <a href="shots.csv">shots.csv</a> · <a href="report.json">report.json</a>'
         + (f' · <a href="{E(video_name)}">差异视频 {E(video_name)}</a>' if video_name else "") + "</div>"]
    # 全片汇总
    h.append('<h2>全片汇总</h2><div class="tiles">')
    tiles = [("已拟合镜头", f"{len(fitted)} / {len(shots)}"),
             ("普通帧达标镜头", f"{sts.count('达标')} / {sts.count('达标') + sts.count('不达标')}")]
    for cls in metric.CLASSES:
        r = class_row(ov, cls)
        tiles.append((f"{CLS_ZH[cls]}帧 SSIM@360 均值（{r['valid']} 帧）", fmt(r["mean"])))
    r = class_row(ov, "normal")
    tiles.append(("普通帧 <0.7 帧数", str(r["below0.7"])))
    h += [f'<div class="tile"><div class="k">{E(k)}</div><div class="v">{E(v)}</div></div>' for k, v in tiles]
    h.append("</div>")
    h.append('<div class="scroll"><table><tr><th>类别</th><th>帧数</th><th>有效帧</th><th>均值</th><th>最小</th><th>5% 分位</th><th>≥0.7</th><th>&lt;0.7 帧数</th><th>最差帧</th></tr>')
    for cls in metric.CLASSES:
        r = class_row(ov, cls)
        h.append(f'<tr><td><span class="dot" style="background:var(--{cls})"></span>{CLS_ZH[cls]}帧</td><td>{r["frames"]}</td><td>{r["valid"]}</td>'
                 f'<td>{fmt(r["mean"])}</td><td>{fmt(r["min"])}</td><td>{fmt(r["p5"])}</td><td>{pct(r["fracGE0.7"])}</td>'
                 f'<td>{r["below0.7"]}</td><td>{r["worstFrame"] if r["worstFrame"] is not None else "—"}</td></tr>')
    h.append("</table></div>")
    h.append('<p class="sub">普通帧门槛：逐帧 SSIM@360 ≥ 0.7；目测帧（≥2 个可见实例，或最大头高 &lt;200px）与特效帧照常统计、不卡门槛，'
             '以并排、叠加对比为准。判定优先级：特效 &gt; 目测 &gt; 普通。统计只含已拟合镜头。</p>')
    # 排行
    h.append("<h2>最差镜头排行</h2><div class='cols'>")
    titles = {"normal": "普通帧（按 &lt;0.7 帧数、均值排序）", "visual": "目测帧（按均值排序，不卡门槛）", "effect": "特效帧（按均值排序，不卡门槛）"}
    for cls in metric.CLASSES:
        rows = rep["ranking"][cls]
        h.append(f'<div><h3><span class="dot" style="background:var(--{cls})"></span>{titles[cls]}</h3><div class="scroll"><table>'
                 '<tr><th>#</th><th>镜头</th><th>有效帧</th>' + ("<th>&lt;0.7</th>" if cls == "normal" else "")
                 + '<th>均值</th><th>最小</th><th>最差帧</th></tr>')
        for i, r in enumerate(rows, 1):
            h.append(f'<tr><td>{i}</td><td><a href="#{r["shot"]}">{r["shot"]}</a></td><td>{r["valid"]}</td>'
                     + (f'<td>{r["below0.7"]}</td>' if cls == "normal" else "")
                     + f'<td>{fmt(r["mean"])}</td><td>{fmt(r["min"])}</td><td>{r["worstFrame"]}</td></tr>')
        if not rows:
            h.append('<tr><td colspan="7" class="sub">（无）</td></tr>')
        h.append("</table></div></div>")
    h.append("</div>")
    # 逐镜头表
    h.append('<h2>逐镜头</h2><div class="scroll"><table><tr><th rowspan="2">镜头</th><th rowspan="2">帧区间</th><th rowspan="2">状态</th>'
             + "".join(f'<th class="g" colspan="{7 if c == "normal" else 4}"><span class="dot" style="background:var(--{c})"></span>{CLS_ZH[c]}帧</th>' for c in metric.CLASSES)
             + "<th rowspan='2'>覆盖率均值</th></tr><tr>"
             + "<th>帧数</th><th>有效</th><th>均值</th><th>最小</th><th>5%</th><th>≥0.7</th><th>&lt;0.7</th>"
             + "<th>帧数</th><th>有效</th><th>均值</th><th>最小</th>" * 2 + "</tr>")
    for sid, v in shots.items():
        st = v["status"]
        cells = [f'<td><a href="#{sid}">{sid}</a></td><td>{v["start"]}–{v["end"] - 1}</td><td class="st {status_cls(st)}">{st}</td>']
        for cls in metric.CLASSES:
            nfr = v["classFrames"][cls]
            if v["fitted"]:
                r = class_row(v["summary"], cls)
                cells.append(f'<td class="gl">{nfr}</td><td>{r["valid"]}</td><td>{fmt(r["mean"])}</td><td>{fmt(r["min"])}</td>')
                if cls == "normal":
                    cells.append(f'<td>{fmt(r["p5"])}</td><td>{pct(r["fracGE0.7"])}</td><td>{r["below0.7"]}</td>')
            else:
                cells.append(f'<td class="gl">{nfr}</td>' + "<td>—</td>" * (6 if cls == "normal" else 3))
        cov = v["summary"].get("coverageMean") if v["fitted"] else None
        cells.append(f"<td>{fmt(cov)}</td>")
        h.append(f'<tr class="{"" if v["fitted"] else "unfit"}">{"".join(cells)}</tr>')
    h.append("</table></div>")
    # 每镜头详情
    h.append('<h2>镜头详情</h2><div class="legend">' + "".join(
        f'<span><span class="dot" style="background:var(--{c})"></span>{CLS_ZH[c]}帧 SSIM@360</span>' for c in metric.CLASSES)
        + '<span><span class="dot" style="background:var(--muted)"></span>SSIM@720（灰线）</span><span>红色虚线 = 0.7 门槛（只对普通帧）</span>'
        '<span>底部色带 = 每帧类别</span></div>')
    for sid, v in shots.items():
        if not v["fitted"]:
            continue
        h.append(f'<div class="shot" id="{sid}"><h3>{sid} · {v["start"]}–{v["end"] - 1} · '
                 f'<span class="st {status_cls(v["status"])}">{v["status"]}</span></h3>')
        h.append(svg_curve(sid, v["start"], v["end"], v["perFrame"]))
        if v["thumbs"]:
            h.append('<div class="thumbs">')
            for t in v["thumbs"]:
                h.append(f'<figure><img loading="lazy" src="{E(t["file"])}" alt="{sid} f={t["frame"]}">'
                         f'<figcaption>最差{CLS_ZH[t["class"]]}帧 f={t["frame"]} SSIM@360={t["ssim360"]:.3f} · '
                         f'<span class="why">{E(" ".join(t["why"]) or "—")}</span></figcaption></figure>')
            h.append("</div>")
        h.append("</div>")
    h.append("</main></body></html>")
    (out / "index.html").write_text("\n".join(h), encoding="utf-8")


# ---------------- 主流程 ----------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(SCENE), help="场景文件（判断哪些镜头已拟合；--render 时用它渲染）")
    ap.add_argument("--shots", help="只跑这些镜头，逗号分隔（缺省全片 46 个）")
    ap.add_argument("--renders", help="渲染帧目录 frame_NNNNNN.png（缺省 <out>/renders）")
    ap.add_argument("--render", action="store_true", help="先用 pnpm render 渲染已拟合镜头的全部帧到 --renders")
    ap.add_argument("--out", default=str(OUT), help="输出目录（缺省 .data/qa/report）")
    ap.add_argument("--layout", default="sbs", choices=LAYOUTS,
                    help="差异视频布局：sbs 并排 2560×720 | overlay 50%% 叠加 | heat SSIM 热力图 | quad 四宫格")
    ap.add_argument("--no-video", action="store_true", help="不生成差异视频")
    ap.add_argument("--worst", type=int, default=2, help="每镜头每类保留的最差帧缩略图数")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--rehtml", action="store_true", help="只用 <out>/report.json 重新生成 index.html（不重算）")
    args = ap.parse_args(argv)
    if args.rehtml:
        out = Path(args.out)
        vids = sorted(out.glob("diff_*.mp4"))
        write_html(out, json.loads((out / "report.json").read_text(encoding="utf-8")), vids[0].name if vids else None)
        return

    out = Path(args.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    renders = Path(args.renders) if args.renders else out / "renders"
    scene_path = Path(args.scene)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    lspans = layer_spans(scene)
    all_shots = sorted(metric.load_shots()["shots"], key=lambda s: s["start"])
    if args.shots:
        want = args.shots.split(",")
        unknown = set(want) - {s["id"] for s in all_shots}
        if unknown:
            ap.error(f"未知镜头：{sorted(unknown)}")
        sel = [s for s in all_shots if s["id"] in want]
    else:
        sel = all_shots
    covered = {s["id"]: covered_frames(s, lspans) for s in sel}
    fit_sel = [s for s in sel if covered[s["id"]]]
    log(f"镜头 {len(sel)} 个，其中已拟合（与图层区间有交集）{len(fit_sel)} 个：{[s['id'] for s in fit_sel]}")

    need = [f for s in fit_sel for f in covered[s["id"]]]
    if args.render and need:
        run_render(scene_path, need, renders)
    missing = [f for f in need if not (renders / f"frame_{f:06d}.png").exists()]
    if missing:
        sys.exit(f"渲染帧缺失 {len(missing)} 帧（如 {missing[:5]}），目录 {renders}；加 --render 先渲染")

    segs = [(sp[0]["start"], sp[-1]["end"]) for sp in spans(sel)]
    video_name = None
    enc = None
    if not args.no_video:
        video_name = f"diff_{args.layout}.mp4"
        size = (2 * W, H) if args.layout == "sbs" else (W, H)
        enc = open_encoder(out / video_name, size, segs, args.crf, args.preset)

    # 旧缩略图清掉，避免 HTML 之外残留过期文件
    for p in (out / "thumbs").glob("*.jpg"):
        p.unlink()

    shots_out: dict = {}
    all_rows: list[dict] = []
    t0 = time.time()
    done = 0
    total = sum(b - a for a, b in segs)
    for sp in spans(sel):
        stream = ref_stream(sp[0]["start"], sp[-1]["end"])
        for shot in sp:
            sid = shot["id"]
            cov = set(covered[sid])
            fitted = bool(cov)
            rows: list[dict] = []
            worst: dict[str, list] = {c: [] for c in metric.CLASSES}  # 最大堆（取负）：保留最差 K 帧
            counter = 0
            for f in range(shot["start"], shot["end"]):
                ref = next(stream)
                if fitted:
                    if f in cov:
                        rgb, alpha = metric.load_render(renders / f"frame_{f:06d}.png")
                    else:
                        # 没有任何图层覆盖的帧：渲染器在这里输出全透明，直接按空渲染算（遮罩 <1% → 无角色，不计有效帧）
                        rgb, alpha = np.zeros((H, W, 3)), np.zeros((H, W))
                    info, comp, smap, mask = metric.frame_metric(rgb, alpha, ref, shot, f, want_maps=True)
                    fr = {"comp": comp, "rgb": rgb, "alpha": alpha, "smap": smap, "mask": mask,
                          "excl": metric.exclusion_mask(shot, f)}
                else:
                    cls, why = metric.classify(shot, f)
                    vwhy, nvis, max_h = metric.visual_reasons(shot, f)
                    info = {"frame": f, "class": cls, "why": why, "maskArea": None, "visibleInstances": nvis,
                            "maxHeadH": max_h, "whyVisual": vwhy, "coverage": None, "ssim360": None, "ssim720": None}
                    fr = {}
                info["shot"], info["fitted"] = sid, fitted
                rows.append(info)
                if enc is not None:
                    img = compose(args.layout, ref, fr)
                    burn(img, frame_label(info), size=26 if args.layout == "sbs" else 22)
                    enc.stdin.write(img.tobytes())
                if fitted and info["ssim360"] is not None and args.worst > 0:
                    hp = worst[info["class"]]
                    key = (-info["ssim360"], counter)
                    counter += 1
                    if len(hp) < args.worst or key > hp[0][0]:
                        item = (key, info, triptych(ref, comp, info))
                        if len(hp) < args.worst:
                            heapq.heappush(hp, item)
                        else:
                            heapq.heapreplace(hp, item)
                done += 1
                if done % 200 == 0:
                    log(f"{done}/{total} 帧（{done / (time.time() - t0):.1f} fps）")
            # 镜头汇总
            thumbs = []
            for cls in metric.CLASSES:
                for _, info, jpg in sorted(worst[cls], key=lambda t: -t[0][0]):
                    name = f"thumbs/{sid}_{cls}_{info['frame']}.jpg"
                    (out / name).write_bytes(jpg)
                    thumbs.append({"file": name, "frame": info["frame"], "class": cls, "ssim360": info["ssim360"],
                                   "why": info["why"]})
            classes = {c: sum(1 for r in rows if r["class"] == c) for c in metric.CLASSES}
            summ = metric.summarize(rows) if fitted else None
            shots_out[sid] = {"start": shot["start"], "end": shot["end"], "fitted": fitted, "classFrames": classes,
                              "layerFrames": [list(r) for r in frame_ranges(covered[sid])],
                              "status": shot_status(fitted, summ), "summary": summ, "thumbs": thumbs, "perFrame": rows}
            if fitted:
                all_rows += rows
                n, v, e = (class_row(summ, c) for c in metric.CLASSES)
                log(f"{sid}: 普通 {n['frames']} 帧 均值 {fmt(n['mean'])} ≥0.7 {pct(n['fracGE0.7'])} | "
                    f"目测 {v['frames']} 帧 均值 {fmt(v['mean'])} | 特效 {e['frames']} 帧 均值 {fmt(e['mean'])} → {shots_out[sid]['status']}")
        stream.close()

    if enc is not None:
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg 编码失败")
        log(f"差异视频 → {out / video_name}")

    overall = metric.summarize(all_rows) if all_rows else metric.summarize([])
    rep = {"definition": "docs/pipeline.md#验收指标的定义", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "scene": str(scene_path), "renders": str(renders), "threshold360": THR, "overall": overall,
           "shots": shots_out}
    rep["ranking"] = ranking(shots_out)

    # CSV
    cols = ["frame", "shot", "fitted", "class", "why", "ssim360", "ssim720", "coverage", "maskArea",
            "visibleInstances", "maxHeadH", "whyVisual"]
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for v in shots_out.values():
            for r in v["perFrame"]:
                w.writerow([int(r["fitted"]) if c == "fitted" else
                            ";".join(r[c]) if c in ("why", "whyVisual") else
                            (f"{r[c]:.5f}" if isinstance(r[c], float) else ("" if r[c] is None else r[c]))
                            for c in cols])
    with open(out / "shots.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        head = ["shot", "start", "end", "fitted", "status"]
        for c in metric.CLASSES:
            head += [f"{c}_frames", f"{c}_valid", f"{c}_mean", f"{c}_min", f"{c}_p5", f"{c}_fracGE0.7", f"{c}_below0.7"]
        w.writerow(head + ["coverageMean"])
        for sid, v in shots_out.items():
            row = [sid, v["start"], v["end"], int(v["fitted"]), v["status"]]
            for c in metric.CLASSES:
                if v["fitted"]:
                    r = class_row(v["summary"], c)
                    row += [r["frames"], r["valid"]] + ["" if r[k] is None else f"{r[k]:.5f}" for k in ("mean", "min", "p5", "fracGE0.7")] + [r["below0.7"]]
                else:
                    row += [v["classFrames"][c]] + [""] * 6
            cov = v["summary"].get("coverageMean") if v["fitted"] else None
            w.writerow(row + ["" if cov is None else f"{cov:.5f}"])
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    write_html(out, rep, video_name)
    log(f"报告 → {out / 'index.html'}（{time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
