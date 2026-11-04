"""第 4 步：由人工标注 annotations.json + 自动跟踪生成镜头表、缩略图拼图、总览图和 index.html。

  annotations.json（本目录，人工维护）里每个实例写「锚点」anchors（人工看图量出的 head/eyes/bbox 等），
  本脚本用 track.py 在锚点之间做光流相似变换跟踪，加密成约每 10 帧一个的 track 采样。

产出：
  scenes/deepseek-chan/shots.json
  .data/recon/thumbs/<shotId>.jpg   每镜头均匀取帧（含首尾帧），烧录帧号，叠加 head 框/眼睛/排除区
  .data/recon/overview.jpg          每镜头首帧+中帧的总览
  .data/recon/index.html            校对页
用法：tools/.venv/bin/python -I tools/recon/build.py [--no-track] [--only s05,s06] [--no-thumbs]
"""
from __future__ import annotations

import html
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ANNOT, DATA, H, SHOTS_JSON, THUMBS, VIDEO_URI, W, load_json  # noqa: E402
from sheet import grid, save  # noqa: E402

TRACK_CACHE = DATA / "track_cache.json"
EYE_STATS: dict[str, tuple[int, int]] = {}
_FRAME_CACHE: dict[int, np.ndarray] = {}


def _plausible(t: dict, new, check_shift: bool = True) -> bool:
    """细化结果的几何合理性检查（防止锁到胸前蝴蝶结的青色宝石、头发高光等）：
    两眼都在 head 框内、纵向位于框高 15%~80%，两眼间距为头宽的 18%~65%，
    相对粗估位置的位移不超过头宽的 25%（check_shift=False 时不查位移：用于在 head 框内整体搜索的结果）。
    没有 head 框时只要求位移 < 两眼粗估间距的 40%。"""
    (ax, ay), (bx, by) = new
    (px0, py0), (px1, py1) = t["eyes"]
    head = t.get("head")
    if head:
        x, y, w, h = head
        for ex, ey in new:
            if not (x <= ex <= x + w and y + 0.15 * h <= ey <= y + 0.8 * h):
                return False
        dd = np.hypot(bx - ax, by - ay)
        if not (0.18 * w <= dd <= 0.65 * w):
            return False
        lim = 0.25 * w
        if not check_shift:
            return True
    else:
        lim = 0.4 * np.hypot(px1 - px0, py1 - py0)
    return np.hypot(ax - px0, ay - py0) <= lim and np.hypot(bx - px1, by - py1) <= lim


def refine_anchor(a: dict, pair_search: bool = True) -> dict:
    """人工锚点的细化：先按普通采样细化；失败时在 2 倍大的 head 框内搜一对虹膜，
    找到且合理则把 eyes 换成检测值，并把 head/bbox 平移同样的位移（人工读数常有整体偏移）。
    平移量记录到 _shift，便于复核。"""
    t = refine_sample(dict(a), pair_search)
    if not ALLOW_ANCHOR_SHIFT or t.get("_eyesRefined") or not t.get("head") or not t.get("eyes"):
        return t
    from eyes import find_pair, refine
    img = _FRAME_CACHE.get(t["frame"])
    x, y, w, h = t["head"]
    pair = find_pair(img, [x - 0.3 * w, y - 0.3 * h, 1.6 * w, 1.6 * h], t["eyes"])
    if not pair:
        return t
    new, ok = refine(img, pair)
    if not all(ok):
        return t
    (px0, py0), (px1, py1) = t["eyes"]
    dx = (new[0][0] + new[1][0] - px0 - px1) / 2
    dy = (new[0][1] + new[1][1] - py0 - py1) / 2
    if np.hypot(dx, dy) > 0.3 * w:   # 位移过大多半是锁到了别的实例/别的青色物体
        return t
    cand = dict(t, head=[x + dx, y + dy, w, h], eyes=t["eyes"])
    if not _plausible(cand, new, check_shift=False):
        return t
    t["head"] = cand["head"]
    if t.get("bbox"):
        bx, by, bw, bh = t["bbox"]
        t["bbox"] = [bx + dx, by + dy, bw, bh]
    t["eyes"] = new
    t["_eyesRefined"] = True
    ANCHOR_SHIFTS.append((t["frame"], round(dx, 1), round(dy, 1)))
    return t


ANCHOR_SHIFTS: list = []
SPECIAL_EYES = ("key3", "key4", "key5", "key6")


def _switch_state(inst, f):
    v = dict(inst.get("variant", {}))
    for e in sorted(inst.get("switchEvents", []), key=lambda e: e["frame"]):
        if e["frame"] <= f:
            v[e["param"]] = e["value"]
    return v


def annotate_eyes_state(track, inst):
    """eyesState：
    hidden  —— bbox 为 null、没有 eyes、或已明确标为 hidden；
    special —— 该帧 key3/key4/key5/key6（圈圈眼、爱心眼、><、屑屑眼）处于开启；
    open    —— 虹膜细化成功，或两眼附近都找到青色虹膜像素；
    closed  —— 两眼附近都没有青色虹膜像素（闭眼/笑眼）。单眼有（wink）记 open 并降低 confidence。
    钢盆（key10）阴影、强模糊会让 open 被误判成 closed，这种情况 confidence 降到 0.3。"""
    from common import read_frame
    from eyes import HI, LO
    for t in track:
        if t.get("eyesState"):
            if t["eyesState"] == "hidden":
                t.pop("eyes", None)
            continue
        if t.get("bbox", 0) is None or not t.get("eyes"):
            t["eyesState"] = "hidden"
            t.pop("eyes", None)
            continue
        sw = _switch_state(inst, t["frame"])
        if any(sw.get(k, 0) >= 0.5 for k in SPECIAL_EYES):
            t["eyesState"] = "special"
            continue
        if t.get("_eyesRefined"):
            t["eyesState"] = "open"
            continue
        f = t["frame"]
        if f not in _FRAME_CACHE:
            if len(_FRAME_CACHE) > 64:
                _FRAME_CACHE.clear()
            _FRAME_CACHE[f] = read_frame(f)
        img = _FRAME_CACHE[f]
        (lx, ly), (rx, ry) = t["eyes"]
        d = max(12.0, float(np.hypot(rx - lx, ry - ly)))
        r = int(0.3 * d)
        hits = []
        for ex, ey in t["eyes"]:
            x0, y0 = int(max(0, ex - r)), int(max(0, ey - r))
            x1, y1 = int(min(img.shape[1], ex + r)), int(min(img.shape[0], ey + r))
            if x1 - x0 < 3 or y1 - y0 < 3:
                hits.append(False); continue
            hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
            hits.append(int((cv2.inRange(hsv, LO, HI) > 0).sum()) >= max(8, (0.05 * d) ** 2))
        if all(hits):
            t["eyesState"] = "open"
        elif any(hits):
            t["eyesState"] = "open"
            t["confidence"] = min(t.get("confidence", 0.5), 0.4)
        else:
            t["eyesState"] = "closed"
            t["confidence"] = min(t.get("confidence", 0.5), 0.45)
        if sw.get("key10", 0) >= 0.5 or (t.get("blurSigma") or 0) > 3:
            t["confidence"] = min(t.get("confidence", 0.5), 0.3)
# 锚点已人工二次精修（manual.py FIX），整体平移容易锁到别的实例，默认关闭
ALLOW_ANCHOR_SHIFT = False


def refine_sample(t: dict, pair_search: bool = True) -> dict:
    """用 eyes.refine 校正一个采样的 eyes；两眼都成功且几何合理才采用（标记 _eyesRefined）。
    pair_search=False（实例字段 eyePairSearch: false）时不做 head 框内整体搜索：多个实例重叠/景深时
    整体搜索会锁到别的实例的眼睛（s15）。"""
    if not t.get("eyes") or t.get("bbox", 0) is None:
        return t
    from common import read_frame
    from eyes import refine
    f = t["frame"]
    if f not in _FRAME_CACHE:
        if len(_FRAME_CACHE) > 64:
            _FRAME_CACHE.clear()
        _FRAME_CACHE[f] = read_frame(f)
    new, ok = refine(_FRAME_CACHE[f], t["eyes"])
    if all(ok) and _plausible(t, new):
        t["eyes"] = new
        t["_eyesRefined"] = True
        return t
    # 退路：在 head 框（外扩 10%）里整体搜一对虹膜，再用 refine 复核
    if pair_search and t.get("head"):
        from eyes import find_pair
        x, y, w, h = t["head"]
        pair = find_pair(_FRAME_CACHE[f], [x - 0.1 * w, y - 0.1 * h, 1.2 * w, 1.2 * h], t["eyes"])
        if pair:
            new, ok = refine(_FRAME_CACHE[f], pair)
            if all(ok) and _plausible(t, new, check_shift=False):
                t["eyes"] = new
                t["_eyesRefined"] = True
    return t
INST_KEYS = ("id", "continues", "role", "z", "variant", "switchEvents", "motionHints", "clip", "flipX", "colorGrade",
             "notes")


def r1(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [r1(x) for x in v]
    return round(float(v), 1)


def interp_exclusion(ex, f):
    rs = ex["rects"]
    if f <= rs[0]["frame"]:
        return rs[0]["rect"]
    for a, b in zip(rs, rs[1:]):
        if a["frame"] <= f <= b["frame"]:
            t = (f - a["frame"]) / max(1e-6, b["frame"] - a["frame"])
            return [a["rect"][i] * (1 - t) + b["rect"][i] * t for i in range(4)]
    return rs[-1]["rect"]


def interp_track(track, f):
    """与 shots.ts 约定一致：相邻采样线性插值，区间外保持首/尾值。"""
    out = {}
    for key in ("bbox", "head", "eyes", "rotation", "opacity"):
        pts = [(s["frame"], s[key]) for s in track if key in s]
        if not pts:
            continue
        if f <= pts[0][0]:
            out[key] = pts[0][1]; continue
        if f >= pts[-1][0]:
            out[key] = pts[-1][1]; continue
        for (fa, va), (fb, vb) in zip(pts, pts[1:]):
            if fa <= f <= fb:
                if va is None or vb is None:
                    out[key] = va if f - fa < fb - f else vb
                else:
                    t = (f - fa) / max(1e-6, fb - fa)
                    out[key] = (np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t).tolist()
                break
    return out


COLORS = [(255, 80, 80), (80, 220, 80), (80, 160, 255), (255, 200, 0), (255, 0, 255), (0, 255, 255)]


def overlay_fn(shot):
    def draw(f, im):
        for k, inst in enumerate(shot["instances"]):
            c = COLORS[k % len(COLORS)]
            s = interp_track(inst["track"], f)
            if s.get("bbox"):
                x, y, w, h = map(int, s["bbox"]); cv2.rectangle(im, (x, y), (x + w, y + h), c, 2)
            if s.get("head"):
                x, y, w, h = map(int, s["head"]); cv2.rectangle(im, (x, y), (x + w, y + h), c, 4)
                cv2.putText(im, inst["id"], (x + 4, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3)
            prev = [t for t in inst["track"] if t["frame"] <= f]
            if s.get("eyes") and not (prev and prev[-1].get("eyesState") == "hidden"):
                for ex, ey in s["eyes"]:
                    cv2.circle(im, (int(ex), int(ey)), 7, c, 3)
            if inst.get("clip"):
                ck = inst["clip"]
                r = ck[0]["rect"]
                for a, b in zip(ck, ck[1:]):
                    if a["frame"] <= f <= b["frame"]:
                        t = (f - a["frame"]) / max(1e-6, b["frame"] - a["frame"])
                        r = [a["rect"][i] * (1 - t) + b["rect"][i] * t for i in range(4)]
                        break
                else:
                    if f > ck[-1]["frame"]:
                        r = ck[-1]["rect"]
                x, y, w, h = map(int, r); cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
        for ex in shot["exclusions"]:
            if ex["start"] <= f < ex["end"]:
                x, y, w, h = map(int, interp_exclusion(ex, f))
                sub = im[max(0, y):y + h, max(0, x):x + w]
                sub[:] = (sub * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
                cv2.putText(im, ex["reason"], (x + 4, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return draw


def thumb_frames(s, e, n=12):
    if e - s <= n:
        return list(range(s, e))
    return sorted(set(int(round(v)) for v in np.linspace(s, e - 1, n)))


def make_thumbs(shots, only=None):
    THUMBS.mkdir(parents=True, exist_ok=True)
    for sh in shots:
        if only and sh["id"] not in only:
            continue
        fr = thumb_frames(sh["start"], sh["end"])
        img = grid(fr, 4, 320, overlay=overlay_fn(sh) if sh["hasCharacter"] else None)
        save(img, THUMBS / f"{sh['id']}.jpg")


def make_overview(shots):
    fr, lab = [], []
    for sh in shots:
        for f in (sh["start"], (sh["start"] + sh["end"] - 1) // 2):
            fr.append(f); lab.append(f"{sh['id']} {f}")
    save(grid(fr, 10, 192, labels=lab), DATA / "overview.jpg")


def make_index(doc):
    rows = []
    for sh in doc["shots"]:
        insts = "<br>".join(
            html.escape(f"{i['id']} ({i['role']}, z={i['z']})"
                        + (f" continues={i['continues']}" if i.get('continues') else "")
                        + (f" clip×{len(i['clip'])}" if i.get('clip') else "")
                        + f" variant={json.dumps(i['variant'], ensure_ascii=False)} "
                        f"switch={[(e['frame'], e['param'], e['value'], 'fade', e.get('fade', 0)) for e in i['switchEvents']]} "
                        f"motion={[(m['motion'], m['start']) for m in i['motionHints']]}")
            for i in sh["instances"])
        ex = "<br>".join(html.escape(f"{e['reason']} [{e['start']},{e['end']})") for e in sh["exclusions"])
        ef = "<br>".join(html.escape(f"{e['kind']} [{e['start']},{e['end']}) {e.get('note', '')}") for e in sh["effects"])
        tr = sh["transitionIn"]
        rows.append(f"""<tr id="{sh['id']}"><td><b>{sh['id']}</b><br>[{sh['start']}, {sh['end']})<br>{sh['end'] - sh['start']} 帧<br>
{sh['start'] / 60:.2f}s–{sh['end'] / 60:.2f}s<br>{html.escape(sh['category'])}<br>hasCharacter={sh['hasCharacter']}<br>
in: {tr['type']}/{tr['duration']}<br>conf={sh.get('confidence', '')}</td>
<td><a href="thumbs/{sh['id']}.jpg"><img src="thumbs/{sh['id']}.jpg" width="640"></a></td>
<td class="t">{insts}<hr>排除区：<br>{ex}<hr>效果：<br>{ef}<hr>{html.escape(sh.get('notes', ''))}</td></tr>""")
    page = f"""<!doctype html><meta charset="utf-8"><title>CharaForge recon — 镜头表校对</title>
<style>body{{font:13px sans-serif;background:#222;color:#ddd}} td{{vertical-align:top;border-bottom:1px solid #555;padding:6px}}
.t{{max-width:640px;font-size:12px}} a{{color:#9cf}}</style>
<h2>镜头表校对（{len(doc['shots'])} 个镜头，视频 {doc['video']['frameCount']} 帧 @ {doc['video']['fps']}fps）</h2>
<p>缩略图：每镜头均匀 12 帧（含首帧 start 与末帧 end-1），左上角为帧号；粗框=head，圆=eyes，细框=bbox，白框=clip，红色半透明=排除区。
切点逐帧复核图在 review/ 目录（s*.jpg：每行为候选切点前 4 帧+后 4 帧）。</p>
<p><a href="overview.jpg">总览图</a></p>
<p>cadence（每两帧一张新画面的段，左闭右开）：{html.escape(', '.join(f"[{c['start']},{c['end']})" for c in doc.get('cadence', [])))}</p>
<table>{''.join(rows)}</table>"""
    (DATA / "index.html").write_text(page, encoding="utf-8")


def build(no_track=False, only=None, thumbs=True):
    ann = load_json(ANNOT)
    vinfo = load_json(DATA / "video.json")
    cache = load_json(TRACK_CACHE) if TRACK_CACHE.exists() else {}
    shots = []
    for sh in ann["shots"]:
        out = {k: sh[k] for k in ("id", "start", "end", "transitionIn", "category", "hasCharacter")}
        out["instances"] = []
        for inst in sh["instances"]:
            o = {k: inst[k] for k in INST_KEYS if k in inst}
            o.setdefault("variant", {}); o.setdefault("switchEvents", []); o.setdefault("motionHints", [])
            # 第 1 次细化：人工锚点的 eyes 用虹膜颜色校正（失败则保留人工值），再拿去跟踪
            inst = dict(inst)
            ps = inst.get("eyePairSearch", True)
            # irisRefine: false（manual.py 实例字段）：整段没有青色虹膜可找（><、圈圈眼等替换眼），
            # 虹膜细化只会锁到头发/饰品上的青色，直接用人工锚点 + 跟踪传播的值
            ir = inst.get("irisRefine", True)
            inst["anchors"] = [refine_anchor(dict(a), ps) if ir else dict(a) for a in inst["anchors"]]
            key = f"{sh['id']}/{inst['id']}"
            sig = json.dumps([sh["start"], sh["end"], inst["anchors"], inst.get("trackMode", "lk")], sort_keys=True)
            if inst.get("trackMode") == "fixed":
                cache[key] = {"sig": sig, "track": [dict(a) for a in inst["anchors"]]}
            elif not no_track and (only is None or sh["id"] in only) and cache.get(key, {}).get("sig") != sig:
                from track import densify
                cache[key] = {"sig": sig, "track": densify(sh["start"], sh["end"], inst)}
                TRACK_CACHE.write_text(json.dumps(cache), encoding="utf-8")
            track = cache.get(key, {}).get("track") or [dict(a) for a in inst["anchors"]]
            track = [t for t in track if sh["start"] <= t["frame"] < sh["end"]]
            # 镜头首帧/末帧必须有采样：缺失时按「区间外保持首/尾值」补一个
            if track[0]["frame"] != sh["start"]:
                track.insert(0, {**track[0], "frame": sh["start"], "source": "hold", "confidence": 0.2})
            if track[-1]["frame"] != sh["end"] - 1:
                track.append({**track[-1], "frame": sh["end"] - 1, "source": "hold",
                              "confidence": min(0.3, track[-1].get("confidence", 0.3))})
            track = [refine_sample(dict(t), ps) if ir else dict(t) for t in track]   # 第 2 次细化：跟踪加密后的每个采样
            # 两眼都经虹膜细化的采样：rotation 用眼睛连线角度（顺时针为正）。脸部有偏航时会有偏差，仅作初值。
            for t in track:
                if t.get("_eyesRefined"):
                    if t.get("source") in ("manual", "flow", "linear", "hold"):
                        t["source"] = t["source"] + "+iris-refine"
                        t["confidence"] = min(0.9, t.get("confidence", 0.5) + 0.2)
                    (ax, ay), (bx, by) = t["eyes"]
                    t["rotation"] = float(np.degrees(np.arctan2(by - ay, bx - ax)))
            n_eye = sum(1 for t in track if t.get("eyes"))
            n_ok = sum(1 for t in track if t.get("_eyesRefined"))
            EYE_STATS[f"{sh['id']}/{inst['id']}"] = (n_ok, n_eye)
            if n_eye:
                ok_frames = [t["frame"] for t in track if t.get("_eyesRefined")]
                o["notes"] = (o.get("notes", "") + "；" if o.get("notes") else "") + (
                    f"eyes 虹膜细化 {n_ok}/{n_eye} 个采样（细化成功帧：{ok_frames}；其余为人工锚点/光流传播，误差可达 10~30px）")
            annotate_eyes_state(track, inst)
            for t in track:
                for k in ("trackRegion", "note", "_eyesRefined"):
                    t.pop(k, None)
                if t.get("bbox", 0) is None:
                    t["head"] = None
                    t.pop("eyes", None)
                if t.get("eyes") and t["eyes"][0][0] > t["eyes"][1][0]:
                    t["eyes"] = [t["eyes"][1], t["eyes"][0]]
            o["track"] = [{k: (r1(v) if k in ("bbox", "head", "eyes", "rotation", "opacity", "blurSigma") else (round(v, 2) if k == "confidence" else v)) for k, v in s.items()} for s in track]
            out["instances"].append(o)
        out["exclusions"] = sh["exclusions"]
        out["effects"] = sh["effects"]
        out["thumbnail"] = f"../../.data/recon/thumbs/{sh['id']}.jpg"
        for k in ("confidence", "notes"):
            if k in sh:
                out[k] = sh[k]
        shots.append(out)
    cad_path = DATA / "cadence.json"
    cadence = load_json(cad_path) if cad_path.exists() else []
    doc = {"$schema": "../../packages/scene/schema/shots.v1.json", "format": "charaforge.shots", "version": 1,
           "video": {"uri": VIDEO_URI, "sha256": vinfo["sha256"], "width": W, "height": H,
                     "fps": vinfo["fps"], "frameCount": vinfo["frameCount"]},
           "cadence": cadence,
           "shots": shots}
    SHOTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SHOTS_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if thumbs:
        make_thumbs(shots, only)
        make_overview(shots)
    make_index(doc)
    (DATA / "eye_stats.json").write_text(json.dumps(EYE_STATS, indent=0), encoding="utf-8")
    tot_ok = sum(a for a, _ in EYE_STATS.values()); tot = sum(b for _, b in EYE_STATS.values())
    print(f"eyes 虹膜细化成功 {tot_ok}/{tot} 个采样；锚点整体平移 {len(ANCHOR_SHIFTS)} 个：{ANCHOR_SHIFTS}")
    print("写出", SHOTS_JSON, len(shots), "个镜头")


if __name__ == "__main__":
    a = sys.argv[1:]
    only = None
    if "--only" in a:
        only = set(a[a.index("--only") + 1].split(","))
    build("--no-track" in a, only, "--no-thumbs" not in a)
