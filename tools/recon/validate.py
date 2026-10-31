"""校验 scenes/deepseek-chan/shots.json：
  1. JSON Schema（draft-07，packages/scene/schema/shots.v1.json）结构校验；
  2. 语义检查：
     - 镜头首尾相接覆盖 [0, frameCount)，end > start，ID 唯一；
     - 视频元数据与实际文件一致（尺寸/帧率/帧数，--sha 时校验 sha256）；
     - 实例：ID 镜头内唯一；track 按帧升序、都在 [start, end) 内、首尾帧（start 与 end-1）都有采样；
       bbox/head 在画面附近（允许出画 50%）、宽高 > 0；eyes 两点按画面左/右排序；
     - variant / switchEvents 的参数 ID 必须是 cdi3 中的开关参数（key*/Clothes*/Movement*），取值在 moc3 范围 [0,1] 内，
       variant 不能列默认值；switchEvents/motionHints 帧号在镜头内；motionHints 引用的 motion3 文件存在；
     - exclusions/effects 区间在镜头内且 end > start，rects 关键帧在区间内；
     - hasCharacter=false 的镜头不应有实例；hasCharacter=true 的镜头至少一个实例；
     - cadence 段升序不重叠、在视频范围内（--check-cadence 时逐帧核对段内重复帧确实与前一帧相同）；
     - 实例 ID 全局唯一；continues 指向更早镜头里存在的实例（不是紧邻上一镜头时警告）；
     - clip 关键帧升序且在镜头区间内；effects.instances 引用本镜头实例、strength 升序；
     - eyesState=hidden 时不能有 eyes，不可见帧（bbox=null）的 eyesState 只能是 hidden。
用法：tools/.venv/bin/python -I tools/recon/validate.py [path] [--sha]
退出码非 0 表示失败。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from jsonschema import Draft7Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_DIR, SCHEMA, SHOTS_JSON, VIDEO, load_json, probe, sha256  # noqa: E402

SWITCH_RE = re.compile(r"^(key\d+|Clothes\d+|Movement\d+)$")
# 由 moc3（Cubism Core 读取）核对得到的默认值；范围均为 [0, 1]
SWITCH_DEFAULTS = {**{f"key{i}": 0 for i in range(1, 14)}, "Movement1": 0, "Movement2": 0, "Movement3": 0,
                   "Clothes1": 1, "Clothes2": 0, "Clothes3": 0, "Clothes4": 0, "Clothes5": 1}


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv and not argv[0].startswith("--") else SHOTS_JSON
    doc = load_json(path)
    errs: list[str] = []
    warns: list[str] = []

    schema = load_json(SCHEMA)
    for e in sorted(Draft7Validator(schema).iter_errors(doc), key=lambda e: list(e.path)):
        errs.append(f"schema: /{'/'.join(map(str, e.path))}: {e.message}")
    if errs:
        print("\n".join(errs[:30]))
        print(f"失败：{len(errs)} 个结构错误")
        return 1

    cdi = load_json(MODEL_DIR / "大肥鱼.cdi3.json")
    cdi_ids = {p["Id"] for p in cdi["Parameters"]}
    motions = {p.name[: -len(".motion3.json")] for p in (MODEL_DIR / "Animation").glob("*.motion3.json")}

    v = doc["video"]
    info = probe()
    for k in ("width", "height", "fps", "frameCount"):
        if v[k] != info[k]:
            errs.append(f"video.{k}={v[k]} 与实际 {info[k]} 不符")
    if not (Path(__file__).resolve().parents[2] / v["uri"]).resolve() == VIDEO.resolve():
        warns.append(f"video.uri={v['uri']} 不是预期的 {VIDEO}")
    if "--sha" in argv and v["sha256"] != sha256():
        errs.append("video.sha256 与文件不符")
    W, H, N = v["width"], v["height"], v["frameCount"]

    # cadence：按起点升序、互不重叠、在 [0, N) 内、长度至少 every 帧；段内成对帧与视频实际差分相符（抽查）
    cad = doc.get("cadence") or []
    prev_end = 0
    for k, c in enumerate(cad):
        if not (0 <= c["start"] < c["end"] <= v["frameCount"]):
            errs.append(f"cadence[{k}] [{c['start']},{c['end']}) 超出视频范围")
        if c["start"] < prev_end:
            errs.append(f"cadence[{k}] [{c['start']},{c['end']}) 与前一段重叠或未按起点排序")
        if c["end"] - c["start"] < c["every"]:
            errs.append(f"cadence[{k}] 长度小于 every")
        if (c["end"] - c["start"]) % c["every"]:
            warns.append(f"cadence[{k}] 长度不是 every 的整数倍（末尾不完整的一组）")
        prev_end = max(prev_end, c["end"])
    if cad and "--check-cadence" in argv:
        import numpy as np
        from common import load_small
        sm = load_small()
        bad = 0
        for c in cad:
            for f in range(c["start"], c["end"]):
                if (f - c["start"]) % c["every"]:
                    d = np.abs(sm[f].astype(np.int16) - sm[f - 1].astype(np.int16)).max(2)
                    if (d > 12).mean() >= 0.005:
                        bad += 1
        if bad:
            errs.append(f"cadence 段内有 {bad} 帧与前一帧并不相同")

    cursor = 0
    ids = set()
    inst_shot: dict[str, int] = {}   # 全局实例 ID → 镜头序号
    for sh in doc["shots"]:
        sid, s, e = sh["id"], sh["start"], sh["end"]
        p = f"镜头 {sid}"
        if sid in ids:
            errs.append(f"{p}: ID 重复")
        ids.add(sid)
        if s != cursor:
            errs.append(f"{p}: start={s} 与上一镜头 end={cursor} 不衔接")
        if not e > s:
            errs.append(f"{p}: end 必须大于 start")
        if s != int(s) or e != int(e):
            errs.append(f"{p}: 切点必须是整数帧")
        cursor = e
        inside = lambda f: s <= f < e  # noqa: E731
        if not sh["hasCharacter"] and sh["instances"]:
            errs.append(f"{p}: hasCharacter=false 却有实例")
        if sh["hasCharacter"] and not sh["instances"]:
            errs.append(f"{p}: hasCharacter=true 但没有实例")
        if sh["transitionIn"]["type"] == "cut" and sh["transitionIn"]["duration"] != 0:
            errs.append(f"{p}: 硬切 duration 应为 0")
        iids = set()
        cur = len(ids) - 1   # 本镜头序号
        for inst in sh["instances"]:
            q = f"{p}/{inst['id']}"
            if inst["id"] in iids:
                errs.append(f"{q}: 实例 ID 重复")
            if inst["id"] in inst_shot:
                errs.append(f"{q}: 实例 ID 与更早镜头重复（本表约定实例 ID 全局唯一）")
            iids.add(inst["id"])
            cont = inst.get("continues")
            if cont is not None:
                if cont not in inst_shot:
                    errs.append(f"{q}: continues={cont} 不是更早镜头里的实例")
                elif inst_shot[cont] != cur - 1:
                    warns.append(f"{q}: continues={cont} 不在紧邻的上一镜头")
            for f in inst["switchEvents"]:
                if f.get("fade", 0) > 0 and f["frame"] + f["fade"] > e + 60:
                    warns.append(f"{q}: switchEvent {f['param']} 淡入远超镜头结束")
            ck = inst.get("clip")
            if ck:
                fr = [c["frame"] for c in ck]
                if fr != sorted(fr) or len(set(fr)) != len(fr):
                    errs.append(f"{q}: clip 关键帧必须严格升序")
                for c in ck:
                    if not inside(c["frame"]):
                        errs.append(f"{q}: clip 关键帧 {c['frame']} 不在镜头区间内")
                    if c["rect"][2] <= 0 or c["rect"][3] <= 0:
                        errs.append(f"{q}: clip 宽高必须为正")
            for k, val in inst["variant"].items():
                if not SWITCH_RE.match(k) or k not in cdi_ids:
                    errs.append(f"{q}: variant 参数 {k} 不是 cdi3 中的开关参数")
                elif not 0 <= val <= 1:
                    errs.append(f"{q}: variant {k}={val} 超出 [0,1]")
                elif val == SWITCH_DEFAULTS.get(k):
                    errs.append(f"{q}: variant {k}={val} 是默认值，不应列出")
            for ev in inst["switchEvents"]:
                if not SWITCH_RE.match(ev["param"]) or ev["param"] not in cdi_ids:
                    errs.append(f"{q}: switchEvent 参数 {ev['param']} 不是 cdi3 中的开关参数")
                if not 0 <= ev["value"] <= 1:
                    errs.append(f"{q}: switchEvent {ev['param']}={ev['value']} 超出 [0,1]")
                if not inside(ev["frame"]):
                    errs.append(f"{q}: switchEvent 帧 {ev['frame']} 不在镜头内")
            for mh in inst["motionHints"]:
                if mh["motion"] not in motions:
                    errs.append(f"{q}: motion {mh['motion']} 不存在")
                if mh["start"] >= e:
                    errs.append(f"{q}: motionHint 起点 {mh['start']} 在镜头结束之后")
            tr = inst["track"]
            frames = [t["frame"] for t in tr]
            if frames != sorted(frames) or len(set(frames)) != len(frames):
                errs.append(f"{q}: track 帧号必须严格升序")
            if any(not inside(f) for f in frames):
                errs.append(f"{q}: track 有采样不在镜头区间 [{s},{e}) 内")
            if frames[0] != s or frames[-1] != e - 1:
                errs.append(f"{q}: track 必须包含镜头首帧 {s} 和末帧 {e - 1}（实际 {frames[0]}..{frames[-1]}）")
            for t in tr:
                for key in ("bbox", "head"):
                    r = t.get(key)
                    if r is None:
                        continue
                    x, y, w, h = r
                    if w <= 0 or h <= 0:
                        errs.append(f"{q}@{t['frame']}: {key} 宽高必须为正")
                    elif x + w < -0.5 * w or y + h < -0.5 * h or x > W + 0.5 * w or y > H + 0.5 * h:
                        warns.append(f"{q}@{t['frame']}: {key} 大幅出画")
                if t.get("eyes") and t["eyes"][0][0] > t["eyes"][1][0]:
                    errs.append(f"{q}@{t['frame']}: eyes 应按画面左、右排序")
            for t in tr:
                if t.get("eyesState") in ("hidden",) and t.get("eyes"):
                    errs.append(f"{q}@{t['frame']}: eyesState=hidden 却有 eyes")
                if t.get("bbox", 0) is None and t.get("eyesState") not in (None, "hidden"):
                    errs.append(f"{q}@{t['frame']}: 不可见帧的 eyesState 应为 hidden")
        for ex in sh["exclusions"]:
            if not (s <= ex["start"] < ex["end"] <= e):
                errs.append(f"{p}: exclusion {ex['reason']} 区间 [{ex['start']},{ex['end']}) 不在镜头内")
            for r in ex["rects"]:
                if not ex["start"] <= r["frame"] < ex["end"]:
                    warns.append(f"{p}: exclusion {ex['reason']} 关键帧 {r['frame']} 不在其区间内")
                if r["rect"][2] <= 0 or r["rect"][3] <= 0:
                    errs.append(f"{p}: exclusion {ex['reason']} 矩形宽高必须为正")
        for fx in sh["effects"]:
            if not (s <= fx["start"] < fx["end"] <= e):
                errs.append(f"{p}: effect {fx['kind']} 区间 [{fx['start']},{fx['end']}) 不在镜头内")
            for i in fx.get("instances") or []:
                if i not in iids:
                    errs.append(f"{p}: effect {fx['kind']} 引用了本镜头不存在的实例 {i}")
            st = [x["frame"] for x in fx.get("strength") or []]
            if st != sorted(st):
                errs.append(f"{p}: effect {fx['kind']} strength 关键帧未升序")
            if any(not (fx["start"] <= f <= fx["end"]) for f in st):
                warns.append(f"{p}: effect {fx['kind']} strength 关键帧超出效果区间")
        for i in iids:
            inst_shot[i] = cur
    if cursor != N:
        errs.append(f"镜头没有覆盖到最后一帧（{cursor} ≠ {N}）")

    for w in warns:
        print("警告:", w)
    for x in errs:
        print("错误:", x)
    if errs:
        print(f"失败：{len(errs)} 个错误，{len(warns)} 个警告")
        return 1
    n_inst = sum(len(s["instances"]) for s in doc["shots"])
    print(f"通过：{len(doc['shots'])} 个镜头，{n_inst} 个实例，{len(warns)} 个警告")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
