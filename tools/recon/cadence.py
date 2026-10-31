"""帧重复段（cadence）检测。

对每帧 i 与 i-1 做 320x180 RGB 逐像素差：
  D（重复帧）：差值 > 12 的像素占比 < 0.5%（H.264 重编码噪声级别；平均差可到 1 左右）
  P（准重复）：平均差 < 0.3 × 两侧帧平均差，但不满足 D（多为背景/粒子 60fps 而角色 30fps，或整帧亚像素缓动）
  N（新画面）：其余
cadence 段（every=2）判定：取相位 p∈{0,1}，把帧分成对 (k, k+1)，k≡p (mod 2)。
  对合法 ⇔ 第 k+1 帧是 D；对有“变化证据” ⇔ 第 k 帧不是 D（成对帧的第一帧带来新画面）。
  合法对的极大连续段，裁剪到第一个/最后一个有证据的对，至少 4 个有证据的对才算 cadence 段。
  这样「画面静止」（连续 D、没有交替变化）不会被单独标成 cadence，只在两侧有交替证据时被包含在段内
  （静止帧也满足“每两帧重复”的定义，不矛盾）。
段的 start 对齐成对帧的第一帧，end 为最后一对的第二帧 + 1（左闭右开）。
P 类段落不进 cadence（不满足“画面完全相同”），单独输出到 .data/recon/cadence_partial.json 供参考。

用法：tools/.venv/bin/python -I tools/recon/cadence.py   → .data/recon/cadence.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import DATA, load_small  # noqa: E402

MIN_EVIDENCE = 4


def classify(small) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = small.shape[0]
    mn = np.zeros(n)
    frac = np.zeros(n)
    for i in range(1, n):
        d = np.abs(small[i].astype(np.int16) - small[i - 1].astype(np.int16)).max(2)
        mn[i] = d.mean()
        frac[i] = (d > 12).mean()
    t = np.full(n, "N")
    for i in range(1, n):
        nb = [mn[j] for j in (i - 1, i + 1) if 1 <= j < n]
        ref = float(np.mean(nb)) if nb else 0
        if frac[i] < 0.005:
            t[i] = "D"
        elif mn[i] < 0.3 * ref:
            t[i] = "P"
    return t, mn, frac


def segments(t: np.ndarray, want: str = "D") -> list[dict]:
    n = len(t)
    segs = []
    for p in (0, 1):
        k = p
        run = []
        def flush():
            ev = [kk for kk, e in run if e]
            if len(ev) >= MIN_EVIDENCE:
                segs.append({"start": ev[0], "end": ev[-1] + 2, "every": 2, "evidence": len(ev)})
        while k + 1 < n:
            ok = t[k + 1] == want if want == "D" else t[k + 1] in ("D", "P")
            if ok:
                run.append((k, t[k] != "D"))
            else:
                flush(); run = []
            k += 2
        flush()
    # 两个相位的段若重叠（只会出现在静止画面附近），保留证据多的，另一段裁掉重叠部分
    segs.sort(key=lambda s: -s["evidence"])
    out: list[dict] = []
    for s in segs:
        for o in out:
            if s["start"] < o["end"] and o["start"] < s["end"]:
                if s["start"] < o["start"]:
                    s["end"] = min(s["end"], o["start"] - (o["start"] - s["start"]) % 2)
                else:
                    s["start"] = max(s["start"], o["end"] + (o["end"] - s["start"]) % 2)
        if s["end"] - s["start"] >= 2 * MIN_EVIDENCE:
            out.append(s)
    out.sort(key=lambda s: s["start"])
    return out


def main() -> None:
    t, mn, frac = classify(load_small())
    segs = segments(t, "D")
    part = [s for s in segments(t, "P")
            if not any(s["start"] < d["end"] and d["start"] < s["end"] and d["end"] - d["start"] > 0.8 * (s["end"] - s["start"]) for d in segs)]
    np.savez(DATA / "cadence_metrics.npz", mn=mn, frac=frac, t=t)
    (DATA / "cadence.json").write_text(json.dumps([{k: s[k] for k in ("start", "end", "every")} for s in segs], indent=0))
    (DATA / "cadence_partial.json").write_text(json.dumps(part, indent=0))
    print("D/P/N 帧数:", {c: int((t == c).sum()) for c in "DPN"})
    print("cadence 段:")
    for s in segs:
        print(f"  [{s['start']}, {s['end']}) {s['end'] - s['start']} 帧，证据 {s['evidence']} 对")
    print("准重复（D 或 P）段（参考，不写入 cadence）:")
    for s in part:
        print(f"  [{s['start']}, {s['end']}) 证据 {s['evidence']}")


if __name__ == "__main__":
    main()
