"""A5：全片批量驱动（多单元批处理 + 断点续跑 + 合成全片场景）。

    tools/.venv/bin/python -I tools/gpu/scripts/a5_batch.py [--units all|s23,s33,...] [--batch 4] [--out .data/gpu/fit/full]
        [--max-frames-per-batch 900] [--assemble-only] [--dry]

流程：
  1. 拟合单元列表 = tools/fit 的 fitUnits（`cli.ts units`；打样镜头 s15 / s24 / s43 各自成单元，同样走 planAuto）；
  2. 已完成的单元跳过（<out>/units/<单元>/done.json 的指纹 = shots.json + tools/gpu/fit 代码 + 拟合参数的哈希，一致才复用）；
  3. 剩下的按时间顺序每 --batch 个摞成一批（总帧数超过 --max-frames-per-batch 就提前截断），一批一个子进程跑 a5_fit.py
     （进程退出即释放显存与内存）；每个单元的 layers.json / fit.json 单独落盘，写 done.json；
  4. 合成全片场景 <out>/scene.json（tools/gpu/scripts/a5_assemble.ts，parseScene 校验通过才写出；不覆盖 scenes/deepseek-chan/scene.json）。
断点续跑：原样重跑同一命令即可（单元级跳过）；中断在一批中间时，这一批整批重跑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = str(REPO / "tools/.venv/bin/python")
NODE = ["node", "--experimental-strip-types", "--disable-warning=ExperimentalWarning"]


def list_units() -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "units.json"
        subprocess.run(NODE + [str(REPO / "tools/fit/src/cli.ts"), "units", "--out", str(out)], check=True, cwd=REPO, capture_output=True)
        rows = json.loads(out.read_text())
    return [{"unit": r["unit"], "shots": r["shots"], "range": r["range"]} for r in rows]


def fingerprint(fit_args: list[str]) -> str:
    h = hashlib.sha256()
    h.update((REPO / "scenes/deepseek-chan/shots.json").read_bytes())
    for p in sorted((REPO / "tools/gpu/fit").glob("*.py")) + [REPO / "tools/gpu/scripts/a5_fit.py", REPO / "tools/gpu/scripts/a5_plan.ts",
                                                              REPO / "tools/fit/src/plans.ts"]:
        h.update(p.read_bytes())
    h.update(json.dumps(fit_args).encode())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="all")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-frames-per-batch", type=int, default=900)
    ap.add_argument("--out", default=str(REPO / ".data/gpu/fit/full"))
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--fit-args", default="", help="追加给 a5_fit.py 的参数（进指纹）")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    logf = open(out / "batch.log", "a", encoding="utf-8")

    def log(s: str) -> None:
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    fit_args = args.fit_args.split() if args.fit_args else []
    fp = fingerprint(fit_args)
    units = list_units()
    if args.units != "all":
        want = set(args.units.split(","))
        units = [u for u in units if u["unit"] in want or set(u["shots"]) & want]

    def done(u):
        f = out / "units" / u["unit"] / "done.json"
        return f.exists() and json.loads(f.read_text()).get("fp") == fp

    pending = [u for u in units if not done(u)]
    log(f"单元 {len(units)} 个，已完成 {len(units) - len(pending)}，待拟合 {len(pending)}（指纹 {fp}）")
    batches, cur = [], []
    for u in pending:
        n = u["range"][1] - u["range"][0]
        if cur and (len(cur) >= args.batch or sum(x["range"][1] - x["range"][0] for x in cur) + n > args.max_frames_per_batch):
            batches.append(cur)
            cur = []
        cur.append(u)
    if cur:
        batches.append(cur)
    for k, b in enumerate(batches):
        log(f"批 {k}：{[u['unit'] for u in b]}（{sum(u['range'][1] - u['range'][0] for u in b)} 帧）")
    if args.dry or args.assemble_only:
        batches = []
    t_all = time.time()
    frames_all = 0
    for k, b in enumerate(batches):
        bdir = out / "batches" / "_".join(u["unit"] for u in b)
        if bdir.exists():
            shutil.rmtree(bdir)
        bdir.mkdir(parents=True)
        t0 = time.time()
        cmd = [PY, "-I", str(REPO / "tools/gpu/scripts/a5_fit.py"), "--shot", ",".join(u["shots"][0] for u in b), "--out", str(bdir)] + fit_args
        with open(bdir / "stdout.log", "w") as so:
            r = subprocess.run(cmd, cwd=REPO, stdout=so, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if r.returncode != 0:
            log(f"批 {k} 失败（返回码 {r.returncode}，{dt:.0f}s），见 {bdir / 'stdout.log'}；继续下一批")
            continue
        nfr = 0
        for u in b:
            src = bdir / u["unit"] if len(b) > 1 else bdir
            dst = out / "units" / u["unit"]
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            for name in ("layers.json", "fit.json"):
                shutil.copy(src / name, dst / name)
            nfr += u["range"][1] - u["range"][0]
            (dst / "done.json").write_text(json.dumps({"fp": fp, "batch": [x["unit"] for x in b], "sec": dt, "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
        frames_all += nfr
        log(f"批 {k} 完成：{[u['unit'] for u in b]}，{nfr} 帧，{dt:.0f}s（{nfr / dt * 60:.1f} 帧/分钟）")
    if batches:
        el = time.time() - t_all
        log(f"本次共 {frames_all} 帧，{el:.0f}s（{frames_all / max(el, 1) * 60:.1f} 帧/分钟）")
    # 合成全片场景
    r = subprocess.run(NODE + [str(REPO / "tools/gpu/scripts/a5_assemble.ts"), "--units", str(out / "units"), "--out", str(out / "scene.json")],
                       cwd=REPO, capture_output=True, text=True)
    log((r.stdout + r.stderr).strip())
    if r.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
