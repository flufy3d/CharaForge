"""A1：调用 Node 版 Cubism Core（a1_core_dump.mjs）求一批参数的参考输出，并读回 numpy。

用法（Python）：
    from a1_core import run_core, load_ref
    ref = run_core(params)            # params: [N, P] float32；返回 dict（顶点已换成画布像素）
    ref = load_ref('.data/gpu/ref/random')
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MOC = REPO / 'assets/models/deepseek-chan/DeepSeek/大肥鱼.moc3'
SCRIPT = REPO / 'tools/gpu/scripts/a1_core_dump.mjs'
SCRATCH = Path(os.environ.get('A1_SCRATCH', Path.home() / 'cf-scratch/a1'))


def load_ref(prefix: str | Path, pixels: bool = True) -> dict:
    """读 a1_core_dump.mjs 的输出。pixels=True 时顶点换成画布像素（float64）：x = originX + x·ppu，y = originY − y·ppu。"""
    prefix = str(prefix)
    meta = json.loads(Path(prefix + '.json').read_text())
    raw = np.memmap(prefix + '.bin', 'u1', 'r')
    out = {'meta': meta}
    for name, s in meta['sections'].items():
        dt = {'f4': '<f4', 'i4': '<i4', 'u1': 'u1'}[s['dtype']]
        n = int(np.prod(s['shape']))
        out[name] = np.frombuffer(raw, dt, n, s['offset']).reshape(s['shape'])
    N, D = meta['n'], meta['D']
    out['multiply'] = out['multiply'].reshape(N, D, 4)
    out['screen'] = out['screen'].reshape(N, D, 4)
    v = out['verts'].reshape(N, -1, 2)
    if pixels:
        c = meta['canvas']
        v = v.astype(np.float64)
        v = np.stack([c['originX'] + v[..., 0] * c['ppu'], c['originY'] - v[..., 1] * c['ppu']], -1)
    out['verts'] = v
    out['visible'] = (out['dynFlags'] & 1).astype(bool)
    if Path(prefix + '.params.npy').exists():
        out['params'] = np.load(prefix + '.params.npy')
    if Path(prefix + '.parts.npy').exists():
        out['parts'] = np.load(prefix + '.parts.npy')
    if Path(prefix + '.static.json').exists():
        # 静态信息（a1_core_dump.mjs 的 static 选项）：json + 二进制里的 UV（Core 口径）与三角形索引
        st = json.loads(Path(prefix + '.static.json').read_text())
        sb = Path(prefix + '.static.bin').read_bytes()
        V, I = int(np.sum(st['drawables']['vertexCounts'])), int(np.sum(st['drawables']['indexCounts']))
        st['uvs'] = np.frombuffer(sb, '<f4', 2 * V, st['uvOffset']).reshape(-1, 2)
        st['indices'] = np.frombuffer(sb, '<i4', I, st['idxOffset'])
        out['static'] = st
    out['prefix'] = prefix
    return out


def run_core(params: np.ndarray, parts: np.ndarray | None = None, patches: list | None = None, out: str | Path | None = None,
             moc: str | Path = MOC, static: bool = False, pixels: bool = True) -> dict:
    """对 params [N, P] 逐个调用 Core，返回 load_ref 的结果。out 缺省写到临时目录。"""
    params = np.ascontiguousarray(params, np.float32)
    N = params.shape[0]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = None
    if out is None:
        tmp = tempfile.mkdtemp(dir=SCRATCH)
        out = Path(tmp) / 'ref'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + '.params.npy', params)
    params.tofile(str(out) + '.params.bin')
    job = {'moc': str(moc), 'n': N, 'params': str(out) + '.params.bin', 'out': str(out), 'static': static}
    if parts is not None:
        parts = np.ascontiguousarray(parts, np.float32)
        np.save(str(out) + '.parts.npy', parts)
        parts.tofile(str(out) + '.parts.bin')
        job['parts'] = str(out) + '.parts.bin'
    if patches:
        job['patches'] = [{'offset': int(o), 'hex': bytes(b).hex()} for o, b in patches]
    jf = str(out) + '.job.json'
    Path(jf).write_text(json.dumps(job))
    r = subprocess.run(['node', str(SCRIPT), jf], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'a1_core_dump 失败：{r.stderr[-2000:]}')
    Path(str(out) + '.params.bin').unlink()
    if parts is not None:
        Path(str(out) + '.parts.bin').unlink()
    res = load_ref(out, pixels=pixels)
    if tmp is not None:
        # 临时目录：读进内存后删掉
        res = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in res.items()}
        shutil.rmtree(tmp, ignore_errors=True)
        res['prefix'] = None
    return res
