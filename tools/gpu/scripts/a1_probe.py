"""A1 探测：把一个 art mesh 挂到指定变形器下、顶点换成探测点（只在内存里改 moc 字节），
用 Core 求出该变形器对任意局部点的世界坐标，与 deform.py 的结果逐点比较。用于定位变形器语义差异。

    from a1_probe import probe
    core_px, ours_px = probe(deformer=233, pts=np.array([[0.5, 0.5], ...]), params=P)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1_core import MOC, run_core  # noqa: E402
from l2d import moc3  # noqa: E402
from l2d.deform import L2D  # noqa: E402


def pick_mesh(m: moc3.Moc3, need: int) -> int:
    """挑一个顶点数够、没有 blend shape、没有 glue 的 mesh 作为探针。"""
    g = m.groups
    bad = set(g['blendShapesArtMeshes']['targetIndices'].tolist()) if m.version >= 4 else set()
    bad |= set(g['glue']['artMeshIndicesA'].tolist()) | set(g['glue']['artMeshIndicesB'].tolist())
    vc = g['artMeshes']['vertexCounts']
    for i in np.argsort(-vc):
        if int(i) not in bad and vc[i] >= need:
            return int(i)
    raise ValueError('没有合适的探针 mesh')


def make_patches(m: moc3.Moc3, mesh: int, deformer: int, pts: np.ndarray, extra: list | None = None):
    g = m.groups
    vc = int(g['artMeshes']['vertexCounts'][mesh])
    P = np.empty((vc, 2), np.float32)
    P[:len(pts)] = pts
    P[len(pts):] = pts[0]
    patches = [(m.offsets['artMeshes.parentDeformerIndices'] + 4 * mesh, struct.pack('<i', deformer))]
    b, n = int(g['artMeshes']['keyformSourcesBeginIndices'][mesh]), int(g['artMeshes']['keyformSourcesCounts'][mesh])
    for k in range(b, b + n):
        pb = int(g['artMeshKeyforms']['keyformPositionSourcesBeginIndices'][k])
        patches.append((m.offsets['keyformPositions.xys'] + 4 * pb, P.tobytes()))
        # 不透明度设为 1，免得被判不可见
        patches.append((m.offsets['artMeshKeyforms.opacities'] + 4 * k, struct.pack('<f', 1.0)))
    return patches + (extra or [])


def apply_patches(buf: bytes, patches) -> bytes:
    b = bytearray(buf)
    for off, data in patches:
        b[off:off + len(data)] = data
    return bytes(b)


def probe(deformer: int, pts: np.ndarray, params: np.ndarray, extra: list | None = None, dtype=torch.float64, mesh: int | None = None):
    """返回 (Core 像素 [N, K, 2], 我们的像素 [N, K, 2], mesh 号)。extra：额外的字节补丁 [(offset, bytes)]。"""
    buf = Path(MOC).read_bytes()
    m = moc3.parse(buf)
    pts = np.asarray(pts, np.float32)
    mesh = pick_mesh(m, len(pts)) if mesh is None else mesh
    patches = make_patches(m, mesh, deformer, pts, extra)
    ref = run_core(np.atleast_2d(params), patches=patches)
    m2 = moc3.parse(apply_patches(buf, patches))
    model = L2D(m2, dtype=dtype)
    with torch.no_grad():
        o = model(torch.as_tensor(np.atleast_2d(params)))
    vo = model.vertex_offsets
    K = len(pts)
    core = ref['verts'][:, vo[mesh]:vo[mesh] + K]
    ours = o.verts[:, vo[mesh]:vo[mesh] + K].double().cpu().numpy()
    return core, ours, mesh
