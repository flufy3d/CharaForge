"""模型装配：moc3（A1 变形器）+ physics3（A2 物理）+ 贴图与 drawable 静态数据（A3 光栅化），全部在 GPU 上。

静态数据直接从 moc3 生成（与 A3 dump 的 model_<ID>.json 同一口径，`check_static` 可逐项比对），不依赖 .data 里的 dump。
参数写入语义（runtime instance.setParamByIndex → Framework setParameterValueByIndex）：每次写都夹到 [min, max]
（repeat 参数按周期折回），存进 Float32Array。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

GPU_DIR = Path(__file__).resolve().parents[1]
if str(GPU_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_DIR))

from l2d import moc3  # noqa: E402
from l2d.deform import L2D  # noqa: E402
from l2d.physics import PhysicsRig  # noqa: E402
from l2d.raster import ModelGeom, Rasterizer, TextureSet  # noqa: E402

REPO = GPU_DIR.parents[1]
MODEL_DIR = REPO / "assets/models/deepseek-chan/DeepSeek"
MODEL3 = MODEL_DIR / "大肥鱼.model3.json"
VTUBE = MODEL_DIR / "大肥鱼.vtube.json"
IRIS_MESHES = ("ArtMesh236", "ArtMesh134")  # 两眼虹膜主网格（与 tools/fit 的锚点先验同一组：画面左眼、右眼）
CLIP_MAX_DEFAULT = 36
CLIP_MAX_MULTI = 32


def static_from_moc(m) -> dict:
    """A3 ModelGeom 需要的静态数据（与 a3_dump 的 model_<ID>.json 同字段、同口径）。"""
    st = moc3.drawable_static(m)
    D = len(st["ids"])
    vo = np.asarray(st["vertex_offsets"], np.int64)
    idx = st["indices"]
    io = np.concatenate([[0], np.cumsum([len(t) * 3 for t in idx])])
    masks = st["masks"]
    mo = np.concatenate([[0], np.cumsum([len(x) for x in masks])])
    # clipping context 个数（findSameClip 语义，与 ModelGeom 相同）→ Framework 的 mask 缓冲张数
    ctx: list[list[int]] = []
    for mk in masks:
        if not mk:
            continue
        if not any(len(c) == len(mk) and all(x in mk for x in c) for c in ctx):
            ctx.append(list(mk))
    rt = 1 if len(ctx) <= CLIP_MAX_DEFAULT else math.ceil(len(ctx) / CLIP_MAX_MULTI)
    cv = m.canvas
    return {
        "canvas": {"width": cv["width"], "height": cv["height"], "originX": cv["originX"], "originY": cv["originY"],
                   "pixelsPerUnit": cv["pixelsPerUnit"]},
        "drawableCount": D,
        "drawableIds": st["ids"],
        "maskBufferSize": 2048,
        "maskRenderTextureCount": rt,
        "arrays": {
            "vertexOffsets": vo,
            "indexOffsets": io,
            "maskOffsets": mo,
            "uvs": st["uvs"],
            "indices": np.concatenate([t.reshape(-1) for t in idx]).astype(np.uint16),
            "masks": np.asarray([x for mk in masks for x in mk], np.int32),
            "textureIndices": st["texture"].astype(np.int32),
            "constantFlags": st["flags"].astype(np.uint8),
            "parentPartIndices": st["parent_part"].astype(np.int32),
        },
    }


def check_static(st: dict, dump_model: dict) -> list[str]:
    """与 A3 dump 的静态数据逐项比对，返回不一致项（空 = 一致）。"""
    bad = []
    for k in ("vertexOffsets", "indexOffsets", "maskOffsets", "indices", "masks", "textureIndices", "constantFlags"):
        a, b = np.asarray(st["arrays"][k]).astype(np.int64), np.asarray(dump_model["arrays"][k]).astype(np.int64)
        if a.shape != b.shape or not np.array_equal(a, b):
            bad.append(k)
    if not np.allclose(st["arrays"]["uvs"], dump_model["arrays"]["uvs"], atol=1e-7):
        bad.append("uvs")
    for k in ("maskBufferSize", "maskRenderTextureCount", "drawableCount"):
        if st[k] != dump_model[k]:
            bad.append(k)
    return bad


class Rig:
    """一个模型在 GPU 上的完整前向：参数 → 物理 → 变形 → 光栅化。"""

    def __init__(self, device="cuda", phys_compile=False, deform_dtype=torch.float32):
        self.device = torch.device(device)
        self.moc = moc3.load(str(MODEL_DIR / "大肥鱼.moc3"))
        # deform_dtype = float64：A1 的 float32 稀疏矩阵乘（warp 网格的 keyform 插值）不是逐次确定的（同一输入两次差约 0.0065 px），
        # 拟合里这点噪声会经光流目标的阈值判断被放大，导致同一配置两次运行结果不同；float64 时噪声降到 1e-12 量级
        self.l2d = L2D(self.moc, device=device, dtype=deform_dtype)
        self.param_ids = list(self.l2d.param_ids)
        self.P = len(self.param_ids)
        self.pindex = {k: i for i, k in enumerate(self.param_ids)}
        self.pmin = np.asarray(self.l2d.param_min, np.float64)
        self.pmax = np.asarray(self.l2d.param_max, np.float64)
        self.pdef = np.asarray(self.l2d.param_default, np.float64)
        self.repeat = np.asarray(self.moc.groups["parameters"]["isRepeat"]).astype(bool)
        self.part_ids = list(self.l2d.part_ids)
        self.drawable_ids = list(self.l2d.drawable_ids)
        phys = json.loads((MODEL_DIR / "大肥鱼.physics3.json").read_text(encoding="utf-8"))
        self.phys = PhysicsRig(phys, self.param_ids, self.pmin.astype(np.float32).tolist(), self.pmax.astype(np.float32).tolist(),
                               device=device, dtype=torch.float64)
        if phys_compile:
            self.phys.compile()
        self.static = static_from_moc(self.moc)
        m3 = json.loads(MODEL3.read_text(encoding="utf-8"))
        self.raster = Rasterizer(ModelGeom(self.static, device), TextureSet([MODEL_DIR / t for t in m3["FileReferences"]["Textures"]], device))
        vo = self.l2d.vertex_offsets
        self.iris = [(int(vo[self.drawable_ids.index(d)]), int(vo[self.drawable_ids.index(d) + 1])) for d in IRIS_MESHES]
        self.tmin = torch.tensor(self.pmin, device=device, dtype=torch.float32)
        self.tmax = torch.tensor(self.pmax, device=device, dtype=torch.float32)
        self.tdef = torch.tensor(self.pdef, device=device, dtype=torch.float32)

    def idx(self, pid: str) -> int:
        return self.pindex[pid]

    def clamp(self, i: int, v: torch.Tensor) -> torch.Tensor:
        """Framework 的写参数语义（夹取）；repeat 参数不支持（本模型的 4 个 repeat 参数没有 keyform，拟合不写它们）。"""
        if self.repeat[i]:
            raise ValueError(f"repeat 参数 {self.param_ids[i]} 不在拟合范围")
        return v.clamp(float(np.float32(self.pmin[i])), float(np.float32(self.pmax[i])))

    use_graph = False  # A7 的 CUDA graph（L2D.graphed）；批大小按 GRAPH_BUCKET 补齐成少数几种，图可以复用

    GRAPH_BUCKET = 16

    def deform(self, params: torch.Tensor):
        """[B, P] 物理之后的参数 → A1 输出。use_graph 时把 B 补齐到 16 的倍数（重复最后一行）走 CUDA graph，再截回 B；
        输出与普通 forward 逐位相同（A7 实测），被补的行不参与求导（截掉了）。"""
        x = params.to(self.l2d.dtype)
        B = x.shape[0]
        if not self.use_graph or B == 0:
            return self.l2d(x)
        Bp = -(-B // self.GRAPH_BUCKET) * self.GRAPH_BUCKET
        if Bp > B:
            x = torch.cat([x, x[-1:].detach().expand(Bp - B, -1)], 0)
        grad = torch.is_grad_enabled() and x.requires_grad
        out = self.l2d.graphed(Bp, grad=grad, max_cached=10)(x)
        if Bp == B:
            return out
        import dataclasses
        return dataclasses.replace(out, **{f.name: getattr(out, f.name)[:B] for f in dataclasses.fields(out)})

    def _iris_offsets(self) -> torch.Tensor:
        """虹膜中心标定（只用模型本身）：参考姿态（缺省参数）下渲染两眼区域，取各虹膜网格附近「青色」像素（可见的虹膜；
        被眼白 clip 与上眼睑遮住的部分自然不计）的质心，与网格顶点质心之差（画布像素）。shots 的 eyes 锚点是可见虹膜的中心，
        顶点质心比它偏上（网格含被遮住的上半部分）。返回 [2, 2]，拟合时加在顶点质心上（随图层变换一起缩放旋转）。"""
        from .render import raster_layer
        with torch.no_grad():
            out = self.l2d(self.tdef[None])
            c = torch.stack([out.verts[0, a:b].mean(0) for a, b in self.iris]).double()
            mid = c.mean(0)
            d = float((c[1] - c[0]).norm())
            k = 1024 / (2.0 * d)  # 两眼区域放大到 1024 宽
            ox, oy = 512 - k * float(mid[0]), 256 - k * float(mid[1])
            ts = torch.tensor([[k, 0, ox], [0, k, oy]], device=self.device)
            buf, _ = raster_layer(self, self.tdef[None], ts[None].float(), 1024, 512, aa=False, quantize=True)
            img = buf[0]
            a = img[..., 3].clamp_min(1e-6)
            r, g, b = (img[..., i] / a for i in range(3))
            cyan = (img[..., 3] > 0.9) & (b > 0.6) & (g > 0.45) & (r < 0.5) & (b - r > 0.3)
            ys, xs = torch.meshgrid(torch.arange(512, device=self.device), torch.arange(1024, device=self.device), indexing="ij")
            px = (xs.double() + 0.5 - ox) / k
            py = (ys.double() + 0.5 - oy) / k
            offs = []
            for j in range(2):
                near = cyan & ((px - c[j, 0]) ** 2 + (py - c[j, 1]) ** 2 < (0.3 * d) ** 2)
                if int(near.sum()) < 50:
                    offs.append(torch.zeros(2, device=self.device, dtype=torch.float64))
                    continue
                offs.append(torch.stack([px[near].mean(), py[near].mean()]) - c[j])
            self.iris_calib = {"meshCentroid": c.tolist(), "offset": torch.stack(offs).tolist(), "eyeDist": d,
                               "cyanPixels": [int((cyan & ((px - c[j, 0]) ** 2 + (py - c[j, 1]) ** 2 < (0.3 * d) ** 2)).sum()) for j in range(2)]}
            return torch.stack(offs).float()

    def iris_centers(self, verts: torch.Tensor) -> torch.Tensor:
        """[B, V, 2] 画布像素 → [B, 2, 2] 两只虹膜的中心（顶点质心 + 标定偏移，画布像素）。"""
        if not hasattr(self, "_ioff"):
            self._ioff = self._iris_offsets()
        return torch.stack([verts[:, a:b].mean(1) for a, b in self.iris], 1) + self._ioff.to(verts.dtype)
