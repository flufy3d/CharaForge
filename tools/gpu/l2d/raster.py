"""可微分的 Live2D 图层光栅化（nvdiffrast + PyTorch），对齐 runtime（Cubism Framework R5 的 WebGL 渲染器，MSAA 0）。

输入（与 A1 变形器输出同一口径）：每个图层实例每帧的 drawable 数据
  positions   [B, V, 2]  顶点，画布像素（moc3 画布左上为原点，y 向下）；V = 全部 drawable 顶点数之和（按 drawable 顺序拼接）
  opacities   [B, n]     drawable 不透明度（Core）
  multiply    [B, n, 4]  乘算色（Core，rgba，只用 rgb）
  screen      [B, n, 4]  屏幕色（Core）
  render_orders [B, n]   Core 的 render order（drawable i 在绘制序列中的位置）
  dyn_flags   [B, n]     Core 的动态标志（bit0 可见，bit5 顶点已更新）
  to_screen   [B, 2, 3]  画布像素 → 合成像素的仿射矩阵 [[a, c, e], [b, d, f]]（图层变换 ∘ 相机，规范第 2 条）
输出：图层缓冲（合成分辨率，预乘 alpha）[B, H, W, 4]，float32，0..1。

与 Framework 对齐的要点（详细说明见 tools/gpu/docs/raster.md）：
  - 绘制顺序：按 render order 依次画；只画动态标志「可见」的 drawable（不透明度 0 的贡献恰为 0，也跳过）。
  - 混合：Normal (ONE, ONE_MINUS_SRC_ALPHA)、Additive (ONE, ONE; alpha 不变)、Multiplicative (DST_COLOR, ONE_MINUS_SRC_ALPHA; alpha 不变)。
  - 着色：tex = texture(预乘贴图)；tex.rgb *= multiply；tex.rgb = tex.rgb + screen·tex.a − tex.rgb·screen；color = tex·opacity。
  - 贴图：createImageBitmap 预乘（8 位取整）+ generateMipmap（逐级 2×2 平均、8 位取整）+ LINEAR_MIPMAP_LINEAR，CLAMP_TO_EDGE。
  - clipping mask：上下文的建立（findSameClip）、布局（setupLayoutBounds：多缓冲、RGBA 通道、1/2/4/9 分格）、
    被裁切 drawable 的包围盒（含 Number.MIN_VALUE 初值的怪癖）与 5% 外扩、mask 矩阵，全部照 Framework 移植；
    mask 缓冲按 Framework 的分辨率（缺省 2048）在各上下文的布局矩形里实际光栅化（只画「顶点已更新」的 mask drawable，
    值 = Π(1 − tex.a)，与可见性和不透明度无关），绘制时按 clipPos 双线性采样，反转遮罩取 1 − maskVal。
  - 背面剔除：Framework 以模型单位（y 向上）中的逆时针为正面，runtime 翻转时修正绕序，所以正面判定与图层是否翻转无关。

实现：
  - 可见性用 nvdiffrast 的深度剥离（DepthPeeler）求出每个像素上按绘制顺序排列的全部片元。深度 = render order
    （后画的更近），同一 drawable 内的顶点加一个落在本 drawable 深度槽内的小偏移，使自身重叠的三角形也被逐层剥出。
  - 片元着色在压缩后的片元列表上做（不在整幅图上做），重心坐标和其屏幕导数由三角形屏幕坐标自己算（对顶点可导），
    贴图 LOD 由 uv 的屏幕导数决定（与 GL 相同的各向同性 LOD）。
  - 合成：按 GL 的顺序从后往前 D ← B_k + A_k ⊙ D（每通道仿射，三种混合都是这个形式）。
  - 边缘梯度：每个剥离层用 nvdiffrast.antialias 求「直通」项（前向值为 0，只贡献梯度），
    梯度 = 上方透射率 ⊙ (ΔB + ΔA ⊙ 下方合成)，与 runtime 的无抗锯齿前向逐像素一致。
    按 Normal 语义取 4 通道（Additive 也精确；Multiplicative 叠层没有边缘项）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import nvdiffrast.torch as dr
from torch.utils.checkpoint import checkpoint
from PIL import Image

# Core 常量标志 / 动态标志位（live2dcubismcore.d.ts 的 Utils）
CF_ADD = 1
CF_MUL = 2
CF_DOUBLE_SIDED = 4
CF_INVERTED_MASK = 8
DF_VISIBLE = 1
DF_VPOS_CHANGED = 32

CLIP_MAX_DEFAULT = 36        # ClippingMaskMaxCountOnDefault
CLIP_MAX_MULTI = 32          # ClippingMaskMaxCountOnMultiRenderTexture
COLOR_CHANNELS = 4
JS_MIN_VALUE = 5e-324        # Number.MIN_VALUE（Framework 的包围盒初值怪癖）


# ---------------------------------------------------------------------------
# 贴图：预乘 + GL 风格 mipmap + 自己实现的 LINEAR_MIPMAP_LINEAR 采样
# ---------------------------------------------------------------------------

def _premultiply_u8(rgba: np.ndarray) -> np.ndarray:
    """createImageBitmap(premultiplyAlpha:'premultiply') 的 8 位预乘：round(c·a/255)。"""
    c = rgba[..., :3].astype(np.uint32) * rgba[..., 3:4].astype(np.uint32)
    out = rgba.copy()
    out[..., :3] = ((c + 127) // 255).astype(np.uint8)
    return out


def _mip_down_u8(t: torch.Tensor) -> torch.Tensor:
    """generateMipmap 的一级：2×2 平均、8 位取整（四舍五入）。t 为 uint8 [H, W, 4]；奇数尺寸按 GL 规则向下取整。"""
    H, W = t.shape[0], t.shape[1]
    h2, w2 = max(1, H // 2), max(1, W // 2)
    x = t[: h2 * 2 if H > 1 else 1, : w2 * 2 if W > 1 else 1].to(torch.int16)
    x = x.reshape(h2, 2 if H > 1 else 1, w2, 2 if W > 1 else 1, 4).sum(dim=(1, 3), dtype=torch.int16)
    k = (2 if H > 1 else 1) * (2 if W > 1 else 1)
    return ((x + k // 2) // k).to(torch.uint8)


class TextureSet:
    """模型的全部贴图（尺寸可以不同）：预乘后的 uint8 mip 链，按 (贴图, 层级) 展平进一个大缓冲。

    采样与 WebGL 的 LINEAR_MIPMAP_LINEAR + CLAMP_TO_EDGE 一致：λ = log2(max(|∂(s,t)·size/∂X|, |∂(s,t)·size/∂Y|))，
    λ ≤ 0 时按 MAG_FILTER LINEAR 在第 0 级双线性，否则在 floor(λ)、floor(λ)+1 两级各自双线性后按 frac(λ) 插值。
    对 uv 可导（双线性权重），对 λ 不求导。显存：uint8 存储，本模型 5 张贴图（最大 8192²）约 0.8 GB。
    """

    def __init__(self, paths: list[Path], device="cuda"):
        bufs = []
        meta = []  # 每张贴图：[(H, W, offset)]（逐级）
        off = 0
        for p in paths:
            im = np.asarray(Image.open(p).convert("RGBA"), dtype=np.uint8)
            lv = torch.from_numpy(_premultiply_u8(im)).to(device)
            del im
            chain = []
            while True:
                H, W = int(lv.shape[0]), int(lv.shape[1])
                chain.append((H, W, off))
                bufs.append(lv.reshape(-1, 4))
                off += H * W
                if H == 1 and W == 1:
                    break
                lv = _mip_down_u8(lv)
            meta.append(chain)
        self.data = torch.cat(bufs, 0).contiguous()  # [Ntexel, 4] uint8
        del bufs
        self.ntex = len(paths)
        self.nlev = max(len(c) for c in meta)
        tab = np.zeros((self.ntex, self.nlev, 3), dtype=np.int64)
        for t, chain in enumerate(meta):
            for l in range(self.nlev):
                tab[t, l] = chain[min(l, len(chain) - 1)]
        self.tab = torch.tensor(tab, device=device)  # (H, W, offset)
        self.maxlev = torch.tensor([len(c) - 1 for c in meta], device=device)
        self.sizes = [(c[0][0], c[0][1]) for c in meta]
        self.lod_bias = 0.0  # 调试用：LOD 偏置

    def _bilinear(self, t: torch.Tensor, lev: torch.Tensor, s: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        e = self.tab[t, lev]
        H, W, off = e[:, 0], e[:, 1], e[:, 2]
        x = (s * W - 0.5).clamp(min=0)
        y = (tt * H - 0.5).clamp(min=0)
        x = torch.minimum(x, (W - 1).to(x.dtype))
        y = torch.minimum(y, (H - 1).to(y.dtype))
        x0 = x.detach().floor().long()
        y0 = y.detach().floor().long()
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        x1 = torch.minimum(x0 + 1, W - 1)
        y1 = torch.minimum(y0 + 1, H - 1)
        g = lambda yy, xx: self.data[off + yy * W + xx].to(s.dtype)
        return (g(y0, x0) * (1 - fx) + g(y0, x1) * fx) * (1 - fy) + (g(y1, x0) * (1 - fx) + g(y1, x1) * fx) * fy

    def sample(self, tex_index: torch.Tensor, uv: torch.Tensor, uv_da: torch.Tensor) -> torch.Tensor:
        """片元采样。uv：Cubism UV（v 向上，着色器里 t = 1 − v）；uv_da：(du/dX, du/dY, dv/dX, dv/dY)（每像素）。
        返回预乘 RGBA [N, 4]（0..1）。"""
        n = uv.shape[0]
        if n == 0:
            return uv.new_zeros((0, 4))
        t = tex_index.long()
        s = uv[:, 0]
        tt = 1.0 - uv[:, 1]
        e0 = self.tab[t, 0]
        H0, W0 = e0[:, 0].to(uv.dtype), e0[:, 1].to(uv.dtype)
        da = uv_da.detach()
        dsx, dsy = da[:, 0] * W0, da[:, 1] * W0
        dtx, dty = da[:, 2] * H0, da[:, 3] * H0
        rho2 = torch.maximum(dsx * dsx + dtx * dtx, dsy * dsy + dty * dty)
        lam = 0.5 * torch.log2(rho2.clamp_min(1e-30)) + self.lod_bias
        lam = torch.minimum(lam, self.maxlev[t].to(lam.dtype))
        mag = lam <= 0
        lam = lam.clamp_min(0)
        l0 = lam.floor().long()
        l1 = torch.minimum(l0 + 1, self.maxlev[t])
        fr = (lam - l0.to(lam.dtype))[:, None]
        fr = torch.where(mag[:, None], torch.zeros_like(fr), fr)
        c = self._bilinear(t, l0, s, tt) * (1 - fr) + self._bilinear(t, l1, s, tt) * fr
        return c / 255.0


# ---------------------------------------------------------------------------
# 模型静态数据
# ---------------------------------------------------------------------------

class ModelGeom:
    """drawable 拓扑、UV、常量标志与 clipping context（与 CubismClippingManager.initializeForDrawable 相同的建立方式）。"""

    def __init__(self, st: dict, device="cuda"):
        A = st["arrays"]
        self.canvas = st["canvas"]
        n = self.n = int(st["drawableCount"])
        self.device = device
        vo = A["vertexOffsets"].astype(np.int64)
        io = A["indexOffsets"].astype(np.int64)
        mo = A["maskOffsets"].astype(np.int64)
        self.vo = vo
        self.V = int(vo[-1])
        self.vtx_draw_np = np.repeat(np.arange(n), np.diff(vo))
        tris = []
        tri_draw = []
        for i in range(n):
            t = A["indices"][io[i]:io[i + 1]].astype(np.int64).reshape(-1, 3) + vo[i]
            tris.append(t)
            tri_draw.append(np.full(len(t), i))
        self.tri_np = np.concatenate(tris)
        self.tri_draw_np = np.concatenate(tri_draw)
        self.tri = torch.tensor(self.tri_np, dtype=torch.int32, device=device)
        self.tri_l = self.tri.long()
        self.tri_draw = torch.tensor(self.tri_draw_np, dtype=torch.long, device=device)
        self.vtx_draw = torch.tensor(self.vtx_draw_np, dtype=torch.long, device=device)
        self.uv = torch.tensor(A["uvs"], dtype=torch.float32, device=device)
        cf = A["constantFlags"].astype(np.int64)
        self.cflags = cf
        blend = np.where(cf & CF_ADD, 1, np.where(cf & CF_MUL, 2, 0))
        self.blend = torch.tensor(blend, dtype=torch.long, device=device)
        self.culling = torch.tensor((cf & CF_DOUBLE_SIDED) == 0, device=device)
        self.any_culling = bool(((cf & CF_DOUBLE_SIDED) == 0).any())
        self.inverted = torch.tensor((cf & CF_INVERTED_MASK) != 0, device=device)
        self.tex_index = torch.tensor(A["textureIndices"], dtype=torch.long, device=device)
        # 自身重叠的三角形也要被剥离出来：同一 drawable 内的顶点给一个落在深度槽内的小偏移（黄金分割散列）
        loc = np.arange(self.V) - vo[self.vtx_draw_np]
        self.vtx_hash = torch.tensor(((loc * 0.6180339887) % 1.0) - 0.5, dtype=torch.float32, device=device)
        self.topo = dr.antialias_construct_topology_hash(self.tri)

        # clipping context：与 Framework 相同的建立顺序与 findSameClip 语义
        masks = [A["masks"][mo[i]:mo[i + 1]].astype(np.int64).tolist() for i in range(n)]
        self.masks = masks
        ctx_masks: list[list[int]] = []
        ctx_clipped: list[list[int]] = []
        draw_ctx = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if len(masks[i]) <= 0:
                continue
            found = -1
            for c, cm in enumerate(ctx_masks):
                if len(cm) != len(masks[i]):
                    continue
                same = sum(1 for cid in cm if cid in masks[i])
                if same == len(cm):
                    found = c
                    break
            if found < 0:
                ctx_masks.append(list(masks[i]))
                ctx_clipped.append([])
                found = len(ctx_masks) - 1
            ctx_clipped[found].append(i)
            draw_ctx[i] = found
        self.ctx_masks = ctx_masks
        self.ctx_clipped = ctx_clipped
        self.draw_ctx_np = draw_ctx
        self.draw_ctx = torch.tensor(draw_ctx, dtype=torch.long, device=device)
        self.mask_size = int(st.get("maskBufferSize") or 2048)
        self.mask_rt_count = int(st.get("maskRenderTextureCount") or 1)
        # 每个上下文的 mask drawable 三角形（局部重新编号）
        self.ctx_mesh = []
        for cm in ctx_masks:
            vids = np.concatenate([np.arange(vo[m], vo[m + 1]) for m in cm]) if cm else np.zeros(0, np.int64)
            remap = {int(v): k for k, v in enumerate(vids)}
            tri = np.concatenate([self.tri_np[self.tri_draw_np == m] for m in cm]) if cm else np.zeros((0, 3), np.int64)
            tri_local = np.vectorize(remap.__getitem__, otypes=[np.int64])(tri) if len(tri) else tri
            self.ctx_mesh.append(
                {
                    "vids": torch.tensor(vids, dtype=torch.long, device=device),
                    "tri": torch.tensor(tri_local, dtype=torch.int32, device=device).contiguous(),
                    "tri_draw": torch.tensor(np.concatenate([self.tri_draw_np[self.tri_draw_np == m] for m in cm]) if cm else np.zeros(0, np.int64), dtype=torch.long, device=device),
                    "draw": torch.tensor(cm, dtype=torch.long, device=device),
                }
            )
        for mesh in self.ctx_mesh:
            cm = mesh["draw"].tolist()
            nv = [int(vo[m + 1] - vo[m]) for m in cm]
            vdraw = torch.repeat_interleave(torch.arange(len(cm), device=device), torch.tensor(nv, device=device)) if cm else torch.zeros(0, dtype=torch.long, device=device)
            mesh["vdraw"] = vdraw
            # mask 内部顺序无关（Π(1 − a) 可交换），深度只用来把重叠的三角形逐层剥出
            zslot = 2.0 / (len(cm) + 2)
            mesh["z"] = (1 - (vdraw.float() + 1) * zslot + zslot * 0.8 * self.vtx_hash[mesh["vids"]]).contiguous()
            mesh["uv"] = self.uv[mesh["vids"]].contiguous()
            mesh["tri_tex"] = self.tex_index[mesh["tri_draw"]]
        pc, pv, pd = [], [], []
        for c, cm in enumerate(ctx_masks):
            for m in cm:
                pv.append(np.arange(vo[m], vo[m + 1]))
                pc.append(np.full(int(vo[m + 1] - vo[m]), c))
                pd.append(np.full(int(vo[m + 1] - vo[m]), m))
        self.pair_ctx = torch.tensor(np.concatenate(pc) if pc else np.zeros(0, np.int64), dtype=torch.long, device=device)
        self.pair_vid = torch.tensor(np.concatenate(pv) if pv else np.zeros(0, np.int64), dtype=torch.long, device=device)
        self.pair_draw = torch.tensor(np.concatenate(pd) if pd else np.zeros(0, np.int64), dtype=torch.long, device=device)
        # 对网格：三角形、所属上下文/drawable/贴图、UV、深度（每个 (上下文, drawable) 组一个深度槽，组内加顶点散列）
        ptri, ptc, ptd = [], [], []
        base = 0
        groups = []
        for c, cm in enumerate(ctx_masks):
            mesh = self.ctx_mesh[c]
            t = mesh["tri"].long().cpu().numpy()
            ptri.append(t + base)
            ptc.append(np.full(len(t), c))
            ptd.append(mesh["tri_draw"].cpu().numpy())
            for m in cm:
                groups.append(np.full(int(vo[m + 1] - vo[m]), len(groups)))
            base += len(mesh["vids"])
        ng = max(1, len(groups))
        grp = torch.tensor(np.concatenate(groups) if groups else np.zeros(0, np.int64), device=device)
        zslot = 2.0 / (ng + 2)
        self.pair_z = (1 - (grp.float() + 1) * zslot + zslot * 0.8 * self.vtx_hash[self.pair_vid]).contiguous()
        self.pair_tri = torch.tensor(np.concatenate(ptri) if ptri else np.zeros((0, 3), np.int64), dtype=torch.int32, device=device).contiguous()
        self.pair_tri_ctx = torch.tensor(np.concatenate(ptc) if ptc else np.zeros(0, np.int64), dtype=torch.long, device=device)
        self.pair_tri_draw = torch.tensor(np.concatenate(ptd) if ptd else np.zeros(0, np.int64), dtype=torch.long, device=device)
        self.pair_tri_tex = self.tex_index[self.pair_tri_draw]
        self.pair_tri_l = self.pair_tri.long()
        self.pair_uv = self.uv[self.pair_vid].contiguous()
        self.layouts = layout_bounds(len(ctx_masks), self.mask_rt_count)

    # 画布像素（y 向下） ↔ Cubism 单位（y 向上）
    def to_units(self, p: torch.Tensor) -> torch.Tensor:
        c = self.canvas
        return torch.stack([(p[..., 0] - c["originX"]) / c["pixelsPerUnit"], (c["originY"] - p[..., 1]) / c["pixelsPerUnit"]], -1)


def layout_bounds(using: int, rt_count: int) -> list[dict]:
    """CubismClippingManager.setupLayoutBounds 的移植（按上下文列表顺序分配通道、矩形、缓冲）。"""
    out = [dict(channel=0, bounds=(0.0, 0.0, 1.0, 1.0), buffer=0) for _ in range(using)]
    max_count = CLIP_MAX_DEFAULT if rt_count <= 1 else CLIP_MAX_MULTI * rt_count
    if using <= 0 or using > max_count:
        return out
    layout_max = 9 if rt_count <= 1 else 8
    per_sheet = math.ceil(using / rt_count)
    reduce_rt = using % rt_count
    div = per_sheet // COLOR_CHANNELS
    mod = per_sheet % COLOR_CHANNELS
    cur = 0
    for rti in range(rt_count):
        for ch in range(COLOR_CHANNELS):
            cnt = div + (1 if ch < mod else 0)
            check = mod + (-1 if div < 1 else 0)
            if ch == check and reduce_rt > 0:
                cnt -= 0 if rti < reduce_rt else 1
            if cnt == 0:
                continue
            for i in range(cnt):
                if cnt == 1:
                    b = (0.0, 0.0, 1.0, 1.0)
                elif cnt == 2:
                    b = ((i % 2) * 0.5, 0.0, 0.5, 1.0)
                elif cnt <= 4:
                    b = ((i % 2) * 0.5, (i // 2) * 0.5, 0.5, 0.5)
                elif cnt <= layout_max:
                    b = ((i % 3) / 3.0, (i // 3) / 3.0, 1.0 / 3.0, 1.0 / 3.0)
                else:
                    out[cur] = dict(channel=0, bounds=(0.0, 0.0, 1.0, 1.0), buffer=0)
                    cur += 1
                    continue
                out[cur] = dict(channel=ch, bounds=b, buffer=rti)
                cur += 1
    return out


def clip_matrices(geom: ModelGeom, units: torch.Tensor) -> torch.Tensor:
    """每个上下文的 matrixForDraw（2×3，模型单位 → mask 贴图坐标 0..1），按 setupClippingContext（非高精度）计算。
    units：[B, V, 2]（不求导：包围盒不光滑，Framework 也是逐帧重算）。返回 [B, nctx, 2, 3]（float64）。"""
    B = units.shape[0]
    u = units.detach().double()
    vd = geom.vtx_draw
    n = geom.n
    big = torch.finfo(torch.float64).max
    # 每个 drawable 的包围盒
    mn = torch.full((B, n, 2), big, dtype=torch.float64, device=u.device)
    mx = torch.full((B, n, 2), -big, dtype=torch.float64, device=u.device)
    mn.scatter_reduce_(1, vd[None, :, None].expand(B, -1, 2), u, "amin")
    mx.scatter_reduce_(1, vd[None, :, None].expand(B, -1, 2), u, "amax")
    out = torch.zeros((B, len(geom.ctx_masks), 2, 3), dtype=torch.float64, device=u.device)
    for c, clipped in enumerate(geom.ctx_clipped):
        idx = torch.tensor(clipped, device=u.device)
        cmn = mn[:, idx].amin(1)
        cmx = mx[:, idx].amax(1).clamp_min(JS_MIN_VALUE)  # Number.MIN_VALUE 初值
        w = cmx - cmn
        bx, by = cmn[:, 0] - w[:, 0] * 0.05, cmn[:, 1] - w[:, 1] * 0.05
        bw, bh = w[:, 0] * 1.1, w[:, 1] * 1.1
        lx, ly, lw, lh = geom.layouts[c]["bounds"]
        sx, sy = lw / bw, lh / bh
        out[:, c, 0, 0] = sx
        out[:, c, 0, 2] = lx - sx * bx
        out[:, c, 1, 1] = sy
        out[:, c, 1, 2] = ly - sy * by
    return out


# ---------------------------------------------------------------------------
# 变换
# ---------------------------------------------------------------------------

def affine(a, b, c, d, e, f) -> torch.Tensor:
    return torch.stack([torch.stack([a, c, e], -1), torch.stack([b, d, f], -1)], -2)


def affine_mul(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """m1 ∘ m2（[..., 2, 3]）。"""
    A = m1[..., :, :2] @ m2[..., :, :2]
    t = (m1[..., :, :2] @ m2[..., :, 2:3])[..., 0] + m1[..., :, 2]
    return torch.cat([A, t[..., None]], -1)


def layer_matrix(x, y, scale, rotation, anchor_x, anchor_y, scale_x=1.0, scale_y=1.0, flip_x=False) -> torch.Tensor:
    """规范第 2 条：world = [x,y] + R(rot)·diag(sx, sy)·(p − anchor)，sx = scale·scaleX·(flip ? −1 : 1)，sy = scale·scaleY。
    各参数可为张量（可求导）。返回 [..., 2, 3]。"""
    t = lambda v: v if torch.is_tensor(v) else torch.tensor(float(v), dtype=torch.float64)
    x, y, scale, rotation, scale_x, scale_y = map(t, (x, y, scale, rotation, scale_x, scale_y))
    sx = scale * scale_x * (-1.0 if flip_x else 1.0)
    sy = scale * scale_y
    r = rotation * (math.pi / 180.0)
    c, s = torch.cos(r), torch.sin(r)
    # R·S：[[c·sx, −s·sy], [s·sx, c·sy]]
    a, b, cc, d = c * sx, s * sx, -s * sy, c * sy
    e = x - (a * anchor_x + cc * anchor_y)
    f = y - (b * anchor_x + d * anchor_y)
    return affine(a, b, cc, d, e, f)


def camera_matrix(x, y, zoom, rotation, W: int, H: int) -> torch.Tensor:
    """screen = [W/2, H/2] + zoom·R(rot)·(world − [x, y])。"""
    t = lambda v: v if torch.is_tensor(v) else torch.tensor(float(v), dtype=torch.float64)
    x, y, zoom, rotation = map(t, (x, y, zoom, rotation))
    r = rotation * (math.pi / 180.0)
    c, s = torch.cos(r) * zoom, torch.sin(r) * zoom
    e = W / 2 - (c * x - s * y)
    f = H / 2 - (s * x + c * y)
    return affine(c, s, -s, c, e, f)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

@dataclass
class LayerInput:
    positions: torch.Tensor      # [B, V, 2] 画布像素
    opacities: torch.Tensor      # [B, n]
    multiply: torch.Tensor       # [B, n, 4]
    screen: torch.Tensor         # [B, n, 4]
    render_orders: torch.Tensor  # [B, n] long
    dyn_flags: torch.Tensor      # [B, n] long
    to_screen: torch.Tensor      # [B, 2, 3]

    @staticmethod
    def from_arrays(positions, opacities, multiply, screen, render_orders, visible, to_screen, vpos_changed=None) -> "LayerInput":
        """由变形器输出（如 A1 的 l2d.deform，口径相同：画布像素）组装。visible [B, n] 布尔；
        vpos_changed 缺省视为全部已更新（Core 每帧都会重算顶点；只有 Framework 生成 mask 时用到这一位）。
        multiply / screen 可为 [B, n, 3] 或 [B, n, 4]（只用 rgb）。to_screen [B, 2, 3] 或 [2, 3]。"""
        B = positions.shape[0]
        flags = visible.long() * DF_VISIBLE
        flags = flags + (DF_VPOS_CHANGED if vpos_changed is None else vpos_changed.long() * DF_VPOS_CHANGED)
        pad = lambda c: c if c.shape[-1] == 4 else torch.cat([c, torch.ones_like(c[..., :1])], -1)
        ts = to_screen if to_screen.dim() == 3 else to_screen[None].expand(B, -1, -1)
        return LayerInput(positions, opacities, pad(multiply), pad(screen), render_orders.long(), flags, ts)

    @staticmethod
    def from_dump(layers: list[dict], device="cuda") -> "LayerInput":
        """由 dumpio.Dump.frame() 的若干图层（同一模型）拼成一个批。"""
        g = lambda k, dt: torch.tensor(np.stack([L["arrays"][k] for L in layers]), dtype=dt, device=device)
        ts = torch.tensor(np.array([[[L["toScreen"][0], L["toScreen"][2], L["toScreen"][4]], [L["toScreen"][1], L["toScreen"][3], L["toScreen"][5]]] for L in layers]), dtype=torch.float64, device=device)
        return LayerInput(
            positions=g("positions", torch.float32),
            opacities=g("opacities", torch.float32),
            multiply=g("multiplyColors", torch.float32),
            screen=g("screenColors", torch.float32),
            render_orders=g("renderOrders", torch.long),
            dyn_flags=g("dynamicFlags", torch.long),
            to_screen=ts,
        )


class _Quant8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x * 255.0 + 0.5) / 255.0

    @staticmethod
    def backward(ctx, g):
        return g


def _q8(x: torch.Tensor) -> torch.Tensor:
    """8 位取整（直通估计）。"""
    return _Quant8.apply(x)


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


class Rasterizer:
    """单个模型（drawable 拓扑 + 贴图）的可微分图层光栅化器。"""

    def __init__(self, geom: ModelGeom, atlas: "TextureSet", glctx: dr.RasterizeCudaContext | None = None):
        self.geom = geom
        self.atlas = atlas
        self.glctx = glctx or dr.RasterizeCudaContext(device=geom.device)

    # ---------- 片元：重心坐标与插值（可导） ----------
    def _frag_bary(self, S: torch.Tensor, vi: torch.Tensor, px: torch.Tensor, py: torch.Tensor):
        """S：[M, 2] 展平后的屏幕坐标（像素）；vi：[N, 3] 展平后的顶点下标；px/py：片元像素中心。
        返回 (w0, w1, w2) 与其屏幕导数 (dw1/dX, dw1/dY, dw2/dX, dw2/dY)。"""
        p0, p1, p2 = S[vi[:, 0]], S[vi[:, 1]], S[vi[:, 2]]
        e1x, e1y = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        e2x, e2y = p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1]
        den = _cross(e1x, e1y, e2x, e2y)
        den = torch.where(den.abs() < 1e-20, torch.full_like(den, 1e-20), den)
        qx, qy = px - p0[:, 0], py - p0[:, 1]
        w1 = _cross(qx, qy, e2x, e2y) / den
        w2 = _cross(e1x, e1y, qx, qy) / den
        w0 = 1.0 - w1 - w2
        d = (e2y / den, -e2x / den, -e1y / den, e1x / den)
        return w0, w1, w2, d, den

    # ---------- clipping mask 缓冲 ----------
    def _mask_windows(self, inp: LayerInput, units: torch.Tensor, Mdraw: torch.Tensor, draw_on: torch.Tensor):
        """需要渲染的上下文及其像素窗口（mask 缓冲像素坐标，含两端）。
        窗口 = 布局矩形 ∩（mask drawable 投影包围盒 ± 2）∩（本帧实际绘制的被裁切 drawable 投影包围盒 ± 2）。
        被裁切片元只会落在后者内部（双线性再多取 1 个纹素），窗口外而在布局矩形内的纹素都是清屏值 1。"""
        g = self.geom
        R = g.mask_size
        B = units.shape[0]
        dev = units.device
        nctx = len(g.ctx_masks)
        u = units.detach().double()
        big = 1e30
        vpos_ok = (inp.dyn_flags & DF_VPOS_CHANGED) != 0
        # 被裁切 drawable 的顶点（只算本帧绘制的）
        vctx = g.draw_ctx[g.vtx_draw]  # [V]
        on_c = draw_on[:, g.vtx_draw] & (vctx >= 0)[None]
        cidx = vctx.clamp_min(0)
        M = Mdraw[:, cidx]  # [B, V, 2, 3]
        tc = (M[..., :2] @ u[..., None])[..., 0] + M[..., 2]
        pix = tc * R

        def bbox(p, ok, key, K):
            lo = torch.full((K, 2), big, dtype=p.dtype, device=dev)
            hi = torch.full((K, 2), -big, dtype=p.dtype, device=dev)
            k = key[None].expand(B, -1).reshape(-1)
            pp = p.reshape(-1, 2)
            okf = ok.reshape(-1)
            lo.scatter_reduce_(0, k[okf, None].expand(-1, 2), pp[okf], "amin")
            hi.scatter_reduce_(0, k[okf, None].expand(-1, 2), pp[okf], "amax")
            return lo, hi

        s_lo, s_hi = bbox(pix, on_c, cidx, nctx)
        # mask drawable 的顶点（(上下文, 顶点) 对）
        pc, pv, pd = g.pair_ctx, g.pair_vid, g.pair_draw
        Mp = Mdraw[:, pc]
        mp = ((Mp[..., :2] @ u[:, pv][..., None])[..., 0] + Mp[..., 2]) * R
        m_lo, m_hi = bbox(mp, vpos_ok[:, pd], pc, nctx)
        lo = torch.maximum(s_lo, m_lo).floor() - 2
        hi = torch.minimum(s_hi, m_hi).ceil() + 2
        lo, hi = lo.cpu().numpy(), hi.cpu().numpy()
        out = {}
        for c in range(nctx):
            if hi[c, 0] < -1e29 or hi[c, 0] < lo[c, 0] or hi[c, 1] < lo[c, 1]:
                continue
            lx, ly, lw, lh = g.layouts[c]["bounds"]
            # 像素中心落在布局矩形内（step 判定，两端含）的像素范围
            i0 = max(math.ceil(lx * R - 0.5), int(lo[c, 0]))
            i1 = min(math.floor((lx + lw) * R - 0.5), int(hi[c, 0]))
            j0 = max(math.ceil(ly * R - 0.5), int(lo[c, 1]))
            j1 = min(math.floor((ly + lh) * R - 0.5), int(hi[c, 1]))
            if i1 >= i0 and j1 >= j0:
                out[c] = (i0, j0, i1 - i0 + 1, j1 - j0 + 1)
        return out

    def _render_masks(self, inp: LayerInput, units: torch.Tensor, Mdraw: torch.Tensor, draw_on: torch.Tensor, aa: bool):
        """各上下文窗口内的 mask 图（单通道，值 = Π(1 − tex.a)，与 Framework 的 mask 缓冲对应通道一致）。
        为减少调用与同步，所有窗口按行打包进一张画布，一次深度剥离画完：每个上下文的 mask 网格平移到自己的槽位，
        越出槽位的片元（落在别的槽位里）按「片元所属上下文 ≠ 像素所属槽位」丢弃，等价于 Framework 的 isInside 判定。
        返回 (画布 [B, CH·CW], 查表 [nctx, 6]：(偏移, i0, j0, w, h, 行跨度))，w = 0 表示整块为清屏值 1。"""
        g = self.geom
        B = units.shape[0]
        R = g.mask_size
        dev = units.device
        nctx = len(g.ctx_masks)
        wins = self._mask_windows(inp, units, Mdraw, draw_on)
        table = torch.zeros((nctx, 6), dtype=torch.long, device=dev)
        if not wins:
            return torch.ones((B, 1), device=dev), table
        # 按高度降序做行打包
        order = sorted(wins, key=lambda c: -wins[c][3])
        CW = max(1024, max(wins[c][2] for c in wins))
        slots = {}
        x = y = rowh = 0
        for c in order:
            w, h = wins[c][2], wins[c][3]
            if x + w > CW:
                x, y, rowh = 0, y + rowh, 0
            slots[c] = (x, y)
            x += w
            rowh = max(rowh, h)
        CH = y + rowh
        tab = np.zeros((nctx, 6), dtype=np.int64)
        slot_np = np.full((CH, CW), -1, dtype=np.int64)
        shift = np.zeros((nctx, 2), dtype=np.float64)
        need = np.zeros(nctx, dtype=bool)
        for c, (i0, j0, w, h) in wins.items():
            ox, oy = slots[c]
            tab[c] = (oy * CW + ox, i0, j0, w, h, CW)
            slot_np[oy:oy + h, ox:ox + w] = c
            shift[c] = (ox - i0, oy - j0)
            need[c] = True
        table = torch.tensor(tab, device=dev)
        slot_map = torch.tensor(slot_np, device=dev).reshape(-1)
        shift_t = torch.tensor(shift, device=dev)
        need_t = torch.tensor(need, device=dev)
        # 全部 (上下文, mask 顶点) 对 → 画布像素
        pc, pv, pd = g.pair_ctx, g.pair_vid, g.pair_draw
        Mp = Mdraw[:, pc].to(units.dtype)
        pix = ((Mp[..., :2] @ units[:, pv][..., None])[..., 0] + Mp[..., 2]) * R + shift_t[pc].to(units.dtype)
        vpos_ok = (inp.dyn_flags & DF_VPOS_CHANGED) != 0  # 只画「顶点已更新」的 mask drawable
        ok_v = vpos_ok[:, pd] & need_t[pc][None]
        nx = pix[..., 0] / CW * 2 - 1
        ny = pix[..., 1] / CH * 2 - 1
        pos = torch.stack([nx.float(), ny.float(), g.pair_z[None].expand(B, -1), torch.ones_like(nx, dtype=torch.float32)], -1)
        pos = torch.where(ok_v[..., None], pos, torch.tensor([-9.0, -9.0, 0.0, 1.0], device=dev)).contiguous()
        tri = g.pair_tri
        HW = CH * CW
        Np = pix.shape[1]
        self.last_mask_layers = 0

        def peel(pos, Pf, B):
            prod = torch.ones(B * HW, device=dev)
            nl = 0
            with dr.DepthPeeler(self.glctx, pos, tri, (CH, CW), grad_db=False) as peeler:
                while True:
                    rast, _ = peeler.rasterize_next_layer()
                    tidf = rast[..., 3].reshape(-1)
                    hidx = (tidf > 0).nonzero()[:, 0]
                    if hidx.numel() == 0:
                        break
                    nl += 1
                    chunk = hidx.numel() if torch.is_grad_enabled() else (1 << 20)
                    avs = []
                    for s0 in range(0, hidx.numel(), chunk):
                        hc = hidx[s0:s0 + chunk]
                        tid = tidf[hc].long() - 1
                        bidx = hc // HW
                        q = hc % HW
                        own = g.pair_tri_ctx[tid] == slot_map[q]  # 片元属于本像素所在槽位的上下文
                        # 片元重心坐标与 uv 导数（画布像素 = mask 缓冲像素）
                        vi = g.pair_tri_l[tid] + (bidx * Np)[:, None]
                        w0, w1, w2, dw, _ = self._frag_bary(Pf, vi, (q % CW).to(Pf.dtype) + 0.5, (q // CW).to(Pf.dtype) + 0.5)
                        U0, U1, U2 = g.pair_uv[vi[:, 0] - bidx * Np], g.pair_uv[vi[:, 1] - bidx * Np], g.pair_uv[vi[:, 2] - bidx * Np]
                        uv = U0 * w0[:, None] + U1 * w1[:, None] + U2 * w2[:, None]
                        dU1, dU2 = U1 - U0, U2 - U0
                        uv_da = torch.stack(
                            [dw[0] * dU1[:, 0] + dw[2] * dU2[:, 0], dw[1] * dU1[:, 0] + dw[3] * dU2[:, 0],
                             dw[0] * dU1[:, 1] + dw[2] * dU2[:, 1], dw[1] * dU1[:, 1] + dw[3] * dU2[:, 1]], -1)
                        av = self.atlas.sample(g.pair_tri_tex[tid], uv.float(), uv_da.float())[:, 3]
                        av = torch.where(own, av, torch.zeros_like(av))
                        if g.any_culling:
                            av = av * self._front_frag(pos, tri, tid, bidx, g.pair_tri_draw)
                        avs.append(av)
                    av = torch.cat(avs) if len(avs) > 1 else avs[0]
                    f = torch.ones(B * HW, device=dev, dtype=av.dtype).index_put((hidx,), 1.0 - av)
                    if aa:
                        fa = dr.antialias(f.reshape(B, CH, CW, 1), rast, pos, tri).reshape(-1)
                        f = f + (fa - fa.detach())
                    prod = prod * f
            self.last_mask_layers = max(self.last_mask_layers, nl)
            return prod.reshape(B, HW)

        if torch.is_grad_enabled():
            prod = peel(pos, pix.reshape(B * Np, 2), B)
        else:
            # 不求导时逐帧画（窗口与查表对整批相同），显存与批大小无关
            prod = torch.cat([peel(pos[i:i + 1].contiguous(), pix[i], 1) for i in range(B)], 0)
        return prod, table

    def _front_frag(self, pos, tri, tid, bidx, tri_draw):
        """（mask 生成用）背面剔除：mask 矩阵行列式恒正，正面 = 裁剪空间逆时针；被剔除的背面片元返回 0。"""
        t = tri.long()[tid]
        p0, p1, p2 = pos[bidx, t[:, 0], :2], pos[bidx, t[:, 1], :2], pos[bidx, t[:, 2], :2]
        area = _cross(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1], p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1])
        return torch.where(self.geom.culling[tri_draw[tid]] & (area < 0), 0.0, 1.0)

    def _sample_masks(self, masks, ctx: torch.Tensor, b: torch.Tensor, cpos: torch.Tensor) -> torch.Tensor:
        """片元的 mask 值：在所属上下文的 mask 图上按 GL LINEAR 双线性采样，返回 1 − texel（未反转）。"""
        flat, table = masks
        R = self.geom.mask_size
        t = table[ctx]
        off, i0, j0, w, h, stride = t[:, 0], t[:, 1], t[:, 2], t[:, 3], t[:, 4], t[:, 5]
        x = cpos[:, 0] * R - 0.5 - i0
        y = cpos[:, 1] * R - 0.5 - j0
        x0 = x.detach().floor()
        y0 = y.detach().floor()
        fx, fy = x - x0, y - y0
        x0l, y0l = x0.long(), y0.long()

        def tex(yy, xx):
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            i = off + yy.clamp(min=0) * stride + xx.clamp(min=0)
            v = flat[b, torch.where(ok, i, torch.zeros_like(i))]
            return torch.where(ok, v, torch.ones_like(v))

        v = (tex(y0l, x0l) * (1 - fx) + tex(y0l, x0l + 1) * fx) * (1 - fy) + (tex(y0l + 1, x0l) * (1 - fx) + tex(y0l + 1, x0l + 1) * fx) * fy
        return 1.0 - v

    def _shade(self, idx, t, Sf, Uf, P, opacities, multiply, screen, masks, Mdraw, W: int, HW: int):
        """一个剥离层的片元着色 → 混合系数 (B_k, A_k)：D ← B_k + A_k ⊙ D（每通道）。"""
        g = self.geom
        V = g.V
        B = P.shape[0]
        b = idx // HW
        pix = idx % HW
        py = (pix // W).to(Sf.dtype) + 0.5
        px = (pix % W).to(Sf.dtype) + 0.5
        vi = g.tri_l[t] + (b * V)[:, None]
        d = g.tri_draw[t]
        w0, w1, w2, dw, den = self._frag_bary(Sf, vi, px, py)
        # uv 与其屏幕导数
        U0, U1, U2 = g.uv[vi[:, 0] - b * V], g.uv[vi[:, 1] - b * V], g.uv[vi[:, 2] - b * V]
        uv = U0 * w0[:, None] + U1 * w1[:, None] + U2 * w2[:, None]
        dU1, dU2 = U1 - U0, U2 - U0
        uv_da = torch.stack(
            [dw[0] * dU1[:, 0] + dw[2] * dU2[:, 0], dw[1] * dU1[:, 0] + dw[3] * dU2[:, 0],
             dw[0] * dU1[:, 1] + dw[2] * dU2[:, 1], dw[1] * dU1[:, 1] + dw[3] * dU2[:, 1]], -1)
        tex = self.atlas.sample(g.tex_index[d], uv, uv_da)
        # 乘算色 / 屏幕色 / 不透明度
        mul = multiply[b, d, :3]
        scr = screen[b, d, :3]
        rgb = tex[:, :3] * mul
        rgb = rgb + scr * tex[:, 3:4] - rgb * scr
        op = opacities[b, d][:, None]
        col = torch.cat([rgb, tex[:, 3:4]], -1) * op
        # clipping mask（对全部片元计算，未被裁切的取 1，避免数据相关的同步）
        if masks is not None:
            ctx = g.draw_ctx[d]
            masked = ctx >= 0
            cc = ctx.clamp_min(0)
            pu = Uf[vi[:, 0]] * w0[:, None] + Uf[vi[:, 1]] * w1[:, None] + Uf[vi[:, 2]] * w2[:, None]
            M = Mdraw[b, cc].to(pu.dtype)
            cpos = (M[:, :, :2] @ pu[..., None])[..., 0] + M[:, :, 2]
            mv = self._sample_masks(masks, cc, b, cpos)
            mv = torch.where(g.inverted[d], 1.0 - mv, mv)
            col = col * torch.where(masked, mv.to(op.dtype), torch.ones_like(mv, dtype=op.dtype))[:, None]
        # 背面剔除（模型单位中逆时针为正面 ⇔ 画布像素中顺时针）
        if g.any_culling:
            pc = P.reshape(B * V, 2)
            q0, q1, q2 = pc[vi[:, 0]], pc[vi[:, 1]], pc[vi[:, 2]]
            ar = _cross(q1[:, 0] - q0[:, 0], q1[:, 1] - q0[:, 1], q2[:, 0] - q0[:, 0], q2[:, 1] - q0[:, 1])
            col = torch.where((g.culling[d] & (ar.detach() > 0))[:, None], torch.zeros_like(col), col)
        # 混合系数：Normal (ONE, 1−αs)；Additive (ONE, ONE; α 不变)；Multiplicative (DST_COLOR, 1−αs; α 不变)
        bl = g.blend[d]
        a = col[:, 3:4]
        one = torch.ones_like(col)
        zero = torch.zeros_like(col)
        Bn, An = col, (1.0 - a).expand(-1, 4)
        Ba = torch.cat([col[:, :3], zero[:, :1]], -1)
        Am = torch.cat([col[:, :3] + 1.0 - a, one[:, :1]], -1)
        Bk = torch.where((bl == 0)[:, None], Bn, torch.where((bl == 1)[:, None], Ba, zero))
        Ak = torch.where((bl == 0)[:, None], An, torch.where((bl == 1)[:, None], one, Am))
        return Bk, Ak

    def _aa_delta(self, Xv, idx, rast, clip, eidx, B: int, H: int, W: int):
        """边缘梯度直通项：antialias(X) − 其自身的 detach，在 eidx 上取值（前向恒为 0）。"""
        X = torch.zeros((B * H * W, 4), device=Xv.device, dtype=Xv.dtype).index_put((idx,), Xv)
        Xa = dr.antialias(X.reshape(B, H, W, 4), rast, clip, self.geom.tri, topology_hash=self.geom.topo).reshape(B * H * W, 4)
        return (Xa - Xa.detach())[eidx]

    # ---------- 主绘制 ----------
    def render(self, inp: LayerInput, W: int, H: int, aa: bool = True, quantize: bool = True, mask_grad: bool = False, aa_layers: int | None = None) -> torch.Tensor:
        """渲染一批图层缓冲（预乘 RGBA，[B, H, W, 4]）。
        aa=True：附加边缘梯度直通项（前向值不变，只在求导时生效）。
        quantize=True：每次混合后按 RGBA8 目标夹到 [0,1] 并 8 位取整（梯度直通），与 GL 逐 drawable 写入 8 位缓冲一致；
        False 时只夹取不取整。
        mask_grad=True：梯度也穿过 mask 缓冲的生成传到 mask drawable 的顶点（mask 缓冲是 2048² 级别，显存开销大）；
        缺省 False 时 mask 图不求导，被裁切 drawable 对顶点的梯度仍经采样位置（clipPos）传回。
        aa_layers：只对最前面的 K 个剥离层加边缘梯度项（省显存；更深的层大多被遮挡），None = 全部。"""
        g = self.geom
        dev = inp.positions.device
        B = inp.positions.shape[0]
        V = g.V
        need_grad = torch.is_grad_enabled()
        aa = aa and need_grad
        mask_grad = mask_grad and need_grad

        P = inp.positions
        ts = inp.to_screen.to(P.dtype)
        S = (ts[:, None, :, :2] @ P[..., None])[..., 0] + ts[:, None, :, 2]  # [B, V, 2] 合成像素
        units = g.to_units(P)

        # 绘制集合：可见且不透明度 > 0（不透明度 0 的 drawable 贡献恰为 0）
        draw_on = ((inp.dyn_flags & DF_VISIBLE) != 0) & (inp.opacities > 0)  # [B, n]
        n = g.n
        slot = 2.0 / (n + 2)
        zd = 1.0 - (inp.render_orders.float() + 1.0) * slot  # 后画的更近
        z = zd[:, g.vtx_draw] + slot * 0.8 * g.vtx_hash[None]
        clip = torch.stack([S[..., 0] / W * 2 - 1, S[..., 1] / H * 2 - 1, z, torch.ones_like(z)], -1).float()
        on_v = draw_on[:, g.vtx_draw]
        clip = torch.where(on_v[..., None], clip, torch.tensor([-9.0, -9.0, 0.0, 1.0], device=dev)).contiguous()

        # clipping mask
        any_masked = bool((draw_on & (g.draw_ctx[None] >= 0)).any())
        masks = None
        Mdraw = None
        if any_masked:
            Mdraw = clip_matrices(g, units)
            with torch.set_grad_enabled(mask_grad):
                masks = self._render_masks(inp, units, Mdraw, draw_on, aa and mask_grad)

        Sf = S.reshape(B * V, 2)
        Uf = units.reshape(B * V, 2)
        HW = H * W
        layers = []  # (idx, Bk, Ak, edge)
        T_full = torch.ones((B * HW, 4), device=dev) if aa else None
        self.last_layers = 0
        with dr.DepthPeeler(self.glctx, clip, g.tri, (H, W), grad_db=False) as peeler:
            while True:
                rast, _ = peeler.rasterize_next_layer()
                tidf = rast[..., 3].reshape(-1)
                idx = (tidf > 0).nonzero()[:, 0]
                if idx.numel() == 0:
                    break
                self.last_layers += 1
                t = tidf[idx].long() - 1
                if need_grad:
                    Bk, Ak = checkpoint(self._shade, idx, t, Sf, Uf, P, inp.opacities, inp.multiply, inp.screen, masks, Mdraw, W, HW, use_reentrant=False)
                else:
                    Bk, Ak = self._shade(idx, t, Sf, Uf, P, inp.opacities, inp.multiply, inp.screen, masks, Mdraw, W, HW)
                edge = None
                if aa and (aa_layers is None or self.last_layers <= aa_layers):
                    # 4 通道、按 Normal 语义解释：ΔB = ΔX，ΔA = −ΔX_α。Normal 与 Additive（X = (B_rgb, 0)）都精确；
                    # Multiplicative（B = 0、A_rgb = col + 1 − α）取 X = 0，即乘算叠层没有边缘梯度（省一半显存）。
                    Xv = torch.where((g.blend[g.tri_draw[t]] == 2)[:, None], torch.zeros_like(Bk), Bk)
                    with torch.no_grad():
                        X0 = torch.zeros((B * HW, 4), device=dev, dtype=Bk.dtype).index_put((idx,), Xv)
                        Xa0 = dr.antialias(X0.reshape(B, H, W, 4), rast, clip, g.tri, topology_hash=g.topo).reshape(B * HW, 4)
                        eidx = (Xa0 != X0).any(-1).nonzero()[:, 0]
                        del X0, Xa0
                    if eidx.numel():
                        # 反向时重算 antialias（只保留 rast 与压缩的片元值）
                        delta = checkpoint(self._aa_delta, Xv, idx, rast, clip, eidx, B, H, W, use_reentrant=False)
                        edge = (eidx, delta, T_full[eidx].clone())
                if aa:
                    T_full[idx] = T_full[idx] * Ak.detach()
                layers.append((idx, Bk, Ak, edge))

        # 从后往前合成（与 GL 绘制顺序一致）
        D = torch.zeros((B * HW, 4), device=dev, dtype=torch.float32)
        extra = []
        for idx, Bk, Ak, edge in reversed(layers):
            if edge is not None:
                eidx, delta, Te = edge
                Cb = D.detach()[eidx]
                extra.append((eidx, Te * (delta - delta[:, 3:4] * Cb)))
            v = (Bk + Ak * D[idx]).clamp(0.0, 1.0)  # UNORM8 目标：每次混合后夹取
            if quantize:
                v = _q8(v)
            D = D.index_put((idx,), v)
        for eidx, v in extra:
            D = D.index_add(0, eidx, v)
        return D.reshape(B, H, W, 4)


def load_rasterizer(dump, model_id: str, device="cuda", assets_root: Path | None = None) -> Rasterizer:
    """从 A3 dump 的静态数据建立光栅化器（贴图从 assets/ 只读加载）。"""
    from urllib.parse import unquote, urlparse

    st = dump.model(model_id)
    repo = Path(__file__).resolve().parents[3]
    root = assets_root or (repo / "assets")
    model_rel = unquote(urlparse(st["modelUrl"]).path).split("/assets/", 1)[1]
    model_dir = (root / model_rel).parent
    paths = [model_dir / t for t in st["textures"]]
    return Rasterizer(ModelGeom(st, device), TextureSet(paths, device))
