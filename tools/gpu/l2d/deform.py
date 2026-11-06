"""PyTorch 版 Cubism 变形器（可批量、可求导），与 Cubism Core 的 csmUpdateModel 对拍。

输入参数 [B, P]（顺序同 moc3 / Core），输出每个 drawable 的顶点（画布像素）、不透明度、draw order、
render order、乘算色 / 屏幕色、可见性。算法说明与对拍结果见 tools/gpu/docs/moc3.md。

实现要点：
  - keyform 插值：每个参数绑定求 (段号, t)，keyform 绑定做多线性组合，得到「每个 keyform 的权重」Wkf[B, NKF]；
    blend shape 也只是往 Wkf 里再加权重（blend shape 的 keyform 存的是增量）。之后所有 keyform 属性
    （网格点、顶点、不透明度、颜色、角度…）都是 Wkf 的线性函数：稀疏矩阵乘 / index_add。
  - 变形器按深度分层：第 L 层的父变形器都在第 L−1 层。第 L 步把「第 L 层 warp 的网格点、第 L 层 rotation 的原点与方向点、
    父在第 L−1 层的 mesh 顶点」按父类型（warp / rotation）分段，各用一个自定义 autograd Function 向量化变换（反向只存输入）。
  - 内部坐标与 Core 内部一致：y 向下的「单位」坐标（1 单位 = PixelsPerUnit 像素）；输出换成画布像素。
以下语义都由 Core 对拍 / 改字节探测确认（见文档）：key 吸附与范围外不可见（容差 0.001）、blend shape 不吸附、
约束取最小值、warp 网格外 (−2, 3) 带状区三角形插值 + 远处仿射、rotation 方向位移（warp 父 0.1、rotation 父 10）
与 baseAngle、draw order 取 int(x + 0.001)、颜色沿变形器链级联（乘算相乘、屏幕 1−(1−a)(1−b)）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import moc3 as M

# 旋转变形器在父变形器上测方向时用的位移（父为 warp 时是局部 [0,1] 坐标，父为 rotation 时是父的局部坐标）
DIR_EPS_WARP = 0.1
DIR_EPS_ROT = 10.0
# warp 外插：网格外 (−2, 3) 的带状区域在网格边界与仿射外推之间（三角形）插值，再往外纯仿射
WARP_BAND_LO = -2.0
WARP_BAND_HI = 3.0
# 参数离 key 不到 0.001 时吸附到 key；超出绑定的 key 范围 0.001 以上算「范围外」（实测 Core 的阈值）
KEY_RANGE_EPS = 1e-3
BLEND_SNAP = False       # blend shape 参数不吸附到 key（实测）
CONSTRAINT_SNAP = False  # blend shape 约束曲线不吸附（本模型的约束 key 都是整数，未单独验证）
# warp 变换按块计算：每块 B × 点数 ≈ 这么多元素（越大越快、临时显存越多；2^20 时 float32 每个临时量 4MB）
WARP_CHUNK_ELEMS = 1 << 20


def _t(a, device, dtype=None):
    t = torch.as_tensor(np.asarray(a), device=device)
    return t.to(dtype) if dtype is not None else t


@dataclass
class Output:
    verts: torch.Tensor         # [B, V, 2] 画布像素（所有 drawable 顶点按 drawable 顺序拼接，偏移见 L2D.vertex_offsets）
    opacity: torch.Tensor       # [B, D] 最终不透明度（含部件不透明度级联；不可见时按 Core 的约定，见 forward）
    draw_order: torch.Tensor    # [B, D] int64（Core 的 csmGetDrawableDrawOrders）
    render_order: torch.Tensor  # [B, D] int64（Core 的 csmGetRenderOrders：drawable 在绘制序列中的位置，0 最先画）
    multiply: torch.Tensor      # [B, D, 3] 乘算色
    screen: torch.Tensor        # [B, D, 3] 屏幕色
    visible: torch.Tensor       # [B, D] bool（Core 动态标志 csmIsVisible）
    vertex_updated: torch.Tensor  # [B, D] bool（Core 动态标志 csmVertexPositionsDidChange：本次变形了顶点 = 父变形器链有效且自身在 key 范围内）

    def vertices_of(self, model: 'L2D', d: int) -> torch.Tensor:
        o = model.vertex_offsets
        return self.verts[:, o[d]:o[d + 1]]

    def draw_sequence(self) -> torch.Tensor:
        """[B, D]：按绘制先后排列的 drawable 号（render_order 的逆排列）。"""
        return torch.argsort(self.render_order, dim=1)


def _warp_core(p, grid, c, du, dv, pid, C_i, R_i, goff, quad, want_grad):
    """warp 变换的核心（一块点）。p [B, N, 2]；grid [B, G, 2]（父所在层的世界网格）；c/du/dv [B, nW, 2]（父的仿射外推项）；
    pid/C_i/R_i/goff/quad [N]（父在层内的编号、列数、行数、网格起点、isQuadSource）。

    q = Σ_corner w_c·P_c，P_c 是网格点，或仿射外推点 A(u_c, v_c) = c + (u_c − ½)du + (v_c − ½)dv；(−2, 3)² 以外 q = A(u, v)。
    want_grad 时额外返回反向需要的量。"""
    B, N = p.shape[0], p.shape[1]
    u, v = p[..., 0], p[..., 1]
    C_i, R_i, goff = C_i[None], R_i[None], goff[None]

    def axis(s, n_i):
        # 一个轴：两端位置 a0/a1（局部坐标）、是否在网格线上 g0/g1、格线号 i0/i1、插值系数 t、dt/ds
        n = n_i.to(s.dtype)
        inside = (s >= 0) & (s <= 1)
        lo = s < 0
        x = s * n
        i = torch.minimum(torch.clamp(torch.floor(x).long(), min=0), n_i - 1)
        fi = i.to(s.dtype)
        band = 1.0 / (0 - WARP_BAND_LO)  # 带状区的 dt/ds（两侧带宽都是 2）
        t = torch.where(inside, x - fi, torch.where(lo, (s - WARP_BAND_LO) * band, (s - 1) * (1.0 / (WARP_BAND_HI - 1))))
        a0 = torch.where(inside, fi / n, torch.where(lo, WARP_BAND_LO, 1.0))
        a1 = torch.where(inside, (fi + 1) / n, torch.where(lo, 0.0, WARP_BAND_HI))
        i0 = torch.where(inside, i, torch.where(lo, 0, n_i.expand_as(i)))
        i1 = torch.where(inside, i + 1, i0)
        k = torch.where(inside, n.expand_as(s), band)
        return a0, a1, inside | ~lo, inside | lo, i0, i1, t, k, inside

    ua0, ua1, ug0, ug1, ui0, ui1, tx, kx, inu = axis(u, C_i)
    va0, va1, vg0, vg1, vi0, vi1, ty, ky, inv_ = axis(v, R_i)
    stride = C_i + 1
    cs = []  # 四个角点 (00, 10, 01, 11)：(位置 u, 位置 v, 在网格上?, 网格索引)
    for ua, ug, ui, va, vg, vi in ((ua0, ug0, ui0, va0, vg0, vi0), (ua1, ug1, ui1, va0, vg0, vi0),
                                   (ua0, ug0, ui0, va1, vg1, vi1), (ua1, ug1, ui1, va1, vg1, vi1)):
        on = ug & vg
        cs.append((ua, va, on, torch.where(on, goff + vi * stride + ui, 0)))
    # 角点权重：网格内双线性（isQuadSource = 0 时三角形），带状区三角形（沿 (1,0)–(0,1) 对角线分两半）
    tri = ~(inu & inv_ & quad[None])
    upper = tx + ty > 1
    zero = torch.zeros_like(tx)
    far = ~((u > WARP_BAND_LO) & (u < WARP_BAND_HI) & (v > WARP_BAND_LO) & (v < WARP_BAND_HI))
    ws = [torch.where(tri, torch.where(upper, zero, 1 - tx - ty), (1 - tx) * (1 - ty)),
          torch.where(tri, torch.where(upper, 1 - ty, tx), tx * (1 - ty)),
          torch.where(tri, torch.where(upper, 1 - tx, ty), (1 - tx) * ty),
          torch.where(tri, torch.where(upper, tx + ty - 1, zero), tx * ty)]
    ws = [torch.where(far, zero, w) for w in ws]
    # 仿射部分合起来：Σ_{不在网格上的角} w_c·A(u_c, v_c) + far·A(u, v) = sw·c + su·du + sv·dv
    sw = far.to(p.dtype)
    su = torch.where(far, u - 0.5, zero)
    sv = torch.where(far, v - 0.5, zero)
    q = None
    for w, (ua, va, on, gidx) in zip(ws, cs):
        off = torch.where(on, zero, w)
        sw = sw + off
        su = su + off * (ua - 0.5)
        sv = sv + off * (va - 0.5)
        g = torch.gather(grid, 1, gidx[..., None].expand(B, N, 2)) * torch.where(on, w, zero)[..., None]
        q = g if q is None else q + g
    cp = c[:, pid]
    dup = du[:, pid]
    dvp = dv[:, pid]
    q = q + sw[..., None] * cp + su[..., None] * dup + sv[..., None] * dvp
    if not want_grad:
        return q
    one = torch.ones_like(tx)
    # ∂w/∂tx、∂w/∂ty（角点 00、10、01、11）
    dwx = [torch.where(tri, torch.where(upper, zero, -one), ty - 1), torch.where(tri, torch.where(upper, zero, one), 1 - ty),
           torch.where(tri, torch.where(upper, -one, zero), -ty), torch.where(tri, torch.where(upper, one, zero), ty)]
    dwy = [torch.where(tri, torch.where(upper, zero, -one), tx - 1), torch.where(tri, torch.where(upper, -one, zero), -tx),
           torch.where(tri, torch.where(upper, zero, one), 1 - tx), torch.where(tri, torch.where(upper, one, zero), tx)]
    return ws, cs, dwx, dwy, kx, ky, far, sw, su, sv, cp, dup, dvp


class _WarpApply(torch.autograd.Function):
    """warp 变换。自定义反向：只保存输入，反向时按块重算，不保存逐点的中间量（显存约降一个数量级）。"""

    @staticmethod
    def forward(ctx, p, grid, c, du, dv, pid, C_i, R_i, goff, quad, chunk):
        ctx.save_for_backward(p, grid, c, du, dv, pid, C_i, R_i, goff, quad)
        ctx.chunk = chunk
        outs = [_warp_core(p[:, s:s + chunk], grid, c, du, dv, pid[s:s + chunk], C_i[s:s + chunk], R_i[s:s + chunk], goff[s:s + chunk],
                           quad[s:s + chunk], False) for s in range(0, p.shape[1], chunk)]
        return torch.cat(outs, 1) if len(outs) > 1 else outs[0]

    @staticmethod
    def backward(ctx, gq):
        p, grid, c, du, dv, pid, C_i, R_i, goff, quad = ctx.saved_tensors
        chunk = ctx.chunk
        B, N = p.shape[0], p.shape[1]
        need_p, need_grid = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_aff = any(ctx.needs_input_grad[2:5])
        gp = torch.empty_like(p) if need_p else None
        ggrid = torch.zeros_like(grid) if need_grid else None
        gc, gdu, gdv = (torch.zeros_like(c), torch.zeros_like(du), torch.zeros_like(dv)) if need_aff else (None, None, None)
        with torch.no_grad():
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                n = e - s
                ws, cs, dwx, dwy, kx, ky, far, sw, su, sv, cp, dup, dvp = _warp_core(p[:, s:e], grid, c, du, dv, pid[s:e], C_i[s:e], R_i[s:e],
                                                                                     goff[s:e], quad[s:e], True)
                g = gq[:, s:e]
                if need_aff:
                    idx = pid[s:e][None, :, None].expand(B, n, 2)
                    gc.scatter_add_(1, idx, g * sw[..., None])
                    gdu.scatter_add_(1, idx, g * su[..., None])
                    gdv.scatter_add_(1, idx, g * sv[..., None])
                dqx = dqy = None
                for k in range(4):
                    ua, va, on, gidx = cs[k]
                    gi = gidx[..., None].expand(B, n, 2)
                    if need_grid:
                        ggrid.scatter_add_(1, gi, g * torch.where(on, ws[k], 0)[..., None])
                    if need_p:
                        # 角点的值 P_c（网格点或仿射外推点）
                        P = torch.where(on[..., None], torch.gather(grid, 1, gi), cp + (ua - 0.5)[..., None] * dup + (va - 0.5)[..., None] * dvp)
                        tx_ = P * dwx[k][..., None]
                        ty_ = P * dwy[k][..., None]
                        dqx = tx_ if dqx is None else dqx + tx_
                        dqy = ty_ if dqy is None else dqy + ty_
                if need_p:
                    gp[:, s:e, 0] = torch.where(far, (g * dup).sum(-1), (g * dqx).sum(-1) * kx)
                    gp[:, s:e, 1] = torch.where(far, (g * dvp).sum(-1), (g * dqy).sum(-1) * ky)
        return gp, ggrid, gc, gdu, gdv, None, None, None, None, None, None


class _RotApply(torch.autograd.Function):
    """rotation 变换 q = o + sc·R(θ)·diag(rx, ry)·p（按 pid 取父的参数）。自定义反向，只保存输入。"""

    @staticmethod
    def forward(ctx, p, o, cs, sn, sc, rx, ry, pid):
        ctx.save_for_backward(p, o, cs, sn, sc, rx, ry, pid)
        x = p[..., 0] * rx[:, pid]
        y = p[..., 1] * ry[:, pid]
        c_, s_, k_ = cs[:, pid], sn[:, pid], sc[:, pid]
        return torch.stack([o[:, pid, 0] + k_ * (c_ * x - s_ * y), o[:, pid, 1] + k_ * (s_ * x + c_ * y)], -1)

    @staticmethod
    def backward(ctx, g):
        p, o, cs, sn, sc, rx, ry, pid = ctx.saved_tensors
        B, n = p.shape[0], p.shape[1]
        rxp, ryp = rx[:, pid], ry[:, pid]
        x = p[..., 0] * rxp
        y = p[..., 1] * ryp
        c_, s_, k_ = cs[:, pid], sn[:, pid], sc[:, pid]
        gx, gy = g[..., 0], g[..., 1]
        gp = torch.stack([k_ * (c_ * gx + s_ * gy) * rxp, k_ * (c_ * gy - s_ * gx) * ryp], -1) if ctx.needs_input_grad[0] else None
        idx = pid[None].expand(B, n)
        go = torch.zeros_like(o).scatter_add_(1, idx[..., None].expand(B, n, 2), g) if ctx.needs_input_grad[1] else None
        gcs = torch.zeros_like(cs).scatter_add_(1, idx, k_ * (x * gx + y * gy)) if ctx.needs_input_grad[2] else None
        gsn = torch.zeros_like(sn).scatter_add_(1, idx, k_ * (x * gy - y * gx)) if ctx.needs_input_grad[3] else None
        gsc = torch.zeros_like(sc).scatter_add_(1, idx, (c_ * x - s_ * y) * gx + (s_ * x + c_ * y) * gy) if ctx.needs_input_grad[4] else None
        return gp, go, gcs, gsn, gsc, None, None, None


class _Family:
    """一类带 keyform 的对象（部件 / warp / rotation / art mesh / glue）：keyform 绑定、keyform 区间、blend shape。"""

    def __init__(self, n, kb, kf_begin, nkf, owner, bs=None):
        self.n, self.kb, self.kf_begin, self.nkf, self.owner = n, kb, kf_begin, nkf, owner
        self.bs = bs  # (target[K], blend shape 参数绑定[K], keyform 起点[K], 约束索引[K, Cm]) 或 None


class _Stage:
    """第 L 步的常量：本层的 warp / rotation、父在上一层的 mesh，以及各段点的切片、父编号（见 L2D._plan_levels）。"""


class L2D(torch.nn.Module):
    """moc3 → 可微分变形器。forward(params[B,P], part_opacity[B,Q]=None) → Output。"""

    def __init__(self, moc: M.Moc3, device='cuda', dtype=torch.float32):
        super().__init__()
        self.moc = moc
        self.device = torch.device(device)
        self.dtype = dtype
        self.debug = None  # 设成 {} 时 forward 会把各层变形器的世界状态记进去（调试用）
        g, c = moc.groups, moc.counts
        dev = self.device
        cv = moc.canvas
        self.ppu, self.origin_x, self.origin_y = float(cv['pixelsPerUnit']), float(cv['originX']), float(cv['originY'])
        self.P, self.D, self.Q = c['parameters'], c['artMeshes'], c['parts']
        self.param_ids = list(g['parameters']['ids'])
        self.part_ids = list(g['parts']['ids'])
        self.drawable_ids = list(g['artMeshes']['ids'])
        self.param_min = g['parameters']['minValues'].copy()
        self.param_max = g['parameters']['maxValues'].copy()
        self.param_default = g['parameters']['defaultValues'].copy()

        # ---------- 参数绑定（普通）----------
        pb = g['parameterBindings']
        self._pb_param, self._pb_keys, self._pb_n = self._key_table(moc.binding_param, pb['keysSourcesBeginIndices'], pb['keysSourcesCounts'])
        self._pmin = _t(self.param_min, dev, dtype)[None]
        self._pmax = _t(self.param_max, dev, dtype)[None]
        kv = g['keys']['values']
        nbind = c['parameterBindings']
        lo = np.array([kv[b] for b in pb['keysSourcesBeginIndices']], np.float64) if nbind else np.zeros(1)
        hi = np.array([kv[b + n - 1] for b, n in zip(pb['keysSourcesBeginIndices'], pb['keysSourcesCounts'])], np.float64) if nbind else np.zeros(1)
        self._pb_lo = _t(lo - KEY_RANGE_EPS, dev, dtype)
        self._pb_hi = _t(hi + KEY_RANGE_EPS, dev, dtype)

        # ---------- keyform 绑定 ----------
        kbg = g['keyformBindings']
        NKB = c['keyformBindings']
        dims = [list(g['parameterBindingIndices']['bindingSourcesIndices'][b:b + n])
                for b, n in zip(kbg['parameterBindingIndexSourcesBeginIndices'], kbg['parameterBindingIndexSourcesCounts'])]
        Mx = max([len(d) for d in dims] + [1])
        kb_dims = np.full((max(NKB, 1), Mx), -1, np.int64)
        kb_stride = np.zeros((max(NKB, 1), Mx), np.int64)
        for i, d in enumerate(dims):
            s = 1
            for j, b in enumerate(d):
                kb_dims[i, j] = b
                kb_stride[i, j] = s
                s *= int(pb['keysSourcesCounts'][b])
        self._kb_has = _t(kb_dims >= 0, dev)
        self._kb_dims = _t(np.maximum(kb_dims, 0), dev)
        self._kb_stride = _t(kb_stride, dev)
        self._kb_nk1 = _t(np.maximum(np.asarray(pb['keysSourcesCounts'], np.int64)[np.maximum(kb_dims, 0)] - 1, 0) if nbind
                          else np.zeros_like(kb_dims), dev)
        corners = np.arange(1 << Mx)
        self._corner_bits = _t((corners[:, None] >> np.arange(Mx)[None]) & 1, dev)  # [C, M]

        # ---------- blend shape 参数绑定与约束 ----------
        self._bpb_param = None
        self._con_dummy = 0
        if moc.version >= 4 and c['blendShapeParameterBindings']:
            bpb = g['blendShapeParameterBindings']
            self._bpb_param, self._bpb_keys, self._bpb_n = self._key_table(moc.blendshape_binding_param, bpb['keysSourcesBeginIndices'],
                                                                           bpb['keysSourcesCounts'])
            # 约束：分段线性曲线（参数值 → 权重）；多个约束取最小值
            bc, bcv = g['blendShapeConstraints'], g['blendShapeConstraintValues']
            NC = c['blendShapeConstraints']
            Kc = int(max(bc['blendShapeConstraintValueSourcesCounts'].max(), 2)) if NC else 2
            ck = np.zeros((NC + 1, Kc), np.float64)
            cw = np.ones((NC + 1, Kc), np.float64)  # 最后一行是哑约束（恒为 1）
            cn = np.ones(NC + 1, np.int64)
            for i in range(NC):
                b, n = int(bc['blendShapeConstraintValueSourcesBeginIndices'][i]), int(bc['blendShapeConstraintValueSourcesCounts'][i])
                ck[i, :n], cw[i, :n], cn[i] = bcv['keys'][b:b + n], bcv['weights'][b:b + n], n
                ck[i, n:], cw[i, n:] = ck[i, n - 1], cw[i, n - 1]
            self._con_param = _t(np.concatenate([bc['parameterIndices'], [0]]).astype(np.int64), dev)
            self._con_keys, self._con_w, self._con_n = _t(ck, dev, dtype), _t(cw, dev, dtype), _t(cn, dev)
            self._con_dummy = NC

        # ---------- 各类对象 ----------
        self.parts = self._family('parts', 'partKeyforms', 'blendShapesParts')
        self.warps = self._family('warpDeformers', 'warpDeformerKeyforms', 'blendShapesWarpDeformers')
        self.rots = self._family('rotationDeformers', 'rotationDeformerKeyforms', 'blendShapesRotationDeformers')
        self.meshes = self._family('artMeshes', 'artMeshKeyforms', 'blendShapesArtMeshes')
        self.glues = self._family('glue', 'glueKeyforms', 'blendShapesGlue')

        # keyform 标量属性
        wk, rk, ak = g['warpDeformerKeyforms'], g['rotationDeformerKeyforms'], g['artMeshKeyforms']
        self._part_kf = _t(g['partKeyforms']['drawOrders'][:, None], dev, dtype)
        # warp：[不透明度, 乘算 RGB, 屏幕 RGB]
        self._warp_kf = torch.cat([_t(wk['opacities'][:, None], dev, dtype), self._kf_colors('warpDeformers', 'warpDeformerKeyforms')], 1)
        # rotation：[不透明度, 乘算 RGB, 屏幕 RGB, 角度, 原点 x, 原点 y, 缩放, 反射 x, 反射 y]
        self._rot_kf = torch.cat([_t(rk['opacities'][:, None], dev, dtype), self._kf_colors('rotationDeformers', 'rotationDeformerKeyforms'),
                                  _t(np.stack([rk['angles'], rk['originX'], rk['originY'], rk['scales'], rk['isReflectX'], rk['isReflectY']], 1),
                                     dev, dtype)], 1)
        # mesh：[不透明度, 乘算 RGB, 屏幕 RGB, draw order]
        self._mesh_kf = torch.cat([_t(ak['opacities'][:, None], dev, dtype), self._kf_colors('artMeshes', 'artMeshKeyforms'),
                                   _t(ak['drawOrders'][:, None], dev, dtype)], 1)
        self._glue_kf = _t(g['glueKeyforms']['intensities'][:, None], dev, dtype)

        self._plan_levels()
        self._plan_parts()
        self._glue_waves = self._glue_plan()
        self._draw_plan()

    # ================================================================ 编译

    def _key_table(self, owner_param, begin, count):
        keys = self.moc['keys']['values']
        n = np.asarray(count, np.int64)
        K = int(max(n.max(), 2)) if len(n) else 2
        kt = np.zeros((max(len(n), 1), K), np.float64)
        for i, (b, c) in enumerate(zip(begin, count)):
            b, c = int(b), int(c)
            kt[i, :c] = keys[b:b + c]
            kt[i, c:] = keys[b + c - 1] if c else 0
        own = np.asarray(owner_param, np.int64) if len(n) else np.zeros(1, np.int64)
        return _t(own, self.device), _t(kt, self.device, self.dtype), _t(n if len(n) else np.ones(1, np.int64), self.device)

    def _family(self, grp, kfgrp, bsgrp) -> _Family:
        g, c = self.moc.groups, self.moc.counts
        G = g[grp]
        n = c[grp]
        nkf = c[kfgrp]
        kb = G['keyformBindingSourcesIndices'].astype(np.int64)
        b, k = G['keyformSourcesBeginIndices'].astype(np.int64), G['keyformSourcesCounts'].astype(np.int64)
        owner = np.full(nkf, -1, np.int64)
        for i in range(n):
            owner[b[i]:b[i] + k[i]] = i
        bs = None
        if bsgrp and bsgrp in g and c.get(bsgrp, 0) and self._bpb_param is not None:
            S = g[bsgrp]
            bkb = g['blendShapeKeyformBindings']
            ci_all = g['blendShapeConstraintIndices']['blendShapeConstraintSourcesIndices']
            tg, pbs, kfb, cons = [], [], [], []
            for j in range(c[bsgrp]):
                t = int(S['targetIndices'][j])
                bb, bn = int(S['blendShapeKeyformBindingSourcesBeginIndices'][j]), int(S['blendShapeKeyformBindingSourcesCounts'][j])
                for q in range(bb, bb + bn):
                    tg.append(t)
                    pbs.append(int(bkb['parameterBindingSourcesIndices'][q]))
                    k0, kn = int(bkb['keyformSourcesBlendShapeIndices'][q]), int(bkb['keyformSourcesBlendShapeCounts'][q])
                    kfb.append(k0)
                    owner[k0:k0 + kn] = t
                    cb, cn = int(bkb['blendShapeConstraintIndexSourcesBeginIndices'][q]), int(bkb['blendShapeConstraintIndexSourcesCounts'][q])
                    cons.append(list(ci_all[cb:cb + cn]))
            Cm = max([len(x) for x in cons] + [1])
            ca = np.full((len(cons), Cm), self._con_dummy, np.int64)
            for i, x in enumerate(cons):
                ca[i, :len(x)] = x
            dev = self.device
            bs = tuple(_t(np.asarray(x, np.int64), dev) for x in (tg, pbs, kfb, ca))
        owner[owner < 0] = 0  # 没有对象引用的 keyform：权重恒为 0，归到 0 号对象无影响
        dev = self.device
        return _Family(n, _t(kb, dev), _t(b, dev), nkf, _t(owner, dev), bs)

    def _kf_colors(self, grp, kfgrp):
        """每个 keyform 的 [乘算 RGB, 屏幕 RGB]。"""
        g = self.moc.groups
        nkf = self.moc.counts[kfgrp]
        if self.moc.version < 4:
            return _t(np.concatenate([np.ones((nkf, 3)), np.zeros((nkf, 3))], 1), self.device, self.dtype)
        mul = np.stack([g['keyformColorsMultiply'][ch] for ch in 'rgb'], 1)
        scr = np.stack([g['keyformColorsScreen'][ch] for ch in 'rgb'], 1)
        if self.moc.version >= 5:
            mi = g[kfgrp]['keyformMultiplyColorSourcesIndices'].astype(np.int64)
            si = g[kfgrp]['keyformScreenColorSourcesIndices'].astype(np.int64)
        else:
            # V4.02：颜色按对象的 keyformColorSourcesBeginIndices 连续存放（blend shape keyform 不带颜色 → 增量 0；未验证）
            G = g[grp]
            mi = np.full(nkf, -1, np.int64)
            for b, k, cb in zip(G['keyformSourcesBeginIndices'], G['keyformSourcesCounts'], G['keyformColorSourcesBeginIndices']):
                mi[b:b + k] = np.arange(cb, cb + k)
            si = mi
        col = np.concatenate([np.where(mi[:, None] >= 0, mul[np.maximum(mi, 0)], 0), np.where(si[:, None] >= 0, scr[np.maximum(si, 0)], 0)], 1)
        return _t(col, self.device, self.dtype)

    def _pos_matrix(self, grp, kfgrp, fam: _Family, row_perm):
        """位置稀疏矩阵 A [2·Npts, NKF]（CSR）：A @ Wkf^T = 插值后的点坐标；行按 row_perm（新序 → 原点号）排列。"""
        g = self.moc.groups
        G = g[grp]
        xy = g['keyformPositions']['xys']
        vc = G['vertexCounts'].astype(np.int64)
        pofs = np.concatenate([[0], np.cumsum(vc)])
        posb = g[kfgrp]['keyformPositionSourcesBeginIndices'].astype(np.int64)
        owner = fam.owner.cpu().numpy()
        newpos = np.empty(len(row_perm), np.int64)
        newpos[row_perm] = np.arange(len(row_perm))
        rows, cols, vals = [], [], []
        for k in range(fam.nkf):
            o = owner[k]
            n = vc[o]
            v = xy[posb[k]:posb[k] + 2 * n]
            if not v.any():
                continue
            pts = newpos[pofs[o]:pofs[o] + n]
            rows.append(np.stack([2 * pts, 2 * pts + 1], 1).reshape(-1))
            cols.append(np.full(2 * n, k, np.int64))
            vals.append(v.astype(np.float64))
        rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        with torch.sparse.check_sparse_tensor_invariants(False):
            A = torch.sparse_coo_tensor(torch.as_tensor(np.stack([rows, cols])), torch.as_tensor(vals), (2 * len(row_perm), max(fam.nkf, 1)),
                                        dtype=self.dtype, device=self.device).coalesce()
        return A.to_sparse_csr()

    def _plan_levels(self):
        """按深度分层，并为每一步准备切片与索引（全部是常量张量，forward 里没有逐对象的 Python 循环）。

        每层内的 warp / rotation、每步的 mesh 都按「父是 warp 的在前、父是 rotation 的在后」排好，
        这样网格点、mesh 顶点在 grid_local / mesh_local 里都是连续切片，变换时不用拼接复制。"""
        g, c = self.moc.groups, self.moc.counts
        dev, dt = self.device, self.dtype
        D, W, R, A = g['deformers'], g['warpDeformers'], g['rotationDeformers'], g['artMeshes']
        pdef = D['parentDeformerIndices'].astype(np.int64)
        nd = len(pdef)
        types = D['types'].astype(np.int64)
        spec = D['specificSourcesIndices'].astype(np.int64)
        depth = np.zeros(nd, np.int64)
        for i in range(nd):  # 父总在前面（moc3.validate 保证）
            depth[i] = depth[pdef[i]] + 1 if pdef[i] >= 0 else 0
        nL = int(depth.max()) + 1 if nd else 0
        enabled = (D['isEnabled'] != 0) & (D['isVisible'] != 0)
        ptype = np.where(pdef >= 0, types[np.maximum(pdef, 0)], -1)    # 父的类型（-1 = 没有父）

        def by_parent(ids, par_type):
            """父为 warp 的在前，父为 rotation 的在后（稳定）。"""
            return ids[np.argsort(par_type != M.DEFORMER_WARP, kind='stable')] if len(ids) else ids

        lw = [by_parent(np.nonzero((depth == L) & (types == M.DEFORMER_WARP))[0], ptype[(depth == L) & (types == M.DEFORMER_WARP)])
              for L in range(nL)]
        lr = [by_parent(np.nonzero((depth == L) & (types == M.DEFORMER_ROTATION))[0], ptype[(depth == L) & (types == M.DEFORMER_ROTATION)])
              for L in range(nL)]
        local = np.zeros(nd, np.int64)  # 变形器在本层内的编号（warp、rotation 分开编）
        for L in range(nL):
            local[lw[L]] = np.arange(len(lw[L]))
            local[lr[L]] = np.arange(len(lr[L]))
        self._def_depth, self._def_local, self._def_type = depth, local, types
        self.n_levels = nL

        vcw = W['vertexCounts'].astype(np.int64)
        gofs = np.concatenate([[0], np.cumsum(vcw)])
        vcm = A['vertexCounts'].astype(np.int64)
        self.vertex_offsets = np.concatenate([[0], np.cumsum(vcm)])
        self.V = int(self.vertex_offsets[-1])
        mpar = A['parentDeformerIndices'].astype(np.int64)
        mptype = np.where(mpar >= 0, types[np.maximum(mpar, 0)], -1)
        mstage = np.where(mpar >= 0, depth[np.maximum(mpar, 0)] + 1, 0)   # mesh 在哪一步变换

        def parent_rows(par, Lp):
            """父变形器 → 上一层属性表里的行号（warp 在前，rotation 在后）。"""
            return np.where(types[par] == M.DEFORMER_WARP, local[par], len(lw[Lp]) + local[par])

        def warp_meta(par, prev):
            """父为 warp 的点：(父在上一层的编号, 列数, 行数, 网格起点, isQuadSource)。"""
            pid = local[par]
            return (_t(pid, dev), prev.w_cols[_t(pid, dev)], prev.w_rows[_t(pid, dev)], prev.w_goff[_t(pid, dev)], prev.w_quad[_t(pid, dev)])

        stages = []
        grid_perm, mesh_order, vert_order = [], [], []
        g_start = v_start = 0
        prev = None
        for L in range(nL + 1):
            st = _Stage()
            ws = lw[L] if L < nL else np.zeros(0, np.int64)
            rs = lr[L] if L < nL else np.zeros(0, np.int64)
            ms = by_parent(np.nonzero(mstage == L)[0], mptype[mstage == L])
            wi, ri = spec[ws], spec[rs]
            st.L = L
            st.nW, st.nR, st.nM = len(ws), len(rs), len(ms)
            st.w_spec, st.r_spec = _t(wi, dev), _t(ri, dev)
            st.w_enabled, st.r_enabled = _t(enabled[ws], dev), _t(enabled[rs], dev)
            # warp 网格块（本层 warp 的网格点连续存放）
            vc = vcw[wi]
            goff = np.concatenate([[0], np.cumsum(vc)])
            st.gs, st.gn = g_start, int(goff[-1])
            g_start += st.gn
            grid_perm.append(np.concatenate([np.arange(gofs[w], gofs[w + 1]) for w in wi]) if len(wi) else np.zeros(0, np.int64))
            cols, rows = W['columns'][wi].astype(np.int64), W['rows'][wi].astype(np.int64)
            st.w_goff, st.w_cols, st.w_rows = _t(goff[:-1], dev), _t(cols, dev), _t(rows, dev)
            quad = (W['isQuadSource'][wi] != 0) if self.moc.version >= 2 else np.zeros(len(wi), bool)
            st.w_quad = _t(quad, dev)
            st.w_corners = _t(np.stack([goff[:-1], goff[:-1] + cols, goff[:-1] + rows * (cols + 1), goff[:-1] + rows * (cols + 1) + cols], 1)
                              if len(wi) else np.zeros((0, 4), np.int64), dev)
            st.r_base = _t(np.deg2rad(R['baseAngles'][ri].astype(np.float64)), dev, dt)
            # 本步的 mesh 顶点（连续存放）
            mesh_order.append(ms)
            vs = np.concatenate([np.arange(self.vertex_offsets[m], self.vertex_offsets[m + 1]) for m in ms]) if len(ms) else np.zeros(0, np.int64)
            vert_order.append(vs)
            st.vs, st.nV = v_start, len(vs)
            v_start += st.nV
            if L > 0:
                gpar = np.repeat(pdef[ws], vc)
                vpar = np.repeat(mpar[ms], vcm[ms])
                st.gw = int((types[gpar] == M.DEFORMER_WARP).sum())      # 父为 warp 的网格点个数（在前）
                st.rw = int((types[pdef[rs]] == M.DEFORMER_WARP).sum())  # 父为 warp 的 rotation 个数（在前）
                st.vw = int((types[vpar] == M.DEFORMER_WARP).sum())      # 父为 warp 的 mesh 顶点个数（在前）
                # 原点 + 方向点：父为 warp 的 [org_w, tip_w]，父为 rotation 的 [org_r, tip_r]
                rp = pdef[rs]
                st.meta_g = warp_meta(gpar[:st.gw], prev) if st.gw else None
                st.meta_o = warp_meta(np.concatenate([rp[:st.rw], rp[:st.rw]]), prev) if st.rw else None
                st.meta_v = warp_meta(vpar[:st.vw], prev) if st.vw else None
                st.rpid_g = _t(local[gpar[st.gw:]], dev)
                st.rpid_o = _t(local[np.concatenate([rp[st.rw:], rp[st.rw:]])], dev)
                st.rpid_v = _t(local[vpar[st.vw:]], dev)
                # 方向位移：父为 warp 时 0.1（局部坐标），父为 rotation 时 10
                st.r_eps = _t(np.where(types[rp] == M.DEFORMER_WARP, DIR_EPS_WARP, DIR_EPS_ROT), dev, dt)
                # 对象（warp、rotation、mesh）的父在上一层属性表里的行号
                st.w_prow = _t(parent_rows(pdef[ws], L - 1), dev)
                st.r_prow = _t(parent_rows(rp, L - 1), dev)
                st.m_prow = _t(parent_rows(mpar[ms], L - 1), dev)
            stages.append(st)
            if st.nW or st.nR:
                prev = st
        self.stages = stages
        self._grid_perm = np.concatenate(grid_perm) if grid_perm else np.zeros(0, np.int64)
        vo = np.concatenate(vert_order)
        self._mesh_inv = _t(np.argsort(np.concatenate(mesh_order), kind='stable'), dev)
        self._vert_inv = _t(np.argsort(vo, kind='stable'), dev)
        self._A_warp = self._pos_matrix('warpDeformers', 'warpDeformerKeyforms', self.warps, self._grid_perm)
        self._A_mesh = self._pos_matrix('artMeshes', 'artMeshKeyforms', self.meshes, vo)
        self._mesh_enabled = _t((A['isEnabled'] != 0) & (A['isVisible'] != 0), dev)

    def _plan_parts(self):
        g = self.moc.groups
        Pt, A = g['parts'], g['artMeshes']
        Q = self.Q
        ppar = Pt['parentPartIndices'].astype(np.int64)
        chains = []
        for i in range(Q):
            ch, x = [], i
            while x >= 0:
                ch.append(x)
                x = int(ppar[x])
            chains.append(ch)
        Lp = max([len(ch) for ch in chains] + [1])
        pc = np.full((Q + 1, Lp), Q, np.int64)  # 填充指向「恒为 1」的哑部件；最后一行 = 没有父部件
        for i, ch in enumerate(chains):
            pc[i, :len(ch)] = ch
        self._part_chain = _t(pc, self.device)
        self._mesh_part = _t(np.where(A['parentPartIndices'] < 0, Q, A['parentPartIndices']).astype(np.int64), self.device)
        self._part_enabled = _t((Pt['isEnabled'] != 0) & (Pt['isVisible'] != 0), self.device)

    def _glue_plan(self):
        """glue 按文件顺序执行；相邻且互不共享 mesh 的 glue 合成一波，一波一次向量化。"""
        g, c = self.moc.groups, self.moc.counts
        if not c['glue']:
            return []
        gl, gi = g['glue'], g['glueInfo']
        waves, cur, used = [], [], set()
        for i in range(c['glue']):
            a, b = int(gl['artMeshIndicesA'][i]), int(gl['artMeshIndicesB'][i])
            if a in used or b in used:
                waves.append(cur)
                cur, used = [], set()
            cur.append(i)
            used |= {a, b}
        if cur:
            waves.append(cur)
        vo = self.vertex_offsets
        dev = self.device
        raw = []
        for wv in waves:
            ga, gb, wa, wb, gid = [], [], [], [], []
            for i in wv:
                b, n = int(gl['glueInfoSourcesBeginIndices'][i]), int(gl['glueInfoSourcesCounts'][i])
                ma, mb = int(gl['artMeshIndicesA'][i]), int(gl['artMeshIndicesB'][i])
                ga.append(vo[ma] + gi['positionIndices'][b:b + n:2].astype(np.int64))
                gb.append(vo[mb] + gi['positionIndices'][b + 1:b + n:2].astype(np.int64))
                wa.append(gi['weights'][b:b + n:2])
                wb.append(gi['weights'][b + 1:b + n:2])
                gid.append(np.full(n // 2, i, np.int64))
            raw.append([np.concatenate(x) for x in (ga, gb, wa, wb, gid)])
        # 只在「被 glue 到的顶点」的子集上逐波计算，最后一次写回（避免每波复制整个顶点数组）
        U = np.unique(np.concatenate([np.concatenate([r[0], r[1]]) for r in raw]))
        self._glue_verts = _t(U, dev)
        out = []
        for ga, gb, wa, wb, gid in raw:
            out.append((_t(np.searchsorted(U, ga), dev), _t(np.searchsorted(U, gb), dev), _t(wa, dev, self.dtype), _t(wb, dev, self.dtype),
                        _t(gid, dev)))
        return out

    def _draw_plan(self):
        """render order：从根 draw order 组开始，组内按 draw order 稳定排序，部件对象展开成它自己的组。
        每个 drawable 记下路径上每层的 (排序键来源, 组内位置)，求值时从内到外逐层稳定排序。"""
        g = self.moc.groups
        dg, go = g['drawOrderGroups'], g['drawOrderGroupObjects']
        D = self.D
        paths = [None] * D  # drawable → [(kind, index, pos)]，kind 0 = art mesh 的 draw order，1 = 部件的 draw order

        def walk(grp, prefix, depth=0):
            if depth > 64:
                raise M.Moc3Error('draw order 组嵌套过深（可能有环）')
            b, n = int(dg['objectSourcesBeginIndices'][grp]), int(dg['objectSourcesCounts'][grp])
            for pos in range(n):
                j = b + pos
                t, idx, selfi = int(go['types'][j]), int(go['indices'][j]), int(go['selfIndices'][j])
                if t == 0:
                    paths[idx] = prefix + [(0, idx, pos)]
                elif selfi >= 0:
                    walk(selfi, prefix + [(1, idx, pos)], depth + 1)
        if self.moc.counts['drawOrderGroups']:
            walk(0, [])
        if any(p is None for p in paths):
            raise M.Moc3Error('有 drawable 不在 draw order 组里')
        Lm = max([len(p) for p in paths] + [1])
        kind = np.zeros((D, Lm), np.int64)
        idx = np.zeros((D, Lm), np.int64)
        pos = np.full((D, Lm), -1, np.int64)
        for d, p in enumerate(paths):
            for lvl, (k, i, ps) in enumerate(p):
                kind[d, lvl], idx[d, lvl], pos[d, lvl] = k, i, ps
        if pos.max() >= 1 << 20:
            raise M.Moc3Error('draw order 组太大')
        self._do_kind, self._do_idx, self._do_pos = (_t(x, self.device) for x in (kind, idx, pos))
        self._do_levels = Lm

    # ================================================================ 求值：keyform

    @staticmethod
    def _segment(v, keys, n, snap=True):
        """v [B, J]，keys [J, K]（升序，尾部用最后一个 key 填充），n [J] → 段号 idx [B, J]、t [B, J]（夹到 [0,1]）。"""
        B = v.shape[0]
        K = keys.shape[1]
        valid = torch.arange(K, device=v.device)[None] < n[:, None]
        cnt = ((keys[None] <= v[..., None]) & valid[None]).sum(-1)
        idx = torch.minimum(torch.clamp(cnt - 1, min=0), torch.clamp(n - 2, min=0)[None])
        kk = keys[None].expand(B, -1, -1)
        k0 = torch.gather(kk, 2, idx[..., None])[..., 0]
        k1 = torch.gather(kk, 2, torch.minimum(idx + 1, (n - 1).clamp(min=0)[None])[..., None])[..., 0]
        den = k1 - k0
        single = (n <= 1)[None] | (den == 0)
        t = torch.where(single, torch.zeros_like(v), (v - k0) / torch.where(single, torch.ones_like(den), den))
        if snap:
            # 普通绑定：离 key 不到 KEY_RANGE_EPS 时吸附到 key 上（Core 的行为，实测；blend shape 不吸附）
            t = torch.where(v - k0 <= KEY_RANGE_EPS, torch.zeros_like(t), t)
            t = torch.where(k1 - v <= KEY_RANGE_EPS, torch.ones_like(t), t)
        return idx, t.clamp(0, 1)

    def _keyform_weights(self, fam: _Family, kb_w, kb_off, bs_seg):
        """对象族的 keyform 权重 Wkf [B, NKF]。"""
        B = kb_w.shape[0]
        W = kb_w.new_zeros(B, max(fam.nkf, 1))
        if fam.n == 0 or fam.nkf == 0:
            return W
        w = kb_w[:, fam.kb]                     # [B, n, C]
        off = kb_off[:, fam.kb] + fam.kf_begin[None, :, None]
        W = W.scatter_add(1, off.reshape(B, -1), w.reshape(B, -1))
        if fam.bs is not None:
            tg, pbs, kfb, cons = fam.bs
            idx, t, cw = bs_seg
            idx, t = idx[:, pbs], t[:, pbs]
            wgt = cw[:, cons].amin(-1)          # 多个约束取最小值（实测，不是乘积）
            i0 = kfb[None] + idx
            i1 = kfb[None] + torch.minimum(idx + 1, (self._bpb_n[pbs] - 1)[None])
            W = W.scatter_add(1, i0, (1 - t) * wgt).scatter_add(1, i1, t * wgt)
        return W

    @staticmethod
    def _per_object(Wkf, fam: _Family, kf_vals):
        """Σ_kf Wkf·val → 每个对象 [B, n, F]。"""
        out = Wkf.new_zeros(Wkf.shape[0], fam.n, kf_vals.shape[1])
        if fam.n == 0:
            return out
        return out.index_add(1, fam.owner, Wkf[..., None] * kf_vals[None])

    @staticmethod
    def _spmm(A, Wkf):
        """A [R, NKF] 稀疏 × Wkf^T → [B, R/2, 2]（[R, B] 结果的置换视图，不复制）。"""
        r = torch.sparse.mm(A, Wkf.t().contiguous())
        return r.view(-1, 2, Wkf.shape[0]).permute(2, 0, 1)

    # ================================================================ 求值：变形

    @staticmethod
    def _warp_affine(grid, st):
        """warp 的网格外仿射近似 A(u, v) = c + (u − ½)·du + (v − ½)·dv，由四个角点求得。返回 [B, nW, 2] × 3。"""
        B = grid.shape[0]
        p = torch.gather(grid, 1, st.w_corners.reshape(1, -1, 1).expand(B, -1, 2)).reshape(B, -1, 4, 2)
        p00, p10, p01, p11 = p.unbind(2)
        c = (p00 + p10 + p01 + p11) * 0.25
        du = ((p10 - p00) + (p11 - p01)) * 0.5
        dv = ((p01 - p00) + (p11 - p10)) * 0.5
        return c, du, dv

    @staticmethod
    def _warp_apply(p, grid, c, du, dv, meta):
        """p [B, N, 2]（父 warp 的局部坐标，网格 = [0,1]²）→ 世界坐标 [B, N, 2]。meta = (pid, 列数, 行数, 网格起点, isQuadSource)（各 [N]）。

        网格内：格子内双线性（isQuadSource = 0 时三角形）；网格外 (−2, 3)² 的带状区：在网格边界点与仿射外推点
        （A 在虚拟网格线 −2 / 3 上的值）之间做三角形插值；更远处：纯仿射 A(u, v)。"""
        B = p.shape[0]
        chunk = max(1024, WARP_CHUNK_ELEMS // max(B, 1))  # 按块计算，限制临时量大小
        return _WarpApply.apply(p, grid, c, du, dv, *meta, chunk)

    def _apply_parents(self, st: _Stage, src, w_meta, r_pid, w_n, prev_grid, aff, rot):
        """把一段点（前 w_n 个的父是 warp，其余父是 rotation）变换到世界坐标。"""
        out = []
        if w_n:
            out.append(self._warp_apply(src[:, :w_n], prev_grid, *aff, w_meta))
        if src.shape[1] > w_n:
            out.append(_RotApply.apply(src[:, w_n:], *rot, r_pid))
        if not out:
            return src
        return out[0] if len(out) == 1 else torch.cat(out, 1)

    # ================================================================ 主流程

    def forward(self, params: torch.Tensor, part_opacity: torch.Tensor | None = None) -> Output:
        """params [B, P]（模型参数值，顺序同 param_ids），part_opacity [B, Q]（部件不透明度，缺省全 1）。"""
        params = params.to(self.device, self.dtype)
        B = params.shape[0]
        dev, dt = self.device, self.dtype
        # Core 先把参数夹到 [min, max]
        params = torch.maximum(torch.minimum(params, self._pmax), self._pmin)

        # 1. 参数绑定 → 段号与 t；keyform 绑定的角点权重与偏移；key 范围检查
        pidx, pt = self._segment(params[:, self._pb_param], self._pb_keys, self._pb_n)   # [B, NB]
        has = self._kb_has[None, :, None, :]                    # [1, NKB, 1, M]
        bit = self._corner_bits[None, None]                     # [1, 1, C, M]
        tt = pt[:, self._kb_dims][:, :, None, :]                # [B, NKB, 1, M]
        f = torch.where(bit == 1, tt, 1 - tt)
        f = torch.where(has, f, (bit == 0).to(dt))
        kb_w = f.prod(-1)                                       # [B, NKB, C]
        ki = torch.minimum(pidx[:, self._kb_dims][:, :, None, :] + bit, self._kb_nk1[None, :, None, :])
        kb_off = (torch.where(has, ki, 0) * self._kb_stride[None, :, None, :]).sum(-1)
        # 参数超出绑定的 key 范围（容差 KEY_RANGE_EPS）→ 对象不可见，且其子孙不会被 Core 求值
        pv = params[:, self._pb_param]
        pb_ok = (pv >= self._pb_lo[None]) & (pv <= self._pb_hi[None])
        kb_ok = (pb_ok[:, self._kb_dims] | ~self._kb_has[None]).all(-1)   # [B, NKB]

        bs_seg = None
        if self._bpb_param is not None:
            bidx, bt = self._segment(params[:, self._bpb_param], self._bpb_keys, self._bpb_n, snap=BLEND_SNAP)
            ci, ct = self._segment(params[:, self._con_param], self._con_keys, self._con_n, snap=CONSTRAINT_SNAP)
            cwt = self._con_w[None].expand(B, -1, -1)
            w0 = torch.gather(cwt, 2, ci[..., None])[..., 0]
            w1 = torch.gather(cwt, 2, torch.minimum(ci + 1, (self._con_n - 1)[None])[..., None])[..., 0]
            cw = w0 + (w1 - w0) * ct
            cw = torch.cat([cw[:, :-1], torch.ones_like(cw[:, -1:])], 1)  # 哑约束恒为 1
            bs_seg = (bidx, bt, cw)

        # 2. 各族 keyform 权重与插值属性
        Wp = self._keyform_weights(self.parts, kb_w, kb_off, bs_seg)
        Ww = self._keyform_weights(self.warps, kb_w, kb_off, bs_seg)
        Wr = self._keyform_weights(self.rots, kb_w, kb_off, bs_seg)
        Wm = self._keyform_weights(self.meshes, kb_w, kb_off, bs_seg)
        part_do = self._per_object(Wp, self.parts, self._part_kf)[..., 0]     # [B, Q]
        warp_v = self._per_object(Ww, self.warps, self._warp_kf)                # [B, nW, 7]
        rot_v = self._per_object(Wr, self.rots, self._rot_kf)                   # [B, nR, 13]
        mesh_v = self._per_object(Wm, self.meshes, self._mesh_kf)               # [B, D, 8]
        grid_local = self._spmm(self._A_warp, Ww)                               # [B, G, 2]（层序）
        mesh_local = self._spmm(self._A_mesh, Wm)                               # [B, V, 2]

        # 3. 逐层：变形器的世界状态（网格 / 仿射），以及父在上一层的 mesh 顶点
        # 属性表 props：[不透明度, 乘算 RGB, 屏幕 RGB, 有效(1/0), 总缩放]，本层 warp 在前、rotation 在后
        prev = None
        prev_state = None
        mesh_pts, mesh_props = [], []
        for st in self.stages:
            L = st.L
            wv = warp_v[:, st.w_spec]                              # [B, nW, 7]
            rv = rot_v[:, st.r_spec]                               # [B, nR, 13]
            w_ok = (kb_ok[:, self.warps.kb[st.w_spec]] & st.w_enabled[None]).to(dt)
            r_ok = (kb_ok[:, self.rots.kb[st.r_spec]] & st.r_enabled[None]).to(dt)
            # rv 列：0 不透明度，1:4 乘算，4:7 屏幕，7 角度（度），8:10 原点，10 缩放，11/12 反射 x/y
            ang = rv[..., 7] * (math.pi / 180)
            org = rv[..., 8:10]
            sc = rv[..., 10]
            rx = torch.where(rv[..., 11] > 0.5, -1.0, 1.0).to(dt)
            ry = torch.where(rv[..., 12] > 0.5, -1.0, 1.0).to(dt)
            grid = grid_local[:, st.gs:st.gs + st.gn]
            mloc = mesh_local[:, st.vs:st.vs + st.nV]
            if L == 0:
                w_props = torch.cat([wv, w_ok[..., None], torch.ones_like(w_ok)[..., None]], -1)
                r_props = torch.cat([rv[..., :7], r_ok[..., None], sc[..., None]], -1)
                mesh_pts.append(mloc)
                root = torch.tensor([1, 1, 1, 1, 0, 0, 0, 1, 1], device=dev, dtype=dt)  # 没有父变形器
                mesh_props.append(root.expand(B, st.nM, 9))
            else:
                # 网格点、rotation 原点与方向点、mesh 顶点，各自按父类型分两段变换
                ow = org[:, :st.rw]
                orr = org[:, st.rw:]
                eps = st.r_eps[None]
                tip = torch.stack([org[..., 0], org[..., 1] - eps], -1)
                ot = torch.cat([ow, tip[:, :st.rw], orr, tip[:, st.rw:]], 1)
                grid = self._apply_parents(st, grid, st.meta_g, st.rpid_g, st.gw, grid_prev, aff_prev, rot_prev)
                otw = self._apply_parents(st, ot, st.meta_o, st.rpid_o, 2 * st.rw, grid_prev, aff_prev, rot_prev)
                mesh_pts.append(self._apply_parents(st, mloc, st.meta_v, st.rpid_v, st.vw, grid_prev, aff_prev, rot_prev))
                o_w = torch.cat([otw[:, :st.rw], otw[:, 2 * st.rw:2 * st.rw + st.nR - st.rw]], 1)
                t_w = torch.cat([otw[:, st.rw:2 * st.rw], otw[:, 2 * st.rw + st.nR - st.rw:]], 1)
                # 父的属性（不透明度、颜色、有效性、总缩放）
                pw = prev_state[:, st.w_prow]
                pr = prev_state[:, st.r_prow]
                mesh_props.append(prev_state[:, st.m_prow])
                w_props = torch.cat([wv[..., :1] * pw[..., :1], wv[..., 1:4] * pw[..., 1:4], 1 - (1 - wv[..., 4:7]) * (1 - pw[..., 4:7]),
                                     w_ok[..., None] * pw[..., 7:8], pw[..., 8:9]], -1)
                d = t_w - o_w
                # 方向：父变换后「向上」那一小段的转角（y 向下坐标中顺时针为正）；世界角 = keyform 角 + 父方向角 + baseAngle
                ang = ang + torch.atan2(d[..., 0], -d[..., 1]) + st.r_base[None]
                sc = sc * pr[..., 8]
                org = o_w
                r_props = torch.cat([rv[..., :1] * pr[..., :1], rv[..., 1:4] * pr[..., 1:4], 1 - (1 - rv[..., 4:7]) * (1 - pr[..., 4:7]),
                                     r_ok[..., None] * pr[..., 7:8], sc[..., None]], -1)
            if st.nW or st.nR:
                grid_prev = grid
                aff_prev = self._warp_affine(grid, st) if st.nW else (grid.new_zeros(B, 0, 2),) * 3
                rot_prev = (org, torch.cos(ang), torch.sin(ang), sc, rx, ry)  # 各 [B, nR(, 2)]
                prev_state = torch.cat([w_props, r_props], 1)
                prev = st
                if self.debug is not None:
                    self.debug.setdefault('levels', []).append({'grid': grid, 'aff': aff_prev, 'rot': rot_prev, 'props': prev_state, 'stage': st})

        # 4. art mesh：顶点、不透明度、颜色
        verts = torch.cat(mesh_pts, 1)[:, self._vert_inv]                 # [B, V, 2]（内部单位）
        par = torch.cat(mesh_props, 1)[:, self._mesh_inv]                 # [B, D, 9]
        computed = par[..., 7] > 0.5                                       # 父变形器链都有效，Core 才会求值这个 mesh
        own_ok = kb_ok[:, self.meshes.kb] & self._mesh_enabled[None]      # 自身绑定在 key 范围内
        mop = mesh_v[..., 0] * par[..., 0]
        mmul = mesh_v[..., 1:4] * par[..., 1:4]
        mscr = 1 - (1 - mesh_v[..., 4:7]) * (1 - par[..., 4:7])

        # 5. glue（世界坐标里把成对的顶点按权重拉到一起）
        if self.glues.n:
            Wg = self._keyform_weights(self.glues, kb_w, kb_off, bs_seg)
            glue_int = self._per_object(Wg, self.glues, self._glue_kf)[..., 0]
            sub = verts[:, self._glue_verts]
            for ga, gb, wa, wb, gid in self._glue_waves:
                it = glue_int[:, gid][..., None]
                pa, pb_ = sub[:, ga], sub[:, gb]
                na = pa + (pb_ - pa) * (wa[None, :, None] * it)
                nb = pb_ + (pa - pb_) * (wb[None, :, None] * it)
                sub = sub.index_copy(1, ga, na).index_copy(1, gb, nb)
            verts = verts.index_copy(1, self._glue_verts, sub)

        # 6. 部件不透明度级联（祖先部件相乘）
        if part_opacity is None:
            part_opacity = torch.ones(B, self.Q, device=dev, dtype=dt)
        po = torch.cat([part_opacity.to(dev, dt) * self._part_enabled.to(dt), torch.ones(B, 1, device=dev, dtype=dt)], 1)
        part_eff = po[:, self._part_chain].prod(-1)                       # [B, Q+1]（最后一列 = 没有父部件）
        mop = mop * part_eff[:, self._mesh_part]
        visible = computed & own_ok & (mop > 0)
        # 自身绑定超出 key 范围：Core 只做了 keyform 插值（不透明度 / 颜色 / draw order 是自身 keyform 的值，不乘父与部件），
        # 变形与级联都跳过，且不可见。父链无效（没有求值）时 Core 不写这些值（新模型里是 0）。
        cmp_ = computed.to(dt)
        opacity = torch.where(own_ok, mop, mesh_v[..., 0]) * cmp_
        mmul = torch.where(own_ok[..., None], mmul, mesh_v[..., 1:4]) * cmp_[..., None]
        mscr = torch.where(own_ok[..., None], mscr, mesh_v[..., 4:7]) * cmp_[..., None]

        # 7. draw order / render order（render order 用插值得到的 draw order，不管可见与否）
        draw_order = self._int_draw_order(mesh_v[..., 7])
        render_order = self._render_order(draw_order, self._int_draw_order(part_do))
        draw_order = draw_order * computed.long()

        # 坐标：内部 y 向下单位 → 画布像素
        px = torch.stack([self.origin_x + verts[..., 0] * self.ppu, self.origin_y + verts[..., 1] * self.ppu], -1)
        return Output(verts=px, opacity=opacity, draw_order=draw_order, render_order=render_order, multiply=mmul, screen=mscr,
                      visible=visible, vertex_updated=computed & own_ok)

    @staticmethod
    def _int_draw_order(x):
        # Core：int(插值后的 draw order + 0.001)（实测 500.68 → 500，500.9990 → 501，500.9988 → 500）；按 float32 计算
        return torch.floor(x.detach().float() + 0.001).long()

    def _render_order(self, draw_order, part_order):
        """逐层稳定排序：从最内层到最外层，键 = 该层对象的 draw order，再按组内位置。"""
        B, D = draw_order.shape
        kind, idx, pos = self._do_kind, self._do_idx, self._do_pos
        perm = torch.arange(D, device=self.device)[None].expand(B, D)
        Q = part_order.shape[1]
        for lvl in range(self._do_levels - 1, -1, -1):
            valid = pos[:, lvl] >= 0
            dov = torch.where(kind[:, lvl] == 0, draw_order[:, idx[:, lvl].clamp(max=D - 1)], part_order[:, idx[:, lvl].clamp(max=Q - 1)])
            key = torch.where(valid[None], dov * (1 << 20) + pos[None, :, lvl], torch.full_like(dov, -1))
            o = torch.argsort(torch.gather(key, 1, perm), dim=1, stable=True)
            perm = torch.gather(perm, 1, o)
        ro = torch.empty_like(perm)
        ro.scatter_(1, perm, torch.arange(D, device=self.device)[None].expand(B, D))
        return ro


def load(path, device='cuda', dtype=torch.float32, **kw) -> L2D:
    return L2D(M.load(path), device=device, dtype=dtype, **kw)
