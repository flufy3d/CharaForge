"""A7：光栅化（raster.py）快速路径的 Triton 融合核。

  - mask_layer：mask 缓冲深度剥离的一层，整幅画布逐像素（不压缩片元）：重心坐标 → uv / LOD → 贴图 alpha → Π(1 − a)；
  - composite_fwd：主绘制，每像素沿剥离层从后往前（GL 绘制顺序）着色 + 混合（含每步夹取 / 8 位取整）；
  - bwd_pass1：同上，并记录每层混合前的下层合成 D_below（以及边缘梯度项要用的 X = B_k、A_k）；
  - bwd_pass2：从前往后反传：夹取遮罩、∂/∂B_k、∂/∂A_k，再手写的着色反向（重心 → 顶点、uv → 三线性采样、mask 采样位置、
    乘算 / 屏幕色、不透明度、混合方式），用原子加累加到顶点 / drawable。

着色公式与 raster.Rasterizer._shade / TextureSet.sample / _sample_masks / _render_masks 逐式相同（同样的运算顺序，关闭 FMA 合并），
前向与原实现逐像素相同（误差只来自 bmm 等库函数的舍入，见 docs/perf.md）。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 128
_OPTS = dict(num_warps=4, enable_fp_fusion=False)


# =============================================================================== 公共小函数

@triton.jit
def _texel(data_ptr, idx, m):
    v = tl.load(data_ptr + idx, m, other=0)
    return ((v & 255).to(tl.float32), ((v >> 8) & 255).to(tl.float32), ((v >> 16) & 255).to(tl.float32),
            ((v >> 24) & 255).to(tl.float32))


@triton.jit
def _texel_a(data_ptr, idx, m):
    v = tl.load(data_ptr + idx, m, other=0)
    return ((v >> 24) & 255).to(tl.float32)


@triton.jit
def _bil_setup(tab_ptr, tex, lev, s, tt, m, NLEV):
    """TextureSet._bilinear 的坐标部分。返回 4 个纹素下标、fx、fy，以及 ∂x/∂s、∂y/∂tt（含夹取的梯度遮罩）。"""
    e = (tex * NLEV + lev) * 3
    H = tl.load(tab_ptr + e, m, other=1)
    Wd = tl.load(tab_ptr + e + 1, m, other=1)
    off = tl.load(tab_ptr + e + 2, m, other=0)
    Hf = H.to(tl.float32)
    Wf = Wd.to(tl.float32)
    xr = s * Wf - 0.5
    yr = tt * Hf - 0.5
    xc = tl.maximum(xr, 0.0)
    yc = tl.maximum(yr, 0.0)
    xm = Wf - 1.0
    ym = Hf - 1.0
    x = tl.minimum(xc, xm)
    y = tl.minimum(yc, ym)
    x0f = tl.floor(x)
    y0f = tl.floor(y)
    fx = x - x0f
    fy = y - y0f
    x0 = x0f.to(tl.int64)
    y0 = y0f.to(tl.int64)
    x1 = tl.minimum(x0 + 1, Wd - 1)
    y1 = tl.minimum(y0 + 1, H - 1)
    # 反向（PyTorch 2.14 的约定，实测）：clamp(min=0) 只在 xr > 0 处通过（端点处为 0）；minimum 在 x < W−1 处通过、相等处各一半
    kx = tl.where(xr > 0, 1.0, 0.0) * tl.where(xc < xm, 1.0, tl.where(xc == xm, 0.5, 0.0)) * Wf
    ky = tl.where(yr > 0, 1.0, 0.0) * tl.where(yc < ym, 1.0, tl.where(yc == ym, 0.5, 0.0)) * Hf
    return off + y0 * Wd + x0, off + y0 * Wd + x1, off + y1 * Wd + x0, off + y1 * Wd + x1, fx, fy, kx, ky


@triton.jit
def _lerp4(g00, g01, g10, g11, fx, fy):
    return (g00 * (1 - fx) + g01 * fx) * (1 - fy) + (g10 * (1 - fx) + g11 * fx) * fy


@triton.jit
def _lod(tab_ptr, maxlev_ptr, tex, da0, da1, da2, da3, m, NLEV, lod_bias):
    """TextureSet.sample 的 LOD：返回 l0、l1、fr（不求导）。"""
    e = tex * NLEV * 3
    H0 = tl.load(tab_ptr + e, m, other=1).to(tl.float32)
    W0 = tl.load(tab_ptr + e + 1, m, other=1).to(tl.float32)
    dsx = da0 * W0
    dsy = da1 * W0
    dtx = da2 * H0
    dty = da3 * H0
    rho2 = tl.maximum(dsx * dsx + dtx * dtx, dsy * dsy + dty * dty)
    lam = 0.5 * libdevice.log2(tl.maximum(rho2, 1e-30)) + lod_bias
    ml = tl.load(maxlev_ptr + tex, m, other=0)
    lam = tl.minimum(lam, ml.to(tl.float32))
    mag = lam <= 0
    lam = tl.maximum(lam, 0.0)
    l0f = tl.floor(lam)
    l0 = l0f.to(tl.int64)
    l1 = tl.minimum(l0 + 1, ml)
    fr = lam - l0f
    fr = tl.where(mag, 0.0, fr)
    return l0, l1, fr


@triton.jit
def _sample_rgba(data_ptr, tab_ptr, tex, l0, l1, fr, s, tt, m, NLEV):
    i00, i01, i10, i11, fx, fy, kx, ky = _bil_setup(tab_ptr, tex, l0, s, tt, m, NLEV)
    r00, g00, b00, a00 = _texel(data_ptr, i00, m)
    r01, g01, b01, a01 = _texel(data_ptr, i01, m)
    r10, g10, b10, a10 = _texel(data_ptr, i10, m)
    r11, g11, b11, a11 = _texel(data_ptr, i11, m)
    r0 = _lerp4(r00, r01, r10, r11, fx, fy)
    g0 = _lerp4(g00, g01, g10, g11, fx, fy)
    b0 = _lerp4(b00, b01, b10, b11, fx, fy)
    a0 = _lerp4(a00, a01, a10, a11, fx, fy)
    i00, i01, i10, i11, fx, fy, kx, ky = _bil_setup(tab_ptr, tex, l1, s, tt, m, NLEV)
    r00, g00, b00, a00 = _texel(data_ptr, i00, m)
    r01, g01, b01, a01 = _texel(data_ptr, i01, m)
    r10, g10, b10, a10 = _texel(data_ptr, i10, m)
    r11, g11, b11, a11 = _texel(data_ptr, i11, m)
    r1 = _lerp4(r00, r01, r10, r11, fx, fy)
    g1 = _lerp4(g00, g01, g10, g11, fx, fy)
    b1 = _lerp4(b00, b01, b10, b11, fx, fy)
    a1 = _lerp4(a00, a01, a10, a11, fx, fy)
    return ((r0 * (1 - fr) + r1 * fr) / 255.0, (g0 * (1 - fr) + g1 * fr) / 255.0, (b0 * (1 - fr) + b1 * fr) / 255.0,
            (a0 * (1 - fr) + a1 * fr) / 255.0)


@triton.jit
def _dlerp(g00, g01, g10, g11, fx, fy):
    """双线性对 fx、fy 的偏导。"""
    return (g01 - g00) * (1 - fy) + (g11 - g10) * fy, (g10 * (1 - fx) + g11 * fx) - (g00 * (1 - fx) + g01 * fx)


@triton.jit
def _sample_rgba_grad(data_ptr, tab_ptr, tex, l0, l1, fr, s, tt, m, NLEV, dr_, dg_, db_, da_):
    """给定 ∂L/∂(r,g,b,a)（采样结果，已 /255 之后），返回 ∂L/∂s、∂L/∂tt。"""
    i00, i01, i10, i11, fx, fy, kx, ky = _bil_setup(tab_ptr, tex, l0, s, tt, m, NLEV)
    r00, g00, b00, a00 = _texel(data_ptr, i00, m)
    r01, g01, b01, a01 = _texel(data_ptr, i01, m)
    r10, g10, b10, a10 = _texel(data_ptr, i10, m)
    r11, g11, b11, a11 = _texel(data_ptr, i11, m)
    w0 = (1 - fr) / 255.0
    rx, ry = _dlerp(r00, r01, r10, r11, fx, fy)
    gx, gy = _dlerp(g00, g01, g10, g11, fx, fy)
    bx, by = _dlerp(b00, b01, b10, b11, fx, fy)
    ax, ay = _dlerp(a00, a01, a10, a11, fx, fy)
    ds = (dr_ * rx + dg_ * gx + db_ * bx + da_ * ax) * w0 * kx
    dt = (dr_ * ry + dg_ * gy + db_ * by + da_ * ay) * w0 * ky
    i00, i01, i10, i11, fx, fy, kx, ky = _bil_setup(tab_ptr, tex, l1, s, tt, m, NLEV)
    r00, g00, b00, a00 = _texel(data_ptr, i00, m)
    r01, g01, b01, a01 = _texel(data_ptr, i01, m)
    r10, g10, b10, a10 = _texel(data_ptr, i10, m)
    r11, g11, b11, a11 = _texel(data_ptr, i11, m)
    w1 = fr / 255.0
    rx, ry = _dlerp(r00, r01, r10, r11, fx, fy)
    gx, gy = _dlerp(g00, g01, g10, g11, fx, fy)
    bx, by = _dlerp(b00, b01, b10, b11, fx, fy)
    ax, ay = _dlerp(a00, a01, a10, a11, fx, fy)
    ds = ds + (dr_ * rx + dg_ * gx + db_ * bx + da_ * ax) * w1 * kx
    dt = dt + (dr_ * ry + dg_ * gy + db_ * by + da_ * ay) * w1 * ky
    return ds, dt


@triton.jit
def _sample_alpha(data_ptr, tab_ptr, tex, l0, l1, fr, s, tt, m, NLEV):
    i00, i01, i10, i11, fx, fy, kx, ky = _bil_setup(tab_ptr, tex, l0, s, tt, m, NLEV)
    a0 = _lerp4(_texel_a(data_ptr, i00, m), _texel_a(data_ptr, i01, m), _texel_a(data_ptr, i10, m), _texel_a(data_ptr, i11, m), fx, fy)
    i00, i01, i10, i11, fx, fy, kx, ky = _bil_setup(tab_ptr, tex, l1, s, tt, m, NLEV)
    a1 = _lerp4(_texel_a(data_ptr, i00, m), _texel_a(data_ptr, i01, m), _texel_a(data_ptr, i10, m), _texel_a(data_ptr, i11, m), fx, fy)
    return (a0 * (1 - fr) + a1 * fr) / 255.0


@triton.jit
def _bary(p0x, p0y, p1x, p1y, p2x, p2y, px, py):
    """Rasterizer._frag_bary。"""
    e1x = p1x - p0x
    e1y = p1y - p0y
    e2x = p2x - p0x
    e2y = p2y - p0y
    den = e1x * e2y - e1y * e2x
    small = tl.abs(den) < 1e-20
    den = tl.where(small, 1e-20, den)
    qx = px - p0x
    qy = py - p0y
    w1 = libdevice.div_rn(qx * e2y - qy * e2x, den)
    w2 = libdevice.div_rn(e1x * qy - e1y * qx, den)
    w0 = 1.0 - w1 - w2
    # 除法一律用 IEEE 舍入（Triton 的 / 是近似除法，与 PyTorch 不逐位相同）
    return (w0, w1, w2, libdevice.div_rn(e2y, den), libdevice.div_rn(-e2x, den), libdevice.div_rn(-e1y, den), libdevice.div_rn(e1x, den),
            small, den, e1x, e1y, e2x, e2y, qx, qy)


@triton.jit
def _uv_and_da(U0x, U0y, U1x, U1y, U2x, U2y, w0, w1, w2, d0, d1, d2, d3):
    uvx = U0x * w0 + U1x * w1 + U2x * w2
    uvy = U0y * w0 + U1y * w1 + U2y * w2
    dU1x = U1x - U0x
    dU1y = U1y - U0y
    dU2x = U2x - U0x
    dU2y = U2y - U0y
    return uvx, uvy, d0 * dU1x + d2 * dU2x, d1 * dU1x + d3 * dU2x, d0 * dU1y + d2 * dU2y, d1 * dU1y + d3 * dU2y


# =============================================================================== mask 缓冲（一层）

@triton.jit
def _mask_layer_kernel(rast_ptr, prod_ptr, flag_ptr, total, HWc, CW, Np,
                       Pf_ptr, pos_ptr, ptri_ptr, pctx_ptr, pdraw_ptr, ptex_ptr, puv_ptr, rect_ptr, cull_ptr,
                       data_ptr, tab_ptr, maxlev_ptr, NLEV, lod_bias, CULL: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    tid = tl.load(rast_ptr + offs * 4 + 3, m, other=0.0).to(tl.int64)
    act = m & (tid > 0)
    if tl.sum(act.to(tl.int32)) > 0:
        tl.store(flag_ptr, 1)  # 本层非空（剥离循环据此结束，省掉单独的整幅判空）
        bidx = offs // HWc
        q = offs % HWc
        t = tl.where(act, tid - 1, 0)
        c = tl.load(pctx_ptr + t, act, other=0)
        xi = q % CW
        yi = q // CW
        ox = tl.load(rect_ptr + c * 4, act, other=0)
        oy = tl.load(rect_ptr + c * 4 + 1, act, other=0)
        ow = tl.load(rect_ptr + c * 4 + 2, act, other=0)
        oh = tl.load(rect_ptr + c * 4 + 3, act, other=0)
        own = (xi >= ox) & (xi < ox + ow) & (yi >= oy) & (yi < oy + oh)
        v0 = tl.load(ptri_ptr + t * 3, act, other=0).to(tl.int64)
        v1 = tl.load(ptri_ptr + t * 3 + 1, act, other=0).to(tl.int64)
        v2 = tl.load(ptri_ptr + t * 3 + 2, act, other=0).to(tl.int64)
        base = bidx * Np
        p0x = tl.load(Pf_ptr + (base + v0) * 2, act, other=0.0)
        p0y = tl.load(Pf_ptr + (base + v0) * 2 + 1, act, other=0.0)
        p1x = tl.load(Pf_ptr + (base + v1) * 2, act, other=0.0)
        p1y = tl.load(Pf_ptr + (base + v1) * 2 + 1, act, other=1.0)
        p2x = tl.load(Pf_ptr + (base + v2) * 2, act, other=1.0)
        p2y = tl.load(Pf_ptr + (base + v2) * 2 + 1, act, other=0.0)
        px = xi.to(tl.float32) + 0.5
        py = yi.to(tl.float32) + 0.5
        w0, w1, w2, d0, d1, d2, d3, small, den, e1x, e1y, e2x, e2y, qx, qy = _bary(p0x, p0y, p1x, p1y, p2x, p2y, px, py)
        U0x = tl.load(puv_ptr + v0 * 2, act, other=0.0)
        U0y = tl.load(puv_ptr + v0 * 2 + 1, act, other=0.0)
        U1x = tl.load(puv_ptr + v1 * 2, act, other=0.0)
        U1y = tl.load(puv_ptr + v1 * 2 + 1, act, other=0.0)
        U2x = tl.load(puv_ptr + v2 * 2, act, other=0.0)
        U2y = tl.load(puv_ptr + v2 * 2 + 1, act, other=0.0)
        uvx, uvy, da0, da1, da2, da3 = _uv_and_da(U0x, U0y, U1x, U1y, U2x, U2y, w0, w1, w2, d0, d1, d2, d3)
        tex = tl.load(ptex_ptr + t, act, other=0)
        l0, l1, fr = _lod(tab_ptr, maxlev_ptr, tex, da0, da1, da2, da3, act, NLEV, lod_bias)
        a = _sample_alpha(data_ptr, tab_ptr, tex, l0, l1, fr, uvx, 1.0 - uvy, act, NLEV)
        a = tl.where(own, a, 0.0)
        if CULL:
            # mask 矩阵行列式恒正：正面 = 裁剪空间逆时针；被剔除的背面片元贡献 0
            c0x = tl.load(pos_ptr + (base + v0) * 4, act, other=0.0)
            c0y = tl.load(pos_ptr + (base + v0) * 4 + 1, act, other=0.0)
            c1x = tl.load(pos_ptr + (base + v1) * 4, act, other=0.0)
            c1y = tl.load(pos_ptr + (base + v1) * 4 + 1, act, other=0.0)
            c2x = tl.load(pos_ptr + (base + v2) * 4, act, other=0.0)
            c2y = tl.load(pos_ptr + (base + v2) * 4 + 1, act, other=0.0)
            area = (c1x - c0x) * (c2y - c0y) - (c1y - c0y) * (c2x - c0x)
            dr = tl.load(pdraw_ptr + t, act, other=0)
            cl = tl.load(cull_ptr + dr, act, other=0) != 0
            a = a * tl.where(cl & (area < 0), 0.0, 1.0)
        pr = tl.load(prod_ptr + offs, act, other=1.0)
        tl.store(prod_ptr + offs, pr * (1.0 - a), act)


def mask_layer(rast, prod, Pf, pos, geom_t, atlas_t, CW, culling) -> bool:
    """rast [nb, CH, CW, 4]（剥离层），prod [nb·CH·CW]（就地乘）。geom_t / atlas_t 见 raster.Rasterizer._fast_tensors。
    返回本层是否有片元（一次同步）。"""
    nb, CH = rast.shape[0], rast.shape[1]
    total = nb * CH * CW
    Np = Pf.shape[1]
    flag = torch.zeros(1, dtype=torch.int32, device=rast.device)
    _mask_layer_kernel[(triton.cdiv(total, 256),)](rast, prod, flag, total, CH * CW, CW, Np, Pf, pos,
                                                   geom_t["ptri"], geom_t["pctx"], geom_t["pdraw"], geom_t["ptex"], geom_t["puv"], geom_t["rect"],
                                                   geom_t["cull"], atlas_t["data"], atlas_t["tab"], atlas_t["maxlev"], atlas_t["nlev"],
                                                   atlas_t["bias"], CULL=culling, BLOCK=256, **_OPTS)
    return bool(flag.item())


@triton.jit
def _tid_plane_kernel(rast_ptr, out_ptr, flag_ptr, total, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    t = tl.load(rast_ptr + offs * 4 + 3, m, other=0.0).to(tl.int32)
    tl.store(out_ptr + offs, t, m)
    if tl.sum((t > 0).to(tl.int32)) > 0:
        tl.store(flag_ptr, 1)


def tid_plane(rast):
    """剥离层 rast [B, H, W, 4] → 三角形号（id + 1，0 = 空）int32 [B·H·W]，以及本层是否非空（一次同步）。"""
    total = rast.shape[0] * rast.shape[1] * rast.shape[2]
    out = torch.empty(total, dtype=torch.int32, device=rast.device)
    flag = torch.zeros(1, dtype=torch.int32, device=rast.device)
    _tid_plane_kernel[(triton.cdiv(total, 1024),)](rast, out, flag, total, BLOCK=1024, num_warps=4)
    return out, bool(flag.item())


# =============================================================================== 主绘制：一个片元的着色

@triton.jit
def _frag_geom(p, tid, act, W, HW, V, S_ptr, tri_ptr, tdraw_ptr, uv_ptr):
    b = p // HW
    pix = p % HW
    py = (pix // W).to(tl.float32) + 0.5
    px = (pix % W).to(tl.float32) + 0.5
    t = tl.where(act, tid - 1, 0)
    v0 = tl.load(tri_ptr + t * 3, act, other=0).to(tl.int64)
    v1 = tl.load(tri_ptr + t * 3 + 1, act, other=1).to(tl.int64)
    v2 = tl.load(tri_ptr + t * 3 + 2, act, other=2).to(tl.int64)
    d = tl.load(tdraw_ptr + t, act, other=0).to(tl.int64)
    g0 = b * V + v0
    g1 = b * V + v1
    g2 = b * V + v2
    p0x = tl.load(S_ptr + g0 * 2, act, other=0.0)
    p0y = tl.load(S_ptr + g0 * 2 + 1, act, other=0.0)
    p1x = tl.load(S_ptr + g1 * 2, act, other=1.0)
    p1y = tl.load(S_ptr + g1 * 2 + 1, act, other=0.0)
    p2x = tl.load(S_ptr + g2 * 2, act, other=0.0)
    p2y = tl.load(S_ptr + g2 * 2 + 1, act, other=1.0)
    U0x = tl.load(uv_ptr + v0 * 2, act, other=0.0)
    U0y = tl.load(uv_ptr + v0 * 2 + 1, act, other=0.0)
    U1x = tl.load(uv_ptr + v1 * 2, act, other=0.0)
    U1y = tl.load(uv_ptr + v1 * 2 + 1, act, other=0.0)
    U2x = tl.load(uv_ptr + v2 * 2, act, other=0.0)
    U2y = tl.load(uv_ptr + v2 * 2 + 1, act, other=0.0)
    return b, px, py, v0, v1, v2, d, g0, g1, g2, p0x, p0y, p1x, p1y, p2x, p2y, U0x, U0y, U1x, U1y, U2x, U2y


@triton.jit
def _mask_sample(mflat_ptr, mtab_ptr, b, c, cx, cy, act, R, MHW):
    """Rasterizer._sample_masks：返回 1 − texel 的双线性值，及其对 (x, y)（mask 像素）的偏导。"""
    off = tl.load(mtab_ptr + c * 6, act, other=0)
    i0 = tl.load(mtab_ptr + c * 6 + 1, act, other=0)
    j0 = tl.load(mtab_ptr + c * 6 + 2, act, other=0)
    w = tl.load(mtab_ptr + c * 6 + 3, act, other=0)
    h = tl.load(mtab_ptr + c * 6 + 4, act, other=0)
    st = tl.load(mtab_ptr + c * 6 + 5, act, other=0)
    x = cx * R - 0.5 - i0.to(tl.float32)
    y = cy * R - 0.5 - j0.to(tl.float32)
    x0f = tl.floor(x)
    y0f = tl.floor(y)
    fx = x - x0f
    fy = y - y0f
    x0 = x0f.to(tl.int64)
    y0 = y0f.to(tl.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    mb = mflat_ptr + b * MHW
    ok = act & (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
    t00 = tl.load(mb + off + tl.maximum(y0, 0) * st + tl.maximum(x0, 0), ok, other=1.0)
    t00 = tl.where(ok, t00, 1.0)
    ok = act & (x1 >= 0) & (x1 < w) & (y0 >= 0) & (y0 < h)
    t01 = tl.load(mb + off + tl.maximum(y0, 0) * st + tl.maximum(x1, 0), ok, other=1.0)
    t01 = tl.where(ok, t01, 1.0)
    ok = act & (x0 >= 0) & (x0 < w) & (y1 >= 0) & (y1 < h)
    t10 = tl.load(mb + off + tl.maximum(y1, 0) * st + tl.maximum(x0, 0), ok, other=1.0)
    t10 = tl.where(ok, t10, 1.0)
    ok = act & (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
    t11 = tl.load(mb + off + tl.maximum(y1, 0) * st + tl.maximum(x1, 0), ok, other=1.0)
    t11 = tl.where(ok, t11, 1.0)
    v = _lerp4(t00, t01, t10, t11, fx, fy)
    dvx, dvy = _dlerp(t00, t01, t10, t11, fx, fy)
    return 1.0 - v, dvx, dvy


@triton.jit
def _shade(p, tid, act, W, HW, V, n, NLEV, lod_bias, R, MHW,
           S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr, cull_ptr,
           op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr, Md_ptr, NCTX,
           HAS_MASK: tl.constexpr, CULL: tl.constexpr):
    """Rasterizer._shade 的一个片元：返回混合系数 B_k（4）、A_k（4）与 blend 方式。"""
    (b, px, py, v0, v1, v2, d, g0, g1, g2, p0x, p0y, p1x, p1y, p2x, p2y,
     U0x, U0y, U1x, U1y, U2x, U2y) = _frag_geom(p, tid, act, W, HW, V, S_ptr, tri_ptr, tdraw_ptr, uv_ptr)
    w0, w1, w2, d0, d1, d2, d3, small, den, e1x, e1y, e2x, e2y, qx, qy = _bary(p0x, p0y, p1x, p1y, p2x, p2y, px, py)
    uvx, uvy, da0, da1, da2, da3 = _uv_and_da(U0x, U0y, U1x, U1y, U2x, U2y, w0, w1, w2, d0, d1, d2, d3)
    tex = tl.load(texi_ptr + d, act, other=0).to(tl.int64)
    l0, l1, fr = _lod(tab_ptr, maxlev_ptr, tex, da0, da1, da2, da3, act, NLEV, lod_bias)
    tr, tg, tb, ta = _sample_rgba(data_ptr, tab_ptr, tex, l0, l1, fr, uvx, 1.0 - uvy, act, NLEV)
    bd = b * n + d
    mr = tl.load(mul_ptr + bd * 4, act, other=1.0)
    mg = tl.load(mul_ptr + bd * 4 + 1, act, other=1.0)
    mbb = tl.load(mul_ptr + bd * 4 + 2, act, other=1.0)
    sr = tl.load(scr_ptr + bd * 4, act, other=0.0)
    sg = tl.load(scr_ptr + bd * 4 + 1, act, other=0.0)
    sb = tl.load(scr_ptr + bd * 4 + 2, act, other=0.0)
    xr1 = tr * mr
    xg1 = tg * mg
    xb1 = tb * mbb
    xr2 = xr1 + sr * ta - xr1 * sr
    xg2 = xg1 + sg * ta - xg1 * sg
    xb2 = xb1 + sb * ta - xb1 * sb
    op = tl.load(op_ptr + bd, act, other=0.0)
    cr = xr2 * op
    cg = xg2 * op
    cb = xb2 * op
    ca = ta * op
    if HAS_MASK:
        c = tl.load(dctx_ptr + d, act, other=-1).to(tl.int64)
        masked = act & (c >= 0)
        cc = tl.maximum(c, 0)
        u0x = tl.load(U_ptr + g0 * 2, masked, other=0.0)
        u0y = tl.load(U_ptr + g0 * 2 + 1, masked, other=0.0)
        u1x = tl.load(U_ptr + g1 * 2, masked, other=0.0)
        u1y = tl.load(U_ptr + g1 * 2 + 1, masked, other=0.0)
        u2x = tl.load(U_ptr + g2 * 2, masked, other=0.0)
        u2y = tl.load(U_ptr + g2 * 2 + 1, masked, other=0.0)
        pux = u0x * w0 + u1x * w1 + u2x * w2
        puy = u0y * w0 + u1y * w1 + u2y * w2
        mo = (b * NCTX + cc) * 6
        m00 = tl.load(Md_ptr + mo, masked, other=0.0)
        m01 = tl.load(Md_ptr + mo + 1, masked, other=0.0)
        m02 = tl.load(Md_ptr + mo + 2, masked, other=0.0)
        m10 = tl.load(Md_ptr + mo + 3, masked, other=0.0)
        m11 = tl.load(Md_ptr + mo + 4, masked, other=0.0)
        m12 = tl.load(Md_ptr + mo + 5, masked, other=0.0)
        cx = (m00 * pux + m01 * puy) + m02
        cy = (m10 * pux + m11 * puy) + m12
        mv, dvx, dvy = _mask_sample(mflat_ptr, mtab_ptr, b, cc, cx, cy, masked, R, MHW)
        inv = tl.load(inv_ptr + d, masked, other=0) != 0
        mv = tl.where(inv, 1.0 - mv, mv)
        f = tl.where(masked, mv, 1.0)
        cr = cr * f
        cg = cg * f
        cb = cb * f
        ca = ca * f
    if CULL:
        q0x = tl.load(P_ptr + g0 * 2, act, other=0.0)
        q0y = tl.load(P_ptr + g0 * 2 + 1, act, other=0.0)
        q1x = tl.load(P_ptr + g1 * 2, act, other=0.0)
        q1y = tl.load(P_ptr + g1 * 2 + 1, act, other=0.0)
        q2x = tl.load(P_ptr + g2 * 2, act, other=0.0)
        q2y = tl.load(P_ptr + g2 * 2 + 1, act, other=0.0)
        ar = (q1x - q0x) * (q2y - q0y) - (q1y - q0y) * (q2x - q0x)
        cl = (tl.load(cull_ptr + d, act, other=0) != 0) & (ar > 0)
        cr = tl.where(cl, 0.0, cr)
        cg = tl.where(cl, 0.0, cg)
        cb = tl.where(cl, 0.0, cb)
        ca = tl.where(cl, 0.0, ca)
    bl = tl.load(blend_ptr + d, act, other=0)
    nrm = bl == 0
    add = bl == 1
    ia = 1.0 - ca
    B0 = tl.where(nrm | add, cr, 0.0)
    B1 = tl.where(nrm | add, cg, 0.0)
    B2 = tl.where(nrm | add, cb, 0.0)
    B3 = tl.where(nrm, ca, 0.0)
    A0 = tl.where(nrm, ia, tl.where(add, 1.0, cr + 1.0 - ca))
    A1 = tl.where(nrm, ia, tl.where(add, 1.0, cg + 1.0 - ca))
    A2 = tl.where(nrm, ia, tl.where(add, 1.0, cb + 1.0 - ca))
    A3 = tl.where(nrm, ia, 1.0)
    return B0, B1, B2, B3, A0, A1, A2, A3, bl


@triton.jit
def _blend_step(D, Bk, Ak, QUANT: tl.constexpr):
    v = Bk + Ak * D
    v = tl.minimum(tl.maximum(v, 0.0), 1.0)
    if QUANT:
        v = tl.floor(v * 255.0 + 0.5) / 255.0
    return v


@triton.jit
def _composite_kernel(tidp_ptr, D_ptr, NP, L, p_lo, p_hi,
                      Db_ptr, X_ptr, A_ptr, LA,
                      W, HW, V, n, NLEV, lod_bias, R, MHW,
                      S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr, cull_ptr,
                      op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr, Md_ptr, NCTX,
                      HAS_MASK: tl.constexpr, CULL: tl.constexpr, QUANT: tl.constexpr, SAVE: tl.constexpr, BLOCK: tl.constexpr):
    """从后往前合成像素 [p_lo, p_hi)。SAVE：记录 D_below[k]（本层混合前的值，全部层），以及前 LA 层的 X = B_k（乘算为 0）与 A_k。"""
    p = p_lo + tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = p < p_hi
    nloc = p_hi - p_lo
    lp = p - p_lo
    D0 = tl.zeros([BLOCK], tl.float32)
    D1 = tl.zeros([BLOCK], tl.float32)
    D2 = tl.zeros([BLOCK], tl.float32)
    D3 = tl.zeros([BLOCK], tl.float32)
    for kk in range(0, L):
        k = L - 1 - kk
        tid = tl.load(tidp_ptr + k * NP + p, m, other=0).to(tl.int64)
        act = m & (tid > 0)
        if SAVE:
            if k < LA:
                # 边缘梯度项要用到本层片元以外的像素（antialias 会改动边缘两侧）：D_below、X、A 全部像素都写
                o = (k * nloc + lp) * 4
                tl.store(Db_ptr + o, D0, m)
                tl.store(Db_ptr + o + 1, D1, m)
                tl.store(Db_ptr + o + 2, D2, m)
                tl.store(Db_ptr + o + 3, D3, m)
                z = tl.zeros([BLOCK], tl.float32)
                tl.store(X_ptr + o, z, m)
                tl.store(X_ptr + o + 1, z, m)
                tl.store(X_ptr + o + 2, z, m)
                tl.store(X_ptr + o + 3, z, m)
                tl.store(A_ptr + o, z + 1.0, m)
                tl.store(A_ptr + o + 1, z + 1.0, m)
                tl.store(A_ptr + o + 2, z + 1.0, m)
                tl.store(A_ptr + o + 3, z + 1.0, m)
        if tl.sum(act.to(tl.int32)) > 0:
            B0, B1, B2, B3, A0, A1, A2, A3, bl = _shade(p, tid, act, W, HW, V, n, NLEV, lod_bias, R, MHW,
                                                        S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr,
                                                        cull_ptr, op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr,
                                                        Md_ptr, NCTX, HAS_MASK, CULL)
            if SAVE:
                o = (k * nloc + lp) * 4
                if k >= LA:
                    tl.store(Db_ptr + o, D0, act)
                    tl.store(Db_ptr + o + 1, D1, act)
                    tl.store(Db_ptr + o + 2, D2, act)
                    tl.store(Db_ptr + o + 3, D3, act)
                if k < LA:
                    xm = act & (bl != 2)
                    tl.store(X_ptr + o, B0, xm)
                    tl.store(X_ptr + o + 1, B1, xm)
                    tl.store(X_ptr + o + 2, B2, xm)
                    tl.store(X_ptr + o + 3, B3, xm)
                    tl.store(A_ptr + o, A0, act)
                    tl.store(A_ptr + o + 1, A1, act)
                    tl.store(A_ptr + o + 2, A2, act)
                    tl.store(A_ptr + o + 3, A3, act)
            D0 = tl.where(act, _blend_step(D0, B0, A0, QUANT), D0)
            D1 = tl.where(act, _blend_step(D1, B1, A1, QUANT), D1)
            D2 = tl.where(act, _blend_step(D2, B2, A2, QUANT), D2)
            D3 = tl.where(act, _blend_step(D3, B3, A3, QUANT), D3)
    if not SAVE:
        tl.store(D_ptr + p * 4, D0, m)
        tl.store(D_ptr + p * 4 + 1, D1, m)
        tl.store(D_ptr + p * 4 + 2, D2, m)
        tl.store(D_ptr + p * 4 + 3, D3, m)


# =============================================================================== 反向（从前往后）

@triton.jit
def _shade_bwd(p, tid, act, W, HW, V, n, NLEV, lod_bias, R, MHW,
               S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr, cull_ptr,
               op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr, Md_ptr, NCTX,
               dB0, dB1, dB2, dB3, dA0, dA1, dA2, dA3,
               gS_ptr, gU_ptr, gop_ptr, gmul_ptr, gscr_ptr,
               HAS_MASK: tl.constexpr, CULL: tl.constexpr, NEED_S: tl.constexpr, NEED_U: tl.constexpr, NEED_OP: tl.constexpr,
               NEED_COL: tl.constexpr):
    """_shade 的反向：给定 ∂L/∂B_k、∂L/∂A_k，原子加到 ∂S（屏幕坐标）、∂U（模型单位，经 mask 采样位置）、不透明度、乘算 / 屏幕色。"""
    (b, px, py, v0, v1, v2, d, g0, g1, g2, p0x, p0y, p1x, p1y, p2x, p2y,
     U0x, U0y, U1x, U1y, U2x, U2y) = _frag_geom(p, tid, act, W, HW, V, S_ptr, tri_ptr, tdraw_ptr, uv_ptr)
    w0, w1, w2, d0, d1, d2, d3, small, den, e1x, e1y, e2x, e2y, qx, qy = _bary(p0x, p0y, p1x, p1y, p2x, p2y, px, py)
    uvx, uvy, da0, da1, da2, da3 = _uv_and_da(U0x, U0y, U1x, U1y, U2x, U2y, w0, w1, w2, d0, d1, d2, d3)
    tex = tl.load(texi_ptr + d, act, other=0).to(tl.int64)
    l0, l1, fr = _lod(tab_ptr, maxlev_ptr, tex, da0, da1, da2, da3, act, NLEV, lod_bias)
    s = uvx
    tt = 1.0 - uvy
    tr, tg, tb, ta = _sample_rgba(data_ptr, tab_ptr, tex, l0, l1, fr, s, tt, act, NLEV)
    bd = b * n + d
    mr = tl.load(mul_ptr + bd * 4, act, other=1.0)
    mg = tl.load(mul_ptr + bd * 4 + 1, act, other=1.0)
    mbb = tl.load(mul_ptr + bd * 4 + 2, act, other=1.0)
    sr = tl.load(scr_ptr + bd * 4, act, other=0.0)
    sg = tl.load(scr_ptr + bd * 4 + 1, act, other=0.0)
    sb = tl.load(scr_ptr + bd * 4 + 2, act, other=0.0)
    xr1 = tr * mr
    xg1 = tg * mg
    xb1 = tb * mbb
    xr2 = xr1 + sr * ta - xr1 * sr
    xg2 = xg1 + sg * ta - xg1 * sg
    xb2 = xb1 + sb * ta - xb1 * sb
    op = tl.load(op_ptr + bd, act, other=0.0)
    f = tl.zeros_like(op) + 1.0
    if HAS_MASK:
        c = tl.load(dctx_ptr + d, act, other=-1).to(tl.int64)
        masked = act & (c >= 0)
        cc = tl.maximum(c, 0)
        u0x = tl.load(U_ptr + g0 * 2, masked, other=0.0)
        u0y = tl.load(U_ptr + g0 * 2 + 1, masked, other=0.0)
        u1x = tl.load(U_ptr + g1 * 2, masked, other=0.0)
        u1y = tl.load(U_ptr + g1 * 2 + 1, masked, other=0.0)
        u2x = tl.load(U_ptr + g2 * 2, masked, other=0.0)
        u2y = tl.load(U_ptr + g2 * 2 + 1, masked, other=0.0)
        pux = u0x * w0 + u1x * w1 + u2x * w2
        puy = u0y * w0 + u1y * w1 + u2y * w2
        mo = (b * NCTX + cc) * 6
        m00 = tl.load(Md_ptr + mo, masked, other=0.0)
        m01 = tl.load(Md_ptr + mo + 1, masked, other=0.0)
        m02 = tl.load(Md_ptr + mo + 2, masked, other=0.0)
        m10 = tl.load(Md_ptr + mo + 3, masked, other=0.0)
        m11 = tl.load(Md_ptr + mo + 4, masked, other=0.0)
        m12 = tl.load(Md_ptr + mo + 5, masked, other=0.0)
        cx = (m00 * pux + m01 * puy) + m02
        cy = (m10 * pux + m11 * puy) + m12
        mv, dvx, dvy = _mask_sample(mflat_ptr, mtab_ptr, b, cc, cx, cy, masked, R, MHW)
        inv = tl.load(inv_ptr + d, masked, other=0) != 0
        mv = tl.where(inv, 1.0 - mv, mv)
        f = tl.where(masked, mv, 1.0)
    alive = act
    if CULL:
        q0x = tl.load(P_ptr + g0 * 2, act, other=0.0)
        q0y = tl.load(P_ptr + g0 * 2 + 1, act, other=0.0)
        q1x = tl.load(P_ptr + g1 * 2, act, other=0.0)
        q1y = tl.load(P_ptr + g1 * 2 + 1, act, other=0.0)
        q2x = tl.load(P_ptr + g2 * 2, act, other=0.0)
        q2y = tl.load(P_ptr + g2 * 2 + 1, act, other=0.0)
        ar = (q1x - q0x) * (q2y - q0y) - (q1y - q0y) * (q2x - q0x)
        alive = act & ~((tl.load(cull_ptr + d, act, other=0) != 0) & (ar > 0))
    # ---- 混合系数 → col
    bl = tl.load(blend_ptr + d, act, other=0)
    nrm = bl == 0
    add = bl == 1
    mul_ = bl == 2
    dcr = tl.where(nrm | add, dB0, tl.where(mul_, dA0, 0.0))
    dcg = tl.where(nrm | add, dB1, tl.where(mul_, dA1, 0.0))
    dcb = tl.where(nrm | add, dB2, tl.where(mul_, dA2, 0.0))
    dca = tl.where(nrm, dB3 - (dA0 + dA1 + dA2 + dA3), tl.where(mul_, -(dA0 + dA1 + dA2), 0.0))
    dcr = tl.where(alive, dcr, 0.0)
    dcg = tl.where(alive, dcg, 0.0)
    dcb = tl.where(alive, dcb, 0.0)
    dca = tl.where(alive, dca, 0.0)
    # col = [rgb2, ta]·op·f
    of = op * f
    dr2 = dcr * of
    dg2 = dcg * of
    db2 = dcb * of
    dta = dca * of
    if NEED_OP:
        dop = (dcr * xr2 + dcg * xg2 + dcb * xb2 + dca * ta) * f
        tl.atomic_add(gop_ptr + bd, dop, alive)
    # rgb2 = rgb1 + scr·ta − rgb1·scr
    dr1 = dr2 * (1.0 - sr)
    dg1 = dg2 * (1.0 - sg)
    db1 = db2 * (1.0 - sb)
    dta = dta + dr2 * sr + dg2 * sg + db2 * sb
    if NEED_COL:
        tl.atomic_add(gscr_ptr + bd * 4, dr2 * (ta - xr1), alive)
        tl.atomic_add(gscr_ptr + bd * 4 + 1, dg2 * (ta - xg1), alive)
        tl.atomic_add(gscr_ptr + bd * 4 + 2, db2 * (ta - xb1), alive)
        tl.atomic_add(gmul_ptr + bd * 4, dr1 * tr, alive)
        tl.atomic_add(gmul_ptr + bd * 4 + 1, dg1 * tg, alive)
        tl.atomic_add(gmul_ptr + bd * 4 + 2, db1 * tb, alive)
    if NEED_S:
        # 贴图采样 → uv → 重心坐标
        ds, dtt = _sample_rgba_grad(data_ptr, tab_ptr, tex, l0, l1, fr, s, tt, alive, NLEV, dr1 * mr, dg1 * mg, db1 * mbb, dta)
        duvx = ds
        duvy = -dtt
        dw0 = duvx * U0x + duvy * U0y
        dw1 = duvx * U1x + duvy * U1y
        dw2 = duvx * U2x + duvy * U2y
        if HAS_MASK:
            # col·f，f = mv 或 1 − mv；mv = 1 − bilinear(x, y)，x = cx·R − …
            dfm = (dcr * xr2 + dcg * xg2 + dcb * xb2 + dca * ta) * op
            dmv = tl.where(inv, -dfm, dfm)
            dcx = -dmv * dvx * R
            dcy = -dmv * dvy * R
            dcx = tl.where(masked & alive, dcx, 0.0)
            dcy = tl.where(masked & alive, dcy, 0.0)
            dpux = m00 * dcx + m10 * dcy
            dpuy = m01 * dcx + m11 * dcy
            dw0 = dw0 + dpux * u0x + dpuy * u0y
            dw1 = dw1 + dpux * u1x + dpuy * u1y
            dw2 = dw2 + dpux * u2x + dpuy * u2y
            if NEED_U:
                mk = masked & alive
                tl.atomic_add(gU_ptr + g0 * 2, dpux * w0, mk)
                tl.atomic_add(gU_ptr + g0 * 2 + 1, dpuy * w0, mk)
                tl.atomic_add(gU_ptr + g1 * 2, dpux * w1, mk)
                tl.atomic_add(gU_ptr + g1 * 2 + 1, dpuy * w1, mk)
                tl.atomic_add(gU_ptr + g2 * 2, dpux * w2, mk)
                tl.atomic_add(gU_ptr + g2 * 2 + 1, dpuy * w2, mk)
        # w0 = 1 − w1 − w2
        a1 = dw1 - dw0
        a2 = dw2 - dw0
        dN1 = a1 / den
        dN2 = a2 / den
        dden = tl.where(small, 0.0, -(a1 * w1 + a2 * w2) / den)
        dqx = dN1 * e2y - dN2 * e1y
        dqy = -dN1 * e2x + dN2 * e1x
        de1x = dN2 * qy + dden * e2y
        de1y = -dN2 * qx - dden * e2x
        de2x = -dN1 * qy - dden * e1y
        de2y = dN1 * qx + dden * e1x
        tl.atomic_add(gS_ptr + g0 * 2, -dqx - de1x - de2x, alive)
        tl.atomic_add(gS_ptr + g0 * 2 + 1, -dqy - de1y - de2y, alive)
        tl.atomic_add(gS_ptr + g1 * 2, de1x, alive)
        tl.atomic_add(gS_ptr + g1 * 2 + 1, de1y, alive)
        tl.atomic_add(gS_ptr + g2 * 2, de2x, alive)
        tl.atomic_add(gS_ptr + g2 * 2 + 1, de2y, alive)


@triton.jit
def _backward_kernel(tidp_ptr, G_ptr, NP, L, p_lo, p_hi, Db_ptr, dX_ptr, LA,
                     W, HW, V, n, NLEV, lod_bias, R, MHW,
                     S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr, cull_ptr,
                     op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr, Md_ptr, NCTX,
                     gS_ptr, gU_ptr, gop_ptr, gmul_ptr, gscr_ptr,
                     HAS_MASK: tl.constexpr, CULL: tl.constexpr, HAS_DX: tl.constexpr, NEED_S: tl.constexpr, NEED_U: tl.constexpr,
                     NEED_OP: tl.constexpr, NEED_COL: tl.constexpr, BLOCK: tl.constexpr):
    p = p_lo + tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = p < p_hi
    nloc = p_hi - p_lo
    lp = p - p_lo
    g0 = tl.load(G_ptr + p * 4, m, other=0.0)
    g1 = tl.load(G_ptr + p * 4 + 1, m, other=0.0)
    g2 = tl.load(G_ptr + p * 4 + 2, m, other=0.0)
    g3 = tl.load(G_ptr + p * 4 + 3, m, other=0.0)
    for k in range(0, L):
        tid = tl.load(tidp_ptr + k * NP + p, m, other=0).to(tl.int64)
        act = m & (tid > 0)
        if tl.sum(act.to(tl.int32)) > 0:
            B0, B1, B2, B3, A0, A1, A2, A3, bl = _shade(p, tid, act, W, HW, V, n, NLEV, lod_bias, R, MHW,
                                                        S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr,
                                                        cull_ptr, op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr,
                                                        Md_ptr, NCTX, HAS_MASK, CULL)
            o = (k * nloc + lp) * 4
            e0 = tl.load(Db_ptr + o, act, other=0.0)
            e1 = tl.load(Db_ptr + o + 1, act, other=0.0)
            e2 = tl.load(Db_ptr + o + 2, act, other=0.0)
            e3 = tl.load(Db_ptr + o + 3, act, other=0.0)
            # 夹取：torch.clamp(0, 1) 的反向只在 0 < v < 1 处通过（端点处为 0，实测）；8 位取整为直通
            v = B0 + A0 * e0
            h0 = tl.where((v > 0) & (v < 1), g0, 0.0)
            v = B1 + A1 * e1
            h1 = tl.where((v > 0) & (v < 1), g1, 0.0)
            v = B2 + A2 * e2
            h2 = tl.where((v > 0) & (v < 1), g2, 0.0)
            v = B3 + A3 * e3
            h3 = tl.where((v > 0) & (v < 1), g3, 0.0)
            dB0 = h0
            dB1 = h1
            dB2 = h2
            dB3 = h3
            if HAS_DX:
                if k < LA:
                    xm = act & (bl != 2)
                    dB0 = dB0 + tl.load(dX_ptr + o, xm, other=0.0)
                    dB1 = dB1 + tl.load(dX_ptr + o + 1, xm, other=0.0)
                    dB2 = dB2 + tl.load(dX_ptr + o + 2, xm, other=0.0)
                    dB3 = dB3 + tl.load(dX_ptr + o + 3, xm, other=0.0)
            _shade_bwd(p, tid, act, W, HW, V, n, NLEV, lod_bias, R, MHW,
                       S_ptr, P_ptr, U_ptr, tri_ptr, tdraw_ptr, uv_ptr, texi_ptr, blend_ptr, dctx_ptr, inv_ptr, cull_ptr,
                       op_ptr, mul_ptr, scr_ptr, data_ptr, tab_ptr, maxlev_ptr, mflat_ptr, mtab_ptr, Md_ptr, NCTX,
                       dB0, dB1, dB2, dB3, h0 * e0, h1 * e1, h2 * e2, h3 * e3,
                       gS_ptr, gU_ptr, gop_ptr, gmul_ptr, gscr_ptr, HAS_MASK, CULL, NEED_S, NEED_U, NEED_OP, NEED_COL)
            g0 = tl.where(act, h0 * A0, g0)
            g1 = tl.where(act, h1 * A1, g1)
            g2 = tl.where(act, h2 * A2, g2)
            g3 = tl.where(act, h3 * A3, g3)


# =============================================================================== Python 侧启动

def _common(st):
    """st：raster._FastState。返回着色核共用的位置参数。"""
    g, a = st.gt, st.at
    return (st.W, st.HW, st.V, st.n, a["nlev"], a["bias"], st.R, st.MHW,
            st.S, st.P, st.U, g["tri"], g["tdraw"], g["uv"], g["texi"], g["blend"], g["dctx"], g["inv"], g["cull"],
            st.op, st.mul, st.scr, a["data"], a["tab"], a["maxlev"], st.mflat, st.mtab, st.Md, st.nctx)


def composite_forward(st, quantize: bool) -> torch.Tensor:
    NP = st.NP
    D = torch.empty((NP, 4), device=st.S.device, dtype=torch.float32)
    d = D
    _composite_kernel[(triton.cdiv(NP, BLOCK),)](st.tidp, D, NP, st.L, 0, NP, d, d, d, 0, *_common(st),
                                                 HAS_MASK=st.has_mask, CULL=st.cull, QUANT=quantize, SAVE=False, BLOCK=BLOCK, **_OPTS)
    return D


def composite_save(st, quantize: bool, p_lo: int, p_hi: int, LA: int):
    """反向第一遍：返回 D_below [L, n, 4]，以及 LA > 0 时的 X、A [LA, n, 4]（n = p_hi − p_lo）。"""
    nl = p_hi - p_lo
    dev = st.S.device
    Db = torch.empty((st.L, nl, 4), device=dev, dtype=torch.float32)
    X = torch.empty((LA, nl, 4), device=dev, dtype=torch.float32) if LA else Db
    A = torch.empty((LA, nl, 4), device=dev, dtype=torch.float32) if LA else Db
    _composite_kernel[(triton.cdiv(nl, BLOCK),)](st.tidp, Db, st.NP, st.L, p_lo, p_hi, Db, X, A, LA, *_common(st),
                                                 HAS_MASK=st.has_mask, CULL=st.cull, QUANT=quantize, SAVE=True, BLOCK=BLOCK, **_OPTS)
    return Db, (X if LA else None), (A if LA else None)


def composite_backward(st, G, p_lo, p_hi, Db, dX, LA, grads, needs):
    nl = p_hi - p_lo
    gS, gU, gop, gmul, gscr = grads
    d = Db
    _backward_kernel[(triton.cdiv(nl, BLOCK),)](st.tidp, G, st.NP, st.L, p_lo, p_hi, Db, dX if dX is not None else d, LA, *_common(st),
                                                gS if gS is not None else d, gU if gU is not None else d, gop if gop is not None else d,
                                                gmul if gmul is not None else d, gscr if gscr is not None else d,
                                                HAS_MASK=st.has_mask, CULL=st.cull, HAS_DX=dX is not None,
                                                NEED_S=needs[0], NEED_U=needs[1] and st.has_mask, NEED_OP=needs[2], NEED_COL=needs[3],
                                                BLOCK=BLOCK, **_OPTS)
