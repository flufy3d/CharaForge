"""A7：变形器（deform.py）的 Triton 融合核：warp 变换、rotation 变换的前向与反向。

与 deform._warp_core / _RotApply 的 PyTorch 实现逐式相同（同样的运算顺序，关闭 FMA 合并），
一次内核代替原来每段约 100（前向）/ 200（反向）个小算子。反向的散射累加用原子加（与 scatter_add_ 一样不保证求和顺序）。
支持 float32 / float64。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 256
_OPTS = dict(num_warps=4, enable_fp_fusion=False)


@triton.jit
def _axis(s, n_i, LO: tl.constexpr, HI: tl.constexpr):
    """deform._warp_core 里的 axis()：返回 a0, a1, g0, g1, i0, i1, t, k, inside。"""
    n = n_i.to(s.dtype)
    inside = (s >= 0) & (s <= 1)
    lo = s < 0
    x = s * n
    # torch：min(clamp(floor(x).long(), 0), n − 1)；先在浮点里夹取，避免超大值转整数溢出（结果相同）
    xf = tl.minimum(tl.maximum(x, 0.0), (n_i - 1).to(s.dtype))
    i = tl.floor(xf).to(tl.int64)
    fi = i.to(s.dtype)
    band = 1.0 / (0.0 - LO)
    hib = 1.0 / (HI - 1.0)
    t = tl.where(inside, x - fi, tl.where(lo, (s - LO) * band, (s - 1.0) * hib))
    a0 = tl.where(inside, libdevice.div_rn(fi, n), tl.where(lo, LO, 1.0))  # IEEE 除法（Triton 的 / 是近似除法）
    a1 = tl.where(inside, libdevice.div_rn(fi + 1.0, n), tl.where(lo, 0.0, HI))
    i0 = tl.where(inside, i, tl.where(lo, 0, n_i))
    i1 = tl.where(inside, i + 1, i0)
    k = tl.where(inside, n, band)
    return a0.to(s.dtype), a1.to(s.dtype), inside | (~lo), inside | lo, i0, i1, t, k.to(s.dtype), inside


@triton.jit
def _warp_geom(u, v, Ci, Ri, goff, quad, LO: tl.constexpr, HI: tl.constexpr):
    ua0, ua1, ug0, ug1, ui0, ui1, tx, kx, inu = _axis(u, Ci, LO, HI)
    va0, va1, vg0, vg1, vi0, vi1, ty, ky, inv = _axis(v, Ri, LO, HI)
    stride = Ci + 1
    on00 = ug0 & vg0
    on10 = ug1 & vg0
    on01 = ug0 & vg1
    on11 = ug1 & vg1
    gi00 = tl.where(on00, goff + vi0 * stride + ui0, 0)
    gi10 = tl.where(on10, goff + vi0 * stride + ui1, 0)
    gi01 = tl.where(on01, goff + vi1 * stride + ui0, 0)
    gi11 = tl.where(on11, goff + vi1 * stride + ui1, 0)
    tri = ~(inu & inv & quad)
    upper = tx + ty > 1
    zero = tl.zeros_like(tx)
    one = zero + 1.0
    far = ~((u > LO) & (u < HI) & (v > LO) & (v < HI))
    w00 = tl.where(tri, tl.where(upper, zero, 1 - tx - ty), (1 - tx) * (1 - ty))
    w10 = tl.where(tri, tl.where(upper, 1 - ty, tx), tx * (1 - ty))
    w01 = tl.where(tri, tl.where(upper, 1 - tx, ty), (1 - tx) * ty)
    w11 = tl.where(tri, tl.where(upper, tx + ty - 1, zero), tx * ty)
    w00 = tl.where(far, zero, w00)
    w10 = tl.where(far, zero, w10)
    w01 = tl.where(far, zero, w01)
    w11 = tl.where(far, zero, w11)
    return (ua0, ua1, va0, va1, on00, on10, on01, on11, gi00, gi10, gi01, gi11, w00, w10, w01, w11,
            tri, upper, far, tx, ty, kx, ky, zero, one)


@triton.jit
def _warp_fwd_kernel(p_ptr, grid_ptr, c_ptr, du_ptr, dv_ptr, pid_ptr, C_ptr, R_ptr, goff_ptr, quad_ptr, q_ptr,
                     total, N, G, NW, LO: tl.constexpr, HI: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    b = offs // N
    n = offs % N
    u = tl.load(p_ptr + offs * 2, m, other=0.0)
    v = tl.load(p_ptr + offs * 2 + 1, m, other=0.0)
    pid = tl.load(pid_ptr + n, m, other=0)
    Ci = tl.load(C_ptr + n, m, other=1)
    Ri = tl.load(R_ptr + n, m, other=1)
    goff = tl.load(goff_ptr + n, m, other=0)
    quad = tl.load(quad_ptr + n, m, other=0) != 0
    (ua0, ua1, va0, va1, on00, on10, on01, on11, gi00, gi10, gi01, gi11, w00, w10, w01, w11,
     tri, upper, far, tx, ty, kx, ky, zero, one) = _warp_geom(u, v, Ci, Ri, goff, quad, LO, HI)
    sw = tl.where(far, one, zero)
    su = tl.where(far, u - 0.5, zero)
    sv = tl.where(far, v - 0.5, zero)
    gb = grid_ptr + b * (G * 2)
    # 角点 00
    off = tl.where(on00, zero, w00)
    sw = sw + off
    su = su + off * (ua0 - 0.5)
    sv = sv + off * (va0 - 0.5)
    ww = tl.where(on00, w00, zero)
    qx = tl.load(gb + gi00 * 2, m, other=0.0) * ww
    qy = tl.load(gb + gi00 * 2 + 1, m, other=0.0) * ww
    # 角点 10
    off = tl.where(on10, zero, w10)
    sw = sw + off
    su = su + off * (ua1 - 0.5)
    sv = sv + off * (va0 - 0.5)
    ww = tl.where(on10, w10, zero)
    qx = qx + tl.load(gb + gi10 * 2, m, other=0.0) * ww
    qy = qy + tl.load(gb + gi10 * 2 + 1, m, other=0.0) * ww
    # 角点 01
    off = tl.where(on01, zero, w01)
    sw = sw + off
    su = su + off * (ua0 - 0.5)
    sv = sv + off * (va1 - 0.5)
    ww = tl.where(on01, w01, zero)
    qx = qx + tl.load(gb + gi01 * 2, m, other=0.0) * ww
    qy = qy + tl.load(gb + gi01 * 2 + 1, m, other=0.0) * ww
    # 角点 11
    off = tl.where(on11, zero, w11)
    sw = sw + off
    su = su + off * (ua1 - 0.5)
    sv = sv + off * (va1 - 0.5)
    ww = tl.where(on11, w11, zero)
    qx = qx + tl.load(gb + gi11 * 2, m, other=0.0) * ww
    qy = qy + tl.load(gb + gi11 * 2 + 1, m, other=0.0) * ww
    a = (b * NW + pid) * 2
    qx = qx + sw * tl.load(c_ptr + a, m, other=0.0) + su * tl.load(du_ptr + a, m, other=0.0) + sv * tl.load(dv_ptr + a, m, other=0.0)
    qy = qy + sw * tl.load(c_ptr + a + 1, m, other=0.0) + su * tl.load(du_ptr + a + 1, m, other=0.0) + sv * tl.load(dv_ptr + a + 1, m, other=0.0)
    tl.store(q_ptr + offs * 2, qx, m)
    tl.store(q_ptr + offs * 2 + 1, qy, m)


@triton.jit
def _corner_P(gb, gi, on, ua, va, cx, cy, dux, duy, dvx, dvy, m):
    gx = tl.load(gb + gi * 2, m & on, other=0.0)
    gy = tl.load(gb + gi * 2 + 1, m & on, other=0.0)
    px = tl.where(on, gx, cx + (ua - 0.5) * dux + (va - 0.5) * dvx)
    py = tl.where(on, gy, cy + (ua - 0.5) * duy + (va - 0.5) * dvy)
    return px, py


@triton.jit
def _warp_bwd_kernel(p_ptr, grid_ptr, c_ptr, du_ptr, dv_ptr, pid_ptr, C_ptr, R_ptr, goff_ptr, quad_ptr, gq_ptr,
                     gp_ptr, ggrid_ptr, gc_ptr, gdu_ptr, gdv_ptr,
                     total, N, G, NW, LO: tl.constexpr, HI: tl.constexpr,
                     NEED_P: tl.constexpr, NEED_GRID: tl.constexpr, NEED_AFF: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    b = offs // N
    n = offs % N
    u = tl.load(p_ptr + offs * 2, m, other=0.0)
    v = tl.load(p_ptr + offs * 2 + 1, m, other=0.0)
    gx = tl.load(gq_ptr + offs * 2, m, other=0.0)
    gy = tl.load(gq_ptr + offs * 2 + 1, m, other=0.0)
    pid = tl.load(pid_ptr + n, m, other=0)
    Ci = tl.load(C_ptr + n, m, other=1)
    Ri = tl.load(R_ptr + n, m, other=1)
    goff = tl.load(goff_ptr + n, m, other=0)
    quad = tl.load(quad_ptr + n, m, other=0) != 0
    (ua0, ua1, va0, va1, on00, on10, on01, on11, gi00, gi10, gi01, gi11, w00, w10, w01, w11,
     tri, upper, far, tx, ty, kx, ky, zero, one) = _warp_geom(u, v, Ci, Ri, goff, quad, LO, HI)
    a = (b * NW + pid) * 2
    if NEED_AFF:
        sw = tl.where(far, one, zero)
        su = tl.where(far, u - 0.5, zero)
        sv = tl.where(far, v - 0.5, zero)
        off = tl.where(on00, zero, w00)
        sw = sw + off
        su = su + off * (ua0 - 0.5)
        sv = sv + off * (va0 - 0.5)
        off = tl.where(on10, zero, w10)
        sw = sw + off
        su = su + off * (ua1 - 0.5)
        sv = sv + off * (va0 - 0.5)
        off = tl.where(on01, zero, w01)
        sw = sw + off
        su = su + off * (ua0 - 0.5)
        sv = sv + off * (va1 - 0.5)
        off = tl.where(on11, zero, w11)
        sw = sw + off
        su = su + off * (ua1 - 0.5)
        sv = sv + off * (va1 - 0.5)
        tl.atomic_add(gc_ptr + a, gx * sw, m)
        tl.atomic_add(gc_ptr + a + 1, gy * sw, m)
        tl.atomic_add(gdu_ptr + a, gx * su, m)
        tl.atomic_add(gdu_ptr + a + 1, gy * su, m)
        tl.atomic_add(gdv_ptr + a, gx * sv, m)
        tl.atomic_add(gdv_ptr + a + 1, gy * sv, m)
    gb = grid_ptr + b * (G * 2)
    if NEED_GRID:
        ggb = ggrid_ptr + b * (G * 2)
        ww = tl.where(on00, w00, zero)
        tl.atomic_add(ggb + gi00 * 2, gx * ww, m & on00)
        tl.atomic_add(ggb + gi00 * 2 + 1, gy * ww, m & on00)
        ww = tl.where(on10, w10, zero)
        tl.atomic_add(ggb + gi10 * 2, gx * ww, m & on10)
        tl.atomic_add(ggb + gi10 * 2 + 1, gy * ww, m & on10)
        ww = tl.where(on01, w01, zero)
        tl.atomic_add(ggb + gi01 * 2, gx * ww, m & on01)
        tl.atomic_add(ggb + gi01 * 2 + 1, gy * ww, m & on01)
        ww = tl.where(on11, w11, zero)
        tl.atomic_add(ggb + gi11 * 2, gx * ww, m & on11)
        tl.atomic_add(ggb + gi11 * 2 + 1, gy * ww, m & on11)
    if NEED_P:
        cx = tl.load(c_ptr + a, m, other=0.0)
        cy = tl.load(c_ptr + a + 1, m, other=0.0)
        dux = tl.load(du_ptr + a, m, other=0.0)
        duy = tl.load(du_ptr + a + 1, m, other=0.0)
        dvx = tl.load(dv_ptr + a, m, other=0.0)
        dvy = tl.load(dv_ptr + a + 1, m, other=0.0)
        # ∂w/∂tx、∂w/∂ty
        dwx00 = tl.where(tri, tl.where(upper, zero, -one), ty - 1)
        dwx10 = tl.where(tri, tl.where(upper, zero, one), 1 - ty)
        dwx01 = tl.where(tri, tl.where(upper, -one, zero), -ty)
        dwx11 = tl.where(tri, tl.where(upper, one, zero), ty)
        dwy00 = tl.where(tri, tl.where(upper, zero, -one), tx - 1)
        dwy10 = tl.where(tri, tl.where(upper, -one, zero), -tx)
        dwy01 = tl.where(tri, tl.where(upper, zero, one), 1 - tx)
        dwy11 = tl.where(tri, tl.where(upper, one, zero), tx)
        Px, Py = _corner_P(gb, gi00, on00, ua0, va0, cx, cy, dux, duy, dvx, dvy, m)
        dqxx = Px * dwx00
        dqxy = Py * dwx00
        dqyx = Px * dwy00
        dqyy = Py * dwy00
        Px, Py = _corner_P(gb, gi10, on10, ua1, va0, cx, cy, dux, duy, dvx, dvy, m)
        dqxx = dqxx + Px * dwx10
        dqxy = dqxy + Py * dwx10
        dqyx = dqyx + Px * dwy10
        dqyy = dqyy + Py * dwy10
        Px, Py = _corner_P(gb, gi01, on01, ua0, va1, cx, cy, dux, duy, dvx, dvy, m)
        dqxx = dqxx + Px * dwx01
        dqxy = dqxy + Py * dwx01
        dqyx = dqyx + Px * dwy01
        dqyy = dqyy + Py * dwy01
        Px, Py = _corner_P(gb, gi11, on11, ua1, va1, cx, cy, dux, duy, dvx, dvy, m)
        dqxx = dqxx + Px * dwx11
        dqxy = dqxy + Py * dwx11
        dqyx = dqyx + Px * dwy11
        dqyy = dqyy + Py * dwy11
        gpx = tl.where(far, gx * dux + gy * duy, (gx * dqxx + gy * dqxy) * kx)
        gpy = tl.where(far, gx * dvx + gy * dvy, (gx * dqyx + gy * dqyy) * ky)
        tl.store(gp_ptr + offs * 2, gpx, m)
        tl.store(gp_ptr + offs * 2 + 1, gpy, m)


def warp_forward(p, grid, c, du, dv, pid, C_i, R_i, goff, quad, lo, hi):
    B, N = p.shape[0], p.shape[1]
    p = p.contiguous()
    q = torch.empty_like(p)
    total = B * N
    if total == 0:
        return q
    _warp_fwd_kernel[(triton.cdiv(total, BLOCK),)](p, grid.contiguous(), c.contiguous(), du.contiguous(), dv.contiguous(), pid, C_i, R_i, goff,
                                                    quad.view(torch.uint8), q, total, N, grid.shape[1], c.shape[1], LO=lo, HI=hi, BLOCK=BLOCK, **_OPTS)
    return q


def warp_backward(gq, p, grid, c, du, dv, pid, C_i, R_i, goff, quad, lo, hi, need_p, need_grid, need_aff):
    B, N = p.shape[0], p.shape[1]
    p = p.contiguous()
    gp = torch.empty_like(p) if need_p else None
    ggrid = torch.zeros_like(grid) if need_grid else None
    gc, gdu, gdv = (torch.zeros_like(c), torch.zeros_like(du), torch.zeros_like(dv)) if need_aff else (None, None, None)
    total = B * N
    if total == 0:
        return (torch.zeros_like(p) if need_p else None), ggrid, gc, gdu, gdv
    dummy = p
    _warp_bwd_kernel[(triton.cdiv(total, BLOCK),)](p, grid.contiguous(), c.contiguous(), du.contiguous(), dv.contiguous(), pid, C_i, R_i, goff,
                                                    quad.view(torch.uint8), gq.contiguous(),
                                                    gp if need_p else dummy, ggrid if need_grid else dummy,
                                                    gc if need_aff else dummy, gdu if need_aff else dummy, gdv if need_aff else dummy,
                                                    total, N, grid.shape[1], c.shape[1], LO=lo, HI=hi,
                                                    NEED_P=need_p, NEED_GRID=need_grid, NEED_AFF=need_aff, BLOCK=BLOCK, **_OPTS)
    return gp, ggrid, gc, gdu, gdv


@triton.jit
def _rot_fwd_kernel(p_ptr, o_ptr, cs_ptr, sn_ptr, sc_ptr, rx_ptr, ry_ptr, pid_ptr, q_ptr, total, N, NR, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    b = offs // N
    n = offs % N
    pid = tl.load(pid_ptr + n, m, other=0)
    j = b * NR + pid
    x = tl.load(p_ptr + offs * 2, m, other=0.0) * tl.load(rx_ptr + j, m, other=0.0)
    y = tl.load(p_ptr + offs * 2 + 1, m, other=0.0) * tl.load(ry_ptr + j, m, other=0.0)
    c = tl.load(cs_ptr + j, m, other=0.0)
    s = tl.load(sn_ptr + j, m, other=0.0)
    k = tl.load(sc_ptr + j, m, other=0.0)
    tl.store(q_ptr + offs * 2, tl.load(o_ptr + j * 2, m, other=0.0) + k * (c * x - s * y), m)
    tl.store(q_ptr + offs * 2 + 1, tl.load(o_ptr + j * 2 + 1, m, other=0.0) + k * (s * x + c * y), m)


@triton.jit
def _rot_bwd_kernel(p_ptr, cs_ptr, sn_ptr, sc_ptr, rx_ptr, ry_ptr, pid_ptr, g_ptr, gp_ptr, go_ptr, gcs_ptr, gsn_ptr, gsc_ptr,
                    total, N, NR, NEED_P: tl.constexpr, NEED_O: tl.constexpr, NEED_CS: tl.constexpr, NEED_SN: tl.constexpr,
                    NEED_SC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    b = offs // N
    n = offs % N
    pid = tl.load(pid_ptr + n, m, other=0)
    j = b * NR + pid
    rxp = tl.load(rx_ptr + j, m, other=0.0)
    ryp = tl.load(ry_ptr + j, m, other=0.0)
    x = tl.load(p_ptr + offs * 2, m, other=0.0) * rxp
    y = tl.load(p_ptr + offs * 2 + 1, m, other=0.0) * ryp
    c = tl.load(cs_ptr + j, m, other=0.0)
    s = tl.load(sn_ptr + j, m, other=0.0)
    k = tl.load(sc_ptr + j, m, other=0.0)
    gx = tl.load(g_ptr + offs * 2, m, other=0.0)
    gy = tl.load(g_ptr + offs * 2 + 1, m, other=0.0)
    if NEED_P:
        tl.store(gp_ptr + offs * 2, k * (c * gx + s * gy) * rxp, m)
        tl.store(gp_ptr + offs * 2 + 1, k * (c * gy - s * gx) * ryp, m)
    if NEED_O:
        tl.atomic_add(go_ptr + j * 2, gx, m)
        tl.atomic_add(go_ptr + j * 2 + 1, gy, m)
    if NEED_CS:
        tl.atomic_add(gcs_ptr + j, k * (x * gx + y * gy), m)
    if NEED_SN:
        tl.atomic_add(gsn_ptr + j, k * (x * gy - y * gx), m)
    if NEED_SC:
        tl.atomic_add(gsc_ptr + j, (c * x - s * y) * gx + (s * x + c * y) * gy, m)


def rot_forward(p, o, cs, sn, sc, rx, ry, pid):
    B, N = p.shape[0], p.shape[1]
    p = p.contiguous()
    q = torch.empty_like(p)
    total = B * N
    if total:
        _rot_fwd_kernel[(triton.cdiv(total, BLOCK),)](p, o.contiguous(), cs.contiguous(), sn.contiguous(), sc.contiguous(), rx.contiguous(),
                                                       ry.contiguous(), pid, q, total, N, o.shape[1], BLOCK=BLOCK, **_OPTS)
    return q


def rot_backward(g, p, o, cs, sn, sc, rx, ry, pid, needs):
    B, N = p.shape[0], p.shape[1]
    p = p.contiguous()
    gp = torch.empty_like(p) if needs[0] else None
    go = torch.zeros_like(o) if needs[1] else None
    gcs = torch.zeros_like(cs) if needs[2] else None
    gsn = torch.zeros_like(sn) if needs[3] else None
    gsc = torch.zeros_like(sc) if needs[4] else None
    total = B * N
    if total == 0:
        return (torch.zeros_like(p) if needs[0] else None), go, gcs, gsn, gsc
    d = p
    _rot_bwd_kernel[(triton.cdiv(total, BLOCK),)](p, cs.contiguous(), sn.contiguous(), sc.contiguous(), rx.contiguous(), ry.contiguous(), pid,
                                                   g.contiguous(), gp if needs[0] else d, go if needs[1] else d, gcs if needs[2] else d,
                                                   gsn if needs[3] else d, gsc if needs[4] else d, total, N, o.shape[1],
                                                   NEED_P=needs[0], NEED_O=needs[1], NEED_CS=needs[2], NEED_SN=needs[3], NEED_SC=needs[4],
                                                   BLOCK=BLOCK, **_OPTS)
    return gp, go, gcs, gsn, gsc
