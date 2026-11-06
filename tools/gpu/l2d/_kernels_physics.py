"""A8：可微分物理（physics.py）的 Triton 融合核：整段 T 帧的前向一个核、反向一个核（每个批元素一个 CTA）。

前向与 physics.PhysicsRig 的 eager 实现逐式相同：同样的运算顺序、关闭 FMA 合并、IEEE 除法 / 开方（div_rn / sqrt_rn）、
libdevice 的 sin / cos / atan2（与 PyTorch CUDA 调用的是同一套 CUDA 数学库），「张量 / Python 标量」按 PyTorch 的做法乘以倒数，
Float32Array 的位置做 float32 舍入。所以前向与 eager 的 rollout 逐位相同（a8_compare.py 对拍）。

反向是手写的伴随（不是 autograd），公式与 eager autograd 的各个算子一致（atan2 在 (0,0) 处梯度为 0、normalize 长度 0 处为 0、
where / 夹取 / 吸附按所在分支、float32 舍入直通、输入恰在中点时取两侧斜率平均），只有浮点求和顺序不同。
所有累加都按固定顺序（逆表，见 _Plan），没有原子加：同一输入两次结果逐位相同。

结构（一帧，一个批元素；setting 依赖层 l = 0..NL−1，与 physics.py 的 stages 相同，要求每层一组，即 split_by_length=False）：
  子步 k：C = f32(ic·(1−w) + pq·w)，ic ← C，prev ← cur；
          每层：输入归一化 + 旋转 → 重力 / 空气阻力 → 粒子链（链内顺序、链间并行）→ 输出（写 vals 与 cur）→ 按轮写回 C；
  interpolate：out = f32(params)，物理输出参数 = 按轮写入的 prev·(1−α) + cur·α。
同一 CTA 内各阶段之间用 tl.debug_barrier()（__syncthreads）同步，阶段间的数据经全局显存（L1 / L2 命中）交换。
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

def _find_libdevice():
    """PyTorch 编译时所用 CUDA 版本的 libdevice。Triton 自带的 libdevice 版本不同：atan2 有约 0.3% 的输入差 1 ulp
    （sin / cos / sqrt / 除法相同），而物理链是混沌的，1 ulp 会被放大。找不到时退回 Triton 自带的（前向不再逐位相同）。"""
    import os
    roots = []
    if torch.version.cuda:
        roots.append(f'/usr/local/cuda-{torch.version.cuda}')
    roots += [os.environ.get('CUDA_HOME', ''), os.environ.get('CUDA_PATH', ''), '/usr/local/cuda']
    for r in roots:
        p = os.path.join(r, 'nvvm', 'libdevice', 'libdevice.10.bc') if r else ''
        if p and os.path.exists(p):
            return p
    return None


LIBDEVICE = _find_libdevice()
_OPTS = dict(num_warps=2, enable_fp_fusion=False)
if LIBDEVICE:
    _OPTS['extern_libs'] = (('libdevice', LIBDEVICE),)

# ---------------------------------------------------------------- 常量表布局
# int32 表 I 的表头：各段的起始偏移与尺寸
_HDR = ['QGLOB', 'OUTP', 'PMAP', 'META', 'SLOT', 'INQ', 'HPOS', 'HNEG', 'OI1', 'OPAR', 'OTYP', 'OGID',
        'RTAB', 'RSRC', 'RDST', 'RFUL', 'ITAB', 'ISRC', 'IDST', 'IFUL', 'NZ', 'INVIN', 'INVOUT',
        'HASC', 'P', 'Q', 'NO', 'NOP', 'NP', 'NS', 'NL', 'NOVT', 'NSLOT', 'DIN', 'DOUT', 'NIRI']
# float64 表 F 的表头（偏移存放在 I 表里）
_FHDR = ['NMIN', 'NMAX', 'NMID', 'RPOS', 'RNEG', 'MNORM', 'RMID', 'SW0', 'SW1', 'TYPE', 'CONST',
         'ACC', 'RADI', 'MOB', 'THR', 'OSGN', 'RSCL', 'RMIN', 'RMAX', 'RW', 'ISCL', 'IMIN', 'IMAX', 'IW']
_NH = len(_HDR) + len(_FHDR)
# 每层的元数据（I[META + l·16 + 字段]）
_META = ['R', 'L', 'K', 'NI', 'ONP', 'OSET', 'OSLOT', 'OINP', 'NOV', 'OOUT', 'R0', 'NR']

for _i, _n in enumerate(_HDR):
    globals()['H_' + _n] = tl.constexpr(_i)
for _i, _n in enumerate(_FHDR):
    globals()['HF_' + _n] = tl.constexpr(len(_HDR) + _i)
for _i, _n in enumerate(_META):
    globals()['M_' + _n] = tl.constexpr(_i)

PI = tl.constexpr(math.pi)
TWO_PI = tl.constexpr(math.pi * 2.0)
INV180 = tl.constexpr(1.0 / 180.0)   # PyTorch：张量 / Python 标量 = 张量 × (1 / 标量)
INV5 = tl.constexpr(1.0 / 5.0)       # AIR_RESISTANCE
NSF = tl.constexpr(12)   # 磁带：每个 setting 的中间量个数
NPF = tl.constexpr(4)    # 磁带：每个粒子的中间量个数


@triton.jit
def _r32(x, EMU: tl.constexpr):
    if EMU:
        return x.to(tl.float32).to(tl.float64)
    else:
        return x


@triton.jit
def _h(I, k: tl.constexpr):
    return tl.load(I + k)


@triton.jit
def _wrap(ret):
    ret = tl.where(ret < -PI, ret + TWO_PI, ret)
    return tl.where(ret > PI, ret - TWO_PI, ret)


# ---------------------------------------------------------------- 前向各阶段（前向核与反向核的重算共用）

@triton.jit
def _ph_c0(xrow, I, F, Cc, ic, w, EMU: tl.constexpr, BQ: tl.constexpr):
    """C = f32(ic·(1−w) + f32(params)[qglob]·w)；ic ← C。"""
    Q = _h(I, H_Q)
    oq = _h(I, H_QGLOB)
    w1 = 1.0 - w
    for q0 in range(0, Q, BQ):
        q = q0 + tl.arange(0, BQ)
        m = q < Q
        g = tl.load(I + oq + q, m, other=0)
        pq = _r32(tl.load(xrow + g, m, other=0.0), EMU)
        a = tl.load(ic + q, m, other=0.0)
        c = _r32(a * w1 + pq * w, EMU)
        tl.store(Cc + q, c, m)
        tl.store(ic + q, c, m)


@triton.jit
def _copy(dst, src, n, BLK: tl.constexpr):
    for o in range(0, n, BLK):
        j = o + tl.arange(0, BLK)
        m = j < n
        tl.store(dst + j, tl.load(src + j, m, other=0.0), m)


@triton.jit
def _inputs(I, F, Cc, l, r, m):
    """各 setting 的输入（[BR] 向量）：返回 旋转后的 tx、ty，angle，旋转前的 tx、ty，以及旋转用的 cos / sin。"""
    mb = _h(I, H_META) + l * 16
    K = tl.load(I + mb + M_K)
    oslot = tl.load(I + mb + M_OSLOT)
    oinp = tl.load(I + mb + M_OINP)
    oset = tl.load(I + mb + M_OSET)
    o_slot = _h(I, H_SLOT)
    o_inq = _h(I, H_INQ)
    o_hp = _h(I, H_HPOS)
    o_hn = _h(I, H_HNEG)
    f_nmin = _h(I, HF_NMIN)
    f_nmax = _h(I, HF_NMAX)
    f_nmid = _h(I, HF_NMID)
    f_rpos = _h(I, HF_RPOS)
    f_rneg = _h(I, HF_RNEG)
    f_m = _h(I, HF_MNORM)
    f_rmid = _h(I, HF_RMID)
    f_sw0 = _h(I, HF_SW0)
    f_sw1 = _h(I, HF_SW1)
    f_typ = _h(I, HF_TYPE)
    tx = tl.zeros_like(r).to(tl.float64)
    ty = tx
    ang = tx
    for k in range(K):
        sp = oslot + r * K + k                       # 槽位（全局）
        s = tl.load(I + o_slot + sp, m, other=0)     # 本层输入表里的下标（NI = 填充）
        e = oinp + s
        q = tl.load(I + o_inq + e, m, other=-1)
        v = tl.load(Cc + q, m & (q >= 0), other=0.0)
        nmax = tl.load(F + f_nmax + e, m, other=0.0)
        nmin = tl.load(F + f_nmin + e, m, other=0.0)
        v = tl.where(nmax < v, nmax, v)
        v = tl.where(nmin > v, nmin, v)
        pv = v - tl.load(F + f_nmid + e, m, other=0.0)
        mn = tl.load(F + f_m + e, m, other=0.0)
        hp = tl.load(I + o_hp + e, m, other=0) != 0
        hn = tl.load(I + o_hn + e, m, other=0) != 0
        rp = tl.where(hp, pv * tl.load(F + f_rpos + e, m, other=0.0) + mn, 0.0)
        rn = tl.where(hn, pv * tl.load(F + f_rneg + e, m, other=0.0) + mn, 0.0)
        r0 = pv * tl.load(F + f_rmid + e, m, other=0.0) + mn
        res = tl.where(pv > 0, rp, tl.where(pv < 0, rn, r0))
        c = (res * tl.load(F + f_sw0 + e, m, other=0.0)) * tl.load(F + f_sw1 + e, m, other=0.0)
        cx = c * tl.load(F + f_typ + sp * 3 + 0, m, other=0.0)
        cy = c * tl.load(F + f_typ + sp * 3 + 1, m, other=0.0)
        ca = c * tl.load(F + f_typ + sp * 3 + 2, m, other=0.0)
        tx = tl.where(k == 0, cx, tx + cx)
        ty = tl.where(k == 0, cy, ty + cy)
        ang = tl.where(k == 0, ca, ang + ca)
    f_c = _h(I, HF_CONST)
    if _h(I, H_HASC) != 0:   # has_const：模型里不存在的输入参数的常量贡献
        tx = tx + tl.load(F + f_c + (oset + r) * 3 + 0, m, other=0.0)
        ty = ty + tl.load(F + f_c + (oset + r) * 3 + 1, m, other=0.0)
        ang = ang + tl.load(F + f_c + (oset + r) * 3 + 2, m, other=0.0)
    ra = ((-ang) * INV180) * PI
    cri = libdevice.cos(ra)
    sri = libdevice.sin(ra)
    tx2 = tx * cri - ty * sri
    ty2 = tx2 * sri + ty * cri
    return tx2, ty2, ang, tx, ty, cri, sri


@triton.jit
def _gravity(ang):
    tr = (ang * INV180) * PI
    s = libdevice.sin(tr)
    c = libdevice.cos(tr)
    l2 = s * s + c * c
    ok = l2 > 0
    ln = libdevice.sqrt_rn(tl.where(ok, l2, 1.0))
    nan = float('nan')
    return tl.where(ok, libdevice.div_rn(s, ln), nan), tl.where(ok, libdevice.div_rn(c, ln), nan), s, c, ln, ok


@triton.jit
def _air(lgx, lgy, cgx, cgy):
    ret = libdevice.atan2(cgy, cgx) - libdevice.atan2(lgy, lgx)
    rad = _wrap(ret) * INV5
    return libdevice.cos(rad), libdevice.sin(rad)


@triton.jit
def _ph_particles(I, F, DL, DS, Cc, spx, spy, svx, svy, sgx, sgy, l, TS, TP, WX: tl.constexpr, WY: tl.constexpr,
                  SAVE: tl.constexpr, BR: tl.constexpr):
    """updateParticles：第 l 层的全部链（就地更新 s* 状态）。SAVE 时把反向要用的中间量写进 TS（每个 setting）/ TP（每个粒子）。"""
    mb = _h(I, H_META) + l * 16
    R = tl.load(I + mb + M_R)
    L = tl.load(I + mb + M_L)
    onp = tl.load(I + mb + M_ONP)
    oset = tl.load(I + mb + M_OSET)
    NS = _h(I, H_NS)
    NP = _h(I, H_NP)
    o_nz = _h(I, H_NZ)
    f_acc = _h(I, HF_ACC)
    f_rad = _h(I, HF_RADI)
    f_mob = _h(I, HF_MOB)
    f_thr = _h(I, HF_THR)
    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        m = r < R
        tx2, ty2, ang, tx, ty, cri, sri = _inputs(I, F, Cc, l, r, m)
        cgx, cgy, sg, cg, lng, okg = _gravity(ang)
        lgx = tl.load(sgx + oset + r, m, other=0.0)
        lgy = tl.load(sgy + oset + r, m, other=1.0)
        cr, sr = _air(lgx, lgy, cgx, cgy)
        if SAVE:
            s = oset + r
            tl.store(TS + 0 * NS + s, tx, m)
            tl.store(TS + 1 * NS + s, ty, m)
            tl.store(TS + 2 * NS + s, cri, m)
            tl.store(TS + 3 * NS + s, sri, m)
            tl.store(TS + 4 * NS + s, tx2, m)
            tl.store(TS + 5 * NS + s, cgx, m)
            tl.store(TS + 6 * NS + s, cgy, m)
            tl.store(TS + 7 * NS + s, sg, m)
            tl.store(TS + 8 * NS + s, cg, m)
            tl.store(TS + 9 * NS + s, tl.where(okg, lng, 0.0), m)   # 0 = 长度为 0（梯度为 0）
            tl.store(TS + 10 * NS + s, cr, m)
            tl.store(TS + 11 * NS + s, sr, m)
        thr = tl.load(F + f_thr + oset + r, m, other=0.0)
        base = onp + r * L
        tl.store(spx + base, tx2, m)
        tl.store(spy + base, ty2, m)
        ox = tx2
        oy = ty2
        for i in range(1, L):
            p = base + i
            a = tl.load(F + f_acc + p, m, other=1.0)
            d = tl.load(DL + p, m, other=1.0)
            fx = cgx * a + WX
            fy = cgy * a + WY
            lpx = tl.load(spx + p, m, other=0.0)
            lpy = tl.load(spy + p, m, other=0.0)
            vx = tl.load(svx + p, m, other=0.0)
            vy = tl.load(svy + p, m, other=0.0)
            dx = lpx - ox
            dy = lpy - oy
            dx = cr * dx - dy * sr
            dy = sr * dx + dy * cr   # Framework 原样：用已旋转的 x
            qx = ox + dx
            qy = oy + dy
            qx = (qx + vx * d) + (fx * d) * d
            qy = (qy + vy * d) + (fy * d) * d
            ux = qx - ox
            uy = qy - oy
            l2 = ux * ux + uy * uy
            ok = l2 > 0
            ln = libdevice.sqrt_rn(tl.where(ok, l2, 1.0))
            ex = tl.where(ok, libdevice.div_rn(ux, ln), float('nan'))
            ey = tl.where(ok, libdevice.div_rn(uy, ln), float('nan'))
            rr = tl.load(F + f_rad + p, m, other=1.0)
            qx = ox + ex * rr
            qy = oy + ey * rr
            snap = tl.abs(qx) < thr
            qx = tl.where(snap, 0.0, qx)
            nz = tl.load(I + o_nz + p, m, other=0) != 0
            ds = tl.load(DS + p, m, other=1.0)
            mob = tl.load(F + f_mob + p, m, other=1.0)
            nvx = tl.where(nz, libdevice.div_rn(qx - lpx, ds) * mob, vx)
            nvy = tl.where(nz, libdevice.div_rn(qy - lpy, ds) * mob, vy)
            tl.store(spx + p, qx, m)
            tl.store(spy + p, qy, m)
            tl.store(svx + p, nvx, m)
            tl.store(svy + p, nvy, m)
            if SAVE:
                tl.store(TP + 0 * NP + p, tl.where(ok, ex, 0.0), m)
                tl.store(TP + 1 * NP + p, tl.where(ok, ey, 0.0), m)
                tl.store(TP + 2 * NP + p, tl.where(ok, ln, 0.0), m)        # 0 = 长度为 0（梯度为 0）
                tl.store(TP + 3 * NP + p, tl.where(snap, 1.0, 0.0), m)
            ox = qx
            oy = qy
        tl.store(sgx + oset + r, cgx, m)
        tl.store(sgy + oset + r, cgy, m)


@triton.jit
def _out_geom(spx, spy, g1, par, GXN: tl.constexpr, GYN: tl.constexpr, m):
    x1 = tl.load(spx + g1, m, other=0.0)
    y1 = tl.load(spy + g1, m, other=0.0)
    x0 = tl.load(spx + g1 - 1, m, other=0.0)
    y0 = tl.load(spy + g1 - 1, m, other=0.0)
    i2 = tl.where(par, g1 - 2, g1 - 1)
    x2 = tl.load(spx + i2, m, other=0.0)
    y2 = tl.load(spy + i2, m, other=0.0)
    tx = x1 - x0
    ty = y1 - y0
    pxp = tl.where(par, x0 - x2, GXN)
    pyp = tl.where(par, y0 - y2, GYN)
    return tx, ty, pxp, pyp


@triton.jit
def _ph_outputs(I, F, spx, spy, vals, cur, l, GXN: tl.constexpr, GYN: tl.constexpr, BO: tl.constexpr):
    """getOutputTranslationX / Y / Angle：vals[OOUT + o] 与 cur[gidx]。"""
    mb = _h(I, H_META) + l * 16
    NOV = tl.load(I + mb + M_NOV)
    oo = tl.load(I + mb + M_OOUT)
    o_i1 = _h(I, H_OI1)
    o_par = _h(I, H_OPAR)
    o_typ = _h(I, H_OTYP)
    o_gid = _h(I, H_OGID)
    f_sgn = _h(I, HF_OSGN)
    for o0 in range(0, NOV, BO):
        o = oo + o0 + tl.arange(0, BO)
        m = (o0 + tl.arange(0, BO)) < NOV
        g1 = tl.load(I + o_i1 + o, m, other=2)
        par = tl.load(I + o_par + o, m, other=0) != 0
        tx, ty, pxp, pyp = _out_geom(spx, spy, g1, par, GXN, GYN, m)
        angle = _wrap(libdevice.atan2(ty, tx) - libdevice.atan2(pyp, pxp))
        t = tl.load(I + o_typ + o, m, other=2)
        val = tl.where(t == 0, tx, tl.where(t == 1, ty, angle))
        val = val * tl.load(F + f_sgn + o, m, other=1.0)
        tl.store(vals + o, val, m)
        gid = tl.load(I + o_gid + o, m, other=0)
        tl.store(cur + gid, val, m)


@triton.jit
def _ph_rounds(I, F, tgt, vals, r_first, n_rounds, TAB: tl.constexpr, SRC: tl.constexpr, DST: tl.constexpr,
               FUL: tl.constexpr, SCL: tl.constexpr, VMIN: tl.constexpr, VMAX: tl.constexpr, W: tl.constexpr,
               EMU: tl.constexpr, BRD: tl.constexpr):
    """updateOutputParameterValue（按轮）：tgt[dst] = f32(混合(夹取(vals[src]·scale), tgt[dst]))。"""
    o_tab = _h(I, TAB)
    o_src = _h(I, SRC)
    o_dst = _h(I, DST)
    o_ful = _h(I, FUL)
    f_scl = _h(I, SCL)
    f_min = _h(I, VMIN)
    f_max = _h(I, VMAX)
    f_w = _h(I, W)
    for rr in range(r_first, r_first + n_rounds):
        st = tl.load(I + o_tab + rr * 2)
        n = tl.load(I + o_tab + rr * 2 + 1)
        for j0 in range(0, n, BRD):
            j = st + j0 + tl.arange(0, BRD)
            m = (j0 + tl.arange(0, BRD)) < n
            src = tl.load(I + o_src + j, m, other=0)
            dst = tl.load(I + o_dst + j, m, other=0)
            v = tl.load(vals + src, m, other=0.0) * tl.load(F + f_scl + j, m, other=0.0)
            vmin = tl.load(F + f_min + j, m, other=0.0)
            vmax = tl.load(F + f_max + j, m, other=0.0)
            v = tl.where(v < vmin, vmin, tl.where(v > vmax, vmax, v))
            base = tl.load(tgt + dst, m, other=0.0)
            w = tl.load(F + f_w + j, m, other=0.0)
            full = tl.load(I + o_ful + j, m, other=0) != 0
            v = tl.where(full, v, base * (1.0 - w) + v * w)
            tl.store(tgt + dst, _r32(v, EMU), m)
        tl.debug_barrier()


@triton.jit
def _ph_interp(I, F, xrow, orow, ivals, tgt, prev, cur, a, resolved, EMU: tl.constexpr, BP: tl.constexpr,
               BRD: tl.constexpr):
    """interpolate + 写回：orow = f32(params)，resolved 时物理输出参数按轮写入 prev·(1−α) + cur·α。"""
    P = _h(I, H_P)
    o_pmap = _h(I, H_PMAP)
    if resolved != 0:
        NOP = _h(I, H_NOP)
        NO = _h(I, H_NO)
        o_outp = _h(I, H_OUTP)
        for j0 in range(0, NOP, BP):
            j = j0 + tl.arange(0, BP)
            m = j < NOP
            p = tl.load(I + o_outp + j, m, other=0)
            tl.store(tgt + j, _r32(tl.load(xrow + p, m, other=0.0), EMU), m)
        a1 = 1.0 - a
        for j0 in range(0, NO, BP):
            j = j0 + tl.arange(0, BP)
            m = j < NO
            tl.store(ivals + j, tl.load(prev + j, m, other=0.0) * a1 + tl.load(cur + j, m, other=0.0) * a, m)
        tl.debug_barrier()
        _ph_rounds(I, F, tgt, ivals, 0, _h(I, H_NIRI), H_ITAB, H_ISRC, H_IDST, H_IFUL, HF_ISCL, HF_IMIN, HF_IMAX, HF_IW,
                   EMU, BRD)
    for p0 in range(0, P, BP):
        p = p0 + tl.arange(0, BP)
        m = p < P
        v = _r32(tl.load(xrow + p, m, other=0.0), EMU)
        if resolved != 0:
            j = tl.load(I + o_pmap + p, m, other=-1)
            v = tl.where(j >= 0, tl.load(tgt + j, m & (j >= 0), other=0.0), v)
        tl.store(orow + p, v, m)


@triton.jit
def _fwd_kernel(X, OUT, ST, SNAP, TAPE, SCR, SCHN, SCHW, SCHA, SCHR, I, F, DL, DS, T, B, STLEN, TLEN, SCRLEN, MAXS,
                SAVE: tl.constexpr, EMU: tl.constexpr, WX: tl.constexpr, WY: tl.constexpr, GXN: tl.constexpr,
                GYN: tl.constexpr, BQ: tl.constexpr, BR: tl.constexpr, BO: tl.constexpr, BRD: tl.constexpr,
                BP: tl.constexpr, BST: tl.constexpr):
    """整段前向。SAVE 时每帧写 SNAP（帧开始时的状态）与 TAPE（每个子步：子步后的粒子状态、各层开始时的 C、各层输出、
    setting / 粒子中间量；帧末的 cur / prev），供反向核使用。"""
    b = tl.program_id(0)
    P = _h(I, H_P)
    Q = _h(I, H_Q)
    NO = _h(I, H_NO)
    NP = _h(I, H_NP)
    NS = _h(I, H_NS)
    NL = _h(I, H_NL)
    NOVT = _h(I, H_NOVT)
    SPLEN = 4 * NP + 2 * NS
    SUBLEN = SPLEN + NL * Q + NOVT + NS * NSF + NP * NPF
    st = ST + b.to(tl.int64) * STLEN
    spx = st
    spy = st + NP
    svx = st + 2 * NP
    svy = st + 3 * NP
    sgx = st + 4 * NP
    sgy = sgx + NS
    cur = sgy + NS
    prev = cur + NO
    ic = prev + NO
    scr = SCR + b.to(tl.int64) * SCRLEN
    Cc = scr
    vals0 = Cc + Q
    ivals = vals0 + NOVT
    tgt = ivals + NO
    for t in range(T):
        if SAVE:
            _copy(SNAP + (t * B + b).to(tl.int64) * STLEN, st, STLEN, BST)
        tape = TAPE + (t * B + b).to(tl.int64) * TLEN
        xrow = X + (t * B + b).to(tl.int64) * P
        ns = tl.load(SCHN + t)
        for k in range(ns):
            w = tl.load(SCHW + t * MAXS + k)
            sub = tape + k * SUBLEN
            vals = vals0
            if SAVE:
                vals = sub + SPLEN + NL * Q
            tl.debug_barrier()
            _ph_c0(xrow, I, F, Cc, ic, w, EMU, BQ)
            _copy(prev, cur, NO, BQ)
            tl.debug_barrier()
            for l in range(NL):
                if SAVE:
                    _copy(sub + SPLEN + l * Q, Cc, Q, BQ)
                ts = sub + SPLEN + NL * Q + NOVT
                _ph_particles(I, F, DL, DS, Cc, spx, spy, svx, svy, sgx, sgy, l, ts, ts + NS * NSF, WX, WY, SAVE, BR)
                tl.debug_barrier()
                _ph_outputs(I, F, spx, spy, vals, cur, l, GXN, GYN, BO)
                tl.debug_barrier()
                mb = _h(I, H_META) + l * 16
                _ph_rounds(I, F, Cc, vals, tl.load(I + mb + M_R0), tl.load(I + mb + M_NR),
                           H_RTAB, H_RSRC, H_RDST, H_RFUL, HF_RSCL, HF_RMIN, HF_RMAX, HF_RW, EMU, BRD)
            if SAVE:
                _copy(sub, st, SPLEN, BST)
        tl.debug_barrier()
        _ph_interp(I, F, xrow, OUT + (t * B + b).to(tl.int64) * P, ivals, tgt, prev, cur,
                   tl.load(SCHA + t), tl.load(SCHR + t), EMU, BP, BRD)
        if SAVE:
            _copy(tape + MAXS * SUBLEN, cur, 2 * NO, BST)
        tl.debug_barrier()


# ---------------------------------------------------------------- 反向

@triton.jit
def _bwd_level(I, F, DL, MDS, Cl, opx, opy, ovx, ovy, ogx, ogy, npx, npy, vals, TS, TP,
               gpx, gpy, gvx, gvy, ggx, ggy, gC, gvals, GOX, GOY, GSL, l,
               WX: tl.constexpr, WY: tl.constexpr, GXN: tl.constexpr, GYN: tl.constexpr,
               BQ: tl.constexpr, BR: tl.constexpr, BO: tl.constexpr, BRD: tl.constexpr, BNP: tl.constexpr):
    """第 l 层一个子步的伴随。o* = 子步前的状态，n* = 子步后的位置，Cl = 本层开始时的 C，vals / TS / TP = 前向记下的中间量；
    g* 进来时是对子步后状态的梯度，出去时是对子步前状态的梯度；gC 进来是对本层写回之后的 C 的梯度，出去是对 Cl 的梯度。"""
    mb = _h(I, H_META) + l * 16
    R = tl.load(I + mb + M_R)
    L = tl.load(I + mb + M_L)
    K = tl.load(I + mb + M_K)
    onp = tl.load(I + mb + M_ONP)
    oset = tl.load(I + mb + M_OSET)
    oslot = tl.load(I + mb + M_OSLOT)
    oinp = tl.load(I + mb + M_OINP)
    NOV = tl.load(I + mb + M_NOV)
    oo = tl.load(I + mb + M_OOUT)
    NS = _h(I, H_NS)
    NP = _h(I, H_NP)
    # ---- 写回轮次（逆序）
    o_tab = _h(I, H_RTAB)
    o_src = _h(I, H_RSRC)
    o_dst = _h(I, H_RDST)
    o_ful = _h(I, H_RFUL)
    f_scl = _h(I, HF_RSCL)
    f_min = _h(I, HF_RMIN)
    f_max = _h(I, HF_RMAX)
    f_w = _h(I, HF_RW)
    r_first = tl.load(I + mb + M_R0)
    n_rounds = tl.load(I + mb + M_NR)
    for rr_ in range(0, n_rounds):
        rr = r_first + n_rounds - 1 - rr_
        st = tl.load(I + o_tab + rr * 2)
        n = tl.load(I + o_tab + rr * 2 + 1)
        for j0 in range(0, n, BRD):
            j = st + j0 + tl.arange(0, BRD)
            m = (j0 + tl.arange(0, BRD)) < n
            src = tl.load(I + o_src + j, m, other=0)
            dst = tl.load(I + o_dst + j, m, other=0)
            scl = tl.load(F + f_scl + j, m, other=0.0)
            v = tl.load(vals + src, m, other=0.0) * scl
            vmin = tl.load(F + f_min + j, m, other=0.0)
            vmax = tl.load(F + f_max + j, m, other=0.0)
            w = tl.load(F + f_w + j, m, other=0.0)
            full = tl.load(I + o_ful + j, m, other=0) != 0
            gv = tl.load(gC + dst, m, other=0.0)
            tl.store(gC + dst, tl.where(full, 0.0, gv * (1.0 - w)), m)
            gcl = tl.where(full, gv, gv * w)
            pas = (~(v < vmin)) & (~(v > vmax))
            gs = tl.where(pas, gcl * scl, 0.0)
            tl.store(gvals + src, tl.load(gvals + src, m, other=0.0) + gs, m)
        tl.debug_barrier()
    # ---- 输出（每个输出对三处粒子位置的梯度，写进 GOX / GOY[o·3 + 角色]）
    o_i1 = _h(I, H_OI1)
    o_par = _h(I, H_OPAR)
    o_typ = _h(I, H_OTYP)
    f_sgn = _h(I, HF_OSGN)
    for o0 in range(0, NOV, BO):
        o = oo + o0 + tl.arange(0, BO)
        m = (o0 + tl.arange(0, BO)) < NOV
        g1 = tl.load(I + o_i1 + o, m, other=2)
        par = tl.load(I + o_par + o, m, other=0) != 0
        tx, ty, pxp, pyp = _out_geom(npx, npy, g1, par, GXN, GYN, m)
        g = tl.load(gvals + o, m, other=0.0) * tl.load(F + f_sgn + o, m, other=1.0)
        t = tl.load(I + o_typ + o, m, other=2)
        gA = tl.where(t == 2, g, 0.0)
        r2a = tx * tx + ty * ty
        inva = tl.where(r2a > 0, libdevice.div_rn(gA, tl.where(r2a > 0, r2a, 1.0)), 0.0)
        gtx = tl.where(t == 0, g, 0.0) - ty * inva
        gty = tl.where(t == 1, g, 0.0) + tx * inva
        r2p = pxp * pxp + pyp * pyp
        invp = tl.where(r2p > 0, libdevice.div_rn(-gA, tl.where(r2p > 0, r2p, 1.0)), 0.0)
        gpxp = tl.where(par, -pyp * invp, 0.0)
        gpyp = tl.where(par, pxp * invp, 0.0)
        tl.store(GOX + o * 3 + 0, gtx, m)
        tl.store(GOX + o * 3 + 1, gpxp - gtx, m)
        tl.store(GOX + o * 3 + 2, -gpxp, m)
        tl.store(GOY + o * 3 + 0, gty, m)
        tl.store(GOY + o * 3 + 1, gpyp - gty, m)
        tl.store(GOY + o * 3 + 2, -gpyp, m)
    tl.debug_barrier()
    # ---- 粒子位置收集输出的梯度（逆表，固定顺序）
    o_inv = _h(I, H_INVOUT)
    D = _h(I, H_DOUT)
    nlp = R * L
    for p0 in range(0, nlp, BNP):
        pl = p0 + tl.arange(0, BNP)
        m = pl < nlp
        p = onp + pl
        ax = tl.load(gpx + p, m, other=0.0)
        ay = tl.load(gpy + p, m, other=0.0)
        for d in range(D):
            e = tl.load(I + o_inv + p * D + d, m, other=-1)
            ok = m & (e >= 0)
            ax = ax + tl.load(GOX + e, ok, other=0.0)
            ay = ay + tl.load(GOY + e, ok, other=0.0)
        tl.store(gpx + p, ax, m)
        tl.store(gpy + p, ay, m)
    tl.debug_barrier()
    # ---- 粒子链（逆序）+ 重力 / 空气阻力 / 输入
    o_nz = _h(I, H_NZ)
    f_acc = _h(I, HF_ACC)
    f_rad = _h(I, HF_RADI)
    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        m = r < R
        s = oset + r
        txp = tl.load(TS + 0 * NS + s, m, other=0.0)
        typ_ = tl.load(TS + 1 * NS + s, m, other=0.0)
        cri = tl.load(TS + 2 * NS + s, m, other=1.0)
        sri = tl.load(TS + 3 * NS + s, m, other=0.0)
        tx2 = tl.load(TS + 4 * NS + s, m, other=0.0)
        cgx = tl.load(TS + 5 * NS + s, m, other=0.0)
        cgy = tl.load(TS + 6 * NS + s, m, other=1.0)
        sg = tl.load(TS + 7 * NS + s, m, other=0.0)
        cg = tl.load(TS + 8 * NS + s, m, other=1.0)
        lng = tl.load(TS + 9 * NS + s, m, other=1.0)
        cr = tl.load(TS + 10 * NS + s, m, other=1.0)
        sr = tl.load(TS + 11 * NS + s, m, other=0.0)
        lgx = tl.load(ogx + s, m, other=0.0)
        lgy = tl.load(ogy + s, m, other=1.0)
        base = onp + r * L
        cx = tl.zeros_like(cr)
        cy = cx
        g_cgx = cx
        g_cgy = cx
        g_cr = cx
        g_sr = cx
        for ii in range(0, L - 1):
            i = L - 1 - ii
            p = base + i
            ox = tl.load(npx + p - 1, m, other=0.0)
            oy = tl.load(npy + p - 1, m, other=0.0)
            lpx = tl.load(opx + p, m, other=0.0)
            lpy = tl.load(opy + p, m, other=0.0)
            a = tl.load(F + f_acc + p, m, other=1.0)
            d = tl.load(DL + p, m, other=1.0)
            rr = tl.load(F + f_rad + p, m, other=1.0)
            nz = tl.load(I + o_nz + p, m, other=0) != 0
            mds = tl.load(MDS + p, m, other=0.0)
            ex = tl.load(TP + 0 * NP + p, m, other=0.0)
            ey = tl.load(TP + 1 * NP + p, m, other=0.0)
            ln = tl.load(TP + 2 * NP + p, m, other=0.0)
            snap = tl.load(TP + 3 * NP + p, m, other=0.0) != 0.0
            dx0 = lpx - ox
            dy0 = lpy - oy
            dx1 = cr * dx0 - dy0 * sr
            # 伴随
            gnx = tl.load(gpx + p, m, other=0.0) + cx
            gny = tl.load(gpy + p, m, other=0.0) + cy
            gnvx = tl.load(gvx + p, m, other=0.0)
            gnvy = tl.load(gvy + p, m, other=0.0)
            gbx = tl.where(nz, gnvx * mds, 0.0)      # nv = ((n − lp) / ds)·mob
            gby = tl.where(nz, gnvy * mds, 0.0)
            gvxo = tl.where(nz, 0.0, gnvx)
            gvyo = tl.where(nz, 0.0, gnvy)
            gnx = gnx + gbx
            gny = gny + gby
            glpx = -gbx
            glpy = -gby
            gq3x = tl.where(snap, 0.0, gnx)            # 吸附
            gq3y = gny
            gex = gq3x * rr                            # q3 = o + e·r
            gey = gq3y * rr
            dot = ex * gex + ey * gey                  # e = u / |u|
            il = tl.where(ln > 0, 1.0 / tl.where(ln > 0, ln, 1.0), 0.0)
            gux = (gex - ex * dot) * il
            guy = (gey - ey * dot) * il
            gox = gq3x                                 # o 的梯度：q3 = o + …，u = q2 − o，q1 = o + d1，d0 = lp − o
            goy = gq3y
            gvxo = gvxo + gux * d                      # q2 = (q1 + v·d) + (f·d)·d
            gvyo = gvyo + guy * d
            gfx = (gux * d) * d
            gfy = (guy * d) * d
            gdx1 = gux
            gdy1 = guy
            g_sr = g_sr + gdy1 * dx1                   # dy1 = sr·dx1 + dy0·cr
            gdx1 = gdx1 + gdy1 * sr
            gdy0 = gdy1 * cr
            g_cr = g_cr + gdy1 * dy0
            g_cr = g_cr + gdx1 * dx0                   # dx1 = cr·dx0 − dy0·sr
            gdx0 = gdx1 * cr
            gdy0 = gdy0 - gdx1 * sr
            g_sr = g_sr - gdx1 * dy0
            glpx = glpx + gdx0
            glpy = glpy + gdy0
            gox = gox - gdx0
            goy = goy - gdy0
            g_cgx = g_cgx + gfx * a                    # f = g·acc + wind
            g_cgy = g_cgy + gfy * a
            tl.store(gpx + p, glpx, m)
            tl.store(gpy + p, glpy, m)
            tl.store(gvx + p, gvxo, m)
            tl.store(gvy + p, gvyo, m)
            cx = gox
            cy = goy
        # 粒子 0：新位置 = (tx2, ty2)；旧位置不用；速度原样
        gtx2 = tl.load(gpx + base, m, other=0.0) + cx
        gty2 = tl.load(gpy + base, m, other=0.0) + cy
        tl.store(gpx + base, tl.zeros_like(cx), m)
        tl.store(gpy + base, tl.zeros_like(cx), m)
        # lastGravity ← 本步重力
        g_cgx = g_cgx + tl.load(ggx + s, m, other=0.0)
        g_cgy = g_cgy + tl.load(ggy + s, m, other=0.0)
        # 空气阻力：rad = wrap(atan2(cg) − atan2(lg)) / 5
        g_ret = (-g_cr * sr + g_sr * cr) * INV5
        r2c = cgx * cgx + cgy * cgy
        invc = tl.where(r2c > 0, libdevice.div_rn(g_ret, tl.where(r2c > 0, r2c, 1.0)), 0.0)
        g_cgy = g_cgy + cgx * invc
        g_cgx = g_cgx - cgy * invc
        r2l = lgx * lgx + lgy * lgy
        invl = tl.where(r2l > 0, libdevice.div_rn(-g_ret, tl.where(r2l > 0, r2l, 1.0)), 0.0)
        tl.store(ggx + s, -lgy * invl, m)
        tl.store(ggy + s, lgx * invl, m)
        # 重力 normalize(sin, cos)
        dotg = cgx * g_cgx + cgy * g_cgy
        okg = lng > 0
        gsg = tl.where(okg, libdevice.div_rn(g_cgx - cgx * dotg, tl.where(okg, lng, 1.0)), 0.0)
        gcg = tl.where(okg, libdevice.div_rn(g_cgy - cgy * dotg, tl.where(okg, lng, 1.0)), 0.0)
        g_ang = ((gsg * cg - gcg * sg) * PI) * INV180
        # 输入的旋转：tx2 = tx·cri − ty·sri；ty2 = tx2·sri + ty·cri
        gtx2 = gtx2 + gty2 * sri
        g_sri = gty2 * tx2
        g_ty = gty2 * cri
        g_cri = gty2 * typ_
        g_tx = gtx2 * cri
        g_cri = g_cri + gtx2 * txp
        g_ty = g_ty - gtx2 * sri
        g_sri = g_sri - gtx2 * typ_
        g_ang = g_ang - ((-g_cri * sri + g_sri * cri) * PI) * INV180
        # 输入归一化：每个槽位对 C 的梯度写进 GSL[槽位]
        o_slot = _h(I, H_SLOT)
        o_inq = _h(I, H_INQ)
        o_hp = _h(I, H_HPOS)
        o_hn = _h(I, H_HNEG)
        for k in range(K):
            sp = oslot + r * K + k
            si = tl.load(I + o_slot + sp, m, other=0)
            e = oinp + si
            q = tl.load(I + o_inq + e, m, other=-1)
            v0 = tl.load(Cl + q, m & (q >= 0), other=0.0)
            nmax = tl.load(F + _h(I, HF_NMAX) + e, m, other=0.0)
            nmin = tl.load(F + _h(I, HF_NMIN) + e, m, other=0.0)
            pass1 = ~(nmax < v0)
            v1 = tl.where(nmax < v0, nmax, v0)
            pass2 = ~(nmin > v1)
            v2 = tl.where(nmin > v1, nmin, v1)
            pv = v2 - tl.load(F + _h(I, HF_NMID) + e, m, other=0.0)
            hp = tl.load(I + o_hp + e, m, other=0) != 0
            hn = tl.load(I + o_hn + e, m, other=0) != 0
            slope = tl.where(pv > 0, tl.where(hp, tl.load(F + _h(I, HF_RPOS) + e, m, other=0.0), 0.0),
                             tl.where(pv < 0, tl.where(hn, tl.load(F + _h(I, HF_RNEG) + e, m, other=0.0), 0.0),
                                      tl.load(F + _h(I, HF_RMID) + e, m, other=0.0)))
            gc = g_tx * tl.load(F + _h(I, HF_TYPE) + sp * 3 + 0, m, other=0.0) \
                + g_ty * tl.load(F + _h(I, HF_TYPE) + sp * 3 + 1, m, other=0.0) \
                + g_ang * tl.load(F + _h(I, HF_TYPE) + sp * 3 + 2, m, other=0.0)
            gres = (gc * tl.load(F + _h(I, HF_SW1) + e, m, other=0.0)) * tl.load(F + _h(I, HF_SW0) + e, m, other=0.0)
            gv = tl.where(pass1 & pass2 & (q >= 0), gres * slope, 0.0)
            tl.store(GSL + sp, gv, m)
    tl.debug_barrier()
    # ---- C 收集各槽位的梯度（逆表，固定顺序）
    o_ivn = _h(I, H_INVIN)
    Q = _h(I, H_Q)
    DI = _h(I, H_DIN)
    for q0 in range(0, Q, BQ):
        q = q0 + tl.arange(0, BQ)
        m = q < Q
        acc = tl.load(gC + q, m, other=0.0)
        for d in range(DI):
            e = tl.load(I + o_ivn + (l * Q + q) * DI + d, m, other=-1)
            acc = acc + tl.load(GSL + e, m & (e >= 0), other=0.0)
        tl.store(gC + q, acc, m)
    tl.debug_barrier()


@triton.jit
def _bwd_kernel(GOUT, GX, SNAP, TAPE, GST, SCR, SCHN, SCHW, SCHA, SCHR, I, F, DL, MDS, T, B, STLEN, TLEN, SCRLEN, MAXS,
                WX: tl.constexpr, WY: tl.constexpr, GXN: tl.constexpr, GYN: tl.constexpr,
                BQ: tl.constexpr, BR: tl.constexpr, BO: tl.constexpr, BRD: tl.constexpr, BP: tl.constexpr,
                BNP: tl.constexpr):
    """GST：进来是对最终状态的梯度，出去是对初始状态的梯度（就地）。GX：对 X 的梯度。用前向记下的 SNAP / TAPE，不重算前向。"""
    b = tl.program_id(0)
    P = _h(I, H_P)
    Q = _h(I, H_Q)
    NO = _h(I, H_NO)
    NOP = _h(I, H_NOP)
    NP = _h(I, H_NP)
    NS = _h(I, H_NS)
    NL = _h(I, H_NL)
    NOVT = _h(I, H_NOVT)
    NSLOT = _h(I, H_NSLOT)
    SPLEN = 4 * NP + 2 * NS
    SUBLEN = SPLEN + NL * Q + NOVT + NS * NSF + NP * NPF
    g = GST + b.to(tl.int64) * STLEN
    gpx = g
    gpy = g + NP
    gvx = g + 2 * NP
    gvy = g + 3 * NP
    ggx = g + 4 * NP
    ggy = ggx + NS
    gcur = ggy + NS
    gprev = gcur + NO
    gic = gprev + NO
    scr = SCR + b.to(tl.int64) * SCRLEN
    GOX = scr                                 # 3·NOVT
    GOY = GOX + 3 * NOVT
    GSL = GOY + 3 * NOVT                      # NSLOT
    gC = GSL + NSLOT
    gpr = gC + Q
    gvals = gpr + P
    gtgt = gvals + NOVT
    ivals = gtgt + NOP
    o_pmap = _h(I, H_PMAP)
    o_outp = _h(I, H_OUTP)
    o_gid = _h(I, H_OGID)
    o_qg = _h(I, H_QGLOB)
    for tt in range(T):
        t = T - 1 - tt
        snap = SNAP + (t * B + b).to(tl.int64) * STLEN
        tape = TAPE + (t * B + b).to(tl.int64) * TLEN
        wcur = tape + MAXS * SUBLEN
        wprev = wcur + NO
        ns = tl.load(SCHN + t)
        tl.debug_barrier()
        # ---- interpolate 的伴随
        grow = GOUT + (t * B + b).to(tl.int64) * P
        a = tl.load(SCHA + t)
        res = tl.load(SCHR + t)
        for p0 in range(0, P, BP):
            p = p0 + tl.arange(0, BP)
            m = p < P
            gv = tl.load(grow + p, m, other=0.0)
            if res != 0:
                j = tl.load(I + o_pmap + p, m, other=-1)
                gv = tl.where(j >= 0, 0.0, gv)
            tl.store(gpr + p, gv, m)
        if res != 0:
            for j0 in range(0, NOP, BP):
                j = j0 + tl.arange(0, BP)
                m = j < NOP
                p = tl.load(I + o_outp + j, m, other=0)
                tl.store(gtgt + j, tl.load(grow + p, m, other=0.0), m)
            a1 = 1.0 - a
            for j0 in range(0, NO, BP):
                j = j0 + tl.arange(0, BP)
                m = j < NO
                tl.store(ivals + j, tl.load(wprev + j, m, other=0.0) * a1 + tl.load(wcur + j, m, other=0.0) * a, m)
            tl.debug_barrier()
            o_tab = _h(I, H_ITAB)
            o_src = _h(I, H_ISRC)
            o_dst = _h(I, H_IDST)
            o_ful = _h(I, H_IFUL)
            f_scl = _h(I, HF_ISCL)
            f_min = _h(I, HF_IMIN)
            f_max = _h(I, HF_IMAX)
            f_w = _h(I, HF_IW)
            nr = _h(I, H_NIRI)
            for rr_ in range(0, nr):
                rr = nr - 1 - rr_
                st = tl.load(I + o_tab + rr * 2)
                n = tl.load(I + o_tab + rr * 2 + 1)
                for j0 in range(0, n, BRD):
                    j = st + j0 + tl.arange(0, BRD)
                    m = (j0 + tl.arange(0, BRD)) < n
                    src = tl.load(I + o_src + j, m, other=0)
                    dst = tl.load(I + o_dst + j, m, other=0)
                    scl = tl.load(F + f_scl + j, m, other=0.0)
                    v = tl.load(ivals + src, m, other=0.0) * scl
                    vmin = tl.load(F + f_min + j, m, other=0.0)
                    vmax = tl.load(F + f_max + j, m, other=0.0)
                    w = tl.load(F + f_w + j, m, other=0.0)
                    full = tl.load(I + o_ful + j, m, other=0) != 0
                    gv = tl.load(gtgt + dst, m, other=0.0)
                    tl.store(gtgt + dst, tl.where(full, 0.0, gv * (1.0 - w)), m)
                    gcl = tl.where(full, gv, gv * w)
                    pas = (~(v < vmin)) & (~(v > vmax))
                    gs = tl.where(pas, gcl * scl, 0.0)
                    tl.store(gprev + src, tl.load(gprev + src, m, other=0.0) + gs * a1, m)
                    tl.store(gcur + src, tl.load(gcur + src, m, other=0.0) + gs * a, m)
                tl.debug_barrier()
            for j0 in range(0, NOP, BP):
                j = j0 + tl.arange(0, BP)
                m = j < NOP
                p = tl.load(I + o_outp + j, m, other=0)
                tl.store(gpr + p, tl.load(gpr + p, m, other=0.0) + tl.load(gtgt + j, m, other=0.0), m)
        tl.debug_barrier()
        # ---- 子步（逆序）
        for kk in range(ns):
            k = ns - 1 - kk
            sub = tape + k * SUBLEN
            old = snap
            if k > 0:
                old = tape + (k - 1) * SUBLEN
            # cur 的伴随：输出写进 cur[gidx]；prev ← cur（子步开始时）
            for o0 in range(0, NOVT, BO):
                o = o0 + tl.arange(0, BO)
                m = o < NOVT
                gid = tl.load(I + o_gid + o, m, other=0)
                tl.store(gvals + o, tl.load(gcur + gid, m, other=0.0), m)
            tl.debug_barrier()
            for o0 in range(0, NOVT, BO):
                o = o0 + tl.arange(0, BO)
                m = o < NOVT
                gid = tl.load(I + o_gid + o, m, other=0)
                tl.store(gcur + gid, tl.zeros([BO], tl.float64), m)
            tl.debug_barrier()
            for j0 in range(0, NO, BQ):
                j = j0 + tl.arange(0, BQ)
                m = j < NO
                tl.store(gcur + j, tl.load(gcur + j, m, other=0.0) + tl.load(gprev + j, m, other=0.0), m)
                tl.store(gprev + j, tl.zeros([BQ], tl.float64), m)
            for q0 in range(0, Q, BQ):
                q = q0 + tl.arange(0, BQ)
                tl.store(gC + q, tl.zeros([BQ], tl.float64), q < Q)
            tl.debug_barrier()
            ts = sub + SPLEN + NL * Q + NOVT
            for ll in range(NL):
                l = NL - 1 - ll
                _bwd_level(I, F, DL, MDS, sub + SPLEN + l * Q, old, old + NP, old + 2 * NP, old + 3 * NP, old + 4 * NP,
                           old + 4 * NP + NS, sub, sub + NP, sub + SPLEN + NL * Q, ts, ts + NS * NSF,
                           gpx, gpy, gvx, gvy, ggx, ggy, gC, gvals, GOX, GOY, GSL, l,
                           WX, WY, GXN, GYN, BQ, BR, BO, BRD, BNP)
            # C0 = f32(ic·(1−w) + pq·w)，ic_new = C0
            w = tl.load(SCHW + t * MAXS + k)
            w1 = 1.0 - w
            for q0 in range(0, Q, BQ):
                q = q0 + tl.arange(0, BQ)
                m = q < Q
                gc0 = tl.load(gC + q, m, other=0.0) + tl.load(gic + q, m, other=0.0)
                tl.store(gic + q, gc0 * w1, m)
                pg = tl.load(I + o_qg + q, m, other=0)
                tl.store(gpr + pg, tl.load(gpr + pg, m, other=0.0) + gc0 * w, m)
            tl.debug_barrier()
        _copy(GX + (t * B + b).to(tl.int64) * P, gpr, P, BP)


# ---------------------------------------------------------------- 主机端：常量表、调度、autograd

class Plan:
    """由 PhysicsRig 建好的常量表（int32 表 I、float64 表 F）与核的块大小。只支持 float64 rig、split_by_length=False。"""

    BR = 64
    BO = 64
    BRD = 64
    BQ = 128
    BP = 128
    BST = 256
    BNP = 128

    def __init__(self, rig):
        if rig.dtype != torch.float64 or rig.split_by_length or any(len(g) != 1 for g, _ in rig.stages):
            raise ValueError('fused 路径只支持 float64、split_by_length=False 的 rig')
        dev = rig.device
        self.rig = rig
        self.emu = bool(rig.emulate_f32)
        NL = len(rig.levels)
        meta = []
        slot, inq, hpos, hneg = [], [], [], []
        fin = {k: [] for k in ('NMIN', 'NMAX', 'NMID', 'RPOS', 'RNEG', 'MNORM', 'RMID', 'SW0', 'SW1')}
        ftype, fconst, acc, radi, mob, thr = [], [], [], [], [], []
        oi1, opar, otyp, ogid, osgn = [], [], [], [], []
        rtab, rsrc, rdst, rful, rscl, rmin, rmax, rw = [], [], [], [], [], [], [], []
        onp = oset = oslot = oinp = oout = 0
        c = lambda t: t.detach().double().cpu().reshape(-1).tolist()
        ci = lambda t: [int(x) for x in t.detach().long().cpu().reshape(-1).tolist()]
        for l, (lv, (gids, rounds)) in enumerate(zip(rig.levels, rig.stages)):
            R, L, K = len(lv.rigs), lv.L, lv.K
            NI = len(lv.in_q)
            NOV = len(lv.out_gidx) if lv.out_vertex_ok else 0
            meta.append([R, L, K, NI, onp, oset, oslot, oinp, NOV, oout, len(rtab) // 2, len(rounds)] + [0] * 4)
            slot += ci(lv.in_slot)
            inq += ci(lv.in_q) + [-1]
            hpos += ci(lv.has_pos)
            hneg += ci(lv.has_neg)
            # r_mid：与 physics._inputs 完全相同的 GPU 运算（逐位相同）
            rmid = torch.where(lv.has_pos, lv.r_pos, 0.0) + torch.where(lv.has_neg, lv.r_neg, 0.0)
            rmid = rmid / torch.where(lv.has_pos & lv.has_neg, 2.0, 1.0)
            for k_, t_ in (('NMIN', lv.n_min), ('NMAX', lv.n_max), ('NMID', lv.n_mid), ('RPOS', lv.r_pos),
                           ('RNEG', lv.r_neg), ('MNORM', lv.m_norm), ('RMID', rmid), ('SW0', lv.in_sw[:, 0]),
                           ('SW1', lv.in_sw[:, 1])):
                fin[k_] += c(t_)
            ftype += c(lv.in_type)
            fconst += c(lv.in_const)
            acc += c(lv.acc)
            radi += c(lv.radius)
            mob += c(lv.mobility)
            thr += c(lv.thr)
            if NOV:
                oi1 += [onp + x for x in ci(lv.out_flat)]
                opar += ci(lv.out_parent)
                otyp += ci(lv.out_type)
                ogid += ci(lv.out_gidx)
                osgn += c(lv.out_sign)
            for rd in rounds:
                n = len(rd['src'])
                rtab += [len(rsrc), n]
                rsrc += [oout + x for x in ci(rd['src'])]
                rdst += ci(rd['dst'])
                rful += ci(rd['full'])
                rscl += c(rd['scale'])
                rmin += c(rd['vmin'])
                rmax += c(rd['vmax'])
                rw += c(rd['w'])
            onp += R * L
            oset += R
            oslot += R * K
            oinp += NI + 1
            oout += NOV
        NP, NS, NSLOT, NOVT = onp, oset, oslot, oout
        itab, isrc, idst, iful, iscl, imin, imax, iw = [], [], [], [], [], [], [], []
        for rd in rig.interp_rounds:
            n = len(rd['src'])
            itab += [len(isrc), n]
            isrc += ci(rd['src'])
            idst += ci(rd['dst'])
            iful += ci(rd['full'])
            iscl += c(rd['scale'])
            imin += c(rd['vmin'])
            imax += c(rd['vmax'])
            iw += c(rd['w'])
        P, Q, NO = rig.P, rig.Q, rig.NO
        outp = list(rig.output_params)
        pmap = [-1] * P
        for j, p in enumerate(outp):
            pmap[p] = j
        # 逆表 1：每层、每个 q → 读它的槽位（全局槽位号，升序）
        inv_in = [[[] for _ in range(Q)] for _ in range(NL)]
        for l, m_ in enumerate(meta):
            R, L, K, NI, _, _, osl, oin = m_[:8]
            for sp in range(osl, osl + R * K):
                s = slot[sp]
                if s < NI:
                    inv_in[l][inq[oin + s]].append(sp)
        DIN = max([1] + [len(x) for lv_ in inv_in for x in lv_])
        invin = [x[d] if d < len(x) else -1 for lv_ in inv_in for x in lv_ for d in range(DIN)]
        # 逆表 2：每个粒子 → 用到它的 (输出, 角色)：0 = 本顶点，1 = 上一顶点，2 = 父方向的再上一顶点
        inv_out = [[] for _ in range(NP)]
        for o, (g1, par) in enumerate(zip(oi1, opar)):
            inv_out[g1].append(o * 3 + 0)
            inv_out[g1 - 1].append(o * 3 + 1)
            inv_out[g1 - 2 if par else g1 - 1].append(o * 3 + 2)
        DOUT = max([1] + [len(x) for x in inv_out])
        invout = [x[d] if d < len(x) else -1 for x in inv_out for d in range(DOUT)]
        segs_i = dict(QGLOB=list(rig.q_glob), OUTP=outp, PMAP=pmap, META=[x for m_ in meta for x in m_], SLOT=slot,
                      INQ=inq, HPOS=hpos, HNEG=hneg, OI1=oi1, OPAR=opar, OTYP=otyp, OGID=ogid, RTAB=rtab, RSRC=rsrc,
                      RDST=rdst, RFUL=rful, ITAB=itab, ISRC=isrc, IDST=idst, IFUL=iful, NZ=[0] * NP, INVIN=invin,
                      INVOUT=invout)
        segs_f = dict(fin, TYPE=ftype, CONST=fconst, ACC=acc, RADI=radi, MOB=mob, THR=thr, OSGN=osgn, RSCL=rscl,
                      RMIN=rmin, RMAX=rmax, RW=rw, ISCL=iscl, IMIN=imin, IMAX=imax, IW=iw)
        hdr = [0] * _NH
        body_i = []
        off = _NH
        for k_ in _HDR:
            if k_ in segs_i:
                hdr[_HDR.index(k_)] = off
                body_i += segs_i[k_]
                off += len(segs_i[k_])
        body_f = []
        off = 0
        for k_ in _FHDR:
            hdr[len(_HDR) + _FHDR.index(k_)] = off
            body_f += segs_f[k_]
            off += len(segs_f[k_])
        for k_, v in dict(HASC=int(rig.has_const), P=P, Q=Q, NO=NO, NOP=len(outp), NP=NP, NS=NS, NL=NL, NOVT=NOVT,
                          NSLOT=NSLOT, DIN=DIN, DOUT=DOUT, NIRI=len(rig.interp_rounds)).items():
            hdr[_HDR.index(k_)] = v
        self.I = torch.tensor(hdr + body_i, dtype=torch.int32, device=dev)
        self.F = torch.tensor(body_f, dtype=torch.float64, device=dev)
        self._nz_off = hdr[_HDR.index('NZ')]
        self.P, self.Q, self.NO, self.NOP, self.NP, self.NS, self.NL = P, Q, NO, len(outp), NP, NS, NL
        self.NOVT, self.NSLOT = NOVT, NSLOT
        self.STLEN = 4 * NP + 2 * NS + 2 * NO + Q
        self.level_shapes = [(len(lv.rigs), lv.L) for lv in rig.levels]
        self.WX, self.WY = rig.opt_wind
        self.GXN, self.GYN = -rig.opt_gravity[0], -rig.opt_gravity[1]
        self._dl: dict = {}
        self._sch: dict = {}

    def delays(self, phys_dt: float):
        """(DL, DS, MDS, I)：每个粒子的 delay·dt·30、除数（就是 physics._delays 的结果）、反向用的 mobility / 除数，
        以及写好 delay ≠ 0 标志的 I 表。"""
        v = self._dl.get(phys_dt)
        if v is None:
            rig = self.rig
            ds = rig._delays_fixed or [rig._delays(lv, phys_dt) for lv in rig.levels]
            DL = torch.cat([d[0].reshape(-1) for d in ds]).contiguous()
            NZ = torch.cat([d[1].reshape(-1) for d in ds]).to(torch.int32)
            DS = torch.cat([d[2].reshape(-1) for d in ds]).contiguous()
            MDS = (torch.cat([lv.mobility.reshape(-1) for lv in rig.levels]) / DS).contiguous()
            I = self.I.clone()
            I[self._nz_off:self._nz_off + self.NP] = NZ
            v = (DL, DS, MDS, I)
            if len(self._dl) < 64:
                self._dl[phys_dt] = v
        return v

    def schedule(self, sched):
        """每帧 (inputWeights, α, resolved) → 设备上的 (子步数 [T], inputWeight [T, MAXS], α [T], resolved [T], MAXS)。按内容缓存。"""
        key = tuple(sched)
        v = self._sch.get(key)
        if v is None:
            T = len(sched)
            MAXS = max([1] + [len(ws) for ws, _, _ in sched])
            dev = self.rig.device
            w = [list(ws) + [0.0] * (MAXS - len(ws)) for ws, _, _ in sched]
            v = (torch.tensor([len(ws) for ws, _, _ in sched], dtype=torch.int32, device=dev),
                 torch.tensor(w, dtype=torch.float64, device=dev).reshape(T, MAXS),
                 torch.tensor([a for _, a, _ in sched], dtype=torch.float64, device=dev),
                 torch.tensor([int(r) for _, _, r in sched], dtype=torch.int32, device=dev), MAXS)
            if len(self._sch) < 256:
                self._sch[key] = v
        return v

    # ---- 状态打包 / 拆包（[B, STLEN]：px、py、vx、vy、gx、gy、cur、prev、ic）
    def pack(self, st, ic):
        f = lambda ts: [t.reshape(t.shape[0], -1) for t in ts]
        return torch.cat(f(st.px) + f(st.py) + f(st.vx) + f(st.vy) + list(st.gx) + list(st.gy) + [st.cur, st.prev, ic], 1)

    def unpack(self, S):
        B = S.shape[0]
        o = 0
        out = []
        for _ in range(4):
            lst = []
            for R, L in self.level_shapes:
                lst.append(S[:, o:o + R * L].view(B, R, L))
                o += R * L
            out.append(tuple(lst))
        for _ in range(2):
            lst = []
            for R, L in self.level_shapes:
                lst.append(S[:, o:o + R])
                o += R
            out.append(tuple(lst))
        cur = S[:, o:o + self.NO]
        prev = S[:, o + self.NO:o + 2 * self.NO]
        ic = S[:, o + 2 * self.NO:o + 2 * self.NO + self.Q]
        return out, cur, prev, ic

    def kw(self):
        return dict(EMU=self.emu, WX=self.WX, WY=self.WY, GXN=self.GXN, GYN=self.GYN, BQ=self.BQ, BR=self.BR,
                    BO=self.BO, BRD=self.BRD, BP=self.BP, BST=self.BST)


    def tape_len(self, MAXS: int) -> int:
        """每帧、每个批元素的磁带长度（float64 个数）。"""
        SPLEN = 4 * self.NP + 2 * self.NS
        return MAXS * (SPLEN + self.NL * self.Q + self.NOVT + self.NS * NSF.value + self.NP * NPF.value) + 2 * self.NO


def fused_forward(plan: Plan, X, S0, sched_dev, phys_dt, save: bool):
    """X [T, B, P]、S0 [B, STLEN]（float64、连续）→ (OUT [T, B, P], S [B, STLEN], SNAP, TAPE)；不求导时后两个为 None。"""
    T, B, P = X.shape
    ns, w, a, res, MAXS = sched_dev
    DL, DS, MDS, I = plan.delays(phys_dt)
    S = S0.clone()
    OUT = torch.empty_like(X)
    TLEN = plan.tape_len(MAXS)
    if save:
        SNAP = torch.empty(T, B, plan.STLEN, dtype=torch.float64, device=X.device)
        TAPE = torch.empty(T, B, TLEN, dtype=torch.float64, device=X.device)
    else:
        SNAP = TAPE = S
    SCRLEN = plan.Q + plan.NOVT + plan.NO + plan.NOP
    SCR = torch.empty(B, SCRLEN, dtype=torch.float64, device=X.device)
    _fwd_kernel[(B,)](X, OUT, S, SNAP, TAPE, SCR, ns, w, a, res, I, plan.F, DL, DS, T, B, plan.STLEN, TLEN, SCRLEN,
                      MAXS, SAVE=save, **plan.kw(), **_OPTS)
    return OUT, S, (SNAP if save else None), (TAPE if save else None)


def fused_backward(plan: Plan, SNAP, TAPE, GOUT, GS, sched_dev, phys_dt):
    """反向核：返回 (对 X 的梯度 [T, B, P], 对初始状态的梯度 [B, STLEN])。"""
    T, B, P = GOUT.shape
    ns, w, a, res, MAXS = sched_dev
    DL, DS, MDS, I = plan.delays(phys_dt)
    GST = GS.clone()
    GX = torch.empty_like(GOUT)
    SCRLEN = 6 * plan.NOVT + plan.NSLOT + plan.Q + plan.P + plan.NOVT + plan.NOP + plan.NO
    SCR = torch.empty(B, SCRLEN, dtype=torch.float64, device=GOUT.device)
    kw = plan.kw()
    kw.pop('EMU')
    kw.pop('BST')
    _bwd_kernel[(B,)](GOUT, GX, SNAP, TAPE, GST, SCR, ns, w, a, res, I, plan.F, DL, MDS, T, B, plan.STLEN,
                      plan.tape_len(MAXS), SCRLEN, MAXS, BNP=plan.BNP, **kw, **_OPTS)
    return GX, GST


class FusedRollout(torch.autograd.Function):
    """(X [T, B, P], S0 [B, STLEN]) → (OUT [T, B, P], S [B, STLEN])；前向一个核（求导时顺带记磁带），反向一个核（确定性）。"""

    @staticmethod
    def forward(ctx, plan, sched_dev, phys_dt, X, S0):
        save = any(ctx.needs_input_grad[3:5])
        OUT, S, SNAP, TAPE = fused_forward(plan, X.contiguous(), S0.contiguous(), sched_dev, phys_dt, save)
        if save:
            ctx.save_for_backward(SNAP, TAPE)
            ctx.plan, ctx.sched_dev, ctx.phys_dt = plan, sched_dev, phys_dt
        return OUT, S

    @staticmethod
    def backward(ctx, gOUT, gS):
        SNAP, TAPE = ctx.saved_tensors
        T, B = SNAP.shape[:2]
        dev = SNAP.device
        gOUT = torch.zeros(T, B, ctx.plan.P, dtype=torch.float64, device=dev) if gOUT is None else gOUT.contiguous()
        gS = torch.zeros(B, ctx.plan.STLEN, dtype=torch.float64, device=dev) if gS is None else gS.contiguous()
        GX, GST = fused_backward(ctx.plan, SNAP, TAPE, gOUT, gS, ctx.sched_dev, ctx.phys_dt)
        return None, None, None, (GX if ctx.needs_input_grad[3] else None), (GST if ctx.needs_input_grad[4] else None)
