"""可微分的图层合成器：逐像素对齐 packages/runtime/src/compositor.ts 与 shaders.ts（MSAA 0）。

每个图层：图层缓冲（预乘 RGBA，合成分辨率）→ filters（按数组顺序：blur / motionBlur / resample / color）
→ 输出 pass（圆角矩形羽化 clip × opacity）→ 按 blendMode（Pixi 预乘混合）混入合成缓冲。图层从下到上依次合成。

张量约定：[B, H, W, 4]，float，0..1，预乘 alpha，行从上到下（与 runtime 读回一致）。
runtime 的每张中间缓冲都是 RGBA8：quantize=True 时每个 pass 的输出按 8 位取整（直通估计，梯度当恒等），
前向与 runtime 逐像素对齐；quantize=False 时全程浮点（拟合时可用）。

参数（与 runtime 的 ResolvedFilter / LayerDrawParams 同名同义，见 a3_dump 的 draw 字段）：
  blur        {sigma}                         可分离高斯，半径 ceil(3σ)，整数采样，缓冲外透明，按全部权重归一化
  motionBlur  {angle, length}                 沿 (cos, sin)（y 向下、顺时针为正）取 max(2, ceil(L)+1) 个双线性样本平均
  resample    {factor, method}                factor ∈ (0,1) 才处理：k×k 区域平均缩小到 round(W·f)×round(H·f)，
                                              再按像素中心对齐放大（bilinear / Catmull-Rom bicubic，越界钳制）
  color       {brightness, contrast, saturation, multiply, offset}  在反预乘颜色上做，夹到 [0,1] 再预乘
  clip        {x, y, width, height, radius, feather}  覆盖率 = clamp(0.5 − d/max(feather,1), 0, 1)
  opacity, blendMode ∈ {normal, add, multiply, screen}
标量参数可以是 Python 数或张量（形状 [] 或 [B]），张量可求导。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# A7：求导时每个滤镜 / 输出 pass 用 torch.utils.checkpoint（只保存输入，反向时重算）。前向、梯度都不变，
# 省掉滤镜的中间量（640×360 下 color + resample + blur 每帧约 45 MB）。缺省关闭，也可按调用传 checkpoint_filters。
CHECKPOINT = False
CHECKPOINT_FRAMES = 4  # 分帧检查点：反向重算时每次展开的帧数


def _ck(on, fn, *args):
    if on and torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

class _Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x.clamp(0, 1) * 255.0 + 0.5) / 255.0

    @staticmethod
    def backward(ctx, g):
        return g


def quantize8(x: torch.Tensor) -> torch.Tensor:
    """RGBA8 渲染目标的写入：夹到 [0,1] 后按 8 位取整（梯度直通）。"""
    return _Quant.apply(x)


def _q(x, on: bool):
    return quantize8(x) if on else x


def _scalar(v, B: int, ref: torch.Tensor) -> torch.Tensor:
    """标量或 [B] 张量 → [B, 1, 1, 1]。"""
    t = v if torch.is_tensor(v) else torch.tensor(float(v))
    t = t.to(device=ref.device, dtype=ref.dtype)
    return t.reshape(-1, 1, 1, 1).expand(B, 1, 1, 1) if t.numel() == 1 else t.reshape(B, 1, 1, 1)


def _val(v) -> float:
    """取标量值（用于决定半径、采样数等离散结构）。"""
    if torch.is_tensor(v):
        return float(v.detach().reshape(-1)[0].item()) if v.numel() == 1 else float(v.detach().max().item())
    return float(v)


def _js_round(x: float) -> int:
    return int(math.floor(x + 0.5))


def bilinear(img: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """GL LINEAR + CLAMP_TO_EDGE 采样。img [B,H,W,C]；x、y 为像素坐标（纹素中心在 i+0.5），形状 [B, ...]。返回 [B, ..., C]。"""
    B, H, W, C = img.shape
    tx = (x - 0.5).clamp(0, W - 1)
    ty = (y - 0.5).clamp(0, H - 1)
    x0 = tx.detach().floor().clamp(max=max(W - 2, 0))
    y0 = ty.detach().floor().clamp(max=max(H - 2, 0))
    fx = (tx - x0)[..., None]
    fy = (ty - y0)[..., None]
    x0l, y0l = x0.long(), y0.long()
    x1l = (x0l + 1).clamp(max=W - 1)
    y1l = (y0l + 1).clamp(max=H - 1)
    flat = img.reshape(B, H * W, C)
    shp = x.shape

    def g(yy, xx):
        i = (yy * W + xx).reshape(B, -1)
        return torch.gather(flat, 1, i[..., None].expand(-1, -1, C)).reshape(*shp, C)

    return (g(y0l, x0l) * (1 - fx) + g(y0l, x1l) * fx) * (1 - fy) + (g(y1l, x0l) * (1 - fx) + g(y1l, x1l) * fx) * fy


def _pix_grid(B, H, W, ref):
    ys = torch.arange(H, device=ref.device, dtype=ref.dtype) + 0.5
    xs = torch.arange(W, device=ref.device, dtype=ref.dtype) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gx[None].expand(B, -1, -1), gy[None].expand(B, -1, -1)


# ---------------------------------------------------------------------------
# 滤镜
# ---------------------------------------------------------------------------

def blur(img: torch.Tensor, sigma, quantize=True) -> torch.Tensor:
    """BLUR_FRAG：两遍（先横后竖），每遍输出写入 RGBA8 缓冲。"""
    s = _val(sigma)
    if not (s > 0):
        return img
    B, H, W, C = img.shape
    r = int(math.ceil(3.0 * s))
    sg = sigma if torch.is_tensor(sigma) else torch.tensor(s)
    sg = sg.to(device=img.device, dtype=img.dtype).reshape(-1)[:1]
    i = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    w = torch.exp(-0.5 * i * i / (sg * sg))
    w = w / w.sum()
    x = img.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    x = F.conv2d(x, w.reshape(1, 1, 1, -1), padding=(0, r))
    x = _q(x, quantize)
    x = F.conv2d(x, w.reshape(1, 1, -1, 1), padding=(r, 0))
    x = _q(x, quantize)
    return x.reshape(B, C, H, W).permute(0, 2, 3, 1)


def motion_blur(img: torch.Tensor, angle, length, quantize=True) -> torch.Tensor:
    L = _val(length)
    if not (L > 0):
        return img
    B, H, W, C = img.shape
    n = max(2, int(math.ceil(L)) + 1)
    a = _scalar(angle, B, img)[..., 0] * (math.pi / 180.0)  # [B,1,1]
    Lt = _scalar(length, B, img)[..., 0]
    dx, dy = torch.cos(a), torch.sin(a)
    gx, gy = _pix_grid(B, H, W, img)
    acc = torch.zeros_like(img)
    for k in range(n):
        t = -0.5 * Lt + Lt * (k / (n - 1))
        qx, qy = gx + dx * t, gy + dy * t
        inside = ((qx >= 0) & (qy >= 0) & (qx <= W) & (qy <= H))[..., None]
        acc = acc + torch.where(inside, bilinear(img, qx, qy), torch.zeros_like(img))
    return _q(acc / n, quantize)


def resample(img: torch.Tensor, factor, method="bilinear", quantize=True) -> torch.Tensor:
    f = _val(factor)
    if not (0 < f < 1):
        return img
    B, H, W, C = img.shape
    sw, sh = max(1, _js_round(W * f)), max(1, _js_round(H * f))
    k = min(8, int(math.ceil(1.0 / f - 1e-9)))
    # 1) 区域平均缩小（RESAMPLE_FRAG，与缩放读回同一着色器）
    cx, cy = W / sw, H / sh
    oy, ox = torch.meshgrid(torch.arange(sh, device=img.device, dtype=img.dtype), torch.arange(sw, device=img.device, dtype=img.dtype), indexing="ij")
    acc = 0
    for j in range(k):
        for i in range(k):
            qx = (ox + (i + 0.5) / k) * cx
            qy = (oy + (j + 0.5) / k) * cy
            acc = acc + bilinear(img, qx[None].expand(B, -1, -1), qy[None].expand(B, -1, -1))
    small = _q(acc / (k * k), quantize)
    # 2) 放大（UPSAMPLE_FRAG）：q = (p + 0.5)·small/out − 0.5
    py, px = torch.meshgrid(torch.arange(H, device=img.device, dtype=img.dtype), torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    qx = (px + 0.5) * sw / W - 0.5
    qy = (py + 0.5) * sh / H - 0.5
    bx, by = torch.floor(qx), torch.floor(qy)
    tx, ty = (qx - bx)[None, ..., None], (qy - by)[None, ..., None]
    ix, iy = bx.long(), by.long()
    flat = small.reshape(B, sh * sw, C)

    def tap(dy, dx):
        yy = (iy + dy).clamp(0, sh - 1)
        xx = (ix + dx).clamp(0, sw - 1)
        i = (yy * sw + xx).reshape(1, -1, 1).expand(B, -1, C)
        return torch.gather(flat, 1, i).reshape(B, H, W, C)

    if method != "bicubic":
        a = tap(0, 0) * (1 - tx) + tap(0, 1) * tx
        c = tap(1, 0) * (1 - tx) + tap(1, 1) * tx
        out = a * (1 - ty) + c * ty
    else:
        def cr(x):
            x = x.abs()
            return torch.where(x < 1, 1.5 * x ** 3 - 2.5 * x ** 2 + 1, torch.where(x < 2, -0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2, torch.zeros_like(x)))
        out = 0
        for j in range(-1, 3):
            wy = cr(j - ty)
            for i in range(-1, 3):
                out = out + cr(i - tx) * wy * tap(j, i)
        out = out.clamp(0, 1)
        out = torch.cat([torch.minimum(out[..., :3], out[..., 3:4]), out[..., 3:4]], -1)
    return _q(out, quantize)


def color(img: torch.Tensor, brightness=0.0, contrast=1.0, saturation=1.0, multiply=(1, 1, 1), offset=(0, 0, 0), quantize=True) -> torch.Tensor:
    B = img.shape[0]
    a = img[..., 3:4]
    ok = a > 0
    c = img[..., :3] / torch.where(ok, a, torch.ones_like(a))
    c = (c - 0.5) * _scalar(contrast, B, img) + 0.5 + _scalar(brightness, B, img)
    lw = torch.tensor([0.2126, 0.7152, 0.0722], device=img.device, dtype=img.dtype)
    l = (c * lw).sum(-1, keepdim=True)
    c = l + (c - l) * _scalar(saturation, B, img)
    mul = multiply if torch.is_tensor(multiply) else torch.tensor(multiply)
    off = offset if torch.is_tensor(offset) else torch.tensor(offset)
    mul = mul.to(img.device, img.dtype).reshape(-1, 1, 1, 3)
    off = off.to(img.device, img.dtype).reshape(-1, 1, 1, 3)
    c = (c * mul + off).clamp(0, 1)
    out = torch.where(ok, torch.cat([c * a, a], -1), torch.zeros_like(img))
    return _q(out, quantize)


def apply_filters(img: torch.Tensor, filters: list[dict], quantize=True, checkpoint_filters: bool | None = None) -> torch.Tensor:
    ck = CHECKPOINT if checkpoint_filters is None else checkpoint_filters
    for f in filters:
        t = f["type"]
        if t == "blur":
            img = _ck(ck, blur, img, f["sigma"], quantize)
        elif t == "motionBlur":
            img = _ck(ck, motion_blur, img, f["angle"], f["length"], quantize)
        elif t == "resample":
            img = _ck(ck, resample, img, f["factor"], f.get("method", "bilinear"), quantize)
        elif t == "color":
            img = _ck(ck, color, img, f.get("brightness", 0), f.get("contrast", 1), f.get("saturation", 1), f.get("multiply", (1, 1, 1)),
                      f.get("offset", (0, 0, 0)), quantize)
        else:
            raise ValueError(f"未知滤镜 {t}")
    return img


# ---------------------------------------------------------------------------
# clip / opacity / 混合
# ---------------------------------------------------------------------------

def clip_coverage(clip: dict, B: int, H: int, W: int, ref: torch.Tensor) -> torch.Tensor:
    """OUTPUT_FRAG 的圆角矩形羽化覆盖率 [B, H, W, 1]。"""
    gx, gy = _pix_grid(B, H, W, ref)
    s = lambda k: _scalar(clip[k], B, ref)[..., 0]
    x, y, w, h, rad, fe = s("x"), s("y"), s("width"), s("height"), s("radius"), s("feather")
    hx, hy = 0.5 * w, 0.5 * h
    r = torch.minimum(rad.clamp_min(0), torch.minimum(hx, hy))
    qx = (gx - (x + hx)).abs() - hx + r
    qy = (gy - (y + hy)).abs() - hy + r
    d = torch.sqrt(qx.clamp_min(0) ** 2 + qy.clamp_min(0) ** 2 + 1e-30) * ((qx > 0) | (qy > 0)) + torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx)) - r
    return (0.5 - d / fe.clamp_min(1.0)).clamp(0, 1)[..., None]


def blend(dst: torch.Tensor, src: torch.Tensor, mode: str) -> torch.Tensor:
    """Pixi v8 预乘混合（mapWebGLBlendModesToPixi）。"""
    sa = src[..., 3:4]
    if mode == "normal":
        out = src + dst * (1 - sa)
    elif mode == "add":
        out = src + dst
    elif mode == "multiply":  # [DST_COLOR, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA]
        rgb = src[..., :3] * dst[..., :3] + dst[..., :3] * (1 - sa)
        out = torch.cat([rgb, sa + dst[..., 3:4] * (1 - sa)], -1)
    elif mode == "screen":  # [ONE, ONE_MINUS_SRC_COLOR, ONE, ONE_MINUS_SRC_ALPHA]
        rgb = src[..., :3] + dst[..., :3] * (1 - src[..., :3])
        out = torch.cat([rgb, sa + dst[..., 3:4] * (1 - sa)], -1)
    else:
        raise ValueError(f"未知 blendMode {mode}")
    return out.clamp(0, 1)


def output_layer(comp: torch.Tensor, src: torch.Tensor, draw: dict, quantize=True) -> torch.Tensor:
    """输出 pass：src × (opacity × clip 覆盖率)，按 blendMode 混入 comp。"""
    B, H, W, _ = src.shape
    a = _scalar(draw.get("opacity", 1.0), B, src)
    if draw.get("clip"):
        a = a * clip_coverage(draw["clip"], B, H, W, src)
    return _q(blend(comp, src * a, draw.get("blendMode", "normal")), quantize)


def parse_color(c) -> tuple[float, float, float, float]:
    """compositor.ts parseColor 的子集：None/transparent/#rgb/#rrggbb/#rrggbbaa → 预乘 RGBA。"""
    if c is None or c == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    s = c.strip()
    if s.startswith("#"):
        h = s[1:]
        if len(h) in (3, 4):
            v = [int(ch, 16) / 15 for ch in h]
        else:
            v = [int(h[i:i + 2], 16) / 255 for i in range(0, len(h), 2)]
        r, g, b = v[:3]
        a = v[3] if len(v) > 3 else 1.0
        return (r * a, g * a, b * a, a)
    raise ValueError(f"不支持的颜色：{c}")


def _per_frame(d) -> bool:
    """draw 参数里是否有逐帧（多于一个元素）的张量。"""
    if torch.is_tensor(d):
        return d.numel() > 1
    if isinstance(d, dict):
        return any(_per_frame(v) for v in d.values())
    if isinstance(d, (list, tuple)):
        return any(_per_frame(v) for v in d)
    return False


def _layer_pass(comp, buf, draw, quantize):
    src = apply_filters(_q(buf, quantize), draw.get("filters", []), quantize, False)
    return output_layer(comp, src, draw, quantize)


def composite(layers: list[tuple[torch.Tensor, dict]], H: int, W: int, background=None, quantize=True, B: int | None = None,
              checkpoint_filters: bool | None = None) -> torch.Tensor:
    """layers：[(图层缓冲 [B,H,W,4], draw 参数)]，按从下到上的顺序；draw 含 filters / clip / opacity / blendMode。
    返回合成缓冲 [B,H,W,4]。checkpoint_filters：见模块变量 CHECKPOINT（None = 用它）。"""
    ck = CHECKPOINT if checkpoint_filters is None else checkpoint_filters
    if B is None:
        B = layers[0][0].shape[0] if layers else 1
    ref = layers[0][0] if layers else torch.zeros(1)
    bg = torch.tensor(parse_color(background), device=ref.device, dtype=torch.float32)
    comp = _q(bg.reshape(1, 1, 1, 4).expand(B, H, W, 4).clone(), quantize)
    for buf, draw in layers:
        if ck and torch.is_grad_enabled() and B > CHECKPOINT_FRAMES and not _per_frame(draw):
            # 分帧检查点：反向重算时一次只展开 CHECKPOINT_FRAMES 帧的滤镜中间量（各帧互不相关）。
            # 有逐帧参数时不分：blur 半径、motionBlur 采样数取整批的最大值，分开算会改变结果
            comp = torch.cat([_ck(True, _layer_pass, comp[i:i + CHECKPOINT_FRAMES], buf[i:i + CHECKPOINT_FRAMES], draw, quantize)
                              for i in range(0, B, CHECKPOINT_FRAMES)], 0)
        else:
            comp = _ck(ck, _layer_pass, comp, buf, draw, quantize)
    return comp
