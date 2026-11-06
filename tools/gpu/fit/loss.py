"""可微分的目标函数项。

ssim_map：与 tools/qa/metric.py（skimage.structural_similarity, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
  data_range=255, K1=0.01, K2=0.03）逐项相同：高斯窗截断 3.5σ（11 抽头），边界按 scipy 的 'reflect'（含边缘像素的对称延拓），
  RGB 三通道 SSIM 图取平均。tests/test_a5_loss.py 用 metric.py 对拍。
metric_ssim：metric.py 的口径——渲染（预乘）按 alpha 合成到参考帧上，在角色遮罩（dilate(α > 0.1) − 排除区）内取 SSIM 图均值。
edge_loss：边缘项（模糊亮度的梯度幅值差，Charbonnier），见 fit.md。
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

K1, K2, RANGE = 0.01, 0.03, 255.0
C1, C2 = (K1 * RANGE) ** 2, (K2 * RANGE) ** 2


def gauss_kernel(sigma: float, truncate: float = 3.5, device="cuda", dtype=torch.float32) -> torch.Tensor:
    r = int(truncate * sigma + 0.5)
    i = torch.arange(-r, r + 1, device=device, dtype=torch.float64)
    w = torch.exp(-0.5 * (i / sigma) ** 2)
    return (w / w.sum()).to(dtype)


def _sym_pad(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """scipy.ndimage mode='reflect'：d c b a | a b c d | d c b a（r 不超过尺寸时）。"""
    n = x.shape[dim]
    if r <= n:
        lo = x.narrow(dim, 0, r).flip(dim)
        hi = x.narrow(dim, n - r, r).flip(dim)
        return torch.cat([lo, x, hi], dim)
    idx = torch.arange(-r, n + r, device=x.device)
    period = 2 * n
    m = idx % period
    m = torch.where(m >= n, period - 1 - m, m)
    return x.index_select(dim, m)


def gfilter(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """可分离高斯滤波，x [B, C, H, W]。"""
    B, C, H, W = x.shape
    r = (w.numel() - 1) // 2
    y = _sym_pad(x, r, 3).reshape(B * C, 1, H, W + 2 * r)
    y = F.conv2d(y, w.reshape(1, 1, 1, -1)).reshape(B, C, H, W)
    y = _sym_pad(y, r, 2).reshape(B * C, 1, H + 2 * r, W)
    return F.conv2d(y, w.reshape(1, 1, -1, 1)).reshape(B, C, H, W)


def ssim_map(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """x, y [B, H, W, 3]，0..255 → [B, H, W]（三通道平均）。"""
    w = gauss_kernel(sigma, device=x.device, dtype=x.dtype)
    X, Y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
    st = gfilter(torch.cat([X, Y, X * X, Y * Y, X * Y], 1), w)
    ux, uy, uxx, uyy, uxy = st.split(3, 1)
    vx, vy, vxy = uxx - ux * ux, uyy - uy * uy, uxy - ux * uy
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    return S.mean(1)


def composite_on_ref(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """img 预乘 RGBA 0..1 [B, H, W, 4]，ref 0..255 [B, H, W, 3] → 合成结果 0..255（metric.py 第 2 条）。"""
    return img[..., :3] * 255.0 + ref * (1.0 - img[..., 3:4])


def metric_mask(alpha: torch.Tensor, excl_w: torch.Tensor, scale: float) -> torch.Tensor:
    """近似 metric.py 的降采样遮罩：dilate(α > 0.1, 半径 3px@720) 在分辨率比例 scale 下按 max-pool 近似，× (有效区 > 0.5)。不求导。"""
    with torch.no_grad():
        m = (alpha > 0.1).float()[:, None]
        r = max(1, int(round(3 * scale)))
        m = F.max_pool2d(m, 2 * r + 1, 1, r)[:, 0]
        return m * (excl_w > 0.5).float()


def masked_ssim(img: torch.Tensor, ref: torch.Tensor, excl_w: torch.Tensor, scale: float, mask: torch.Tensor | None = None):
    """逐帧 SSIM（metric 口径，在当前分辨率上）。返回 ([B] SSIM, 遮罩)。"""
    comp = composite_on_ref(img, ref)
    S = ssim_map(comp, ref)
    m = metric_mask(img[..., 3], excl_w, scale) if mask is None else mask
    return (S * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1.0), m


def blur_l1(img: torch.Tensor, ref: torch.Tensor, excl_w: torch.Tensor, sigma720: float, scale: float) -> torch.Tensor:
    """粗尺度光度项：合成图与参考都做高斯模糊（σ 以 720p 像素计），在放宽的角色遮罩内取 RGB 平均绝对差（0..255）。返回 [B]。
    实测（s24，fit4 解附近横移 ±16px）：SSIM@360 的吸引域只有约 ±4px，σ = 8～16px 的模糊 L1 在 ±16px 内单调，
    用来把姿态、前倾、身体这些大偏差拉进 SSIM 的吸引域。"""
    comp = composite_on_ref(img, ref)
    k = gauss_kernel(max(0.5, sigma720 * scale), device=img.device, dtype=img.dtype)
    x = torch.cat([comp, ref], -1).permute(0, 3, 1, 2)
    xb = gfilter(x, k)
    with torch.no_grad():
        r = max(2, int(round((12 + 2 * sigma720) * scale)))
        m = F.max_pool2d((img[..., 3] > 0.1).float()[:, None], 2 * r + 1, 1, r)[:, 0] * (excl_w > 0.5).float()
    d = (xb[:, :3] - xb[:, 3:]).abs().mean(1)
    return (d * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1.0)


_DIS = None


def _gray_u8(x: torch.Tensor) -> np.ndarray:
    return (_lum(x).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def dis_flows(ours: torch.Tensor, ref: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """OpenCV DIS 光流（不求导）：ours / ref [B, H, W, 3] 0..255 → 每帧 (前向 ours→ref, 后向 ref→ours)，[H, W, 2]。"""
    global _DIS
    from concurrent.futures import ThreadPoolExecutor
    a, b = _gray_u8(ours), _gray_u8(ref)

    def one(i):
        d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return d.calc(a[i], b[i], None), d.calc(b[i], a[i], None)

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, range(a.shape[0])))


def _sample(field: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """双线性采样 field [H, W, C] 在像素坐标 p [N, 2]（像素中心为 +0.5）。"""
    H, W, _ = field.shape
    g = torch.stack([(p[:, 0]) / W * 2 - 1, (p[:, 1]) / H * 2 - 1], -1)[None, None].float()
    return F.grid_sample(field.permute(2, 0, 1)[None].float(), g, align_corners=False, padding_mode="border")[0, :, 0].t()


def flow_loss(verts_screen: torch.Tensor, valid: torch.Tensor, img: torch.Tensor, ref: torch.Tensor, scale_c: float,
              fb_thr: float = 1.5) -> tuple[torch.Tensor, list[int]]:
    """光流项：把我们的合成图（渲染合成到参考帧上）与参考帧之间的 DIS 光流当作每个顶点的目标位移，
    目标 = 顶点当前位置 + flow(顶点)（不求导），损失 = Cauchy(|p − 目标| / c) 的均值。
    只用前后向一致（误差 < fb_thr 像素）、落在我们 α > 0.5 处、所属 drawable 可见的顶点。
    verts_screen [B, V, 2]（当前分辨率像素）、valid [B, V] 布尔 → ([B] 损失, 每帧有效点数)。"""
    with torch.no_grad():
        comp = composite_on_ref(img.detach(), ref)
        flows = dis_flows(comp, ref)
    out, counts = [], []
    for i, (fw, bw) in enumerate(flows):
        p = verts_screen[i]
        with torch.no_grad():
            fwt = torch.tensor(fw, device=p.device)
            bwt = torch.tensor(bw, device=p.device)
            pd = p.detach()
            f1 = _sample(fwt, pd)
            f2 = _sample(bwt, pd + f1)
            H, W = fw.shape[:2]
            a = _sample(img[i, ..., 3:4].detach(), pd)[:, 0]
            ok = valid[i] & ((f1 + f2).norm(dim=-1) < fb_thr) & (a > 0.5) & (pd[:, 0] > 0) & (pd[:, 1] > 0) & (pd[:, 0] < W) & (pd[:, 1] < H)
            tgt = pd + f1
        n = int(ok.sum())
        counts.append(n)
        if n == 0:
            out.append(p.sum() * 0)
            continue
        d = (p[ok] - tgt[ok]).norm(dim=-1)
        out.append(torch.log1p((d / scale_c) ** 2).mean())
    return torch.stack(out), counts


def _lum(x):
    return (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114) / 255.0


def edge_loss(img: torch.Tensor, ref: torch.Tensor, excl_w: torch.Tensor, sigma: float, scale: float, eps: float = 0.01) -> torch.Tensor:
    """边缘项：亮度经高斯 σ 模糊后的梯度幅值（Sobel），合成图与参考之差的 Charbonnier，均值取在「放宽的角色遮罩」内。返回 [B]。
    σ 大时梯度场是平滑的长程吸引（粗对齐），σ 小时贴合线稿；只看结构、不看颜色与亮度偏差（与 SSIM 互补）。"""
    comp = composite_on_ref(img, ref)
    L = torch.stack([_lum(comp), _lum(ref)], 1)  # [B, 2, H, W]
    if sigma > 0:
        L = gfilter(L, gauss_kernel(sigma, device=L.device, dtype=L.dtype))
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=L.device, dtype=L.dtype) / 8.0
    B = L.shape[0]
    Lp = F.pad(L.reshape(B * 2, 1, *L.shape[2:]), (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(Lp, kx[None, None])
    gy = F.conv2d(Lp, kx.t()[None, None])
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8).reshape(B, 2, *L.shape[2:])
    with torch.no_grad():
        r = max(2, int(round(12 * scale)) + int(math.ceil(2 * sigma)))
        m = F.max_pool2d((img[..., 3] > 0.1).float()[:, None], 2 * r + 1, 1, r)[:, 0] * (excl_w > 0.5).float()
    d = mag[:, 0] - mag[:, 1]
    rho = torch.sqrt(d * d + eps * eps) - eps
    return (rho * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1.0) * 100.0
