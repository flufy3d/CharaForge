"""参数 → 图层缓冲 → 合成（A1 变形 + A3 光栅化与合成）。合成分辨率可以按比例 s 缩小（640×360 时 s = 0.5）：
to_screen 整体乘 s，屏幕像素单位的滤镜参数（blur σ、motionBlur 长度、clip 几何与羽化）同样乘 s，resample factor 不变。"""
from __future__ import annotations

import torch

from l2d import composite as C
from l2d.raster import LayerInput, layer_matrix


def to_screen(t: dict, s: float = 1.0, device="cuda") -> torch.Tensor:
    """图层变换（数值或张量）→ [2, 3] 画布像素 → 合成像素（无相机；本项目场景没有相机）。"""
    m = layer_matrix(t["x"], t["y"], t["scale"], t["rotation"], t["anchorX"], t["anchorY"], t.get("scaleX", 1.0), t.get("scaleY", 1.0), t.get("flipX", False))
    return (m * s).to(device)


def scale_draw(d: dict, s: float) -> dict:
    """draw 参数（runtime 的 LayerDrawParams 口径）缩放到合成分辨率比例 s。"""
    if s == 1.0:
        return d
    fs = []
    for f in d.get("filters", []):
        f = dict(f)
        if f["type"] == "blur":
            f["sigma"] = f["sigma"] * s
        elif f["type"] == "motionBlur":
            f["length"] = f["length"] * s
        fs.append(f)
    out = {**d, "filters": fs}
    if d.get("clip"):
        out["clip"] = {k: (v * s if k != "feather" else max(1.0, v) * s) for k, v in d["clip"].items()}
    return out


def raster_layer(rig, params: torch.Tensor, ts: torch.Tensor, W: int, H: int, aa=True, quantize=True, aa_layers=None) -> tuple[torch.Tensor, object]:
    """[B, P] 最终参数、[B, 2, 3] to_screen → (图层缓冲 [B, H, W, 4] 预乘, A1 输出)。"""
    out = rig.deform(params)
    # A3 只收 float32（变形器可以是 float64，见 rig.Rig 的 deform_dtype）
    f = lambda x: x.float()
    inp = LayerInput.from_arrays(f(out.verts), f(out.opacity), f(out.multiply), f(out.screen), out.render_order, out.visible, ts, out.vertex_updated)
    buf = rig.raster.render(inp, W, H, aa=aa, quantize=quantize, aa_layers=aa_layers)
    return buf, out


def compose_one(buf: torch.Tensor, draw: dict, quantize=True) -> torch.Tensor:
    """单图层合成（透明背景）：buf [B, H, W, 4] 与同一组 draw 参数（标量可为 [B] 张量）。"""
    B, H, W, _ = buf.shape
    return C.composite([(buf, draw)], H, W, None, quantize=quantize, B=B)
