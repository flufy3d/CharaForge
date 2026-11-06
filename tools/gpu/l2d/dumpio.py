"""A3 dump 的读取（格式见 tools/gpu/docs/raster.md「dump 格式」与 tools/gpu/scripts/a3_dump.ts）。

  from l2d.dumpio import Dump
  d = Dump(".data/gpu/frames/main")
  st = d.model("dsc")          # 静态数据（dict，数组为 numpy）
  fr = d.frame(2400)           # 每帧数据：fr["layers"][i]["arrays"]["positions"] 等
  ref = d.ref_premul(2400)     # runtime 合成结果，预乘 RGBA uint8 [H, W, 4]

只依赖 numpy / PIL，不依赖 torch。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

_DT = {"f4": np.float32, "i4": np.int32, "u1": np.uint8, "u2": np.uint16}


def _load_blob(json_path: Path) -> dict:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    raw = json_path.with_suffix(".bin").read_bytes()
    arrays = {}
    for name, sp in meta.pop("arrays").items():
        a = np.frombuffer(raw, _DT[sp["dtype"]], count=int(np.prod(sp["shape"])) if sp["shape"] else 1, offset=sp["offset"])
        arrays[name] = a.reshape(sp["shape"])
    meta["arrays"] = arrays
    return meta


def _png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


class Dump:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self._models: dict[str, dict] = {}

    @property
    def frames(self) -> list[int]:
        return list(self.manifest["frames"])

    @property
    def size(self) -> tuple[int, int]:
        c = self.manifest["composition"]
        return int(c["width"]), int(c["height"])

    def model(self, model_id: str) -> dict:
        if model_id not in self._models:
            m = _load_blob(self.root / f"{self.manifest['models'][model_id]}.json")
            self._models[model_id] = m
        return self._models[model_id]

    def frame(self, f: int) -> dict:
        """每帧数据。layers[i]["arrays"] 为该图层的数组（键：positions、opacities、multiplyColors、screenColors、
        renderOrders、drawOrders、dynamicFlags、params、partOpacities）。"""
        m = _load_blob(self.root / f"f{f:06d}.json")
        arrs = m.pop("arrays")
        for i, L in enumerate(m["layers"]):
            L["arrays"] = {k.split("/", 1)[1]: v for k, v in arrs.items() if k.startswith(f"{i}/")}
        return m

    def ref_premul(self, f: int) -> np.ndarray:
        """runtime 合成结果（预乘 RGBA，uint8，行从上到下）。"""
        return _png(self.root / f"f{f:06d}_ref_premul.png")

    def ref_png(self, f: int) -> Path:
        """与 pnpm render 相同编码的参考 PNG（非预乘）。"""
        return self.root / f"f{f:06d}_ref.png"

    def layer_buffers(self, f: int, layer_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """--layers 时 dump 的图层缓冲：(Cubism 绘制结果, 滤镜之后)，均为预乘 RGBA uint8。"""
        safe = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in layer_id)
        a = self.root / f"f{f:06d}_{safe}_model.png"
        b = self.root / f"f{f:06d}_{safe}_src.png"
        if not a.exists():
            return None
        return _png(a), _png(b)
