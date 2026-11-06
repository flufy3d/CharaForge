"""VTube Studio 前向模型的 torch 实现，语义与 tools/fit/src/vts.ts 逐条一致（单元测试 tests/test_a5_vts.py 与 TS 版对拍）：

  - 映射表读自模型的 *.vtube.json（ParameterSettings）：区间线性映射，不夹紧（模型参数由 Framework 写入时夹到范围）；
  - decouple（vts.ts 的 MODELX.decouple）：输出为 PositionX/Y/Z 的设置改由独立输入 BodyX/BodyY/BodyZ 驱动；
  - 中性值：每个输入取「第一个用到它的设置」把模型缺省值逆映射得到的输入值（outHi = outLo 时取 inLo）；
  - 平滑：y_0 = map(x_0)，y_i = y_{i−1} + α·(map(x_i) − y_{i−1})，α = 1 / (1 + c·smoothing)（逐帧，60 fps）；
    这是线性运算，这里写成下三角矩阵 K（每种 α 一个）乘逐帧序列，整段一次 matmul，可求导；
  - 自动呼吸（UseBreathing）：0.5 + 0.5·sin(2π·f/60/T + φ)，f 为场景绝对帧号。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

# 与 vts.ts INPUT_DIMS 相同：拟合的面捕输入、范围、时间尺度 tau（相邻节点之间「正常」的变化幅度）
INPUT_DIMS: dict[str, dict] = {
    "FaceAngleX": {"lo": -40, "hi": 40, "sd": 3, "tau": 8},
    "FaceAngleY": {"lo": -30, "hi": 30, "sd": 2, "tau": 6},
    "FaceAngleZ": {"lo": -40, "hi": 40, "sd": 3, "tau": 8},
    "EyeOpenLeft": {"lo": 0, "hi": 1, "sd": 0.06, "tau": 0.4},
    "EyeOpenRight": {"lo": 0, "hi": 1, "sd": 0.06, "tau": 0.4},
    "EyeRightX": {"lo": -0.8, "hi": 0.8, "sd": 0.05, "tau": 0.3},
    "EyeRightY": {"lo": -0.9, "hi": 0.9, "sd": 0.05, "tau": 0.3},
    "MouthSmile": {"lo": -0.2, "hi": 1.2, "sd": 0.06, "tau": 0.3},
    "MouthOpen": {"lo": 0, "hi": 1, "sd": 0.05, "tau": 0.3},
    "Brows": {"lo": 0, "hi": 0.9, "sd": 0.04, "tau": 0.2},
    "BodyX": {"lo": -40, "hi": 40, "sd": 3, "tau": 8},
    "BodyY": {"lo": -40, "hi": 40, "sd": 3, "tau": 8},
    "BodyZ": {"lo": -40, "hi": 40, "sd": 3, "tau": 8},
}


class Vts:
    def __init__(self, path: str | Path, model_defaults: dict[str, float], decouple: bool = False):
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        self.settings = []
        for p in v["ParameterSettings"]:
            s = {"name": p["Name"], "input": p.get("Input") or "", "output": p["OutputLive2D"],
                 "inLo": float(p["InputRangeLower"]), "inHi": float(p["InputRangeUpper"]),
                 "outLo": float(p["OutputRangeLower"]), "outHi": float(p["OutputRangeUpper"]),
                 "smoothing": float(p.get("Smoothing") or 0), "breathing": bool(p.get("UseBreathing"))}
            if decouple and s["output"] in ("PositionX", "PositionY", "PositionZ"):
                s["input"] = "Body" + s["output"][-1]
            self.settings.append(s)
        self.neutral: dict[str, float] = {}
        for s in self.settings:
            if not s["input"] or s["input"] in self.neutral:
                continue
            d = model_defaults.get(s["output"], 0.0)
            self.neutral[s["input"]] = s["inLo"] if s["outHi"] == s["outLo"] else s["inLo"] + (d - s["outLo"]) / (s["outHi"] - s["outLo"]) * (s["inHi"] - s["inLo"])
        # 拟合的输入：INPUT_DIMS 里、且有设置用到的（顺序同 INPUT_DIMS）
        self.inputs = [k for k in INPUT_DIMS if any(s["input"] == k for s in self.settings)]
        self.outputs = [s["output"] for s in self.settings]
        self._K: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def alpha(smoothing: float, c: float) -> float:
        return 1.0 / (1.0 + c * smoothing)

    def _kernel(self, a: float, n: int, device, dtype) -> torch.Tensor:
        """下三角平滑矩阵：y = K·v（y_0 = v_0）。"""
        key = (a, n, str(device), dtype)
        K = self._K.get(key)
        if K is None:
            i = torch.arange(n, dtype=torch.float64)
            d = (i[:, None] - i[None, :])
            K = torch.where(d >= 0, a * (1 - a) ** d.clamp_min(0), torch.zeros(()))
            K[:, 0] = (1 - a) ** i  # 第 0 帧的权重：(1−α)^i
            K = K.to(device=device, dtype=dtype)
            self._K[key] = K
        return K

    def forward(self, inputs: dict[str, torch.Tensor], n: int, f0: int, smooth_c: float,
                breath_period, breath_phase, device="cuda", dtype=torch.float64) -> dict[str, torch.Tensor]:
        """逐帧输入（name → [n]，帧 f0..f0+n−1；缺的输入取中性值）→ 逐帧模型参数（output → [n]）。
        breath_period / breath_phase 可为张量（可求导）。"""
        out: dict[str, torch.Tensor] = {}
        frames = torch.arange(f0, f0 + n, device=device, dtype=dtype)
        for s in self.settings:
            if s["breathing"]:
                out[s["output"]] = 0.5 + 0.5 * torch.sin(2 * math.pi * frames / 60.0 / breath_period + breath_phase)
                continue
            x = inputs.get(s["input"])
            if x is None:
                x = torch.full((n,), float(self.neutral.get(s["input"], 0.0)), device=device, dtype=dtype)
            slope = (s["outHi"] - s["outLo"]) / (s["inHi"] - s["inLo"])
            v = s["outLo"] + (x.to(dtype) - s["inLo"]) * slope
            a = self.alpha(s["smoothing"], smooth_c)
            out[s["output"]] = v if a == 1.0 else self._kernel(a, n, device, dtype) @ v
        return out
