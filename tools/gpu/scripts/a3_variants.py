"""A3 对拍用的派生场景：在 scenes/deepseek-chan/scene.json 的基础上改几个图层属性，覆盖原片里没有用到的合成特性。

  tools/.venv/bin/python -I tools/gpu/scripts/a3_variants.py [输出路径，缺省 .data/gpu/scenes/variants.scene.json]

改动（只影响下面这些图层；原场景文件不动）：
  - 相机：x=640, y=360, zoom=1.08, rotation=3°（所有图层）
  - s24.main：flipX = true
  - s33.main：scaleX = −0.8、scaleY = 1.15（负 scaleX 等价于翻转 + 非等比缩放）
  - s10.right：blendMode = screen（与 s10.left 叠加）
  - s19.glasses：blendMode = add（与 s19.bowl 叠加）
  - s08.loading：blendMode = multiply（在 s08.bowl 之上）
  - s16.next：在原滤镜前加 motionBlur {angle: 30, length: 12}
  - s12.main：resample.method = bicubic
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / ".data/gpu/scenes/variants.scene.json"
    s = json.loads((REPO / "scenes/deepseek-chan/scene.json").read_text(encoding="utf-8"))
    s["camera"] = {"x": 640, "y": 360, "zoom": 1.08, "rotation": 3}
    L = {l["id"]: l for l in s["layers"]}
    L["s24.main"]["transform"]["flipX"] = True
    L["s33.main"]["transform"]["scaleX"] = -0.8
    L["s33.main"]["transform"]["scaleY"] = 1.15
    L["s10.right"]["blendMode"] = "screen"
    L["s19.glasses"]["blendMode"] = "add"
    L["s08.loading"]["blendMode"] = "multiply"
    L["s16.next"]["filters"] = [{"type": "motionBlur", "angle": 30, "length": 12}] + L["s16.next"].get("filters", [])
    for f in L["s12.main"]["filters"]:
        if f["type"] == "resample":
            f["method"] = "bicubic"
    s["meta"]["title"] = s["meta"].get("title", "") + "（A3 对拍派生）"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
