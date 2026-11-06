# 闪烁尖峰统计：已统一到 tools/qa/flicker.py（唯一实现），这里只是转调，保留旧用法。
#   tools/.venv/bin/python -I tools/fit/scripts/flicker.py <渲染目录> <起帧-止帧(含)> [--json out.json]
# 口径：我们的帧间变化 > 3 × max(原视频, 1) 且 > 4（320×180 灰度，相邻两帧渲染 alpha 都 > 0.5 的区域），
# 镜头切点不判定；原视频重复帧（原视频 < 0.3、我们 > 2）单独统计。详见 tools/qa/README.md。
import runpy
import sys
from pathlib import Path

qa = Path(__file__).resolve().parents[2] / "qa"
sys.path.insert(0, str(qa))
sys.argv[0] = str(qa / "flicker.py")
runpy.run_path(str(qa / "flicker.py"), run_name="__main__")
