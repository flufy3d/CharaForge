"""A7：打开快速路径后运行另一个脚本（对拍 / 梯度检查用），不改那个脚本本身。

    tools/.venv/bin/python -I tools/gpu/scripts/a7_fast_run.py [--raster-fast 1] [--checkpoint 1] [--auto-graph 0] \
        -- tools/gpu/scripts/a3_compare.py --dump .data/gpu/frames/main ...

  --raster-fast 1：raster.FAST_DEFAULT = True（render 缺省走 Triton 快速路径）
  --checkpoint 1 ：composite.CHECKPOINT = True（滤镜分帧检查点）
  --auto-graph 1 ：之后新建的 L2D 都设 auto_graph = True（CUDA graph）
  --deform-triton 0/1：deform.USE_TRITON
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools/gpu"))


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster-fast", type=int, default=1)
    ap.add_argument("--checkpoint", type=int, default=1)
    ap.add_argument("--auto-graph", type=int, default=0)
    ap.add_argument("--deform-triton", type=int, default=None)
    a = ap.parse_args(argv[:i])
    script, rest = argv[i + 1], argv[i + 2:]
    from l2d import composite, deform, raster
    raster.FAST_DEFAULT = bool(a.raster_fast)
    composite.CHECKPOINT = bool(a.checkpoint)
    if a.deform_triton is not None:
        deform.USE_TRITON = bool(a.deform_triton)
    if a.auto_graph:
        init = deform.L2D.__init__

        def patched(self, *args, **kw):
            init(self, *args, **kw)
            self.auto_graph = True
        deform.L2D.__init__ = patched
    print(f"[a7_fast_run] raster.FAST_DEFAULT={raster.FAST_DEFAULT} composite.CHECKPOINT={composite.CHECKPOINT} "
          f"deform.USE_TRITON={deform.USE_TRITON} auto_graph={bool(a.auto_graph)}", flush=True)
    sys.argv = [script] + rest
    sys.path.insert(0, str(Path(script).resolve().parent))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
