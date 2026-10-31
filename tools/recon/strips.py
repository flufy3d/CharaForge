"""人工复核切点用：每个候选切点 b 生成一行连续帧 b-4..b+3（全帧，不跳帧），多行拼一张图。
用法：strips.py b1,b2,... out.jpg [--tw 240] [--before 4] [--after 4]
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from sheet import grid, save  # noqa: E402

def main(argv):
    tw, before, after = 240, 4, 4
    for k in ("--tw", "--before", "--after"):
        if k in argv:
            i = argv.index(k); v = int(argv[i + 1]); del argv[i:i + 2]
            if k == "--tw": tw = v
            elif k == "--before": before = v
            else: after = v
    bs = [int(x) for x in argv[0].split(",")]
    frames = []
    for b in bs:
        frames += list(range(b - before, b + after))
    save(grid(frames, before + after, tw), argv[1])
    print(argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])
