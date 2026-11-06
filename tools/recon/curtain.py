"""上下“窗帘”装饰边框（前景，挡在角色前面）的逐帧内边缘检测 → tools/recon/curtain.json。

窗帘随节拍上下起伏（内边缘在 y≈40~110 / 590~690 之间摆动），固定的 BORDER 排除区（62 / 652）盖不全。
窗帘的深色布条是很均匀的蓝紫色（HSV 约 H114 S135 V180）；背景圆点同色相但更浅（S≈100 V≈225），头发更暗。
做法：逐行统计整行里窗帘蓝像素的比例；以靠边 24 行的中位数为窗帘带水平 L，从画面上沿往下（下沿往上）
找第一处连续 RUN 行比例 < L/2 的位置，再外扩 MARGIN 像素盖住拱形尖端。窗帘在角色前面，所以角色不会降低窗帘行的比例；没有窗帘（第 0 行比例很低）记 0 / 720。

用法：tools/.venv/bin/python -I tools/recon/curtain.py            （全片逐帧检测，写 curtain.json，约 1 分钟）
      tools/.venv/bin/python -I tools/recon/curtain.py --frames 700,701 （只打印）
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import read_frame  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "curtain.json")
LO = np.array([109, 112, 150], np.uint8)
HI = np.array([119, 175, 205], np.uint8)
MARGIN = 8          # 半高阈值行之外再外扩（拱形尖端约 8px）
RUN = 6             # 连续 RUN 行低于阈值才算出了窗帘带（窗帘内部有 2~4 行的横向白条）
TOP_MAX, BOT_MIN = 115, 600
NO_CURTAIN = 0.15   # 边缘 24 行窗帘蓝比例中位数低于它 = 这一侧没有窗帘


def edges(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    frac = (cv2.inRange(hsv, LO, HI) > 0).mean(axis=1)

    def scan(rows):
        level = float(np.median(frac[rows[2:26]]))
        if level < NO_CURTAIN:
            return None   # 这一侧没有窗帘
        thr = 0.5 * level
        for k in range(2, len(rows) - RUN):
            if all(frac[rows[k + j]] < thr for j in range(RUN)):
                return rows[k]
        return rows[-1]
    t = scan(list(range(0, 180)))
    b = scan(list(range(719, 520, -1)))
    # 头发 / 深色衣服 / 问号等与窗帘同色时会把边缘拖进画面（s23 2330 上沿 187、s25 2466 下沿 539）：
    # 截到实测的起伏范围（s06/s15/s25 逐帧量过：上 39~110，下 590~696），宁大勿小
    top = 0 if t is None else min(TOP_MAX, t + MARGIN)
    bot = 720 if b is None else max(BOT_MIN, b - MARGIN)
    return int(top), int(bot)


def main(argv):
    if argv and argv[0] == "--frames":
        for f in map(int, argv[1].split(",")):
            print(f, edges(read_frame(f)))
        return
    from common import FRAMES
    n = len(list(FRAMES.glob("*.jpg")))
    raw = np.array([edges(read_frame(f)) for f in range(n)], np.int32)
    # 时间上 5 帧中值，压掉个别帧被深色衣服/头发拉偏的检测（窗帘起伏周期约 45 帧，中值不会抹掉真实起伏）
    sm = raw.copy()
    for f in range(n):
        a, b = max(0, f - 2), min(n, f + 3)
        sm[f] = np.median(raw[a:b], axis=0)
    with open(OUT, "w", encoding="utf-8") as fo:
        json.dump({"note": "逐帧 [上窗帘内边缘 y, 下窗帘内边缘 y]（已外扩 MARGIN、5 帧中值）；0 / 720 = 该侧没有窗帘。"
                           "只在 manual.py 的 BORDER() 区间内使用", "margin": MARGIN,
                   "frames": sm.tolist()}, fo, separators=(",", ":"))
    print("写出", OUT, n, "帧")


if __name__ == "__main__":
    main(sys.argv[1:])
