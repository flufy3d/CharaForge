"""眼睛中心细化：人工/跟踪给出的 eyes 只是粗估（±20px），这里在每只眼附近用虹膜颜色定位。

角色虹膜是高饱和、高亮的青蓝色（HSV 色相约 85~110/180），与深蓝头发、肤色、白底差异明显。
做法：以粗估点为中心、半径 r = 0.45 × 两眼距离的窗口内，取青色像素掩膜 → 连通域 →
选「面积足够大且离粗估点最近」的连通域，用其像素质心作为眼睛中心。

拒绝条件（保留粗估值，计为未细化）：
  - 找不到面积 ≥ max(12, (0.05×两眼距离)²) 的青色连通域（闭眼、><眼、圈圈眼、爱心眼、被钢盆遮挡、强模糊等）；
  - 两只眼细化后落到同一个连通域、或细化后两眼距离变化超过 35%；
  - 位移超过 0.3 × 两眼距离；连通域细长（头发高光）或过大。
注意：青色质心是「虹膜中心」，眼球转动时会偏离眼眶中心；高光和眼镜反光会让质心有 1~3px 偏差。
"""
from __future__ import annotations

import cv2
import numpy as np

LO = np.array([84, 70, 80], np.uint8)
HI = np.array([108, 255, 255], np.uint8)


def refine(img_rgb: np.ndarray, eyes):
    """返回 (新 eyes, 每只眼是否成功细化 [bool, bool])。"""
    (lx, ly), (rx, ry) = eyes
    d = float(np.hypot(rx - lx, ry - ly))
    if d < 12:
        return eyes, [False, False]
    r = 0.45 * d
    min_area = max(12.0, (0.05 * d) ** 2)
    out, ok, labels_used = [], [], []
    H, W = img_rgb.shape[:2]
    for ex, ey in eyes:
        x0, y0 = int(max(0, ex - r)), int(max(0, ey - r))
        x1, y1 = int(min(W, ex + r)), int(min(H, ey + r))
        if x1 - x0 < 4 or y1 - y0 < 4:
            out.append([ex, ey]); ok.append(False); labels_used.append(None); continue
        hsv = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
        m = cv2.inRange(hsv, LO, HI)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
        best, bd = None, 1e9
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # 虹膜是紧凑的椭圆块；头发高光是细长条 → 用长宽比和填充率排除
            if a < min_area or not (0.35 <= bw / bh <= 2.5) or a < 0.3 * bw * bh or max(bw, bh) > 0.6 * d:
                continue
            cx, cy = cents[i][0] + x0, cents[i][1] + y0
            dist = np.hypot(cx - ex, cy - ey)
            if dist < bd:
                best, bd = (cx, cy, i), dist
        if best is None or bd > 0.3 * d:
            out.append([ex, ey]); ok.append(False); labels_used.append(None)
        else:
            # 用该连通域的外接框中心与质心平均：高光（白）会在掩膜里挖洞，使质心略偏
            i = best[2]
            bx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2 + x0
            by = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2 + y0
            out.append([round((best[0] + bx) / 2, 1), round((best[1] + by) / 2, 1)])
            ok.append(True); labels_used.append((round(best[0]), round(best[1])))
    if all(ok):
        nd = float(np.hypot(out[1][0] - out[0][0], out[1][1] - out[0][1]))
        if labels_used[0] == labels_used[1] or abs(nd - d) / d > 0.35:
            return eyes, [False, False]
    return out, ok


def _blobs(img_rgb, box, min_area):
    x, y, w, h = [int(round(v)) for v in box]
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    hsv = cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, LO, HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if a < min_area or not (0.35 <= bw / bh <= 2.5) or a < 0.3 * bw * bh:
            continue
        bx = stats[i, cv2.CC_STAT_LEFT] + bw / 2 + x0
        by = stats[i, cv2.CC_STAT_TOP] + bh / 2 + y0
        out.append(((cents[i][0] + x0 + bx) / 2, (cents[i][1] + y0 + by) / 2, float(a), bw, bh))
    return out


def find_pair(img_rgb, head, prior=None):
    """在 head 框内找一对虹膜：两个大小相近的紧凑青色块，水平间距 0.2~0.75 × 头宽，
    高度差 < 0.25 × 间距，大小不超过头宽的 0.3。prior（粗估 eyes）用于在多个候选里择优。
    返回 eyes 或 None。"""
    x, y, w, h = head
    if w < 30:
        return None
    bl = [b for b in _blobs(img_rgb, head, max(12.0, (0.02 * w) ** 2)) if max(b[3], b[4]) < 0.3 * w]
    best, bs = None, 1e18
    for i in range(len(bl)):
        for j in range(len(bl)):
            a, b = bl[i], bl[j]
            if a[0] >= b[0]:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            if not (0.2 * w <= dx <= 0.75 * w) or abs(dy) > 0.25 * dx:
                continue
            ratio = max(a[2], b[2]) / min(a[2], b[2])
            if ratio > 3.0:
                continue
            score = -min(a[2], b[2]) * (1.0 / ratio)
            if prior is not None:
                (px0, py0), (px1, py1) = prior
                score += 3.0 * (np.hypot(a[0] - px0, a[1] - py0) + np.hypot(b[0] - px1, b[1] - py1)) ** 2
            if score < bs:
                best, bs = [[round(a[0], 1), round(a[1], 1)], [round(b[0], 1), round(b[1], 1)]], score
    return best
