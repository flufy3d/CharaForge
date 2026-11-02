"""人工标注源（唯一事实来源）→ 生成 tools/recon/annotations.json。

坐标约定（重要）：
  人工读数来自 measure.py 生成的「网格图」review/g_<shot>.jpg：整帧缩放 0.75，2x2 排列，
  每格 960x540，格间距 4px；四格依次是镜头的 [start, start+len//3, start+2*len//3, end-1] 帧，
  顺序为 左上(TL) 右上(TR) 左下(BL) 右下(BR)。
  下面所有 head/bbox/eyes/clip/排除区 都直接写「网格图上的显示像素」，由 G() 换算成视频像素：
     video = (display - tile_origin) / 0.75
  这样每个数字都能对着网格图复核。读数误差约 ±5 显示像素 ≈ ±7 视频像素。
  非网格帧的锚点用 V() 直接写视频像素。

镜头以起始帧为键（镜头 ID 由 build 时按顺序生成 s01..）。
用法：tools/.venv/bin/python -I tools/recon/manual.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import ANNOT  # noqa: E402

SCALE = 0.75
ORIG = [(0, 0), (964, 0), (0, 544), (964, 544)]

# ---------------------------------------------------------------- 镜头边界（逐帧复核过）
# (start, category, hasCharacter, transitionIn.type, transitionIn.duration, note)
BOUNDS = [
    (0, "noCharacter", False, "cut", 0, "片头：点阵背景上聊天窗口弹出放大（纯 UI）"),
    (18, "noCharacter", False, "cut", 0, "窗口放大到全屏的白色画面（纯 UI）"),
    (34, "noCharacter", False, "cut", 0, "聊天窗口：鲸鱼 logo、逐字打出“嗨！我是deepseek 有什么想要问我的吗？”、输入框、鼠标光标点击（纯 UI）"),
    (254, "pip", True, "other", 12, "角色窗口（Photo Booth 样式）从聊天窗口中部点击处缩放弹出，约 266 帧后窗口大小稳定"),
    (376, "transition", False, "wipe", 0, "蓝色颜料泼溅擦除转场（角色被大面积遮挡，不参与拟合）"),
    (400, "multi", True, "wipe", 0, "泼溅擦除后的新画面；之后陆续弹出两个画中画窗口、手指、对话气泡"),
    (654, "transition", False, "wipe", 0, "白色剪影擦除转场"),
    (684, "multi", True, "wipe", 0, "白色剪影转场后先出背景，三个实例依次从左/下方滑入"),
    (778, "transition", False, "wipe", 0, "白色剪影擦除转场"),
    (795, "multi", True, "wipe", 0, "两个实例并排：左侧大幅旋转的躯干特写（头部基本出画），右侧半身"),
    (950, "transition", False, "other", 0, "白色爆炸形剪影转场"),
    (970, "extremeCloseUp", True, "other", 0, "裙子/大腿极近特写（无头部）"),
    (1041, "pip", True, "other", 6, "圆角窗口从画面中心放大弹出覆盖上一镜头（1041~1046 窗口外仍是上一镜头残留画面）"),
    (1162, "transition", False, "wipe", 0, "蓝色颜料泼溅擦除转场"),
    (1194, "multi", True, "wipe", 0, ""),
    (1509, "transition", False, "other", 0, "旋转/万花筒模糊转场（角色严重扭曲，不参与拟合）"),
    (1527, "halfBody", True, "other", 0, ""),
    (1707, "closeUp", True, "cut", 0, ""),
    (1726, "multi", True, "cut", 0, ""),
    (1899, "transition", False, "other", 0, "白色爆炸形剪影转场"),
    (1918, "multi", True, "wipe", 0, ""),
    (2137, "multi", True, "cut", 0, ""),
    (2304, "closeUp", True, "cut", 0, ""),
    (2368, "closeUp", True, "cut", 0, ""),
    (2442, "closeUp", True, "wipe", 4, "圆角窗口从上方展开覆盖上一镜头，2442~2445 窗口外仍是上一镜头"),
    (2530, "closeUp", True, "other", 0, "白色笔刷线条遮挡下的切换（2526 起线条已出现在上一镜头）"),
    (2608, "transition", False, "wipe", 0, "翻页式斜向擦除转场"),
    (2614, "multi", True, "wipe", 0, ""),
    (2639, "multi", True, "cut", 0, ""),
    (2690, "multi", True, "other", 7, "模糊切入：2690 起清晰度逐渐恢复，约 2697 清晰"),
    (2969, "transition", False, "wipe", 0, "深蓝颜料泼溅擦除转场"),
    (3000, "multi", True, "wipe", 0, "泼溅擦除后：左侧大特写 + 右侧全身"),
    (3097, "halfBody", True, "cut", 0, "3097~3104 快速推拉（同一镜头内连续缩放，非切点）"),
    (3145, "halfBody", True, "cut", 0, ""),
    (3191, "halfBody", True, "cut", 0, "3191~3197 快速拉远"),
    (3241, "fullBody", True, "cut", 0, "硬切到 Q 版（光流残差检测补出的切点，3240→3241）"),
    (3285, "transition", False, "other", 0, "半调网点（橙色圆点）转场"),
    (3303, "multi", True, "other", 0, "网点转场残留到约 3308"),
    (3382, "transition", False, "other", 0, "浅蓝云雾/水彩擦除转场"),
    (3399, "multi", True, "other", 0, "云雾转场残留到约 3410"),
    (3478, "multi", True, "cut", 0, "3471 起上一镜头径向模糊推出，3478 硬切，新画面带径向模糊约到 3488 清晰"),
    (3662, "transition", False, "wipe", 0, "深蓝颜料泼溅擦除转场"),
    (3684, "halfBody", True, "wipe", 0, "泼溅残留到约 3692"),
    (3722, "halfBody", True, "cut", 0, ""),
    (3768, "halfBody", True, "cut", 0, ""),
    (3818, "closeUp", True, "cut", 0, ""),
]
FRAME_COUNT = 3868


def shot_range(start):
    i = [b[0] for b in BOUNDS].index(start)
    end = BOUNDS[i + 1][0] if i + 1 < len(BOUNDS) else FRAME_COUNT
    return start, end


def grid_frames(start):
    s, e = shot_range(start)
    n = e - s
    return sorted(set([s, s + n // 3, s + 2 * n // 3, e - 1]))


def G(start, k, kind, v):
    """网格图第 k 格（0=TL,1=TR,2=BL,3=BR）的显示像素 → 视频像素。
    kind: 'pt' (x,y) | 'pts' ((x,y),(x,y)) | 'box' (x0,y0,x1,y1) → [x,y,w,h]"""
    ox, oy = ORIG[k]
    c = lambda x, y: [round((x - ox) / SCALE, 1), round((y - oy) / SCALE, 1)]
    if kind == "pt":
        return c(*v)
    if kind == "pts":
        return [c(*p) for p in v]
    x0, y0 = c(v[0], v[1]); x1, y1 = c(v[2], v[3])
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(1280, x1), min(720, y1)
    return [x0, y0, round(x1 - x0, 1), round(y1 - y0, 1)]


def A(start, k, head=None, eyes=None, bbox=None, **kw):
    """网格格 k 上的锚点。"""
    f = grid_frames(start)[k]
    a = {"frame": f}
    if bbox is not None:
        a["bbox"] = G(start, k, "box", bbox) if bbox != "null" else None
    if head is not None:
        a["head"] = G(start, k, "box", head)
    if eyes is not None:
        a["eyes"] = G(start, k, "pts", eyes)
    a.update(kw)
    return a


def V(frame, head=None, eyes=None, bbox=None, hidden=False, **kw):
    """非网格帧的锚点，直接写视频像素 [x,y,w,h] / [[x,y],[x,y]]。hidden=True 表示该帧实例不可见（bbox/head 为 null）。"""
    a = {"frame": frame}
    if hidden:
        a["bbox"] = None
        a["head"] = None
    if bbox is not None:
        a["bbox"] = bbox if bbox != "null" else None
    if head is not None:
        a["head"] = head
    if eyes is not None:
        a["eyes"] = eyes
    a.update(kw)
    return a


def R(start, k, box):
    return G(start, k, "box", box)


def EX(reason, s, e, rects):
    return {"start": s, "end": e, "reason": reason, "rects": [{"frame": f, "rect": r} for f, r in rects]}


def BORDER(s, e):
    """片中常驻的上下“窗帘”装饰边框，叠在角色前面。"""
    return [EX("frame", s, e, [(s, [0, 0, 1280, 62])]), EX("frame", s, e, [(s, [0, 652, 1280, 68])])]


def FX(kind, s, e, note=""):
    d = {"kind": kind, "start": s, "end": e}
    if note:
        d["note"] = note
    return d


def SW(frame, param, value, conf=0.6, note="", fade=None):
    """fade 缺省：表情热键（key*/Clothes*）按 VTS 0.5s 淡入 = 30 帧；Movement* 由 motion 触发，阶跃 = 0。
    实测到明显不同的淡入时长时显式给出。"""
    if fade is None:
        fade = 0 if param.startswith("Movement") else 30
    d = {"frame": frame, "param": param, "value": value, "fade": fade, "confidence": conf}
    if note:
        d["note"] = note
    return d


def MH(motion, start, conf=0.5, note=""):
    d = {"motion": motion, "start": start, "confidence": conf}
    if note:
        d["note"] = note
    return d


def I(id, role, z, anchors, variant=None, sw=None, mh=None, clip=None, flipX=False, grade=None, notes="",
      mode="lk"):
    d = {"id": id, "role": role, "z": z, "variant": variant or {}, "switchEvents": sw or [],
         "motionHints": mh or [], "anchors": anchors, "trackMode": mode, "flipX": flipX}
    if clip is not None:
        d["clip"] = clip
    if grade:
        d["colorGrade"] = grade
    if notes:
        d["notes"] = notes
    return d


HALFTONE = "全屏半调网点风格背景；角色本身未见明显压暗，但整帧有轻微锐化/色阶压缩（疑似全屏滤镜），偏蓝"

# ---------------------------------------------------------------- 逐镜头标注
SHOTS: dict[int, dict] = {}

# s04 画中画：钢盆
S = 254
SHOTS[S] = dict(
    instances=[I("main", "pip", 1, [
        V(254, bbox=[573, 313, 133, 87], head=[606, 320, 67, 47], opacity=1.0),
        A(S, 1, head=(1305, 200, 1535, 330), eyes=((1400, 262), (1475, 260)), bbox=(1290, 85, 1640, 420)),
        A(S, 2, head=(360, 735, 560, 870), eyes=((440, 795), (515, 790)), bbox=(330, 640, 660, 960)),
        A(S, 3, head=(1325, 700, 1560, 835), eyes=((1395, 718), (1478, 718)), bbox=(1300, 640, 1680, 960)),
    ], variant={"key10": 1},
        clip={"rect": R(S, 3, (1215, 640, 1675, 960)), "radius": 6},
        notes="头部框不含钢盆（钢盆是 key10 部件，单独在 bbox 内）；254~266 窗口在缩放弹出，clip 取稳定后的窗口；"
              "前半段闭眼低头，后半段抬头睁眼（眼睛被钢盆阴影半遮）")],
    exclusions=[
        EX("cursor", 254, 376, [(294, R(S, 1, (1595, 415, 1645, 475))), (335, R(S, 2, (460, 650, 535, 735))),
                                 (375, R(S, 3, (1515, 700, 1570, 765)))]),
        EX("ui", 254, 376, [(294, R(S, 1, (1190, 425, 1700, 470)))]),
    ] + BORDER(254, 376),
    effects=[FX("scale", 254, 268, "窗口整体从小到大弹出")],
    confidence=0.7)

# s06 多实例：大半身 + 左上画中画(眼镜/竖食指) + 右下画中画(脱捕：加载+圈圈眼+狗尾草)
S = 400
SHOTS[S] = dict(
    instances=[
        I("main", "main", 1, [
            A(S, 0, head=(290, 40, 700, 330), eyes=((510, 170), (620, 165)), bbox=(280, 30, 960, 540), rotation=-12),
            A(S, 1, head=(1240, 20, 1560, 330), eyes=((1385, 183), (1462, 183)), bbox=(1130, 0, 1820, 540), rotation=-5),
            A(S, 2, head=(320, 640, 700, 840), eyes=((425, 745), (545, 738)), bbox=(300, 590, 960, 1088), rotation=0),
            A(S, 3, head=(1285, 600, 1690, 860), eyes=((1395, 790), (1530, 790)), bbox=(1260, 590, 1928, 1088)),
        ], sw=[SW(547, "key10", 1, 0.8, "钢盆约 547~562 淡入（实测约 15 帧，短于 VTS 默认 0.5s）", fade=15)],
            notes="400~480 单独一个半身（闭眼），之后镜头推近；547 起戴钢盆；后段眼睛被钢盆阴影遮住"),
        I("pipL", "pip", 2, [
            V(479, hidden=True),
            A(S, 1, head=(1060, 115, 1160, 195), eyes=((1100, 148), (1135, 148)), bbox=(1000, 110, 1225, 265)),
            A(S, 2, head=(40, 615, 245, 760), eyes=((95, 690), (180, 690)), bbox=(0, 615, 265, 850)),
            A(S, 3, head=(1010, 610, 1250, 770), eyes=((1075, 710), (1160, 710)), bbox=(965, 600, 1280, 850)),
        ], variant={"key11": 1, "Movement3": 1}, mode="linear",
            notes="约 480 帧窗口弹出；黑框眼镜；竖食指（Movement3，可能来自 竖手指切出来动画，起点未见）；窗口裁切随画面移动，未给 clip"),
        I("pipR", "pip", 3, [
            V(519, hidden=True),
            A(S, 2, head=(710, 780, 860, 880), eyes=((742, 838), (797, 836)), bbox=(615, 760, 955, 1000)),
            A(S, 3, head=(1640, 780, 1800, 900), eyes=((1690, 862), (1745, 862)), bbox=(1585, 760, 1928, 1000)),
        ], variant={"key8": 1, "key3": 1, "Movement1": 1}, mode="linear",
            mh=[MH("脱捕", 470, 0.4, "加载图标+圈圈眼+狗尾草同时出现，与 脱捕.motion3 的 key3/key8/Movement1 曲线一致；起点不可见，按窗口出现前推估")],
            notes="约 520 帧窗口弹出"),
    ],
    exclusions=[
        EX("finger", 514, 654, [(569, R(S, 2, (0, 830, 250, 1010))), (653, R(S, 3, (965, 830, 1205, 1010)))]),
        EX("bubble", 538, 654, [(569, R(S, 2, (220, 890, 615, 1045))), (653, R(S, 3, (1180, 880, 1565, 1045)))]),
        EX("effect", 400, 420, [(400, R(S, 0, (0, 60, 330, 330)))]),
    ] + BORDER(400, 654),
    effects=[FX("zoom", 480, 654, "主实例整体推近")],
    confidence=0.55)

# s08 三实例依次滑入
S = 684
SHOTS[S] = dict(
    instances=[
        I("bowl", "side", 1, [
            V(689, hidden=True),
            A(S, 1, head=(1035, 150, 1260, 330), eyes=((1115, 235), (1185, 238)), bbox=(965, 85, 1290, 540)),
            A(S, 2, head=(60, 690, 290, 880), eyes=((115, 775), (190, 778)), bbox=(0, 595, 330, 1088)),
            A(S, 3, head=(1020, 690, 1260, 880), eyes=((1080, 770), (1150, 770)), bbox=(965, 600, 1290, 1088)),
        ], variant={"key10": 1}, notes="约 690 从左侧滑入；头部框不含钢盆"),
        I("loading", "side", 2, [
            V(710, hidden=True),
            A(S, 1, head=(1260, 70, 1520, 330), eyes=((1388, 258), (1460, 255)), bbox=(1180, 60, 1640, 540)),
            A(S, 2, head=(320, 610, 580, 880), eyes=((462, 780), (522, 775)), bbox=(210, 590, 620, 1088)),
            A(S, 3, head=(1280, 585, 1530, 880), eyes=((1405, 745), (1475, 738)), bbox=(1200, 580, 1560, 1088)),
        ], variant={"key8": 1, "key3": 1}, notes="约 711 滑入；加载图标 + 圈圈眼（白色圆眼）"),
        I("wink", "main", 3, [
            V(740, hidden=True),
            A(S, 2, head=(590, 545, 930, 860), eyes=((690, 770), (780, 765)), bbox=(540, 545, 960, 1088)),
            A(S, 3, head=(1500, 570, 1928, 900), eyes=((1615, 870), (1710, 858)), bbox=(1480, 560, 1928, 1088), rotation=-10),
        ], notes="约 741 从右下滑入；后段右眼闭（wink，为眼睛开合参数而非开关）"),
    ],
    exclusions=[EX("silhouette", 684, 692, [(684, [0, 0, 1280, 720])]),
                EX("silhouette", 774, 778, [(777, R(S, 3, (1080, 860, 1290, 1088)))])] + BORDER(684, 778),
    effects=[FX("silhouette", 684, 692, "白色剪影转场残留"), FX("motionBlur", 688, 745, "实例滑入时有运动模糊")],
    confidence=0.55)

# s10 两实例并排
S = 795
SHOTS[S] = dict(
    instances=[
        I("left", "side", 1, [
            A(S, 0, bbox=(0, 70, 400, 540), rotation=-30),
            A(S, 3, bbox=(965, 545, 1330, 1088), rotation=0),
        ], mode="linear", notes="躯干特写，头部基本出画，仅末段可见下巴/嘴；旋转约 -30° 逐渐回正，数值很粗"),
        I("right", "main", 2, [
            A(S, 0, head=(430, 0, 930, 300), eyes=((625, 150), (765, 160)), bbox=(400, 0, 960, 540)),
            A(S, 1, head=(1400, 0, 1830, 290), eyes=((1555, 170), (1690, 165)), bbox=(1330, 0, 1928, 540)),
            A(S, 2, head=(420, 545, 900, 840), eyes=((590, 830), (705, 828)), bbox=(380, 545, 960, 1088)),
            A(S, 3, head=(1290, 545, 1890, 1000), eyes=((1525, 820), (1700, 805)), bbox=(1290, 545, 1928, 1088)),
        ], sw=[SW(860, "key7", 1, 0.5, "脸红约 860 起出现"), SW(895, "key2", 1, 0.4, "泪滴约 895~930 出现，时间不确定")],
            notes="头顶出画；镜头逐渐推近"),
    ],
    exclusions=BORDER(795, 950),
    effects=[FX("zoom", 900, 950, "推近")],
    confidence=0.5)

# s12 裙子特写（无头）
S = 970
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, bbox=(0, 0, 960, 540)), A(S, 3, bbox=(965, 0, 1928, 1088))], mode="linear",
        notes="无头部；短裙为默认（Clothes1=1）；画面整体向上平移/推近，从裙子到大腿袜口")],
    exclusions=[EX("effect", 970, 985, [(970, R(S, 0, (0, 0, 960, 540)))])] + BORDER(970, 1041),
    effects=[FX("pan", 970, 1041, "镜头沿身体向下移动")],
    confidence=0.6)

# s13 画中画：腿/脱鞋
S = 1041
SHOTS[S] = dict(
    instances=[I("main", "pip", 1, [
        A(S, 0, bbox=(315, 315, 610, 470)), A(S, 1, bbox=(1100, 60, 1820, 470)),
        A(S, 2, bbox=(370, 600, 600, 1010)), A(S, 3, bbox=(1300, 600, 1560, 1010))], mode="linear",
        sw=[SW(1117, "Clothes4", 1, 0.85, "鞋子 1117~1125 淡出（脱鞋，实测约 8 帧）", fade=8)],
        clip={"rect": R(S, 2, (90, 590, 865, 1020)), "radius": 25},
        notes="无头部；腿部特写，镜头沿腿下移；1041~1046 窗口弹出")],
    exclusions=[EX("sign", 1135, 1162, [(1137, [490, 520, 80, 80]), (1145, R(S, 3, (1245, 615, 1645, 1010))),
                                         (1161, R(S, 3, (1245, 615, 1645, 1010)))]),
                EX("effect", 1155, 1162, [(1161, R(S, 3, (1800, 840, 1928, 1088)))])] + BORDER(1041, 1162),
    effects=[FX("pan", 1041, 1135)],
    confidence=0.6)

# s33 半身（网点背景）
S = 3097
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(350, 140, 640, 420), eyes=((452, 330), (545, 330)), bbox=(300, 120, 700, 540)),
        A(S, 1, head=(1320, 30, 1560, 260), eyes=((1398, 205), (1478, 205)), bbox=(1240, 20, 1700, 540)),
        A(S, 2, head=(380, 600, 660, 860), eyes=((455, 740), (545, 740)), bbox=(280, 590, 700, 1088)),
        A(S, 3, head=(1340, 590, 1560, 800), eyes=((1405, 735), (1495, 735)), bbox=(1230, 580, 1700, 1088)),
    ], grade=HALFTONE, notes="3097~3104 快速拉远")],
    exclusions=[EX("effect", 3097, 3104, [(3097, R(S, 0, (310, 180, 530, 300)))])] + BORDER(3097, 3145),
    effects=[FX("zoom", 3097, 3105, "快速拉远"), FX("particles", 3097, 3145, "白圈、黄色方块等几何粒子")],
    confidence=0.55)

# s34 抚胸+脸红
S = 3145
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(380, 90, 640, 370), eyes=((468, 305), (548, 300)), bbox=(330, 80, 700, 540)),
        A(S, 1, head=(1350, 50, 1555, 260), eyes=((1418, 195), (1490, 195)), bbox=(1200, 40, 1650, 540)),
        A(S, 2, head=(390, 610, 640, 860), eyes=((468, 770), (553, 770)), bbox=(300, 600, 680, 1088)),
        A(S, 3, head=(1350, 600, 1600, 850), eyes=((1428, 770), (1510, 768)), bbox=(1260, 590, 1650, 1088)),
    ], variant={"Movement2": 1, "key7": 1}, grade=HALFTONE,
        notes="右手抚胸（Movement2，可能来自 默认手切举手 motion，起点在镜头前）；脸红 key7")],
    exclusions=BORDER(3145, 3191),
    effects=[FX("zoom", 3145, 3153, "快速拉远"), FX("particles", 3145, 3191)],
    confidence=0.55)

# s35 ><眼 + 狗尾草
S = 3191
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(330, 160, 610, 420), eyes=((432, 300), (515, 300)), bbox=(290, 140, 650, 540)),
        A(S, 1, head=(1340, 70, 1560, 280), eyes=((1400, 190), (1478, 185)), bbox=(1220, 40, 1700, 540)),
    ], variant={"key5": 1, "Movement1": 1}, grade=HALFTONE,
        mh=[MH("狗尾草切换出来", 3180, 0.3, "镜头开始时狗尾草已在手中（3194 可见），起点在镜头前")],
        notes="><眼（key5）；手持狗尾草（Movement1）")],
    exclusions=BORDER(3191, 3241),
    effects=[FX("zoom", 3191, 3198, "快速拉远"), FX("particles", 3191, 3241)],
    confidence=0.5)

# s36 Q 版
S = 3241
SHOTS[S] = dict(
    instances=[I("main", "chibi", 1, [
        V(3253, head=[307, 48, 626, 427], eyes=[[573, 408], [747, 395]], bbox=[200, 40, 900, 680]),
        V(3284, head=[381, 21, 587, 427], eyes=[[568, 381], [741, 368]], bbox=[300, 0, 800, 680]),
    ], variant={"Clothes2": 1}, grade=HALFTONE, mode="linear",
        notes="Q 版（Clothes2=1）；3241~3250 由远推近；坐标精度较低")],
    exclusions=BORDER(3241, 3285) + [EX("effect", 3278, 3285, [(3284, [0, 60, 330, 600])])],
    effects=[FX("zoom", 3241, 3252, "推近"), FX("particles", 3241, 3285)],
    confidence=0.4)

# s38 左大特写 + 右全身
S = 3303
SHOTS[S] = dict(
    instances=[
        I("main", "main", 2, [
            A(S, 0, head=(120, 0, 500, 330), eyes=((235, 245), (365, 238)), bbox=(0, 0, 560, 540)),
            A(S, 1, head=(1084, 0, 1480, 330), eyes=((1215, 230), (1325, 225)), bbox=(964, 0, 1520, 540)),
            A(S, 2, head=(100, 560, 480, 900), eyes=((270, 855), (390, 845)), bbox=(0, 545, 560, 1088)),
            A(S, 3, head=(1080, 570, 1480, 950), eyes=((1240, 860), (1360, 860)), bbox=(964, 560, 1520, 1088)),
        ], sw=[SW(3357, "Clothes2", 1, 0.6, "约 3357~3366 变成 Q 版（实测约 9 帧）", fade=9)], notes="后段笑眼(^^)，眼睛中心为估计"),
        I("side", "fullBody", 1, [
            A(S, 0, bbox=(610, 110, 760, 480)),
            A(S, 1, head=(1590, 230, 1680, 310), eyes=((1620, 270), (1655, 268)), bbox=(1580, 230, 1720, 525)),
            A(S, 2, bbox=(630, 870, 720, 1040)),
            A(S, 3, bbox=(1600, 950, 1680, 1050)),
        ], sw=[SW(3307, "Clothes2", 1, 0.4, "右侧全身约 3306~3309 变 Q 版，不确定")], mode="linear",
            notes="开头为常规全身（带尾巴），随后为 Q 版并逐渐缩小；短裙"),
    ],
    exclusions=[EX("text", 3303, 3382, [(3329, R(S, 1, (1690, 320, 1880, 375))), (3381, R(S, 3, (1670, 865, 1880, 920)))]),
                EX("effect", 3303, 3309, [(3303, [500, 0, 780, 720])])],
    effects=[FX("other", 3303, 3309, "网点转场残留")],
    confidence=0.45)

# s40 左 Q 版 + 右全身
S = 3399
SHOTS[S] = dict(
    instances=[
        I("chibi", "chibi", 2, [
            A(S, 1, head=(1090, 30, 1450, 330), eyes=((1205, 250), (1300, 250)), bbox=(1040, 20, 1500, 540)),
            A(S, 2, head=(150, 600, 490, 900), eyes=((250, 840), (355, 840)), bbox=(100, 580, 520, 1088)),
            V(3470, bbox=[100, 30, 600, 690]),
        ], variant={"Clothes2": 1}, sw=[SW(3440, "key7", 1, 0.4, "脸红约 3440 出现")], mode="linear"),
        I("full", "fullBody", 1, [
            A(S, 1, head=(1570, 50, 1680, 190), eyes=((1605, 120), (1650, 120)), bbox=(1540, 40, 1760, 530)),
            A(S, 2, head=(600, 590, 720, 720), eyes=((625, 665), (675, 665)), bbox=(590, 590, 780, 1075)),
        ], mode="linear", notes="短裙+尾巴；后段抚胸（Movement2）+脸红，切换时间不确定",
            sw=[SW(3440, "Movement2", 1, 0.3)]),
    ],
    exclusions=[EX("effect", 3399, 3412, [(3399, [0, 0, 700, 720])]),
                EX("text", 3399, 3478, [(3425, R(S, 1, (1720, 320, 1890, 375))), (3451, R(S, 2, (730, 860, 920, 920)))])],
    effects=[FX("radialBlur", 3470, 3478, "径向推镜模糊，接下一镜头")],
    confidence=0.4)

# s41 戴眼镜：左大 + 右全身
S = 3478
SHOTS[S] = dict(
    instances=[
        I("main", "main", 2, [
            V(3490, bbox=[0, 0, 700, 720], blurSigma=3),
            A(S, 1, head=(1100, 0, 1460, 330), eyes=((1225, 225), (1305, 222)), bbox=(964, 0, 1500, 540), blurSigma=0),
            A(S, 2, head=(130, 550, 440, 880), eyes=((240, 815), (320, 815)), bbox=(0, 545, 560, 1088)),
            A(S, 3, head=(1030, 545, 1420, 950), eyes=((1240, 830), (1345, 810)), bbox=(964, 545, 1640, 1088)),
        ], variant={"key11": 1}, sw=[SW(3596, "Movement3", 1, 0.6, "约 3596 起竖食指")],
            mh=[MH("竖手指切出来动画", 3575, 0.4)], notes="3478~3488 径向模糊；后段推近"),
        I("full", "fullBody", 1, [
            A(S, 1, head=(1590, 0, 1700, 110), eyes=((1620, 55), (1660, 55)), bbox=(1530, 0, 1760, 540)),
            A(S, 2, head=(600, 550, 720, 700), eyes=((640, 615), (690, 612)), bbox=(580, 545, 800, 1088)),
            A(S, 3, bbox=(1600, 545, 1928, 1088)),
        ], variant={"key11": 1}, sw=[SW(3596, "Movement3", 1, 0.5)], mode="linear", notes="短裙+尾巴"),
    ],
    exclusions=[EX("text", 3478, 3662, [(3539, R(S, 1, (1690, 300, 1928, 370))), (3600, R(S, 2, (790, 860, 960, 920)))])],
    effects=[FX("radialBlur", 3478, 3489, "径向模糊逐渐恢复")],
    confidence=0.45)

# s43 戴眼镜半身
S = 3684
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(330, 60, 700, 330), eyes=((470, 240), (545, 235)), bbox=(280, 50, 760, 540)),
        A(S, 1, head=(1290, 60, 1600, 300), eyes=((1405, 195), (1490, 195)), bbox=(1200, 40, 1700, 540)),
        A(S, 2, head=(360, 600, 650, 860), eyes=((450, 740), (530, 740)), bbox=(260, 590, 700, 1088)),
        A(S, 3, head=(1330, 610, 1640, 860), eyes=((1420, 740), (1505, 740)), bbox=(1230, 590, 1700, 1088)),
    ], variant={"key11": 1}, grade=HALFTONE)],
    exclusions=[EX("effect", 3684, 3693, [(3684, R(S, 0, (0, 0, 960, 540)))])] + BORDER(3684, 3722),
    effects=[FX("particles", 3684, 3722)],
    confidence=0.55)

# s44 圈圈眼 + 加载
S = 3722
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(360, 150, 650, 430), eyes=((445, 325), (535, 325)), bbox=(300, 140, 700, 540)),
        A(S, 1, head=(1370, 40, 1600, 300), eyes=((1450, 185), (1530, 190)), bbox=(1250, 30, 1700, 540)),
        A(S, 2, head=(400, 590, 670, 870), eyes=((500, 735), (580, 750)), bbox=(300, 580, 720, 1088)),
        A(S, 3, head=(1380, 600, 1640, 860), eyes=((1470, 745), (1560, 755)), bbox=(1280, 590, 1700, 1088)),
    ], variant={"key3": 1, "key8": 1}, grade=HALFTONE,
        mh=[MH("脱捕", 3700, 0.3, "加载+圈圈眼与 脱捕 中段一致，但未见狗尾草，也可能是直接按 key3/key8")])],
    exclusions=BORDER(3722, 3768),
    effects=[FX("zoom", 3722, 3730), FX("particles", 3722, 3768)],
    confidence=0.5)

# s45 钢盆
S = 3768
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(330, 230, 620, 460), eyes=((420, 345), (520, 335)), bbox=(290, 160, 700, 540), rotation=-10),
        A(S, 1, head=(1320, 140, 1520, 330), eyes=((1395, 225), (1480, 225)), bbox=(1260, 50, 1650, 540)),
        A(S, 2, head=(370, 720, 640, 880), eyes=((450, 795), (540, 795)), bbox=(300, 590, 700, 1088)),
        A(S, 3, head=(1360, 720, 1600, 880), eyes=((1420, 790), (1510, 790)), bbox=(1300, 590, 1650, 1088)),
    ], variant={"key10": 1}, grade=HALFTONE, notes="头部框不含钢盆；眼睛被钢盆阴影遮挡")],
    exclusions=BORDER(3768, 3818),
    effects=[FX("zoom", 3768, 3776), FX("particles", 3768, 3818)],
    confidence=0.5)

# s46 结尾 wink
S = 3818
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(350, 70, 650, 350), eyes=((475, 245), (550, 245)), bbox=(300, 60, 760, 540)),
        A(S, 1, head=(1250, 0, 1720, 330), eyes=((1400, 205), (1510, 205)), bbox=(1200, 0, 1800, 540)),
        A(S, 2, head=(330, 560, 760, 950), eyes=((440, 780), (590, 790)), bbox=(280, 545, 900, 1088), rotation=-10),
        A(S, 3, head=(1290, 570, 1700, 960), eyes=((1420, 800), (1550, 800)), bbox=(1250, 560, 1800, 1088)),
    ], grade=HALFTONE, notes="后段 wink（单眼闭）")],
    exclusions=BORDER(3818, 3868),
    effects=[FX("zoom", 3818, 3868, "缓慢推近"), FX("particles", 3818, 3868)],
    confidence=0.5)

# s15 多窗口 → 景深
S = 1194
SHOTS[S] = dict(
    instances=[
        I("left", "side", 1, [
            A(S, 0, head=(190, 80, 262, 150), eyes=((215, 120), (240, 120)), bbox=(175, 75, 300, 445)),
            A(S, 1, head=(1110, 60, 1300, 260), eyes=((1180, 198), (1250, 190)), bbox=(1045, 55, 1370, 500)),
            A(S, 2, head=(150, 610, 330, 760), bbox=(130, 600, 380, 1088), blurSigma=8),
        ], mode="linear", notes="1194 为细线框内的全身（短裙+尾巴），约 1250~1300 放大成大半身 wink，再退到背景虚化。"
                                "细线框只是装饰：1320~1340 角色明显超出框外，所以不给 clip"),
        I("chibi", "side", 1, [
            V(1229, hidden=True),
            V(1236, bbox=[880, 80, 160, 560]),
            A(S, 1, head=(1540, 90, 1800, 290), eyes=((1655, 255), (1740, 255)), bbox=(1525, 80, 1810, 495)),
            A(S, 2, head=(600, 680, 850, 900), bbox=(580, 670, 880, 1088), blurSigma=8),
        ], sw=[SW(1264, "Clothes2", 1, 0.5, "约 1264~1272 由常规全身变成 Q 版（实测约 8 帧完成）", fade=8)], mode="linear",
            notes="右侧实例：约 1230 以“翻牌”方式（先是竖直细条）出现，常规全身（短裙+尾巴）；约 1264 变 Q 版；后退为背景虚化"),
        I("center", "main", 2, [
            V(1330, hidden=True),
            A(S, 2, head=(380, 600, 660, 830), eyes=((478, 760), (560, 758)), bbox=(300, 600, 700, 1088)),
            A(S, 3, head=(1250, 545, 1850, 830), eyes=((1425, 830), (1540, 808)), bbox=(1150, 545, 1928, 1088)),
        ], variant={"Movement2": 1, "key7": 1}, sw=[SW(1450, "key4", 1, 0.3, "爱心眼出现时间不确定")], mode="linear",
            notes="抚胸+脸红，笑眼；后段推近成特写、爱心眼；出现时间约 1330（不确定）"),
    ],
    exclusions=[EX("effect", 1194, 1215, [(1194, R(S, 0, (0, 0, 460, 540)))])] + BORDER(1194, 1509),
    effects=[FX("depthOfField", 1340, 1509, "背景两个实例强虚化（blurSigma 约 8）"),
             FX("zoom", 1420, 1509, "推近"), FX("vignette", 1480, 1509, "四周模糊/暗角，接下一个转场")],
    confidence=0.35,
    notes="镜头内构图变化很大（窗口→景深三实例→特写），实例对应关系和出现时间是推测")

# s17 暗角半身
S = 1527
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, bbox=(0, 0, 960, 540), rotation=-35, blurSigma=4),
        A(S, 1, head=(1290, 30, 1580, 290), eyes=((1400, 240), (1492, 238)), bbox=(1260, 20, 1780, 540), rotation=-5, blurSigma=0),
        A(S, 2, head=(340, 560, 650, 800), eyes=((468, 712), (560, 712)), bbox=(300, 550, 700, 1088), rotation=0),
        A(S, 3, head=(1180, 545, 1720, 800), eyes=((1360, 658), (1540, 652)), bbox=(1150, 545, 1928, 1088)),
    ], mode="linear", grade="整体压暗约 30%、去饱和偏青灰；四周重暗角",
        notes="开头为旋转模糊的极近特写（1527~约 1545）；末段闭眼张嘴，推近")],
    exclusions=[EX("effect", 1527, 1535, [(1527, R(S, 0, (420, 270, 660, 540)))])],
    effects=[FX("vignette", 1527, 1707), FX("colorGrade", 1527, 1707, "压暗去饱和"),
             FX("motionBlur", 1527, 1548, "旋转进入的模糊"), FX("zoom", 1650, 1707, "推近")],
    confidence=0.45)

# s18 极近特写 → 模糊发白
S = 1707
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(240, 0, 840, 280), eyes=((395, 95), (690, 95)), bbox=(0, 0, 960, 540), blurSigma=1),
        A(S, 1, head=(1200, 0, 1700, 230), eyes=((1330, 70), (1560, 70)), bbox=(964, 0, 1928, 540), blurSigma=4),
        A(S, 3, bbox=(964, 544, 1928, 1088), blurSigma=20),
    ], mode="linear", grade="开头整体压暗、重暗角；随后迅速提亮发白、去饱和",
        notes="闭眼张嘴；眼睛位置为估计（眼睛被刘海遮住）；1715 后模糊很重，基本不可用于拟合")],
    exclusions=[],
    effects=[FX("vignette", 1707, 1716), FX("gaussianBlur", 1710, 1726, "模糊逐渐加重到 sigma≈20"),
             FX("fadeToWhite", 1712, 1726, "整体提亮发白，衔接下一镜头")],
    confidence=0.35)

# s19 钢盆 + 眼镜滑入
S = 1726
SHOTS[S] = dict(
    instances=[
        I("bowl", "main", 1, [
            V(1745, bbox=[560, 60, 720, 660], opacity=0.6),
            A(S, 1, head=(1440, 180, 1810, 360), eyes=((1520, 212), (1610, 212)), bbox=(1380, 45, 1928, 540), opacity=1.0),
            A(S, 2, head=(480, 700, 860, 880), eyes=((630, 790), (745, 800)), bbox=(450, 590, 960, 1088)),
            A(S, 3, head=(1400, 690, 1880, 890), eyes=((1515, 760), (1640, 775)), bbox=(1370, 545, 1928, 1088)),
        ], variant={"key10": 1}, notes="头部框不含钢盆；1726~1745 从白闪中淡入（opacity 为粗估）"),
        I("glasses", "side", 2, [
            V(1835, hidden=True),
            A(S, 2, bbox=(0, 545, 470, 1088), blurSigma=12),
            A(S, 3, head=(1000, 545, 1320, 800), eyes=((1115, 690), (1215, 680)), bbox=(964, 545, 1400, 1088), blurSigma=0),
        ], variant={"key11": 1, "Movement3": 1}, mode="linear", notes="约 1836 从左侧带运动模糊滑入，1846 后清晰；竖食指"),
    ],
    exclusions=BORDER(1726, 1899),
    effects=[FX("fadeFromWhite", 1726, 1746, "白闪淡入"), FX("motionBlur", 1836, 1846, "眼镜实例滑入")],
    confidence=0.5)

# s21 设定稿底图：左大 + 右全身（钢盆、圈圈眼、狗尾草）
S = 1918
SHOTS[S] = dict(
    instances=[
        I("main", "main", 2, [
            A(S, 0, head=(120, 0, 560, 330), eyes=((370, 160), (455, 155)), bbox=(60, 0, 700, 540), rotation=12),
            A(S, 1, head=(1150, 30, 1460, 330), eyes=((1230, 215), (1310, 220)), bbox=(970, 20, 1520, 540), rotation=0),
            A(S, 2, head=(130, 560, 520, 880), eyes=((270, 805), (370, 800)), bbox=(0, 545, 600, 1088)),
            A(S, 3, head=(1080, 560, 1470, 900), eyes=((1220, 800), (1335, 790)), bbox=(964, 545, 1560, 1088)),
        ], sw=[SW(1950, "Movement2", 1, 0.3, "抚胸手出现时间未逐帧确认")], notes="开头闭眼张嘴、头部右倾"),
        I("side", "fullBody", 1, [
            A(S, 0, head=(790, 70, 870, 130), bbox=(715, 40, 950, 520)),
            A(S, 1, head=(1560, 70, 1680, 160), eyes=((1595, 105), (1640, 105)), bbox=(1540, 30, 1720, 520)),
            A(S, 2, head=(590, 610, 700, 720), eyes=((620, 655), (670, 655)), bbox=(570, 575, 780, 1088)),
            A(S, 3, head=(1580, 830, 1680, 910), eyes=((1610, 855), (1650, 855)), bbox=(1540, 800, 1720, 1088)),
        ], variant={"key10": 1, "key3": 1, "Movement1": 1}, mode="linear",
            mh=[MH("狗尾草切换出来", 1918, 0.3)], notes="短裙+尾巴+钢盆+圈圈眼，手拿狗尾草；后段下移出画"),
    ],
    exclusions=[EX("effect", 1918, 1936, [(1918, R(S, 0, (0, 0, 960, 540)))]),
                EX("text", 1918, 2137, [(1991, R(S, 1, (1670, 325, 1880, 375))), (2064, R(S, 2, (750, 860, 915, 920))),
                                         (2136, R(S, 3, (1670, 860, 1880, 920)))])],
    effects=[FX("other", 1918, 1936, "白色颜料泼溅转场残留")],
    confidence=0.5)

# s22 设定稿：左大（脸红→爱心眼）+ 右全身
S = 2137
SHOTS[S] = dict(
    instances=[
        I("main", "main", 2, [
            A(S, 0, head=(60, 0, 560, 300), eyes=((235, 175), (355, 170)), bbox=(0, 0, 720, 540)),
            A(S, 1, head=(1110, 40, 1450, 330), eyes=((1225, 280), (1315, 280)), bbox=(1030, 30, 1550, 540)),
            A(S, 2, head=(100, 560, 520, 880), eyes=((275, 825), (395, 825)), bbox=(0, 545, 600, 1088)),
            A(S, 3, head=(1000, 545, 1500, 900), eyes=((1200, 835), (1330, 800)), bbox=(964, 545, 1560, 1088)),
        ], variant={"key7": 1}, sw=[SW(2280, "key4", 1, 0.4, "爱心眼约 2280 出现")]),
        I("side", "fullBody", 1, [
            A(S, 0, bbox=(600, 330, 700, 470), blurSigma=3),
            A(S, 1, head=(1580, 60, 1700, 180), eyes=((1610, 140), (1650, 140)), bbox=(1520, 50, 1760, 520), blurSigma=0),
            A(S, 2, head=(600, 580, 720, 700), eyes=((625, 665), (675, 665)), bbox=(575, 575, 790, 1088)),
            A(S, 3, head=(1570, 580, 1700, 700), eyes=((1595, 655), (1645, 655)), bbox=(1520, 570, 1760, 1088)),
        ], mode="linear", notes="短裙+尾巴；笑眼"),
    ],
    exclusions=[EX("text", 2137, 2304, [(2137, R(S, 0, (705, 325, 915, 375))), (2192, R(S, 1, (1670, 325, 1880, 375))),
                                         (2248, R(S, 2, (705, 860, 915, 920)))]),
                EX("effect", 2290, 2304, [(2303, R(S, 3, (1100, 600, 1928, 1000)))])],
    effects=[FX("particles", 2290, 2304, "白色速度线划过，衔接下一镜头")],
    confidence=0.5)

# s23 特写：? + 加载 + 圈圈眼
S = 2304
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(0, 0, 900, 500), eyes=((310, 120), (540, 190)), bbox=(0, 0, 960, 540), rotation=20, blurSigma=3),
        A(S, 1, head=(1030, 0, 1760, 470), eyes=((1345, 220), (1512, 225)), bbox=(964, 0, 1928, 540), rotation=0, blurSigma=0),
        A(S, 2, head=(180, 545, 830, 920), eyes=((515, 735), (680, 690)), bbox=(0, 545, 960, 1088), rotation=-10),
        A(S, 3, head=(1180, 580, 1650, 900), eyes=((1450, 770), (1550, 760)), bbox=(1100, 570, 1800, 1088), rotation=0),
    ], sw=[SW(2318, "key8", 1, 0.4, "加载图标约 2318 出现"), SW(2335, "key3", 1, 0.5, "圈圈眼约 2335 出现")],
        mh=[MH("脱捕", 2300, 0.3, "key8→key3 的出现顺序与 脱捕 一致，但未见狗尾草")],
        notes="开头带运动模糊的倾斜大特写，后拉远")],
    exclusions=[EX("question", 2318, 2368, [(2325, R(S, 1, (1150, 60, 1360, 330))), (2346, R(S, 2, (80, 610, 400, 880))),
                                             (2367, R(S, 3, (1070, 605, 1360, 875)))]),
                EX("effect", 2304, 2312, [(2304, R(S, 0, (100, 80, 800, 480)))]),
                EX("effect", 2360, 2368, [(2367, R(S, 3, (1300, 740, 1640, 850)))])] + BORDER(2304, 2368),
    effects=[FX("motionBlur", 2304, 2312), FX("zoom", 2304, 2330, "拉远")],
    confidence=0.45)

# s24 特写
S = 2368
SHOTS[S] = dict(
    instances=[I("main", "main", 1, [
        A(S, 0, head=(0, 30, 720, 500), eyes=((200, 215), (380, 190)), bbox=(0, 0, 900, 540), rotation=15),
        A(S, 1, head=(1220, 20, 1690, 330), eyes=((1350, 190), (1460, 190)), bbox=(1100, 0, 1830, 540), rotation=0),
        A(S, 2, head=(210, 545, 880, 950), eyes=((480, 745), (660, 740)), bbox=(100, 545, 960, 1088)),
        A(S, 3, head=(1170, 575, 1830, 950), eyes=((1440, 745), (1590, 745)), bbox=(1050, 560, 1928, 1088)),
    ])],
    exclusions=[EX("effect", 2368, 2376, [(2368, R(S, 0, (100, 60, 500, 300)))]),
                EX("effect", 2438, 2442, [(2441, R(S, 3, (1080, 585, 1820, 630)))])] + BORDER(2368, 2442),
    effects=[FX("zoom", 2368, 2442, "先拉远后推近"), FX("particles", 2368, 2392, "白色划痕粒子扫过脸和肩（2026-11-02 总负责人补标）")],
    confidence=0.5)

# s25 圆角窗口：单片眼镜
S = 2442
SHOTS[S] = dict(
    instances=[I("main", "pip", 1, [
        A(S, 0, head=(160, 0, 800, 240), bbox=(100, 0, 860, 540)),
        A(S, 1, head=(1150, 60, 1750, 470), eyes=((1355, 298), (1530, 240)), bbox=(1070, 50, 1800, 470), rotation=-20),
        A(S, 2, head=(180, 575, 800, 1020), eyes=((390, 925), (560, 875)), bbox=(105, 575, 840, 1020), rotation=-15),
        A(S, 3, head=(1110, 580, 1760, 1020), eyes=((1300, 870), (1460, 870)), bbox=(1070, 580, 1800, 1020), rotation=0),
    ], variant={"key12": 1}, clip={"rect": R(S, 2, (105, 585, 840, 1020)), "radius": 25},
        notes="2442~2445 窗口从上方展开，窗口外是上一镜头；单片眼镜在画面右眼")],
    exclusions=[EX("other-shot", 2442, 2446, [(2442, R(S, 0, (0, 230, 960, 540)))]),
                EX("effect", 2515, 2530, [(2529, R(S, 3, (1200, 800, 1700, 1020)))]),
                EX("check", 2520, 2530, [(2529, R(S, 3, (1600, 915, 1670, 985)))])] + BORDER(2442, 2530),
    effects=[FX("zoom", 2442, 2530, "缓慢推近/旋转")],
    confidence=0.5)

# s26 圆角窗口：><眼 + 脸红 + 红勾
S = 2530
SHOTS[S] = dict(
    instances=[I("main", "pip", 1, [
        A(S, 0, head=(190, 40, 810, 480), bbox=(105, 40, 840, 480), rotation=-25),
        A(S, 1, head=(1100, 40, 1700, 470), eyes=((1300, 270), (1475, 270)), bbox=(1070, 40, 1800, 475), rotation=0),
        A(S, 2, head=(200, 580, 800, 1020), eyes=((465, 915), (620, 915)), bbox=(105, 580, 840, 1020)),
        A(S, 3, head=(1180, 580, 1780, 1020), eyes=((1430, 915), (1600, 880)), bbox=(1070, 580, 1800, 1020), rotation=-15),
    ], variant={"key5": 1, "key7": 1}, clip={"rect": R(S, 2, (105, 585, 840, 1020)), "radius": 25},
        notes="与上一镜头同一窗口；本镜头未见单片眼镜（上一镜头有），说明切换发生在白色笔刷遮挡期间")],
    exclusions=[EX("effect", 2530, 2540, [(2530, R(S, 0, (160, 300, 700, 480)))]),
                EX("check", 2530, 2608, [(2530, R(S, 0, (640, 345, 790, 470))), (2556, R(S, 1, (1600, 260, 1928, 490))),
                                          (2582, R(S, 2, (630, 800, 930, 1040))), (2607, R(S, 3, (1600, 800, 1900, 1040)))]),
                EX("effect", 2602, 2608, [(2607, R(S, 3, (964, 700, 1100, 900)))])] + BORDER(2530, 2608),
    effects=[FX("zoom", 2530, 2608)],
    confidence=0.5)

# s28 设定稿：左躯干 + 右窗口全身（短裙）
S = 2614
SHOTS[S] = dict(
    instances=[
        I("body", "side", 1, [A(S, 0, bbox=(0, 0, 500, 540)), A(S, 1, bbox=(964, 0, 1460, 540)),
                              A(S, 3, bbox=(1030, 545, 1480, 1088))], mode="linear", notes="腰部到大腿特写，无头部；短裙"),
        I("full", "fullBody", 2, [
            A(S, 0, head=(680, 40, 760, 110), eyes=((710, 85), (740, 85)), bbox=(630, 30, 830, 510)),
            A(S, 1, head=(1640, 40, 1720, 110), eyes=((1665, 85), (1700, 85)), bbox=(1590, 30, 1790, 500)),
            A(S, 2, head=(680, 590, 770, 660), eyes=((705, 630), (740, 630)), bbox=(630, 580, 830, 1050)),
            A(S, 3, head=(1640, 590, 1720, 660), eyes=((1665, 625), (1700, 625)), bbox=(1600, 580, 1780, 1040)),
        ], clip={"rect": R(S, 1, (1565, 40, 1805, 505)), "radius": 8}, notes="闭眼；短裙"),
    ],
    exclusions=[EX("wipe", 2614, 2618, [(2614, R(S, 0, (720, 0, 960, 540)))])] + BORDER(2614, 2639),
    effects=[],
    confidence=0.55)

# s29 设定稿：左躯干 + 右窗口全身（长裙）
S = 2639
SHOTS[S] = dict(
    instances=[
        I("body", "side", 1, [A(S, 0, bbox=(0, 30, 540, 490)), A(S, 1, bbox=(964, 0, 1460, 540)),
                              A(S, 2, bbox=(0, 560, 500, 1088)), A(S, 3, bbox=(964, 544, 1928, 1088), blurSigma=20)],
          variant={"Clothes1": 0}, mode="linear", notes="腰部到裙摆特写，无头部；长裙（Clothes1=0）"),
        I("full", "fullBody", 2, [
            A(S, 0, head=(690, 60, 770, 130), eyes=((710, 100), (745, 100)), bbox=(630, 50, 830, 490)),
            A(S, 1, head=(1640, 50, 1720, 125), eyes=((1665, 95), (1700, 95)), bbox=(1580, 40, 1790, 495)),
            A(S, 2, head=(680, 595, 770, 670), eyes=((705, 640), (740, 640)), bbox=(620, 585, 830, 1050)),
            V(2689, hidden=True),
        ], variant={"Clothes1": 0}, clip={"rect": R(S, 1, (1560, 45, 1795, 495)), "radius": 8}, mode="linear",
            notes="长裙（Clothes1=0）"),
    ],
    exclusions=BORDER(2639, 2690),
    effects=[FX("radialBlur", 2680, 2690, "径向推镜模糊出场")],
    confidence=0.55)

# s30 设定稿：左大 + 右全身（长裙、单片眼镜）
S = 2690
SHOTS[S] = dict(
    instances=[
        I("main", "main", 2, [
            A(S, 0, head=(0, 0, 450, 330), bbox=(0, 0, 690, 540), blurSigma=12),
            V(2698, blurSigma=0),
            A(S, 1, head=(1150, 0, 1490, 300), eyes=((1290, 225), (1385, 220)), bbox=(1030, 0, 1560, 540), rotation=-10, blurSigma=0),
            A(S, 2, head=(170, 590, 470, 880), eyes=((270, 795), (355, 800)), bbox=(60, 580, 560, 1088), rotation=0),
            A(S, 3, head=(1000, 545, 1500, 950), eyes=((1180, 830), (1300, 810)), bbox=(964, 545, 1560, 1088)),
        ], variant={"key12": 1}, notes="2690~2697 模糊切入；单片眼镜在画面右眼；后段推近、wink"),
        I("full", "fullBody", 1, [
            A(S, 0, bbox=(690, 0, 960, 540), blurSigma=12),
            A(S, 1, head=(1600, 30, 1700, 120), eyes=((1625, 80), (1665, 80)), bbox=(1520, 25, 1790, 530), blurSigma=0),
            A(S, 2, head=(610, 590, 720, 690), eyes=((640, 640), (680, 640)), bbox=(560, 585, 790, 1088)),
            A(S, 3, head=(1620, 545, 1780, 700), eyes=((1660, 595), (1740, 595)), bbox=(1550, 545, 1928, 1088)),
        ], variant={"key12": 1, "Clothes1": 0}, mode="linear", notes="长裙+尾巴+单片眼镜"),
    ],
    exclusions=[EX("text", 2690, 2969, [(2783, R(S, 1, (1720, 320, 1880, 375))), (2876, R(S, 2, (780, 860, 915, 920))),
                                         (2968, R(S, 3, (1800, 880, 1928, 950)))])],
    effects=[FX("gaussianBlur", 2690, 2698, "模糊切入"), FX("zoom", 2876, 2969, "推近")],
    confidence=0.5)

# s32 设定稿：左大 + 右全身（长裙）
S = 3000
SHOTS[S] = dict(
    instances=[
        I("main", "main", 2, [
            A(S, 0, head=(140, 30, 500, 350), eyes=((300, 265), (415, 265)), bbox=(0, 0, 560, 540)),
            A(S, 1, head=(1140, 20, 1540, 380), eyes=((1255, 295), (1370, 285)), bbox=(964, 0, 1560, 540)),
            A(S, 2, head=(140, 570, 500, 880), eyes=((275, 810), (365, 815)), bbox=(0, 560, 560, 1088)),
            A(S, 3, head=(1130, 575, 1510, 900), eyes=((1265, 820), (1370, 810)), bbox=(964, 560, 1560, 1088)),
        ]),
        I("full", "fullBody", 1, [
            A(S, 0, head=(640, 20, 760, 140), eyes=((680, 100), (730, 100)), bbox=(580, 10, 880, 500)),
            A(S, 1, head=(1690, 20, 1820, 160), eyes=((1730, 115), (1770, 110)), bbox=(1620, 10, 1928, 540)),
            A(S, 2, head=(650, 570, 760, 700), eyes=((690, 640), (735, 635)), bbox=(610, 560, 860, 1088)),
            A(S, 3, head=(1570, 590, 1690, 700), eyes=((1600, 650), (1650, 650)), bbox=(1520, 580, 1760, 1088)),
        ], variant={"Clothes1": 0}, mode="linear", notes="长裙+尾巴"),
    ],
    exclusions=[EX("splash", 3000, 3012, [(3000, R(S, 0, (0, 0, 960, 540)))]),
                EX("text", 3000, 3097, [(3000, R(S, 0, (780, 330, 960, 400))), (3064, R(S, 2, (870, 860, 960, 920))),
                                         (3096, R(S, 3, (1720, 860, 1880, 920)))]),
                EX("effect", 3088, 3097, [(3096, R(S, 3, (1200, 640, 1500, 760)))])],
    effects=[FX("other", 3000, 3012, "颜料泼溅转场残留")],
    confidence=0.55)

# ---------------------------------------------------------------- 锚点精修（第二轮）
# 第一轮从 0.75 缩放网格图读数误差较大（20~60px）。第二轮在 review/fix_<shot>.jpg
# （0.5 缩放、带视频像素坐标网格的整帧图）上逐个重读，直接写视频像素：
#   (镜头起点, 实例, 帧) -> (head 的 x0,y0,x1,y1, eyes 或 None, 可选 bbox 的 x0,y0,x1,y1)
# 读数误差约 ±6px；eyes 之后再由 build.py 用虹膜颜色自动细化（成功则到 ±2px 级）。
# 未提供 bbox 时，bbox 按 head 中心的修正量整体平移。
FIX = {
    (254, "main", 294): ((460, 270, 740, 410), ((570, 344), (680, 344)), None),
    (254, "main", 335): ((490, 260, 750, 430), ((590, 320), (690, 310)), None),
    (254, "main", 375): ((480, 200, 780, 390), ((570, 236), (670, 236)), None),
    (400, "main", 400): ((440, 60, 920, 430), ((680, 296), (820, 310)), None),
    (400, "main", 484): ((360, 20, 800, 400), ((444, 236), (550, 236)), None),
    (400, "main", 569): ((430, 90, 930, 390), ((610, 210), (740, 200)), None),
    (400, "main", 653): ((420, 160, 970, 410), ((570, 326), (756, 326)), None),
    (400, "pipL", 484): ((150, 160, 260, 260), ((194, 200), (234, 200)), (130, 150, 330, 370)),
    (400, "pipL", 569): ((60, 100, 310, 280), ((130, 196), (240, 196)), (0, 100, 360, 400)),
    (400, "pipL", 653): ((80, 100, 360, 300), ((146, 220), (262, 220)), (0, 100, 420, 420)),
    (400, "pipR", 569): ((970, 310, 1140, 430), ((1030, 360), (1090, 360)), (940, 300, 1180, 620)),
    (400, "pipR", 653): ((940, 320, 1110, 470), ((990, 436), (1044, 436)), (900, 300, 1220, 620)),
    (684, "bowl", 715): ((90, 180, 400, 440), ((210, 300), (300, 310)), None),
    (684, "bowl", 746): ((80, 200, 380, 450), ((160, 310), (250, 310)), None),
    (684, "bowl", 777): ((70, 200, 380, 440), ((160, 300), (250, 296)), None),
    (684, "loading", 715): ((410, 120, 760, 500), ((520, 390), (665, 385)), None),
    (684, "loading", 746): ((480, 80, 780, 440), ((610, 304), (700, 300)), None),
    (684, "loading", 777): ((430, 70, 740, 440), ((590, 320), (676, 296)), None),
    (684, "wink", 746): ((780, 0, 1280, 440), ((916, 254), (1050, 250)), (760, 0, 1280, 720)),
    (684, "wink", 777): ((740, 40, 1280, 560), ((870, 440), (1030, 430)), (720, 40, 1280, 720)),
    (795, "right", 795): ((600, 0, 1120, 400), ((860, 200), (994, 200)), (560, 0, 1280, 720)),
    (795, "right", 846): ((580, 0, 1160, 400), ((720, 260), (930, 260)), (520, 0, 1280, 720)),
    (795, "right", 898): ((600, 0, 1200, 400), ((760, 220), (910, 210)), (560, 0, 1280, 720)),
    (795, "right", 949): ((460, 0, 1200, 560), ((720, 320), (970, 320)), (440, 0, 1280, 720)),
    (1194, "left", 1194): ((270, 120, 350, 200), ((294, 164), (320, 164)), (220, 110, 400, 600)),
    (1194, "left", 1299): ((210, 110, 460, 370), ((266, 270), (344, 256)), (150, 100, 520, 660)),
    (1194, "left", 1404): ((200, 80, 440, 280), ((280, 190), (360, 190)), (160, 60, 500, 720)),
    (1194, "chibi", 1299): ((780, 140, 1120, 420), ((920, 326), (994, 326)), (760, 130, 1140, 660)),
    (1194, "chibi", 1404): ((800, 180, 1120, 480), ((880, 320), (1000, 320)), (760, 160, 1180, 720)),
    (1194, "center", 1404): ((520, 80, 820, 400), ((624, 274), (704, 274)), (400, 60, 880, 720)),
    (1194, "center", 1508): ((380, 0, 1020, 400), ((556, 160), (760, 156)), (100, 0, 1280, 720)),
    (1527, "main", 1587): ((450, 40, 890, 400), ((600, 316), (710, 316)), None),
    (1527, "main", 1647): ((480, 20, 860, 350), ((620, 220), (740, 220)), None),
    (1527, "main", 1706): ((300, 0, 1040, 360), ((520, 150), (770, 144)), None),
    (1707, "main", 1707): ((320, 0, 1120, 370), ((540, 150), (880, 150)), None),
    (1707, "main", 1713): ((320, 0, 980, 310), ((490, 90), (800, 90)), None),
    (1726, "bowl", 1783): ((640, 240, 1120, 490), ((724, 290), (830, 290)), None),
    (1726, "bowl", 1841): ((640, 220, 1140, 450), ((870, 296), (996, 260)), None),
    (1726, "bowl", 1898): ((580, 190, 1220, 460), ((740, 244), (900, 300)), None),
    (1726, "glasses", 1898): ((40, 20, 480, 340), ((176, 190), (320, 160)), (0, 0, 560, 720)),
    (1918, "main", 1918): ((160, 0, 740, 440), ((444, 216), (570, 210)), None),
    (1918, "main", 1991): ((250, 50, 650, 440), ((390, 300), (480, 296)), None),
    (1918, "main", 2064): ((150, 20, 660, 360), ((360, 270), (480, 264)), None),
    (1918, "main", 2136): ((160, 30, 670, 480), ((330, 260), (484, 260)), None),
    (1918, "side", 1918): ((1050, 80, 1130, 170), ((1070, 126), (1106, 126)), (960, 40, 1280, 660)),
    (1918, "side", 1991): ((820, 100, 950, 210), ((864, 154), (912, 154)), (760, 40, 1060, 690)),
    (1918, "side", 2064): ((800, 70, 930, 170), ((840, 130), (890, 130)), (740, 40, 1060, 720)),
    (1918, "side", 2136): ((820, 380, 950, 470), ((864, 420), (906, 420)), (780, 380, 1020, 720)),
    (2137, "main", 2137): ((80, 0, 740, 400), ((330, 250), (500, 250)), None),
    (2137, "main", 2192): ((200, 50, 640, 440), ((376, 360), (500, 356)), None),
    (2137, "main", 2248): ((130, 20, 700, 450), ((370, 380), (500, 380)), None),
    (2137, "main", 2303): ((50, 0, 720, 480), ((314, 312), (492, 330)), None),
    (2137, "side", 2192): ((820, 80, 980, 220), ((880, 134), (940, 134)), (780, 60, 1060, 700)),
    (2137, "side", 2248): ((810, 40, 960, 200), ((860, 110), (910, 110)), (760, 40, 1060, 700)),
    (2137, "side", 2303): ((810, 50, 980, 220), ((850, 150), (910, 150)), (740, 40, 1080, 700)),
    (2304, "main", 2304): ((0, 0, 1200, 660), ((414, 160), (690, 256)), None),
    (2304, "main", 2325): ((90, 20, 1060, 640), ((524, 300), (756, 306)), None),
    (2304, "main", 2346): ((240, 20, 1100, 500), ((686, 254), (904, 194)), None),
    (2304, "main", 2367): ((360, 80, 920, 480), ((644, 296), (780, 286)), None),
    (2368, "main", 2368): ((0, 40, 960, 660), ((270, 280), (500, 260)), None),
    (2368, "main", 2392): ((340, 30, 960, 440), ((510, 260), (656, 260)), None),
    (2368, "main", 2417): ((280, 20, 1170, 540), ((650, 250), (860, 250)), None),
    (2368, "main", 2441): ((270, 40, 1150, 540), ((630, 210), (840, 210)), None),
    (2442, "main", 2442): ((280, 0, 1120, 300), ((510, 80), (750, 100)), None),
    (2442, "main", 2471): ((240, 80, 1050, 620), ((510, 390), (750, 316)), None),
    (2442, "main", 2500): ((280, 50, 1050, 640), ((524, 490), (744, 444)), None),
    (2442, "main", 2529): ((200, 50, 1040, 600), ((450, 436), (670, 436)), None),
    (2530, "main", 2530): ((250, 60, 1080, 640), None, None),
    (2530, "main", 2556): ((160, 50, 980, 620), ((440, 366), (650, 366)), None),
    (2530, "main", 2582): ((240, 50, 1060, 640), ((610, 490), (840, 496)), None),
    (2530, "main", 2607): ((280, 50, 1080, 660), ((610, 510), (840, 450)), None),
    (2614, "full", 2614): ((910, 50, 1020, 160), ((940, 114), (990, 114)), (850, 40, 1100, 680)),
    (2614, "full", 2622): ((900, 50, 1010, 160), ((936, 114), (980, 114)), (840, 40, 1060, 670)),
    (2614, "full", 2630): ((910, 50, 1024, 160), ((952, 116), (996, 116)), (840, 40, 1090, 660)),
    (2614, "full", 2638): ((900, 50, 1010, 160), ((936, 116), (980, 116)), (840, 40, 1060, 660)),
    (2639, "full", 2639): ((920, 80, 1030, 180), ((940, 130), (994, 130)), (850, 70, 1110, 660)),
    (2639, "full", 2656): ((900, 70, 1010, 180), ((940, 136), (984, 136)), (810, 60, 1090, 660)),
    (2639, "full", 2673): ((910, 70, 1030, 180), ((960, 130), (994, 130)), (840, 60, 1110, 670)),
    (2690, "main", 2690): ((0, 0, 600, 440), ((120, 220), (320, 250)), None),
    (2690, "main", 2783): ((250, 10, 700, 400), ((436, 310), (532, 296)), None),
    (2690, "main", 2876): ((230, 60, 630, 450), ((360, 334), (460, 320)), None),
    (2690, "main", 2968): ((50, 10, 720, 540), ((290, 386), (440, 350)), None),
    (2690, "full", 2783): ((850, 50, 980, 160), ((890, 104), (930, 104)), (740, 40, 1080, 710)),
    (2690, "full", 2876): ((820, 70, 950, 180), ((866, 130), (904, 130)), (720, 60, 1060, 720)),
    (2690, "full", 2968): ((880, 10, 1080, 210), ((930, 70), (1020, 70)), (780, 0, 1280, 720)),
    (3000, "main", 3000): ((190, 40, 660, 470), ((410, 344), (544, 340)), None),
    (3000, "main", 3032): ((230, 30, 770, 500), ((370, 410), (530, 380)), None),
    (3000, "main", 3064): ((190, 30, 690, 460), ((356, 360), (474, 370)), None),
    (3000, "main", 3096): ((130, 60, 640, 500), ((384, 400), (510, 380)), None),
    (3000, "full", 3000): ((860, 30, 1020, 190), ((904, 136), (984, 136)), (790, 20, 1200, 700)),
    (3000, "full", 3032): ((970, 30, 1140, 210), ((1020, 146), (1080, 146)), (900, 20, 1280, 720)),
    (3000, "full", 3064): ((870, 40, 1020, 210), ((936, 126), (990, 126)), (800, 30, 1160, 720)),
    (3000, "full", 3096): ((810, 60, 970, 210), ((856, 144), (912, 144)), (740, 50, 1100, 720)),
    (3097, "main", 3097): ((510, 190, 830, 560), ((600, 444), (720, 444)), None),
    (3097, "main", 3113): ((480, 40, 800, 350), ((626, 214), (726, 214)), None),
    (3097, "main", 3129): ((500, 70, 880, 420), ((604, 260), (724, 260)), None),
    (3097, "main", 3144): ((500, 60, 800, 340), ((594, 254), (710, 254)), None),
    (3145, "main", 3145): ((530, 230, 770, 490), ((620, 410), (684, 410)), None),
    (3145, "main", 3160): ((510, 70, 790, 340), ((600, 224), (690, 224)), None),
    (3145, "main", 3175): ((520, 70, 840, 420), ((630, 260), (700, 260)), None),
    (3145, "main", 3190): ((530, 140, 870, 470), ((634, 256), (710, 256)), None),
    (3191, "main", 3191): ((440, 220, 810, 560), ((586, 410), (684, 410)), None),
    (3191, "main", 3207): ((500, 90, 790, 370), ((590, 250), (680, 250)), None),
    (3241, "main", 3253): ((320, 60, 940, 470), ((564, 346), (710, 320)), None),
    (3241, "main", 3284): ((380, 30, 880, 450), ((544, 370), (690, 340)), None),
    (3303, "main", 3303): ((160, 0, 670, 440), ((360, 314), (490, 314)), None),
    (3303, "main", 3329): ((160, 0, 670, 440), ((360, 340), (480, 310)), None),
    (3303, "main", 3355): ((140, 30, 660, 480), ((350, 416), (474, 400)), None),
    (3303, "main", 3381): ((150, 30, 630, 460), ((360, 334), (460, 334)), None),
    (3303, "side", 3303): ((860, 170, 930, 260), ((880, 210), (910, 210)), (820, 170, 980, 650)),
    (3303, "side", 3329): ((850, 320, 950, 410), ((880, 364), (916, 364)), (830, 310, 1040, 700)),
    (3303, "side", 3355): ((850, 430, 940, 510), ((880, 480), (910, 480)), (820, 420, 960, 720)),
    (3303, "side", 3381): ((860, 540, 940, 620), ((886, 590), (916, 590)), (840, 530, 960, 720)),
    (3399, "chibi", 3425): ((180, 50, 640, 440), ((344, 366), (450, 366)), (100, 40, 700, 720)),
    (3399, "chibi", 3451): ((200, 90, 650, 480), ((344, 390), (474, 390)), (120, 80, 720, 720)),
    (3399, "full", 3425): ((830, 80, 980, 220), ((880, 156), (924, 156)), (760, 70, 1060, 710)),
    (3399, "full", 3451): ((820, 80, 980, 230), ((880, 160), (930, 160)), (760, 70, 1060, 710)),
    (3478, "main", 3539): ((180, 0, 660, 440), ((350, 296), (464, 280)), None),
    (3478, "main", 3600): ((170, 0, 590, 450), ((310, 280), (430, 280)), None),
    (3478, "main", 3661): ((90, 0, 610, 540), ((370, 220), (510, 204)), None),
    (3478, "full", 3539): ((840, 0, 980, 150), ((880, 70), (920, 70)), (760, 0, 1060, 720)),
    (3478, "full", 3600): ((810, 10, 960, 210), ((890, 90), (920, 90)), (740, 0, 1080, 720)),
    (3684, "main", 3684): ((440, 80, 930, 440), ((614, 330), (720, 326)), None),
    (3684, "main", 3696): ((430, 80, 850, 400), ((584, 266), (700, 266)), None),
    (3684, "main", 3709): ((480, 70, 870, 420), ((594, 240), (716, 260)), None),
    (3684, "main", 3721): ((490, 80, 900, 420), ((606, 236), (726, 256)), None),
    (3722, "main", 3722): ((480, 200, 870, 570), ((600, 430), (704, 430)), None),
    (3722, "main", 3737): ((540, 50, 850, 400), ((654, 226), (756, 256)), None),
    (3722, "main", 3752): ((530, 60, 890, 430), ((710, 245), (800, 292)), None),
    (3722, "main", 3767): ((550, 70, 900, 420), ((704, 240), (794, 292)), None),
    (3768, "main", 3768): ((450, 300, 830, 610), ((555, 424), (640, 395)), None),
    (3768, "main", 3784): ((470, 190, 740, 440), ((576, 300), (690, 300)), None),
    (3768, "main", 3801): ((490, 230, 850, 450), ((600, 264), (730, 264)), None),
    (3768, "main", 3817): ((530, 230, 850, 450), ((610, 256), (730, 256)), None),
    (3818, "main", 3818): ((470, 90, 860, 460), ((634, 300), (744, 310)), None),
    (3818, "main", 3834): ((380, 0, 1010, 440), ((596, 286), (764, 276)), None),
    (3818, "main", 3851): ((440, 30, 1010, 540), ((592, 330), (750, 300)), None),
    (3818, "main", 3867): ((440, 40, 980, 550), ((590, 320), (740, 280)), None),
}
# 新增锚点（第一轮没有的帧）：(镜头起点, 实例) -> [V(...)]
EXTRA = {
    (3303, "side"): [V(3303, head=[860, 170, 70, 90], eyes=[[880, 210], [910, 210]], bbox=[820, 170, 160, 480])],
    (3478, "full"): [V(3661, bbox=[860, 0, 420, 720])],
}
# 画中画窗口裁切修正（视频像素 [x,y,w,h]）
CLIP_FIX = {
    (2442, "main"): {"rect": [140, 50, 830, 540], "radius": 25},
    (2530, "main"): {"rect": [140, 50, 830, 540], "radius": 25},
    (2614, "full"): {"rect": [800, 50, 310, 620], "radius": 12},
    (2639, "full"): {"rect": [800, 60, 330, 610], "radius": 12},
}


def apply_fixes():
    for (s, iid, f), (hd, ey, bb) in FIX.items():
        inst = next(i for i in SHOTS[s]["instances"] if i["id"] == iid)
        a = next((a for a in inst["anchors"] if a["frame"] == f), None)
        if a is None:
            a = {"frame": f}
            inst["anchors"].append(a)
            inst["anchors"].sort(key=lambda a: a["frame"])
        new_head = [hd[0], hd[1], hd[2] - hd[0], hd[3] - hd[1]]
        if bb is not None:
            a["bbox"] = [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]]
        elif a.get("bbox") and a.get("head"):
            ox, oy = a["head"][0] + a["head"][2] / 2, a["head"][1] + a["head"][3] / 2
            dx, dy = hd[0] + new_head[2] / 2 - ox, hd[1] + new_head[3] / 2 - oy
            x, y, w, h = a["bbox"]
            x0, y0 = max(0, x + dx), max(0, y + dy)
            a["bbox"] = [round(x0, 1), round(y0, 1), round(min(1280, x + dx + w) - x0, 1), round(min(720, y + dy + h) - y0, 1)]
        a["head"] = new_head
        if ey is not None:
            a["eyes"] = [list(ey[0]), list(ey[1])]
        else:
            a.pop("eyes", None)
    for (s, iid), vs in EXTRA.items():
        inst = next(i for i in SHOTS[s]["instances"] if i["id"] == iid)
        have = {a["frame"] for a in inst["anchors"]}
        inst["anchors"] += [v for v in vs if v["frame"] not in have]
        inst["anchors"].sort(key=lambda a: a["frame"])
    for (s, iid), c in CLIP_FIX.items():
        inst = next(i for i in SHOTS[s]["instances"] if i["id"] == iid)
        inst["clip"] = c


# ================================================================ 第二版（审查意见修订）
# 1) 转场与角色镜头边界精确到帧（review/tr_*.jpg、onA/onB/c_all/c_new.jpg 逐帧确认）；
# 2) 转场镜头里角色仍可见的部分：加实例（continues 指向上一镜头实例），track 用 transitions.py 的
#    “与转场前干净帧比对”结果，覆盖到角色完全被遮住；遮挡物随帧扩大的矩形写成排除区；
# 3) 实例 ID 全局唯一：<镜头ID>.<名字>，continues / effects.instances 都用全局 ID；
# 4) clip 改成关键帧数组；effects 加 instances / strength。
#
# BOUNDS（上面）只用于把第一轮网格图读数映射回帧号（grid_frames），不再决定镜头边界。

NEW_BOUNDS = [
    (0, "noCharacter", False, "cut", 0, "片头：点阵背景上聊天窗口弹出放大（纯 UI）"),
    (18, "noCharacter", False, "cut", 0, "窗口放大到全屏的白色画面（纯 UI）"),
    (34, "noCharacter", False, "cut", 0, "聊天窗口：鲸鱼 logo、逐字打出文字、输入框、鼠标光标点击（纯 UI）"),
    (254, "pip", True, "other", 14, "角色窗口（Photo Booth 样式）从聊天窗口中部缩放弹出，254~268 窗口在放大"),
    (372, "transition", True, "wipe", 22, "蓝色颜料泼溅擦除：372 首帧出现泼溅，角色约 390 完全被盖住；新画面 394 出现"),
    (394, "multi", True, "wipe", 10, "泼溅退出，394~403 新画面左侧仍有泼溅残留"),
    (652, "transition", True, "wipe", 24, "白色剪影擦除：652 首帧出现白色形状，角色约 664 完全被盖住；新画面背景 676 开始出现"),
    (676, "multi", True, "wipe", 8, "白底上点阵背景逐步显现（676~683），三个实例依次从左/下方滑入（约 689 起）"),
    (776, "transition", True, "wipe", 13, "白色剪影擦除：776 首帧出现白色形状，角色约 782 完全被盖住；新画面 789 出现"),
    (789, "multi", True, "wipe", 6, "两个实例从左下旋转扫入（789~794 带剪影残留）"),
    (948, "transition", True, "other", 14, "白色爆炸形剪影：948 首帧出现，角色约 956 完全被盖住；新画面 962 出现"),
    (962, "extremeCloseUp", True, "other", 8, "白色爆炸形剪影退出（962~969）后为裙子/大腿极近特写（无头部）"),
    (1041, "pip", True, "other", 6, "圆角窗口从画面中心放大弹出覆盖上一镜头（1041~1046 窗口外仍是上一镜头的画面）"),
    (1157, "transition", True, "wipe", 30, "蓝色颜料泼溅擦除：1157 首帧泼溅从右侧进入，角色约 1177 完全被盖住；新画面 1187 出现"),
    (1187, "multi", True, "wipe", 12, "泼溅退出（1187~1198）"),
    (1509, "transition", True, "other", 18, "旋转/万花筒模糊转场：前半段是上一镜头（中间实例）被旋转扭曲，约 1517 起混入下一镜头画面"),
    (1527, "halfBody", True, "other", 0, ""),
    (1707, "closeUp", True, "cut", 0, ""),
    (1726, "multi", True, "cut", 0, ""),
    (1896, "transition", True, "other", 14, "白色爆炸形剪影：1896 首帧出现，角色约 1904 完全被盖住；新画面 1910 出现"),
    (1910, "multi", True, "other", 12, "灰白爆炸形剪影退出（1910~1921）"),
    (2137, "multi", True, "cut", 0, ""),
    (2304, "closeUp", True, "cut", 0, ""),
    (2368, "closeUp", True, "cut", 0, ""),
    (2442, "closeUp", True, "wipe", 4, "圆角窗口从上方展开覆盖上一镜头，2442~2445 窗口外仍是上一镜头"),
    (2530, "closeUp", True, "other", 0, "白色笔刷线条遮挡下的切换（2526 起线条已出现在上一镜头）"),
    (2608, "transition", True, "wipe", 6, "翻页式斜向擦除：2608~2613 旧画面（左上）与新画面（右下）同时可见"),
    (2614, "multi", True, "wipe", 0, ""),
    (2639, "multi", True, "cut", 0, ""),
    (2690, "multi", True, "other", 8, "模糊切入：2690 起清晰度逐渐恢复，约 2698 清晰"),
    (2969, "transition", True, "wipe", 13, "深蓝颜料泼溅擦除：2969 首帧出现，角色约 2981 完全被盖住；新画面 2982 起从泼溅缝隙中显现"),
    (2982, "multi", True, "wipe", 22, "泼溅逐步退出（2982~3003）：左侧大特写 + 右侧全身"),
    (3097, "halfBody", True, "cut", 0, "3097~3104 快速推拉（同一镜头内连续缩放，非切点）"),
    (3145, "halfBody", True, "cut", 0, ""),
    (3191, "halfBody", True, "cut", 0, "3191~3197 快速拉远"),
    (3241, "fullBody", True, "cut", 0, "硬切到 Q 版（光流残差检测补出的切点，3240→3241）"),
    (3280, "transition", True, "other", 16, "半调网点（橙色圆点）转场：3280 首帧出现网点，角色约 3292 完全被盖住；新画面 3296 起透过网点显现"),
    (3296, "multi", True, "other", 9, "新画面透过网点逐步显现（3296~3304）"),
    (3382, "transition", True, "other", 10, "浅蓝云雾/水彩擦除：3382 首帧，角色约 3389 完全被盖住；新画面 3392 起从云雾中显现"),
    (3392, "multi", True, "other", 12, "云雾逐步退出（3392~3403）"),
    (3478, "multi", True, "cut", 0, "3471 起上一镜头径向模糊推出，3478 硬切，新画面带径向模糊约到 3488 清晰"),
    (3662, "transition", True, "wipe", 6, "深蓝颜料泼溅擦除：3662 首帧，角色约 3667 完全被盖住；新画面 3668 起从泼溅中显现"),
    (3668, "halfBody", True, "wipe", 20, "泼溅逐步退出（3668~3687）"),
    (3722, "halfBody", True, "cut", 0, ""),
    (3768, "halfBody", True, "cut", 0, ""),
    (3818, "closeUp", True, "cut", 0, ""),
]
# 旧起点（SHOTS 的键）→ 新起点
REMAP = {400: 394, 684: 676, 795: 789, 970: 962, 1194: 1187, 1918: 1910, 3000: 2982, 3303: 3296, 3399: 3392,
         3684: 3668}
# 转场镜头里旧画面实例延续：新起点 → [(上一镜头实例全局 ID, 擦除物名称)]
TRANS_OLD = {
    372: (["s04.main"], "splash"), 652: (["s06.main", "s06.pipL", "s06.pipR"], "silhouette"),
    776: (["s08.bowl", "s08.loading", "s08.wink"], "silhouette"), 948: (["s10.left", "s10.right"], "silhouette"),
    1157: (["s13.main"], "splash"), 1896: (["s19.bowl", "s19.glasses"], "silhouette"),
    2608: (["s26.main"], "pageTurn"), 2969: (["s30.main", "s30.full"], "splash"),
    3280: (["s36.main"], "halftone"), 3382: (["s38.main", "s38.side"], "cloud"),
    3662: (["s41.main", "s41.full"], "splash"),
}
# 新镜头开头的擦除物残留：新起点 → 擦除物名称（矩形来自 transitions.py 的 new 模式）
TRANS_NEW = {394: "splash", 676: "silhouette", 789: "silhouette", 962: "silhouette", 1187: "splash",
             1910: "silhouette", 2982: "splash", 3296: "halftone", 3392: "cloud", 3668: "splash"}

# clip 关键帧（视频像素，在 review/clip*.jpg 的 0.5 倍整帧坐标网格图上读数，±10px）
CLIPS = {
    ("s04", "main"): [(254, [570, 310, 140, 90], 4), (258, [520, 260, 240, 190], 6), (260, [470, 160, 340, 330], 6),
                      (262, [430, 180, 420, 340], 6), (264, [390, 160, 490, 390], 6), (266, [370, 140, 550, 430], 6),
                      (268, [350, 130, 580, 450], 6), (330, [330, 110, 620, 490], 6)],
    ("s06", "pipL"): [(480, [130, 200, 150, 110], 4), (482, [90, 170, 230, 170], 4), (486, [100, 120, 270, 250], 6),
                      (500, [90, 70, 350, 340], 6), (520, [0, 80, 400, 330], 6), (540, [20, 90, 380, 300], 6),
                      (600, [0, 80, 420, 310], 6)],
    ("s06", "pipR"): [(519, [940, 380, 220, 160], 4), (521, [940, 350, 250, 210], 4), (525, [870, 320, 410, 300], 6),
                      (540, [810, 270, 470, 350], 6), (600, [820, 290, 460, 310], 6), (651, [830, 280, 450, 330], 6)],
    ("s13", "main"): [(1041, [430, 416, 390, 220], 20), (1043, [280, 250, 680, 410], 25), (1045, [190, 140, 860, 520], 25),
                      (1047, [130, 66, 980, 594], 25), (1050, [120, 61, 1033, 574], 25)],
    ("s25", "main"): [(2442, [140, 0, 980, 300], 30), (2443, [140, 0, 980, 300], 30), (2444, [140, 0, 980, 450], 30),
                      (2445, [140, 20, 980, 580], 30), (2446, [140, 40, 980, 510], 30), (2450, [140, 60, 980, 590], 30),
                      (2500, [140, 55, 980, 590], 30)],
    ("s26", "main"): [(2530, [140, 55, 980, 590], 30)],
    ("s27", "s26cont"): [(2608, [140, 55, 980, 590], 30)],
    ("s28", "full"): [(2614, [800, 50, 310, 620], 12)],
    ("s29", "full"): [(2639, [800, 60, 330, 610], 12)],
}
CLIP_NULL = {("s15", "left"), ("s15", "chibi")}   # 细线框不裁切

# effects 的作用实例与强度（键：新镜头 ID, kind）
FX_EXTRA = {
    ("s15", "depthOfField"): (["left", "chibi"], [(1340, 0), (1360, 8)], "强度单位：blurSigma（像素）"),
    ("s17", "motionBlur"): (None, [(1527, 1), (1548, 0)], ""),
    ("s18", "gaussianBlur"): (None, [(1710, 1), (1726, 20)], "强度单位：blurSigma（像素）"),
    ("s18", "fadeToWhite"): (None, [(1712, 0), (1725, 1)], "0~1，白色叠加比例"),
    ("s19", "fadeFromWhite"): (None, [(1726, 1), (1746, 0)], "0~1，白色叠加比例"),
    ("s19", "motionBlur"): (["glasses"], [(1836, 1), (1846, 0)], ""),
    ("s29", "radialBlur"): (None, [(2680, 0), (2689, 1)], ""),
    ("s30", "gaussianBlur"): (None, [(2690, 12), (2698, 0)], "强度单位：blurSigma（像素）"),
    ("s40", "radialBlur"): (None, [(3470, 0), (3477, 1)], ""),
    ("s41", "radialBlur"): (None, [(3478, 1), (3489, 0)], ""),
    ("s08", "motionBlur"): (["bowl", "loading", "wink"], None, ""),
}
# 跨镜头同一实例（画面连续）：(镜头, 实例) → 被延续的全局 ID
CONTINUES = {("s26", "main"): "s25.main", ("s29", "body"): None}

TRANS_JSON = os.path.join(os.path.dirname(__file__), "transitions.json")


def _variant_at_end(inst):
    v = dict(inst.get("variant", {}))
    for e in sorted(inst.get("switchEvents", []), key=lambda e: e["frame"]):
        v[e["param"]] = e["value"]
    from validate import SWITCH_DEFAULTS
    return {k: x for k, x in v.items() if SWITCH_DEFAULTS.get(k) != x}


def _trans_records():
    if not os.path.exists(TRANS_JSON):
        return []
    with open(TRANS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _occluder_exclusion(rec, gids, reason, s, e, step=2):
    rects = []
    last_vis = None
    for i, row in enumerate(rec["instances"][gids[0]]):
        f = row["frame"]
        if not (s <= f < e):
            continue
        boxes = [rec["instances"][g][i]["occluder"] for g in gids if rec["instances"][g][i]["occluder"]]
        anyvis = any(rec["instances"][g][i]["visible"] for g in gids)
        if anyvis:
            last_vis = f
        if not boxes or (f - rec["start"]) % step and f != e - 1:
            continue
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        rects.append((f, [x0, y0, round(x1 - x0, 1), round(y1 - y0, 1)]))
    if rec["mode"] == "old" and last_vis is not None:
        rects = [r for r in rects if r[0] <= last_vis + 2]
        e = min(e, last_vis + 3)
    if not rects:
        return None
    return EX(reason, rects[0][0], max(min(e, rec["end"]), rects[-1][0] + 1), rects)


def _continuation_instance(rec, gid, sid, prev_inst):
    name = gid.split(".", 1)[1]
    st = rec["refState"][gid]
    track = []
    rows = rec["instances"][gid]
    gone = False
    for i, row in enumerate(rows):
        f = row["frame"]
        last = i == len(rows) - 1
        vis = row["visible"]
        if vis is None:
            if not gone:
                track.append({"frame": f, "bbox": None, "head": None, "eyesState": "hidden",
                              "source": "diff-vs-ref", "confidence": 0.6})
                gone = True
            continue
        gone = False
        if (f - rec["start"]) % 2 and not last and (i + 1 < len(rows) and rows[i + 1]["visible"] is not None):
            continue
        smp = {"frame": f, "bbox": vis, "source": "diff-vs-ref", "confidence": 0.4}
        if st.get("head") and (row["headFrac"] or 0) >= 0.5:
            smp["head"] = st["head"]
            if st.get("eyes") and row["eyesVisible"]:
                smp["eyes"] = st["eyes"]
        else:
            smp["head"] = None
        if not smp.get("eyes"):
            smp["eyesState"] = "hidden"
        if st.get("rotation") is not None:
            smp["rotation"] = st["rotation"]
        track.append(smp)
    if not track:
        return None
    return {"id": name, "continues": gid, "role": prev_inst["role"], "z": prev_inst["z"],
            "variant": _variant_at_end(prev_inst), "switchEvents": [], "motionHints": [],
            "anchors": track, "trackMode": "fixed", "flipX": prev_inst.get("flipX", False),
            **({"colorGrade": prev_inst["colorGrade"]} if prev_inst.get("colorGrade") else {}),
            "notes": "转场期间旧画面的延续：bbox 为与转场前一帧（" + str(rec["ref"]) +
                     "）比对仍未被遮挡部分的外接框；head/eyes 沿用转场前一帧的值（角色在转场中只有小幅动作），"
                     "被遮挡后为 null"}


def build():
    apply_fixes()
    recs = _trans_records()
    rec_by = {(r["start"], r["mode"]): r for r in recs}
    old_start_of = {v: k for k, v in REMAP.items()}
    shots = []
    by_gid = {}
    for k, (s, cat, hc, tt, dur, note) in enumerate(NEW_BOUNDS):
        sid = f"s{k + 1:02d}"
        e = NEW_BOUNDS[k + 1][0] if k + 1 < len(NEW_BOUNDS) else FRAME_COUNT
        d = {"id": sid, "start": s, "end": e, "transitionIn": {"type": tt, "duration": dur},
             "category": cat, "hasCharacter": hc, "instances": [], "exclusions": [], "effects": [],
             "confidence": 0.9 if not hc else 0.4, "notes": note}
        old = old_start_of.get(s, s)
        if old in SHOTS:
            m = SHOTS[old]
            d["instances"] = [dict(i) for i in m.get("instances", [])]
            d["exclusions"] = [x for x in m.get("exclusions", [])]
            d["effects"] = [dict(x) for x in m.get("effects", [])]
            d["confidence"] = m.get("confidence", 0.5)
            if m.get("notes"):
                d["notes"] = (note + "；" if note else "") + m["notes"]
        # 转场里旧画面的延续实例
        if s in TRANS_OLD:
            gids, reason = TRANS_OLD[s]
            rec = rec_by.get((s, "old"))
            if rec:
                for g in gids:
                    prev = by_gid.get(g)
                    if prev is None:
                        continue
                    ci = _continuation_instance(rec, g, sid, prev)
                    if ci:
                        d["instances"].append(ci)
                ex = _occluder_exclusion(rec, gids, reason, s, e)
                if ex:
                    d["exclusions"].append(ex)
                d["effects"].append(FX(reason, s, e, "擦除/转场叠加物"))
                d["confidence"] = 0.4
        # 新镜头开头的擦除残留
        if s in TRANS_NEW:
            rec = rec_by.get((s, "new"))
            if rec:
                ex = _occluder_exclusion(rec, list(rec["instances"]), TRANS_NEW[s], s, min(e, rec["end"]))
                if ex:
                    d["exclusions"].append(ex)
        # 裁掉落在新区间外的排除区/效果，夹到区间内
        def clamp(x):
            a, b = max(s, x["start"]), min(e, x["end"])
            if b <= a:
                return None
            x = dict(x, start=a, end=b)
            if "rects" in x:
                rs = [r for r in x["rects"] if a <= r["frame"] < b] or [dict(x["rects"][0], frame=a)]
                x["rects"] = rs
            return x
        d["exclusions"] = [y for y in (clamp(x) for x in d["exclusions"]) if y]
        d["effects"] = [y for y in (clamp(x) for x in d["effects"]) if y]
        # 实例 ID 全局唯一、clip 关键帧、continues
        for inst in d["instances"]:
            name = inst["id"]
            inst["id"] = f"{sid}.{name}"
            key = (sid, name)
            if key in CLIPS:
                inst["clip"] = [{"frame": f, "rect": r, "radius": rad} for f, r, rad in CLIPS[key]]
            elif key in CLIP_NULL or "clip" in inst and isinstance(inst.get("clip"), dict):
                inst.pop("clip", None)
            if CONTINUES.get(key):
                inst["continues"] = CONTINUES[key]
            by_gid[inst["id"]] = inst
        for fx in d["effects"]:
            ex = FX_EXTRA.get((sid, fx["kind"]))
            if ex:
                ids, strength, n = ex
                if ids:
                    fx["instances"] = [f"{sid}.{i}" for i in ids]
                if strength:
                    fx["strength"] = [{"frame": f, "value": v} for f, v in strength]
                if n:
                    fx["note"] = (fx.get("note", "") + "；" if fx.get("note") else "") + n
        shots.append(d)
    _special_instances(shots, by_gid)
    ANNOT.write_text(json.dumps({"shots": shots}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", ANNOT, len(shots), "个镜头；有角色", sum(1 for s in shots if s["hasCharacter"]))


def _special_instances(shots, by_gid):
    """无法自动比对的转场与“窗口外残留上一镜头”的实例。"""
    sh = {s["id"]: s for s in shots}
    # s16 旋转万花筒：前半段是 s15 中间实例被扭曲；后半段混入 s17 画面
    s16 = sh["s16"]
    prev = by_gid["s15.center"]
    s16["instances"] = [
        {"id": "s16.center", "continues": "s15.center", "role": "main", "z": 2, "variant": _variant_at_end(prev),
         "switchEvents": [], "motionHints": [], "trackMode": "fixed", "flipX": False,
         "anchors": [{"frame": 1509, "bbox": [0, 0, 1280, 720], "head": [380, 0, 640, 400], "blurSigma": 6,
                      "eyesState": "special", "source": "manual", "confidence": 0.2},
                     {"frame": 1518, "bbox": [0, 0, 1280, 720], "head": None, "blurSigma": 15, "eyesState": "hidden",
                      "source": "manual", "confidence": 0.1},
                     {"frame": 1522, "bbox": None, "head": None, "eyesState": "hidden", "source": "manual", "confidence": 0.2}],
         "notes": "旋转扭曲 + 强模糊，几何量只表示“占满全屏”，不可用于拟合"},
        {"id": "s16.next", "role": "main", "z": 1, "variant": {}, "switchEvents": [], "motionHints": [],
         "trackMode": "fixed", "flipX": False,
         "anchors": [{"frame": 1509, "bbox": None, "head": None, "eyesState": "hidden", "source": "manual", "confidence": 0.2},
                     {"frame": 1516, "bbox": [0, 0, 1280, 720], "head": None, "blurSigma": 15, "eyesState": "hidden",
                      "source": "manual", "confidence": 0.1},
                     {"frame": 1526, "bbox": [0, 0, 1280, 720], "head": None, "blurSigma": 6, "eyesState": "hidden",
                      "source": "manual", "confidence": 0.2}],
         "colorGrade": "整体压暗、偏青灰（同 s17）",
         "notes": "下一镜头（s17）的画面以旋转方式进入"}]
    s16["effects"] = [{"kind": "swirl", "start": 1509, "end": 1527, "strength": [{"frame": 1509, "value": 0},
                       {"frame": 1516, "value": 1}, {"frame": 1526, "value": 0.6}], "note": "旋转/万花筒扭曲 + 径向模糊"}]
    s16["exclusions"] = [EX("effect", 1509, 1527, [(1509, [0, 0, 1280, 720])])]
    s16["confidence"] = 0.2
    for i in sh["s17"]["instances"]:
        if i["id"] == "s17.main":
            i["continues"] = "s16.next"
    # s27 翻页：新画面（s28 的两个实例）在右下逐步露出
    s27 = sh["s27"]
    for name in ("body", "full"):
        src = by_gid.get(f"s28.{name}")
        if not src:
            continue
        s27["instances"].append({
            "id": f"s27.{name}", "role": src["role"], "z": src["z"] - 10, "variant": dict(src["variant"]),
            "switchEvents": [], "motionHints": [], "trackMode": "fixed", "flipX": False,
            "anchors": [{"frame": 2608, "bbox": None, "head": None, "eyesState": "hidden", "source": "manual", "confidence": 0.5},
                        {"frame": 2611, "bbox": [640, 200, 640, 520] if name == "body" else [900, 300, 300, 400],
                         "head": None, "eyesState": "hidden", "source": "manual", "confidence": 0.3},
                        {"frame": 2613, "bbox": dict(src["anchors"][0]).get("bbox") or [0, 0, 1280, 720],
                         "head": dict(src["anchors"][0]).get("head"), "source": "manual", "confidence": 0.3}],
            **({"clip": [{"frame": 2608, "rect": [800, 50, 310, 620], "radius": 12}]} if name == "full" else {}),
            "notes": "翻页擦除中逐步露出的下一镜头画面（被翻起的页面遮挡部分见排除区）"})
        src["continues"] = f"s27.{name}"
    for i in s27["instances"]:
        if i.get("continues") == "s26.main":
            i["id"] = "s27.s26cont"
            i["clip"] = [{"frame": 2608, "rect": [140, 55, 980, 590], "radius": 30}]
    # 窗口弹出期间窗口外仍是上一镜头：s13 开头（1041~1046）、s25 开头（2442~2445）
    for sid, prev_gid, a, b in (("s13", "s12.main", 1041, 1047), ("s25", "s24.main", 2442, 2446)):
        prev = by_gid.get(prev_gid)
        if not prev:
            continue
        last = [x for x in prev["anchors"] if x.get("bbox")][-1]
        sh[sid]["instances"].append({
            "id": f"{sid}.prev", "continues": prev_gid, "role": prev["role"], "z": 0,
            "variant": _variant_at_end(prev), "switchEvents": [], "motionHints": [], "trackMode": "fixed", "flipX": False,
            "anchors": [{**{k: v for k, v in last.items() if k in ("bbox", "head", "eyes", "rotation")}, "frame": a,
                         "source": "hold", "confidence": 0.3},
                        {**{k: v for k, v in last.items() if k in ("bbox", "head", "eyes", "rotation")}, "frame": b - 1,
                         "source": "hold", "confidence": 0.3},
                        {"frame": b, "bbox": None, "head": None, "eyesState": "hidden", "source": "manual", "confidence": 0.8}],
            "notes": "窗口展开期间窗口外仍是上一镜头的画面（被 z 更高的窗口实例自然遮挡）；几何量沿用上一镜头末锚点"})


if __name__ == "__main__":
    build()
