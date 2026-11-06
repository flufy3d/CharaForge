"""moc3 解析器（内部拟合工具专用，不进播放器、不对外分发）。

只读解析 Cubism moc3 二进制（小端，版本 1～5，即 moc3 V3.00～V5.00），读出变形求值需要的全部结构，
并能 dump 成可检查的 json 摘要。格式说明见 tools/gpu/docs/moc3.md。

文件布局：
  - 0x00 文件头 64 字节：'MOC3'、版本字节、端序字节（1 = 大端，本实现只支持小端）；
  - 0x40 段偏移表：每项 u32，指向一个数组（数组一般按 64 字节对齐）；项数取决于版本；
  - 计数表（段 0）：各类对象的个数，决定每个数组的长度；
  - 画布信息（段 1）。

assets 下的文件是不可信数据：所有读取都做边界检查，索引类字段在 validate() 里逐一校验范围，
不认识的版本、多出来的偏移项、多出来的计数、段之间的非零字节都会明确报出来（strict 时直接抛错）。
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HEADER_SIZE = 64
MAGIC = b'MOC3'
# 版本字节 → 名称（与 Core 的 csmMocVersion 一致）
VERSIONS = {1: 'V3.00', 2: 'V3.03', 3: 'V4.00', 4: 'V4.02', 5: 'V5.00'}

# 计数表各项（按文件中的顺序），以及从哪个版本开始出现
COUNTS = [
    ('parts', 1), ('deformers', 1), ('warpDeformers', 1), ('rotationDeformers', 1), ('artMeshes', 1), ('parameters', 1),
    ('partKeyforms', 1), ('warpDeformerKeyforms', 1), ('rotationDeformerKeyforms', 1), ('artMeshKeyforms', 1),
    ('keyformPositions', 1), ('parameterBindingIndices', 1), ('keyformBindings', 1), ('parameterBindings', 1), ('keys', 1),
    ('uvs', 1), ('positionIndices', 1), ('drawableMasks', 1), ('drawOrderGroups', 1), ('drawOrderGroupObjects', 1),
    ('glue', 1), ('glueInfo', 1), ('glueKeyforms', 1),
    ('keyformColors', 4), ('keyformScreenColorsCount', 4),
    ('blendShapeParameterBindings', 4), ('blendShapeKeyformBindings', 4), ('blendShapesWarpDeformers', 4), ('blendShapesArtMeshes', 4),
    ('blendShapeConstraintIndices', 4), ('blendShapeConstraints', 4), ('blendShapeConstraintValues', 4),
    ('blendShapesParts', 5), ('blendShapesRotationDeformers', 5), ('blendShapesGlue', 5),
]
COUNT_TABLE_SLOTS = 64  # 计数表在文件里占 64 个 u32（其余必须为 0）

# 段偏移表：(组, 字段, 计数名, dtype, 起始版本)。dtype：'rt' = 运行时占位（每项 8 字节，文件里全 0），'id' = 64 字节 ID 字符串
S = []


def _sec(group, cnt, ver, *fields):
    for f, dt in fields:
        S.append((group, f, cnt, dt, ver))


_sec('parts', 'parts', 1, ('runtimeSpace0', 'rt'), ('ids', 'id'), ('keyformBindingSourcesIndices', 'i4'),
     ('keyformSourcesBeginIndices', 'i4'), ('keyformSourcesCounts', 'i4'), ('isVisible', 'i4'), ('isEnabled', 'i4'),
     ('parentPartIndices', 'i4'))
_sec('deformers', 'deformers', 1, ('runtimeSpace0', 'rt'), ('ids', 'id'), ('keyformBindingSourcesIndices', 'i4'),
     ('isVisible', 'i4'), ('isEnabled', 'i4'), ('parentPartIndices', 'i4'), ('parentDeformerIndices', 'i4'), ('types', 'i4'),
     ('specificSourcesIndices', 'i4'))
_sec('warpDeformers', 'warpDeformers', 1, ('keyformBindingSourcesIndices', 'i4'), ('keyformSourcesBeginIndices', 'i4'),
     ('keyformSourcesCounts', 'i4'), ('vertexCounts', 'i4'), ('rows', 'i4'), ('columns', 'i4'))
_sec('rotationDeformers', 'rotationDeformers', 1, ('keyformBindingSourcesIndices', 'i4'), ('keyformSourcesBeginIndices', 'i4'),
     ('keyformSourcesCounts', 'i4'), ('baseAngles', 'f4'))
_sec('artMeshes', 'artMeshes', 1, ('runtimeSpace0', 'rt'), ('runtimeSpace1', 'rt'), ('runtimeSpace2', 'rt'), ('runtimeSpace3', 'rt'),
     ('ids', 'id'), ('keyformBindingSourcesIndices', 'i4'), ('keyformSourcesBeginIndices', 'i4'), ('keyformSourcesCounts', 'i4'),
     ('isVisible', 'i4'), ('isEnabled', 'i4'), ('parentPartIndices', 'i4'), ('parentDeformerIndices', 'i4'), ('textureNos', 'i4'),
     ('drawableFlags', 'u1'), ('vertexCounts', 'i4'), ('uvSourcesBeginIndices', 'i4'), ('positionIndexSourcesBeginIndices', 'i4'),
     ('positionIndexSourcesCounts', 'i4'), ('drawableMaskSourcesBeginIndices', 'i4'), ('drawableMaskSourcesCounts', 'i4'))
_sec('parameters', 'parameters', 1, ('runtimeSpace0', 'rt'), ('ids', 'id'), ('maxValues', 'f4'), ('minValues', 'f4'),
     ('defaultValues', 'f4'), ('isRepeat', 'i4'), ('decimalPlaces', 'i4'), ('parameterBindingSourcesBeginIndices', 'i4'),
     ('parameterBindingSourcesCounts', 'i4'))
_sec('partKeyforms', 'partKeyforms', 1, ('drawOrders', 'f4'))
_sec('warpDeformerKeyforms', 'warpDeformerKeyforms', 1, ('opacities', 'f4'), ('keyformPositionSourcesBeginIndices', 'i4'))
_sec('rotationDeformerKeyforms', 'rotationDeformerKeyforms', 1, ('opacities', 'f4'), ('angles', 'f4'), ('originX', 'f4'),
     ('originY', 'f4'), ('scales', 'f4'), ('isReflectX', 'i4'), ('isReflectY', 'i4'))
_sec('artMeshKeyforms', 'artMeshKeyforms', 1, ('opacities', 'f4'), ('drawOrders', 'f4'), ('keyformPositionSourcesBeginIndices', 'i4'))
_sec('keyformPositions', 'keyformPositions', 1, ('xys', 'f4'))
_sec('parameterBindingIndices', 'parameterBindingIndices', 1, ('bindingSourcesIndices', 'i4'))
_sec('keyformBindings', 'keyformBindings', 1, ('parameterBindingIndexSourcesBeginIndices', 'i4'),
     ('parameterBindingIndexSourcesCounts', 'i4'))
_sec('parameterBindings', 'parameterBindings', 1, ('keysSourcesBeginIndices', 'i4'), ('keysSourcesCounts', 'i4'))
_sec('keys', 'keys', 1, ('values', 'f4'))
_sec('uvs', 'uvs', 1, ('xys', 'f4'))
_sec('positionIndices', 'positionIndices', 1, ('indices', 'u2'))
_sec('drawableMasks', 'drawableMasks', 1, ('artMeshSourcesIndices', 'i4'))
_sec('drawOrderGroups', 'drawOrderGroups', 1, ('objectSourcesBeginIndices', 'i4'), ('objectSourcesCounts', 'i4'),
     ('objectSourcesTotalCounts', 'i4'), ('maximumDrawOrders', 'i4'), ('minimumDrawOrders', 'i4'))
_sec('drawOrderGroupObjects', 'drawOrderGroupObjects', 1, ('types', 'i4'), ('indices', 'i4'), ('selfIndices', 'i4'))
_sec('glue', 'glue', 1, ('runtimeSpace0', 'rt'), ('ids', 'id'), ('keyformBindingSourcesIndices', 'i4'),
     ('keyformSourcesBeginIndices', 'i4'), ('keyformSourcesCounts', 'i4'), ('artMeshIndicesA', 'i4'), ('artMeshIndicesB', 'i4'),
     ('glueInfoSourcesBeginIndices', 'i4'), ('glueInfoSourcesCounts', 'i4'))
_sec('glueInfo', 'glueInfo', 1, ('weights', 'f4'), ('positionIndices', 'u2'))
_sec('glueKeyforms', 'glueKeyforms', 1, ('intensities', 'f4'))
# V3.03
_sec('warpDeformers', 'warpDeformers', 2, ('isQuadSource', 'i4'))
# V4.00：参数的 key 表（供 csmGetParameterKeyValues）
_sec('parameterExtensions', 'parameters', 3, ('runtimeSpace0', 'rt'), ('keysSourcesBeginIndices', 'i4'), ('keysSourcesCounts', 'i4'))
# V4.02：乘算色/屏幕色、参数类型、blend shape
_sec('warpDeformers', 'warpDeformers', 4, ('keyformColorSourcesBeginIndices', 'i4'))
_sec('rotationDeformers', 'rotationDeformers', 4, ('keyformColorSourcesBeginIndices', 'i4'))
_sec('artMeshes', 'artMeshes', 4, ('keyformColorSourcesBeginIndices', 'i4'))
_sec('keyformColorsMultiply', 'keyformColors', 4, ('r', 'f4'), ('g', 'f4'), ('b', 'f4'))
_sec('keyformColorsScreen', 'keyformColors', 4, ('r', 'f4'), ('g', 'f4'), ('b', 'f4'))
_sec('parameters', 'parameters', 4, ('parameterTypes', 'i4'), ('blendShapeParameterBindingSourcesBeginIndices', 'i4'),
     ('blendShapeParameterBindingSourcesCounts', 'i4'))
_sec('blendShapeParameterBindings', 'blendShapeParameterBindings', 4, ('keysSourcesBeginIndices', 'i4'), ('keysSourcesCounts', 'i4'),
     ('baseKeyIndices', 'i4'))
_sec('blendShapeKeyformBindings', 'blendShapeKeyformBindings', 4, ('parameterBindingSourcesIndices', 'i4'),
     ('keyformSourcesBlendShapeIndices', 'i4'), ('keyformSourcesBlendShapeCounts', 'i4'),
     ('blendShapeConstraintIndexSourcesBeginIndices', 'i4'), ('blendShapeConstraintIndexSourcesCounts', 'i4'))
_sec('blendShapesWarpDeformers', 'blendShapesWarpDeformers', 4, ('targetIndices', 'i4'),
     ('blendShapeKeyformBindingSourcesBeginIndices', 'i4'), ('blendShapeKeyformBindingSourcesCounts', 'i4'))
_sec('blendShapesArtMeshes', 'blendShapesArtMeshes', 4, ('targetIndices', 'i4'),
     ('blendShapeKeyformBindingSourcesBeginIndices', 'i4'), ('blendShapeKeyformBindingSourcesCounts', 'i4'))
_sec('blendShapeConstraintIndices', 'blendShapeConstraintIndices', 4, ('blendShapeConstraintSourcesIndices', 'i4'))
_sec('blendShapeConstraints', 'blendShapeConstraints', 4, ('parameterIndices', 'i4'),
     ('blendShapeConstraintValueSourcesBeginIndices', 'i4'), ('blendShapeConstraintValueSourcesCounts', 'i4'))
_sec('blendShapeConstraintValues', 'blendShapeConstraintValues', 4, ('keys', 'f4'), ('weights', 'f4'))
# V5.00：每个 keyform 独立的颜色索引（乘算、屏幕），部件 / 旋转变形器 / glue 的 blend shape
_sec('warpDeformerKeyforms', 'warpDeformerKeyforms', 5, ('keyformMultiplyColorSourcesIndices', 'i4'),
     ('keyformScreenColorSourcesIndices', 'i4'))
_sec('rotationDeformerKeyforms', 'rotationDeformerKeyforms', 5, ('keyformMultiplyColorSourcesIndices', 'i4'),
     ('keyformScreenColorSourcesIndices', 'i4'))
_sec('artMeshKeyforms', 'artMeshKeyforms', 5, ('keyformMultiplyColorSourcesIndices', 'i4'),
     ('keyformScreenColorSourcesIndices', 'i4'))
_sec('blendShapesParts', 'blendShapesParts', 5, ('targetIndices', 'i4'), ('blendShapeKeyformBindingSourcesBeginIndices', 'i4'),
     ('blendShapeKeyformBindingSourcesCounts', 'i4'))
_sec('blendShapesRotationDeformers', 'blendShapesRotationDeformers', 5, ('targetIndices', 'i4'),
     ('blendShapeKeyformBindingSourcesBeginIndices', 'i4'), ('blendShapeKeyformBindingSourcesCounts', 'i4'))
_sec('blendShapesGlue', 'blendShapesGlue', 5, ('targetIndices', 'i4'), ('blendShapeKeyformBindingSourcesBeginIndices', 'i4'),
     ('blendShapeKeyformBindingSourcesCounts', 'i4'))

SECTIONS = [('countInfo', None, None, None, 1), ('canvasInfo', None, None, None, 1)] + S
DT = {'i4': ('<i4', 4), 'f4': ('<f4', 4), 'u1': ('u1', 1), 'u2': ('<u2', 2), 'rt': (None, 8), 'id': (None, 64)}

# drawableFlags 位
FLAG_BLEND_ADDITIVE = 1
FLAG_BLEND_MULTIPLICATIVE = 2
FLAG_DOUBLE_SIDED = 4
FLAG_INVERTED_MASK = 8

DEFORMER_WARP = 0
DEFORMER_ROTATION = 1


class Moc3Error(ValueError):
    pass


@dataclass
class Moc3:
    """解析结果。各组是 {字段: numpy 数组 / 列表}，命名与文件结构一致（见 SECTIONS）。"""
    version: int
    counts: dict
    canvas: dict
    groups: dict
    warnings: list = field(default_factory=list)
    offsets: dict = field(default_factory=dict)  # '组.字段' → 文件偏移（调试 / 探测用）
    path: str = ''

    def __getitem__(self, k):
        return self.groups[k]

    # ---------- 便捷视图 ----------
    @property
    def version_name(self) -> str:
        return VERSIONS.get(self.version, f'?{self.version}')

    def parameter_keys(self, p: int) -> np.ndarray:
        """参数 p 的 key 值（V4.00+ 的 parameterExtensions；与 Core 的 keyValues 一致）。"""
        ext = self.groups.get('parameterExtensions')
        if ext is None:
            return np.zeros(0, np.float32)
        b, c = int(ext['keysSourcesBeginIndices'][p]), int(ext['keysSourcesCounts'][p])
        return self['keys']['values'][b:b + c]

    def keyform_binding_params(self, kb: int) -> list[tuple[int, np.ndarray]]:
        """keyform 绑定 kb 涉及的 [(参数号, keys)]，顺序即 keyform 多维数组的维度顺序（第一个变化最快）。"""
        g = self['keyformBindings']
        b, c = int(g['parameterBindingIndexSourcesBeginIndices'][kb]), int(g['parameterBindingIndexSourcesCounts'][kb])
        out = []
        for pb in self['parameterBindingIndices']['bindingSourcesIndices'][b:b + c]:
            pb = int(pb)
            kb0, kc = int(self['parameterBindings']['keysSourcesBeginIndices'][pb]), int(self['parameterBindings']['keysSourcesCounts'][pb])
            out.append((self.binding_param[pb], self['keys']['values'][kb0:kb0 + kc]))
        return out

    @property
    def binding_param(self) -> np.ndarray:
        """parameterBindings[i] 属于哪个参数（由参数的 parameterBindingSources 区间反推）。"""
        if not hasattr(self, '_binding_param'):
            g = self['parameters']
            bp = np.full(self.counts['parameterBindings'], -1, np.int64)
            for p in range(self.counts['parameters']):
                b, c = int(g['parameterBindingSourcesBeginIndices'][p]), int(g['parameterBindingSourcesCounts'][p])
                bp[b:b + c] = p
            self._binding_param = bp
        return self._binding_param

    @property
    def blendshape_binding_param(self) -> np.ndarray:
        """blendShapeParameterBindings[i] 属于哪个参数。"""
        if not hasattr(self, '_bs_binding_param'):
            g = self['parameters']
            n = self.counts.get('blendShapeParameterBindings', 0)
            bp = np.full(n, -1, np.int64)
            if n:
                for p in range(self.counts['parameters']):
                    b = int(g['blendShapeParameterBindingSourcesBeginIndices'][p])
                    c = int(g['blendShapeParameterBindingSourcesCounts'][p])
                    bp[b:b + c] = p
            self._bs_binding_param = bp
        return self._bs_binding_param

    def summary(self, full: bool = False) -> dict:
        return summarize(self, full)


# ---------------------------------------------------------------- 读取

class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.n = len(buf)

    def check(self, off: int, size: int, what: str):
        if off < HEADER_SIZE or size < 0 or off + size > self.n:
            raise Moc3Error(f'{what}: 越界（offset={off}, size={size}, 文件 {self.n} 字节）')

    def array(self, off: int, count: int, dt: str, what: str):
        np_dt, size = DT[dt]
        self.check(off, count * size, what)
        if dt == 'rt':
            return None  # 运行时占位，不读
        if dt == 'id':
            raw = self.buf[off:off + 64 * count]
            ids = []
            for i in range(count):
                s = raw[64 * i:64 * (i + 1)]
                z = s.find(b'\0')
                if z < 0:
                    raise Moc3Error(f'{what}[{i}]: ID 没有结尾的 0')
                ids.append(s[:z].decode('utf-8', errors='strict'))
            return ids
        return np.frombuffer(self.buf, np_dt, count, off).copy()


def load(path: str | Path, strict: bool = True) -> Moc3:
    """读取并解析 moc3。strict 时遇到任何不认识的内容直接抛 Moc3Error；否则记入 warnings。"""
    path = Path(path)
    buf = path.read_bytes()
    m = parse(buf, strict=strict)
    m.path = str(path)
    return m


def parse(buf: bytes, strict: bool = True) -> Moc3:
    r = _Reader(buf)
    warnings: list[str] = []

    def problem(msg: str):
        if strict:
            raise Moc3Error(msg)
        warnings.append(msg)

    if len(buf) < HEADER_SIZE + 4 * 2:
        raise Moc3Error('文件太短')
    if buf[:4] != MAGIC:
        raise Moc3Error(f'文件头不是 MOC3：{buf[:4]!r}')
    version = buf[4]
    if buf[5] != 0:
        raise Moc3Error('大端 moc3 不支持')
    if version not in VERSIONS:
        raise Moc3Error(f'不认识的 moc3 版本 {version}（已知 {sorted(VERSIONS)}；6 = V5.03 起有离屏等新结构，未实现）')
    if any(buf[6:HEADER_SIZE]):
        problem('文件头保留字节非 0')

    secs = [s for s in SECTIONS if s[4] <= version]
    nsec = len(secs)
    r.check(HEADER_SIZE, 4 * nsec, '段偏移表')
    offsets = struct.unpack_from(f'<{nsec}I', buf, HEADER_SIZE)
    # 偏移表之后、第一个段之前的多余偏移项：必须全 0
    nz = [o for o in offsets if o > 0]
    if not nz:
        raise Moc3Error('段偏移表全为 0')
    first = min(nz)
    r.check(HEADER_SIZE, first - HEADER_SIZE, '段偏移表与第一个段之间')
    if first < HEADER_SIZE + 4 * nsec:
        raise Moc3Error('第一个段与段偏移表重叠')
    tail = np.frombuffer(buf, '<u4', (first - HEADER_SIZE) // 4 - nsec, HEADER_SIZE + 4 * nsec)
    if tail.any():
        problem(f'段偏移表在第 {nsec} 项之后还有非零项（不认识的段）：{[(nsec + i, int(v)) for i, v in enumerate(tail) if v][:8]}')

    # 计数表
    cnt_names = [n for n, v in COUNTS if v <= version]
    r.check(offsets[0], 4 * COUNT_TABLE_SLOTS, '计数表')
    raw_counts = struct.unpack_from(f'<{COUNT_TABLE_SLOTS}I', buf, offsets[0])
    counts = dict(zip(cnt_names, raw_counts))
    if any(raw_counts[len(cnt_names):]):
        problem(f'计数表第 {len(cnt_names)} 项之后有非零计数（不认识的对象类型）：{raw_counts[len(cnt_names):]}')
    counts['keyformScreenColors'] = counts.pop('keyformScreenColorsCount', 0)
    if version >= 4 and counts['keyformScreenColors'] != counts['keyformColors']:
        problem('乘算色与屏幕色 keyform 个数不一致')

    # 画布
    r.check(offsets[1], 64, '画布信息')
    ppu, ox, oy, w, h = struct.unpack_from('<5f', buf, offsets[1])
    cflags = buf[offsets[1] + 20]
    canvas = {'pixelsPerUnit': ppu, 'originX': ox, 'originY': oy, 'width': w, 'height': h, 'flags': cflags}
    if any(buf[offsets[1] + 21:offsets[1] + 64]):
        problem('画布信息保留字节非 0')

    groups: dict = {}
    section_offsets: dict = {}
    extents = [(offsets[0], 4 * COUNT_TABLE_SLOTS, 'countInfo'), (offsets[1], 64, 'canvasInfo')]
    for (grp, fld, cnt, dt, _v), off in zip(secs[2:], offsets[2:]):
        n = counts[cnt]
        what = f'{grp}.{fld}'
        section_offsets[what] = off
        if n == 0:
            groups.setdefault(grp, {})[fld] = np.zeros(0, DT[dt][0] or 'u1') if dt not in ('id', 'rt') else ([] if dt == 'id' else None)
            continue
        groups.setdefault(grp, {})[fld] = r.array(off, n, dt, what)
        extents.append((off, n * DT[dt][1], what))

    # 段之间的空隙必须是 0（否则说明有我们没解析的数据）
    extents.sort()
    for (o0, s0, n0), (o1, _s1, n1) in zip(extents, extents[1:]):
        if o0 + s0 > o1:
            problem(f'段重叠：{n0} 与 {n1}')
        elif any(buf[o0 + s0:o1]):
            problem(f'{n0} 与 {n1} 之间有 {o1 - o0 - s0} 字节非零数据（未解析）')
    o_last, s_last, n_last = extents[-1]
    if any(buf[o_last + s_last:]):
        problem(f'{n_last} 之后的文件尾部有非零数据（未解析）')

    m = Moc3(version=version, counts=counts, canvas=canvas, groups=groups, warnings=warnings, offsets=section_offsets)
    validate(m, problem)
    return m


# ---------------------------------------------------------------- 校验

def validate(m: Moc3, problem) -> None:
    """检查所有索引 / 区间都在范围内，并检查我们依赖的不变量。出错时调用 problem。"""
    c = m.counts
    g = m.groups

    def rng(arr, lo, hi, what):  # lo <= arr < hi
        a = np.asarray(arr)
        if a.size and (a.min() < lo or a.max() >= hi):
            problem(f'{what} 越界：[{a.min()}, {a.max()}] 不在 [{lo}, {hi})')

    def span(begin, count, total, what, allow_empty=True):
        b, n = np.asarray(begin, np.int64), np.asarray(count, np.int64)
        if b.size == 0:
            return
        if (n < 0).any():
            problem(f'{what}: 负的个数')
        nz = n > 0 if allow_empty else np.ones_like(n, bool)
        if ((b < 0) & nz).any() or ((b + n > total) & nz).any():
            problem(f'{what}: 区间越界（total={total}）')

    rng(g['parts']['parentPartIndices'], -1, c['parts'], 'parts.parentPartIndices')
    rng(g['parts']['keyformBindingSourcesIndices'], 0, max(1, c['keyformBindings']), 'parts.keyformBindingSourcesIndices')
    span(g['parts']['keyformSourcesBeginIndices'], g['parts']['keyformSourcesCounts'], c['partKeyforms'], 'parts.keyforms')

    d = g['deformers']
    rng(d['parentPartIndices'], -1, c['parts'], 'deformers.parentPartIndices')
    rng(d['parentDeformerIndices'], -1, c['deformers'], 'deformers.parentDeformerIndices')
    rng(d['types'], 0, 2, 'deformers.types')
    rng(d['keyformBindingSourcesIndices'], 0, max(1, c['keyformBindings']), 'deformers.keyformBindingSourcesIndices')
    for t, sub, n in ((0, 'warpDeformers', c['warpDeformers']), (1, 'rotationDeformers', c['rotationDeformers'])):
        sel = d['specificSourcesIndices'][d['types'] == t]
        rng(sel, 0, max(1, n), f'deformers.specificSourcesIndices({sub})')
        if len(np.unique(sel)) != n or len(sel) != n:
            problem(f'{sub}: 与 deformers 不是一一对应')
    # 父变形器必须排在前面（求值按数组顺序进行）
    pd = d['parentDeformerIndices']
    if (pd >= np.arange(len(pd))).any():
        problem('deformers: 存在父变形器排在子变形器之后的情况（未实现拓扑排序）')

    w = g['warpDeformers']
    span(w['keyformSourcesBeginIndices'], w['keyformSourcesCounts'], c['warpDeformerKeyforms'], 'warp.keyforms')
    if c['warpDeformers'] and ((w['rows'] < 1).any() or (w['columns'] < 1).any() or
                               (w['vertexCounts'] != (w['rows'] + 1) * (w['columns'] + 1)).any()):
        problem('warp: vertexCounts != (rows+1)(columns+1)')
    rot = g['rotationDeformers']
    span(rot['keyformSourcesBeginIndices'], rot['keyformSourcesCounts'], c['rotationDeformerKeyforms'], 'rotation.keyforms')

    a = g['artMeshes']
    rng(a['parentPartIndices'], -1, c['parts'], 'artMeshes.parentPartIndices')
    rng(a['parentDeformerIndices'], -1, c['deformers'], 'artMeshes.parentDeformerIndices')
    rng(a['keyformBindingSourcesIndices'], 0, max(1, c['keyformBindings']), 'artMeshes.keyformBindingSourcesIndices')
    span(a['keyformSourcesBeginIndices'], a['keyformSourcesCounts'], c['artMeshKeyforms'], 'artMeshes.keyforms')
    span(a['uvSourcesBeginIndices'], 2 * a['vertexCounts'], c['uvs'], 'artMeshes.uvs')
    span(a['positionIndexSourcesBeginIndices'], a['positionIndexSourcesCounts'], c['positionIndices'], 'artMeshes.positionIndices')
    span(a['drawableMaskSourcesBeginIndices'], a['drawableMaskSourcesCounts'], c['drawableMasks'], 'artMeshes.masks')
    if (a["drawableFlags"] & 0xF0).any():
        problem(f'artMeshes.drawableFlags 有不认识的位：{np.unique(a["drawableFlags"])}')
    rng(g['drawableMasks']['artMeshSourcesIndices'], -1, c['artMeshes'], 'drawableMasks')
    # 三角形索引不超出各自顶点数
    pi = g['positionIndices']['indices']
    for i in range(c['artMeshes']):
        b, n = int(a['positionIndexSourcesBeginIndices'][i]), int(a['positionIndexSourcesCounts'][i])
        if n % 3:
            problem(f'artMesh {i}: 索引数不是 3 的倍数')
        if n and int(pi[b:b + n].max()) >= int(a['vertexCounts'][i]):
            problem(f'artMesh {i}: 三角形索引越界')

    p = g['parameters']
    span(p['parameterBindingSourcesBeginIndices'], p['parameterBindingSourcesCounts'], c['parameterBindings'], 'parameters.bindings')
    kb = g['keyformBindings']
    span(kb['parameterBindingIndexSourcesBeginIndices'], kb['parameterBindingIndexSourcesCounts'], c['parameterBindingIndices'],
         'keyformBindings')
    rng(g['parameterBindingIndices']['bindingSourcesIndices'], 0, max(1, c['parameterBindings']), 'parameterBindingIndices')
    pb = g['parameterBindings']
    span(pb['keysSourcesBeginIndices'], pb['keysSourcesCounts'], c['keys'], 'parameterBindings.keys')
    if c['parameterBindings'] and (pb['keysSourcesCounts'] < 1).any():
        problem('parameterBindings: 有 key 个数为 0 的绑定')
    if (m.binding_param < 0).any():
        problem('parameterBindings: 有不属于任何参数的绑定')

    # keyform 个数 = 各维 key 数之积
    def kf_count_check(grp, kfname):
        gg = g[grp]
        for i in range(len(gg['keyformBindingSourcesIndices'])):
            dims = m.keyform_binding_params(int(gg['keyformBindingSourcesIndices'][i]))
            need = int(np.prod([len(k) for _, k in dims])) if dims else 1
            if int(gg['keyformSourcesCounts'][i]) != need:
                problem(f'{grp}[{i}]: keyform 个数 {int(gg["keyformSourcesCounts"][i])} != 各维 key 数之积 {need}')
    for grp in ('parts', 'warpDeformers', 'rotationDeformers', 'artMeshes', 'glue'):
        if c.get(grp, 0):
            kf_count_check(grp, None)

    # keyform 位置
    wk = g['warpDeformerKeyforms']
    for i in range(c['warpDeformers']):
        b, n = int(w['keyformSourcesBeginIndices'][i]), int(w['keyformSourcesCounts'][i])
        span(wk['keyformPositionSourcesBeginIndices'][b:b + n], np.full(n, 2 * int(w['vertexCounts'][i])), c['keyformPositions'],
             f'warp {i} positions')
    ak = g['artMeshKeyforms']
    span(ak['keyformPositionSourcesBeginIndices'], np.zeros(c['artMeshKeyforms'], np.int64), c['keyformPositions'], 'artMeshKeyforms')
    for i in range(c['artMeshes']):
        b, n = int(a['keyformSourcesBeginIndices'][i]), int(a['keyformSourcesCounts'][i])
        span(ak['keyformPositionSourcesBeginIndices'][b:b + n], np.full(n, 2 * int(a['vertexCounts'][i])), c['keyformPositions'],
             f'artMesh {i} positions')

    # draw order groups
    dg = g['drawOrderGroups']
    span(dg['objectSourcesBeginIndices'], dg['objectSourcesCounts'], c['drawOrderGroupObjects'], 'drawOrderGroups')
    go = g['drawOrderGroupObjects']
    rng(go['types'], 0, 2, 'drawOrderGroupObjects.types')
    for t, n in ((0, c['artMeshes']), (1, c['parts'])):
        rng(go['indices'][go['types'] == t], 0, max(1, n), f'drawOrderGroupObjects.indices(type {t})')
    rng(go['selfIndices'], -1, c['drawOrderGroups'], 'drawOrderGroupObjects.selfIndices')

    # glue
    gl = g['glue']
    if c['glue']:
        rng(gl['artMeshIndicesA'], 0, c['artMeshes'], 'glue.artMeshIndicesA')
        rng(gl['artMeshIndicesB'], 0, c['artMeshes'], 'glue.artMeshIndicesB')
        span(gl['keyformSourcesBeginIndices'], gl['keyformSourcesCounts'], c['glueKeyforms'], 'glue.keyforms')
        span(gl['glueInfoSourcesBeginIndices'], gl['glueInfoSourcesCounts'], c['glueInfo'], 'glue.info')
        if (gl['glueInfoSourcesCounts'] % 2).any():
            problem('glue: info 个数不是偶数')
        gi = g['glueInfo']['positionIndices']
        for i in range(c['glue']):
            b, n = int(gl['glueInfoSourcesBeginIndices'][i]), int(gl['glueInfoSourcesCounts'][i])
            ia, ib = gi[b:b + n:2], gi[b + 1:b + n:2]
            if ia.size and (ia.max() >= a['vertexCounts'][gl['artMeshIndicesA'][i]] or ib.max() >= a['vertexCounts'][gl['artMeshIndicesB'][i]]):
                problem(f'glue {i}: 顶点索引越界')

    if m.version >= 3:
        e = g['parameterExtensions']
        span(e['keysSourcesBeginIndices'], e['keysSourcesCounts'], c['keys'], 'parameterExtensions.keys')
    if m.version >= 4:
        nk = c['keyformColors']
        for grp in ('warpDeformers', 'rotationDeformers', 'artMeshes'):
            span(g[grp]['keyformColorSourcesBeginIndices'], g[grp]['keyformSourcesCounts'], nk, f'{grp}.colors')
        rng(p['parameterTypes'], 0, 2, 'parameters.parameterTypes')
        span(p['blendShapeParameterBindingSourcesBeginIndices'], p['blendShapeParameterBindingSourcesCounts'],
             c['blendShapeParameterBindings'], 'parameters.blendShapeBindings')
        bpb = g['blendShapeParameterBindings']
        span(bpb['keysSourcesBeginIndices'], bpb['keysSourcesCounts'], c['keys'], 'blendShapeParameterBindings.keys')
        if c['blendShapeParameterBindings'] and ((bpb['baseKeyIndices'] < 0) | (bpb['baseKeyIndices'] >= bpb['keysSourcesCounts'])).any():
            problem('blendShapeParameterBindings.baseKeyIndices 越界')
        bkb = g['blendShapeKeyformBindings']
        rng(bkb['parameterBindingSourcesIndices'], 0, max(1, c['blendShapeParameterBindings']), 'blendShapeKeyformBindings.param')
        span(bkb['blendShapeConstraintIndexSourcesBeginIndices'], bkb['blendShapeConstraintIndexSourcesCounts'],
             c['blendShapeConstraintIndices'], 'blendShapeKeyformBindings.constraints')
        if c['blendShapeKeyformBindings']:
            nkeys = bpb['keysSourcesCounts'][bkb['parameterBindingSourcesIndices']]
            if (bkb['keyformSourcesBlendShapeCounts'] != nkeys).any():
                problem('blendShapeKeyformBindings: keyform 个数 != 参数 key 数')
        rng(g['blendShapeConstraintIndices']['blendShapeConstraintSourcesIndices'], 0, max(1, c['blendShapeConstraints']),
            'blendShapeConstraintIndices')
        bc = g['blendShapeConstraints']
        rng(bc['parameterIndices'], 0, c['parameters'], 'blendShapeConstraints.parameterIndices')
        span(bc['blendShapeConstraintValueSourcesBeginIndices'], bc['blendShapeConstraintValueSourcesCounts'],
             c['blendShapeConstraintValues'], 'blendShapeConstraints.values')
        targets = {'blendShapesWarpDeformers': ('warpDeformers', 'warpDeformerKeyforms'),
                   'blendShapesArtMeshes': ('artMeshes', 'artMeshKeyforms'),
                   'blendShapesParts': ('parts', 'partKeyforms'),
                   'blendShapesRotationDeformers': ('rotationDeformers', 'rotationDeformerKeyforms'),
                   'blendShapesGlue': ('glue', 'glueKeyforms')}
        for bs, (tg, kfn) in targets.items():
            if bs not in g or not c.get(bs, 0):
                continue
            gg = g[bs]
            rng(gg['targetIndices'], 0, c[tg], f'{bs}.targetIndices')
            span(gg['blendShapeKeyformBindingSourcesBeginIndices'], gg['blendShapeKeyformBindingSourcesCounts'],
                 c['blendShapeKeyformBindings'], f'{bs}.bindings')
            for j in range(c[bs]):
                b, n = int(gg['blendShapeKeyformBindingSourcesBeginIndices'][j]), int(gg['blendShapeKeyformBindingSourcesCounts'][j])
                span(bkb['keyformSourcesBlendShapeIndices'][b:b + n], bkb['keyformSourcesBlendShapeCounts'][b:b + n], c[kfn],
                     f'{bs}[{j}].keyforms')
    if m.version >= 5:
        nk = c['keyformColors']
        for grp in ('warpDeformerKeyforms', 'rotationDeformerKeyforms', 'artMeshKeyforms'):
            rng(g[grp]['keyformMultiplyColorSourcesIndices'], 0, nk, f'{grp}.multiplyColor')
            rng(g[grp]['keyformScreenColorSourcesIndices'], 0, nk, f'{grp}.screenColor')
    # warp blend shape 的 keyform 也要有足够的位置数据
    if m.version >= 4 and c.get('blendShapesWarpDeformers', 0):
        gg = g['blendShapesWarpDeformers']
        for j in range(c['blendShapesWarpDeformers']):
            wi = int(gg['targetIndices'][j])
            b, n = int(gg['blendShapeKeyformBindingSourcesBeginIndices'][j]), int(gg['blendShapeKeyformBindingSourcesCounts'][j])
            for k in range(b, b + n):
                kb0, kn = int(bkb['keyformSourcesBlendShapeIndices'][k]), int(bkb['keyformSourcesBlendShapeCounts'][k])
                span(wk['keyformPositionSourcesBeginIndices'][kb0:kb0 + kn], np.full(kn, 2 * int(w['vertexCounts'][wi])),
                     c['keyformPositions'], f'warp blend shape {j}')
    if m.version >= 4 and c.get('blendShapesArtMeshes', 0):
        gg = g['blendShapesArtMeshes']
        for j in range(c['blendShapesArtMeshes']):
            ai = int(gg['targetIndices'][j])
            b, n = int(gg['blendShapeKeyformBindingSourcesBeginIndices'][j]), int(gg['blendShapeKeyformBindingSourcesCounts'][j])
            for k in range(b, b + n):
                kb0, kn = int(bkb['keyformSourcesBlendShapeIndices'][k]), int(bkb['keyformSourcesBlendShapeCounts'][k])
                span(ak['keyformPositionSourcesBeginIndices'][kb0:kb0 + kn], np.full(kn, 2 * int(a['vertexCounts'][ai])),
                     c['keyformPositions'], f'artMesh blend shape {j}')


# ---------------------------------------------------------------- drawable 静态数据（Core 口径）

def drawable_static(m: Moc3) -> dict:
    """光栅化需要的 drawable 静态数据，口径与 Core 的 csmGetDrawable* 一致（已逐项对拍）：
      uvs [V, 2]          全部顶点 UV（按 drawable 顺序拼接），v 已翻转：v_core = 1 − v_file；
      indices [list]      每个 drawable 的三角形索引 [T, 3]（int64），每个三角形顶点顺序已反转（Core 的绕序）；
      vertex_offsets [D+1]、texture [D]、flags [D]（bit0 加算、bit1 乘算、bit2 双面、bit3 反转遮罩）、
      masks [list]        每个 drawable 的 mask drawable 号（已去掉 −1）、parent_part [D]、ids [D]。"""
    g = m.groups
    a = g['artMeshes']
    D = m.counts['artMeshes']
    vc = a['vertexCounts'].astype(np.int64)
    uv = np.concatenate([g['uvs']['xys'][int(a['uvSourcesBeginIndices'][i]):int(a['uvSourcesBeginIndices'][i]) + 2 * int(vc[i])]
                         for i in range(D)]).reshape(-1, 2).astype(np.float32)
    uv[:, 1] = 1 - uv[:, 1]
    pi = g['positionIndices']['indices']
    idx = [pi[int(a['positionIndexSourcesBeginIndices'][i]):int(a['positionIndexSourcesBeginIndices'][i]) + int(a['positionIndexSourcesCounts'][i])]
           .astype(np.int64).reshape(-1, 3)[:, ::-1].copy() for i in range(D)]
    mk = g['drawableMasks']['artMeshSourcesIndices']
    masks = [[int(x) for x in mk[int(a['drawableMaskSourcesBeginIndices'][i]):int(a['drawableMaskSourcesBeginIndices'][i])
                                 + int(a['drawableMaskSourcesCounts'][i])] if x >= 0] for i in range(D)]
    return {'ids': list(a['ids']), 'uvs': uv, 'indices': idx, 'vertex_offsets': np.concatenate([[0], np.cumsum(vc)]),
            'texture': a['textureNos'].astype(np.int64), 'flags': a['drawableFlags'].astype(np.int64), 'masks': masks,
            'parent_part': a['parentPartIndices'].astype(np.int64)}


# ---------------------------------------------------------------- 摘要

def _lst(a, n=None):
    a = np.asarray(a)
    if n is not None:
        a = a[:n]
    return a.tolist()


def summarize(m: Moc3, full: bool = False) -> dict:
    """可检查的 json 摘要。full=False 时大数组只给统计量。"""
    g, c = m.groups, m.counts
    out: dict = {'version': m.version, 'versionName': m.version_name, 'counts': c, 'canvas': m.canvas, 'warnings': m.warnings}
    p = g['parameters']
    params = []
    for i in range(c['parameters']):
        e = {'id': p['ids'][i], 'min': float(p['minValues'][i]), 'max': float(p['maxValues'][i]), 'default': float(p['defaultValues'][i]),
             'repeat': bool(p['isRepeat'][i]), 'decimalPlaces': int(p['decimalPlaces'][i]), 'keys': _lst(m.parameter_keys(i))}
        if m.version >= 4:
            e['type'] = 'blendShape' if p['parameterTypes'][i] == 1 else 'normal'
        params.append(e)
    out['parameters'] = params
    pt = g['parts']
    pk = g['partKeyforms']['drawOrders']
    out['parts'] = [{'id': pt['ids'][i], 'parent': int(pt['parentPartIndices'][i]), 'visible': bool(pt['isVisible'][i]),
                     'enabled': bool(pt['isEnabled'][i]),
                     'keyformParams': [m['parameters']['ids'][q] for q, _ in m.keyform_binding_params(int(pt['keyformBindingSourcesIndices'][i]))],
                     'keyformDrawOrders': _lst(pk[pt['keyformSourcesBeginIndices'][i]:pt['keyformSourcesBeginIndices'][i] + pt['keyformSourcesCounts'][i]])}
                    for i in range(c['parts'])]
    d = g['deformers']
    defs = []
    for i in range(c['deformers']):
        t = int(d['types'][i])
        s = int(d['specificSourcesIndices'][i])
        e = {'id': d['ids'][i], 'type': 'warp' if t == 0 else 'rotation', 'parentDeformer': int(d['parentDeformerIndices'][i]),
             'parentPart': int(d['parentPartIndices'][i]), 'visible': bool(d['isVisible'][i]), 'enabled': bool(d['isEnabled'][i]),
             'keyformParams': [m['parameters']['ids'][q] for q, _ in m.keyform_binding_params(int(d['keyformBindingSourcesIndices'][i]))]}
        if t == 0:
            w = g['warpDeformers']
            e.update(rows=int(w['rows'][s]), columns=int(w['columns'][s]), keyforms=int(w['keyformSourcesCounts'][s]),
                     isQuadSource=bool(w['isQuadSource'][s]) if m.version >= 2 else False)
            if full:
                kb = int(w['keyformSourcesBeginIndices'][s])
                wk = g['warpDeformerKeyforms']
                e['keyformOpacities'] = _lst(wk['opacities'][kb:kb + e['keyforms']])
                pos0 = int(wk['keyformPositionSourcesBeginIndices'][kb])
                e['keyform0Grid'] = _lst(g['keyformPositions']['xys'][pos0:pos0 + 2 * int(w['vertexCounts'][s])])
        else:
            rr = g['rotationDeformers']
            rk = g['rotationDeformerKeyforms']
            kb, kn = int(rr['keyformSourcesBeginIndices'][s]), int(rr['keyformSourcesCounts'][s])
            e.update(baseAngle=float(rr['baseAngles'][s]), keyforms=kn)
            if full:
                e['keyformValues'] = {f: _lst(rk[f][kb:kb + kn]) for f in ('opacities', 'angles', 'originX', 'originY', 'scales',
                                                                           'isReflectX', 'isReflectY')}
        defs.append(e)
    out['deformers'] = defs
    a = g['artMeshes']
    ak = g['artMeshKeyforms']
    meshes = []
    for i in range(c['artMeshes']):
        kb, kn = int(a['keyformSourcesBeginIndices'][i]), int(a['keyformSourcesCounts'][i])
        mb, mn = int(a['drawableMaskSourcesBeginIndices'][i]), int(a['drawableMaskSourcesCounts'][i])
        fl = int(a['drawableFlags'][i])
        e = {'id': a['ids'][i], 'parentDeformer': int(a['parentDeformerIndices'][i]), 'parentPart': int(a['parentPartIndices'][i]),
             'texture': int(a['textureNos'][i]), 'flags': fl,
             'blend': 'additive' if fl & 1 else 'multiplicative' if fl & 2 else 'normal', 'doubleSided': bool(fl & 4),
             'invertedMask': bool(fl & 8), 'vertices': int(a['vertexCounts'][i]), 'triangles': int(a['positionIndexSourcesCounts'][i]) // 3,
             'masks': _lst(g['drawableMasks']['artMeshSourcesIndices'][mb:mb + mn]), 'keyforms': kn,
             'keyformParams': [m['parameters']['ids'][q] for q, _ in m.keyform_binding_params(int(a['keyformBindingSourcesIndices'][i]))],
             'keyformOpacities': _lst(ak['opacities'][kb:kb + kn]), 'keyformDrawOrders': _lst(ak['drawOrders'][kb:kb + kn])}
        if m.version >= 5:
            e['keyformMultiplyColors'] = [[float(g['keyformColorsMultiply'][ch][j]) for ch in 'rgb']
                                          for j in ak['keyformMultiplyColorSourcesIndices'][kb:kb + kn]]
            e['keyformScreenColors'] = [[float(g['keyformColorsScreen'][ch][j]) for ch in 'rgb']
                                        for j in ak['keyformScreenColorSourcesIndices'][kb:kb + kn]]
        meshes.append(e)
    out['artMeshes'] = meshes
    dg, go = g['drawOrderGroups'], g['drawOrderGroupObjects']
    out['drawOrderGroups'] = [{'objects': [('artMesh' if go['types'][j] == 0 else 'part', int(go['indices'][j]), int(go['selfIndices'][j]))
                                           for j in range(dg['objectSourcesBeginIndices'][i],
                                                          dg['objectSourcesBeginIndices'][i] + dg['objectSourcesCounts'][i])],
                               'totalCount': int(dg['objectSourcesTotalCounts'][i]), 'maxDrawOrder': int(dg['maximumDrawOrders'][i]),
                               'minDrawOrder': int(dg['minimumDrawOrders'][i])} for i in range(c['drawOrderGroups'])]
    gl, gi = g['glue'], g['glueInfo']
    out['glue'] = [{'id': gl['ids'][i], 'artMeshA': int(gl['artMeshIndicesA'][i]), 'artMeshB': int(gl['artMeshIndicesB'][i]),
                    'pairs': int(gl['glueInfoSourcesCounts'][i]) // 2,
                    'keyformParams': [m['parameters']['ids'][q] for q, _ in m.keyform_binding_params(int(gl['keyformBindingSourcesIndices'][i]))],
                    'intensities': _lst(g['glueKeyforms']['intensities'][gl['keyformSourcesBeginIndices'][i]:
                                                                         gl['keyformSourcesBeginIndices'][i] + gl['keyformSourcesCounts'][i]])}
                   for i in range(c['glue'])]
    if m.version >= 4:
        bs = []
        bpb, bkb = g['blendShapeParameterBindings'], g['blendShapeKeyformBindings']
        for grp, tg in (('blendShapesWarpDeformers', 'warp'), ('blendShapesArtMeshes', 'artMesh'), ('blendShapesParts', 'part'),
                        ('blendShapesRotationDeformers', 'rotation'), ('blendShapesGlue', 'glue')):
            if grp not in g:
                continue
            gg = g[grp]
            for j in range(c.get(grp, 0)):
                b, n = int(gg['blendShapeKeyformBindingSourcesBeginIndices'][j]), int(gg['blendShapeKeyformBindingSourcesCounts'][j])
                binds = []
                for k in range(b, b + n):
                    pb = int(bkb['parameterBindingSourcesIndices'][k])
                    kb0, kn0 = int(bpb['keysSourcesBeginIndices'][pb]), int(bpb['keysSourcesCounts'][pb])
                    cb, cn = int(bkb['blendShapeConstraintIndexSourcesBeginIndices'][k]), int(bkb['blendShapeConstraintIndexSourcesCounts'][k])
                    cons = []
                    for ci in g['blendShapeConstraintIndices']['blendShapeConstraintSourcesIndices'][cb:cb + cn]:
                        bc = g['blendShapeConstraints']
                        vb, vn = int(bc['blendShapeConstraintValueSourcesBeginIndices'][ci]), int(bc['blendShapeConstraintValueSourcesCounts'][ci])
                        cons.append({'param': m['parameters']['ids'][int(bc['parameterIndices'][ci])],
                                     'keys': _lst(g['blendShapeConstraintValues']['keys'][vb:vb + vn]),
                                     'weights': _lst(g['blendShapeConstraintValues']['weights'][vb:vb + vn])})
                    binds.append({'param': m['parameters']['ids'][int(m.blendshape_binding_param[pb])],
                                  'keys': _lst(g['keys']['values'][kb0:kb0 + kn0]), 'baseKey': int(bpb['baseKeyIndices'][pb]),
                                  'keyformBegin': int(bkb['keyformSourcesBlendShapeIndices'][k]), 'constraints': cons})
                bs.append({'target': tg, 'index': int(gg['targetIndices'][j]), 'bindings': binds})
        out['blendShapes'] = bs
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='解析 moc3 并输出 json 摘要')
    ap.add_argument('moc')
    ap.add_argument('-o', '--out', help='输出 json 路径（缺省打印计数）')
    ap.add_argument('--full', action='store_true', help='包含 keyform 明细')
    ap.add_argument('--lenient', action='store_true', help='遇到不认识的内容只警告')
    a = ap.parse_args(argv)
    m = load(a.moc, strict=not a.lenient)
    s = summarize(m, a.full)
    if a.out:
        Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(json.dumps({'version': s['versionName'], 'counts': s['counts'], 'canvas': s['canvas'], 'warnings': s['warnings']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
