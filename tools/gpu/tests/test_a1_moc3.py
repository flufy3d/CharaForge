"""
A1 moc3 解析器测试。

    tools/.venv/bin/python -I tools/gpu/tests/test_a1_moc3.py      （也可用 pytest）

- 本模型能在 strict 模式下解析，没有警告，计数与已知值一致；
- 与 Core 的静态信息逐项一致（参数 id/min/max/default/repeat/type/keys、部件、drawable id/贴图/标志/mask/UV/索引）；
- 损坏的输入（截断、魔数错、段间夹带非零数据、索引越界、不认识的版本）都明确报错。
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import numpy as np  # noqa: E402

from a1_core import MOC, run_core  # noqa: E402
from l2d import moc3  # noqa: E402


class TestMoc3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MOC, "rb") as f:
            cls.buf = f.read()
        cls.m = moc3.parse(cls.buf, strict=True)

    def test_counts(self):
        m = self.m
        self.assertEqual(m.version, 5)
        self.assertEqual(m.warnings, [])
        c = m.counts
        self.assertEqual((c['parts'], c['deformers'], c['warpDeformers'], c['rotationDeformers'], c['artMeshes'], c['parameters']),
                         (147, 822, 723, 99, 715, 509))
        self.assertEqual(c['glue'], 38)
        self.assertEqual(c['blendShapesWarpDeformers'] + c['blendShapesArtMeshes'] + c['blendShapesRotationDeformers'], 349 + 13 + 22)
        self.assertEqual(m.canvas['pixelsPerUnit'], 3749.0)
        s = moc3.summarize(m)
        self.assertEqual(len(s['parameters']), 509)
        self.assertEqual(len(s['blendShapes']), 349 + 13 + 22)

    def test_against_core_static(self):
        m = self.m
        ref = run_core(m['parameters']['defaultValues'][None], static=True)
        st = ref['static']
        p = m['parameters']
        self.assertEqual(p['ids'], st['parameters']['ids'])
        for k, f in (('minValues', 'min'), ('maxValues', 'max'), ('defaultValues', 'default')):
            np.testing.assert_array_equal(p[k], np.float32(st['parameters'][f]))
        np.testing.assert_array_equal(p['isRepeat'], st['parameters']['repeats'])
        np.testing.assert_array_equal(p['parameterTypes'], st['parameters']['types'])
        for i, k in enumerate(st['parameters']['keys']):
            np.testing.assert_array_equal(m.parameter_keys(i), np.float32(k))
        self.assertEqual(m['parts']['ids'], st['parts']['ids'])
        np.testing.assert_array_equal(m['parts']['parentPartIndices'], st['parts']['parents'])
        a, d = m['artMeshes'], st['drawables']
        self.assertEqual(a['ids'], d['ids'])
        np.testing.assert_array_equal(a['textureNos'], d['textureIndices'])
        np.testing.assert_array_equal(a['drawableFlags'], d['constantFlags'])
        np.testing.assert_array_equal(a['parentPartIndices'], d['parentParts'])
        np.testing.assert_array_equal(a['vertexCounts'], d['vertexCounts'])
        mk = m['drawableMasks']['artMeshSourcesIndices']
        for i in range(m.counts['artMeshes']):
            b, n = a['drawableMaskSourcesBeginIndices'][i], a['drawableMaskSourcesCounts'][i]
            self.assertEqual([x for x in mk[b:b + n].tolist() if x >= 0], d['masks'][i])  # Core 去掉了 −1
        uv = st['uvs']
        mine = np.concatenate([m['uvs']['xys'][a['uvSourcesBeginIndices'][i]:a['uvSourcesBeginIndices'][i] + 2 * a['vertexCounts'][i]]
                               for i in range(m.counts['artMeshes'])]).reshape(-1, 2)
        np.testing.assert_array_equal(mine[:, 0], uv[:, 0])
        np.testing.assert_allclose(1 - mine[:, 1], uv[:, 1], atol=1e-7)    # Core 的 v = 1 − 文件里的 v
        idx = st['indices']
        mine = np.concatenate([m['positionIndices']['indices'][a['positionIndexSourcesBeginIndices'][i]:
                                                               a['positionIndexSourcesBeginIndices'][i] + a['positionIndexSourcesCounts'][i]]
                               for i in range(m.counts['artMeshes'])]).reshape(-1, 3)
        np.testing.assert_array_equal(mine[:, ::-1].reshape(-1), idx)       # Core 把每个三角形的顶点顺序反过来
        self.assertEqual(ref['meta']['mocVersion'], 5)
        # drawable_static：直接给出 Core 口径的 UV / 索引 / mask
        ds = moc3.drawable_static(m)
        np.testing.assert_array_equal(ds['uvs'], uv)
        np.testing.assert_array_equal(np.concatenate([t.reshape(-1) for t in ds['indices']]), idx)
        self.assertEqual(ds['masks'], d['masks'])

    def _bad(self, buf, msg=None):
        with self.assertRaises(moc3.Moc3Error) as cm:
            moc3.parse(bytes(buf), strict=True)
        if msg:
            self.assertIn(msg, str(cm.exception))

    def test_corrupt(self):
        b = bytearray(self.buf)
        self._bad(b[:1000])                                     # 截断
        x = bytearray(b)
        x[:4] = b'MOC2'
        self._bad(x, 'MOC3')                                    # 魔数
        x = bytearray(b)
        x[4] = 6
        self._bad(x, '版本')                                    # 不认识的版本（V5.03 起有新结构）
        m = self.m
        # 段之间的填充里夹带数据
        off = m.offsets['rotationDeformers.baseAngles'] + 4 * m.counts['rotationDeformers']
        x = bytearray(b)
        x[off] = 1
        self._bad(x, '非零')
        # 索引越界
        off = m.offsets['artMeshes.parentPartIndices']
        x = bytearray(b)
        x[off:off + 4] = struct.pack('<i', 10 ** 6)
        self._bad(x, '越界')
        # 偏移表在已知项之后多出非零项
        nsec = len([s for s in moc3.SECTIONS if s[4] <= m.version])
        x = bytearray(b)
        x[64 + 4 * nsec:64 + 4 * nsec + 4] = struct.pack('<I', 64)
        self._bad(x, '不认识的段')
        # 非 strict 时只记警告
        x = bytearray(b)
        x[m.offsets['rotationDeformers.baseAngles'] + 4 * m.counts['rotationDeformers']] = 1
        mm = moc3.parse(bytes(x), strict=False)
        self.assertTrue(any('非零' in w for w in mm.warnings))


if __name__ == '__main__':
    unittest.main()
