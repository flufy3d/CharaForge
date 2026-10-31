// 探针：找「眼球」部件下的 drawable，计算参考姿态下的顶点质心（画布像素）
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const h = await openHarness();
try {
  const r = await h.page.evaluate(async () => {
    const H = window.__charaforge;
    const id = await H.createInstance('models/deepseek-chan/DeepSeek/大肥鱼.model3.json');
    for (let i = 0; i < 120; i++) H.stepInstance(id, { physics: true });
    const rt = (window as any).__cfRuntime;
    const inst = rt.instance(id);
    const m = inst.model;
    const c = inst.canvas;
    const nP = m.getPartCount();
    const partIds: string[] = [];
    for (let i = 0; i < nP; i++) partIds.push(m.getPartId(i).getString().s ?? String(m.getPartId(i).getString()));
    const ppi = m.getPartParentPartIndices(); const parentOfPart = (i: number) => ppi[i];
    const chain = (p: number) => { const out: string[] = []; while (p >= 0) { out.push(partIds[p]!); p = parentOfPart(p); } return out; };
    const res: any[] = [];
    for (let d = 0; d < m.getDrawableCount(); d++) {
      const ch = chain(m.getDrawableParentPartIndex(d));
      if (!ch.some((p) => ['Part43', 'Part35', 'Part140', 'Part145'].includes(p))) continue;
      const v = m.getDrawableVertices(d);
      let sx = 0, sy = 0; const n = v.length / 2;
      for (let i = 0; i < n; i++) { sx += v[2 * i]; sy += v[2 * i + 1]; }
      res.push({ d, id: m.getDrawableId(d).getString().s ?? String(m.getDrawableId(d).getString()), chain: ch.slice(0, 3).join('<'), op: +m.getDrawableOpacity(d).toFixed(2), n,
        cx: +(c.originX + (sx / n) * c.pixelsPerUnit).toFixed(1), cy: +(c.originY - (sy / n) * c.pixelsPerUnit).toFixed(1) });
    }
    return res;
  });
  for (const x of r) console.log(JSON.stringify(x));
} finally { await h.close(); }
