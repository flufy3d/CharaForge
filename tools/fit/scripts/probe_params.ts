// 探针：每个参数单独改值（物理推进 120 帧至稳态）后，与默认姿态相比画面变化了多少像素、变化区域在哪
//   node --experimental-strip-types tools/fit/scripts/probe_params.ts '{"Param11":8,"ParamBreath":1}'
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const tests: Record<string, number> = JSON.parse(process.argv[2]!);
const h = await openHarness();
try {
  const r = await h.page.evaluate(async (tests) => {
    const H = window.__charaforge;
    const id = await H.createInstance('models/deepseek-chan/DeepSeek/大肥鱼.model3.json');
    const W = 600, HH = 1232; // 整个画布按 0.16 缩放
    const shot = async (pre: Record<string, number>) => {
      H.resetInstancePhysics(id);
      for (let i = 0; i < 120; i++) H.stepInstance(id, { pre, physics: true });
      const px = await H.renderInstance(id, { transform: { x: W / 2, y: HH / 2, scale: 0.16 }, width: W, height: HH });
      const b = atob(px.data!); const a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return a;
    };
    const base = await shot({});
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(tests)) {
      const im = await shot({ [k]: v });
      let n = 0, x0 = W, y0 = HH, x1 = 0, y1 = 0;
      for (let i = 0; i < W * HH; i++) {
        const d = Math.abs(im[4 * i]! - base[4 * i]!) + Math.abs(im[4 * i + 1]! - base[4 * i + 1]!) + Math.abs(im[4 * i + 2]! - base[4 * i + 2]!) + Math.abs(im[4 * i + 3]! - base[4 * i + 3]!);
        if (d > 24) { n++; const x = i % W, y = (i / W) | 0; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      }
      out[`${k}=${v}`] = n ? { changed: n, bbox: [x0, y0, x1, y1] } : 0;
    }
    return out;
  }, tests);
  for (const [k, v] of Object.entries(r)) console.log(k, JSON.stringify(v));
} finally { await h.close(); }
