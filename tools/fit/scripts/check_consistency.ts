// 校验：拟合引擎（页面里自己逐帧推进 + Compositor 合成）与场景求值路径（renderFrame）对同一份图层是否逐像素一致
//   node --experimental-strip-types tools/fit/scripts/check_consistency.ts s24 2416,2417,2430
import { readFileSync } from 'node:fs';
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
import { installPageHelpers } from '../src/page.ts';
import { MODEL_URI } from '../src/paths.ts';
const [shotId, framesArg] = process.argv.slice(2);
const res = JSON.parse(readFileSync(`.data/fit/${shotId}/result.json`, 'utf8'));
const calib = JSON.parse(readFileSync('.data/fit/calib.json', 'utf8'));
const frames = framesArg!.split(',').map(Number);
const h = await openHarness();
try {
  await h.page.evaluate(installPageHelpers);
  const scene = {
    format: 'charaforge.scene', version: 1, meta: { title: 'check' }, timebase: { fps: 60 }, duration: res.range[1],
    composition: { width: 1280, height: 720, background: null }, assets: { models: [{ id: 'dsc', uri: MODEL_URI }] }, layers: res.layers,
  };
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene as object);
  // 拟合引擎不实现 frameHold：去掉保持段后比较（保持只改变「显示哪一帧」）
  const noHold = res.layers.map((l: any) => ({ ...l, frameHold: undefined }));
  await h.page.evaluate(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [noHold, MODEL_URI, calib.irisOffset] as const);
  const isHeld = (f: number) => res.layers.some((l: any) => (l.frameHold ?? []).some((h: any) => f >= h.start && f < h.end && (f - h.start) % h.every !== 0));
  for (const f of frames) {
    if (isHeld(f)) { console.log(`f=${f} 是某图层的保持帧（画面取自更早的帧），跳过`); continue; }
    const r = await h.page.evaluate(async ([f]) => {
      const scenePx = await window.__charaforge.renderFrame(f, { read: {} });
      const fitPx = (window as any).__fit.render(f, 1);
      const a = atob(scenePx.data!), b = atob(fitPx.data);
      let diff = 0, maxd = 0;
      for (let i = 0; i < a.length; i++) { const d = Math.abs(a.charCodeAt(i) - b.charCodeAt(i)); if (d) diff++; if (d > maxd) maxd = d; }
      return { diff, maxd, n: a.length };
    }, [f] as const);
    console.log(`f=${f} 不同字节 ${r.diff}/${r.n}，最大差 ${r.maxd}`);
  }
} finally { await h.close(); }
