// 调试：在帧 f 上扫描单个维度，打印损失各项（检查损失曲面是否合理）
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
import { makeScene } from '../src/fitshot.ts';
import { planFor } from '../src/plans.ts';
import { loadShots, trackAt } from '../src/shotsutil.ts';
import { installPageHelpers } from '../src/page.ts';
import { RefFrames } from '../src/refvideo.ts';
import { toParams } from '../src/model.ts';
import { readFileSync } from 'node:fs';
const shotId = process.argv[2] ?? 's24', f = Number(process.argv[3] ?? 2408);
const shot = loadShots().shots.find((s) => s.id === shotId)!;
const calib = JSON.parse(readFileSync('.data/fit/calib.json', 'utf8'));
const { plans } = planFor(shot);
const p = plans[0]!;
const eyes = trackAt(p.inst, 'eyes', f) as number[][];
const mdx = calib.eyes[1][0] - calib.eyes[0][0];
const st: Record<string, number> = { x: (eyes[0]![0]! + eyes[1]![0]!) / 2, y: (eyes[0]![1]! + eyes[1]![1]!) / 2, ls: Math.log(Math.hypot(eyes[1]![0]! - eyes[0]![0]!, eyes[1]![1]! - eyes[0]![1]!) / mdx), rot: 0, hx: 0, hy: 0, hz: 0, eyeL: 1, eyeR: 1, es: 0, ebx: 0, eby: 0, brow: 0, mo: 0, mf: 0 };
const scene = makeScene(shot, plans, calib, new Map([[p.id, st]]));
const ref = new RefFrames(shot.start, shot.end);
const h = await openHarness();
try {
  await h.page.evaluate(installPageHelpers);
  await h.page.evaluate(([ls, uri]) => (window as any).__fit.setup(ls, uri), [scene.layers, 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json'] as const);
  await h.page.evaluate(([b64]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl([]); }, [ref.frame(f).toString('base64')] as const);
  for (const [key, vals] of [['x', [-40, -20, -10, 0, 10, 20, 40]], ['hx', [-20, -10, 0, 10, 20]], ['hy', [-15, -5, 0, 5, 15]], ['ls', [-0.1, -0.05, 0, 0.05, 0.1]], ['hz', [-15, -5, 0, 5, 15]], ['c', [0, -10, -20, -30]]] as const) {
    const cands = vals.map((dv) => {
      const s = { ...st, [key]: (st[key] ?? 0) + dv };
      const set: [number, Record<string, number>][] = [];
      for (let g = p.in; g <= f; g++) set.push([g, { ...toParams(s), LeanInput: s.c ?? 0 }]);
      return { [p.id]: { tf: [s.x, s.y, Math.exp(s.ls!), s.rot], set } };
    });
    for (const scale of [0.25, 0.5]) {
      const r = (await h.page.evaluate(([f, c, sc]) => (window as any).__fit.evalBatch(f, c, { scale: sc, r: sc < 0.5 ? 2 : 3, wcol: 1, weye: 0.08, eyesOf: [Object.keys(c[0])[0]] }), [f, cands, scale] as const)) as any[];
      console.log(key, scale, vals.map((v, i) => `${v}:${r[i].loss.toFixed(3)}/${r[i].ssim.toFixed(3)}/${r[i].eye.toFixed(1)}/${r[i].seg.toFixed(3)}`).join('  '));
    }
  }
  console.log(JSON.stringify(await h.page.evaluate(() => (window as any).__fit.timing())));
} finally { await h.close(); }
