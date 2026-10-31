// 调试：一组参数组合，按 shots 的眼睛位置对齐后渲染，与参考帧并排拼图（人工判断参数空间能否覆盖参考姿态）
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { makeScene } from '../src/fitshot.ts';
import { planFor } from '../src/plans.ts';
import { loadShots, trackAt } from '../src/shotsutil.ts';
import { installPageHelpers } from '../src/page.ts';
import { RefFrames } from '../src/refvideo.ts';
import { toParams, DEFAULT_STATE } from '../src/model.ts';
import { readFileSync, writeFileSync } from 'node:fs';
const [shotId, fs, combosJson, outPath] = process.argv.slice(2);
const f = Number(fs);
const combos: Record<string, number>[] = JSON.parse(combosJson!);
const shot = loadShots().shots.find((s) => s.id === shotId)!;
const calib = JSON.parse(readFileSync('.data/fit/calib.json', 'utf8'));
const { plans } = planFor(shot);
const p = plans[0]!;
p.shotConsts = [];
const eyes = trackAt(p.inst, 'eyes', f) as number[][];
const ref = new RefFrames(shot.start, shot.end);
const R = ref.frame(f);
const h = await openHarness();
const tiles: Buffer[] = [];
const S = 0.25, TW = 320, TH = 180;
try {
  await h.page.evaluate(installPageHelpers);
  for (const c of combos) {
    const extra: Record<string, number> = {};
    const st: Record<string, number> = { ...DEFAULT_STATE, x: 640, y: 360, ls: Math.log(0.6), rot: 0 };
    for (const [k, v] of Object.entries(c)) if (k in st) st[k] = v; else extra[k] = v;
    const scene = makeScene(shot, plans, calib, new Map([[p.id, st]]));
    const L = scene.layers[0] as any;
    for (const [k, v] of Object.entries(extra)) L.params.push({ id: k, curve: { keys: [{ t: p.in, v }] } });
    await h.page.evaluate(([ls, uri]) => (window as any).__fit.setup(ls, uri), [scene.layers, 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json'] as const);
    // 两次迭代：按虹膜位置做相似变换对齐到 shots 的 eyes
    for (let it = 0; it < 3; it++) {
      const ir = (await h.page.evaluate(([lid, f]) => (window as any).__fit.irisScreen(lid, f), [p.id, f] as const)) as number[][];
      const dO = Math.hypot(ir[1]![0]! - ir[0]![0]!, ir[1]![1]! - ir[0]![1]!), dT = Math.hypot(eyes[1]![0]! - eyes[0]![0]!, eyes[1]![1]! - eyes[0]![1]!);
      const aO = Math.atan2(ir[1]![1]! - ir[0]![1]!, ir[1]![0]! - ir[0]![0]!), aT = Math.atan2(eyes[1]![1]! - eyes[0]![1]!, eyes[1]![0]! - eyes[0]![0]!);
      st.ls! += Math.log(dT / dO);
      st.rot! += ((aT - aO) * 180) / Math.PI;
      const mO = [(ir[0]![0]! + ir[1]![0]!) / 2, (ir[0]![1]! + ir[1]![1]!) / 2], mT = [(eyes[0]![0]! + eyes[1]![0]!) / 2, (eyes[0]![1]! + eyes[1]![1]!) / 2];
      st.x! += mT[0]! - mO[0]!; st.y! += mT[1]! - mO[1]!;
      await h.page.evaluate(([lid, f, v]) => (window as any).__fit.commit(f, { [lid]: v }), [p.id, f, { tf: [st.x, st.y, Math.exp(st.ls!), st.rot] }] as const);
    }
    const img = (await h.page.evaluate((f) => (window as any).__fit.render(f, 0.25), f)) as { width: number; height: number; data: string };
    const P = Buffer.from(img.data, 'base64');
    const t = Buffer.alloc(TW * TH * 3);
    for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
      const po = (y * TW + x) * 4, ro = (y * 4 * 1280 + x * 4) * 3, a = P[po + 3]! / 255;
      for (let k = 0; k < 3; k++) t[(y * TW + x) * 3 + k] = Math.round(P[po + k]! + (1 - a) * 255);
    }
    tiles.push(t);
  }
  const rt = Buffer.alloc(TW * TH * 3);
  for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) for (let k = 0; k < 3; k++) rt[(y * TW + x) * 3 + k] = R[(y * 4 * 1280 + x * 4) * 3 + k]!;
  const all = [rt, ...tiles];
  const cols = 4, rows = Math.ceil(all.length / cols);
  const out = new Uint8Array(cols * TW * rows * TH * 4);
  all.forEach((t, i) => {
    const ox = (i % cols) * TW, oy = Math.floor(i / cols) * TH;
    for (let y = 0; y < TH; y++) for (let x = 0; x < TW; x++) {
      const o = ((oy + y) * cols * TW + ox + x) * 4;
      out[o] = t[(y * TW + x) * 3]!; out[o + 1] = t[(y * TW + x) * 3 + 1]!; out[o + 2] = t[(y * TW + x) * 3 + 2]!; out[o + 3] = 255;
    }
  });
  writeFileSync(outPath!, encodePng(cols * TW, rows * TH, out, false));
  console.log('ok', combos.length);
} finally { await h.close(); }
