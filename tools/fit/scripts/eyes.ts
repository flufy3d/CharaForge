// shots 修正建议：拟合结果里每个实例在 shots 眼睛锚点帧上的虹膜屏幕位置（页面 irisScreen，与拟合同一实现），
// 与 shots.json 的 eyes 对比，偏差大于阈值的列出来（只输出建议，不改 shots）。
//   node --experimental-strip-types tools/fit/scripts/eyes.ts <scene.json> <shot> [阈值px=8]
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Pool } from '../src/pool.ts';
import { loadShots } from '../src/shotsutil.ts';
import { MODEL_URI, REPO } from '../src/paths.ts';

const [scenePath, shotId, thrArg] = process.argv.slice(2);
const thr = Number(thrArg ?? 8);
const doc = JSON.parse(readFileSync(scenePath!, 'utf8'));
const shot = loadShots().shots.find((s) => s.id === shotId)!;
const layers = doc.layers.filter((l: any) => l.id.startsWith(`${shotId}.`));
const calib = JSON.parse(readFileSync(join(REPO, '.data/fit/calib.json'), 'utf8'));
const pool = await Pool.open(1);
try {
  await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [layers, MODEL_URI, calib.irisOffset] as const);
  for (const inst of shot.instances) {
    const L = layers.find((l: any) => l.id === inst.id);
    if (!L) continue;
    for (const t of (inst as any).track.filter((t: any) => t.eyes)) {
      const f = t.frame as number;
      if (f < L.in || f >= L.out) continue;
      const e = (await pool.pages[0]!.evaluate(([lid, f]) => (window as any).__fit.irisScreen(lid, f), [inst.id, f] as const)) as number[][];
      const d = Math.max(Math.hypot(e[0]![0]! - t.eyes[0][0], e[0]![1]! - t.eyes[0][1]), Math.hypot(e[1]![0]! - t.eyes[1][0], e[1]![1]! - t.eyes[1][1]));
      if (d > thr) console.log(`${shotId}\t${inst.id}\t${f}\teyes\t${JSON.stringify(t.eyes.map((p: number[]) => p.map(Math.round)))}\t→ ${JSON.stringify(e.map((p) => p.map(Math.round)))}\t（偏差 ${d.toFixed(0)}px）`);
    }
  }
} finally { await pool.close(); }
