// 诊断：给定图层（result.json）在指定帧上的拟合损失各项（不做优化），用于比较「真值参数」与「拟合结果」的损失
//   FIT_VIDEO=... FIT_VIDEO_OFFSET=... node --experimental-strip-types tools/fit/scripts/eval_layers.ts <result.json> <shot> 2416,2430
import { readFileSync } from 'node:fs';
import { Pool } from '../src/pool.ts';
import { RefFrames } from '../src/refvideo.ts';
import { exclusionsAt, loadShots } from '../src/shotsutil.ts';
import { MODEL_URI } from '../src/paths.ts';
const [resPath, shotId, framesArg] = process.argv.slice(2);
const res = JSON.parse(readFileSync(resPath!, 'utf8'));
const calib = JSON.parse(readFileSync('.data/fit/calib.json', 'utf8'));
const shot = loadShots().shots.find((s) => s.id === shotId)!;
const ref = new RefFrames(shot.start, shot.end);
const pool = await Pool.open(1);
try {
  await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [res.layers, MODEL_URI, calib.irisOffset] as const);
  for (const f of framesArg!.split(',').map(Number)) {
    await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
    const out: string[] = [];
    for (const [scale, r] of [[0.125, 1], [0.25, 2], [0.5, 3]] as const) {
      const [x] = await pool.evalBatch<any>(f, [{}], { scale, r, wcol: 1, weye: 0.08, wseg: 1, wgrad: 0.5, eyesOf: res.layers.map((l: any) => l.id) });
      out.push(`@${scale}: loss ${x.loss.toFixed(3)} ssim ${x.ssim.toFixed(3)} seg ${x.seg.toFixed(3)} grad ${(x.grad ?? 0).toFixed(3)} eye ${x.eye.toFixed(1)}`);
    }
    console.log(`f=${f}  ${out.join(' | ')}`);
  }
} finally { await pool.close(); }
