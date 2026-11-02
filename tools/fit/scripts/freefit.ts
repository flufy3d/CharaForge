// 诊断（表达能力上限）：在单帧上放开一大组「物理之后」的参数（postPhysics add 偏移，含身体/肩/盆骨/前倾输出/体型/发束组/耳朵组）
// 与图层变换（含各向异性缩放 sx），用像素损失做 CMA 局部搜索，看 SSIM@360 能到多少、哪些维度被用到。
//   node --experimental-strip-types tools/fit/scripts/freefit.ts <scene.json> <layerId> <shot> <frames> [groups] [outTag]
//   groups：逗号分隔，可选 tf,sx,pose,body,lean,c2,hair,ears（缺省全部）
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';
import { CMAES } from '../src/cmaes.ts';
import { Pool } from '../src/pool.ts';
import { RefFrames } from '../src/refvideo.ts';
import { exclusionsAt, loadShots } from '../src/shotsutil.ts';
import { MODEL_URI, REPO } from '../src/paths.ts';

const [scenePath, layerId, shotId, framesArg, groupsArg, tag] = process.argv.slice(2);
const doc = JSON.parse(readFileSync(scenePath!, 'utf8'));
const layer = JSON.parse(JSON.stringify((doc.layers ?? [doc]).find((l: any) => l.id === layerId)));
const calib = JSON.parse(readFileSync(join(REPO, '.data/fit/calib.json'), 'utf8'));
const shot = loadShots().shots.find((s) => s.id === shotId)!;
const groups = new Set((groupsArg || 'tf,sx,pose,body,lean,c2,hair,ears').split(','));
const cdi = JSON.parse(readFileSync(join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.cdi3.json'), 'utf8'));
const allIds: string[] = cdi.Parameters.map((p: any) => p.Id);
const re = (r: RegExp) => allIds.filter((id) => r.test(id));
// 维度：[名字, 参数列表, 尺度]
const dims: { key: string; ids: string[]; sd: number }[] = [];
if (groups.has('pose')) for (const id of ['ParamAngleX', 'ParamAngleY', 'ParamAngleZ']) dims.push({ key: id, ids: [id], sd: 3 });
if (groups.has('body')) for (const id of ['ParamBodyAngleX', 'ParamBodyAngleY', 'ParamBodyAngleZ', 'ParamBodyAngleRX', 'ParamBodyAngleX2', 'ParamBodyAngleY2', 'ShoulderY', 'ShoulderZ', 'HipAngleX', 'HipAngleY', 'HipAngleZ'])
  dims.push({ key: id, ids: [id], sd: 3 });
if (groups.has('lean')) for (const id of ['Param50', 'Param53', 'Param54']) dims.push({ key: id, ids: [id], sd: 3 });
if (groups.has('c2')) dims.push({ key: 'Clothes2', ids: ['Clothes2'], sd: 0.08 }, { key: 'Param', ids: ['Param'], sd: 0.08 });
if (groups.has('hair')) for (const [k, r] of [['FrontHairX', /^FrontHairX\d$/], ['SideHairX', /^SideHairX\d$/], ['SideHairLX', /^SideHairLX\d$/], ['SideHairRX', /^SideHairRX\d$/], ['LateSideHairX', /^LateSideHairX\d$/], ['LateHairX', /^LateHairX\d$/], ['LateSideHairY', /^LateSideHairY\d$/], ['LateHairY', /^LateHairY\d$/]] as const)
  dims.push({ key: k, ids: re(r), sd: 0.25 });
if (groups.has('ears')) for (const [k, r] of [['LEar', /^LEar\d$/], ['REar', /^REar\d$/], ['LFlip', /^LFlip2?$/], ['RFlip', /^RFlip2?$/]] as const) dims.push({ key: k, ids: re(r), sd: 0.25 });
const tfDims = groups.has('tf') ? [{ key: 'dx', sd: 4 }, { key: 'dy', sd: 4 }, { key: 'dls', sd: 0.02 }, { key: 'drot', sd: 1 }, ...(groups.has('sx') ? [{ key: 'lsx', sd: 0.02 }] : [])] : [];
// 给图层加 postPhysics add 轨（全零）
const n = layer.out - layer.in;
const postIds = [...new Set(dims.flatMap((d) => d.ids))];
for (const id of postIds) layer.params.push({ id, stage: 'postPhysics', mode: 'add', curve: { baked: { start: layer.in, values: Array(n).fill(0) } } });
const bake = (v: any, f: number) => v.baked.values[f - v.baked.start];
const outDir = join(REPO, '.data/fit/diag', `free_${shotId}${tag ? '_' + tag : ''}`);
mkdirSync(outDir, { recursive: true });
const ref = new RefFrames(shot.start, shot.end);
const pool = await Pool.open(4);
const results: any[] = [];
const score = (f: number, rgba: Buffer) => {
  writeFileSync(join(outDir, `ours_${f}.rgba`), rgba);
  writeFileSync(join(outDir, 'manifest.json'), JSON.stringify({ frames: [{ frame: f, ours: join(outDir, `ours_${f}.rgba`), ref: ref.file, refIndex: f - shot.start, excl: exclusionsAt(shot, f) }] }));
  const r = spawnSync(join(REPO, 'tools/.venv/bin/python'), ['-I', join(REPO, 'tools/fit/score360.py'), join(outDir, 'manifest.json')], { encoding: 'utf8' });
  return JSON.parse(r.stdout)[String(f)] as number;
};
try {
  await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [[layer], MODEL_URI, calib.irisOffset] as const);
  for (const f of framesArg!.split(',').map(Number)) {
    await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
    const T = layer.transform;
    const tf0 = [bake(T.x, f), bake(T.y, f), bake(T.scale, f), bake(T.rotation, f)];
    const D = tfDims.length + dims.length;
    const cand = (x: number[]) => {
      const v: Record<string, number> = {};
      dims.forEach((d, i) => { for (const id of d.ids) v[`post:${id}`] = (v[`post:${id}`] ?? 0) + x[tfDims.length + i]!; });
      const t = Object.fromEntries(tfDims.map((d, i) => [d.key, x[i]!]));
      return { [layerId!]: { tf: [tf0[0] + (t.dx ?? 0), tf0[1] + (t.dy ?? 0), tf0[2] * Math.exp(t.dls ?? 0), tf0[3] + (t.drot ?? 0)] as [number, number, number, number], ...(groups.has('sx') ? { sx: Math.exp(t.lsx ?? 0) } : {}), set: [[f, v]] } };
    };
    const lossX = async (xs: number[][], scale: number) => (await pool.evalBatch<{ loss: number; ssim: number }>(f, xs.map(cand), { scale, r: 3, wcol: 0.5, weye: 0, wseg: 0, wgrad: 0.2, eyesOf: [] })).map((r) => r.loss);
    const zero = Array(D).fill(0);
    const s0 = score(f, Buffer.from(((await pool.pages[0]!.evaluate(([f, c]) => { (window as any).__fit.commit(f, c); return (window as any).__fit.render(f, 1); }, [f, cand(zero)] as const)) as any).data, 'base64'));
    let best = { x: zero, c: Infinity };
    for (const [scale, gens, sig] of [[0.25, 60, 1.5], [0.5, 50, 0.6]] as const) {
      const es = new CMAES(best.x, { sigma: sig, scales: [...tfDims.map((d) => d.sd), ...dims.map((d) => d.sd)], lambda: 16, seed: f });
      best = { x: best.x, c: (await lossX([best.x], scale))[0]! };
      for (let g = 0; g < gens; g++) {
        const xs = es.ask();
        const cs = await lossX(xs, scale);
        es.tell(cs);
        cs.forEach((c, i) => { if (c < best.c) best = { x: xs[i]!, c }; });
      }
    }
    const img = (await pool.pages[0]!.evaluate(([f, c]) => { (window as any).__fit.commit(f, c); return (window as any).__fit.render(f, 1); }, [f, cand(best.x)] as const)) as any;
    const s1 = score(f, Buffer.from(img.data, 'base64'));
    const xs = Object.fromEntries([...tfDims.map((d) => d.key), ...dims.map((d) => d.key)].map((k, i) => [k, +best.x[i]!.toFixed(3)]));
    results.push({ frame: f, before: s0, after: s1, x: xs });
    console.log(`f=${f} SSIM@360 ${s0.toFixed(3)} → ${s1.toFixed(3)}  ${JSON.stringify(xs)}`);
    // 写一张 PNG（与参考同尺寸 RGBA）供肉眼对比
    writeFileSync(join(outDir, `best_${f}.rgba`), Buffer.from(img.data, 'base64'));
  }
  writeFileSync(join(outDir, 'result.json'), JSON.stringify({ groups: [...groups], results }, null, 1));
} finally { await pool.close(); ref.close(); }
