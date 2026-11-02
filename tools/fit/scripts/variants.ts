// 诊断：同一帧渲染若干「参数变体」（postPhysics add 偏移），变换用眼睛对齐到参考帧的虹膜（shots 轨迹），
// 输出 SSIM@360 并把 参考 | 变体… 拼成一张图，供肉眼判断某个参数是否能解释差异。
//   node --experimental-strip-types tools/fit/scripts/variants.ts <scene.json> <layerId> <shot> <frame> '<[{"Clothes2":0.1},...]>' <out.jpg>
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';
import { Pool } from '../src/pool.ts';
import { RefFrames } from '../src/refvideo.ts';
import { exclusionsAt, loadShots, trackAt } from '../src/shotsutil.ts';
import { MODEL_URI, REPO } from '../src/paths.ts';

const [scenePath, layerId, shotId, frameArg, variantsArg, outJpg] = process.argv.slice(2);
const doc = JSON.parse(readFileSync(scenePath!, 'utf8'));
const layer = JSON.parse(JSON.stringify((doc.layers ?? [doc]).find((l: any) => l.id === layerId)));
const calib = JSON.parse(readFileSync(join(REPO, '.data/fit/calib.json'), 'utf8'));
const shot = loadShots().shots.find((s) => s.id === shotId)!;
const f = Number(frameArg);
const variants: Record<string, number>[] = JSON.parse(variantsArg!);
const n = layer.out - layer.in;
const ids = [...new Set(variants.flatMap((v) => Object.keys(v)))].filter((k) => !k.startsWith('tf.') && k !== 'align');
for (const id of ids) layer.params.push({ id, stage: 'postPhysics', mode: 'add', curve: { baked: { start: layer.in, values: Array(n).fill(0) } } });
const inst = shot.instances.find((i) => layerId!.endsWith(i.id)) ?? shot.instances[0]!;
const eyes = trackAt(inst, 'eyes', f) as [[number, number], [number, number]] | null;
const outDir = join(REPO, '.data/fit/diag/variants');
mkdirSync(outDir, { recursive: true });
const ref = new RefFrames(shot.start, shot.end);
const pool = await Pool.open(1);
const files: string[] = [];
try {
  await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [[layer], MODEL_URI, calib.irisOffset] as const);
  await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
  const man: any[] = [];
  for (const [k, v] of variants.entries()) {
    const set = Object.fromEntries(ids.map((id) => [`post:${id}`, v[id] ?? 0]));
    const vals: any = { set: [[f, set]] };
    if (v.align && eyes) vals.alignEyes = eyes;
    const img = (await pool.pages[0]!.evaluate(([f, lid, c]) => {
      const F = (window as any).__fit;
      if (c.alignEyes) { const r = F.evalBatch(f, [{ [lid]: c }], { scale: 0.25, r: 1, wcol: 0, weye: 0 }); c = { ...c, alignEyes: undefined, tf: r[0].tf }; }
      F.commit(f, { [lid]: c });
      return F.render(f, 1);
    }, [f, layerId!, vals] as const)) as any;
    const p = join(outDir, `v${k}_${f}.rgba`);
    writeFileSync(p, Buffer.from(img.data, 'base64'));
    files.push(p);
    man.push({ frame: f, ours: p, ref: ref.file, refIndex: f - shot.start, excl: exclusionsAt(shot, f) });
  }
  for (const [k, m] of man.entries()) {
    writeFileSync(join(outDir, 'm.json'), JSON.stringify({ frames: [m] }));
    const r = spawnSync(join(REPO, 'tools/.venv/bin/python'), ['-I', join(REPO, 'tools/fit/score360.py'), join(outDir, 'm.json')], { encoding: 'utf8' });
    console.log(`变体 ${k} ${JSON.stringify(variants[k])}: SSIM@360 ${JSON.parse(r.stdout)[String(f)]?.toFixed(3)}`);
  }
  // 拼图（python）
  const py = `import sys,numpy as np,cv2
W,H=1280,720
ref=np.asarray(np.memmap(sys.argv[1],np.uint8,'r')[int(sys.argv[2])*W*H*3:(int(sys.argv[2])+1)*W*H*3]).reshape(H,W,3).astype(np.float64)
ims=[ref]
for p in sys.argv[4:]:
  o=np.fromfile(p,np.uint8).reshape(H,W,4).astype(np.float64); a=o[...,3:]/255.0; ims.append(o[...,:3]+(1-a)*ref)
ims=[cv2.resize(i,(640,360),interpolation=cv2.INTER_AREA) for i in ims]
while len(ims)%2: ims.append(np.zeros_like(ims[0]))
rows=[np.concatenate(ims[i:i+2],1) for i in range(0,len(ims),2)]
cv2.imwrite(sys.argv[3],cv2.cvtColor(np.concatenate(rows,0).astype(np.uint8),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])`;
  writeFileSync(join(outDir, 'mosaic.py'), py);
  spawnSync(join(REPO, 'tools/.venv/bin/python'), ['-I', join(outDir, 'mosaic.py'), ref.file, String(f - shot.start), outJpg!, ...files], { stdio: 'inherit' });
} finally { await pool.close(); ref.close(); }
