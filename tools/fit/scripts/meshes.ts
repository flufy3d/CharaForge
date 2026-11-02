// 诊断：按场景规范求值图层（runtime 的 LayerEvaluator，无 GL），导出指定帧所有 drawable 的屏幕坐标网格，
// 供 tools/fit/regions.py 光栅化出「分区标签图」（脸、前发、侧发、后发、耳、身体、领结……）。
//   node --import tools/fit/src/geom/register.mjs --experimental-strip-types tools/fit/scripts/meshes.ts <scene.json|layer.json> <layerId> <frames> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileFetcher, loadRuntime } from '../src/geom/core.ts';
import { MODEL_URI, REPO } from '../src/paths.ts';

const [scenePath, layerId, framesArg, outPath] = process.argv.slice(2);
const doc = JSON.parse(readFileSync(scenePath!, 'utf8'));
const layer = (doc.layers ?? [doc]).find((l: any) => l.id === layerId);
if (!layer) throw new Error(`没有图层 ${layerId}`);
const rt = await loadRuntime();
const res = await rt.ModelResource.load(null, pathToFileURL(join(REPO, 'assets', MODEL_URI)).href, fileFetcher);
const inst = res.createInstance();
const model = inst.model;
const core = model.getModel();
const c = inst.canvas;
const noClip = (id: string): never => { throw new Error(`不支持 motion/expression：${id}`); };
const ev = new rt.LayerEvaluator(layer, inst, 60, noClip, noClip);
// 部件路径（名字取 cdi3）
const cdi = JSON.parse(readFileSync(join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.cdi3.json'), 'utf8'));
const partName = new Map<string, string>(cdi.Parts.map((p: any) => [p.Id, p.Name]));
const partParents = core.parts.parentIndices as Int32Array;
const pathOf = (d: number) => {
  const out: string[] = [];
  let p = model.getDrawableParentPartIndex(d);
  while (p >= 0) { const id = core.parts.ids[p] as string; out.unshift(partName.get(id) ?? id); p = partParents[p]!; }
  return out.join('/');
};
const bake = (v: any, f: number, dflt: number) => (v?.baked ? v.baked.values[Math.min(v.baked.values.length - 1, Math.max(0, f - v.baked.start))] : typeof v === 'number' ? v : dflt);
const frames: any[] = [];
for (const f of framesArg!.split(',').map(Number).sort((a, b) => a - b)) {
  ev.prepare(f, undefined, 'draw');
  const T = layer.transform;
  const t = { x: bake(T.x, f, 0), y: bake(T.y, f, 0), s: bake(T.scale, f, 1), sx: bake(T.scaleX, f, 1), sy: bake(T.scaleY, f, 1), rot: bake(T.rotation, f, 0), ax: T.anchorX ?? 0, ay: T.anchorY ?? 0 };
  const masks = core.drawables.maskCounts as Int32Array;
  const cs = Math.cos((t.rot * Math.PI) / 180), sn = Math.sin((t.rot * Math.PI) / 180);
  const order = (core.drawables.renderOrders ?? core.renderOrders ?? core.getRenderOrders?.()) as Int32Array;
  if (!order || !masks) throw new Error("keys: " + Object.keys(core) + " | " + Object.keys(core.drawables));

  const ds: any[] = [];
  for (let d = 0; d < model.getDrawableCount(); d++) {
    if (!model.getDrawableDynamicFlagIsVisible(d)) continue;
    const op = model.getDrawableOpacity(d);
    if (op < 0.05) continue;
    const v = model.getDrawableVertices(d) as Float32Array;
    const pts: number[] = [];
    for (let i = 0; i < v.length / 2; i++) {
      const px = (c.originX + v[2 * i]! * c.pixelsPerUnit - t.ax) * t.s * t.sx, py = (c.originY - v[2 * i + 1]! * c.pixelsPerUnit - t.ay) * t.s * t.sy;
      pts.push(+(t.x + cs * px - sn * py).toFixed(2), +(t.y + sn * px + cs * py).toFixed(2));
    }
    ds.push({
      d, id: core.drawables.ids[d], path: pathOf(d), order: order[d], opacity: op,
      blend: model.getDrawableBlendMode(d), masked: masks[d] > 0, tex: model.getDrawableTextureIndex(d),
      pts, uv: Array.from(model.getDrawableVertexUvs(d) as Float32Array, (x) => +x.toFixed(5)), idx: Array.from(model.getDrawableVertexIndices(d) as Uint16Array),
    });
  }
  frames.push({ frame: f, transform: t, drawables: ds });
}
writeFileSync(outPath!, JSON.stringify({ layer: layerId, frames }));
console.log(`写出 ${outPath}：${frames.length} 帧，${frames[0]?.drawables.length} 个可见 drawable`);
