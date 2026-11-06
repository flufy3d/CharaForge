// A1：从 scenes/deepseek-chan/scene.json 按场景规范实际求值出参数帧（真实分布），供 Core 对拍。
// 用 packages/runtime 同一份 LayerEvaluator（含物理），每帧导出交给 Core 的输入：参数值 + 部件不透明度。
//
//   node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types tools/gpu/scripts/a1_scene_params.ts [每层帧数=6]
//
// 输出 .data/gpu/ref/scene_frames.{bin,json}：bin = [N, P] float32 参数，接着 [N, Q] float32 部件不透明度。
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileFetcher, loadRuntime } from '../../fit/src/geom/core.ts';
import { MODEL_URI, REPO } from '../../fit/src/paths.ts';

const perLayer = Number(process.argv[2] ?? 6);
const OUT = join(REPO, '.data/gpu/ref');
mkdirSync(OUT, { recursive: true });

const scene = JSON.parse(readFileSync(join(REPO, 'scenes/deepseek-chan/scene.json'), 'utf8'));
const rt = await loadRuntime();
const res = await rt.ModelResource.load(null, pathToFileURL(join(REPO, 'assets', MODEL_URI)).href, fileFetcher);
const noClip = (id: string): never => { throw new Error(`不支持 motion/expression：${id}`); };

const params: Float32Array[] = [];
const parts: Float32Array[] = [];
const index: { layer: string; frame: number }[] = [];
for (const layer of scene.layers) {
  if (layer.kind && layer.kind !== 'model') continue;
  const inst = res.createInstance();
  let ev;
  try {
    ev = new rt.LayerEvaluator(layer, inst, scene.timebase.fps, noClip, noClip);
  } catch (e) {
    console.warn(`跳过 ${layer.id}：${(e as Error).message}`);
    continue;
  }
  // 图层区间内均匀取帧（按时间顺序求值，物理逐帧推进）
  const n = Math.min(perLayer, layer.out - layer.in);
  const frames = Array.from({ length: n }, (_, i) => Math.floor(layer.in + ((i + 0.5) * (layer.out - layer.in)) / n));
  try {
    for (const f of frames) {
      ev.prepare(f, undefined, 'params');
      const inp = inst.exportInputs();
      params.push(Float32Array.from(inp.params));
      parts.push(Float32Array.from(inp.parts));
      index.push({ layer: layer.id, frame: f });
    }
  } catch (e) {
    console.warn(`跳过 ${layer.id} 的剩余帧：${(e as Error).message}`);
  }
  inst.dispose?.();
}
const N = params.length;
const P = params[0]!.length, Q = parts[0]!.length;
const buf = new Float32Array(N * (P + Q));
params.forEach((p, i) => buf.set(p, i * P));
parts.forEach((q, i) => buf.set(q, N * P + i * Q));
writeFileSync(join(OUT, 'scene_frames.bin'), new Uint8Array(buf.buffer));
writeFileSync(join(OUT, 'scene_frames.json'), JSON.stringify({ n: N, P, Q, index }));
console.log(`场景帧 ${N} 个（${new Set(index.map((x) => x.layer)).size} 个图层），P=${P}，Q=${Q}`);
