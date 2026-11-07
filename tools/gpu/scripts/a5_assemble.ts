/*
 * A5：由各拟合单元的结果（<units>/<单元>/layers.json + fit.json）合成全片场景，结构同 tools/fit/src/assemble.ts
 * （全片时长、原音频、模型署名、每个镜头一个 shot 标记）。还没有 GPU 结果的单元沿用基础场景里该单元的图层（--base，可关闭），
 * 在 meta.provenance 里列出。parseScene 校验通过才写出；不覆盖 scenes/deepseek-chan/scene.json。
 *   node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_assemble.ts \
 *     --units .data/gpu/fit/full/units --out .data/gpu/fit/full/scene.json [--base scenes/deepseek-chan/scene.json | --base none]
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseScene } from '../../../packages/scene/src/index.ts';
import { MODEL_CREDITS } from '../../fit/src/assemble.ts';
import { MODEL_URI, REPO } from '../../fit/src/paths.ts';
import { fitUnits } from '../../fit/src/plans.ts';
import { loadShots } from '../../fit/src/shotsutil.ts';

const arg = (k: string, d?: string) => {
  const i = process.argv.indexOf(`--${k}`);
  if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1]!;
  if (d !== undefined) return d;
  throw new Error(`缺少 --${k}`);
};
const unitsDir = arg('units'), out = arg('out'), basePath = arg('base', join(REPO, 'scenes/deepseek-chan/scene.json'));
if (resolve(out) === resolve(join(REPO, 'scenes/deepseek-chan/scene.json'))) throw new Error('不覆盖 scenes/deepseek-chan/scene.json');
const shots = loadShots();
const base = basePath !== 'none' && existsSync(basePath) ? JSON.parse(readFileSync(basePath, 'utf8')) : null;
const done = new Set(existsSync(unitsDir) ? readdirSync(unitsDir).filter((d) => existsSync(join(unitsDir, d, 'layers.json'))) : []);
const layers: unknown[] = [];
const markers: unknown[] = [];
const gpu: Record<string, unknown> = {};
const fallback: string[] = [];
const missing: string[] = [];
for (const u of fitUnits(shots.shots)) {
  if (done.has(u.id)) {
    const L = JSON.parse(readFileSync(join(unitsDir, u.id, 'layers.json'), 'utf8'));
    const f = JSON.parse(readFileSync(join(unitsDir, u.id, 'fit.json'), 'utf8'));
    layers.push(...L);
    gpu[u.id] = { shots: u.shots, layers: L.map((l: any) => l.id), batch: f.batch ?? [u.id], totalSec: Math.round(f.totalSec) };
  } else if (base) {
    const L = base.layers.filter((l: any) => u.shots.some((s) => l.id.startsWith(`${s}.`)));
    if (L.length) { layers.push(...L); fallback.push(u.id); } else missing.push(u.id);
  } else {
    missing.push(u.id);
  }
  for (const id of u.shots) {
    const s = shots.shots.find((x) => x.id === id)!;
    markers.push({ id: s.id, start: s.start, end: s.end, kind: 'shot', label: `${s.id} ${s.category}` });
  }
}
const scene = {
  $schema: '../../packages/scene/schema/scene.v1.json',
  format: 'charaforge.scene',
  version: 1,
  meta: {
    title: 'DeepSeek-chan PV 复刻（GPU 拟合全片）',
    description: '由 tools/gpu（A5，可微分 GPU 前向 + 整段联合优化）按拟合单元批量拟合；' +
      (fallback.length ? `单元 ${fallback.join(', ')} 尚未重拟合，沿用 ${basePath} 的图层。` : '全部单元均为 GPU 拟合结果。'),
    credits: [MODEL_CREDITS],
    provenance: { source: 'fit', tool: '@charaforge/gpu-fit (A5)', detail: { units: gpu, fallback, missing, reference: shots.video.uri, sha256: shots.video.sha256 } },
  },
  timebase: { fps: shots.video.fps },
  duration: shots.video.frameCount,
  composition: { width: shots.video.width, height: shots.video.height, background: null, antialias: true },
  assets: {
    models: [{ id: 'dsc', uri: MODEL_URI, name: 'DeepSeek-chan（大肥鱼）', credits: MODEL_CREDITS }],
    audio: [{ id: 'pv', uri: 'video/DeepSeek-chan.wav' }],
  },
  layers,
  audio: [{ audio: 'pv', start: 0 }],
  markers,
};
parseScene(scene);
writeFileSync(out, `${JSON.stringify(scene)}\n`);
console.log(`[a5_assemble] GPU 单元 ${Object.keys(gpu).length} 个、沿用旧图层 ${fallback.length} 个、缺 ${missing.length} 个；共 ${layers.length} 个图层 → ${out}（parseScene 通过）`);
