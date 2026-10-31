/*
 * 把逐镜头拟合结果（.data/fit/<shot>/result.json）合成为 scenes/deepseek-chan/scene.json：
 * 全片时长、原音频、模型署名、每个已拟合镜头一个 shot 标记；没拟合的时间段没有图层（留空）。
 * 写出前用 @charaforge/scene 的 parseScene 校验。
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseScene } from '../../../packages/scene/src/index.ts';
import { FIT_DIR, MODEL_URI, REPO } from './paths.ts';
import { loadShots } from './shotsutil.ts';

export const MODEL_CREDITS =
  'DeepSeek-chan — Illustration & PSD Layer Separation: Kamiya Ryoko / Live2D Rigging: 霓虹猫_NeonNeko / ' +
  'Free model distributed by Kamiya Ryoko (https://vgen.co/KAMIYA, https://x.com/kamiya_ryoko)';

export function assembleScene(out?: string): void {
  const shots = loadShots();
  const layers: unknown[] = [];
  const markers: unknown[] = [];
  const fitted: Record<string, unknown> = {};
  for (const shot of shots.shots) {
    const file = join(FIT_DIR, shot.id, 'result.json');
    if (!existsSync(file)) continue;
    const r = JSON.parse(readFileSync(file, 'utf8'));
    layers.push(...r.layers);
    markers.push({ id: shot.id, start: shot.start, end: shot.end, kind: 'shot', label: `${shot.id} ${shot.category}` });
    fitted[shot.id] = { elapsedSec: Math.round(r.elapsedSec), fittedFrames: r.fittedFrames, renders: r.evals };
  }
  const scene = {
    $schema: '../../packages/scene/schema/scene.v1.json',
    format: 'charaforge.scene',
    version: 1,
    meta: {
      title: 'DeepSeek-chan PV 复刻（打样：s15 / s24 / s43）',
      description: '由 tools/fit 从参考视频逐帧反推；只包含已拟合镜头的角色图层，其余时间为空。',
      credits: [MODEL_CREDITS],
      provenance: { source: 'fit', tool: '@charaforge/fit', detail: { shots: fitted, reference: shots.video.uri, sha256: shots.video.sha256 } },
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
  const path = out ?? join(REPO, 'scenes/deepseek-chan/scene.json');
  writeFileSync(path, `${JSON.stringify(scene)}\n`);
  console.log(`[assemble] ${layers.length} 个图层（${Object.keys(fitted).join(', ')}）→ ${path}（parseScene 通过）`);
}
