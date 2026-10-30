import { TypeCompiler } from '@sinclair/typebox/compiler';
import { Scene } from './scene.ts';
import { Shots } from './shots.ts';
import type { Scene as SceneT } from './scene.ts';
import type { Shots as ShotsT } from './shots.ts';

const sceneChecker = TypeCompiler.Compile(Scene);
const shotsChecker = TypeCompiler.Compile(Shots);

function check<T>(checker: { Check(v: unknown): boolean; Errors(v: unknown): Iterable<{ path: string; message: string }> }, v: unknown, what: string): T {
  if (checker.Check(v)) return v as T;
  const errs = [...checker.Errors(v)].slice(0, 10).map((e) => `  ${e.path}: ${e.message}`);
  throw new Error(`${what} 校验失败：\n${errs.join('\n')}`);
}

/** 结构校验 + 语义校验（引用完整性、区间合法性）。失败时抛出带路径的错误。 */
export function parseScene(v: unknown): SceneT {
  const s = check<SceneT>(sceneChecker, v, 'scene');
  const ids = (xs: { id: string }[] | undefined) => new Set((xs ?? []).map((x) => x.id));
  const models = ids(s.assets.models);
  const motions = ids(s.assets.motions);
  const exprs = ids(s.assets.expressions);
  const audio = ids(s.assets.audio);
  const problems: string[] = [];
  const seen = new Set<string>();
  for (const l of s.layers) {
    if (seen.has(l.id)) problems.push(`图层 ID 重复：${l.id}`);
    seen.add(l.id);
    if (!models.has(l.model)) problems.push(`图层 ${l.id} 引用了不存在的模型 ${l.model}`);
    if (!(l.out > l.in)) problems.push(`图层 ${l.id} 的 out 必须大于 in`);
    for (const m of l.motions ?? []) if (!motions.has(m.motion)) problems.push(`图层 ${l.id} 引用了不存在的 motion ${m.motion}`);
    for (const e of l.expressions ?? []) if (!exprs.has(e.expression)) problems.push(`图层 ${l.id} 引用了不存在的 expression ${e.expression}`);
    for (const p of l.params ?? []) if ('keys' in p.curve) {
      for (let i = 1; i < p.curve.keys.length; i++) {
        if (p.curve.keys[i]!.t < p.curve.keys[i - 1]!.t) { problems.push(`图层 ${l.id} 参数 ${p.id} 的关键帧未按时间排序`); break; }
      }
    }
  }
  for (const a of s.audio ?? []) if (!audio.has(a.audio)) problems.push(`音频片段引用了不存在的音频 ${a.audio}`);
  if (problems.length) throw new Error(`scene 语义校验失败：\n  ${problems.join('\n  ')}`);
  return s;
}

export function parseShots(v: unknown): ShotsT {
  const s = check<ShotsT>(shotsChecker, v, 'shots');
  const problems: string[] = [];
  let cursor = 0;
  for (const shot of s.shots) {
    if (shot.start !== cursor) problems.push(`镜头 ${shot.id} 起点 ${shot.start} 与上一镜头终点 ${cursor} 不衔接`);
    if (!(shot.end > shot.start)) problems.push(`镜头 ${shot.id} 的 end 必须大于 start`);
    cursor = shot.end;
  }
  if (cursor !== s.video.frameCount) problems.push(`镜头没有覆盖到最后一帧（${cursor} ≠ ${s.video.frameCount}）`);
  if (problems.length) throw new Error(`shots 语义校验失败：\n  ${problems.join('\n  ')}`);
  return s;
}
