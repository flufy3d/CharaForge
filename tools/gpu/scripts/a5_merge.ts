/*
 * A5：把 GPU 拟合导出的图层合并进一份临时全片场景（其余图层沿用基础场景），parseScene 校验通过才写出。
 *   node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_merge.ts \
 *     --base scenes/deepseek-chan/scene.json --shot s24 --layer .data/gpu/fit/s24/layer.json --out .data/gpu/fit/s24/scene.json
 * 替换规则：基础场景里 id 以「<镜头>.」开头的图层全部去掉，新图层放在原来第一个被替换图层的位置（没有则追加）。
 * 不覆盖基础场景（--out 与 --base 相同时报错）。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseScene } from '../../../packages/scene/src/index.ts';

const arg = (k: string) => {
  const i = process.argv.indexOf(`--${k}`);
  if (i < 0 || !process.argv[i + 1]) throw new Error(`缺少 --${k}`);
  return process.argv[i + 1]!;
};
const base = arg('base'), shot = arg('shot'), out = arg('out');
const layers = ([] as unknown[]).concat(JSON.parse(readFileSync(arg('layer'), 'utf8')));
if (resolve(base) === resolve(out)) throw new Error('不覆盖基础场景：--out 不能等于 --base');
const scene = JSON.parse(readFileSync(base, 'utf8'));
const pos = scene.layers.findIndex((l: any) => l.id.startsWith(`${shot}.`));
const kept = scene.layers.filter((l: any) => !l.id.startsWith(`${shot}.`));
const removed = scene.layers.length - kept.length;
const at = pos < 0 ? kept.length : pos;
kept.splice(at, 0, ...layers);
scene.layers = kept;
scene.meta = {
  ...scene.meta,
  title: `${scene.meta.title}（${shot} 由 GPU 拟合替换）`,
  description: `${scene.meta.description ?? ''} 临时场景：${shot} 的图层由 tools/gpu（A5）重新拟合，其余图层同 ${base}。`,
};
parseScene(scene);
writeFileSync(out, `${JSON.stringify(scene)}\n`);
console.log(`[a5_merge] 去掉 ${removed} 个 ${shot} 图层，插入 ${layers.length} 个（位置 ${at}），共 ${kept.length} 个图层 → ${out}（parseScene 通过）`);
