import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseScene, sample, sampleCurve, cubicBezier } from '../src/index.ts';

const example = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'examples', 'minimal.scene.json'), 'utf8'));
const accessory = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'examples', 'accessory.scene.json'), 'utf8'));
const editorDemo = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'examples', 'editor-demo.scene.json'), 'utf8'));

test('示例场景通过校验', () => {
  parseScene(example);
});

test('饰品示例通过校验（图片资源 + 独立/attach 图片图层）', () => {
  parseScene(accessory);
});

test('编辑器示例通过校验（背景光晕 + 星星 + 音乐轨）', () => {
  const scene = parseScene(editorDemo);
  assert.equal(scene.layers.length, 5);
  assert.equal(scene.assets.images?.length, 3);
  assert.equal(scene.audio?.length, 1);
});

test('图片图层引用不存在的图片会报错', () => {
  const bad = structuredClone(accessory);
  bad.layers[1].image = 'nope';
  assert.throws(() => parseScene(bad), /不存在的图片/);
});

test('图片图层只能绑定模型图层', () => {
  const bad = structuredClone(accessory);
  bad.layers[1].attach.layer = 'badge';
  assert.throws(() => parseScene(bad), /只能绑定模型图层/);
});

test('图片图层绑定不存在的图层会报错', () => {
  const bad = structuredClone(accessory);
  bad.layers[1].attach.layer = 'nope';
  assert.throws(() => parseScene(bad), /绑定了不存在的图层/);
});

test('图片资源 uri 只允许 png/webp 或 PNG/WebP data URI', () => {
  const bad = structuredClone(accessory);
  bad.assets.images[0].uri = 'accessories/hat.jpg';
  assert.throws(() => parseScene(bad), /uri 必须是/);
  const badData = structuredClone(accessory);
  badData.assets.images[0].uri = 'data:image/jpeg;base64,AAAA';
  assert.throws(() => parseScene(badData), /uri 必须是/);
  const badRemote = structuredClone(accessory);
  badRemote.assets.images[0].uri = 'https://example.com/hat.png';
  assert.throws(() => parseScene(badRemote), /uri 必须是/);
  const badEscape = structuredClone(accessory);
  badEscape.assets.images[0].uri = '../secret/hat.png';
  assert.throws(() => parseScene(badEscape), /uri 必须是/);
});

test('图片尺寸超过 schema 上限会报错', () => {
  const bad = structuredClone(accessory);
  bad.assets.images[0].width = 4097;
  assert.throws(() => parseScene(bad), /校验失败/);
});

test('图片资源 ID 重复会报错', () => {
  const bad = structuredClone(accessory);
  bad.assets.images.push(structuredClone(bad.assets.images[0]));
  assert.throws(() => parseScene(bad), /图片资源 ID 重复/);
});

test('图片图层不能带模型字段', () => {
  const bad = structuredClone(accessory);
  bad.layers[1].model = 'ds';
  assert.throws(() => parseScene(bad), /校验失败/);
});

test('attach.anchor 超出 0..1 会报错', () => {
  const bad = structuredClone(accessory);
  bad.layers[1].attach.anchor = [1.2, 0.5];
  assert.throws(() => parseScene(bad), /校验失败/);
});

test('引用不存在的资源会报错', () => {
  const bad = structuredClone(example);
  bad.layers[0].model = 'nope';
  assert.throws(() => parseScene(bad), /不存在的模型/);
});

test('曲线插值语义', () => {
  const c = { keys: [{ t: 0, v: 0 }, { t: 10, v: 10, ease: 'step' as const }, { t: 20, v: 0 }] };
  assert.equal(sampleCurve(c, -5), 0);
  assert.equal(sampleCurve(c, 5), 5);
  assert.equal(sampleCurve(c, 15), 10);
  assert.equal(sampleCurve(c, 20), 0);
  assert.equal(sampleCurve(c, 99), 0);
  assert.deepEqual(sampleCurve({ keys: [{ t: 0, v: [0, 0, 0] }, { t: 2, v: [2, 4, 6] }] }, 1), [1, 2, 3]);
  const b = { baked: { start: 10, values: [1, 2, 3] } };
  assert.equal(sampleCurve(b, 0), 1);
  assert.equal(sampleCurve(b, 11), 2);
  assert.equal(sampleCurve(b, 50), 3);
  assert.equal(sample(undefined, 3, 7), 7);
  assert.equal(sample(4, 3, 7), 4);
});

test('cubicBezier 与 CSS ease-in-out 一致', () => {
  assert.ok(Math.abs(cubicBezier(0.42, 0, 0.58, 1, 0.5) - 0.5) < 1e-9);
  assert.ok(Math.abs(cubicBezier(0.42, 0, 0.58, 1, 0.25) - 0.129) < 1e-3);
});
