import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseScene, sample, sampleCurve, cubicBezier } from '../src/index.ts';

const example = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'examples', 'minimal.scene.json'), 'utf8'));

test('示例场景通过校验', () => {
  parseScene(example);
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
