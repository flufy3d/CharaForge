/*
 * 图片图层 / ImageResource 的运行时测试。
 *
 * 覆盖：
 *   - 固定位置的透明棋盘 PNG：位置、透明边、alpha；
 *   - 图片图层的 in/out 活跃区间；
 *   - attach 跟随目标模型图层的平移；
 *   - scaleWith=false 不继承目标缩放；
 *   - 图片图层在 Worker 路径与同步路径逐位一致；
 *   - 不可见图片图层不影响旧模型图层的像素；
 *   - 声明尺寸与实际解码尺寸不一致时拒绝加载。
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { openHarness, type Harness } from '../../../apps/player/scripts/harness-node.ts';

const MODEL = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
const accessory = JSON.parse(readFileSync(join(import.meta.dirname, '..', '..', 'scene', 'examples', 'accessory.scene.json'), 'utf8'));
const CHECKER = accessory.assets.images[0];

function baseScene(images: object[], layers: object[], models: object[] = []) {
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: { title: 'image-test' },
    timebase: { fps: 60 },
    duration: 30,
    composition: { width: 320, height: 180, background: null },
    assets: { models, images },
    layers,
  };
}

function imageLayer(overrides: Record<string, unknown> = {}) {
  return { id: 'img', kind: 'image', image: 'checker', in: 0, out: 30, space: 'screen', transform: { x: 160, y: 90, scale: 1, anchorX: 32, anchorY: 32 }, ...overrides };
}

type Pixels = { width: number; height: number; data: string };

function alphaBBox(p: Pixels) {
  const d = Buffer.from(p.data, 'base64');
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, count = 0;
  for (let y = 0; y < p.height; y++) {
    for (let x = 0; x < p.width; x++) {
      if (d[(y * p.width + x) * 4 + 3]! > 0) {
        count++;
        x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      }
    }
  }
  return { x0, y0, x1, y1, count };
}

function alphaAt(p: Pixels, x: number, y: number): number {
  return Buffer.from(p.data, 'base64')[(y * p.width + x) * 4 + 3]!;
}

function pixelAt(p: Pixels, x: number, y: number): [number, number, number, number] {
  const d = Buffer.from(p.data, 'base64');
  const i = (y * p.width + x) * 4;
  return [d[i]!, d[i + 1]!, d[i + 2]!, d[i + 3]!];
}

let h: Harness;

before(async () => {
  h = await openHarness();
  console.log(`[gpu] ${h.info.renderer}  (msaa ${h.info.msaa})`);
});
after(async () => h?.close());

async function renderRgba(f: number): Promise<Pixels> {
  const r = await h.page.evaluate((frame) => window.__charaforge.renderFrame(frame, { read: {} }), f);
  return { width: r.width!, height: r.height!, data: r.data! };
}

async function renderSha(f: number): Promise<string> {
  const r = await h.page.evaluate((frame) => window.__charaforge.renderFrame(frame, { read: { format: 'sha256' } }), f);
  return r.sha256!;
}

test('透明棋盘图片固定位置渲染：位置、透明边、alpha', async () => {
  const scene = baseScene([CHECKER], [imageLayer()]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const p = await renderRgba(0);
  assert.equal(alphaAt(p, 10, 10), 0, '图片外应透明');
  assert.equal(alphaAt(p, 128, 90), 0, '图片 4px 透明边框');
  assert.ok(alphaAt(p, 140, 70) > 250, '图片内部应不透明');
  assert.ok(alphaAt(p, 160, 90) > 250, '图片中心应不透明');
  const box = alphaBBox(p);
  assert.ok(box.count > 2000, `图片像素数量：${box.count}`);
  // 64×64 图片有 4px 透明边框，所以可见 alpha 从 128+4=132 到 192−1−4=187
  assert.ok(box.x0 >= 130 && box.x0 <= 134, `左边界：${box.x0}`);
  assert.ok(box.x1 >= 185 && box.x1 <= 189, `右边界：${box.x1}`);
});

test('图片图层遵守 in/out 活跃区间', async () => {
  const scene = baseScene([CHECKER], [imageLayer({ in: 10, out: 20 })]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const outside = alphaBBox(await renderRgba(5));
  assert.equal(outside.count, 0, 'in 之前图片不应绘制');
  const after = alphaBBox(await renderRgba(25));
  assert.equal(after.count, 0, 'out 之后图片不应绘制');
  const inside = alphaBBox(await renderRgba(15));
  assert.ok(inside.count > 2000, `活跃区间内图片应可见：${inside.count}`);
});

test('runtime.addImageLayer 可在运行时追加图片图层', async () => {
  const scene = baseScene([], []);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  await h.page.evaluate(([asset, layer]) => {
    const rt = (window as unknown as { __cfRuntime: { addImageLayer(a: unknown, l: unknown): Promise<void> } }).__cfRuntime;
    return rt.addImageLayer(asset, layer);
  }, [CHECKER, imageLayer()] as const);
  const box = alphaBBox(await renderRgba(0));
  assert.ok(box.count > 2000, `追加的图片应可见：${box.count}`);
});

test('图片图层复用 opacity / clip / color filter', async () => {
  const scene = baseScene([CHECKER], [
    imageLayer({
      opacity: 0.5,
      clip: { x: 0, y: 0, width: 160, height: 180 },
      filters: [{ type: 'color', saturation: 0 }],
    }),
  ]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const p = await renderRgba(0);
  const a = alphaAt(p, 145, 90);
  assert.ok(a >= 110 && a <= 145, `opacity 0.5 的 alpha：${a}`);
  assert.equal(alphaAt(p, 170, 90), 0, 'clip 右半应裁掉');
  const [r, g, b] = pixelAt(p, 145, 90);
  assert.ok(Math.max(r, g, b) - Math.min(r, g, b) <= 2, `saturation=0 应灰度：${r},${g},${b}`);
});

test('z 顺序按 layers 数组；图片 filters 生效', async () => {
  const red = imageLayer({ id: 'red', filters: [{ type: 'color', multiply: [1, 0, 0] }] });
  const blue = imageLayer({ id: 'blue', filters: [{ type: 'color', multiply: [0, 0, 1] }] });
  const scene = baseScene([CHECKER], [blue, red]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const [r, g, b] = pixelAt(await renderRgba(0), 160, 90);
  assert.ok(r > b + 20, `上层红色应覆盖下层蓝色：${r},${g},${b}`);

  const swapped = baseScene([CHECKER], [red, blue]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), swapped);
  const [r2, , b2] = pixelAt(await renderRgba(0), 160, 90);
  assert.ok(b2 > r2 + 20, `交换顺序后蓝色应覆盖红色：${r2},${b2}`);
});

test('attach 跟随目标模型图层的平移', async () => {  const scene = baseScene([CHECKER], [
    {
      id: 'main', kind: 'model', model: 'ds', in: 0, out: 30,
      transform: { x: { keys: [{ t: 0, v: 80 }, { t: 10, v: 240 }] }, y: 90, scale: 0.05 },
      opacity: 0,
    },
    imageLayer({
      space: undefined,
      transform: { x: 0, y: 0, scale: 10, anchorX: 32, anchorY: 32 },
      attach: { layer: 'main', anchor: [0.5, 0.5] },
    }),
  ], [{ id: 'ds', uri: MODEL }]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const a = alphaBBox(await renderRgba(0));
  const b = alphaBBox(await renderRgba(10));
  assert.ok(a.count > 300 && b.count > 300, `图片应可见：${a.count} / ${b.count}`);
  assert.ok(Math.abs((b.x0 - a.x0) - 160) <= 2, `平移应跟随 +160px：${b.x0 - a.x0}`);
  assert.ok(Math.abs((b.y0 - a.y0)) <= 2, `y 不应漂移：${b.y0 - a.y0}`);
});

test('scaleWith=false 不继承目标缩放', async () => {
  const scene = baseScene([CHECKER], [
    {
      id: 'main', kind: 'model', model: 'ds', in: 0, out: 30,
      transform: { x: 160, y: 90, scale: { keys: [{ t: 0, v: 0.5 }, { t: 10, v: 1 }] } },
      opacity: 0,
    },
    imageLayer({
      space: undefined,
      transform: { x: 0, y: 0, scale: 0.5, anchorX: 32, anchorY: 32 },
      attach: { layer: 'main', anchor: [0.5, 0.5], scaleWith: false },
    }),
  ], [{ id: 'ds', uri: MODEL }]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const a = alphaBBox(await renderRgba(0));
  const b = alphaBBox(await renderRgba(10));
  const wa = a.x1 - a.x0 + 1;
  const wb = b.x1 - b.x0 + 1;
  assert.ok(Math.abs(wa - wb) <= 2, `目标缩放变化时图片宽度应不变：${wa} → ${wb}`);
  // 64×64 图片 scale 0.5 → 32px 外框；4px 透明边框缩小到 2px，可见宽度约 28px
  assert.ok(wa >= 26 && wa <= 30, `图片可见宽度应约 28px：${wa}`);
});

test('attach 目标图层不在活跃区间时图片不绘制', async () => {
  const scene = baseScene([CHECKER], [
    { id: 'main', kind: 'model', model: 'ds', in: 0, out: 10, transform: { x: 160, y: 90, scale: 0.05 }, opacity: 0 },
    imageLayer({
      space: undefined,
      transform: { x: 0, y: 0, scale: 10, anchorX: 32, anchorY: 32 },
      attach: { layer: 'main', anchor: [0.5, 0.5] },
    }),
  ], [{ id: 'ds', uri: MODEL }]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  assert.ok(alphaBBox(await renderRgba(5)).count > 300, '目标活跃时图片应可见');
  assert.equal(alphaBBox(await renderRgba(20)).count, 0, '目标不活跃时图片应隐藏');
});

test('图片图层 Worker 路径与同步路径逐位相同', async () => {
  const scene = baseScene([CHECKER], [imageLayer()]);
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
  const sync = await renderSha(0);
  const hw = await openHarness({ workers: 3 });
  try {
    await hw.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
    const r = await hw.page.evaluate(() => window.__charaforge.renderFrame(0, { read: { format: 'sha256' } }));
    assert.equal(r.sha256, sync);
  } finally {
    await hw.close();
  }
});

test('不可见图片图层不影响旧模型图层的像素', async () => {
  const modelOnly = {
    format: 'charaforge.scene', version: 1, meta: { title: 'model-only' }, timebase: { fps: 60 }, duration: 30,
    composition: { width: 320, height: 180, background: '#202830' },
    assets: { models: [{ id: 'ds', uri: MODEL }] },
    layers: [{ id: 'main', kind: 'model', model: 'ds', in: 0, out: 30, transform: { x: 160, y: 300, scale: 0.08 }, physics: { warmup: 10 } }],
  };
  const withInvisible = structuredClone(modelOnly) as typeof modelOnly & { assets: Record<string, unknown>; layers: object[] };
  withInvisible.assets.images = [CHECKER];
  withInvisible.layers.push(imageLayer({ opacity: 0, space: undefined, attach: { layer: 'main', anchor: [0.5, 0.5] } }));
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), modelOnly);
  const a = [await renderSha(0), await renderSha(10), await renderSha(20)];
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), withInvisible);
  const b = [await renderSha(0), await renderSha(10), await renderSha(20)];
  assert.deepEqual(b, a);
});

test('图片声明尺寸与实际解码不一致时拒绝加载', async () => {
  const bad = baseScene([{ ...CHECKER, width: 32, height: 32 }], [imageLayer()]);
  await assert.rejects(
    () => h.page.evaluate((s) => window.__charaforge.loadScene(s), bad),
    /尺寸与 scene 声明不一致/,
  );
});
