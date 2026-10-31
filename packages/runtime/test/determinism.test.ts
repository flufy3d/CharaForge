/*
 * 运行时确定性测试（Playwright 无头 Chromium + GPU）。
 *
 *   pnpm --filter @charaforge/runtime test
 *
 * 断言：
 *   0. WebGL renderer 是 NVIDIA（不是 SwiftShader）
 *   1. 同一场景顺序播放两遍（中间重新加载场景），每帧像素 SHA-256 相同
 *   2. 乱序 seek 到任意帧，与顺序播放同一帧的像素相同
 *   3. saveState → 继续渲染 → restoreState → 再渲染，结果相同；一次性覆盖不污染后续帧
 *   4. 单实例接口：保存/恢复物理状态后结果相同
 * 场景包含多图层、motion、expression、pre/postPhysics 轨道、部件不透明度、全部滤镜、clip、blendMode、相机。
 */
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { openHarness, type Harness } from '../../../apps/player/scripts/harness-node.ts';

const MODEL = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
const FRAMES = 90;

const richScene = {
  format: 'charaforge.scene',
  version: 1,
  meta: { title: 'determinism' },
  timebase: { fps: 60 },
  duration: FRAMES,
  composition: { width: 640, height: 360, background: '#202830' },
  assets: {
    models: [{ id: 'ds', uri: MODEL }],
    motions: [{ id: 'idle', model: 'ds', uri: 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json' }],
    expressions: [{ id: 'blush', model: 'ds', uri: 'models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json' }],
  },
  camera: { x: 320, y: 180, zoom: { keys: [{ t: 0, v: 1 }, { t: 90, v: 1.2, ease: 'smooth' }] }, rotation: 2 },
  layers: [
    {
      id: 'back', kind: 'model', model: 'ds', in: 0, out: FRAMES,
      transform: { x: 200, y: 420, scale: 0.08, flipX: true },
      physics: { warmup: 30 },
      filters: [{ type: 'blur', sigma: 3 }, { type: 'color', brightness: -0.1, saturation: 0.5, multiply: [1, 0.9, 1.1] }],
      opacity: 0.8, blendMode: 'normal',
      params: [{ id: 'ParamAngleX', curve: { keys: [{ t: 0, v: -30 }, { t: 45, v: 30, ease: 'smooth' }, { t: 90, v: -30 }] } }],
    },
    {
      id: 'main', kind: 'model', model: 'ds', in: 5, out: FRAMES,
      transform: { x: 380, y: 400, scale: { keys: [{ t: 0, v: 0.09 }, { t: 90, v: 0.1 }] }, rotation: { keys: [{ t: 0, v: -5 }, { t: 90, v: 5 }] } },
      physics: { warmup: 20 },
      clip: { x: 200, y: 20, width: 380, height: 320, radius: 30, feather: 12 },
      filters: [{ type: 'motionBlur', angle: 30, length: { keys: [{ t: 0, v: 0 }, { t: 90, v: 12 }] } }],
      motions: [{ motion: 'idle', start: 0, loop: true, fadeIn: 0.2 }],
      expressions: [{ expression: 'blush', start: 20, end: 70, fadeIn: 0.3, fadeOut: 0.3 }],
      params: [
        { id: 'ParamAngleY', curve: { keys: [{ t: 0, v: 0 }, { t: 30, v: 25 }, { t: 60, v: -25 }] } },
        { id: 'ParamBodyAngleX', curve: { keys: [{ t: 0, v: 10 }, { t: 90, v: -10 }] }, mode: 'add', weight: 0.5 },
        { id: 'key11', curve: { keys: [{ t: 0, v: 0, ease: 'step' }, { t: 50, v: 1 }] }, stage: 'postPhysics' },
      ],
    },
    {
      id: 'ui', kind: 'model', model: 'ds', in: 0, out: 60, space: 'screen',
      transform: { x: 560, y: 300, scale: 0.04 },
      blendMode: 'screen', physics: { enabled: false },
    },
  ],
};

let h: Harness;

const render = (f: number, overrides?: object) =>
  h.page.evaluate(([f, ov]) => window.__charaforge.renderFrame(f as number, { overrides: ov as never, read: { format: 'sha256' } }), [f, overrides] as const)
    .then((r) => r.sha256!);

before(async () => {
  h = await openHarness();
  console.log(`[gpu] ${h.info.renderer}  (msaa ${h.info.msaa})`);
});
after(async () => h?.close());

test('使用 GPU 渲染（不是 SwiftShader）', () => {
  assert.match(h.info.renderer, /NVIDIA/);
  assert.doesNotMatch(h.info.renderer, /SwiftShader/i);
});

let sequential: string[] = [];

test('顺序播放两遍逐位相同', async () => {
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
  const a: string[] = [];
  for (let f = 0; f < FRAMES; f++) a.push(await render(f));
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
  const b: string[] = [];
  for (let f = 0; f < FRAMES; f++) b.push(await render(f));
  assert.deepEqual(b, a);
  assert.ok(new Set(a).size > FRAMES / 2, '画面应随帧变化');
  sequential = a;
});

test('乱序 seek 与顺序播放逐位相同', async () => {
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
  for (const f of [77, 13, 89, 45, 46, 0, 30, 29, 31, 88, 5, 4, 60, 59]) {
    assert.equal(await render(f), sequential[f], `帧 ${f}`);
  }
});

test('saveState / restoreState 与一次性覆盖', async () => {
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
  await render(40);
  const s = await h.page.evaluate(() => window.__charaforge.saveState());
  const tail1: string[] = [];
  for (let f = 41; f <= 50; f++) tail1.push(await render(f));
  await h.page.evaluate((s) => window.__charaforge.restoreState(s), s);
  const tail2: string[] = [];
  for (let f = 41; f <= 50; f++) tail2.push(await render(f));
  assert.deepEqual(tail2, tail1);
  assert.deepEqual(tail1, sequential.slice(41, 51));

  // 同一前态试多组覆盖
  await h.page.evaluate((s) => window.__charaforge.restoreState(s), s);
  const ovA = { main: { pre: { ParamAngleX: 30 }, post: { ParamEyeLOpen: 0 } } };
  const ovB = { main: { pre: { ParamAngleX: -30 } } };
  const a1 = await render(41, ovA);
  await h.page.evaluate((s) => window.__charaforge.restoreState(s), s);
  const b1 = await render(41, ovB);
  await h.page.evaluate((s) => window.__charaforge.restoreState(s), s);
  const a2 = await render(41, ovA);
  assert.equal(a2, a1);
  assert.notEqual(a1, b1);
  assert.notEqual(a1, sequential[41]);
  // 覆盖只影响那一帧：随后（同一帧无覆盖 / 下一帧）回到规范结果
  assert.equal(await render(41), sequential[41]);
  await render(42, ovB);
  assert.equal(await render(43), sequential[43]);
  await h.page.evaluate((s) => window.__charaforge.dropState(s), s);
});

test('单实例：显式参数 + 物理状态保存/恢复', async () => {
  const id = await h.page.evaluate((m) => window.__charaforge.createInstance(m), MODEL);
  const info = await h.page.evaluate((id) => window.__charaforge.instanceInfo(id), id);
  assert.ok(info.parameters.length > 400);
  const shot = (angle: number) =>
    h.page.evaluate(([id, angle, W, H]) => {
      const cf = window.__charaforge;
      cf.stepInstance(id, { pre: { ParamAngleX: angle, ParamBodyAngleX: angle / 3 }, physics: true });
      return cf.renderInstance(id, { transform: { x: 160, y: 200, scale: 0.05 }, width: 320, height: 240, format: 'sha256' });
    }, [id, angle, info.canvas.width, info.canvas.height] as const).then((r) => r.sha256!);
  for (let i = 0; i < 10; i++) await shot(i * 3);
  const s = await h.page.evaluate((id) => window.__charaforge.saveInstanceState(id), id);
  const x1 = [await shot(30), await shot(-10), await shot(5)];
  await h.page.evaluate(([id, s]) => window.__charaforge.restoreInstanceState(id, s), [id, s] as const);
  const x2 = [await shot(30), await shot(-10), await shot(5)];
  assert.deepEqual(x2, x1);
  await h.page.evaluate((id) => window.__charaforge.destroyInstance(id), id);
});

test('frameHold：段内 f 与 g 像素逐位相同，段外与无保持一致', async () => {
  // main 层在 [20, 50) 每 2 帧保持、[60, 75) 每 3 帧保持；其余图层不保持
  const held = structuredClone(richScene) as typeof richScene & { layers: Record<string, unknown>[] };
  held.layers = held.layers.filter((l) => l.id === 'main');
  (held.layers[0] as Record<string, unknown>).frameHold = [{ start: 20, end: 50, every: 2 }, { start: 60, end: 75, every: 3 }];
  const plain = structuredClone(held) as typeof held;
  delete (plain.layers[0] as Record<string, unknown>).frameHold;

  await h.page.evaluate((s) => window.__charaforge.loadScene(s), plain);
  const ref: string[] = [];
  for (let f = 0; f < FRAMES; f++) ref.push(await render(f));

  await h.page.evaluate((s) => window.__charaforge.loadScene(s), held);
  const seq: string[] = [];
  for (let f = 0; f < FRAMES; f++) seq.push(await render(f));
  const g = (f: number) => (f >= 20 && f < 50 ? 20 + Math.floor((f - 20) / 2) * 2 : f >= 60 && f < 75 ? 60 + Math.floor((f - 60) / 3) * 3 : f);
  for (let f = 0; f < FRAMES; f++) {
    assert.equal(seq[f], seq[g(f)], `保持段内帧 ${f} 应等于帧 ${g(f)}`);
    // 物理逐帧推进：画面 = 无保持场景在 g 的画面
    assert.equal(seq[f], ref[g(f)], `帧 ${f} 应等于无保持场景的帧 ${g(f)}`);
  }
  assert.notEqual(seq[21], ref[21]);
  // seek 路径
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), held);
  for (const f of [73, 21, 49, 62, 5, 61]) assert.equal(await render(f), seq[f], `seek 帧 ${f}`);
});

test('Worker 路径与同步路径逐位相同（顺序 + seek + 覆盖）', async () => {
  const hw = await openHarness({ workers: 3 });
  try {
    assert.equal(hw.info.workers, 3);
    const renderW = (f: number, ov?: object) =>
      hw.page.evaluate(([f, ov]) => window.__charaforge.renderFrame(f as number, { overrides: ov as never, read: { format: 'sha256' } }), [f, ov] as const).then((r) => r.sha256!);
    await hw.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
    for (let f = 0; f < FRAMES; f++) assert.equal(await renderW(f), sequential[f], `顺序帧 ${f}`);
    await hw.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
    for (const f of [77, 13, 89, 0, 45, 46]) assert.equal(await renderW(f), sequential[f], `seek 帧 ${f}`);
    // 覆盖：与同步路径的同一覆盖结果相同
    const ov = { main: { pre: { ParamAngleX: 30 }, post: { ParamEyeLOpen: 0 } } };
    await h.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
    const syncOv = await render(41, ov);
    assert.equal(await renderW(41, ov), syncOv);
    assert.equal(await renderW(42), sequential[42]);
  } finally {
    await hw.close();
  }
});

test('composition.antialias：false 时不做 MSAA，等价于 MSAA=0 的渲染', async () => {
  const noAA = structuredClone(richScene) as typeof richScene & { composition: Record<string, unknown> };
  noAA.composition.antialias = false;
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), noAA);
  const a: string[] = [];
  for (const f of [10, 40, 80]) a.push(await render(f));
  for (const [i, f] of [10, 40, 80].entries()) assert.notEqual(a[i], sequential[f], `帧 ${f} 应与开 MSAA 时不同`);
  // 同一场景（antialias 缺省 true）在 msaa=0 的 harness 上渲染，应与 antialias=false 逐位相同
  const h0 = await openHarness({ msaa: 0 });
  try {
    await h0.page.evaluate((s) => window.__charaforge.loadScene(s), richScene);
    for (const [i, f] of [10, 40, 80].entries()) {
      const r = await h0.page.evaluate((f) => window.__charaforge.renderFrame(f, { read: { format: 'sha256' } }), f);
      assert.equal(r.sha256, a[i], `帧 ${f}`);
    }
  } finally {
    await h0.close();
  }
  // 自身确定（顺序 vs seek）
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), noAA);
  assert.equal(await render(80), a[2]);
});

test('physics.stabilize：预热前置于平衡态，结果不同于未稳定、且确定', async () => {
  const mk = (stabilize: boolean) => {
    const s = structuredClone(richScene) as typeof richScene & { layers: Record<string, unknown>[] };
    s.layers = s.layers.filter((l) => l.id === 'main');
    (s.layers[0] as { physics: Record<string, unknown> }).physics = { warmup: 0, stabilize };
    return s;
  };
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), mk(false));
  const plain = [await render(5), await render(6)];
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), mk(true));
  const stab = [await render(5), await render(6), await render(30)];
  assert.notEqual(stab[0], plain[0], '第一帧应受 stabilize 影响');
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), mk(true));
  assert.equal(await render(30), stab[2], 'seek 与顺序一致');
  assert.equal(await render(5), stab[0]);
  // 物理输出参数在 stabilize 后、第一帧时应已接近平衡（与长预热的结果接近）
  const params = (s: object) => h.page.evaluate(async (s) => { await window.__charaforge.loadScene(s); await window.__charaforge.renderFrame(5); return window.__charaforge.layerParams('main'); }, s);
  const pS = await params(mk(true));
  const pLong = await params({ ...mk(false), layers: [{ ...mk(false).layers[0], physics: { warmup: 600 } }] });
  const p0 = await params(mk(false));
  const dist = (a: Record<string, number>, b: Record<string, number>) => Object.keys(a).reduce((s, k) => s + Math.abs(a[k]! - b[k]!), 0);
  assert.ok(dist(pS, pLong) < dist(p0, pLong), `stabilize 应更接近长预热：${dist(pS, pLong)} vs ${dist(p0, pLong)}`);
});

test('不可见图层（opacity=0 / 出画 / clip 面积 0）跳过 Core update，但物理链连续', async () => {
  const base = structuredClone(richScene) as typeof richScene & { layers: Record<string, unknown>[] };
  base.layers = base.layers.filter((l) => l.id === 'main');
  const hidden = structuredClone(base);
  const L = hidden.layers[0] as Record<string, unknown>;
  // [0,20) opacity 0；[20,40) 移出画面；[40,50) clip 面积为 0；之后正常
  L.opacity = { keys: [{ t: 0, v: 0, ease: 'step' }, { t: 20, v: 1 }] };
  L.transform = { ...(L.transform as object), x: { keys: [{ t: 0, v: 380, ease: 'step' }, { t: 20, v: 5000, ease: 'step' }, { t: 40, v: 380 }] } };
  L.clip = { x: 200, y: 20, width: { keys: [{ t: 0, v: 380, ease: 'step' }, { t: 40, v: 0, ease: 'step' }, { t: 50, v: 380 }] }, height: 320, radius: 30, feather: 12 };
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), base);
  const ref: string[] = [];
  for (let f = 0; f < FRAMES; f++) ref.push(await render(f));
  await h.page.evaluate((s) => window.__charaforge.loadScene(s), hidden);
  const updates: number[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const sha = await render(f);
    updates.push(await h.page.evaluate(() => (window as unknown as { __cfRuntime: { lastUpdates: number } }).__cfRuntime.lastUpdates));
    if (f >= 50) assert.equal(sha, ref[f], `帧 ${f}：隐藏期间物理照常推进，重新可见后与一直可见的场景相同`);
  }
  for (let f = 5; f < 50; f++) assert.equal(updates[f], 0, `帧 ${f} 不应做 Core update`);
  for (let f = 50; f < FRAMES; f++) assert.equal(updates[f], 1, `帧 ${f} 应做 1 次 Core update`);
});
