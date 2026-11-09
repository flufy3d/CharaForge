/*
 * 编辑器增量同步测试（Playwright 无头 Chromium + GPU）。
 *
 *   pnpm --filter @charaforge/runtime test
 *
 * 断言：
 *   1. syncScene(内容相同的场景) 与 loadScene 逐位相同；
 *   2. 修改参数轨 / expression 后，syncScene 与重新 loadScene 逐位相同；
 *   3. rebuildEvaluators 对原地修改生效；Worker 路径也收到新的场景与求值输入；
 *   4. addImageLayer / moveLayer / removeLayer 维护 scene.layers，移除后画面回到原样。
 */
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { openHarness, type Harness } from '../../../apps/player/scripts/harness-node.ts';

const MODEL = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
const MOTION = 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json';
const EXPR = 'models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json';
/** 64×64 透明棋盘图（与 packages/scene/examples/accessory.scene.json 相同）。 */
const CHECKER =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAi0lEQVR4nO3YwQnAIBAEwJSWQlJRqrDT8yuCPgQ5wVnwF8HMa9lHRKTNWyLaE98/Pf33p98HAADAPNkP3H0fAAAAAAAAADBO9gPTi9DpPwAAAAAAAADYAzRBAAAAAAAAwB6gCQIAAAAAAAD2AE0QAAAAAAAAsAdoggAAAAAAAIA9QBMEAGARQESuSgWjFkZObbjgswAAAABJRU5ErkJggg==';
const FRAMES = 48;
const PICK = [0, 6, 13, 24, 37, 47];

interface EditLayer {
  id: string;
  kind: 'model' | 'image';
  model?: string;
  image?: string;
  in: number;
  out: number;
  space?: 'world' | 'screen';
  transform: { x: number; y: number; scale: number; anchorX?: number; anchorY?: number; rotation?: number };
  opacity?: number;
  physics?: { warmup: number };
  motions?: { motion: string; start: number; loop: boolean }[];
  expressions?: { expression: string; start: number; end: number }[];
  params?: { id: string; curve: { keys: { t: number; v: number }[] } }[];
}

interface EditScene {
  format: string;
  version: number;
  meta: { title: string };
  timebase: { fps: number };
  duration: number;
  composition: { width: number; height: number; background: string };
  assets: {
    models: { id: string; uri: string }[];
    motions: { id: string; model: string; uri: string }[];
    expressions: { id: string; model: string; uri: string }[];
    images: { id: string; uri: string; width: number; height: number }[];
  };
  layers: EditLayer[];
}

interface Rt {
  currentScene: EditScene;
  syncScene(scene: unknown): Promise<unknown>;
  rebuildEvaluators(): Promise<void>;
  addImageLayer(asset: { id: string; uri: string; width: number; height: number }, layer: EditLayer): Promise<void>;
  removeLayer(id: string): void;
  moveLayer(id: string, index: number): void;
}

function baseScene(): EditScene {
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: { title: 'edit-sync' },
    timebase: { fps: 60 },
    duration: FRAMES,
    composition: { width: 320, height: 240, background: '#203040' },
    assets: {
      models: [{ id: 'ds', uri: MODEL }],
      motions: [{ id: 'idle', model: 'ds', uri: MOTION }],
      expressions: [{ id: 'blush', model: 'ds', uri: EXPR }],
      images: [{ id: 'checker', uri: CHECKER, width: 64, height: 64 }],
    },
    layers: [
      {
        id: 'main',
        kind: 'model',
        model: 'ds',
        in: 0,
        out: FRAMES,
        transform: { x: 160, y: 260, scale: 0.06 },
        physics: { warmup: 12 },
        motions: [{ motion: 'idle', start: 0, loop: true }],
        expressions: [{ expression: 'blush', start: 8, end: 30 }],
        params: [{ id: 'ParamAngleX', curve: { keys: [{ t: 0, v: -20 }, { t: FRAMES, v: 20 }] } }],
      },
      {
        id: 'badge',
        kind: 'image',
        image: 'checker',
        in: 0,
        out: FRAMES,
        space: 'screen',
        transform: { x: 48, y: 48, scale: 0.5, anchorX: 32, anchorY: 32 },
        opacity: 0.9,
      },
    ],
  };
}

/** 修改求值输入（参数轨 + expression 区间 + 图片位置）。 */
function editedScene(): EditScene {
  const s = baseScene();
  s.layers[0]!.params![0]!.curve.keys = [{ t: 0, v: 10 }, { t: 12, v: 55 }, { t: FRAMES, v: -35 }];
  s.layers[0]!.expressions = [{ expression: 'blush', start: 0, end: FRAMES }];
  s.layers[1]!.transform.x = 250;
  return s;
}

/** 另一组求值输入，用于原地修改 + rebuildEvaluators。 */
function edited2Scene(): EditScene {
  const s = editedScene();
  s.layers[0]!.params![0]!.curve.keys = [{ t: 0, v: -45 }, { t: 30, v: 0 }, { t: FRAMES, v: 45 }];
  s.layers[0]!.expressions = [];
  s.layers[1]!.transform.x = 90;
  return s;
}

let h: Harness;

const render = (harness: Harness, f: number) =>
  harness.page
    .evaluate((frame) => window.__charaforge.renderFrame(frame, { read: { format: 'sha256' } }), f)
    .then((r) => r.sha256!);

const load = (harness: Harness, scene: object) =>
  harness.page.evaluate((s) => window.__charaforge.loadScene(s), scene);

async function hashes(harness: Harness, frames = PICK): Promise<string[]> {
  const out: string[] = [];
  for (const f of frames) out.push(await render(harness, f));
  return out;
}

before(async () => {
  h = await openHarness();
});

after(async () => {
  await h?.close();
});

test('syncScene 对内容相同的场景与 loadScene 逐位相同', async () => {
  await load(h, baseScene());
  const before = await hashes(h);
  await h.page.evaluate((s) => (window as unknown as { __cfRuntime: Rt }).__cfRuntime.syncScene(s), structuredClone(baseScene()));
  const after = await hashes(h);
  assert.deepEqual(after, before);
});

test('syncScene 应用求值输入修改，与重新 loadScene 逐位相同', async () => {
  await load(h, baseScene());
  const original = await hashes(h);

  await h.page.evaluate((s) => (window as unknown as { __cfRuntime: Rt }).__cfRuntime.syncScene(s), editedScene());
  const edited = await hashes(h);

  await load(h, editedScene());
  const reloaded = await hashes(h);

  assert.deepEqual(edited, reloaded);
  assert.notDeepEqual(edited, original);
});

test('rebuildEvaluators 对原地修改生效，且 worker 副本被刷新', async () => {
  // 同步路径的参考结果
  await load(h, editedScene());
  const ref1 = await hashes(h);
  await load(h, edited2Scene());
  const ref2 = await hashes(h);

  const hw = await openHarness({ workers: 3 });
  try {
    assert.equal(hw.info.workers, 3);
    await load(hw, baseScene());
    const original = await hashes(hw);

    await hw.page.evaluate((s) => (window as unknown as { __cfRuntime: Rt }).__cfRuntime.syncScene(s), editedScene());
    const synced = await hashes(hw);
    assert.deepEqual(synced, ref1, 'syncScene 后 Worker 应使用新场景');
    assert.notDeepEqual(synced, original);

    // 原地改 live scene 的求值输入，只重建求值器：Worker 也必须收到新输入
    await hw.page.evaluate(() => {
      const rt = (window as unknown as { __cfRuntime: Rt }).__cfRuntime;
      const main = rt.currentScene.layers.find((l) => l.id === 'main')!;
      main.params![0]!.curve.keys = [{ t: 0, v: -45 }, { t: 30, v: 0 }, { t: 48, v: 45 }];
      main.expressions = [];
      const badge = rt.currentScene.layers.find((l) => l.id === 'badge')!;
      badge.transform.x = 90;
    });
    await hw.page.evaluate(() => (window as unknown as { __cfRuntime: Rt }).__cfRuntime.rebuildEvaluators());
    const rebuilt = await hashes(hw);
    assert.deepEqual(rebuilt, ref2, 'rebuildEvaluators 后 Worker 应使用新求值输入');
  } finally {
    await hw.close();
  }
});

test('addImageLayer / moveLayer / removeLayer 维护 scene.layers', async () => {
  await load(h, baseScene());
  const before = await hashes(h);

  const ids = await h.page.evaluate(async (checker) => {
    const rt = (window as unknown as { __cfRuntime: Rt }).__cfRuntime;
    await rt.addImageLayer(
      { id: 'extra', uri: checker, width: 64, height: 64 },
      {
        id: 'extra',
        kind: 'image',
        image: 'extra',
        in: 0,
        out: 48,
        space: 'screen',
        transform: { x: 280, y: 200, scale: 0.25, anchorX: 32, anchorY: 32 },
      },
    );
    const added = rt.currentScene.layers.map((l) => l.id);
    rt.moveLayer('extra', 0);
    const moved = rt.currentScene.layers.map((l) => l.id);
    rt.removeLayer('extra');
    const removed = rt.currentScene.layers.map((l) => l.id);
    return { added, moved, removed };
  }, CHECKER);

  assert.deepEqual(ids.added, ['main', 'badge', 'extra']);
  assert.deepEqual(ids.moved, ['extra', 'main', 'badge']);
  assert.deepEqual(ids.removed, ['main', 'badge']);
  assert.deepEqual(await hashes(h), before, '移除后应回到原画面');
});
