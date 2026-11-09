/*
 * Playwright smoke：场景 / 动作编辑器。
 *
 *   pnpm --filter @charaforge/player test:editor
 *
 * 覆盖：
 *   - 打开全片场景（47 图层、43 镜头标记、1 条音频），时间线渲染；
 *   - 编辑器预览像素与 harness 同一帧逐位一致；
 *   - 选中图层、关键帧添加 / 撤销；
 *   - AI window.__editor 命令：生成图片图层 + 一键动画 + 删除；
 *   - 动作 / 表情目录添加到图层并撤销；
 *   - 追加音乐轨（多轨）；
 *   - 保存到临时场景文件（服务端写盘）；
 *   - 导出 4 帧 WebM + 音频。
 *
 * 需要模型与音频已通过 pnpm run setup 安装。
 */
import assert from 'node:assert/strict';
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { parseScene } from '@charaforge/scene';
import type { Browser, Page } from 'playwright';
import { launchBrowser, REPO, startServer } from './harness-node.ts';

const TEMP_SCENE = 'packages/scene/examples/editor-smoke.scene.json';
const TEMP_SCENE_ABS = join(REPO, TEMP_SCENE);

declare global {
  interface Window {
    __ed: () => EditorApi;
  }
}

interface EditorApi {
  ready: Promise<void>;
  state(): {
    phase: string;
    frame: number;
    playing: boolean;
    dirty: boolean;
    catalog: { motions: { uri: string }[]; expressions: { uri: string }[] } | null;
    audioDurations: Record<string, number>;
    selection: unknown;
    notices: { text: string; kind: string }[];
  };
  summary(): { layers: number; markers: number; audio: number; duration: number; fps: number };
  scene(): {
    layers: { id: string; kind: string; motions?: unknown[]; expressions?: unknown[]; transform: Record<string, unknown> }[];
    audio?: unknown[];
    assets: { images?: unknown[] };
  };
  apply(commands: unknown, opts?: unknown): Promise<{ changes: unknown[]; effects: string[] }>;
  undo(): Promise<void>;
  redo(): Promise<void>;
  seek(frame: number): void;
  play(): Promise<void>;
  pause(): void;
  select(selection: unknown): void;
  save(path?: string): Promise<void>;
  exportVideo(opts?: unknown): Promise<Record<string, unknown>>;
  addMotion(uri: string, name?: string): Promise<void>;
  addExpression(uri: string): Promise<void>;
  importImage(dataUri: string, opts?: unknown): Promise<void>;
}

let server: Awaited<ReturnType<typeof startServer>>;
let browser: Browser;
let page: Page;
let harness: Page;
const pageErrors: string[] = [];

before(async () => {
  server = await startServer();
  browser = await launchBrowser();
  page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await page.addInitScript(() => {
    // 测试里用 __ed() 拿到页面上的 window.__editor，避免每个 evaluate 都写长类型断言。
    (window as unknown as { __ed: () => unknown }).__ed = () => (window as unknown as { __editor: unknown }).__editor;
  });
  await page.goto(`${server.baseUrl}/editor.html?scene=scenes/deepseek-chan/scene.json&workers=0&frame=2400`, {
    waitUntil: 'domcontentloaded',
    timeout: 120_000,
  });
  await page.waitForFunction(() => !!(window as unknown as { __editor?: unknown }).__editor, null, { timeout: 120_000 });
  await page.evaluate(() => window.__ed().ready);
  // 等待动作目录与音频时长异步加载
  await page.waitForFunction(
    () => {
      const s = window.__ed().state();
      return !!s.catalog && Object.keys(s.audioDurations).length > 0;
    },
    null,
    { timeout: 120_000 },
  );

  harness = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  await harness.goto(`${server.baseUrl}/harness.html?workers=0`, { waitUntil: 'domcontentloaded', timeout: 120_000 });
  await harness.waitForFunction(() => !!(window as unknown as { __charaforge?: unknown }).__charaforge, null, { timeout: 120_000 });
  await harness.evaluate(() => (window as unknown as { __charaforge: { ready: Promise<void> } }).__charaforge.ready);
});

after(async () => {
  if (existsSync(TEMP_SCENE_ABS)) rmSync(TEMP_SCENE_ABS, { force: true });
  if (existsSync(`${TEMP_SCENE_ABS}.bak.json`)) rmSync(`${TEMP_SCENE_ABS}.bak.json`, { force: true });
  // 测试通过编辑器接口上传的 smoke-bg 图片（assets/ 本身不进 git）。
  const editorAssets = join(REPO, 'assets', 'editor');
  if (existsSync(editorAssets)) {
    for (const file of readdirSync(editorAssets)) {
      if (/^smoke-bg(?:-\d+)?\.png$/i.test(file)) rmSync(join(editorAssets, file), { force: true });
    }
  }
  await browser?.close().catch(() => {});
  await server?.server.close().catch(() => {});
});

test('全片场景加载：47 图层 / 43 标记 / 1 音频，时间线渲染', async () => {
  const summary = await page.evaluate(() => window.__ed().summary());
  assert.equal(summary.layers, 47);
  assert.equal(summary.markers, 43);
  assert.equal(summary.audio, 1);
  assert.equal(summary.fps, 60);
  assert.equal(summary.duration, 3868);
  assert.equal(await page.locator('[data-testid^="track-layer-"]').count(), 47);
  assert.equal(await page.locator('[data-testid="timeline"]').count(), 1);
  assert.ok((await page.locator('[data-keyframe]').count()) > 500, '时间线应显示关键帧');
});

test('编辑器预览与 harness 同一帧逐位一致', async () => {
  const editorHash = await page.evaluate(async () => {
    const rt = (window.__ed() as unknown as { store: { engine: { runtime: { readPixels(): { data: Uint8Array } } } } }).store.engine.runtime;
    const px = rt.readPixels();
    const digest = await crypto.subtle.digest('SHA-256', px.data as unknown as ArrayBuffer);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  });
  const harnessHash = await harness.evaluate(async () => {
    const api = (window as unknown as { __charaforge: { loadScene(s: unknown): Promise<unknown>; renderFrame(f: number, o: unknown): Promise<{ sha256?: string }> } }).__charaforge;
    const scene = await (await fetch('scenes/deepseek-chan/scene.json')).json();
    await api.loadScene(scene);
    const r = await api.renderFrame(2400, { read: { format: 'sha256' } });
    return r.sha256;
  });
  assert.equal(editorHash.length, 64);
  assert.equal(editorHash, harnessHash, '编辑器与玩家 harness 的预览必须逐位一致');
});

test('选中图层、添加关键帧并撤销', async () => {
  await page.locator('[data-testid="track-layer-s24.main"] .tl-layer-head').click();
  assert.deepEqual(await page.evaluate(() => window.__ed().state().selection), { kind: 'layer', id: 's24.main' });

  const before = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 's24.main')!;
    const x = layer.transform.x as { keys?: { t: number; v: number }[] } | number;
    return typeof x === 'object' && x.keys ? x.keys.find((k) => k.t === 2400)?.v : x;
  });
  const result = await page.evaluate(() =>
    window.__ed().apply({ op: 'addKeyframe', target: { kind: 'layer', id: 's24.main', path: 'transform.x' }, t: 2400, v: 700, ease: 'smooth' }),
  );
  assert.ok(result.changes.length > 0);
  assert.ok(result.effects.includes('render'));
  const after = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 's24.main')!;
    const x = layer.transform.x as { keys: { t: number; v: number }[] };
    return x.keys.find((k) => k.t === 2400)?.v;
  });
  assert.equal(after, 700);

  await page.evaluate(() => window.__ed().undo());
  const undone = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 's24.main')!;
    const x = layer.transform.x as { keys: { t: number; v: number }[] };
    return x.keys.find((k) => k.t === 2400)?.v;
  });
  assert.equal(undone, before);
});

test('AI 命令：生成元素 + 一键动画 + 删除', async () => {
  const png = await page.evaluate(() => {
    const c = document.createElement('canvas');
    c.width = 96;
    c.height = 96;
    const g = c.getContext('2d')!;
    const grad = g.createLinearGradient(0, 0, 96, 96);
    grad.addColorStop(0, '#ff8ad8');
    grad.addColorStop(1, '#6f8dff');
    g.fillStyle = grad;
    g.fillRect(0, 0, 96, 96);
    return c.toDataURL('image/png');
  });
  const added = await page.evaluate(async (uri) => {
    await window.__ed().importImage(uri, { name: 'smoke-bg', asBackground: true });
    const s = window.__ed().scene();
    return { layers: s.layers.length, images: s.assets.images?.length ?? 0, first: s.layers[0]?.id };
  }, png);
  assert.equal(added.layers, 48);
  assert.equal(added.images, 1);
  assert.equal(added.first, 'smoke-bg');

  const animated = await page.evaluate(() => window.__ed().apply({ op: 'animate', id: 'smoke-bg', preset: 'float' }));
  assert.ok(animated.changes.length > 0);
  const y = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 'smoke-bg')!;
    return typeof layer.transform.y === 'object' && layer.transform.y && 'keys' in layer.transform.y;
  });
  assert.equal(y, true);

  await page.evaluate(() => window.__ed().apply({ op: 'removeLayer', id: 'smoke-bg' }));
  const removed = await page.evaluate(() => {
    const s = window.__ed().scene();
    return { layers: s.layers.length, images: s.assets.images?.length ?? 0 };
  });
  assert.equal(removed.layers, 47);
  assert.equal(removed.images, 0);
});

test('动作 / 表情添加到选中图层并可撤销', async () => {
  await page.evaluate(() => window.__ed().select({ kind: 'layer', id: 's24.main' }));
  const motion = await page.evaluate(() => window.__ed().state().catalog!.motions[0]!.uri);
  const expression = await page.evaluate(() => window.__ed().state().catalog!.expressions[0]!.uri);

  await page.evaluate((uri) => window.__ed().addMotion(uri), motion);
  let counts = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 's24.main')!;
    return { motions: layer.motions?.length ?? 0, expressions: layer.expressions?.length ?? 0 };
  });
  assert.equal(counts.motions, 1);

  await page.evaluate((uri) => window.__ed().addExpression(uri), expression);
  counts = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 's24.main')!;
    return { motions: layer.motions?.length ?? 0, expressions: layer.expressions?.length ?? 0 };
  });
  assert.equal(counts.expressions, 1);

  await page.evaluate(() => window.__ed().undo());
  await page.evaluate(() => window.__ed().undo());
  counts = await page.evaluate(() => {
    const layer = window.__ed().scene().layers.find((l) => l.id === 's24.main')!;
    return { motions: layer.motions?.length ?? 0, expressions: layer.expressions?.length ?? 0 };
  });
  assert.equal(counts.motions, 0);
  assert.equal(counts.expressions, 0);
});

test('追加音乐轨：多轨时间线', async () => {
  await page.evaluate(() => window.__ed().apply({ op: 'addAudio', uri: 'video/DeepSeek-chan.wav', start: 600, gain: 0.5 }));
  const summary = await page.evaluate(() => window.__ed().summary());
  assert.equal(summary.audio, 2);
  assert.equal(await page.locator('[data-testid="audio-0"]').count(), 1);
  assert.equal(await page.locator('[data-testid="audio-1"]').count(), 1);
  await page.evaluate(() => window.__ed().undo());
  assert.equal((await page.evaluate(() => window.__ed().summary())).audio, 1);
});

test('保存到临时场景文件并被 parseScene 接受', async () => {
  await page.evaluate(async (path) => window.__ed().save(path), TEMP_SCENE);
  const notices = await page.evaluate(() => window.__ed().state().notices);
  assert.ok(existsSync(TEMP_SCENE_ABS), `保存接口应写出场景文件；notices=${JSON.stringify(notices)}`);
  const saved = parseScene(JSON.parse(readFileSync(TEMP_SCENE_ABS, 'utf8')));
  assert.equal(saved.duration, 3868);
  assert.equal(saved.layers.length, 47);
  // 大场景应写紧凑 JSON，而不是展开成十几 MB
  assert.ok(statSync(TEMP_SCENE_ABS).size < 8 * 1024 * 1024);
});

test('导出 4 帧 WebM（含音频）', async () => {
  const result = await page.evaluate(() =>
    window.__ed().exportVideo({ from: 2400, to: 2404, format: 'webm', quality: 'low', filename: 'editor-smoke.webm' }),
  );
  assert.equal(result.extension, 'webm');
  assert.equal(result.frames, 4);
  assert.equal(result.hasAudio, true);
  assert.ok(Number(result.bytes) > 0, '导出的视频字节数应大于 0');
  assert.match(String(result.mimeType), /video\/webm/);
});

test('页面无未捕获错误', () => {
  assert.deepEqual(pageErrors, []);
});
