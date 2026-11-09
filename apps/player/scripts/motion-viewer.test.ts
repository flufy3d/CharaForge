/*
 * Playwright smoke：打开 motions.html，确认目录 8 个动作 / 23 个表情，
 * 点击 IDLE 后 seek 到不同帧，像素摘要必须变化；再点击“脸红”确认表情预览可用。
 *
 *   pnpm --filter @charaforge/player test:viewer
 *
 * 需要 Playwright Chromium（仓库已有依赖）；首次使用：
 *   pnpm exec playwright install chromium-headless-shell
 */
import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import type { Browser, Page } from 'playwright';
import { launchBrowser, startServer } from './harness-node.ts';

interface ViewerApi {
  readyState: 'booting' | 'ready' | 'error';
  error: string | null;
  catalog: () => { motions: { file: string }[]; expressions: { file: string }[] } | null;
  preview: () => { kind: 'motion' | 'expression'; file: string; duration: number; loop: boolean; withIdle: boolean } | null;
  select: (kind: 'motion' | 'expression', file: string) => Promise<void>;
  seekAndRender: (frame: number) => Promise<void>;
  hashFrame: () => Promise<string>;
  thumbnails: () => Record<string, string> | null;
  setLoop: (loop: boolean) => Promise<void>;
  setWithIdle: (withIdle: boolean) => Promise<void>;
}

let server: Awaited<ReturnType<typeof startServer>>;
let browser: Browser;
let page: Page;
const pageErrors: string[] = [];

before(async () => {
  server = await startServer();
  browser = await launchBrowser();
  page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
  page.on('pageerror', (e) => pageErrors.push(e.message));
  await page.goto(`${server.baseUrl}/motions.html`);
  await page.waitForFunction(
    () => {
      const v = (window as unknown as { __motionViewer?: { readyState?: string } }).__motionViewer;
      return v?.readyState === 'ready' || v?.readyState === 'error';
    },
    null,
    { timeout: 120_000 },
  );
  const failed = await page.evaluate(() => {
    const v = (window as unknown as { __motionViewer?: ViewerApi }).__motionViewer;
    return v?.readyState === 'error' ? v.error ?? '未知错误' : null;
  });
  assert.equal(failed, null, `查看器启动失败：${failed}`);
});

after(async () => {
  await browser?.close().catch(() => {});
  await server?.server.close().catch(() => {});
});

test('目录列出 8 个 motion3 与 23 个 exp3', async () => {
  const counts = await page.evaluate(() => {
    const c = (window as unknown as { __motionViewer: ViewerApi }).__motionViewer.catalog()!;
    return { motions: c.motions.length, expressions: c.expressions.length };
  });
  assert.equal(counts.motions, 8);
  assert.equal(counts.expressions, 23);
  const initial = await page.evaluate(() => (window as unknown as { __motionViewer: ViewerApi }).__motionViewer.preview());
  assert.equal(initial?.file, 'IDLE.motion3.json', '缺省应预览 IDLE');
});

test('列表缩略图与图集已生成', async () => {
  const count = await page.evaluate(() => Object.keys((window as unknown as { __motionViewer: ViewerApi }).__motionViewer.thumbnails() ?? {}).length);
  assert.equal(count, 31);
  assert.ok((await page.locator('.asset-thumb img').count()) >= 8, '动作 tab 应显示缩略图');
  await page.locator('.gallery-btn').click();
  await page.waitForSelector('.gallery-backdrop');
  assert.equal(await page.locator('.gallery-item').count(), 31);
  assert.equal(await page.locator('.gallery-thumb img').count(), 31);
  await page.keyboard.press('Escape');
  await page.waitForSelector('.gallery-backdrop', { state: 'detached' });
});

test('点击 IDLE 后播放/seek，画面随帧变化', async () => {
  await page.locator('[data-kind="motion"][data-asset="IDLE.motion3.json"]').click();
  await page.waitForFunction(
    () => {
      const p = (window as unknown as { __motionViewer?: ViewerApi }).__motionViewer?.preview();
      return p?.file === 'IDLE.motion3.json' && p.kind === 'motion';
    },
    null,
    { timeout: 60_000 },
  );
  const info = await page.evaluate(() => (window as unknown as { __motionViewer: ViewerApi }).__motionViewer.preview());
  assert.equal(info?.duration, 359);

  const a = await page.evaluate(async () => {
    const v = (window as unknown as { __motionViewer: ViewerApi }).__motionViewer;
    await v.seekAndRender(0);
    return v.hashFrame();
  });
  const b = await page.evaluate(async () => {
    const v = (window as unknown as { __motionViewer: ViewerApi }).__motionViewer;
    await v.seekAndRender(90);
    return v.hashFrame();
  });
  assert.equal(typeof a, 'string');
  assert.equal(a.length, 64);
  assert.notEqual(a, b, 'IDLE 在帧 0 与帧 90 的画面应不同');
});

test('点击“脸红”表情能预览', async () => {
  await page.getByRole('tab', { name: /表情/ }).click();
  await page.locator('[data-kind="expression"][data-asset="脸红.exp3.json"]').click();
  await page.waitForFunction(
    () => {
      const p = (window as unknown as { __motionViewer?: ViewerApi }).__motionViewer?.preview();
      return p?.file === '脸红.exp3.json' && p.kind === 'expression';
    },
    null,
    { timeout: 60_000 },
  );
  const info = await page.evaluate(() => (window as unknown as { __motionViewer: ViewerApi }).__motionViewer.preview());
  assert.equal(info?.duration, 120);
  assert.equal(info?.loop, false);
  assert.equal(info?.withIdle, true, '表情缺省叠加 IDLE');
  const hashes = await page.evaluate(async () => {
    const v = (window as unknown as { __motionViewer: ViewerApi }).__motionViewer;
    await v.seekAndRender(0);
    const a = await v.hashFrame();
    await v.seekAndRender(60);
    const b = await v.hashFrame();
    return { a, b };
  });
  assert.notEqual(hashes.a, hashes.b, '叠加 IDLE 后画面应随帧变化');
  const off = await page.evaluate(async () => {
    const v = (window as unknown as { __motionViewer: ViewerApi }).__motionViewer;
    await v.setWithIdle(false);
    return v.preview()?.withIdle;
  });
  assert.equal(off, false);
  assert.equal(await page.locator('video').count(), 0, '查看器不应加载参考视频');
});

test('页面没有未捕获的运行时错误', () => {
  assert.deepEqual(pageErrors, []);
});
