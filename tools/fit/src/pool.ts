/*
 * 多页面并行：N 个独立的无头 Chromium（各自一个渲染进程、一份 runtime），共用一个 Vite 开发服务器。
 *
 * 所有改变状态的调用（setup / setRef / commit / copyFrames / setHidden）广播到每个页面，使它们保持逐位相同的状态；
 * 一代 CMA-ES 候选（evalBatch）按页面切分并行评估。候选只写本帧涉及的逐帧数组，随后 commit 用同一写入集合覆盖，
 * 所以评估过不同候选的页面在 commit 之后重新一致（物理状态缓存也按写入帧失效后确定性重算）。
 * Core update（CPU，约 10 ms/次）是瓶颈，多进程能吃满多核；GPU 由各页面共享。
 */
import type { Browser, Page } from 'playwright';
import { defaultMsaa, launchBrowser, startServer } from '../../../apps/player/scripts/harness-node.ts';
import { installPageHelpers } from './page.ts';

export class Pool {
  readonly pages: Page[];
  private readonly browsers: Browser[];
  private readonly closeServer: () => Promise<void>;
  readonly renderer: string;

  private constructor(pages: Page[], browsers: Browser[], closeServer: () => Promise<void>, renderer: string) {
    this.pages = pages;
    this.browsers = browsers;
    this.closeServer = closeServer;
    this.renderer = renderer;
  }

  static async open(n: number): Promise<Pool> {
    const { server, baseUrl } = await startServer();
    const browsers = await Promise.all(Array.from({ length: n }, () => launchBrowser()));
    const pages: Page[] = [];
    let renderer = '';
    for (const b of browsers) {
      const page = await b.newPage({ viewport: { width: 1280, height: 720 } });
      page.on('pageerror', (e) => process.stderr.write(`[page error] ${e.message}\n`));
      page.on('console', (m) => { if (m.type() === 'error') process.stderr.write(`[page error] ${m.text()}\n`); });
      const msaa = defaultMsaa();
      await page.goto(`${baseUrl}/harness.html${msaa !== undefined ? `?msaa=${msaa}` : ''}`);
      await page.waitForFunction(() => !!window.__charaforge, null, { timeout: 60_000 });
      await page.evaluate(() => window.__charaforge.ready);
      const info = await page.evaluate(() => window.__charaforge.info());
      if (!info.renderer.includes(process.env.CHARAFORGE_EXPECT_GPU ?? 'NVIDIA')) throw new Error(`不是 GPU 渲染：${info.renderer}`);
      renderer = `${info.renderer}（msaa ${info.msaa}）`;
      await page.evaluate(installPageHelpers);
      pages.push(page);
    }
    return new Pool(pages, browsers, async () => { await server.close(); }, renderer);
  }

  get size(): number {
    return this.pages.length;
  }

  /** 广播到所有页面，返回第 0 个页面的结果 */
  async all<A, R>(fn: (a: A) => R | Promise<R>, arg: A): Promise<R> {
    const rs = await Promise.all(this.pages.map((p) => p.evaluate(fn as never, arg as never)));
    return rs[0] as R;
  }

  /** 只在第 0 个页面执行（只读操作：dump、预览渲染、计时） */
  one<A, R>(fn: (a: A) => R | Promise<R>, arg: A): Promise<R> {
    return this.pages[0]!.evaluate(fn as never, arg as never) as Promise<R>;
  }

  /** 候选切分到各页面并行评估，按原顺序返回 */
  async evalBatch<T>(f: number, cands: unknown[], opt: unknown): Promise<T[]> {
    const n = this.pages.length;
    const chunks: unknown[][] = Array.from({ length: n }, () => []);
    cands.forEach((c, i) => chunks[i % n]!.push(c));
    const res = await Promise.all(
      this.pages.map((p, k) =>
        chunks[k]!.length
          ? (p.evaluate(([f, c, o]) => (window as any).__fit.evalBatch(f, c, o), [f, chunks[k], opt] as const) as Promise<T[]>)
          : Promise.resolve([] as T[]),
      ),
    );
    return cands.map((_, i) => res[i % n]![Math.floor(i / n)]!);
  }

  /** 强制各页面做一次完整 GC（CDP；读回缓冲等外部内存回收滞后，不 GC 会把 chrome-headless 涨到 OOM） */
  async gc(): Promise<void> {
    await Promise.all(this.pages.map(async (p) => {
      const s = await p.context().newCDPSession(p);
      try { await s.send('HeapProfiler.collectGarbage'); } finally { await s.detach().catch(() => {}); }
    }));
  }

  /** 各页面计时之和 */
  async timing(): Promise<Record<string, number>> {
    const ts = await Promise.all(this.pages.map((p) => p.evaluate(() => (window as any).__fit.timing())));
    const out: Record<string, number> = {};
    for (const t of ts) for (const k in t) out[k] = (out[k] ?? 0) + t[k];
    return out;
  }

  async close(): Promise<void> {
    await Promise.all(this.browsers.map((b) => b.close().catch(() => {})));
    await this.closeServer().catch(() => {});
  }
}
