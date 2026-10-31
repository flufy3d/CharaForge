/*
 * 播放器帧率基准：无头 Chromium（GPU）打开播放器，播放 N 秒，统计上屏间隔。
 *
 *   node --experimental-strip-types apps/player/scripts/bench.ts [scene=demo:3] [seconds=10] [workers]
 *   pnpm bench -- demo:3 10
 *
 * 输出：平均 fps（上屏次数 / 秒）、上屏间隔（按 vsync 对齐的 rAF 时间戳计）平均 / p50 / p95 / p99 / 最大值、
 * 掉帧：显示帧号跳变次数与被跳过的帧数。
 */
import { launchBrowser, startServer } from './harness-node.ts';

const scene = process.argv[2] ?? 'demo:3';
const seconds = Number(process.argv[3] ?? 10);
const workers = process.argv[4];

const { server, baseUrl } = await startServer();
const browser = await launchBrowser();
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const q = new URLSearchParams({ scene });
  if (workers !== undefined) q.set('workers', workers);
  await page.goto(`${baseUrl}/?${q}`);
  await page.waitForFunction(() => {
    const p = (window as any).__player;
    return p && p.scene && p.rendered >= 0;
  }, null, { timeout: 120_000, polling: 500 });
  const info = await page.evaluate(() => {
    const gl = (window as any).__player.runtime.gl as WebGL2RenderingContext;
    const e = gl.getExtension('WEBGL_debug_renderer_info');
    return { renderer: gl.getParameter(e ? e.UNMASKED_RENDERER_WEBGL : gl.RENDERER), workers: (window as any).__player.workers, cores: navigator.hardwareConcurrency };
  });
  await page.evaluate(() => (window as any).__player.play());
  await page.waitForTimeout(3000); // 预热（JIT、Worker 流水线、显存分配）
  await page.evaluate(() => (window as any).__player.resetStats());
  await page.waitForTimeout(seconds * 1000);
  const s = await page.evaluate(() => {
    const st = (window as any).__player.stats;
    return { intervals: st.intervals as number[], jumps: st.jumps as number, dropped: st.dropped as number, presented: st.presented as number, log: st.log as unknown[] };
  });
  const iv = [...s.intervals].sort((a, b) => a - b);
  const avg = iv.reduce((a, b) => a + b, 0) / iv.length;
  const p = (q: number) => iv[Math.min(iv.length - 1, Math.floor(q * iv.length))]!;
  console.log(JSON.stringify({
    scene, seconds, gpu: info.renderer, workers: info.workers, cores: info.cores,
    fps: +(s.presented / seconds).toFixed(2),
    frameMs: { avg: +avg.toFixed(2), p50: +p(0.5).toFixed(2), p95: +p(0.95).toFixed(2), p99: +p(0.99).toFixed(2), max: +iv[iv.length - 1]!.toFixed(2) },
    jumps: s.jumps,
    droppedFrames: s.dropped,
    dropRate: +(s.dropped / (s.presented + s.dropped)).toFixed(4),
  }));
  if (process.env.BENCH_LOG) console.log(JSON.stringify(s.log.slice(0, Number(process.env.BENCH_LOG))));
} finally {
  await browser.close();
  await server.close();
}
