/*
 * 播放器帧率与音画同步基准：无头 Chromium（GPU）打开播放器，播放 N 秒，统计上屏间隔与画面相对音频时钟的滞后。
 *
 *   node --experimental-strip-types apps/player/scripts/bench.ts [scene=demo:3] [seconds=10] [workers] [from=0]
 *   pnpm bench demo:3 10            （pnpm 透传的 `--` 会被忽略，`pnpm bench -- demo:3 10` 也可以）
 *   pnpm bench scenes/deepseek-chan/scene.json 60     全片（带原音频）
 *   BENCH_QUERY='prewarm=0' pnpm bench …             追加播放器 URL 参数；BENCH_LOG=N 额外输出前 N 次上屏的明细
 * 缺省等播放器的后台预热做完（prewarmSec）再点播放；BENCH_PLAY_NOW=1 就绪后立即播放。
 *
 * 播放用真实点击「播放」按钮开始（用户手势），这样带音频的场景里 AudioContext 能真正启动，时钟走音频时间。
 * 输出：平均 fps（上屏次数 / 秒）、上屏间隔（按 vsync 对齐的 rAF 时间戳计）平均 / p50 / p95 / p99 / 最大值、
 * 掉帧：显示帧号跳变次数与被跳过的帧数；
 * 音画同步（avLagFrames）：每次上屏时「音频时钟所在的帧 − 上屏的帧」，单位帧（> 0 = 画面落后于声音），
 * 给出均值 / p50 / p95 / 最大，以及前 10% 与后 10% 样本的均值（看长时间播放是否漂移）；
 * frameRate：这段时间里帧号前进的速度（帧 / 墙钟秒），应为 60。
 */
import { launchBrowser, startServer } from './harness-node.ts';

const argv = process.argv.slice(2).filter((a) => a !== '--');
const scene = argv[0] ?? 'demo:3';
const seconds = Number(argv[1] ?? 10);
const workers = argv[2];
const from = argv[3];

const { server, baseUrl } = await startServer();
const browser = await launchBrowser();
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const q = new URLSearchParams({ scene });
  if (workers !== undefined && workers !== '') q.set('workers', workers);
  if (from) q.set('frame', from);
  // 其他播放器 URL 参数，如 BENCH_QUERY='prewarm=0'
  for (const [k, v] of new URLSearchParams(process.env.BENCH_QUERY ?? '')) q.set(k, v);
  await page.goto(`${baseUrl}/?${q}`);
  await page.waitForFunction(() => (window as any).__player?.status?.ready, null, { timeout: 180_000, polling: 200 });
  const info = await page.evaluate(() => {
    const gl = (window as any).__player.runtime.gl as WebGL2RenderingContext;
    const e = gl.getExtension('WEBGL_debug_renderer_info');
    return { renderer: gl.getParameter(e ? e.UNMASKED_RENDERER_WEBGL : gl.RENDERER), workers: (window as any).__player.workers, cores: navigator.hardwareConcurrency };
  });
  const tReady = Date.now();
  // 等后台预热做完再播放（播放器暂停时预热各图层的 in 帧）；BENCH_PLAY_NOW=1 时就绪后立即播放
  if (!process.env.BENCH_PLAY_NOW) {
    await page.waitForFunction(() => {
      const s = (window as any).__player.status;
      return s.prewarm.done >= s.prewarm.total;
    }, null, { timeout: 180_000, polling: 200 });
  }
  const prewarmSec = +((Date.now() - tReady) / 1000).toFixed(1);
  await page.click('button.primary'); // 用户手势：AudioContext 才能 resume
  await page.waitForFunction(() => (window as any).__player.clock.playing, null, { timeout: 10_000 });
  await page.waitForTimeout(3000); // 预热（JIT、Worker 流水线、显存分配）
  const t0 = await page.evaluate(() => {
    const p = (window as any).__player;
    p.resetStats();
    return { f: p.clock.now() * p.clock.fps, t: performance.now() };
  });
  await page.waitForTimeout(seconds * 1000);
  const s = await page.evaluate(() => {
    const p = (window as any).__player;
    const st = p.stats;
    const ctx = (p.clock as any).ctx as AudioContext | null;
    return {
      intervals: st.intervals as number[], jumps: st.jumps as number, dropped: st.dropped as number, presented: st.presented as number,
      log: st.log as unknown[], avLag: st.avLag as number[],
      f: p.clock.now() * p.clock.fps, t: performance.now(), playing: p.clock.playing as boolean,
      audio: { hasAudio: p.clock.hasAudio as boolean, state: ctx?.state ?? null, sampleRate: ctx?.sampleRate ?? null, outputLatency: ctx?.outputLatency ?? null, baseLatency: ctx?.baseLatency ?? null },
    };
  });
  if (!s.playing) console.error('[bench] 测量结束前播放已停止（到片尾了？），结果只覆盖实际播放部分');
  const pct = (arr: number[], q: number) => {
    const a = [...arr].sort((x, y) => x - y);
    return a[Math.min(a.length - 1, Math.floor(q * a.length))]!;
  };
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  const iv = s.intervals;
  const lag = s.avLag;
  const k = Math.max(1, Math.floor(lag.length / 10));
  console.log(JSON.stringify({
    scene, seconds, gpu: info.renderer, workers: info.workers, cores: info.cores, prewarmSec,
    fps: +(s.presented / seconds).toFixed(2),
    frameMs: { avg: +mean(iv).toFixed(2), p50: +pct(iv, 0.5).toFixed(2), p95: +pct(iv, 0.95).toFixed(2), p99: +pct(iv, 0.99).toFixed(2), max: +Math.max(...iv).toFixed(2) },
    jumps: s.jumps,
    droppedFrames: s.dropped,
    dropRate: +(s.dropped / (s.presented + s.dropped)).toFixed(4),
    frameRate: +((s.f - t0.f) / ((s.t - t0.t) / 1000)).toFixed(3),
    audio: s.audio,
    avLagFrames: lag.length
      ? { mean: +mean(lag).toFixed(3), p50: +pct(lag, 0.5).toFixed(3), p95: +pct(lag, 0.95).toFixed(3), max: +Math.max(...lag).toFixed(3), min: +Math.min(...lag).toFixed(3), first10pct: +mean(lag.slice(0, k)).toFixed(3), last10pct: +mean(lag.slice(-k)).toFixed(3) }
      : null,
  }));
  if (process.env.BENCH_LOG) console.log(JSON.stringify(s.log.slice(0, Number(process.env.BENCH_LOG))));
} finally {
  await browser.close();
  await server.close();
}
