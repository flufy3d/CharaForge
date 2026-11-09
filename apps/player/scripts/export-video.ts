/*
 * 离线 MV 导出 CLI：启动 Vite + 无头 Chromium，打开 export.html，逐帧导出带音轨的视频。
 *
 *   node --experimental-strip-types --disable-warning=ExperimentalWarning \
 *     scripts/export-video.ts --scene packages/scene/examples/minimal.scene.json \
 *     --out .data/export/minimal.mp4 --from 0 --to 10
 *
 * 参数：
 *   --scene <path>        场景 JSON（相对当前目录或绝对路径，Node 端读取后传给页面）
 *   --out <file>          输出文件；缺省 .data/export/<scene>-<from>-<to>.<mp4|webm>
 *   --from <frame>        起始帧（含，缺省 0）
 *   --to <frame>          结束帧（不含，缺省场景 duration）
 *   --format auto|mp4|webm  缺省 auto（MP4 不可用时自动回退 WebM）
 *   --quality high|medium|low  缺省 high
 *   --width <px> --height <px> 缺省场景合成分辨率
 *   --background <css>    缺省场景 background ?? '#000'
 *   --workers <n>         求值 Worker 数（缺省 0；2 更快，但同步路径最稳）
 *   --msaa <n>            覆盖 MSAA（缺省按平台，见 harness-node.ts）
 */
import { closeSync, mkdirSync, openSync, readFileSync, writeSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { ExportPageResult } from '../src/export-page.ts';
import { defaultMsaa, launchBrowser, startServer } from './harness-node.ts';

const CHUNK = 4 * 1024 * 1024;

interface SceneFile {
  duration?: number;
  timebase?: { fps?: number };
  meta?: { title?: string };
}

function usage(): never {
  console.error(
    '用法：export-video --scene <path> [--out <file.mp4|webm>] [--from <f>] [--to <f>]\n' +
      '                    [--format auto|mp4|webm] [--quality high|medium|low]\n' +
      '                    [--width N] [--height N] [--background "#000"] [--workers N] [--msaa N]',
  );
  process.exit(2);
}

function frameArg(value: string | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error(`--${name} 不是数字：${value}`);
  return Math.floor(n);
}

async function main(): Promise<void> {
  // pnpm 会把 `--` 原样传下来，去掉它
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const { values } = parseArgs({
    args: argv,
    options: {
      scene: { type: 'string' },
      out: { type: 'string' },
      from: { type: 'string' },
      to: { type: 'string' },
      format: { type: 'string' },
      quality: { type: 'string' },
      width: { type: 'string' },
      height: { type: 'string' },
      background: { type: 'string' },
      workers: { type: 'string' },
      msaa: { type: 'string' },
    },
  });
  if (!values.scene) usage();

  const format = (values.format ?? 'auto') as 'auto' | 'mp4' | 'webm';
  if (format !== 'auto' && format !== 'mp4' && format !== 'webm') throw new Error(`--format 无效：${format}`);
  const quality = (values.quality ?? 'high') as 'high' | 'medium' | 'low';
  if (quality !== 'high' && quality !== 'medium' && quality !== 'low') throw new Error(`--quality 无效：${quality}`);
  const workers = values.workers !== undefined ? Number(values.workers) : 0;
  if (!Number.isInteger(workers) || workers < 0 || workers > 16) throw new Error(`--workers 无效：${values.workers}`);
  const width = values.width !== undefined ? Number(values.width) : undefined;
  const height = values.height !== undefined ? Number(values.height) : undefined;
  if (width !== undefined && (!Number.isInteger(width) || width <= 0)) throw new Error(`--width 无效：${values.width}`);
  if (height !== undefined && (!Number.isInteger(height) || height <= 0)) throw new Error(`--height 无效：${values.height}`);
  const msaa = values.msaa !== undefined ? Number(values.msaa) : defaultMsaa();
  if (msaa !== undefined && (!Number.isInteger(msaa) || msaa < 0)) throw new Error(`--msaa 无效：${values.msaa}`);

  const cwd = process.env.INIT_CWD ?? process.cwd();
  const scenePath = resolve(cwd, values.scene);
  let scene: SceneFile;
  try {
    scene = JSON.parse(readFileSync(scenePath, 'utf8')) as SceneFile;
  } catch (e) {
    throw new Error(`读取场景失败：${scenePath}\n${e instanceof Error ? e.message : String(e)}`);
  }
  const duration = Math.floor(Number(scene.duration));
  const fps = Number(scene.timebase?.fps);
  if (!(duration > 0) || !(fps > 0)) throw new Error(`场景缺少有效的 duration / timebase.fps：${scenePath}`);

  const from = Math.max(0, frameArg(values.from, 0, 'from'));
  const to = Math.min(duration, frameArg(values.to, duration, 'to'));
  if (to <= from) throw new Error(`导出范围无效：[${from}, ${to})（场景共 ${duration} 帧）`);

  const sceneBase = basename(scenePath)
    .replace(/\.scene\.json$/i, '')
    .replace(/\.json$/i, '')
    .replace(/[\\/:*?"<>|]+/g, '_');
  const explicitOut = values.out !== undefined ? resolve(cwd, values.out) : null;

  const { server, baseUrl } = await startServer();
  const browser = await launchBrowser();
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
    page.on('console', (m) => {
      if ((m.type() === 'error' || m.type() === 'warning') && !m.text().includes('GPU stall due to ReadPixels')) {
        process.stderr.write(`[page ${m.type()}] ${m.text()}\n`);
      }
    });
    page.on('pageerror', (e) => process.stderr.write(`[page error] ${e.message}\n`));

    const qs = new URLSearchParams({ workers: String(workers) });
    if (msaa !== undefined) qs.set('msaa', String(msaa));
    console.log(`[export] 打开 ${baseUrl}/export.html?${qs}`);
    await page.goto(`${baseUrl}/export.html?${qs}`, { waitUntil: 'load', timeout: 60_000 });
    await page.waitForFunction(() => !!window.__charaforgeExport, null, { timeout: 60_000 });
    await page.evaluate(() => window.__charaforgeExport.ready);
    const info = await page.evaluate(() => window.__charaforgeExport.info());
    console.log(`[export] GPU: ${info.renderer}（msaa ${info.msaa}, workers ${info.workers}）`);

    let lastStatus = '';
    const poll = setInterval(() => {
      void page
        .evaluate(() => window.__charaforgeExport.status())
        .then((s) => {
          if (s !== lastStatus) {
            lastStatus = s;
            console.log(`[export] ${s}`);
          }
        })
        .catch(() => {});
    }, 1000);
    (poll as { unref?: () => void }).unref?.();

    const t0 = Date.now();
    let meta: ExportPageResult;
    try {
      meta = await page.evaluate(async (o) => window.__charaforgeExport.render(o), {
        scene,
        from,
        to,
        format,
        quality,
        ...(width !== undefined ? { width } : {}),
        ...(height !== undefined ? { height } : {}),
        ...(values.background !== undefined ? { background: values.background } : {}),
      });
    } finally {
      clearInterval(poll);
    }
    console.log(
      `[export] ${meta.frames} 帧 ${meta.width}×${meta.height} @${meta.fps}fps，` +
        `${meta.duration.toFixed(3)}s，${meta.extension}/${meta.codecs.video}` +
        `${meta.codecs.audio ? `+${meta.codecs.audio}` : ''}，音频 ${meta.hasAudio ? '有' : '无'}`,
    );
    if (!(meta.bytes > 0)) throw new Error('导出结果为空');

    const outPath =
      explicitOut ?? resolve(cwd, '.data', 'export', `${sceneBase}-${from}-${to}.${meta.extension}`);
    if (explicitOut) {
      const lower = outPath.toLowerCase();
      if (!lower.endsWith(`.${meta.extension}`)) {
        console.warn(`[export] 注意：实际编码为 ${meta.extension}，但输出文件名为 ${basename(outPath)}`);
      }
    }
    mkdirSync(dirname(outPath), { recursive: true });
    const fd = openSync(outPath, 'w');
    let written = 0;
    try {
      while (written < meta.bytes) {
        const chunk = await page.evaluate(
          ([offset, length]) => window.__charaforgeExport.readResult(offset, length),
          [written, CHUNK] as const,
        );
        if (!chunk.data) throw new Error(`读取导出结果失败：offset ${written}/${meta.bytes}`);
        const buf = Buffer.from(chunk.data, 'base64');
        writeSync(fd, buf);
        written += buf.length;
        const pct = Math.floor((written / meta.bytes) * 100);
        console.log(`[export] 写出 ${written}/${meta.bytes} 字节 (${pct}%)`);
      }
    } finally {
      closeSync(fd);
    }
    const secs = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`[export] 完成 → ${outPath}（${written} 字节，${secs}s）`);
  } finally {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
