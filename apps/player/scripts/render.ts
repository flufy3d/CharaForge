/*
 * 离线渲染 CLI：用 harness（与播放器同一份渲染代码）把场景的指定帧渲染成 PNG。
 *
 *   pnpm render --scene packages/scene/examples/minimal.scene.json --frames 0-60 --out .data/render/a
 *
 * 参数：
 *   --scene <path>    场景文件（相对当前目录或绝对路径；在 Node 端读取后传给页面，所以不受 dev 服务器目录白名单限制）
 *   --frames <spec>   帧列表：'0-60'（含两端）、'0,10,20'、'100'；可混用 '0-10,50'。缺省全片
 *   --out <dir>       输出目录，文件名 frame_000000.png
 *   --scale <s>       输出缩放（缺省 1）
 *   --msaa <n>        覆盖 MSAA 采样数（缺省 4）
 * 帧按升序渲染（顺序播放路径）；输出 PNG 只取决于像素（见 png.ts），两次运行逐位一致。
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { openHarness } from './harness-node.ts';
import { encodePng } from './png.ts';

export function parseFrames(spec: string, duration: number): number[] {
  const out = new Set<number>();
  for (const part of spec.split(',').map((s) => s.trim()).filter(Boolean)) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(part);
    if (!m) throw new Error(`无法解析帧范围：${part}`);
    const a = Number(m[1]);
    const b = m[2] !== undefined ? Number(m[2]) : a;
    for (let f = Math.min(a, b); f <= Math.max(a, b); f++) out.add(f);
  }
  return [...out].filter((f) => f >= 0 && f < duration).sort((x, y) => x - y);
}

async function main() {
  // pnpm 会把 `--` 原样传下来，去掉它
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const { values } = parseArgs({
    args: argv,
    options: {
      scene: { type: 'string' },
      frames: { type: 'string' },
      out: { type: 'string' },
      scale: { type: 'string' },
      msaa: { type: 'string' },
    },
  });
  if (!values.scene || !values.out) {
    console.error('用法：pnpm render --scene <path> [--frames 0-60] --out <dir> [--scale 1] [--msaa 4]');
    process.exit(2);
  }
  const cwd = process.env.INIT_CWD ?? process.cwd();
  const scenePath = resolve(cwd, values.scene);
  const outDir = resolve(cwd, values.out);
  const scene = JSON.parse(readFileSync(scenePath, 'utf8'));
  const scale = values.scale ? Number(values.scale) : 1;

  const h = await openHarness({ msaa: values.msaa !== undefined ? Number(values.msaa) : undefined });
  try {
    console.log(`[render] GPU: ${h.info.renderer}（msaa ${h.info.msaa}）`);
    const sum = await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
    const frames = parseFrames(values.frames ?? `0-${sum.duration - 1}`, sum.duration);
    mkdirSync(outDir, { recursive: true });
    const t0 = Date.now();
    for (const f of frames) {
      const r = await h.page.evaluate(([f, scale]) => window.__charaforge.renderFrame(f, { read: { scale } }), [f, scale] as const);
      const png = encodePng(r.width!, r.height!, Buffer.from(r.data!, 'base64'));
      writeFileSync(join(outDir, `frame_${String(f).padStart(6, '0')}.png`), png);
    }
    console.log(`[render] ${frames.length} 帧 → ${outDir}（${((Date.now() - t0) / 1000).toFixed(1)}s）`);
  } finally {
    await h.close();
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
