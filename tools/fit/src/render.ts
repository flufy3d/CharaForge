/*
 * 临时渲染入口（绕过 runtime 的一个 GL 状态 bug，等子代理 A 修复后即可删除，改用 `pnpm render`）：
 *
 * 现象：Cubism 绘制之后，Compositor 里任何 clear=true 的 Pixi pass（frameHold 缓存拷贝、color/blur 滤镜、缩放读回）
 * 在源透明处输出不透明白色 (255,255,255,255)。原因：Cubism 渲染器绘制遮罩时直接调用 gl.clearColor(1,1,1,1)，
 * Pixi 的 GlRenderTargetAdaptor 缓存了上一次设置的清屏色 [0,0,0,0]，认为无需重设，于是 GL 用白色清屏。
 * 复现：tools/fit/scripts/dbg_pass2.ts（copyOfCleared 正常、drawModel 之后 copyCorner 为白色）。
 * 修法建议（runtime 内）：Compositor.pass() 里用 clear=false 并先调 clearRT()，或在 drawModel 的 finally 里 gl.clearColor(0,0,0,0)。
 *
 * 这里在页面里给 compositor.pass 包一层：每次 pass 之前把 GL 清屏色设回 (0,0,0,0)，与 Pixi 的缓存一致。
 * 除此之外与 apps/player/scripts/render.ts 完全相同（同一 harness、同一场景求值路径、同一 PNG 编码）。
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { parseFrames } from '../../../apps/player/scripts/render.ts';

export async function renderPatched(scenePath: string, frameSpec: string, outDir: string): Promise<void> {
  const scene = JSON.parse(readFileSync(resolve(scenePath), 'utf8'));
  const h = await openHarness();
  try {
    await h.page.evaluate(() => {
      const c = (window as any).__cfRuntime.compositor;
      const orig = c.pass.bind(c);
      c.pass = (...a: unknown[]) => { c.gl.clearColor(0, 0, 0, 0); return orig(...a); };
    });
    const sum = await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
    const frames = parseFrames(frameSpec, sum.duration);
    mkdirSync(outDir, { recursive: true });
    const t0 = Date.now();
    for (const f of frames) {
      const r = await h.page.evaluate((f) => window.__charaforge.renderFrame(f, { read: {} }), f);
      writeFileSync(join(outDir, `frame_${String(f).padStart(6, '0')}.png`), encodePng(r.width!, r.height!, Buffer.from(r.data!, 'base64')));
    }
    console.log(`[render-patched] ${frames.length} 帧 → ${outDir}（${((Date.now() - t0) / 1000).toFixed(1)}s）`);
  } finally {
    await h.close();
  }
}
