/*
 * 拟合命令行。
 *
 *   node --experimental-strip-types tools/fit/src/cli.ts fit --shot s43 [--until 3700] [--preview 4]
 *   node --experimental-strip-types tools/fit/src/cli.ts assemble          # 合成 scenes/deepseek-chan/scene.json
 *
 * 详见 tools/fit/README.md。
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
import { assembleScene } from './assemble.ts';
import { renderPatched } from './render.ts';
import { fitShot, type ModelCalib } from './fitshot.ts';
import { installPageHelpers } from './page.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { FIT_DIR, MODEL_URI } from './paths.ts';
import { planFor } from './plans.ts';
import { loadShots } from './shotsutil.ts';

/** 模型参考姿态（全部默认参数、物理静止）下的双眼虹膜中心（画布像素），用于由 shots 的 eyes 求初始变换 */
async function calibrate(page: import('playwright').Page): Promise<ModelCalib> {
  const file = join(FIT_DIR, 'calib.json');
  if (existsSync(file)) return JSON.parse(readFileSync(file, 'utf8'));
  const r = await page.evaluate(async (m) => {
    const H = window.__charaforge;
    const id = await H.createInstance(m);
    const info = H.instanceInfo(id);
    for (let i = 0; i < 180; i++) H.stepInstance(id, { physics: true });
    // 画布 [1000, 2750] × [300, 2050] → 1750×1750（scale 1）
    const px = await H.renderInstance(id, { transform: { x: 0, y: 0, scale: 1, anchorX: 1000, anchorY: 300 }, width: 1750, height: 1750 });
    // 虹膜主网格（ArtMesh236 左、ArtMesh134 右）的顶点质心，画布像素
    const inst = (window as any).__cfRuntime.instance(id);
    const mdl = inst.model, c = inst.canvas;
    const mesh = ['ArtMesh236', 'ArtMesh134'].map((name) => {
      for (let d = 0; d < mdl.getDrawableCount(); d++) {
        const did = mdl.getDrawableId(d);
        if ((did.getString().s ?? String(did.getString())) !== name) continue;
        const v = mdl.getDrawableVertices(d);
        let x = 0, y = 0;
        for (let k = 0; k < v.length / 2; k++) { x += v[2 * k]; y += v[2 * k + 1]; }
        const n = v.length / 2;
        return [c.originX + (x / n) * c.pixelsPerUnit, c.originY - (y / n) * c.pixelsPerUnit];
      }
      throw new Error(name);
    });
    H.destroyInstance(id);
    return { px, canvas: info.canvas, mesh };
  }, MODEL_URI);
  const d = Buffer.from(r.px.data!, 'base64');
  const W = r.px.width;
  // 虹膜：青色（B、G 都高，R 低）。从画布中线两侧的眼部窗口出发，在半径 70px 内迭代求青色像素质心（排除发梢上的青色高光）
  const cyan = (x: number, y: number) => {
    const o = (y * W + x) * 4;
    const R = d[o]!, G = d[o + 1]!, B = d[o + 2]!;
    return d[o + 3]! > 200 && B > 170 && G > 130 && R < 130 && G > 0.65 * B;
  };
  const centers: [number, number][] = [[705, 1285], [1045, 1285]]; // 输出缓冲坐标（= 画布 − [1000, 300]）的初值
  const acc = [[0, 0, 0], [0, 0, 0]];
  for (let it = 0; it < 4; it++)
    centers.forEach(([cx, cy], k) => {
      let sx = 0, sy = 0, n = 0;
      for (let y = Math.round(cy - 70); y <= cy + 70; y++)
        for (let x = Math.round(cx - 70); x <= cx + 70; x++)
          if ((x - cx) ** 2 + (y - cy) ** 2 <= 4900 && cyan(x, y)) { sx += x; sy += y; n++; }
      if (n) centers[k] = [sx / n, sy / n];
      acc[k] = [centers[k]![0] + 1000.5, centers[k]![1] + 300.5, n]; // 像素中心
    });
  if (acc[0]![2]! < 200 || acc[1]![2]! < 200) throw new Error(`眼睛定位失败：${JSON.stringify(acc)}`);
  const calib: ModelCalib = {
    eyes: [[acc[0]![0]!, acc[0]![1]!], [acc[1]![0]!, acc[1]![1]!]],
    canvas: { width: r.canvas.width, height: r.canvas.height },
    irisOffset: [0, 1].map((k) => [acc[k]![0]! - r.mesh[k]![0]!, acc[k]![1]! - r.mesh[k]![1]!]) as [[number, number], [number, number]],
  };
  mkdirSync(FIT_DIR, { recursive: true });
  writeFileSync(file, JSON.stringify(calib, null, 2));
  // 校验图：标出检测到的眼睛中心
  const out = Buffer.from(d);
  for (const [ex, ey] of calib.eyes)
    for (let k = -12; k <= 12; k++)
      for (const [x, y] of [[Math.round(ex - 1000) + k, Math.round(ey - 300)], [Math.round(ex - 1000), Math.round(ey - 300) + k]] as const) {
        const o = (y * W + x) * 4;
        out[o] = 255; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 255;
      }
  writeFileSync(join(FIT_DIR, 'calib.png'), encodePng(W, r.px.height, out));
  return calib;
}

async function main() {
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const cmd = argv[0];
  const { values } = parseArgs({
    args: argv.slice(1),
    options: { shot: { type: 'string' }, until: { type: 'string' }, start: { type: 'string' }, only: { type: 'string' }, scene: { type: 'string' }, frames: { type: 'string' }, preview: { type: 'string' }, out: { type: 'string' } },
  });
  if (cmd === 'assemble') {
    assembleScene(values.out);
    return;
  }
  if (cmd === 'render') {
    // 绕过 runtime 清屏色 bug 的渲染（见 render.ts 文件头）；参数同 pnpm render
    await renderPatched(values.scene ?? 'scenes/deepseek-chan/scene.json', values.frames!, values.out!);
    return;
  }
  if (cmd !== 'fit' || !values.shot) {
    console.error('用法：cli.ts fit --shot s43[,s24,...] [--until F] [--preview N] | cli.ts assemble [--out path]');
    process.exit(2);
  }
  const shots = loadShots();
  const h = await openHarness();
  try {
    console.log(`[fit] GPU: ${h.info.renderer}（msaa ${h.info.msaa}）`);
    await h.page.evaluate(installPageHelpers);
    const calib = await calibrate(h.page);
    console.log(`[fit] 模型眼睛（画布像素）：${JSON.stringify(calib.eyes)}`);
    for (const id of values.shot.split(',')) {
      const shot = shots.shots.find((s) => s.id === id);
      if (!shot) throw new Error(`没有镜头 ${id}`);
      const { plans, options } = planFor(shot);
      const logFile = join(FIT_DIR, id, 'fit.log');
      mkdirSync(join(FIT_DIR, id), { recursive: true });
      writeFileSync(logFile, '');
      await fitShot(h.page, shot, plans, calib, {
        ...options,
        until: values.until ? Number(values.until) : undefined,
        start: values.start ? Number(values.start) : undefined,
        only: values.only ? values.only.split(',') : undefined,
        previewEvery: values.preview ? Number(values.preview) : options.previewEvery,
        log: (s) => {
          console.log(s);
          appendFileSync(logFile, `${s}\n`);
        },
      });
    }
  } finally {
    await h.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
