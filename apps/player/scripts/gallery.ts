/*
 * 图集：把默认姿态、每个开关参数单独置 1（key1～13、Movement1～3、Clothes1～5；
 * 若参数下限 < 0 再加一张 −1）、每个 exp3 单独生效的效果渲染成带标注的 PNG。
 *
 *   pnpm render:gallery            → .data/gallery/*.png + .data/gallery/contact-sheet.png + index.json
 *
 * 每张图都是一个单图层场景（经由与播放器相同的场景求值路径）：物理预热 120 帧（输入恒定），
 * 渲染帧 0。expression 片段从帧 0 开始、淡入 0。
 */
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { openHarness, REPO } from './harness-node.ts';

const MODEL_DIR = 'models/deepseek-chan/DeepSeek';
const MODEL = `${MODEL_DIR}/大肥鱼.model3.json`;
const SWITCHES = [
  ...Array.from({ length: 13 }, (_, i) => `key${i + 1}`),
  'Movement1', 'Movement2', 'Movement3',
  'Clothes1', 'Clothes2', 'Clothes3', 'Clothes4', 'Clothes5',
];
const OUT = join(REPO, '.data', 'gallery');
const WIDTH = 720;

interface Variant {
  file: string;
  label: string;
  params?: Record<string, number>;
  expression?: string;
}

async function main() {
  const h = await openHarness();
  try {
    console.log(`[gallery] GPU: ${h.info.renderer}`);
    const id = await h.page.evaluate((m) => window.__charaforge.createInstance(m), MODEL);
    const info = await h.page.evaluate((id) => window.__charaforge.instanceInfo(id), id);
    await h.page.evaluate((id) => window.__charaforge.destroyInstance(id), id);
    const { canvas } = info;
    const W = WIDTH;
    const H = Math.round((WIDTH * canvas.height) / canvas.width);
    const byId = new Map(info.parameters.map((p) => [p.id, p]));

    const variants: Variant[] = [{ file: '00_default', label: '默认姿态（全部参数为默认值）' }];
    SWITCHES.forEach((pid, i) => {
      const p = byId.get(pid);
      if (!p) {
        console.warn(`[gallery] 模型没有参数 ${pid}`);
        return;
      }
      const n = String(i + 1).padStart(2, '0');
      variants.push({ file: `10_${n}_${pid}=1`, label: `${pid} = 1　${p.name ?? ''}　[${p.min}, ${p.max}]`, params: { [pid]: 1 } });
      if (p.min < 0) variants.push({ file: `10_${n}_${pid}=-1`, label: `${pid} = −1　${p.name ?? ''}　[${p.min}, ${p.max}]`, params: { [pid]: -1 } });
    });
    const exps = readdirSync(join(REPO, 'assets', MODEL_DIR, 'Exp3')).filter((f) => f.endsWith('.exp3.json')).sort();
    exps.forEach((f, i) => variants.push({ file: `20_${String(i + 1).padStart(2, '0')}_exp_${f.replace('.exp3.json', '')}`, label: `exp3：${f}`, expression: `Exp3/${f}` }));

    mkdirSync(OUT, { recursive: true });
    const index: { file: string; label: string }[] = [];
    for (const v of variants) {
      const scene = {
        format: 'charaforge.scene',
        version: 1,
        meta: { title: v.label },
        timebase: { fps: 60 },
        duration: 1,
        composition: { width: W, height: H, background: '#ffffff' },
        assets: {
          models: [{ id: 'm', uri: MODEL }],
          expressions: v.expression ? [{ id: 'e', model: 'm', uri: `${MODEL_DIR}/${v.expression}` }] : [],
        },
        layers: [
          {
            id: 'l', kind: 'model', model: 'm', in: 0, out: 1,
            transform: { x: W / 2, y: H / 2, scale: W / canvas.width },
            physics: { warmup: 120 },
            expressions: v.expression ? [{ expression: 'e', start: 0, fadeIn: 0 }] : [],
            params: Object.entries(v.params ?? {}).map(([pid, val]) => ({ id: pid, curve: { keys: [{ t: 0, v: val }] } })),
          },
        ],
      };
      const png = await h.page.evaluate(async ([scene, label]) => {
        const cf = window.__charaforge;
        await cf.loadScene(scene);
        const r = await cf.renderFrame(0, { read: {} });
        const bin = atob(r.data!);
        const px = new Uint8ClampedArray(bin.length);
        for (let i = 0; i < bin.length; i++) px[i] = bin.charCodeAt(i);
        // 背景不透明，预乘 = 非预乘
        const band = 44;
        const c = document.createElement('canvas');
        c.width = r.width!;
        c.height = r.height! + band;
        const g = c.getContext('2d')!;
        g.fillStyle = '#ffffff';
        g.fillRect(0, 0, c.width, c.height);
        g.putImageData(new ImageData(px, r.width!, r.height!), 0, 0);
        g.fillStyle = '#1d2330';
        g.fillRect(0, r.height!, c.width, band);
        g.fillStyle = '#ffffff';
        g.font = '20px "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", sans-serif';
        g.textBaseline = 'middle';
        g.fillText(label as string, 12, r.height! + band / 2, c.width - 24);
        return c.toDataURL('image/png').split(',')[1]!;
      }, [scene, v.label] as const);
      writeFileSync(join(OUT, `${v.file}.png`), Buffer.from(png, 'base64'));
      index.push({ file: `${v.file}.png`, label: v.label });
      process.stdout.write('.');
    }
    process.stdout.write('\n');

    // 缩略图拼图
    const sheet = await h.page.evaluate(async (files: string[]) => {
      const cols = 8;
      const tw = 180;
      const imgs = await Promise.all(files.map((d) => { const i = new Image(); i.src = `data:image/png;base64,${d}`; return i.decode().then(() => i); }));
      const th = Math.round((tw * imgs[0]!.height) / imgs[0]!.width);
      const c = document.createElement('canvas');
      c.width = cols * tw;
      c.height = Math.ceil(imgs.length / cols) * th;
      const g = c.getContext('2d')!;
      g.fillStyle = '#ffffff';
      g.fillRect(0, 0, c.width, c.height);
      g.imageSmoothingQuality = 'high';
      imgs.forEach((im, i) => g.drawImage(im, (i % cols) * tw, Math.floor(i / cols) * th, tw, th));
      return c.toDataURL('image/png').split(',')[1]!;
    }, index.map((e) => (readFileB64(join(OUT, e.file)))));
    writeFileSync(join(OUT, 'contact-sheet.png'), Buffer.from(sheet, 'base64'));
    writeFileSync(join(OUT, 'index.json'), JSON.stringify({ model: MODEL, size: [W, H], images: index }, null, 2));
    console.log(`[gallery] ${index.length} 张 → ${OUT}（另有 contact-sheet.png、index.json）`);
  } finally {
    await h.close();
  }
}

function readFileB64(p: string): string {
  return readFileSync(p).toString('base64');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
