/*
 * 外部饰品导入 CLI。
 *
 *   pnpm accessory:import --input .data/hat.png --id hat --name "帽子" \
 *     --scene packages/scene/examples/accessory.scene.json --attach main --anchor 0.5,0.12
 *
 * 做四件事：
 *   1. 读 PNG/WebP，校验尺寸、透明像素（浏览器 canvas 解码，不执行图片内容）；
 *   2. 复制到 assets/accessories/<id>.<png|webp>，生成 96×96 缩略图；
 *   3. 更新 assets/accessories/index.json（id/name/size/sha256/thumbnail）；
 *   4. 在 stdout 打印可粘进 scene 的 ImageAsset + ImageLayer JSON。
 *
 * 默认不改 scene；场景仍是唯一编辑源。所有图片只在 Node/浏览器里解码，不执行。
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..', '..');
const MAX_SIDE = 4096;
const MAX_PIXELS = 16_777_216;
const MAX_BYTES = 16 * 1024 * 1024;
const THUMB_SIZE = 96;
const ID_RE = /^[A-Za-z0-9_.:-]+$/;

function usage() {
  console.error(`用法：pnpm accessory:import --input <png|webp> [选项]

必填：
  --input <path>        输入图片（PNG/WebP）

选项：
  --id <id>             资源/图层 id（缺省取文件名；只允许 [A-Za-z0-9_.:-]）
  --name <name>         显示名（缺省取文件名）
  --scene <path>        读取场景的 duration/composition，并校验 attach 目标；不修改场景
  --attach <layer-id>   绑定到模型图层；缺省为独立屏幕图层
  --anchor <x,y>        attach 锚点，目标模型画布归一化坐标，缺省 0.5,0.5
  --offset <x,y>        attach 偏移，目标模型画布像素，缺省 0,0
  --scale <n>           图层 transform.scale，缺省 1
  --duration <frames>   无 --scene 时的 out，缺省 240
  --out-dir <dir>       输出目录，缺省 assets/accessories
  --force               覆盖已有 id/文件
  --dry-run             只校验并打印 JSON，不写文件
  --help                显示本帮助

输出（stdout）：{ "asset": ImageAsset, "layer": ImageLayer }
`);
}

function fail(msg) {
  throw new Error(msg);
}

function parsePair(s, what) {
  const parts = String(s).split(',').map((v) => Number(v.trim()));
  if (parts.length !== 2 || !parts.every(Number.isFinite)) fail(`${what} 需要两个数字，例如 0.5,0.12`);
  return parts;
}

function sanitizeId(s) {
  const out = s.replace(/[^A-Za-z0-9_.:-]+/g, '-').replace(/^-+|-+$/g, '');
  return out || 'accessory';
}

function detectMime(bytes) {
  if (bytes.length >= 8 && bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47 && bytes[4] === 0x0d && bytes[5] === 0x0a && bytes[6] === 0x1a && bytes[7] === 0x0a) return 'image/png';
  if (bytes.length >= 12 && bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 && bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) return 'image/webp';
  return null;
}

/** 浏览器 canvas 解码：尺寸、alpha 统计、缩略图。不执行图片内容。 */
async function inspectImage(bytes, mime) {
  const browser = await chromium.launch({ headless: true, args: ['--disable-gpu'] });
  try {
    const page = await browser.newPage();
    await page.setContent('<!doctype html><html><body></body></html>');
    const b64 = bytes.toString('base64');
    return await page.evaluate(async ({ b64, mime, thumbSize }) => {
      const res = await fetch(`data:${mime};base64,${b64}`);
      const bmp = await createImageBitmap(await res.blob());
      try {
        if (bmp.width < 1 || bmp.height < 1) throw new Error(`图片尺寸非法：${bmp.width}×${bmp.height}`);
        const c = document.createElement('canvas');
        c.width = bmp.width;
        c.height = bmp.height;
        const g = c.getContext('2d', { willReadFrequently: true });
        if (!g) throw new Error('无法创建 canvas 2d 上下文');
        g.clearRect(0, 0, c.width, c.height);
        g.drawImage(bmp, 0, 0);
        const data = g.getImageData(0, 0, c.width, c.height).data;
        let nonOpaque = 0;
        let visible = 0;
        for (let i = 3; i < data.length; i += 4) {
          const a = data[i];
          if (a < 255) nonOpaque++;
          if (a > 0) visible++;
        }

        const t = document.createElement('canvas');
        t.width = thumbSize;
        t.height = thumbSize;
        const tg = t.getContext('2d');
        if (!tg) throw new Error('无法创建缩略图 canvas');
        tg.clearRect(0, 0, thumbSize, thumbSize);
        tg.imageSmoothingEnabled = true;
        tg.imageSmoothingQuality = 'high';
        const s = Math.min(thumbSize / bmp.width, thumbSize / bmp.height);
        const w = Math.max(1, Math.round(bmp.width * s));
        const h = Math.max(1, Math.round(bmp.height * s));
        tg.drawImage(bmp, Math.floor((thumbSize - w) / 2), Math.floor((thumbSize - h) / 2), w, h);
        const thumbnail = t.toDataURL('image/png').split(',')[1];
        return { width: bmp.width, height: bmp.height, nonOpaque, visible, thumbnail };
      } finally {
        bmp.close();
      }
    }, { b64, mime, thumbSize: THUMB_SIZE });
  } finally {
    await browser.close();
  }
}

async function main() {
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const { values } = parseArgs({
    args: argv,
    options: {
      input: { type: 'string' },
      id: { type: 'string' },
      name: { type: 'string' },
      scene: { type: 'string' },
      attach: { type: 'string' },
      anchor: { type: 'string' },
      offset: { type: 'string' },
      scale: { type: 'string' },
      duration: { type: 'string' },
      'out-dir': { type: 'string' },
      force: { type: 'boolean' },
      'dry-run': { type: 'boolean' },
      help: { type: 'boolean' },
    },
  });
  if (values.help) {
    usage();
    return;
  }
  if (!values.input) {
    usage();
    process.exitCode = 2;
    return;
  }

  const cwd = process.env.INIT_CWD ?? process.cwd();
  const input = resolve(cwd, values.input);
  if (!existsSync(input) || !statSync(input).isFile()) fail(`输入文件不存在：${input}`);
  const bytes = readFileSync(input);
  if (bytes.length > MAX_BYTES) fail(`文件过大：${bytes.length} 字节（上限 ${MAX_BYTES}）`);
  const mime = detectMime(bytes);
  if (!mime) fail('只支持 PNG / WebP（按文件头判断，不看扩展名）');
  const ext = mime === 'image/png' ? '.png' : '.webp';

  const base = basename(input, extname(input));
  const id = values.id !== undefined ? values.id : sanitizeId(base);
  if (!ID_RE.test(id)) fail(`id 非法：${id}（只允许 [A-Za-z0-9_.:-]）`);
  const name = values.name ?? base;
  const layerId = id;
  const scale = values.scale !== undefined ? Number(values.scale) : 1;
  if (!Number.isFinite(scale) || scale <= 0) fail(`--scale 必须是正数：${values.scale}`);
  const anchor = values.anchor !== undefined ? parsePair(values.anchor, '--anchor') : [0.5, 0.5];
  if (anchor.some((v) => v < 0 || v > 1)) fail('--anchor 的每个分量必须在 0..1');
  const offset = values.offset !== undefined ? parsePair(values.offset, '--offset') : [0, 0];

  let duration = values.duration !== undefined ? Number(values.duration) : 240;
  if (!Number.isInteger(duration) || duration < 1) fail(`--duration 必须是正整数：${values.duration}`);
  let composition = { width: 1280, height: 720 };
  let scene = null;
  if (values.scene) {
    const scenePath = resolve(cwd, values.scene);
    if (!existsSync(scenePath)) fail(`场景不存在：${scenePath}`);
    const { parseScene } = await import('@charaforge/scene');
    scene = parseScene(JSON.parse(readFileSync(scenePath, 'utf8')));
    duration = scene.duration;
    composition = { width: scene.composition.width, height: scene.composition.height };
    if (values.attach && !scene.layers.some((l) => l.id === values.attach && l.kind === 'model')) {
      fail(`attach 目标不是模型图层：${values.attach}`);
    }
    if (scene.layers.some((l) => l.id === layerId)) fail(`场景里已有图层 id：${layerId}（换一个 --id 或先删掉旧图层）`);
  }

  const info = await inspectImage(bytes, mime);
  if (info.width > MAX_SIDE || info.height > MAX_SIDE || info.width * info.height > MAX_PIXELS) {
    fail(`图片尺寸超出限制：${info.width}×${info.height}（单边 ≤ ${MAX_SIDE}，总像素 ≤ ${MAX_PIXELS}）`);
  }
  if (info.visible === 0) fail('图片没有任何可见像素（alpha 全为 0）');
  if (info.nonOpaque === 0) fail('图片没有透明/半透明像素（alpha 全为 255）；饰品图需要 alpha');

  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const file = `${id}${ext}`;
  const thumbFile = `${id}.thumb.png`;
  const outDir = resolve(cwd, values['out-dir'] ?? join(REPO, 'assets', 'accessories'));
  const relDir = relative(resolve(REPO, 'assets'), outDir).split(sep).join('/');
  if (isAbsolute(relDir) || relDir === '..' || relDir.startsWith('../')) fail(`--out-dir 必须位于 assets/ 下：${outDir}`);
  const imagePath = join(outDir, file);
  const thumbPath = join(outDir, thumbFile);
  const indexPath = join(outDir, 'index.json');

  if (!values['dry-run']) {
    if (!values.force && existsSync(imagePath)) fail(`文件已存在：${imagePath}（用 --force 覆盖）`);
    mkdirSync(outDir, { recursive: true });
    writeFileSync(imagePath, bytes);
    writeFileSync(thumbPath, Buffer.from(info.thumbnail, 'base64'));

    let index = { version: 1, items: [] };
    if (existsSync(indexPath)) {
      try {
        index = JSON.parse(readFileSync(indexPath, 'utf8'));
      } catch (e) {
        fail(`index.json 解析失败：${String(e?.message ?? e)}`);
      }
      if (!Array.isArray(index.items)) fail('index.json 缺少 items 数组');
    }
    const entry = {
      id,
      name,
      file,
      size: { width: info.width, height: info.height },
      bytes: bytes.length,
      sha256,
      thumbnail: thumbFile,
    };
    const old = index.items.findIndex((x) => x.id === id);
    if (old >= 0) {
      if (!values.force) fail(`index.json 已有 id：${id}（用 --force 覆盖）`);
      index.items[old] = entry;
    } else index.items.push(entry);
    index.items.sort((a, b) => String(a.id).localeCompare(String(b.id)));
    writeFileSync(indexPath, JSON.stringify(index, null, 2) + '\n');
  }

  const uriDir = relDir ? `${relDir}/` : '';
  const asset = { id, uri: `${uriDir}${file}`, width: info.width, height: info.height, sha256 };
  const transform = { x: 0, y: 0, scale, anchorX: info.width / 2, anchorY: info.height / 2 };
  const layer = {
    id: layerId,
    kind: 'image',
    image: id,
    in: 0,
    out: duration,
    ...(values.attach
      ? { transform, attach: { layer: values.attach, anchor, offset, rotateWith: true, scaleWith: true } }
      : { space: 'screen', transform: { ...transform, x: composition.width / 2, y: composition.height / 2 } }),
    opacity: 1,
    provenance: { source: 'import', tool: 'accessory:import', detail: { file, sha256 } },
  };

  console.error(`[accessory] ${info.width}×${info.height}，非不透明像素 ${info.nonOpaque}，可见像素 ${info.visible}`);
  if (!values['dry-run']) {
    console.error(`[accessory] 图片 -> ${imagePath}`);
    console.error(`[accessory] 缩略图 -> ${thumbPath}`);
    console.error(`[accessory] 目录 -> ${indexPath}`);
  }
  console.error(`[accessory] 把 asset 粘到 assets.images，layer 粘到 layers（layer id=${layerId}）`);
  console.log(JSON.stringify({ asset, layer }, null, 2));
}

main().catch((e) => {
  console.error(`[accessory] ${String(e?.message ?? e)}`);
  process.exit(1);
});
