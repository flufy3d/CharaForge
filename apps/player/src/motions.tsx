/*
 * CharaForge 动作 / 表情素材查看器。
 *
 * 独立页面 motions.html：列出当前模型的 motion3 / exp3，点击后构造一个临时预览 scene（不写盘），
 * 复用 PlayerEngine / CharaForgeRuntime / Compositor 播放。不加载原参考视频/音频，也不修改
 * scenes/deepseek-chan/scene.json 或任何模型文件。
 *
 * URL 参数：
 *   kind=motion|expression   初始素材类型（缺省 motion）
 *   name=<文件名>            初始素材文件名（缺省 IDLE；kind=expression 时缺省第一个表情）
 *   frame=<n>                初始帧（缺省 0）
 *   loop=0                   动作循环初始关闭（缺省用 motion3 的 Meta.Loop）
 *   idle=0                   表情预览不叠加 IDLE（缺省叠加）
 *
 * 快捷键：空格 播放/暂停；←/→ ±1 帧；Shift+←/→ ±10 帧；Home/End；L 切换循环。
 * 缩略图：一次 contact sheet 渲染生成所有素材的 160×160 缩略图，缓存到 localStorage；
 *         列表显示缩略图，顶部“图集”打开全部素材的图集。
 */
import { asciiUrl } from '@charaforge/runtime';
import type { Scene } from '@charaforge/scene';
import { StrictMode, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { PlayerEngine, type EngineStatus } from './engine.ts';
import {
  ANIMATION_DIR,
  EXP3_DIR,
  PREVIEW_FPS,
  PREVIEW_MODEL_URI,
  THUMBNAIL_COLUMNS,
  THUMBNAIL_TILE,
  assetKey,
  assetUriFromDir,
  buildPreviewScene,
  buildThumbnailScene,
  catalogAssets,
  declaredAssetUri,
  declaredExpressionFiles,
  declaredMotionFiles,
  displayName,
  fitTransform,
  isExpressionFile,
  isMotionFile,
  parseExpressionInfo,
  parseMotionMeta,
  previewDuration,
  sortAssets,
  type AssetFileInfo,
  type AssetKind,
  type AssetSource,
  type Catalog,
  type CatalogAsset,
  type ExpressionAssetInfo,
  type ExpressionInfo,
  type MotionAssetInfo,
  type MotionMeta,
} from './motionCatalog.ts';
import { pageUrl, sha256Hex, timecode } from './util.ts';

const params = new URLSearchParams(location.search);
const LIST_ENDPOINT = '__charaforge/list-assets';

type Phase = 'booting' | 'ready' | 'error';

interface PreviewInfo {
  kind: AssetKind;
  /** 文件名（含扩展名） */
  file: string;
  /** 显示名（去掉扩展名，或 model3 里声明的 Name） */
  name: string;
  uri: string;
  duration: number;
  fps: number;
  composition: { width: number; height: number };
  loop: boolean;
  /** 表情预览是否叠加了 IDLE 动作 */
  withIdle: boolean;
  source: AssetSource;
  size: number;
  mtime: string;
  meta?: MotionMeta;
  info?: ExpressionInfo;
}

interface ViewerError {
  title: string;
  hint?: string;
  detail: string;
}

interface ViewerState {
  phase: Phase;
  stage: string;
  error: ViewerError | null;
  catalog: Catalog | null;
  warnings: string[];
  selected: { kind: AssetKind; file: string } | null;
  preview: PreviewInfo | null;
  loadingPreview: boolean;
  loop: boolean;
  withIdle: boolean;
  /** kind:file → 160×160 PNG data URL；null = 尚未生成（生成中或失败） */
  thumbnails: Record<string, string> | null;
  status: EngineStatus | null;
}

interface MotionViewerApi {
  ready: Promise<void>;
  readyState: Phase;
  error: string | null;
  catalog: () => Catalog | null;
  warnings: () => string[];
  preview: () => PreviewInfo | null;
  selected: () => { kind: AssetKind; file: string } | null;
  select: (kind: AssetKind, file: string) => Promise<void>;
  seekAndRender: (frame: number) => Promise<void>;
  hashFrame: () => Promise<string>;
  thumbnails: () => Record<string, string> | null;
  status: () => EngineStatus | null;
  setLoop: (loop: boolean) => Promise<void>;
  setWithIdle: (withIdle: boolean) => Promise<void>;
}

declare global {
  interface Window {
    __motionViewer?: MotionViewerApi;
    __charaforgeNoCore?: boolean;
    __player?: PlayerEngine;
  }
}

let view: ViewerState = {
  phase: 'booting',
  stage: '初始化',
  error: null,
  catalog: null,
  warnings: [],
  selected: null,
  preview: null,
  loadingPreview: false,
  loop: false,
  withIdle: true,
  thumbnails: null,
  status: null,
};
const listeners = new Set<(s: ViewerState) => void>();
function setView(patch: Partial<ViewerState>): void {
  view = { ...view, ...patch };
  if ('error' in patch) viewer.error = patch.error?.detail ?? null;
  for (const l of listeners) l(view);
}

let engine: PlayerEngine | null = null;
let booted = false;
let selectQueue: Promise<void> = Promise.resolve();
let readyResolve!: () => void;
const ready = new Promise<void>((resolve) => {
  readyResolve = resolve;
});

function errorMessage(e: unknown): string {
  return String((e as Error)?.message ?? e);
}

function explainError(e: unknown): ViewerError {
  const detail = String((e as Error)?.stack ?? e);
  const msg = errorMessage(e);
  if (/素材目录读取失败 404|assets\/models\/ 不存在|模型文件缺失|加载失败 404/.test(msg)) {
    return {
      title: '模型素材不存在',
      hint: 'assets/ 不进 git。请先在仓库根目录运行 pnpm setup:assets（或 pnpm run setup），再刷新。',
      detail,
    };
  }
  if (/Live2DCubismCore/.test(msg)) {
    return {
      title: '没有找到 Live2D Cubism Core',
      hint: '请先运行 pnpm setup:cubism（或 pnpm run setup），它会下载到 vendor/cubism/core/。',
      detail,
    };
  }
  if (/WebGL2/.test(msg)) {
    return { title: '浏览器不支持 WebGL2', hint: '请换用新版 Chrome / Edge，并确认没有禁用硬件加速。', detail };
  }
  if (/schema|校验|must|expected/i.test(msg)) {
    return { title: '临时预览场景校验失败', hint: '这通常是查看器构造 scene 的 bug，请把详细信息一起反馈。', detail };
  }
  return { title: '动作 / 表情查看器启动失败', detail };
}

interface RawAsset {
  file: string;
  uri: string;
  dir: string;
  size: number;
  mtime: string;
  source: AssetSource;
  declaredName?: string;
}

function rawFromDirectory(dir: string, f: AssetFileInfo): RawAsset {
  return { file: f.name, uri: assetUriFromDir(dir, f.name), dir, size: f.size, mtime: f.mtime, source: 'directory' };
}

function rawFromDeclared(file: string, declaredName?: string): RawAsset {
  const uri = declaredAssetUri(PREVIEW_MODEL_URI, file);
  const slash = uri.lastIndexOf('/');
  const name = slash >= 0 ? uri.slice(slash + 1) : uri;
  return {
    file: name,
    uri,
    dir: slash >= 0 ? `assets/${uri.slice(0, slash)}` : 'assets/',
    size: 0,
    mtime: '',
    source: 'declared',
    declaredName,
  };
}

function dedupeRaw(assets: RawAsset[]): RawAsset[] {
  const seen = new Set<string>();
  return assets.filter((a) => {
    if (seen.has(a.uri)) return false;
    seen.add(a.uri);
    return true;
  });
}

/** 场景资源 uri 相对于 assets/（与 runtime 的 assetBase 一致）。 */
function assetUrl(uri: string): string {
  return new URL(uri.replace(/^\/+/, ''), pageUrl('assets/')).href;
}

async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(asciiUrl(url));
  if (!res.ok) throw new Error(`加载失败 ${res.status}：${url}`);
  return (await res.json()) as unknown;
}

async function listDirectory(dir: string): Promise<AssetFileInfo[]> {
  const u = new URL(pageUrl(LIST_ENDPOINT));
  u.searchParams.set('dir', dir);
  const res = await fetch(u.href);
  if (!res.ok) {
    let detail = '';
    try {
      const body = (await res.json()) as { error?: unknown };
      if (typeof body.error === 'string') detail = `（${body.error}）`;
    } catch {
      /* 非 JSON 错误体 */
    }
    throw new Error(`素材目录读取失败 ${res.status}：${dir}${detail}`);
  }
  const body = (await res.json()) as { files?: unknown };
  if (!Array.isArray(body.files)) throw new Error(`素材目录返回格式无效：${dir}`);
  return body.files as AssetFileInfo[];
}

async function loadCatalog(onStage: (stage: string) => void): Promise<Catalog> {
  const warnings: string[] = [];
  onStage('读取 model3.json（检查是否声明 Motions / Expressions）');
  let model3: unknown = null;
  try {
    model3 = await fetchJson(assetUrl(PREVIEW_MODEL_URI));
  } catch (e) {
    warnings.push(`model3.json 读取失败，将只依赖目录列表：${errorMessage(e)}`);
  }

  const declaredMotions = declaredMotionFiles(model3);
  const declaredExpressions = declaredExpressionFiles(model3);
  const motionSource: AssetSource = declaredMotions.length ? 'declared' : 'directory';
  const expressionSource: AssetSource = declaredExpressions.length ? 'declared' : 'directory';

  let rawMotions: RawAsset[];
  if (declaredMotions.length) {
    rawMotions = dedupeRaw(declaredMotions.map((file) => rawFromDeclared(file)));
  } else {
    try {
      const files = (await listDirectory(ANIMATION_DIR)).filter((f) => isMotionFile(f.name));
      rawMotions = files.map((f) => rawFromDirectory(ANIMATION_DIR, f));
    } catch (e) {
      warnings.push(`动作目录读取失败：${errorMessage(e)}`);
      rawMotions = [];
    }
  }

  let rawExpressions: RawAsset[];
  if (declaredExpressions.length) {
    rawExpressions = dedupeRaw(declaredExpressions.map((e) => rawFromDeclared(e.file, e.name)));
  } else {
    try {
      const files = (await listDirectory(EXP3_DIR)).filter((f) => isExpressionFile(f.name));
      rawExpressions = files.map((f) => rawFromDirectory(EXP3_DIR, f));
    } catch (e) {
      warnings.push(`表情目录读取失败：${errorMessage(e)}`);
      rawExpressions = [];
    }
  }

  const total = rawMotions.length + rawExpressions.length;
  let done = 0;
  const bump = () => {
    done++;
    onStage(`解析素材元数据 ${done}/${total}`);
  };
  onStage(`解析素材元数据 0/${total}`);

  const motions = (
    await Promise.all(
      rawMotions.map(async (raw): Promise<MotionAssetInfo | null> => {
        try {
          const meta = parseMotionMeta(await fetchJson(assetUrl(raw.uri)), raw.file);
          bump();
          return {
            kind: 'motion',
            name: raw.declaredName ?? displayName(raw.file),
            file: raw.file,
            uri: raw.uri,
            dir: raw.dir,
            size: raw.size,
            mtime: raw.mtime,
            source: raw.source,
            meta,
          };
        } catch (e) {
          warnings.push(`动作 ${raw.file} 解析失败：${errorMessage(e)}`);
          bump();
          return null;
        }
      }),
    )
  ).filter((x): x is MotionAssetInfo => x !== null);

  const expressions = (
    await Promise.all(
      rawExpressions.map(async (raw): Promise<ExpressionAssetInfo | null> => {
        try {
          const info = parseExpressionInfo(await fetchJson(assetUrl(raw.uri)), raw.file);
          bump();
          return {
            kind: 'expression',
            name: raw.declaredName ?? displayName(raw.file),
            file: raw.file,
            uri: raw.uri,
            dir: raw.dir,
            size: raw.size,
            mtime: raw.mtime,
            source: raw.source,
            info,
          };
        } catch (e) {
          warnings.push(`表情 ${raw.file} 解析失败：${errorMessage(e)}`);
          bump();
          return null;
        }
      }),
    )
  ).filter((x): x is ExpressionAssetInfo => x !== null);

  sortAssets(motions);
  sortAssets(expressions);
  if (!motions.length && !expressions.length && warnings.length) throw new Error(warnings.join('\n'));
  return {
    modelUri: PREVIEW_MODEL_URI,
    motions,
    expressions,
    sources: { motions: motionSource, expressions: expressionSource },
    warnings,
  };
}

const THUMBNAIL_CACHE_KEY = 'charaforge.motion-thumbs.v1';

function thumbnailCacheKey(cat: Catalog): string {
  const files = [...cat.motions, ...cat.expressions].map((a) => `${a.kind}:${a.file}:${a.size}:${a.mtime}`);
  return [cat.modelUri, ...files].join('|');
}

function readThumbnailCache(cat: Catalog): Record<string, string> | null {
  try {
    const raw = localStorage.getItem(THUMBNAIL_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { key?: unknown; images?: unknown };
    if (parsed.key !== thumbnailCacheKey(cat) || !parsed.images || typeof parsed.images !== 'object' || Array.isArray(parsed.images)) return null;
    const images = parsed.images as Record<string, unknown>;
    for (const a of catalogAssets(cat)) {
      if (typeof images[assetKey(a)] !== 'string') return null;
    }
    return images as Record<string, string>;
  } catch {
    return null;
  }
}

function writeThumbnailCache(cat: Catalog, images: Record<string, string>): void {
  try {
    localStorage.setItem(THUMBNAIL_CACHE_KEY, JSON.stringify({ key: thumbnailCacheKey(cat), images }));
  } catch {
    /* localStorage 不可用 / 超出配额时忽略；缩略图仍可在本次会话使用 */
  }
}

/**
 * 用一次 contact sheet 渲染生成全部缩略图：
 *   - 每个素材一个模型图层，排成 6 列网格，物理关闭，只渲染帧 0；
 *   - motion 通过 offset 取动作中段，避免 31 个图层各从 0 逐帧推进；
 *   - 读出整张图后按 tile 裁成 data URL，供列表缩略图和图集使用。
 */
async function generateThumbnails(eng: PlayerEngine, cat: Catalog, onStage: (stage: string) => void): Promise<Record<string, string>> {
  const assets = catalogAssets(cat);
  if (!assets.length) return {};
  onStage(`渲染缩略图场景（${assets.length} 个素材）`);
  const scene = await eng.runtime.loadScene(buildThumbnailScene(cat));
  const first = scene.layers[0];
  if (!first || first.kind !== 'model') throw new Error('缩略图 scene 没有模型图层');
  const canvas = eng.runtime.layerInstance(first.id)?.canvas;
  if (!canvas) throw new Error('缩略图 scene 没有模型实例');
  const scale = Math.min(THUMBNAIL_TILE / canvas.width, THUMBNAIL_TILE / canvas.height) * 0.86;
  assets.forEach((_, i) => {
    const col = i % THUMBNAIL_COLUMNS;
    const row = Math.floor(i / THUMBNAIL_COLUMNS);
    const layer = scene.layers[i]!;
    (layer as unknown as { transform: { x: number; y: number; scale: number } }).transform = {
      x: col * THUMBNAIL_TILE + THUMBNAIL_TILE / 2,
      y: row * THUMBNAIL_TILE + THUMBNAIL_TILE / 2,
      scale,
    };
  });
  eng.runtime.invalidate();
  eng.redraw();
  await eng.runtime.renderFrameAsync(0);
  const px = eng.runtime.readPixels();
  const sheet = document.createElement('canvas');
  sheet.width = px.width;
  sheet.height = px.height;
  const sheetCtx = sheet.getContext('2d');
  if (!sheetCtx) throw new Error('无法创建 2D canvas 用于裁切缩略图');
  sheetCtx.putImageData(new ImageData(new Uint8ClampedArray(px.data), px.width, px.height), 0, 0);
  onStage(`裁切缩略图（${assets.length} 个素材）`);
  const images: Record<string, string> = {};
  for (const [i, asset] of assets.entries()) {
    const col = i % THUMBNAIL_COLUMNS;
    const row = Math.floor(i / THUMBNAIL_COLUMNS);
    const tile = document.createElement('canvas');
    tile.width = THUMBNAIL_TILE;
    tile.height = THUMBNAIL_TILE;
    const ctx = tile.getContext('2d');
    if (!ctx) throw new Error('无法创建 2D canvas 用于缩略图');
    ctx.drawImage(sheet, col * THUMBNAIL_TILE, row * THUMBNAIL_TILE, THUMBNAIL_TILE, THUMBNAIL_TILE, 0, 0, THUMBNAIL_TILE, THUMBNAIL_TILE);
    images[assetKey(asset)] = tile.toDataURL('image/png');
  }
  return images;
}

function updateUrl(kind: AssetKind, file: string, loop: boolean, withIdle: boolean): void {
  const q = new URLSearchParams(location.search);
  q.set('kind', kind);
  q.set('name', file);
  q.delete('frame');
  if (kind === 'motion' && !loop) q.set('loop', '0');
  else q.delete('loop');
  if (kind === 'expression' && !withIdle) q.set('idle', '0');
  else q.delete('idle');
  const search = q.toString();
  history.replaceState(null, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`);
}

function findAsset(cat: Catalog, kind: AssetKind, file: string | null): MotionAssetInfo | ExpressionAssetInfo | undefined {
  const list: (MotionAssetInfo | ExpressionAssetInfo)[] = kind === 'motion' ? cat.motions : cat.expressions;
  return file ? list.find((a) => a.file === file) : undefined;
}

/** 等待某一帧真正上屏；Playwright smoke / 深链初始帧用。 */
function seekAndRender(frame: number): Promise<void> {
  const eng = engine;
  if (!eng?.scene) return Promise.reject(new Error('尚未加载预览场景'));
  const target = Math.max(0, Math.min(eng.duration - 1, Math.floor(frame)));
  return new Promise<void>((resolve, reject) => {
    let done = false;
    const prev = eng.onStatus;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      eng.onStatus = prev;
      reject(new Error(`等待渲染帧 ${target} 超时`));
    }, 15_000);
    eng.onStatus = (s) => {
      prev(s);
      if (done || s.frame !== target || !s.ready) return;
      done = true;
      clearTimeout(timer);
      eng.onStatus = prev;
      resolve();
    };
    eng.seek(target);
  });
}

let lastStatusAt = 0;
let lastStatusSig = '';
let lastPlaying = false;
let replaying = false;

function onEngineStatus(s: EngineStatus): void {
  const now = performance.now();
  const sig = `${s.frame}|${s.playing}|${s.ready}|${s.waitMs > 0}|${s.fps.toFixed(0)}`;
  if (now - lastStatusAt > 250 || sig !== lastStatusSig) {
    lastStatusAt = now;
    lastStatusSig = sig;
    setView({ status: { ...s } });
  }
  const preview = view.preview;
  const atEnd = !!preview && s.frame >= preview.duration - 1;
  if (view.loop && preview?.kind === 'motion' && lastPlaying && !s.playing && atEnd && !replaying) {
    replaying = true;
    setTimeout(() => {
      replaying = false;
      if (view.loop && view.preview?.kind === 'motion' && engine && !engine.clock.playing) {
        engine.play().catch((e) => console.warn('[motions] 自动重播失败', e));
      }
    }, 30);
  }
  lastPlaying = s.playing;
}

async function selectAsset(
  kind: AssetKind,
  file: string,
  opts: { loop?: boolean; frame?: number; preserveFrame?: boolean; withIdle?: boolean } = {},
): Promise<void> {
  const run = async (): Promise<void> => {
    const cat = view.catalog;
    const eng = engine;
    if (!cat || !eng) throw new Error('查看器尚未就绪');
    const entry = findAsset(cat, kind, file);
    if (!entry) throw new Error(`没有找到${kind === 'motion' ? '动作' : '表情'}：${file}`);

    const currentFrame = eng.scene ? eng.frame : 0;
    const resume = opts.preserveFrame ? currentFrame : Math.max(0, Math.floor(opts.frame ?? 0));
    const wasPlaying = !!eng.clock?.playing;
    if (eng.scene) {
      eng.pause();
      eng.stop();
    }

    const same = view.selected?.kind === kind && view.selected.file === file && view.preview?.kind === 'motion';
    const sameAsset = view.selected?.kind === kind && view.selected.file === file;
    const defaultLoop = kind === 'motion' ? (entry as MotionAssetInfo).meta.loop : false;
    const loop = kind === 'motion' ? (opts.loop ?? (same ? view.loop : defaultLoop)) : false;
    const idleEntry = cat.motions.find((m) => m.file === 'IDLE.motion3.json');
    const withIdleWanted = kind === 'expression' ? (opts.withIdle ?? (sameAsset ? view.withIdle : true)) : false;
    const withIdle = withIdleWanted && !!idleEntry;

    setView({ loadingPreview: true, error: null, loop, withIdle, status: null, stage: '构造临时预览场景' });
    const sceneInput = buildPreviewScene({
      kind,
      file: entry.file,
      uri: entry.uri,
      motionMeta: kind === 'motion' ? (entry as MotionAssetInfo).meta : undefined,
      loop,
      idleMotionUri: withIdle && idleEntry ? idleEntry.uri : undefined,
    });
    const scene: Scene = await eng.load(sceneInput, (stage) => setView({ stage }));
    const canvas = eng.runtime.layerInstance('model')?.canvas;
    if (canvas) {
      const t = fitTransform(canvas, scene.composition.width, scene.composition.height);
      (scene.layers[0] as unknown as { transform: typeof t }).transform = t;
      eng.runtime.invalidate();
      eng.redraw();
    }

    const preview: PreviewInfo = {
      kind,
      file: entry.file,
      name: entry.name,
      uri: entry.uri,
      duration: previewDuration(kind, kind === 'motion' ? (entry as MotionAssetInfo).meta : undefined),
      fps: PREVIEW_FPS,
      composition: { width: scene.composition.width, height: scene.composition.height },
      loop,
      withIdle,
      source: entry.source,
      size: entry.size,
      mtime: entry.mtime,
      meta: kind === 'motion' ? (entry as MotionAssetInfo).meta : undefined,
      info: kind === 'expression' ? (entry as ExpressionAssetInfo).info : undefined,
    };
    setView({ selected: { kind, file: entry.file }, preview, loadingPreview: false, stage: '预览就绪' });

    // rAF 循环要先跑起来，seekAndRender 才能等到新帧上屏。
    eng.start();
    await seekAndRender(resume);
    if (wasPlaying) await eng.play();
    updateUrl(kind, entry.file, loop, withIdle);
  };

  const next = selectQueue.then(run, run);
  selectQueue = next.catch(() => {});
  return next;
}

function setLoop(loop: boolean): Promise<void> {
  const p = view.preview;
  const selected = view.selected;
  if (!p || p.kind !== 'motion' || !selected) return Promise.resolve();
  return selectAsset('motion', selected.file, { loop, preserveFrame: true });
}

function setWithIdle(withIdle: boolean): Promise<void> {
  const p = view.preview;
  const selected = view.selected;
  if (!p || p.kind !== 'expression' || !selected) return Promise.resolve();
  return selectAsset('expression', selected.file, { withIdle, preserveFrame: true });
}

/** 当前合成结果的像素摘要（测试/后续代理用，不依赖 canvas 截图）。 */
async function hashFrame(): Promise<string> {
  const eng = engine;
  if (!eng?.scene) throw new Error('尚未加载预览场景');
  return sha256Hex(eng.runtime.readPixels(undefined, 0.5).data);
}

const viewer: MotionViewerApi = {
  ready,
  readyState: 'booting',
  error: null,
  catalog: () => view.catalog,
  warnings: () => view.warnings,
  preview: () => view.preview,
  selected: () => view.selected,
  select: (kind, file) => selectAsset(kind, file),
  seekAndRender,
  hashFrame,
  thumbnails: () => view.thumbnails,
  status: () => view.status,
  setLoop,
  setWithIdle,
};
window.__motionViewer = viewer;

async function boot(canvas: HTMLCanvasElement): Promise<void> {
  if (booted) return;
  booted = true;
  try {
    setView({ phase: 'booting', stage: '读取素材目录' });
    const cat = await loadCatalog((stage) => setView({ stage }));
    const cachedThumbnails = readThumbnailCache(cat);
    setView({ catalog: cat, warnings: cat.warnings, thumbnails: cachedThumbnails });
    setView({ stage: '初始化 WebGL 与 Cubism Core' });
    const eng = new PlayerEngine();
    engine = eng;
    window.__player = eng;
    await eng.init(canvas, 0);
    eng.onStatus = onEngineStatus;

    if (!cachedThumbnails && (cat.motions.length || cat.expressions.length)) {
      setView({ stage: `生成素材缩略图（${cat.motions.length + cat.expressions.length} 个）` });
      try {
        const images = await generateThumbnails(eng, cat, (stage) => setView({ stage }));
        setView({ thumbnails: images });
        writeThumbnailCache(cat, images);
      } catch (e) {
        setView({ warnings: [...view.warnings, `缩略图生成失败：${errorMessage(e)}`] });
      }
    }

    const requestedKind: AssetKind = params.get('kind') === 'expression' ? 'expression' : 'motion';
    const requested = params.get('name') ? findAsset(cat, requestedKind, params.get('name')) : undefined;
    // 缺省预览 IDLE；没有 IDLE 时用排序后的第一个动作。kind=expression 时缺省第一个表情。
    const fallback =
      requestedKind === 'expression'
        ? cat.expressions[0]
        : cat.motions.find((a) => a.file === 'IDLE.motion3.json') ?? cat.motions[0];
    const entry = requested ?? fallback ?? cat.motions[0] ?? cat.expressions[0];
    if (entry) {
      const frameParam = params.has('frame') ? Number(params.get('frame')) : 0;
      const loopParam = params.get('loop') === '0' ? false : undefined;
      const idleParam = params.get('idle') === '0' ? false : undefined;
      await selectAsset(entry.kind, entry.file, {
        frame: Number.isFinite(frameParam) ? frameParam : 0,
        loop: entry.kind === 'motion' ? loopParam : undefined,
        withIdle: entry.kind === 'expression' ? idleParam : undefined,
      });
    }
    setView({ phase: 'ready', stage: '就绪' });
    viewer.readyState = 'ready';
  } catch (e) {
    const err = explainError(e);
    setView({ phase: 'error', stage: '启动失败', error: err, loadingPreview: false });
    viewer.readyState = 'error';
    viewer.error = err.detail;
  }
  readyResolve();
}

function useViewerState(): ViewerState {
  const [state, setState] = useState(view);
  useEffect(() => {
    const listener = (s: ViewerState) => setState(s);
    listeners.add(listener);
    setState(view);
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return state;
}

function hardBadge(label: string) {
  return <span className="tag">{label}</span>;
}

function AssetList(props: {
  kind: AssetKind;
  assets: (MotionAssetInfo | ExpressionAssetInfo)[];
  selected: ViewerState['selected'];
  thumbnails: Record<string, string> | null;
  onSelect: (kind: AssetKind, file: string) => void;
}) {
  const { kind, assets, selected, thumbnails, onSelect } = props;
  if (!assets.length) return <div className="empty small muted">没有可用的{kind === 'motion' ? '动作' : '表情'}素材。</div>;
  return (
    <div className="asset-list">
      {assets.map((a) => {
        const active = selected?.kind === kind && selected.file === a.file;
        const thumb = thumbnails?.[assetKey(a)];
        return (
          <button
            key={a.file}
            type="button"
            className={`asset-item${active ? ' active' : ''}`}
            data-kind={kind}
            data-asset={a.file}
            onClick={() => onSelect(kind, a.file)}
            title={a.uri}
          >
            <span className={`asset-thumb${thumb ? ' has-image' : ''}`}>
              {thumb ? (
                <img src={thumb} alt="" loading="lazy" />
              ) : (
                <span className="thumb-fallback" aria-hidden="true">
                  {kind === 'motion' ? '动' : '表'}
                </span>
              )}
            </span>
            <span className="asset-text">
              <span className="asset-name">{a.name}</span>
              <span className="asset-meta">
                {a.kind === 'motion'
                  ? `${a.meta.duration.toFixed(3)}s · ${a.meta.loop ? 'loop' : 'once'} · ${a.meta.curveCount} 曲线`
                  : `${a.info.parameters.length} 参数 · ${[...new Set(a.info.parameters.map((p) => p.blend))].join('/') || '—'}`}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

function Gallery(props: {
  assets: CatalogAsset[];
  thumbnails: Record<string, string> | null;
  onSelect: (kind: AssetKind, file: string) => void;
  onClose: () => void;
}) {
  const { assets, thumbnails, onSelect, onClose } = props;
  return (
    <div className="gallery-backdrop" role="dialog" aria-modal="true" aria-label="素材图集" onClick={onClose}>
      <div className="gallery-panel" onClick={(e) => e.stopPropagation()}>
        <header className="gallery-head">
          <strong>素材图集</strong>
          <span className="muted small">{assets.length} 个素材 · 点击缩略图切换预览</span>
          <span className="spacer" />
          <button type="button" onClick={onClose} title="Esc">
            关闭
          </button>
        </header>
        <div className="gallery-grid">
          {assets.map((a) => {
            const thumb = thumbnails?.[assetKey(a)];
            return (
              <button
                key={assetKey(a)}
                type="button"
                className="gallery-item"
                data-gallery-kind={a.kind}
                data-gallery-asset={a.file}
                onClick={() => {
                  onSelect(a.kind, a.file);
                  onClose();
                }}
                title={a.uri}
              >
                <span className="gallery-thumb">
                  {thumb ? <img src={thumb} alt="" loading="lazy" /> : <span className="thumb-fallback">无缩略图</span>}
                </span>
                <span className="gallery-name">{a.name}</span>
                <span className="gallery-meta">
                  {a.kind === 'motion' ? `动作 · ${a.meta.duration.toFixed(2)}s` : `表情 · ${a.info.parameters.length} 参数`}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function MetaPanel(props: { preview: PreviewInfo }) {
  const { preview } = props;
  const motion = preview.kind === 'motion';
  const bytes = (n: number) => (n > 0 ? `${n.toLocaleString()} B` : '—');
  return (
    <div className="meta-panel">
      <div className="meta-head">
        <strong>{preview.name}</strong>
        {motion && hardBadge(preview.loop ? 'loop' : 'once')}
        {hardBadge(preview.source === 'declared' ? 'model3 声明' : '目录扫描')}
      </div>
      <dl className="meta-grid">
        <div>
          <dt>类型</dt>
          <dd>{motion ? 'motion3 动作' : 'exp3 表情'}</dd>
        </div>
        <div>
          <dt>{motion ? '时长' : '预览时长'}</dt>
          <dd>{motion ? `${preview.meta!.duration.toFixed(3)} s` : `${preview.duration / preview.fps} s（${preview.duration} 帧）`}</dd>
        </div>
        <div>
          <dt>场景帧率</dt>
          <dd>{preview.fps} fps</dd>
        </div>
        {motion ? (
          <>
            <div>
              <dt>loop</dt>
              <dd>{preview.meta!.loop ? 'true' : 'false'}</dd>
            </div>
            <div>
              <dt>曲线数</dt>
              <dd>{preview.meta!.curveCount}</dd>
            </div>
            <div>
              <dt>段 / 点</dt>
              <dd>
                {preview.meta!.totalSegmentCount} / {preview.meta!.totalPointCount}
              </dd>
            </div>
            <div>
              <dt>motion3 FPS</dt>
              <dd>{preview.meta!.fps}</dd>
            </div>
          </>
        ) : (
          <>
            <div>
              <dt>参数数</dt>
              <dd>{preview.info!.parameters.length}</dd>
            </div>
            <div>
              <dt>Blend</dt>
              <dd>{[...new Set(preview.info!.parameters.map((p) => p.blend))].join(' / ') || '—'}</dd>
            </div>
            <div>
              <dt>叠加 IDLE</dt>
              <dd>{preview.withIdle ? '是' : '否'}</dd>
            </div>
          </>
        )}
        <div>
          <dt>文件大小</dt>
          <dd>{bytes(preview.size)}</dd>
        </div>
        <div>
          <dt>修改时间</dt>
          <dd>{preview.mtime ? new Date(preview.mtime).toLocaleString() : '—'}</dd>
        </div>
        <div className="wide">
          <dt>文件</dt>
          <dd className="mono">{preview.uri}</dd>
        </div>
      </dl>
      {!motion && preview.info!.parameters.length > 0 && (
        <table className="param-table">
          <thead>
            <tr>
              <th>参数</th>
              <th>值</th>
              <th>Blend</th>
            </tr>
          </thead>
          <tbody>
            {preview.info!.parameters.map((p) => (
              <tr key={p.id}>
                <td className="mono">{p.id}</td>
                <td>{p.value}</td>
                <td>{p.blend}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const s = useViewerState();
  const [tab, setTab] = useState<AssetKind>('motion');
  const [query, setQuery] = useState('');
  const [gallery, setGallery] = useState(false);

  useEffect(() => {
    boot(canvasRef.current!);
  }, []);

  useEffect(() => {
    if (s.selected) setTab(s.selected.kind);
  }, [s.selected?.kind, s.selected?.file]);

  useEffect(() => {
    if (!gallery) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setGallery(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [gallery]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const eng = engine;
      if (gallery || !eng?.scene || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target as HTMLElement;
      if (
        (t instanceof HTMLInputElement && !['range', 'radio', 'checkbox', 'button', 'submit'].includes(t.type)) ||
        t instanceof HTMLSelectElement ||
        t instanceof HTMLTextAreaElement
      ) {
        return;
      }
      const n = e.shiftKey ? 10 : 1;
      let handled = true;
      if (e.key === ' ') void eng.togglePlay().catch((err) => console.warn('[motions]', err));
      else if (e.key === 'ArrowRight') eng.step(n);
      else if (e.key === 'ArrowLeft') eng.step(-n);
      else if (e.key === 'Home') eng.seek(0);
      else if (e.key === 'End') eng.seek(eng.duration - 1);
      else if (e.key === 'l' || e.key === 'L') void setLoop(!view.loop).catch((err) => setView({ error: explainError(err) }));
      else handled = false;
      if (handled) e.preventDefault();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === ' ' && (e.target as HTMLElement).tagName === 'BUTTON') e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [gallery]);

  const normalizedQuery = query.trim().toLowerCase();
  const motions = useMemo(
    () => (s.catalog?.motions ?? []).filter((a) => !normalizedQuery || a.name.toLowerCase().includes(normalizedQuery)),
    [s.catalog, normalizedQuery],
  );
  const expressions = useMemo(
    () => (s.catalog?.expressions ?? []).filter((a) => !normalizedQuery || a.name.toLowerCase().includes(normalizedQuery)),
    [s.catalog, normalizedQuery],
  );
  const assets = tab === 'motion' ? motions : expressions;
  const allAssets = useMemo(() => (s.catalog ? catalogAssets(s.catalog) : []), [s.catalog]);
  const selectedAsset = s.selected?.kind === tab ? findAsset(s.catalog!, tab, s.selected.file) : undefined;
  const loading = s.phase === 'booting' || s.loadingPreview;
  const frame = Math.min(s.status?.frame ?? 0, Math.max(0, (s.preview?.duration ?? 1) - 1));
  const last = Math.max(0, (s.preview?.duration ?? 1) - 1);
  const fps = s.preview?.fps ?? PREVIEW_FPS;

  function choose(kind: AssetKind, file: string): void {
    if (s.phase !== 'ready') return;
    void selectAsset(kind, file).catch((e) => setView({ error: explainError(e), loadingPreview: false }));
  }

  function togglePlay(): void {
    void engine?.togglePlay().catch((e) => console.warn('[motions]', e));
  }

  return (
    <div className="motions-app">
      <header>
        <strong>CharaForge 动作 / 表情素材查看器</strong>
        <span className="muted">DeepSeek-chan（大肥鱼）</span>
        {(s.catalog?.motions.length ?? 0) > 0 && (
          <span className="muted small">
            动作 {s.catalog!.motions.length} · 表情 {s.catalog!.expressions.length}
          </span>
        )}
        <span className="spacer" />
        <button
          type="button"
          className="gallery-btn"
          disabled={!s.thumbnails || allAssets.length === 0 || s.phase !== 'ready'}
          onClick={() => setGallery(true)}
          title={s.thumbnails ? '查看全部素材缩略图' : '缩略图生成中'}
        >
          图集 {s.thumbnails ? allAssets.length : ''}
        </button>
        <a href="index.html">← 返回编辑器</a>
      </header>

      {s.error && (
        <div className="error-box" role="alert">
          <strong>{s.error.title}</strong>
          {s.error.hint && <div>{s.error.hint}</div>}
          <details>
            <summary>详细信息</summary>
            <pre>{s.error.detail}</pre>
          </details>
        </div>
      )}
      {s.warnings.map((w) => (
        <div key={w} className="notice">
          {w}
        </div>
      ))}

      <div className="viewer-grid">
        <aside className="sidebar">
          <div className="tabs" role="tablist">
            <button type="button" className={tab === 'motion' ? 'active' : ''} onClick={() => setTab('motion')} role="tab" aria-selected={tab === 'motion'}>
              动作 <span className="muted">{s.catalog?.motions.length ?? 0}</span>
            </button>
            <button
              type="button"
              className={tab === 'expression' ? 'active' : ''}
              onClick={() => setTab('expression')}
              role="tab"
              aria-selected={tab === 'expression'}
            >
              表情 <span className="muted">{s.catalog?.expressions.length ?? 0}</span>
            </button>
          </div>
          <div className="source-line small muted">
            来源：{tab === 'motion' ? s.catalog?.sources.motions === 'declared' ? 'model3.json 声明' : 'Animation/ 目录' : s.catalog?.sources.expressions === 'declared' ? 'model3.json 声明' : 'Exp3/ 目录'}
          </div>
          <input className="search" type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={`搜索${tab === 'motion' ? '动作' : '表情'}名称…`} />
          {!s.catalog ? (
            <div className="empty small muted">读取素材目录…</div>
          ) : (
            <AssetList kind={tab} assets={assets} selected={s.selected} thumbnails={s.thumbnails} onSelect={choose} />
          )}
          <div className="small muted sidebar-note">
            点击素材即可在右侧预览。预览是临时 scene，不写盘，也不会修改全片场景。
          </div>
        </aside>

        <main className="preview-panel">
          <div className="canvas-box">
            <canvas ref={canvasRef} />
            {loading && (
              <div className="overlay">
                <div className="spinner" />
                <div>{s.stage}…</div>
              </div>
            )}
            {!loading && !s.preview && s.phase !== 'error' && <div className="overlay">选择左侧素材开始预览</div>}
          </div>

          <div className="transport">
            <button className="primary" type="button" disabled={!s.preview} onClick={togglePlay} title="空格">
              {s.status?.playing ? '❚❚ 暂停' : '▶ 播放'}
            </button>
            <button type="button" disabled={!s.preview} onClick={() => engine?.step(-10)} title="Shift+←">
              −10
            </button>
            <button type="button" disabled={!s.preview} onClick={() => engine?.step(-1)} title="←">
              −1
            </button>
            <button type="button" disabled={!s.preview} onClick={() => engine?.step(1)} title="→">
              +1
            </button>
            <button type="button" disabled={!s.preview} onClick={() => engine?.step(10)} title="Shift+→">
              +10
            </button>
            <input
              className="scrub"
              type="range"
              min={0}
              max={last}
              value={frame}
              disabled={!s.preview}
              onChange={(e) => engine?.seek(Number(e.target.value))}
            />
            <span className="mono">
              帧 {frame} / {last}　{timecode(frame, fps)}
            </span>
            <label className="inline" title="L">
              <input
                type="checkbox"
                checked={s.loop}
                disabled={s.preview?.kind !== 'motion'}
                onChange={(e) => void setLoop(e.target.checked).catch((err) => setView({ error: explainError(err), loadingPreview: false }))}
              />
              循环
            </label>
            <label className="inline" title="表情预览叠加 IDLE 动作">
              <input
                type="checkbox"
                checked={s.withIdle}
                disabled={s.preview?.kind !== 'expression'}
                onChange={(e) => void setWithIdle(e.target.checked).catch((err) => setView({ error: explainError(err), loadingPreview: false }))}
              />
              叠加 IDLE
            </label>
          </div>

          {s.preview ? (
            <MetaPanel preview={s.preview} />
          ) : (
            <div className="meta-panel small muted">
              当前模型：{PREVIEW_MODEL_URI}
              <br />
              动作来源：model3.json 的 FileReferences.Motions 优先，没有声明时扫描 Animation/；表情同理（Expressions 优先，否则 Exp3/）。
            </div>
          )}

          <div className="keys small muted">
            快捷键：<kbd>空格</kbd> 播放/暂停　<kbd>←</kbd>/<kbd>→</kbd> ±1 帧（<kbd>Shift</kbd> ±10）　<kbd>Home</kbd>/<kbd>End</kbd>　<kbd>L</kbd> 循环
          </div>
        </main>
      </div>

      {gallery && s.catalog && (
        <Gallery assets={allAssets} thumbnails={s.thumbnails} onSelect={choose} onClose={() => setGallery(false)} />
      )}

      <footer>
        <div>
          预览只使用 CharaForge runtime；不加载原参考视频 / 音频，不写 scene 文件。素材目录来自只读接口{' '}
          <code>/__charaforge/list-assets</code>（`pnpm dev` / `pnpm preview` 提供）。
        </div>
        <div className="muted small">
          模型：DeepSeek-chan — Illustration &amp; PSD Layer Separation: Kamiya Ryoko / Live2D Rigging: 霓虹猫_NeonNeko。
          模型按原作者条款禁止再分发，本页面只供本地查看。
        </div>
        {selectedAsset && <div className="muted small mono">{selectedAsset.uri}</div>}
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
