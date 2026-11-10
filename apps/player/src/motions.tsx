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
import { StrictMode, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
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
  bodyManifestUri,
  buildPreviewScene,
  buildThumbnailScene,
  catalogAssets,
  clampNumber,
  declaredAssetUri,
  declaredExpressionFiles,
  declaredMotionFiles,
  displayName,
  fitTransform,
  fitTransformToBounds,
  isExpressionFile,
  isMotionFile,
  modelAssetDir,
  modelNameFromUri,
  parseBodyManifest,
  parseExpressionInfo,
  parseMotionMeta,
  previewDuration,
  sortAssets,
  splitBodyPresets,
  type AssetFileInfo,
  type AssetKind,
  type AssetSource,
  type BodyAssetInfo,
  type Catalog,
  type CatalogAsset,
  type ExpressionAssetInfo,
  type ExpressionInfo,
  type FitTransform,
  type MotionAssetInfo,
  type MotionMeta,
} from './motionCatalog.ts';
import { pageUrl, sha256Hex, timecode } from './util.ts';

const params = new URLSearchParams(location.search);
const LIST_ENDPOINT = '__charaforge/list-assets';
/** ?model=... 可以查看 assets/models/ 下的任意模型；缺省仍是示例模型。 */
const MODEL_URI = params.get('model')?.trim() || PREVIEW_MODEL_URI;
const MODEL_NAME = params.get('modelName')?.trim() || modelNameFromUri(MODEL_URI);
const MODEL_ANIMATION_DIR = modelAssetDir(MODEL_URI, 'Animation') || ANIMATION_DIR;
const MODEL_EXP3_DIR = modelAssetDir(MODEL_URI, 'Exp3') || EXP3_DIR;

type Phase = 'booting' | 'ready' | 'error';

interface PreviewSelection {
  motion: string | null;
  expression: string | null;
  body: string | null;
}

/** 预览取景：scale 是模型画布像素 → 合成像素；anchor 是缩放中心（模型画布像素）。 */
interface PreviewViewport {
  scale: number;
  /** 自动适应时的 scale；缩放百分比相对它计算。 */
  fitScale: number;
  x: number;
  y: number;
  anchorX: number;
  anchorY: number;
}

interface PreviewInfo {
  /** 最后一次点击的素材（用于列表高亮和元数据面板）。 */
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
  /** 当前预览是否叠加了动作（无动作时只显示体型/表情的静态状态）。 */
  withIdle: boolean;
  /** 当前同时生效的体型 / 表情 / 动作。 */
  selection: PreviewSelection;
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
  /** 当前同时生效的体型 / 表情 / 动作（三者互不覆盖）。 */
  selection: PreviewSelection;
  viewport: PreviewViewport | null;
  preview: PreviewInfo | null;
  loadingPreview: boolean;
  loop: boolean;
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
  selection: () => PreviewSelection;
  select: (kind: AssetKind, file: string) => Promise<void>;
  setSelection: (patch: Partial<PreviewSelection>) => Promise<void>;
  setBody: (file: string | null) => Promise<void>;
  setExpression: (file: string | null) => Promise<void>;
  setMotion: (file: string | null) => Promise<void>;
  seekAndRender: (frame: number) => Promise<void>;
  hashFrame: () => Promise<string>;
  thumbnails: () => Record<string, string> | null;
  status: () => EngineStatus | null;
  viewport: () => PreviewViewport | null;
  fit: () => Promise<void>;
  zoomBy: (factor: number) => void;
  panBy: (dx: number, dy: number) => void;
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
  selection: { motion: null, expression: null, body: null },
  viewport: null,
  preview: null,
  loadingPreview: false,
  loop: false,
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
  const uri = declaredAssetUri(MODEL_URI, file);
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

/** 可选文件：404 视为没有，其他错误继续抛给调用方记 warning。 */
async function fetchOptionalJson(url: string): Promise<unknown | undefined> {
  const res = await fetch(asciiUrl(url));
  if (res.status === 404) return undefined;
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
    model3 = await fetchJson(assetUrl(MODEL_URI));
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
      const files = (await listDirectory(MODEL_ANIMATION_DIR)).filter((f) => isMotionFile(f.name));
      rawMotions = files.map((f) => rawFromDirectory(MODEL_ANIMATION_DIR, f));
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
      const files = (await listDirectory(MODEL_EXP3_DIR)).filter((f) => isExpressionFile(f.name));
      rawExpressions = files.map((f) => rawFromDirectory(MODEL_EXP3_DIR, f));
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

  // 可选的体型 manifest：显式列出哪些 exp3 是体型，并指定默认体型。
  // 没有 manifest 时由 splitBodyPresets 根据名称/参数推断。
  let manifest;
  try {
    const rawManifest = await fetchOptionalJson(assetUrl(bodyManifestUri(MODEL_URI)));
    if (rawManifest !== undefined) manifest = parseBodyManifest(rawManifest, MODEL_URI);
  } catch (e) {
    warnings.push(`体型 manifest 读取失败，将按名称/参数推断：${errorMessage(e)}`);
  }

  sortAssets(motions);
  sortAssets(expressions);
  const split = splitBodyPresets(expressions, manifest);
  sortAssets(split.expressions);
  sortAssets(split.bodies);
  if (manifest) {
    const known = new Set(expressions.map((asset) => asset.uri));
    for (const item of manifest.bodies) {
      if (!known.has(item.uri)) warnings.push(`体型 manifest 引用了不存在或未解析成功的表情：${item.uri}`);
    }
  }
  if (!motions.length && !expressions.length && warnings.length) throw new Error(warnings.join('\n'));
  return {
    modelUri: MODEL_URI,
    modelName: MODEL_NAME,
    motions,
    expressions: split.expressions,
    bodies: split.bodies,
    ...(split.defaultBody ? { defaultBody: split.defaultBody } : {}),
    sources: { motions: motionSource, expressions: expressionSource, bodies: expressionSource },
    warnings,
  };
}

const THUMBNAIL_CACHE_KEY = 'charaforge.motion-thumbs.v2';

function thumbnailCacheKey(cat: Catalog): string {
  const files = catalogAssets(cat).map((a) => `${a.kind}:${a.file}:${a.size}:${a.mtime}`);
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
  const fitTile = (i: number, scale: number, anchorX?: number, anchorY?: number) => {
    const col = i % THUMBNAIL_COLUMNS;
    const row = Math.floor(i / THUMBNAIL_COLUMNS);
    const layer = scene.layers[i]!;
    (layer as unknown as { transform: FitTransform }).transform = {
      x: col * THUMBNAIL_TILE + THUMBNAIL_TILE / 2,
      y: row * THUMBNAIL_TILE + THUMBNAIL_TILE / 2,
      scale,
      ...(anchorX !== undefined ? { anchorX } : {}),
      ...(anchorY !== undefined ? { anchorY } : {}),
    };
  };
  const fullScale = Math.min(THUMBNAIL_TILE / canvas.width, THUMBNAIL_TILE / canvas.height) * 0.86;
  assets.forEach((_, i) => fitTile(i, fullScale));
  eng.redraw();
  await eng.runtime.renderFrameAsync(0);

  // 先按整张画布渲染一次（让每个图层的体型/表情生效），再按各自内容包围盒收紧到 tile。
  assets.forEach((_, i) => {
    const inst = eng.runtime.layerInstance(scene.layers[i]!.id);
    const bounds = inst?.contentBounds();
    if (!bounds) return;
    const bw = bounds.x1 - bounds.x0;
    const bh = bounds.y1 - bounds.y0;
    if (!(bw > 0 && bh > 0)) return;
    const scale = Math.min(THUMBNAIL_TILE / bw, THUMBNAIL_TILE / bh) * 0.86;
    fitTile(i, scale, (bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2);
  });
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

function updateUrl(selection: PreviewSelection, selected: { kind: AssetKind; file: string } | null, loop: boolean): void {
  const q = new URLSearchParams(location.search);
  // kind/name 保留为“最后一次点击”，方便深链直接打开某个素材；三个槽位则各自独立。
  if (selected) {
    q.set('kind', selected.kind);
    q.set('name', selected.file);
  } else {
    q.delete('kind');
    q.delete('name');
  }
  if (selection.body) q.set('body', selection.body);
  else q.delete('body');
  if (selection.expression) q.set('expression', selection.expression);
  else q.delete('expression');
  if (selection.motion) q.set('motion', selection.motion);
  else q.set('motion', '0');
  q.delete('frame');
  q.delete('idle');
  if (!loop) q.set('loop', '0');
  else q.delete('loop');
  const search = q.toString();
  history.replaceState(null, '', `${location.pathname}${search ? `?${search}` : ''}${location.hash}`);
}

function findAsset(
  cat: Catalog,
  kind: AssetKind,
  file: string | null,
): MotionAssetInfo | ExpressionAssetInfo | BodyAssetInfo | undefined {
  const list =
    kind === 'motion' ? cat.motions : kind === 'expression' ? cat.expressions : cat.bodies;
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
  const hasMotion = !!view.selection.motion;
  if (view.loop && hasMotion && lastPlaying && !s.playing && atEnd && !replaying) {
    replaying = true;
    setTimeout(() => {
      replaying = false;
      if (view.loop && view.selection.motion && engine && !engine.clock.playing) {
        engine.play().catch((e) => console.warn('[motions] 自动重播失败', e));
      }
    }, 30);
  }
  lastPlaying = s.playing;
}

/** 按模型“实际内容”取景；没有可见 drawable 时退回整张画布。 */
function fitViewport(scene: Scene, eng: PlayerEngine, margin = 0.92): PreviewViewport {
  const inst = eng.runtime.layerInstance('model');
  if (!inst) throw new Error('预览 scene 没有模型实例');
  const W = scene.composition.width;
  const H = scene.composition.height;
  const bounds = inst.contentBounds();
  if (bounds) {
    const fit = fitTransformToBounds(bounds, W, H, margin);
    return {
      scale: fit.scale,
      fitScale: fit.scale,
      x: fit.x,
      y: fit.y,
      anchorX: fit.anchorX ?? inst.canvas.width / 2,
      anchorY: fit.anchorY ?? inst.canvas.height / 2,
    };
  }
  const fit = fitTransform(inst.canvas, W, H, 0.96);
  return {
    scale: fit.scale,
    fitScale: fit.scale,
    x: fit.x,
    y: fit.y,
    anchorX: inst.canvas.width / 2,
    anchorY: inst.canvas.height / 2,
  };
}

/** 把预览取景写入模型图层的 transform（只影响查看器，不写盘）。 */
function applyViewport(scene: Scene, vp: PreviewViewport): void {
  const layer = scene.layers.find((item) => item.kind === 'model');
  if (!layer) return;
  (layer as unknown as { transform: FitTransform }).transform = {
    x: vp.x,
    y: vp.y,
    scale: vp.scale,
    anchorX: vp.anchorX,
    anchorY: vp.anchorY,
  };
}

interface ReloadOptions {
  selection?: Partial<PreviewSelection>;
  /** 最后一次点击的素材；不传则保留当前选择。 */
  selected?: { kind: AssetKind; file: string } | null;
  loop?: boolean;
  frame?: number;
  preserveFrame?: boolean;
  /** true = 重新按内容取景；false = 保留当前视角。缺省在体型变化或首次加载时重新取景。 */
  fit?: boolean;
}

async function reloadPreview(opts: ReloadOptions = {}): Promise<void> {
  const run = async (): Promise<void> => {
    const cat = view.catalog;
    const eng = engine;
    if (!cat || !eng) throw new Error('查看器尚未就绪');
    const previous = view.selection;
    const selection: PreviewSelection = {
      motion: opts.selection && 'motion' in opts.selection ? opts.selection.motion ?? null : previous.motion,
      expression: opts.selection && 'expression' in opts.selection ? opts.selection.expression ?? null : previous.expression,
      body: opts.selection && 'body' in opts.selection ? opts.selection.body ?? null : previous.body,
    };
    const motionEntry = selection.motion ? cat.motions.find((item) => item.file === selection.motion) : undefined;
    const bodyEntry = selection.body ? cat.bodies.find((item) => item.file === selection.body) : undefined;
    const expressionEntry = selection.expression ? cat.expressions.find((item) => item.file === selection.expression) : undefined;
    if (selection.motion && !motionEntry) throw new Error(`没有找到动作：${selection.motion}`);
    if (selection.body && !bodyEntry) throw new Error(`没有找到体型：${selection.body}`);
    if (selection.expression && !expressionEntry) throw new Error(`没有找到表情：${selection.expression}`);

    const bodyChanged = selection.body !== previous.body;
    const motionChanged = selection.motion !== previous.motion;
    const currentFrame = eng.scene ? eng.frame : 0;
    const resume = opts.preserveFrame ? currentFrame : Math.max(0, Math.floor(opts.frame ?? 0));
    const wasPlaying = !!eng.clock?.playing;
    if (eng.scene) {
      eng.pause();
      eng.stop();
    }

    const loop = opts.loop ?? (motionChanged ? (motionEntry?.meta.loop ?? false) : view.loop);
    setView({ loadingPreview: true, error: null, loop, status: null, stage: '构造临时预览场景' });
    const sceneInput = buildPreviewScene({
      modelUri: cat.modelUri,
      modelName: cat.modelName,
      motion: motionEntry ? { file: motionEntry.file, uri: motionEntry.uri, meta: motionEntry.meta } : undefined,
      body: bodyEntry ? { file: bodyEntry.file, uri: bodyEntry.uri } : undefined,
      expression: expressionEntry ? { file: expressionEntry.file, uri: expressionEntry.uri } : undefined,
      loop,
    });
    const scene: Scene = await eng.load(sceneInput, (stage) => setView({ stage }));

    // rAF 循环要先跑起来，seekAndRender 才能等到新帧上屏。
    eng.start();
    await seekAndRender(resume);

    const shouldFit = opts.fit ?? (bodyChanged || !view.viewport);
    const viewport = shouldFit ? fitViewport(scene, eng) : view.viewport!;
    applyViewport(scene, viewport);
    eng.redraw();
    await seekAndRender(resume);

    const selected = opts.selected === undefined ? view.selected : opts.selected;
    const entry = selected ? findAsset(cat, selected.kind, selected.file) : undefined;
    const preview: PreviewInfo = {
      kind: selected?.kind ?? motionEntry?.kind ?? 'motion',
      file: entry?.file ?? motionEntry?.file ?? '',
      name: entry?.name ?? motionEntry?.name ?? '',
      uri: entry?.uri ?? motionEntry?.uri ?? '',
      duration: previewDuration(motionEntry ? 'motion' : 'expression', motionEntry?.meta),
      fps: PREVIEW_FPS,
      composition: { width: scene.composition.width, height: scene.composition.height },
      loop,
      withIdle: !!motionEntry,
      selection,
      source: entry?.source ?? motionEntry?.source ?? 'directory',
      size: entry?.size ?? 0,
      mtime: entry?.mtime ?? '',
      ...(selected?.kind === 'motion' && entry ? { meta: (entry as MotionAssetInfo).meta } : {}),
      ...(selected?.kind === 'expression' && entry ? { info: (entry as ExpressionAssetInfo).info } : {}),
      ...(selected?.kind === 'body' && entry ? { info: (entry as BodyAssetInfo).info } : {}),
    };
    setView({ selection, viewport, selected: selected ?? null, preview, loadingPreview: false, stage: '预览就绪' });
    if (wasPlaying) await eng.play();
    updateUrl(selection, selected ?? null, loop);
  };

  const next = selectQueue.then(run, run);
  selectQueue = next.catch(() => {});
  return next;
}

/** 点击左侧素材：只改对应槽位，体型 / 表情 / 动作互不覆盖。 */
async function selectAsset(
  kind: AssetKind,
  file: string,
  opts: { loop?: boolean; frame?: number; preserveFrame?: boolean } = {},
): Promise<void> {
  const cat = view.catalog;
  if (!cat) throw new Error('查看器尚未就绪');
  const entry = findAsset(cat, kind, file);
  if (!entry) throw new Error(`没有找到${kind === 'motion' ? '动作' : kind === 'body' ? '体型' : '表情'}：${file}`);
  const patch: Partial<PreviewSelection> =
    kind === 'motion' ? { motion: file } : kind === 'body' ? { body: file } : { expression: file };
  return reloadPreview({
    selection: patch,
    selected: { kind, file },
    ...(opts.loop !== undefined ? { loop: opts.loop } : {}),
    ...(opts.frame !== undefined ? { frame: opts.frame } : {}),
    ...(opts.preserveFrame ? { preserveFrame: true } : {}),
  });
}

async function setSelection(patch: Partial<PreviewSelection>): Promise<void> {
  return reloadPreview({ selection: patch, selected: view.selected });
}

function setLoop(loop: boolean): Promise<void> {
  if (!view.selection.motion) return Promise.resolve();
  return reloadPreview({ loop, preserveFrame: true });
}

function setWithIdle(withIdle: boolean): Promise<void> {
  const cat = view.catalog;
  if (!cat) return Promise.resolve();
  if (withIdle) {
    const idle = cat.motions.find((item) => item.file === 'IDLE.motion3.json') ?? cat.motions[0];
    return reloadPreview({ selection: { motion: idle?.file ?? null }, preserveFrame: true });
  }
  return reloadPreview({ selection: { motion: null }, preserveFrame: true });
}

/** 在当前帧按模型内容重新取景（“适应”按钮 / 双击画面）。 */
async function fitNow(): Promise<void> {
  const eng = engine;
  const scene = eng?.scene;
  if (!eng || !scene) return;
  const viewport = fitViewport(scene, eng);
  applyViewport(scene, viewport);
  eng.redraw();
  setView({ viewport });
  await seekAndRender(eng.frame);
}

/** 以合成画面中的某个点为锚缩放；point 缺省为画面中心。 */
function zoomBy(factor: number, point?: { x: number; y: number }): void {
  const eng = engine;
  const scene = eng?.scene;
  const vp = view.viewport;
  if (!eng || !scene || !vp) return;
  const W = scene.composition.width;
  const H = scene.composition.height;
  const px = point?.x ?? W / 2;
  const py = point?.y ?? H / 2;
  const minScale = vp.fitScale * 0.1;
  const maxScale = vp.fitScale * 20;
  const scale = clampNumber(vp.scale * factor, minScale, maxScale);
  const k = scale / vp.scale;
  const next: PreviewViewport = {
    ...vp,
    scale,
    x: px - (px - vp.x) * k,
    y: py - (py - vp.y) * k,
  };
  applyViewport(scene, next);
  eng.redraw();
  setView({ viewport: next });
}

function panBy(dx: number, dy: number): void {
  const eng = engine;
  const scene = eng?.scene;
  const vp = view.viewport;
  if (!eng || !scene || !vp) return;
  const next: PreviewViewport = { ...vp, x: vp.x + dx, y: vp.y + dy };
  applyViewport(scene, next);
  eng.redraw();
  setView({ viewport: next });
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
  selection: () => view.selection,
  select: (kind, file) => selectAsset(kind, file),
  setSelection,
  setBody: (file) => (file ? selectAsset('body', file) : setSelection({ body: null })),
  setExpression: (file) => (file ? selectAsset('expression', file) : setSelection({ expression: null })),
  setMotion: (file) => (file ? selectAsset('motion', file) : setSelection({ motion: null })),
  seekAndRender,
  hashFrame,
  thumbnails: () => view.thumbnails,
  status: () => view.status,
  viewport: () => view.viewport,
  fit: fitNow,
  zoomBy,
  panBy,
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

    const allAssets = catalogAssets(cat);
    if (!cachedThumbnails && allAssets.length) {
      setView({ stage: `生成素材缩略图（${allAssets.length} 个）` });
      try {
        const images = await generateThumbnails(eng, cat, (stage) => setView({ stage }));
        setView({ thumbnails: images });
        writeThumbnailCache(cat, images);
      } catch (e) {
        setView({ warnings: [...view.warnings, `缩略图生成失败：${errorMessage(e)}`] });
      }
    }

    const kindParam = params.get('kind');
    const requestedKind: AssetKind = kindParam === 'expression' || kindParam === 'body' ? kindParam : 'motion';
    const nameParam = params.get('name');
    const requested = nameParam
      ? findAsset(cat, requestedKind, nameParam) ??
        findAsset(cat, 'body', nameParam) ??
        findAsset(cat, 'expression', nameParam) ??
        findAsset(cat, 'motion', nameParam)
      : undefined;
    const idleEntry = cat.motions.find((a) => a.file === 'IDLE.motion3.json') ?? cat.motions[0];

    const motionParam = params.get('motion');
    const idleOff = params.get('idle') === '0';
    let motionFile: string | null =
      motionParam === '0' || idleOff ? null : motionParam ? findAsset(cat, 'motion', motionParam)?.file ?? null : idleEntry?.file ?? null;
    let bodyFile: string | null = params.get('body')
      ? findAsset(cat, 'body', params.get('body'))?.file ?? null
      : cat.defaultBody ?? cat.bodies[0]?.file ?? null;
    let expressionFile: string | null = params.get('expression')
      ? findAsset(cat, 'expression', params.get('expression'))?.file ?? null
      : null;

    // kind/name 深链：显式槽位参数优先；只有对应槽位没写时才用 kind/name 的兼容入口。
    if (requested) {
      if (requested.kind === 'motion' && !params.has('motion')) motionFile = requested.file;
      else if (requested.kind === 'body' && !params.has('body')) bodyFile = requested.file;
      else if (requested.kind === 'expression' && !params.has('expression')) expressionFile = requested.file;
    }
    const selection: PreviewSelection = { motion: motionFile, expression: expressionFile, body: bodyFile };
    const selected = requested ?? (motionFile ? { kind: 'motion' as const, file: motionFile } : bodyFile ? { kind: 'body' as const, file: bodyFile } : null);
    if (catalogAssets(cat).length) {
      const frameParam = params.has('frame') ? Number(params.get('frame')) : 0;
      const loopParam = params.get('loop') === '0' ? false : undefined;
      await reloadPreview({
        selection,
        selected,
        frame: Number.isFinite(frameParam) ? frameParam : 0,
        ...(loopParam !== undefined ? { loop: loopParam } : {}),
        fit: true,
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

function assetKindLabel(kind: AssetKind, short = false): string {
  if (kind === 'motion') return short ? '动' : '动作';
  if (kind === 'body') return short ? '体' : '体型';
  return short ? '表' : '表情';
}

function AssetList(props: {
  kind: AssetKind;
  assets: CatalogAsset[];
  activeFile: string | null;
  thumbnails: Record<string, string> | null;
  onSelect: (kind: AssetKind, file: string) => void;
}) {
  const { kind, assets, activeFile, thumbnails, onSelect } = props;
  if (!assets.length) return <div className="empty small muted">没有可用的{assetKindLabel(kind)}素材。</div>;
  return (
    <div className="asset-list">
      {assets.map((a) => {
        const active = activeFile === a.file;
        const thumb = thumbnails?.[assetKey(a)];
        return (
          <button
            key={a.file}
            type="button"
            className={`asset-item${active ? ' active' : ''}${kind === 'body' ? ' body-item' : ''}`}
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
                  {assetKindLabel(kind, true)}
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
                  {a.kind === 'motion'
                    ? `动作 · ${a.meta.duration.toFixed(2)}s`
                    : `${assetKindLabel(a.kind)} · ${a.info.parameters.length} 参数`}
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
  const body = preview.kind === 'body';
  const bytes = (n: number) => (n > 0 ? `${n.toLocaleString()} B` : '—');
  const kindLabel = motion ? 'motion3 动作' : body ? 'exp3 体型（基础层）' : 'exp3 表情';
  return (
    <div className="meta-panel">
      <div className="meta-head">
        <strong>{preview.name}</strong>
        {motion && hardBadge(preview.loop ? 'loop' : 'once')}
        {body && hardBadge('体型基础层')}
        {hardBadge(preview.source === 'declared' ? 'model3 声明' : '目录扫描')}
      </div>
      <div className="combo-summary">
        <span>
          <em>体型</em> {preview.selection.body ?? '默认'}
        </span>
        <span>
          <em>表情</em> {preview.selection.expression ?? '无'}
        </span>
        <span>
          <em>动作</em> {preview.selection.motion ?? '无'}
        </span>
      </div>
      <dl className="meta-grid">
        <div>
          <dt>类型</dt>
          <dd>{kindLabel}</dd>
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
              <dt>叠加动作</dt>
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
      {body && (
        <div className="small muted">
          体型是 exp3 基础参数层，会与表情、动作同时生效。当前实现按“内容包围盒”自动取景，因此换小体型后角色仍会占满预览；
          也可以滚轮缩放、拖动平移。
        </div>
      )}
    </div>
  );
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasBoxRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ pointerId: number; lastX: number; lastY: number } | null>(null);
  const [panning, setPanning] = useState(false);
  const s = useViewerState();
  const [tab, setTab] = useState<AssetKind>('motion');
  const [query, setQuery] = useState('');
  const [gallery, setGallery] = useState(false);

  useEffect(() => {
    boot(canvasRef.current!);
  }, []);

  // 滚轮缩放：用原生监听器才能 preventDefault，避免页面跟着滚动。
  useEffect(() => {
    const box = canvasBoxRef.current;
    const canvas = canvasRef.current;
    if (!box || !canvas) return;
    const onWheel = (event: WheelEvent) => {
      const scene = engine?.scene;
      if (!scene || !view.viewport) return;
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      if (!(rect.width > 0 && rect.height > 0)) return;
      const point = {
        x: ((event.clientX - rect.left) / rect.width) * scene.composition.width,
        y: ((event.clientY - rect.top) / rect.height) * scene.composition.height,
      };
      zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, point);
    };
    box.addEventListener('wheel', onWheel, { passive: false });
    return () => box.removeEventListener('wheel', onWheel);
  }, []);

  function canvasPoint(event: { clientX: number; clientY: number }): { x: number; y: number } | null {
    const scene = engine?.scene;
    const canvas = canvasRef.current;
    if (!scene || !canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    return {
      x: ((event.clientX - rect.left) / rect.width) * scene.composition.width,
      y: ((event.clientY - rect.top) / rect.height) * scene.composition.height,
    };
  }

  function onCanvasPointerDown(event: ReactPointerEvent<HTMLDivElement>): void {
    if (loading || !view.viewport) return;
    if (event.button !== 0 && event.button !== 1) return;
    panRef.current = { pointerId: event.pointerId, lastX: event.clientX, lastY: event.clientY };
    setPanning(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function onCanvasPointerMove(event: ReactPointerEvent<HTMLDivElement>): void {
    const pan = panRef.current;
    const point = canvasPoint(event);
    const scene = engine?.scene;
    if (!pan || !point || !scene) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const dx = ((event.clientX - pan.lastX) / rect.width) * scene.composition.width;
    const dy = ((event.clientY - pan.lastY) / rect.height) * scene.composition.height;
    pan.lastX = event.clientX;
    pan.lastY = event.clientY;
    panBy(dx, dy);
  }

  function endPan(event: ReactPointerEvent<HTMLDivElement>): void {
    if (!panRef.current) return;
    panRef.current = null;
    setPanning(false);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      /* 指针已经释放 */
    }
  }

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
      else if (e.key === 'f' || e.key === 'F' || e.key === '0') void fitNow().catch((err) => console.warn('[motions]', err));
      else if (e.key === '+' || e.key === '=') zoomBy(1.12);
      else if (e.key === '-' || e.key === '_') zoomBy(1 / 1.12);
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
  const bodies = useMemo(
    () => (s.catalog?.bodies ?? []).filter((a) => !normalizedQuery || a.name.toLowerCase().includes(normalizedQuery)),
    [s.catalog, normalizedQuery],
  );
  const assets = tab === 'motion' ? motions : tab === 'expression' ? expressions : bodies;
  const activeFile = s.selection[tab];
  const allAssets = useMemo(() => (s.catalog ? catalogAssets(s.catalog) : []), [s.catalog]);
  const selectedAsset = s.selected ? findAsset(s.catalog!, s.selected.kind, s.selected.file) : undefined;
  const loading = s.phase === 'booting' || s.loadingPreview;
  const frame = Math.min(s.status?.frame ?? 0, Math.max(0, (s.preview?.duration ?? 1) - 1));
  const last = Math.max(0, (s.preview?.duration ?? 1) - 1);
  const fps = s.preview?.fps ?? PREVIEW_FPS;
  const zoomPct = s.viewport ? Math.round((s.viewport.scale / s.viewport.fitScale) * 100) : 100;

  function choose(kind: AssetKind, file: string): void {
    if (s.phase !== 'ready') return;
    void selectAsset(kind, file).catch((e) => setView({ error: explainError(e), loadingPreview: false }));
  }

  function chooseSlot(patch: Partial<PreviewSelection>): void {
    if (s.phase !== 'ready') return;
    setTab(patch.body !== undefined ? 'body' : patch.expression !== undefined ? 'expression' : 'motion');
    void setSelection(patch).catch((e) => setView({ error: explainError(e), loadingPreview: false }));
  }

  function togglePlay(): void {
    void engine?.togglePlay().catch((e) => console.warn('[motions]', e));
  }

  return (
    <div className="motions-app">
      <header>
        <strong>CharaForge 动作 / 表情 / 体型查看器</strong>
        <span className="muted" title={s.catalog?.modelUri ?? MODEL_URI}>
          {s.catalog?.modelName ?? MODEL_NAME}
        </span>
        {(s.catalog?.motions.length ?? 0) > 0 && (
          <span className="muted small">
            动作 {s.catalog!.motions.length} · 表情 {s.catalog!.expressions.length} · 体型 {s.catalog!.bodies.length}
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
            <button
              type="button"
              className={tab === 'body' ? 'active' : ''}
              onClick={() => setTab('body')}
              role="tab"
              aria-selected={tab === 'body'}
            >
              体型 <span className="muted">{s.catalog?.bodies.length ?? 0}</span>
            </button>
          </div>
          <div className="source-line small muted">
            来源：
            {tab === 'motion'
              ? s.catalog?.sources.motions === 'declared'
                ? 'model3.json 声明'
                : 'Animation/ 目录'
              : tab === 'body'
                ? 'Exp3/ 目录 + charaforge.bodies.json（可选）'
                : s.catalog?.sources.expressions === 'declared'
                  ? 'model3.json 声明'
                  : 'Exp3/ 目录'}
          </div>
          <input className="search" type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={`搜索${assetKindLabel(tab)}名称…`} />
          {!s.catalog ? (
            <div className="empty small muted">读取素材目录…</div>
          ) : (
            <AssetList kind={tab} assets={assets} activeFile={activeFile} thumbnails={s.thumbnails} onSelect={choose} />
          )}
          <div className="small muted sidebar-note">
            {tab === 'body'
              ? '体型是独立的基础参数层：选好体型后，再切表情或动作不会把它覆盖掉。'
              : '点击素材会更新对应槽位；右侧的体型 / 表情 / 动作可以同时生效。预览是临时 scene，不写盘。'}
          </div>
        </aside>

        <main className="preview-panel">
          <div className="combo-bar">
            <label className="combo-field">
              <span>体型</span>
              <select
                value={s.selection.body ?? ''}
                disabled={!s.catalog || loading}
                onChange={(e) => chooseSlot({ body: e.target.value || null })}
              >
                <option value="">默认</option>
                {(s.catalog?.bodies ?? []).map((body) => (
                  <option key={body.file} value={body.file}>
                    {body.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="combo-field">
              <span>表情</span>
              <select
                value={s.selection.expression ?? ''}
                disabled={!s.catalog || loading}
                onChange={(e) => chooseSlot({ expression: e.target.value || null })}
              >
                <option value="">无</option>
                {(s.catalog?.expressions ?? []).map((expr) => (
                  <option key={expr.file} value={expr.file}>
                    {expr.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="combo-field">
              <span>动作</span>
              <select
                value={s.selection.motion ?? ''}
                disabled={!s.catalog || loading}
                onChange={(e) => chooseSlot({ motion: e.target.value || null })}
              >
                <option value="">无</option>
                {(s.catalog?.motions ?? []).map((motion) => (
                  <option key={motion.file} value={motion.file}>
                    {motion.name}
                  </option>
                ))}
              </select>
            </label>
            <span className="combo-hint small muted">体型 + 表情 + 动作同时生效</span>
          </div>

          <div
            className={`canvas-box${panning ? ' panning' : ''}${loading ? ' loading' : ''}`}
            ref={canvasBoxRef}
            onPointerDown={onCanvasPointerDown}
            onPointerMove={onCanvasPointerMove}
            onPointerUp={endPan}
            onPointerCancel={endPan}
            onDoubleClick={() => void fitNow().catch((err) => console.warn('[motions]', err))}
            title="滚轮缩放 · 拖动平移 · 双击适应"
          >
            <canvas ref={canvasRef} />
            <div className="zoom-controls" onPointerDown={(e) => e.stopPropagation()} onDoubleClick={(e) => e.stopPropagation()}>
              <button type="button" onClick={() => zoomBy(1 / 1.2)} title="缩小（-）">
                −
              </button>
              <button type="button" className="zoom-value" onClick={() => void fitNow()} title="适应角色（F / 0）">
                {zoomPct}%
              </button>
              <button type="button" onClick={() => zoomBy(1.2)} title="放大（+）">
                +
              </button>
              <button type="button" onClick={() => void fitNow()} title="适应角色（F / 0）">
                适应
              </button>
            </div>
            {loading && (
              <div className="overlay">
                <div className="spinner" />
                <div>{s.stage}…</div>
              </div>
            )}
            {!loading && !s.preview && s.phase !== 'error' && <div className="overlay">选择左侧素材开始预览</div>}
            <div className="viewport-hint small">滚轮缩放 · 拖动平移 · 双击适应</div>
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
                disabled={!s.selection.motion}
                onChange={(e) => void setLoop(e.target.checked).catch((err) => setView({ error: explainError(err), loadingPreview: false }))}
              />
              循环
            </label>
            <label className="inline" title="快速开关动作层（不会清掉体型/表情选择）">
              <input
                type="checkbox"
                checked={!!s.selection.motion}
                disabled={!s.catalog?.motions.length}
                onChange={(e) => void setWithIdle(e.target.checked).catch((err) => setView({ error: explainError(err), loadingPreview: false }))}
              />
              叠加动作
            </label>
          </div>

          {s.preview ? (
            <MetaPanel preview={s.preview} />
          ) : (
            <div className="meta-panel small muted">
              当前模型：{MODEL_URI}
              <br />
              动作来源：model3.json 的 FileReferences.Motions 优先，没有声明时扫描 Animation/；表情同理（Expressions 优先，否则 Exp3/）。
            </div>
          )}

          <div className="keys small muted">
            快捷键：<kbd>空格</kbd> 播放/暂停　<kbd>←</kbd>/<kbd>→</kbd> ±1 帧（<kbd>Shift</kbd> ±10）　<kbd>Home</kbd>/<kbd>End</kbd>　
            <kbd>L</kbd> 循环　<kbd>F</kbd>/<kbd>0</kbd> 适应　<kbd>+</kbd>/<kbd>−</kbd> 缩放　滚轮缩放 · 拖动平移 · 双击适应
          </div>
        </main>
      </div>

      {gallery && s.catalog && (
        <Gallery assets={allAssets} thumbnails={s.thumbnails} onSelect={choose} onClose={() => setGallery(false)} />
      )}

      <footer>
        <div>
          预览只使用 CharaForge runtime；不加载原参考视频 / 音频，不写 scene 文件。体型是可选的{' '}
          <code>charaforge.bodies.json</code> 声明 + Exp3/ 自动识别；素材目录来自只读接口{' '}
          <code>/__charaforge/list-assets</code>（`pnpm dev` / `pnpm preview` 提供）。
        </div>
        <div className="muted small">
          {MODEL_URI === PREVIEW_MODEL_URI ? (
            <>
              模型：DeepSeek-chan — Illustration &amp; PSD Layer Separation: Kamiya Ryoko / Live2D Rigging: 霓虹猫_NeonNeko。
              模型按原作者条款禁止再分发，本页面只供本地查看。
            </>
          ) : (
            <>模型：{MODEL_NAME}。使用权以模型目录中的原条款为准；本页面只供本地查看，不写盘。</>
          )}
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
