/*
 * 动作 / 表情素材目录的纯逻辑（不依赖 DOM、不 fetch，便于 Node 测试）。
 *
 * 数据来源优先级：
 *   1. model3.json 的 FileReferences.Motions / Expressions（如果未来声明了）；
 *   2. 没有声明时，由 /__charaforge/list-assets 列出 Animation/ 与 Exp3/（见 apps/player/server/assetList.ts）。
 *
 * 这里只解析 motion3 / exp3 的元数据并构造临时预览 scene；不修改模型文件，也不写盘。
 */

export type AssetKind = 'motion' | 'expression';
export type AssetSource = 'declared' | 'directory';

export const PREVIEW_MODEL_URI = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
export const PREVIEW_MODEL_NAME = 'DeepSeek-chan（大肥鱼）';
export const ANIMATION_DIR = 'assets/models/deepseek-chan/DeepSeek/Animation';
export const EXP3_DIR = 'assets/models/deepseek-chan/DeepSeek/Exp3';
export const PREVIEW_FPS = 60;
export const PREVIEW_PHYSICS_WARMUP = 45;
export const EXPRESSION_PREVIEW_FRAMES = 120;
export const PREVIEW_COMPOSITION = { width: 720, height: 720, background: '#202830' } as const;
export const THUMBNAIL_TILE = 160;
export const THUMBNAIL_COLUMNS = 6;
export const THUMBNAIL_BACKGROUND = '#202830';

/** 动作不足 2 秒时也留 120 帧（2 秒），方便观察动作结束/淡出后的状态。 */
export const MIN_MOTION_PREVIEW_FRAMES = 120;

export interface MotionMeta {
  duration: number;
  fps: number;
  loop: boolean;
  curveCount: number;
  totalSegmentCount: number;
  totalPointCount: number;
  userDataCount: number;
}

export interface ExpressionParameter {
  id: string;
  value: number;
  blend: string;
}

export interface ExpressionInfo {
  parameters: ExpressionParameter[];
}

export interface AssetFileInfo {
  name: string;
  size: number;
  mtime: string;
}

export interface MotionAssetInfo {
  kind: 'motion';
  name: string;
  file: string;
  uri: string;
  dir: string;
  size: number;
  mtime: string;
  source: AssetSource;
  meta: MotionMeta;
}

export interface ExpressionAssetInfo {
  kind: 'expression';
  name: string;
  file: string;
  uri: string;
  dir: string;
  size: number;
  mtime: string;
  source: AssetSource;
  info: ExpressionInfo;
}

export type CatalogAsset = MotionAssetInfo | ExpressionAssetInfo;

export interface Catalog {
  modelUri: string;
  motions: MotionAssetInfo[];
  expressions: ExpressionAssetInfo[];
  sources: { motions: AssetSource; expressions: AssetSource };
  warnings: string[];
}

/** 缩略图 / 图集里的唯一键：kind + 文件名。 */
export function assetKey(a: { kind: AssetKind; file: string }): string {
  return `${a.kind}:${a.file}`;
}

/** 图集顺序：先动作、后表情（与缩略图 contact sheet 的行优先顺序一致）。 */
export function catalogAssets(catalog: Catalog): CatalogAsset[] {
  return [...catalog.motions, ...catalog.expressions];
}

export interface FitTransform {
  x: number;
  y: number;
  scale: number;
}

function record(v: unknown, what: string): Record<string, unknown> {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error(`${what} 不是对象`);
  return v as Record<string, unknown>;
}

function finiteNumber(v: unknown, what: string): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${what} 不是有效数字`);
  return v;
}

function optionalNumber(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

/** 解析 motion3.json 的 Meta。缺 Meta / Duration 时抛错（该素材会被跳过并记 warning）。 */
export function parseMotionMeta(input: unknown, label = 'motion3'): MotionMeta {
  const root = record(input, label);
  const meta = record(root.Meta, `${label} 的 Meta`);
  const duration = finiteNumber(meta.Duration, `${label} 的 Meta.Duration`);
  if (duration < 0) throw new Error(`${label} 的 Meta.Duration 不能为负`);
  const curves = Array.isArray(root.Curves) ? root.Curves.length : 0;
  return {
    duration,
    fps: optionalNumber(meta.Fps, 60),
    loop: meta.Loop === true,
    curveCount: Math.max(0, Math.round(optionalNumber(meta.CurveCount, curves))),
    totalSegmentCount: Math.max(0, Math.round(optionalNumber(meta.TotalSegmentCount, 0))),
    totalPointCount: Math.max(0, Math.round(optionalNumber(meta.TotalPointCount, 0))),
    userDataCount: Math.max(0, Math.round(optionalNumber(meta.UserDataCount, 0))),
  };
}

/** 解析 exp3.json 的 Parameters（可能为空）。 */
export function parseExpressionInfo(input: unknown, label = 'exp3'): ExpressionInfo {
  const root = record(input, label);
  if (root.Parameters === undefined) return { parameters: [] };
  if (!Array.isArray(root.Parameters)) throw new Error(`${label} 的 Parameters 不是数组`);
  const parameters = root.Parameters.map((p, i) => {
    const item = record(p, `${label} 的第 ${i + 1} 个参数`);
    if (typeof item.Id !== 'string' || !item.Id) throw new Error(`${label} 的第 ${i + 1} 个参数缺少 Id`);
    return {
      id: item.Id,
      value: optionalNumber(item.Value, 0),
      blend: typeof item.Blend === 'string' && item.Blend ? item.Blend : 'Add',
    };
  });
  return { parameters };
}

/** 文件显示名：去掉 .motion3.json / .exp3.json。 */
export function displayName(file: string): string {
  return file.replace(/\.(motion3|exp3)\.json$/i, '');
}

/** assets/models/.../Animation + 文件名 → 场景资源根目录（assets/）下的相对 uri。 */
export function assetUriFromDir(dir: string, file: string): string {
  const prefix = 'assets/';
  if (!dir.startsWith(prefix)) throw new Error(`目录必须以 ${prefix} 开头：${dir}`);
  return `${dir.slice(prefix.length)}/${file}`.replace(/\/+/g, '/');
}

/** model3.json 里的相对文件（相对 model3 所在目录）→ assets/ 下的相对 uri。 */
export function declaredAssetUri(modelUri: string, file: string): string {
  const base = modelUri.slice(0, modelUri.lastIndexOf('/') + 1);
  const parts: string[] = [];
  for (const seg of `${base}${file.replace(/\\/g, '/')}`.split('/')) {
    if (!seg || seg === '.') continue;
    if (seg === '..') {
      parts.pop();
      continue;
    }
    parts.push(seg);
  }
  return parts.join('/');
}

/** model3.json 的 FileReferences.Motions（未来声明时）→ 去重后的相对文件列表。 */
export function declaredMotionFiles(model3: unknown): string[] {
  if (!model3 || typeof model3 !== 'object' || Array.isArray(model3)) return [];
  const refs = (model3 as Record<string, unknown>).FileReferences;
  if (!refs || typeof refs !== 'object' || Array.isArray(refs)) return [];
  const raw = (refs as Record<string, unknown>).Motions;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return [];
  const files: string[] = [];
  for (const list of Object.values(raw as Record<string, unknown>)) {
    if (!Array.isArray(list)) continue;
    for (const item of list) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
      const file = (item as Record<string, unknown>).File;
      if (typeof file === 'string' && file) files.push(file);
    }
  }
  return [...new Set(files)];
}

/** model3.json 的 FileReferences.Expressions（未来声明时）→ { file, name }。 */
export function declaredExpressionFiles(model3: unknown): { file: string; name?: string }[] {
  if (!model3 || typeof model3 !== 'object' || Array.isArray(model3)) return [];
  const refs = (model3 as Record<string, unknown>).FileReferences;
  if (!refs || typeof refs !== 'object' || Array.isArray(refs)) return [];
  const raw = (refs as Record<string, unknown>).Expressions;
  if (!Array.isArray(raw)) return [];
  const out: { file: string; name?: string }[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    const rec = item as Record<string, unknown>;
    if (typeof rec.File === 'string' && rec.File) out.push({ file: rec.File, name: typeof rec.Name === 'string' ? rec.Name : undefined });
  }
  return out;
}

export function isMotionFile(name: string): boolean {
  return /\.motion3\.json$/i.test(name);
}

export function isExpressionFile(name: string): boolean {
  return /\.exp3\.json$/i.test(name);
}

export function sortAssets<T extends { name: string }>(assets: T[]): T[] {
  return assets.sort((a, b) => a.name.localeCompare(b.name, 'zh-CN', { numeric: true }));
}

export interface PreviewSceneOptions {
  kind: AssetKind;
  file: string;
  uri: string;
  motionMeta?: MotionMeta;
  /** 动作循环开关；缺省取 motion3 的 Meta.Loop。 */
  loop?: boolean;
  /** 表情预览时叠加的 IDLE motion3（相对 assets/ 的 uri）；不传则只用默认姿态。 */
  idleMotionUri?: string;
  composition?: { width: number; height: number; background?: string | null };
  transform?: FitTransform;
}

export function previewDuration(kind: AssetKind, meta?: MotionMeta): number {
  if (kind === 'expression') return EXPRESSION_PREVIEW_FRAMES;
  const duration = meta?.duration ?? 0;
  return Math.max(MIN_MOTION_PREVIEW_FRAMES, Math.ceil(duration * PREVIEW_FPS));
}

/**
 * 临时预览 scene：一个模型图层 + 一个 motion/expression 片段。transform 为占位值；
 * 页面在 loadScene 之后从 runtime 读模型 canvas，再用 fitTransform 覆盖。
 */
export function buildPreviewScene(o: PreviewSceneOptions): Record<string, unknown> {
  const composition = o.composition ?? PREVIEW_COMPOSITION;
  const duration = previewDuration(o.kind, o.motionMeta);
  const transform = o.transform ?? { x: composition.width / 2, y: composition.height / 2, scale: 0.16 };
  const assets: Record<string, unknown> = {
    models: [{ id: 'm', uri: PREVIEW_MODEL_URI, name: PREVIEW_MODEL_NAME }],
  };
  const layer: Record<string, unknown> = {
    id: 'model',
    kind: 'model',
    model: 'm',
    in: 0,
    out: duration,
    transform,
    physics: { warmup: PREVIEW_PHYSICS_WARMUP },
  };
  if (o.kind === 'motion') {
    assets.motions = [{ id: 'preview', model: 'm', uri: o.uri }];
    layer.motions = [
      {
        motion: 'preview',
        start: 0,
        loop: o.loop ?? o.motionMeta?.loop ?? false,
        fadeIn: 0,
      },
    ];
  } else {
    assets.expressions = [{ id: 'preview', model: 'm', uri: o.uri }];
    layer.expressions = [{ expression: 'preview', start: 0, fadeIn: 0 }];
    if (o.idleMotionUri) {
      // 表情默认仍走默认姿态；传入 IDLE 时按“IDLE 动作 + exp3”叠加，便于观察自然动作下的表情。
      assets.motions = [{ id: 'idle', model: 'm', uri: o.idleMotionUri }];
      layer.motions = [{ motion: 'idle', start: 0, loop: true, fadeIn: 0 }];
    }
  }
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: {
      title: `${o.kind === 'motion' ? '动作' : '表情'}预览：${displayName(o.file)}`,
      description: 'CharaForge 动作 / 表情素材查看器的临时预览场景（不写盘、不修改原场景）',
    },
    timebase: { fps: PREVIEW_FPS },
    duration,
    composition: { ...composition },
    assets,
    layers: [layer],
  };
}

export interface ThumbnailSceneOptions {
  tile?: number;
  columns?: number;
  background?: string;
}

/**
 * 一次性生成全部素材缩略图的 contact sheet scene：每个素材一个模型图层，排成网格。
 * 物理关闭、只渲染帧 0；motion 用 offset 直接取动作中段，避免为每个图层从 0 逐帧推进。
 * 每个图层带 clip，缩略图不会互相覆盖。加载后由页面按模型画布尺寸重新计算 transform。
 */
export function buildThumbnailScene(catalog: Catalog, opts: ThumbnailSceneOptions = {}): Record<string, unknown> {
  const tile = Math.max(32, Math.round(opts.tile ?? THUMBNAIL_TILE));
  const columns = Math.max(1, Math.round(opts.columns ?? THUMBNAIL_COLUMNS));
  const sources = catalogAssets(catalog);
  const rows = Math.max(1, Math.ceil(sources.length / columns));
  const width = columns * tile;
  const height = rows * tile;
  const assets: Record<string, unknown> = {
    models: [{ id: 'm', uri: PREVIEW_MODEL_URI, name: PREVIEW_MODEL_NAME }],
    motions: [],
    expressions: [],
  };
  const motionAssets = assets.motions as Record<string, unknown>[];
  const expressionAssets = assets.expressions as Record<string, unknown>[];
  const layers = sources.map((source, i) => {
    const col = i % columns;
    const row = Math.floor(i / columns);
    const assetId = `${source.kind === 'motion' ? 'm' : 'e'}${i}`;
    const layer: Record<string, unknown> = {
      id: `thumb.${assetId}`,
      kind: 'model',
      model: 'm',
      in: 0,
      out: 1,
      physics: { enabled: false },
      transform: { x: col * tile + tile / 2, y: row * tile + tile / 2, scale: 0.16 },
      clip: { x: col * tile, y: row * tile, width: tile, height: tile },
    };
    if (source.kind === 'motion') {
      motionAssets.push({ id: assetId, model: 'm', uri: source.uri });
      const offset = Math.min(1.5, Math.max(0.1, source.meta.duration * 0.35));
      layer.motions = [{ motion: assetId, start: 0, offset, loop: true, fadeIn: 0 }];
    } else {
      expressionAssets.push({ id: assetId, model: 'm', uri: source.uri });
      layer.expressions = [{ expression: assetId, start: 0, fadeIn: 0 }];
    }
    return layer;
  });
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: { title: '素材缩略图', description: '动作 / 表情查看器的 contact sheet 场景（只读，不写盘）' },
    timebase: { fps: PREVIEW_FPS },
    duration: 1,
    composition: { width, height, background: opts.background ?? THUMBNAIL_BACKGROUND },
    assets,
    layers,
  };
}

/** 按模型画布尺寸把整张画布等比放进 width×height 的画面，居中。 */
export function fitTransform(canvas: { width: number; height: number }, width: number, height: number, margin = 0.96): FitTransform {
  if (!(canvas.width > 0 && canvas.height > 0)) throw new Error('模型画布尺寸无效');
  const scale = Math.min(width / canvas.width, height / canvas.height) * margin;
  return { x: width / 2, y: height / 2, scale };
}
