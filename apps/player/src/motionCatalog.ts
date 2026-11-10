/*
 * 动作 / 表情 / 体型素材目录的纯逻辑（不依赖 DOM、不 fetch，便于 Node 测试）。
 *
 * 数据来源优先级：
 *   1. model3.json 的 FileReferences.Motions / Expressions（如果未来声明了）；
 *   2. 没有声明时，由 /__charaforge/list-assets 列出 Animation/ 与 Exp3/（见 apps/player/server/assetList.ts）。
 *   3. 模型目录下可选的 charaforge.bodies.json：显式声明哪些 exp3 是体型，以及默认体型。
 *
 * 体型（body）本质上仍是 exp3，但它是一个“基础参数层”，和表情/动作可以同时生效；
 * 查看器把它从表情列表里拆出来，避免“选了体型就不能选表情”。
 *
 * 这里只解析 motion3 / exp3 的元数据并构造临时预览 scene；不修改模型文件，也不写盘。
 */

export type AssetKind = 'motion' | 'expression' | 'body';
export type AssetSource = 'declared' | 'directory';

export const PREVIEW_MODEL_URI = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
export const PREVIEW_MODEL_NAME = 'DeepSeek-chan（大肥鱼）';
export const ANIMATION_DIR = 'assets/models/deepseek-chan/DeepSeek/Animation';
export const EXP3_DIR = 'assets/models/deepseek-chan/DeepSeek/Exp3';
export const BODY_MANIFEST_FILE = 'charaforge.bodies.json';
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

export interface BodyAssetInfo {
  kind: 'body';
  name: string;
  file: string;
  uri: string;
  dir: string;
  size: number;
  mtime: string;
  source: AssetSource;
  info: ExpressionInfo;
  /** 来自 manifest 的显式声明（而不是命名 / 参数推断）。 */
  declared?: boolean;
}

export type CatalogAsset = MotionAssetInfo | ExpressionAssetInfo | BodyAssetInfo;

export interface Catalog {
  modelUri: string;
  modelName?: string;
  motions: MotionAssetInfo[];
  /** 普通表情（不包含体型）。 */
  expressions: ExpressionAssetInfo[];
  /** 体型预设（同样是 exp3，但作为基础层独立选择）。 */
  bodies: BodyAssetInfo[];
  sources: { motions: AssetSource; expressions: AssetSource; bodies: AssetSource };
  /** 初始体型文件；没有显式 default 时取“默认体型”或第一个。 */
  defaultBody?: string;
  warnings: string[];
}

/** 缩略图 / 图集里的唯一键：kind + 文件名。 */
export function assetKey(a: { kind: AssetKind; file: string }): string {
  return `${a.kind}:${a.file}`;
}

/** 图集顺序：先动作、后表情、最后体型（与缩略图 contact sheet 的行优先顺序一致）。 */
export function catalogAssets(catalog: Catalog): CatalogAsset[] {
  return [...catalog.motions, ...catalog.expressions, ...catalog.bodies];
}

export interface FitTransform {
  x: number;
  y: number;
  scale: number;
  /** 缺省为模型画布中心。 */
  anchorX?: number;
  anchorY?: number;
}

/** 模型实际内容（所有可见 drawable 顶点）在画布像素坐标里的包围盒。 */
export interface ContentBounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
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

/** model3 所在目录 + Animation / Exp3 → 仓库相对目录（带 assets/ 前缀）。 */
export function modelAssetDir(modelUri: string, leaf: 'Animation' | 'Exp3'): string {
  const slash = modelUri.lastIndexOf('/');
  const dir = slash >= 0 ? modelUri.slice(0, slash) : '';
  return dir ? `assets/${dir}/${leaf}`.replace(/\/+/g, '/') : '';
}

/** 从 model3 uri 猜一个显示名：取文件名去掉 .model3.json。 */
export function modelNameFromUri(modelUri: string): string {
  const file = modelUri.slice(modelUri.lastIndexOf('/') + 1);
  return file.replace(/\.model3\.json$/i, '');
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

/** 体型 manifest 放在 model3 同目录；没有该文件时完全走自动识别。 */
export function bodyManifestUri(modelUri: string): string {
  return declaredAssetUri(modelUri, BODY_MANIFEST_FILE);
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

// ---------- 体型识别 ----------

/** 文件名里的体型关键词。显式 manifest 优先；这里只是没有 manifest 时的兜底。 */
const BODY_NAME_RE = /(体型|身材|身型|形态|身高|body\s*(?:type|shape)|bodytype|bodyshape|proportion)/i;

/**
 * 强体型参数：只改一个也基本可以确定是体型/身材，而不是单件饰品开关。
 * 不包含 Clothes/Nail/Tail 等“单件开关”，避免把脱鞋、尾巴开关误判成体型。
 */
const STRONG_BODY_PARAM_RE =
  /^(?:Param)?(?:Body(?:Scale|Size|Shape|Type|Height|Width)?|Bust|Breast|Chest|Waist|Hip|Height|Chubby|Fat|Slim|Thin|Proportion|Torso|LegLength|ArmLength|HeadSize)$/i;

/** 弱体型参数：只有同时改多组时才视为体型（例如体型预设顺带改衣服）。 */
const WEAK_BODY_PARAM_RE =
  /^(?:Param)?(?:Clothes|Cloth|Outfit|Costume|Dress|Skirt|Pants|Shoe|Foot|Tail|Wing|Horn|Halo)\d*$/i;

/** 文件名/声明名是否像体型。 */
export function isBodyPresetName(name: string): boolean {
  return BODY_NAME_RE.test(name);
}

/**
 * 不依赖 manifest 的体型推断：
 *   - 名字里有“体型/身材/body shape”等关键词；或
 *   - 所有参数都是强体型参数；或
 *   - 至少两个参数，且全部是强/弱体型参数（例如 身高 + 衣服）。
 *
 * 单改 Clothes2 / 脱鞋 / 尾巴开关 这类单件开关不会被当成体型。
 */
export function isBodyPreset(info: ExpressionInfo, name: string): boolean {
  if (isBodyPresetName(name)) return true;
  const ids = info.parameters.map((p) => p.id).filter(Boolean);
  if (!ids.length) return false;
  if (ids.every((id) => STRONG_BODY_PARAM_RE.test(id))) return true;
  return ids.length >= 2 && ids.every((id) => STRONG_BODY_PARAM_RE.test(id) || WEAK_BODY_PARAM_RE.test(id));
}

export interface BodyManifestEntry {
  /** assets/ 下的相对 uri。 */
  uri: string;
  name?: string;
}

export interface BodyManifest {
  bodies: BodyManifestEntry[];
  defaultUri?: string;
}

/**
 * 解析模型目录下的 charaforge.bodies.json：
 *   { "bodies": [{ "file": "Exp3/小萝莉.exp3.json", "name": "小萝莉" }], "default": "Exp3/默认体型.exp3.json" }
 * 也接受 bodies 里直接写字符串。
 */
export function parseBodyManifest(input: unknown, modelUri: string): BodyManifest {
  const root = record(input, '体型 manifest');
  if (!Array.isArray(root.bodies)) throw new Error('体型 manifest 的 bodies 不是数组');
  const bodies: BodyManifestEntry[] = [];
  for (const [i, raw] of root.bodies.entries()) {
    const item = typeof raw === 'string' ? { file: raw } : record(raw, `体型 manifest 的第 ${i + 1} 项`);
    if (typeof item.file !== 'string' || !item.file) throw new Error(`体型 manifest 的第 ${i + 1} 项缺少 file`);
    bodies.push({
      uri: declaredAssetUri(modelUri, item.file),
      name: typeof item.name === 'string' && item.name ? item.name : undefined,
    });
  }
  const defaultFile = typeof root.default === 'string' && root.default ? root.default : undefined;
  return { bodies, defaultUri: defaultFile ? declaredAssetUri(modelUri, defaultFile) : undefined };
}

export interface SplitExpressionsResult {
  expressions: ExpressionAssetInfo[];
  bodies: BodyAssetInfo[];
  defaultBody?: string;
}

/**
 * 把素材目录里的 exp3 拆成“普通表情”和“体型”。
 * manifest 中声明的文件强制视为体型；其余按 isBodyPreset 推断。
 */
export function splitBodyPresets(expressions: ExpressionAssetInfo[], manifest?: BodyManifest): SplitExpressionsResult {
  const declared = new Map((manifest?.bodies ?? []).map((item) => [item.uri, item]));
  const faces: ExpressionAssetInfo[] = [];
  const bodies: BodyAssetInfo[] = [];
  for (const asset of expressions) {
    const entry = declared.get(asset.uri);
    if (entry || isBodyPreset(asset.info, asset.name)) {
      bodies.push({
        kind: 'body',
        name: entry?.name ?? asset.name,
        file: asset.file,
        uri: asset.uri,
        dir: asset.dir,
        size: asset.size,
        mtime: asset.mtime,
        source: asset.source,
        info: asset.info,
        ...(entry ? { declared: true } : {}),
      });
      continue;
    }
    faces.push(asset);
  }
  const defaultBody =
    (manifest?.defaultUri && bodies.find((body) => body.uri === manifest.defaultUri)?.file) ||
    bodies.find((body) => /默认|default/i.test(body.name))?.file ||
    bodies[0]?.file;
  return { expressions: faces, bodies, ...(defaultBody ? { defaultBody } : {}) };
}

// ---------- 预览 scene ----------

export interface PreviewAssetRef {
  file: string;
  uri: string;
}

export interface PreviewSceneOptions {
  /** 模型 model3 uri；缺省用当前示例模型。 */
  modelUri?: string;
  modelName?: string;
  /** 动作；没有动作时画面只显示体型/表情的静态状态。 */
  motion?: PreviewAssetRef & { meta?: MotionMeta };
  /** 体型基础层（先于表情应用）。 */
  body?: PreviewAssetRef;
  /** 表情层（在体型之后应用，因此不会覆盖体型）。 */
  expression?: PreviewAssetRef;
  /** 动作循环开关；缺省取 motion3 的 Meta.Loop。 */
  loop?: boolean;
  composition?: { width: number; height: number; background?: string | null };
  transform?: FitTransform;
}

export function previewDuration(kind: AssetKind, meta?: MotionMeta): number {
  if (kind !== 'motion') return EXPRESSION_PREVIEW_FRAMES;
  const duration = meta?.duration ?? 0;
  return Math.max(MIN_MOTION_PREVIEW_FRAMES, Math.ceil(duration * PREVIEW_FPS));
}

/**
 * 临时预览 scene：一个模型图层 + 可选的 motion / body / expression 片段。
 * transform 为占位值；页面在 loadScene 之后读取模型内容包围盒，再用 fitTransformToBounds 覆盖。
 * expression 片段顺序固定为“体型 → 表情”，后者不会把体型顶掉。
 */
export function buildPreviewScene(o: PreviewSceneOptions): Record<string, unknown> {
  const composition = o.composition ?? PREVIEW_COMPOSITION;
  const duration = o.motion ? previewDuration('motion', o.motion.meta) : EXPRESSION_PREVIEW_FRAMES;
  const transform = o.transform ?? { x: composition.width / 2, y: composition.height / 2, scale: 1 };
  const modelUri = o.modelUri ?? PREVIEW_MODEL_URI;
  const assets: Record<string, unknown> = {
    models: [{ id: 'm', uri: modelUri, name: o.modelName ?? modelNameFromUri(modelUri) }],
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
  if (o.motion) {
    assets.motions = [{ id: 'motion', model: 'm', uri: o.motion.uri }];
    layer.motions = [
      {
        motion: 'motion',
        start: 0,
        loop: o.loop ?? o.motion.meta?.loop ?? false,
        fadeIn: 0,
      },
    ];
  }
  const expressionAssets: Record<string, unknown>[] = [];
  const expressionClips: Record<string, unknown>[] = [];
  if (o.body) {
    expressionAssets.push({ id: 'body', model: 'm', uri: o.body.uri });
    expressionClips.push({ expression: 'body', start: 0, fadeIn: 0 });
  }
  if (o.expression) {
    expressionAssets.push({ id: 'expression', model: 'm', uri: o.expression.uri });
    expressionClips.push({ expression: 'expression', start: 0, fadeIn: 0 });
  }
  if (expressionAssets.length) {
    assets.expressions = expressionAssets;
    layer.expressions = expressionClips;
  }
  const parts = [o.body ? `体型 ${displayName(o.body.file)}` : '', o.expression ? `表情 ${displayName(o.expression.file)}` : '', o.motion ? `动作 ${displayName(o.motion.file)}` : ''].filter(Boolean);
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: {
      title: `${parts.join(' / ') || '素材'}预览`,
      description: 'CharaForge 动作 / 表情 / 体型查看器的临时预览场景（不写盘、不修改原场景）',
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
  const modelUri = catalog.modelUri || PREVIEW_MODEL_URI;
  const assets: Record<string, unknown> = {
    models: [{ id: 'm', uri: modelUri, name: catalog.modelName ?? modelNameFromUri(modelUri) }],
    motions: [],
    expressions: [],
  };
  const motionAssets = assets.motions as Record<string, unknown>[];
  const expressionAssets = assets.expressions as Record<string, unknown>[];
  const layers = sources.map((source, i) => {
    const col = i % columns;
    const row = Math.floor(i / columns);
    const assetId = `${source.kind === 'motion' ? 'm' : source.kind === 'body' ? 'b' : 'e'}${i}`;
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
    meta: { title: '素材缩略图', description: '动作 / 表情 / 体型查看器的 contact sheet 场景（只读，不写盘）' },
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

/**
 * 按“模型实际内容”的包围盒（画布像素）等比放进画面，居中。
 * 体型变小后，画布通常不变，所以不能再用整张画布 fit；用内容包围盒才能让角色占满预览。
 */
export function fitTransformToBounds(
  bounds: ContentBounds,
  width: number,
  height: number,
  margin = 0.92,
): FitTransform {
  const bw = bounds.x1 - bounds.x0;
  const bh = bounds.y1 - bounds.y0;
  if (!(bw > 0 && bh > 0) || !Number.isFinite(bw + bh)) throw new Error('模型内容包围盒无效');
  const scale = Math.min(width / bw, height / bh) * margin;
  return {
    x: width / 2,
    y: height / 2,
    scale,
    anchorX: (bounds.x0 + bounds.x1) / 2,
    anchorY: (bounds.y0 + bounds.y1) / 2,
  };
}

/** 数字输入框/URL 参数用：把任意值夹到 [min, max]。 */
export function clampNumber(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}
