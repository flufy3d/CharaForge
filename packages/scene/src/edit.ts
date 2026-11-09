/*
 * 场景编辑命令层（纯 TypeScript，无框架依赖）。
 *
 * 浏览器编辑器、AI CLI 和 HTTP API 共用这一层：命令在原地修改 Scene，同时记录
 * JSON Pointer 形式的 EditChange，可撤销/重做。任何命令抛错时整批命令回滚。
 *
 * 命令别名（兼容不同调用方；所有 op 严格区分大小写）：
 *   setScene | updateScene
 *   addImageLayer | addImage | addDecoration | addLayer（layer.kind = image 时）
 *   addModelLayer | addModel | addLayer（layer.kind = model 或未写 kind 时）
 *   updateLayer | editLayer | updateDecoration
 *   removeLayer | deleteLayer | removeDecoration
 *   moveLayer | reorderLayer
 *   duplicateLayer | cloneLayer
 *   addMotion | addMotionClip
 *   addExpression | addExpressionClip
 *   updateMotion | updateMotionClip / updateExpression | updateExpressionClip
 *   removeMotion | removeMotionClip | deleteMotion / removeExpression | removeExpressionClip | deleteExpression
 *   addAudio | addAudioClip
 *   updateAudio | updateAudioClip
 *   removeAudio | removeAudioClip | deleteAudio
 *   addMarker / updateMarker / removeMarker | deleteMarker
 *   setCamera / updateCamera
 *   setKeyframes | setCurve | setAnimation
 *   addKeyframe | addKey
 *   updateKeyframe | updateKey
 *   removeKeyframe | removeKey | deleteKeyframe
 *   setEase | ease
 *   animate | autoAnimate
 *   preview / noop | nop | no-op
 *
 * 字段别名：
 *   - motion/expression 命令的 layer 也可以写 id；
 *   - target 的 layer id 也可以写 layer 字段；省略 kind 时会根据字段推断；
 *   - addKeyframe 的 t/v 也可以写 time/value；
 *   - updateKeyframe 的 newT 也可以写 newTime。
 *
 * 关键帧约定：
 *   - setKeyframes 的 keys 会按 t 排序，重复 t（1e-6 内）时最后一个生效；
 *   - 空 keys 表示“取消动画”：可选属性删除，必填属性回到缺省常量；param 目标删除整条轨道；
 *   - addKeyframe 遇到常量会直接换成单关键帧曲线（单关键帧 = 常量曲线）；
 *   - removeKeyframe 只要还剩关键帧就保留曲线；删到 0 个时，flatten 缺省 true 用删除前
 *     在该时刻采样的值变成常量，flatten:false 则保留单关键帧曲线（两者语义等价）；
 *   - baked 曲线在关键帧编辑时展开成逐帧关键帧（语义不变）。
 */

import type {
  Animatable,
  Curve,
  Ease,
  Keyframe,
  Provenance,
  Semantics,
} from './common.ts';
import type {
  AudioAsset,
  AudioClip,
  CameraTransform,
  Clip,
  ExpressionAsset,
  ExpressionClip,
  Filter,
  ImageAsset,
  ImageAttach,
  ImageLayer,
  Layer,
  Marker,
  ModelLayer,
  MotionAsset,
  MotionClip,
  ParamTrack,
  Scene,
  Transform2D,
} from './scene.ts';
import { sample } from './curve.ts';
import { parseScene } from './validate.ts';

type AnimNumber = Animatable<number>;

// ---------- 公共类型 ----------

export type EditEffect = 'render' | 'workers' | 'evaluator' | 'structure' | 'audio' | 'clock';

export type EditTarget =
  | { kind: 'layer'; id: string; path: string }
  | { kind: 'camera'; path: string }
  | { kind: 'param'; id: string; param: string };

/** 与 EditTarget 相同；时间线等调用方可以直接使用 EditTarget。 */
export type EditTargetSpec = EditTarget;

export type EasePreset = 'linear' | 'smooth' | 'easeIn' | 'easeOut' | 'easeInOut' | 'step' | 'hold';

export const EASE_PRESETS: Record<EasePreset, Ease> = {
  linear: 'linear',
  smooth: 'smooth',
  easeIn: [0.42, 0, 1, 1],
  easeOut: [0, 0, 0.58, 1],
  easeInOut: [0.42, 0, 0.58, 1],
  step: 'step',
  hold: 'step',
};

/** 把缓动预设或原始 Ease 规范化为 Ease；未知预设/非法贝塞尔会抛错。 */
export function easePreset(name: EasePreset | Ease): Ease {
  if (typeof name === 'string') {
    if (!Object.prototype.hasOwnProperty.call(EASE_PRESETS, name)) throw new Error(`未知缓动预设：${name}`);
    const found = EASE_PRESETS[name as EasePreset];
    return Array.isArray(found) ? ([...found] as Ease) : found;
  }
  if (!Array.isArray(name) || name.length !== 4 || name.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
    throw new Error('ease 必须是 linear/step/smooth、预设名或 [x1,y1,x2,y2]');
  }
  return [...name] as Ease;
}

/** 给所有关键帧设置同一缓动（写的是“本帧到下一帧”的 ease）。 */
export function easeKeyframes(keys: { t: number; v: number; ease?: Ease }[], ease: Ease): void {
  const normalized = Array.isArray(ease) ? ([...ease] as Ease) : ease;
  for (const key of keys) key.ease = Array.isArray(normalized) ? ([...normalized] as Ease) : normalized;
}

export type AnimationPreset =
  | 'fadeIn'
  | 'fadeOut'
  | 'float'
  | 'pulse'
  | 'spin'
  | 'bounce'
  | 'blink'
  | 'slideIn'
  | 'shake'
  | 'zoomIn';

export interface EditChange {
  op: 'add' | 'remove' | 'replace';
  path: string;
  before?: unknown;
  after?: unknown;
}

export interface FramePreviewCamera {
  x: number;
  y: number;
  zoom: number;
  rotation: number;
}

export interface FramePreviewLayer {
  id: string;
  kind: 'model' | 'image';
  name?: string;
  model?: string;
  image?: string;
  x: number;
  y: number;
  scale: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  opacity: number;
  blendMode?: string;
}

export interface FramePreviewAudio {
  index: number;
  audio: string;
  start: number;
  end: number | null;
  offset: number;
  duration: number | null;
  gain: number;
  active: boolean;
}

export interface FramePreviewMarker {
  id: string;
  start: number;
  end?: number;
  label?: string;
  kind?: string;
}

export interface FramePreview {
  frame: number;
  duration: number;
  camera: FramePreviewCamera | null;
  activeLayers: FramePreviewLayer[];
  audio: FramePreviewAudio[];
  markers: FramePreviewMarker[];
}

export interface EditResult {
  ok: true;
  changes: EditChange[];
  effects: EditEffect[];
  preview?: FramePreview;
}

export interface EditOptions {
  validate?: 'each' | 'final' | 'none';
  frame?: number;
}

// ---------- 命令类型 ----------

export interface ScenePatch {
  title?: string;
  description?: string | null;
  fps?: number;
  duration?: number;
  width?: number;
  height?: number;
  background?: string | null;
}

export interface LayerPatch {
  name?: string | null;
  in?: number;
  out?: number;
  space?: 'world' | 'screen' | null;
  transform?: Partial<Transform2D>;
  opacity?: AnimNumber | null;
  clip?: Clip | null;
  filters?: Filter[] | null;
  blendMode?: 'normal' | 'add' | 'multiply' | 'screen' | null;
  attach?: ImageAttach | null;
  semantics?: Semantics | null;
  physics?: NonNullable<ModelLayer['physics']> | null;
  frameHold?: NonNullable<ModelLayer['frameHold']> | null;
  model?: string;
}

export interface MotionClipPatch {
  motion?: string;
  start?: number;
  end?: number | null;
  offset?: number | null;
  speed?: number | null;
  loop?: boolean | null;
  weight?: AnimNumber | null;
  fadeIn?: number | null;
  fadeOut?: number | null;
  semantics?: Semantics | null;
  provenance?: Provenance | null;
}

export interface ExpressionClipPatch {
  expression?: string;
  start?: number;
  end?: number | null;
  weight?: AnimNumber | null;
  fadeIn?: number | null;
  fadeOut?: number | null;
  semantics?: Semantics | null;
  provenance?: Provenance | null;
}

export interface AudioClipPatch {
  audio?: string;
  start?: number;
  offset?: number | null;
  duration?: number | null;
  gain?: number | null;
}

export interface CameraPatch {
  x?: AnimNumber;
  y?: AnimNumber;
  zoom?: AnimNumber;
  rotation?: AnimNumber | null;
}

export interface MarkerPatch {
  start?: number;
  end?: number | null;
  label?: string | null;
  kind?: string | null;
  semantics?: Semantics | null;
}

export type KeyframeInput = { t: number; v: number; ease?: Ease };

export type EditCommand =
  | { op: 'setScene' | 'updateScene'; patch: ScenePatch }
  | { op: 'addImageLayer' | 'addImage' | 'addDecoration'; asset: ImageAsset; layer: ImageLayer; index?: number }
  | { op: 'addModelLayer' | 'addModel' | 'addLayer'; layer: ModelLayer; index?: number }
  | { op: 'updateLayer' | 'editLayer' | 'updateDecoration'; id: string; patch: LayerPatch }
  | { op: 'removeLayer' | 'deleteLayer' | 'removeDecoration'; id: string }
  | { op: 'moveLayer' | 'reorderLayer'; id: string; index: number }
  | { op: 'duplicateLayer' | 'cloneLayer'; id: string; newId?: string; name?: string }
  | {
      op: 'addMotion' | 'addMotionClip';
      layer: string;
      uri: string;
      assetId?: string;
      start: number;
      end?: number;
      offset?: number;
      speed?: number;
      loop?: boolean;
      fadeIn?: number;
      fadeOut?: number;
      weight?: AnimNumber;
    }
  | {
      op: 'addExpression' | 'addExpressionClip';
      layer: string;
      uri: string;
      assetId?: string;
      start: number;
      end?: number;
      fadeIn?: number;
      fadeOut?: number;
      weight?: AnimNumber;
    }
  | { op: 'updateMotion' | 'updateMotionClip'; layer: string; index: number; patch: MotionClipPatch }
  | { op: 'updateExpression' | 'updateExpressionClip'; layer: string; index: number; patch: ExpressionClipPatch }
  | { op: 'removeMotion' | 'removeMotionClip' | 'deleteMotion'; layer: string; index: number }
  | { op: 'removeExpression' | 'removeExpressionClip' | 'deleteExpression'; layer: string; index: number }
  | {
      op: 'addAudio' | 'addAudioClip';
      uri: string;
      assetId?: string;
      start?: number;
      offset?: number;
      duration?: number;
      gain?: number;
    }
  | { op: 'updateAudio' | 'updateAudioClip'; index: number; patch: AudioClipPatch }
  | { op: 'removeAudio' | 'removeAudioClip' | 'deleteAudio'; index: number }
  | { op: 'addMarker'; marker: Marker }
  | { op: 'updateMarker'; id: string; patch: MarkerPatch }
  | { op: 'removeMarker' | 'deleteMarker'; id: string }
  | { op: 'setCamera'; camera: CameraTransform | null }
  | { op: 'updateCamera'; patch: CameraPatch }
  | { op: 'setKeyframes' | 'setCurve' | 'setAnimation'; target: EditTarget; keys: KeyframeInput[]; ease?: EasePreset | Ease }
  | { op: 'addKeyframe' | 'addKey'; target: EditTarget; t: number; v: number; ease?: EasePreset | Ease }
  | { op: 'updateKeyframe' | 'updateKey'; target: EditTarget; t: number; v?: number; newT?: number; ease?: EasePreset | Ease }
  | { op: 'removeKeyframe' | 'removeKey' | 'deleteKeyframe'; target: EditTarget; t: number; flatten?: boolean }
  | { op: 'setEase' | 'ease'; target: EditTarget; t?: number; times?: number[]; ease: EasePreset | Ease }
  | {
      op: 'animate' | 'autoAnimate';
      id: string;
      preset: AnimationPreset;
      start?: number;
      end?: number;
      intensity?: number;
      ease?: EasePreset | Ease;
    }
  | { op: 'preview'; frame: number }
  | { op: 'noop' | 'nop' | 'no-op' };

// ---------- 内部工具 ----------

const EPS = 1e-6;
const EFFECT_ORDER: EditEffect[] = ['render', 'workers', 'evaluator', 'structure', 'audio', 'clock'];
const ID_RE = /^[A-Za-z0-9_.:-]+$/;
const TRANSFORM_KEYS = new Set(['x', 'y', 'scale', 'scaleX', 'scaleY', 'rotation']);
const TRANSFORM_DEFAULTS: Record<string, number> = { x: 0, y: 0, scale: 1, scaleX: 1, scaleY: 1, rotation: 0 };
const CLIP_KEYS = new Set(['x', 'y', 'width', 'height', 'radius', 'feather']);
const CLIP_DEFAULTS: Record<string, number> = { x: 0, y: 0, width: 0, height: 0, radius: 0, feather: 0 };
const CAMERA_KEYS = new Set(['x', 'y', 'zoom', 'rotation']);
const CAMERA_DEFAULTS: Record<string, number> = { x: 0, y: 0, zoom: 1, rotation: 0 };
const FILTER_ANIMATABLE: Record<Filter['type'], Record<string, number>> = {
  blur: { sigma: 0 },
  motionBlur: { angle: 0, length: 0 },
  resample: { factor: 1 },
  color: { brightness: 0, contrast: 1, saturation: 1 },
};
const FILTER_REQUIRED: Record<Filter['type'], Set<string>> = {
  blur: new Set(['sigma']),
  motionBlur: new Set(['angle', 'length']),
  resample: new Set(['factor']),
  color: new Set(),
};
const LAYER_COMMON_UPDATE_KEYS = new Set([
  'name',
  'in',
  'out',
  'space',
  'transform',
  'opacity',
  'clip',
  'filters',
  'blendMode',
  'semantics',
]);
const LAYER_MODEL_UPDATE_KEYS = new Set(['physics', 'frameHold', 'model']);
const LAYER_IMAGE_UPDATE_KEYS = new Set(['attach']);
const BLEND_MODES = new Set(['normal', 'add', 'multiply', 'screen']);

const DEFAULT_INTENSITY: Record<AnimationPreset, number> = {
  fadeIn: 1,
  fadeOut: 1,
  float: 20,
  pulse: 0.1,
  spin: 360,
  bounce: 24,
  blink: 1,
  slideIn: 100,
  shake: 12,
  zoomIn: 0.2,
};

const PRESET_EASE: Record<AnimationPreset, Ease> = {
  fadeIn: 'smooth',
  fadeOut: 'smooth',
  float: 'smooth',
  pulse: 'smooth',
  spin: 'linear',
  bounce: 'smooth',
  blink: 'linear',
  slideIn: [0, 0, 0.58, 1],
  shake: 'linear',
  zoomIn: 'smooth',
};

type PathSeg = string | number;

interface EditContext {
  scene: Scene;
  changes: EditChange[];
  effects: Set<EditEffect>;
  previewFrame?: number;
}

interface AnimSlot {
  get(): AnimNumber | undefined;
  set(value: AnimNumber): void;
  remove(): void;
  reset(): void;
  fallback: number;
  optional: boolean;
  label: string;
}

interface NumericKey {
  t: number;
  v: number;
  ease?: Ease;
}

function fail(message: string): never {
  throw new Error(message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function plainObject(value: unknown, what: string): Record<string, unknown> {
  if (!isRecord(value)) fail(`${what} 必须是对象`);
  return value;
}

function clone<T>(value: T): T {
  return value === undefined ? value : structuredClone(value);
}

function checkId(id: unknown, what: string): string {
  if (typeof id !== 'string' || !ID_RE.test(id)) fail(`${what} 非法：${String(id)}`);
  return id;
}

function reqString(value: unknown, what: string): string {
  if (typeof value !== 'string' || value.length === 0) fail(`${what} 必须是非空字符串`);
  return value;
}

function optString(value: unknown, what: string): string | undefined {
  if (value === undefined) return undefined;
  return reqString(value, what);
}

function reqNumber(value: unknown, what: string, opts: { min?: number; positive?: boolean; integer?: boolean } = {}): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(`${what} 必须是有限数字`);
  if (opts.integer && !Number.isInteger(value)) fail(`${what} 必须是整数`);
  if (opts.positive && !(value > 0)) fail(`${what} 必须大于 0`);
  if (opts.min !== undefined && value < opts.min) fail(`${what} 不能小于 ${opts.min}`);
  return value;
}

function optNumber(value: unknown, what: string, opts: { min?: number; positive?: boolean; integer?: boolean } = {}): number | undefined {
  return value === undefined ? undefined : reqNumber(value, what, opts);
}

function frameNumber(value: unknown, what: string): number {
  return reqNumber(value, what, { min: 0 });
}

function animNumber(value: unknown, what: string): AnimNumber {
  if (typeof value === 'number') return reqNumber(value, what);
  if (isRecord(value) && ('keys' in value || 'baked' in value)) return clone(value) as AnimNumber;
  fail(`${what} 必须是数字或动画曲线`);
}

function effectList(set: Set<EditEffect>): EditEffect[] {
  return EFFECT_ORDER.filter((effect) => set.has(effect));
}

function addEffects(ctx: EditContext, ...effects: EditEffect[]): void {
  for (const effect of effects) ctx.effects.add(effect);
}

function pointer(path: PathSeg[]): string {
  return '/' + path.map((seg) => String(seg).replace(/~/g, '~0').replace(/\//g, '~1')).join('/');
}

function pointerSegments(path: string): string[] {
  if (path === '') return [];
  if (!path.startsWith('/')) fail(`非法 JSON Pointer：${path}`);
  return path
    .slice(1)
    .split('/')
    .map((seg) => seg.replace(/~1/g, '/').replace(/~0/g, '~'));
}

function setProp(ctx: EditContext, container: Record<string, unknown> | unknown[], key: PathSeg, value: unknown, base: PathSeg[]): void {
  const path = pointer([...base, key]);
  if (Array.isArray(container)) {
    if (typeof key !== 'number' || !Number.isInteger(key) || key < 0 || key > container.length) {
      fail(`数组下标非法：${String(key)}（${path}）`);
    }
    if (key === container.length) {
      ctx.changes.push({ op: 'add', path, after: clone(value) });
      container.push(value);
    } else {
      ctx.changes.push({ op: 'replace', path, before: clone(container[key]), after: clone(value) });
      container[key] = value;
    }
    return;
  }

  const name = String(key);
  if (Object.prototype.hasOwnProperty.call(container, name)) {
    ctx.changes.push({ op: 'replace', path, before: clone(container[name]), after: clone(value) });
  } else {
    ctx.changes.push({ op: 'add', path, after: clone(value) });
  }
  container[name] = value;
}

function removeProp(ctx: EditContext, container: Record<string, unknown> | unknown[], key: PathSeg, base: PathSeg[]): boolean {
  const path = pointer([...base, key]);
  if (Array.isArray(container)) {
    if (typeof key !== 'number' || !Number.isInteger(key) || key < 0 || key >= container.length) return false;
    ctx.changes.push({ op: 'remove', path, before: clone(container[key]) });
    container.splice(key, 1);
    return true;
  }
  const name = String(key);
  if (!Object.prototype.hasOwnProperty.call(container, name)) return false;
  ctx.changes.push({ op: 'remove', path, before: clone(container[name]) });
  delete container[name];
  return true;
}

function insertArrayItem(ctx: EditContext, array: unknown[], index: number, value: unknown, base: PathSeg[]): void {
  if (!Number.isInteger(index) || index < 0 || index > array.length) {
    fail(`数组插入下标非法：${index}（${pointer(base)}）`);
  }
  ctx.changes.push({ op: 'add', path: pointer([...base, index]), after: clone(value) });
  array.splice(index, 0, value);
}

function ensureArray(ctx: EditContext, container: Record<string, unknown>, key: string, base: PathSeg[]): unknown[] {
  const current = container[key];
  if (current === undefined) {
    setProp(ctx, container, key, [] as unknown[], base);
    return container[key] as unknown[];
  }
  if (!Array.isArray(current)) fail(`${pointer([...base, key])} 不是数组`);
  return current;
}

function findLayer(ctx: EditContext, id: unknown): { layer: Layer; index: number } {
  const layerId = checkId(id, '图层 id');
  const index = ctx.scene.layers.findIndex((layer) => layer.id === layerId);
  if (index < 0) fail(`找不到图层：${layerId}`);
  return { layer: ctx.scene.layers[index]!, index };
}

function uniqueId(base: string, used: Set<string>): string {
  let id = base;
  let n = 2;
  while (used.has(id)) id = `${base}-${n++}`;
  return id;
}

function uriStem(uri: string): string {
  const noQuery = uri.split(/[?#]/, 1)[0] ?? uri;
  const base = noQuery.split(/[\\/]/).pop() ?? noQuery;
  let stem = base.replace(/\.(motion3|exp3)\.json$/i, '').replace(/\.[A-Za-z0-9]+$/, '');
  stem = stem.replace(/[^A-Za-z0-9_.:-]+/g, '_').replace(/^_+|_+$/g, '');
  return stem || 'asset';
}

function assetIdFor(uri: string, requested: unknown, used: Set<string>): string {
  if (requested !== undefined) {
    const id = checkId(requested, 'assetId');
    if (used.has(id)) fail(`资源 ID 已存在：${id}`);
    return id;
  }
  return uniqueId(uriStem(uri), used);
}

function normalizePath(raw: unknown): string {
  if (typeof raw !== 'string' || raw.length === 0) fail('target.path 必须是非空字符串');
  let path = raw.trim();
  if (path.startsWith('/')) path = path.slice(1);
  path = path.replace(/\[(\d+)\]/g, '.$1');
  path = path.replace(/\//g, '.');
  path = path.replace(/^\.+|\.+$/g, '');
  return path;
}

function layerTransformSlot(ctx: EditContext, layer: Layer, layerIndex: number, key: string): AnimSlot {
  const container = layer.transform as unknown as Record<string, unknown>;
  const optional = key === 'scaleX' || key === 'scaleY' || key === 'rotation';
  const fallback = TRANSFORM_DEFAULTS[key]!;
  const base: PathSeg[] = ['layers', layerIndex, 'transform'];
  const set = (value: AnimNumber) => setProp(ctx, container, key, value, base);
  const remove = () => {
    if (!optional) fail(`图层 ${layer.id} 的 transform.${key} 是必填属性，不能删除`);
    removeProp(ctx, container, key, base);
  };
  return {
    get: () => container[key] as AnimNumber | undefined,
    set,
    remove,
    reset: () => (optional ? remove() : set(fallback)),
    fallback,
    optional,
    label: `图层 ${layer.id} 的 transform.${key}`,
  };
}

function layerClipSlot(ctx: EditContext, layer: Layer, layerIndex: number, key: string): AnimSlot {
  if (!layer.clip) fail(`图层 ${layer.id} 没有 clip，请先设置 clip 再编辑 clip.${key}`);
  const clip = layer.clip as unknown as Record<string, unknown>;
  const optional = key === 'radius' || key === 'feather';
  const fallback = CLIP_DEFAULTS[key]!;
  const base: PathSeg[] = ['layers', layerIndex, 'clip'];
  const set = (value: AnimNumber) => setProp(ctx, clip, key, value, base);
  const remove = () => {
    if (!optional) fail(`图层 ${layer.id} 的 clip.${key} 是必填属性，不能删除`);
    removeProp(ctx, clip, key, base);
  };
  return {
    get: () => clip[key] as AnimNumber | undefined,
    set,
    remove,
    reset: () => (optional ? remove() : set(fallback)),
    fallback,
    optional,
    label: `图层 ${layer.id} 的 clip.${key}`,
  };
}

function layerFilterSlot(ctx: EditContext, layer: Layer, layerIndex: number, filterIndex: number, prop: string): AnimSlot {
  const filter = layer.filters?.[filterIndex];
  if (!filter) fail(`图层 ${layer.id} 没有第 ${filterIndex} 个 filter`);
  const allowed = FILTER_ANIMATABLE[filter.type];
  if (!(prop in allowed)) {
    fail(`滤镜 ${filter.type} 不支持动画属性 ${prop}（可用：${Object.keys(allowed).join('、')}）`);
  }
  const container = filter as unknown as Record<string, unknown>;
  const optional = !FILTER_REQUIRED[filter.type].has(prop);
  const fallback = allowed[prop]!;
  const base: PathSeg[] = ['layers', layerIndex, 'filters', filterIndex];
  const set = (value: AnimNumber) => setProp(ctx, container, prop, value, base);
  const remove = () => {
    if (!optional) fail(`滤镜 ${filter.type}.${prop} 是必填属性，不能删除`);
    removeProp(ctx, container, prop, base);
  };
  return {
    get: () => container[prop] as AnimNumber | undefined,
    set,
    remove,
    reset: () => (optional ? remove() : set(fallback)),
    fallback,
    optional,
    label: `图层 ${layer.id} 的 filters[${filterIndex}].${prop}`,
  };
}

function cameraSlot(ctx: EditContext, key: string): AnimSlot {
  const camera = ctx.scene.camera;
  if (!camera) fail('场景没有 camera，请先 setCamera');
  const container = camera as unknown as Record<string, unknown>;
  const optional = key === 'rotation';
  const fallback = CAMERA_DEFAULTS[key]!;
  const base: PathSeg[] = ['camera'];
  const set = (value: AnimNumber) => setProp(ctx, container, key, value, base);
  const remove = () => {
    if (!optional) fail(`camera.${key} 是必填属性，不能删除`);
    removeProp(ctx, container, key, base);
  };
  return {
    get: () => container[key] as AnimNumber | undefined,
    set,
    remove,
    reset: () => (optional ? remove() : set(fallback)),
    fallback,
    optional,
    label: `camera.${key}`,
  };
}

function paramSlot(ctx: EditContext, layer: ModelLayer, layerIndex: number, paramId: string, create: boolean): AnimSlot | null {
  const params = layer.params;
  let trackIndex = params?.findIndex((track) => track.id === paramId) ?? -1;
  let created = false;
  if (trackIndex < 0) {
    if (!create) return null;
    const array = ensureArray(ctx, layer as unknown as Record<string, unknown>, 'params', ['layers', layerIndex]);
    const track: ParamTrack = { id: paramId, curve: { keys: [{ t: 0, v: 0 }] } };
    insertArrayItem(ctx, array, array.length, track, ['layers', layerIndex, 'params']);
    trackIndex = array.length - 1;
    created = true;
    addEffects(ctx, 'evaluator');
  }
  const track = (layer.params as ParamTrack[])[trackIndex]!;
  const base: PathSeg[] = ['layers', layerIndex, 'params', trackIndex];
  const set = (value: AnimNumber) => setProp(ctx, track as unknown as Record<string, unknown>, 'curve', value, base);
  const remove = () => {
    const array = layer.params;
    if (!array) return;
    if (removeProp(ctx, array as unknown as unknown[], trackIndex, ['layers', layerIndex, 'params'])) {
      addEffects(ctx, 'evaluator');
    }
  };
  return {
    get: () => track.curve as AnimNumber,
    set,
    remove,
    reset: remove,
    fallback: 0,
    optional: false,
    label: `图层 ${layer.id} 的参数 ${paramId}${created ? '（新建）' : ''}`,
  };
}

function resolveTarget(ctx: EditContext, target: unknown, options: { create: boolean }): AnimSlot | null {
  const t = plainObject(target, 'target');
  let kind = t.kind;
  if (kind === undefined) {
    if (t.param !== undefined) kind = 'param';
    else if ((t.id !== undefined || t.layer !== undefined) && t.path !== undefined) kind = 'layer';
    else if (t.path !== undefined) kind = 'camera';
  }
  if (kind === 'layer') {
    const id = t.id ?? t.layer;
    const { layer, index } = findLayer(ctx, id);
    const raw = normalizePath(t.path);
    const path = raw.startsWith('layer.') ? raw.slice('layer.'.length) : raw;

    if (path === 'opacity') {
      const layerRecord = layer as unknown as Record<string, unknown>;
      const base: PathSeg[] = ['layers', index];
      return {
        get: () => layer.opacity as AnimNumber | undefined,
        set: (value) => setProp(ctx, layerRecord, 'opacity', value, base),
        remove: () => {
          removeProp(ctx, layerRecord, 'opacity', base);
        },
        reset: () => removeProp(ctx, layerRecord, 'opacity', base),
        fallback: 1,
        optional: true,
        label: `图层 ${layer.id} 的 opacity`,
      };
    }

    const transformKey = path.startsWith('transform.') ? path.slice('transform.'.length) : path;
    if (TRANSFORM_KEYS.has(transformKey) && (path === transformKey || path.startsWith('transform.'))) {
      return layerTransformSlot(ctx, layer, index, transformKey);
    }

    if (path.startsWith('clip.')) {
      const key = path.slice('clip.'.length);
      if (!CLIP_KEYS.has(key)) fail(`图层 ${layer.id} 不支持 clip.${key}（可用：${[...CLIP_KEYS].join('、')}）`);
      return layerClipSlot(ctx, layer, index, key);
    }

    const filterMatch = /^(?:filters?)\.(\d+)\.([A-Za-z]+)$/.exec(path);
    if (filterMatch) {
      const filterIndex = Number(filterMatch[1]);
      const prop = filterMatch[2]!;
      if (!Number.isInteger(filterIndex) || filterIndex < 0) fail(`filter 下标非法：${filterMatch[1]}`);
      return layerFilterSlot(ctx, layer, index, filterIndex, prop);
    }

    fail(
      `图层 ${layer.id} 不支持的动画路径：${path}` +
        '（可用：transform.x/y/scale/scaleX/scaleY/rotation、opacity、clip.*、filters.<i>.*）',
    );
  }

  if (kind === 'camera') {
    if (!ctx.scene.camera) fail('场景没有 camera，请先 setCamera');
    const raw = normalizePath(t.path);
    const path = raw.startsWith('camera.') ? raw.slice('camera.'.length) : raw;
    if (!CAMERA_KEYS.has(path)) fail(`camera 不支持的动画路径：${path}（可用：x、y、zoom、rotation）`);
    return cameraSlot(ctx, path);
  }

  if (kind === 'param') {
    const id = t.id ?? t.layer;
    const param = reqString(t.param, 'target.param');
    const { layer, index } = findLayer(ctx, id);
    if (layer.kind !== 'model') fail(`参数轨只能挂在模型图层上，${layer.id} 是 ${layer.kind} 图层`);
    return paramSlot(ctx, layer, index, param, options.create);
  }

  fail(`未知 target.kind：${String(kind)}`);
}

function requireSlot(ctx: EditContext, target: unknown, create: boolean): AnimSlot {
  const slot = resolveTarget(ctx, target, { create });
  if (!slot) fail(`参数轨不存在：${JSON.stringify(target)}`);
  return slot;
}

function addAnimTargetEffects(ctx: EditContext, target: unknown): void {
  const t = plainObject(target, 'target');
  addEffects(ctx, 'render');
  if (t.kind === 'param' || (t.kind === undefined && t.param !== undefined)) addEffects(ctx, 'workers');
}

function normalizeEase(value: unknown, what = 'ease'): Ease {
  if (Array.isArray(value)) {
    if (value.length !== 4 || value.some((n) => typeof n !== 'number' || !Number.isFinite(n))) {
      fail(`${what} 必须是 [x1,y1,x2,y2] 四个有限数字`);
    }
    return [...value] as Ease;
  }
  if (typeof value !== 'string') fail(`${what} 必须是缓动预设或贝塞尔数组`);
  return easePreset(value as EasePreset);
}

function keyOf(t: number, v: number, ease: Ease | undefined): NumericKey {
  const key: NumericKey = { t, v };
  if (ease !== undefined) key.ease = Array.isArray(ease) ? ([...ease] as Ease) : ease;
  return key;
}

function normalizeKeyframes(input: unknown, defaultEase: Ease | undefined): NumericKey[] {
  if (!Array.isArray(input)) fail('keys 必须是数组');
  const entries = input.map((raw, order) => {
    const key = plainObject(raw, `keys[${order}]`);
    const t = frameNumber(key.t ?? key.time, `keys[${order}].t`);
    const v = reqNumber(key.v ?? key.value, `keys[${order}].v`);
    const ease = key.ease == null ? defaultEase : normalizeEase(key.ease, `keys[${order}].ease`);
    return { order, key: keyOf(t, v, ease) };
  });
  entries.sort((a, b) => a.key.t - b.key.t || a.order - b.order);
  const out: NumericKey[] = [];
  for (const entry of entries) {
    const last = out[out.length - 1];
    if (last && Math.abs(entry.key.t - last.t) <= EPS) out[out.length - 1] = entry.key;
    else out.push(entry.key);
  }
  return out;
}

function expandBaked(curve: Extract<Curve<number>, { baked: unknown }>): NumericKey[] {
  return curve.baked.values.map((v, i) => ({ t: curve.baked.start + i, v }));
}

function editableKeys(value: AnimNumber, label: string): NumericKey[] {
  if (typeof value === 'number') fail(`${label} 是常量，没有关键帧`);
  if ('baked' in value) return expandBaked(value);
  return value.keys.map((key) => keyOf(key.t, key.v, key.ease));
}

function findKeyAt(keys: NumericKey[], t: number): number {
  return keys.findIndex((key) => Math.abs(key.t - t) <= EPS);
}

function mergeAnimated(
  current: AnimNumber | undefined,
  start: number,
  end: number,
  generated: NumericKey[],
): AnimNumber {
  if (current === undefined || typeof current === 'number') {
    return { keys: generated.map((key) => keyOf(key.t, key.v, key.ease)) };
  }
  const existing = 'baked' in current ? expandBaked(current) : current.keys.map((key) => keyOf(key.t, key.v, key.ease));
  const outside = existing.filter((key) => key.t < start - EPS || key.t > end + EPS);
  const keys = [...outside, ...generated].sort((a, b) => a.t - b.t);
  return { keys };
}

// ---------- change 应用/撤销 ----------

function resolveParent(root: unknown, path: string): { parent: Record<string, unknown> | unknown[]; key: string } {
  const segments = pointerSegments(path);
  if (segments.length === 0) fail('编辑路径不能为空');
  let current: unknown = root;
  for (let i = 0; i < segments.length - 1; i++) {
    const segment = segments[i]!;
    if (current === null || typeof current !== 'object') fail(`编辑路径不存在：${path}`);
    if (Array.isArray(current)) {
      const index = Number(segment);
      if (!Number.isInteger(index) || index < 0 || index >= current.length) fail(`编辑路径不存在：${path}`);
      current = current[index];
    } else {
      if (!(segment in current)) fail(`编辑路径不存在：${path}`);
      current = (current as Record<string, unknown>)[segment];
    }
  }
  if (current === null || typeof current !== 'object') fail(`编辑路径不存在：${path}`);
  return { parent: current as Record<string, unknown> | unknown[], key: segments[segments.length - 1]! };
}

function addAt(root: unknown, path: string, value: unknown): void {
  const { parent, key } = resolveParent(root, path);
  if (Array.isArray(parent)) {
    const index = Number(key);
    if (!Number.isInteger(index) || index < 0 || index > parent.length) fail(`编辑路径不存在：${path}`);
    parent.splice(index, 0, clone(value));
  } else {
    parent[key] = clone(value);
  }
}

function removeAt(root: unknown, path: string): void {
  const { parent, key } = resolveParent(root, path);
  if (Array.isArray(parent)) {
    const index = Number(key);
    if (!Number.isInteger(index) || index < 0 || index >= parent.length) fail(`编辑路径不存在：${path}`);
    parent.splice(index, 1);
  } else {
    delete parent[key];
  }
}

function replaceAt(root: unknown, path: string, value: unknown): void {
  const { parent, key } = resolveParent(root, path);
  if (Array.isArray(parent)) {
    const index = Number(key);
    if (!Number.isInteger(index) || index < 0 || index >= parent.length) fail(`编辑路径不存在：${path}`);
    parent[index] = clone(value);
  } else {
    parent[key] = clone(value);
  }
}

/** 正向应用 changes（add 在数组中是插入，和 JSON Patch 一致）。 */
export function applyEditChanges(scene: Scene, changes: EditChange[]): void {
  for (const change of changes) {
    if (change.op === 'add') addAt(scene, change.path, change.after);
    else if (change.op === 'remove') removeAt(scene, change.path);
    else replaceAt(scene, change.path, change.after);
  }
}

/** 反向撤销 changes；调用方通常传入 applyEditCommands 返回的 changes。 */
export function undoEditChanges(scene: Scene, changes: EditChange[]): void {
  for (let i = changes.length - 1; i >= 0; i--) {
    const change = changes[i]!;
    if (change.op === 'add') removeAt(scene, change.path);
    else if (change.op === 'remove') addAt(scene, change.path, change.before);
    else replaceAt(scene, change.path, change.before);
  }
}

// ---------- 预览 ----------

/** 在指定帧生成编辑器/AI 可读的摘要；不修改场景。 */
export function previewAt(scene: Scene, frame: number): FramePreview {
  frame = frameNumber(frame, 'preview.frame');
  const fps = scene.timebase.fps;
  const camera = scene.camera
    ? {
        x: sample(scene.camera.x, frame, 0),
        y: sample(scene.camera.y, frame, 0),
        zoom: sample(scene.camera.zoom, frame, 1),
        rotation: sample(scene.camera.rotation, frame, 0),
      }
    : null;

  const activeLayers: FramePreviewLayer[] = [];
  for (const layer of scene.layers) {
    if (!(layer.in <= frame && frame < layer.out)) continue;
    const t = layer.transform;
    const entry: FramePreviewLayer = {
      id: layer.id,
      kind: layer.kind,
      x: sample(t.x, frame, 0),
      y: sample(t.y, frame, 0),
      scale: sample(t.scale, frame, 1),
      scaleX: sample(t.scaleX, frame, 1),
      scaleY: sample(t.scaleY, frame, 1),
      rotation: sample(t.rotation, frame, 0),
      opacity: sample(layer.opacity, frame, 1),
    };
    if (layer.name !== undefined) entry.name = layer.name;
    if (layer.kind === 'model') entry.model = layer.model;
    else entry.image = layer.image;
    if (layer.blendMode !== undefined) entry.blendMode = layer.blendMode;
    activeLayers.push(entry);
  }

  const audio: FramePreviewAudio[] = (scene.audio ?? []).map((clip, index) => {
    const duration = clip.duration ?? null;
    const end = duration === null ? null : clip.start + duration * fps;
    return {
      index,
      audio: clip.audio,
      start: clip.start,
      end,
      offset: clip.offset ?? 0,
      duration,
      gain: clip.gain ?? 1,
      active: frame >= clip.start && (end === null || frame < end),
    };
  });

  const markers: FramePreviewMarker[] = [];
  for (const marker of scene.markers ?? []) {
    if (frame < marker.start || (marker.end !== undefined && frame >= marker.end)) continue;
    const entry: FramePreviewMarker = { id: marker.id, start: marker.start };
    if (marker.end !== undefined) entry.end = marker.end;
    if (marker.label !== undefined) entry.label = marker.label;
    if (marker.kind !== undefined) entry.kind = marker.kind;
    markers.push(entry);
  }

  return { frame, duration: scene.duration, camera, activeLayers, audio, markers };
}

// ---------- 场景命令 ----------

function applySetScene(ctx: EditContext, cmd: Record<string, unknown>): void {
  const patch = plainObject(cmd.patch, 'setScene.patch');
  const allowed = new Set(['title', 'description', 'fps', 'duration', 'width', 'height', 'background']);
  for (const key of Object.keys(patch)) if (!allowed.has(key)) fail(`setScene 不允许修改字段：${key}`);

  if ('title' in patch) {
    const title = reqString(patch.title, 'setScene.title');
    setProp(ctx, ctx.scene.meta as unknown as Record<string, unknown>, 'title', title, ['meta']);
    addEffects(ctx, 'render');
  }
  if ('description' in patch) {
    if (patch.description === null) {
      removeProp(ctx, ctx.scene.meta as unknown as Record<string, unknown>, 'description', ['meta']);
    } else {
      const description = reqString(patch.description, 'setScene.description');
      setProp(ctx, ctx.scene.meta as unknown as Record<string, unknown>, 'description', description, ['meta']);
    }
    addEffects(ctx, 'render');
  }
  if ('fps' in patch) {
    const fps = reqNumber(patch.fps, 'setScene.fps', { positive: true });
    setProp(ctx, ctx.scene.timebase as unknown as Record<string, unknown>, 'fps', fps, ['timebase']);
    addEffects(ctx, 'audio', 'clock');
  }
  if ('duration' in patch) {
    const duration = reqNumber(patch.duration, 'setScene.duration', { positive: true, integer: true });
    setProp(ctx, ctx.scene as unknown as Record<string, unknown>, 'duration', duration, []);
    addEffects(ctx, 'audio', 'clock');
  }
  if ('width' in patch) {
    const width = reqNumber(patch.width, 'setScene.width', { positive: true, integer: true });
    setProp(ctx, ctx.scene.composition as unknown as Record<string, unknown>, 'width', width, ['composition']);
    addEffects(ctx, 'structure', 'render');
  }
  if ('height' in patch) {
    const height = reqNumber(patch.height, 'setScene.height', { positive: true, integer: true });
    setProp(ctx, ctx.scene.composition as unknown as Record<string, unknown>, 'height', height, ['composition']);
    addEffects(ctx, 'structure', 'render');
  }
  if ('background' in patch) {
    if (patch.background !== null && typeof patch.background !== 'string') fail('setScene.background 必须是字符串或 null');
    setProp(
      ctx,
      ctx.scene.composition as unknown as Record<string, unknown>,
      'background',
      patch.background,
      ['composition'],
    );
    addEffects(ctx, 'render');
  }
}

function addImageLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const asset = clone(plainObject(cmd.asset, 'addImageLayer.asset')) as unknown as ImageAsset;
  const layer = clone(plainObject(cmd.layer, 'addImageLayer.layer')) as unknown as ImageLayer;
  checkId(asset.id, 'asset.id');
  checkId(layer.id, 'layer.id');
  if (layer.kind !== 'image') fail('addImageLayer.layer.kind 必须是 "image"');
  if (!ctx.scene.layers.every((item) => item.id !== layer.id)) fail(`图层 ID 已存在：${layer.id}`);
  if ((ctx.scene.assets.images ?? []).some((item) => item.id === asset.id)) fail(`图片资源 ID 已存在：${asset.id}`);
  if (layer.image === undefined) layer.image = asset.id;
  else if (layer.image !== asset.id) fail(`图层 ${layer.id} 的 image 必须等于 asset.id（${asset.id}）`);

  const index = cmd.index === undefined ? ctx.scene.layers.length : reqNumber(cmd.index, 'addImageLayer.index', { integer: true, min: 0 });
  if (index > ctx.scene.layers.length) fail(`addImageLayer.index 超出范围：${index}`);
  const images = ensureArray(ctx, ctx.scene.assets as unknown as Record<string, unknown>, 'images', ['assets']);
  insertArrayItem(ctx, images, images.length, asset, ['assets', 'images']);
  insertArrayItem(ctx, ctx.scene.layers as unknown as unknown[], index, layer, ['layers']);
  addEffects(ctx, 'structure', 'render');
}

function addModelLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const layer = clone(plainObject(cmd.layer, 'addModelLayer.layer')) as unknown as ModelLayer;
  checkId(layer.id, 'layer.id');
  if (layer.kind !== 'model') fail('addModelLayer.layer.kind 必须是 "model"');
  if (!ctx.scene.layers.every((item) => item.id !== layer.id)) fail(`图层 ID 已存在：${layer.id}`);
  if (!ctx.scene.assets.models.some((asset) => asset.id === layer.model)) fail(`模型资源不存在：${layer.model}`);
  const index = cmd.index === undefined ? ctx.scene.layers.length : reqNumber(cmd.index, 'addModelLayer.index', { integer: true, min: 0 });
  if (index > ctx.scene.layers.length) fail(`addModelLayer.index 超出范围：${index}`);
  insertArrayItem(ctx, ctx.scene.layers as unknown as unknown[], index, layer, ['layers']);
  addEffects(ctx, 'structure', 'render');
}

function updateLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer, index } = findLayer(ctx, cmd.id);
  const patch = plainObject(cmd.patch, 'updateLayer.patch');
  const allowed = new Set([...LAYER_COMMON_UPDATE_KEYS]);
  if (layer.kind === 'model') for (const key of LAYER_MODEL_UPDATE_KEYS) allowed.add(key);
  else for (const key of LAYER_IMAGE_UPDATE_KEYS) allowed.add(key);
  for (const key of Object.keys(patch)) {
    if (!allowed.has(key)) {
      if (key === 'id' || key === 'kind') fail(`图层 ${key} 不可修改`);
      fail(`updateLayer 不允许修改字段：${key}`);
    }
  }

  const record = layer as unknown as Record<string, unknown>;
  const base: PathSeg[] = ['layers', index];
  const evaluatorFields = new Set(['in', 'out', 'model', 'physics', 'frameHold']);

  if ('name' in patch) {
    if (patch.name === null) removeProp(ctx, record, 'name', base);
    else setProp(ctx, record, 'name', reqString(patch.name, 'updateLayer.name'), base);
  }
  if ('in' in patch) setProp(ctx, record, 'in', frameNumber(patch.in, 'updateLayer.in'), base);
  if ('out' in patch) setProp(ctx, record, 'out', frameNumber(patch.out, 'updateLayer.out'), base);
  if ('space' in patch) {
    if (patch.space === null) removeProp(ctx, record, 'space', base);
    else {
      if (patch.space !== 'world' && patch.space !== 'screen') fail('updateLayer.space 必须是 "world"、"screen" 或 null');
      setProp(ctx, record, 'space', patch.space, base);
    }
  }
  if ('transform' in patch) {
    const transformPatch = plainObject(patch.transform, 'updateLayer.transform');
    const transform = layer.transform as unknown as Record<string, unknown>;
    for (const key of Object.keys(transformPatch)) {
      const value = transformPatch[key];
      if (!TRANSFORM_KEYS.has(key) && !['anchorX', 'anchorY', 'flipX'].includes(key)) {
        fail(`updateLayer.transform 不允许修改字段：${key}`);
      }
      if (value === null) {
        if (key === 'x' || key === 'y' || key === 'scale') fail(`transform.${key} 是必填属性，不能设为 null`);
        removeProp(ctx, transform, key, ['layers', index, 'transform']);
      } else if (key === 'anchorX' || key === 'anchorY') {
        setProp(ctx, transform, key, reqNumber(value, `updateLayer.transform.${key}`), ['layers', index, 'transform']);
      } else if (key === 'flipX') {
        if (typeof value !== 'boolean') fail('updateLayer.transform.flipX 必须是布尔值');
        setProp(ctx, transform, key, value, ['layers', index, 'transform']);
      } else {
        setProp(ctx, transform, key, animNumber(value, `updateLayer.transform.${key}`), ['layers', index, 'transform']);
      }
    }
  }
  if ('opacity' in patch) {
    if (patch.opacity === null) removeProp(ctx, record, 'opacity', base);
    else setProp(ctx, record, 'opacity', animNumber(patch.opacity, 'updateLayer.opacity'), base);
  }
  if ('clip' in patch) {
    if (patch.clip === null) removeProp(ctx, record, 'clip', base);
    else setProp(ctx, record, 'clip', clone(plainObject(patch.clip, 'updateLayer.clip')), base);
  }
  if ('filters' in patch) {
    if (patch.filters === null) removeProp(ctx, record, 'filters', base);
    else {
      if (!Array.isArray(patch.filters)) fail('updateLayer.filters 必须是数组或 null');
      setProp(ctx, record, 'filters', clone(patch.filters), base);
    }
  }
  if ('blendMode' in patch) {
    if (patch.blendMode === null) removeProp(ctx, record, 'blendMode', base);
    else {
      if (typeof patch.blendMode !== 'string' || !BLEND_MODES.has(patch.blendMode)) fail('updateLayer.blendMode 非法');
      setProp(ctx, record, 'blendMode', patch.blendMode, base);
    }
  }
  if ('attach' in patch) {
    if (patch.attach === null) removeProp(ctx, record, 'attach', base);
    else setProp(ctx, record, 'attach', clone(plainObject(patch.attach, 'updateLayer.attach')), base);
  }
  if ('semantics' in patch) {
    if (patch.semantics === null) removeProp(ctx, record, 'semantics', base);
    else setProp(ctx, record, 'semantics', clone(plainObject(patch.semantics, 'updateLayer.semantics')), base);
  }
  if ('physics' in patch) {
    if (patch.physics === null) removeProp(ctx, record, 'physics', base);
    else setProp(ctx, record, 'physics', clone(plainObject(patch.physics, 'updateLayer.physics')), base);
  }
  if ('frameHold' in patch) {
    if (patch.frameHold === null) removeProp(ctx, record, 'frameHold', base);
    else {
      if (!Array.isArray(patch.frameHold)) fail('updateLayer.frameHold 必须是数组或 null');
      setProp(ctx, record, 'frameHold', clone(patch.frameHold), base);
    }
  }
  if ('model' in patch) {
    if (layer.kind !== 'model') fail('图片图层不能修改 model');
    const model = reqString(patch.model, 'updateLayer.model');
    if (!ctx.scene.assets.models.some((asset) => asset.id === model)) fail(`模型资源不存在：${model}`);
    setProp(ctx, record, 'model', model, base);
  }

  addEffects(ctx, 'render');
  for (const key of Object.keys(patch)) if (evaluatorFields.has(key)) addEffects(ctx, 'evaluator');
}

function removeLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer, index } = findLayer(ctx, cmd.id);
  const imageId = layer.kind === 'image' ? layer.image : null;
  removeProp(ctx, ctx.scene.layers as unknown as unknown[], index, ['layers']);
  if (imageId !== null && !ctx.scene.layers.some((item) => item.kind === 'image' && item.image === imageId)) {
    const images = ctx.scene.assets.images;
    const imageIndex = images?.findIndex((asset) => asset.id === imageId) ?? -1;
    if (images && imageIndex >= 0) removeProp(ctx, images as unknown as unknown[], imageIndex, ['assets', 'images']);
  }
  addEffects(ctx, 'structure', 'render');
}

function moveLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer, index } = findLayer(ctx, cmd.id);
  const target = reqNumber(cmd.index, 'moveLayer.index', { integer: true, min: 0 });
  if (target >= ctx.scene.layers.length) fail(`moveLayer.index 超出范围：${target}`);
  if (target === index) return;
  removeProp(ctx, ctx.scene.layers as unknown as unknown[], index, ['layers']);
  insertArrayItem(ctx, ctx.scene.layers as unknown as unknown[], target, layer, ['layers']);
  addEffects(ctx, 'structure', 'render');
}

function duplicateLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer, index } = findLayer(ctx, cmd.id);
  const copy = structuredClone(layer) as Layer;
  if (cmd.newId !== undefined) {
    copy.id = checkId(cmd.newId, 'duplicateLayer.newId');
    if (ctx.scene.layers.some((item) => item.id === copy.id)) fail(`图层 ID 已存在：${copy.id}`);
  } else {
    const used = new Set(ctx.scene.layers.map((item) => item.id));
    copy.id = uniqueId(`${layer.id}-copy`, used);
  }
  if (cmd.name !== undefined) {
    if (typeof cmd.name !== 'string') fail('duplicateLayer.name 必须是字符串');
    copy.name = cmd.name;
  }
  insertArrayItem(ctx, ctx.scene.layers as unknown as unknown[], index + 1, copy, ['layers']);
  addEffects(ctx, 'structure', 'render');
}

// ---------- motion / expression ----------

function findOrCreateMotionAsset(ctx: EditContext, layer: ModelLayer, uri: string, assetId: unknown): MotionAsset {
  const assets = ensureArray(ctx, ctx.scene.assets as unknown as Record<string, unknown>, 'motions', ['assets']) as MotionAsset[];
  const existing = assets.find((asset) => asset.uri === uri);
  if (existing) return existing;
  const used = new Set(assets.map((asset) => asset.id));
  const asset: MotionAsset = { id: assetIdFor(uri, assetId, used), model: layer.model, uri };
  insertArrayItem(ctx, assets as unknown as unknown[], assets.length, asset, ['assets', 'motions']);
  addEffects(ctx, 'structure');
  return asset;
}

function findOrCreateExpressionAsset(ctx: EditContext, layer: ModelLayer, uri: string, assetId: unknown): ExpressionAsset {
  const assets = ensureArray(ctx, ctx.scene.assets as unknown as Record<string, unknown>, 'expressions', ['assets']) as ExpressionAsset[];
  const existing = assets.find((asset) => asset.uri === uri);
  if (existing) return existing;
  const used = new Set(assets.map((asset) => asset.id));
  const asset: ExpressionAsset = { id: assetIdFor(uri, assetId, used), model: layer.model, uri };
  insertArrayItem(ctx, assets as unknown as unknown[], assets.length, asset, ['assets', 'expressions']);
  addEffects(ctx, 'structure');
  return asset;
}

function addMotion(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer, index } = findLayer(ctx, cmd.layer ?? cmd.id);
  if (layer.kind !== 'model') fail(`motion 只能加到模型图层，${layer.id} 是 ${layer.kind} 图层`);
  const uri = reqString(cmd.uri, 'addMotion.uri');
  const start = frameNumber(cmd.start, 'addMotion.start');
  const end = optNumber(cmd.end, 'addMotion.end', { min: 0 });
  if (end !== undefined && !(end > start)) fail('addMotion.end 必须大于 start');
  const asset = findOrCreateMotionAsset(ctx, layer, uri, cmd.assetId);
  const clip: MotionClip = { motion: asset.id, start };
  if (end !== undefined) clip.end = end;
  if (cmd.offset !== undefined) clip.offset = reqNumber(cmd.offset, 'addMotion.offset');
  if (cmd.speed !== undefined) clip.speed = reqNumber(cmd.speed, 'addMotion.speed', { positive: true });
  if (cmd.loop !== undefined) {
    if (typeof cmd.loop !== 'boolean') fail('addMotion.loop 必须是布尔值');
    clip.loop = cmd.loop;
  }
  if (cmd.fadeIn !== undefined) clip.fadeIn = reqNumber(cmd.fadeIn, 'addMotion.fadeIn', { min: 0 });
  if (cmd.fadeOut !== undefined) clip.fadeOut = reqNumber(cmd.fadeOut, 'addMotion.fadeOut', { min: 0 });
  if (cmd.weight !== undefined) clip.weight = animNumber(cmd.weight, 'addMotion.weight');
  const clips = ensureArray(ctx, layer as unknown as Record<string, unknown>, 'motions', ['layers', index]);
  insertArrayItem(ctx, clips, clips.length, clip, ['layers', index, 'motions']);
  addEffects(ctx, 'evaluator', 'render');
}

function addExpression(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer, index } = findLayer(ctx, cmd.layer ?? cmd.id);
  if (layer.kind !== 'model') fail(`expression 只能加到模型图层，${layer.id} 是 ${layer.kind} 图层`);
  const uri = reqString(cmd.uri, 'addExpression.uri');
  const start = frameNumber(cmd.start, 'addExpression.start');
  const end = optNumber(cmd.end, 'addExpression.end', { min: 0 });
  if (end !== undefined && !(end > start)) fail('addExpression.end 必须大于 start');
  const asset = findOrCreateExpressionAsset(ctx, layer, uri, cmd.assetId);
  const clip: ExpressionClip = { expression: asset.id, start };
  if (end !== undefined) clip.end = end;
  if (cmd.fadeIn !== undefined) clip.fadeIn = reqNumber(cmd.fadeIn, 'addExpression.fadeIn', { min: 0 });
  if (cmd.fadeOut !== undefined) clip.fadeOut = reqNumber(cmd.fadeOut, 'addExpression.fadeOut', { min: 0 });
  if (cmd.weight !== undefined) clip.weight = animNumber(cmd.weight, 'addExpression.weight');
  const clips = ensureArray(ctx, layer as unknown as Record<string, unknown>, 'expressions', ['layers', index]);
  insertArrayItem(ctx, clips, clips.length, clip, ['layers', index, 'expressions']);
  addEffects(ctx, 'evaluator', 'render');
}

function clipAt(ctx: EditContext, layer: Layer, list: 'motions' | 'expressions', rawIndex: unknown): { clips: unknown[]; index: number; clip: MotionClip | ExpressionClip } {
  const index = reqNumber(rawIndex, `${list} index`, { integer: true, min: 0 });
  if (layer.kind !== 'model') fail(`图层 ${layer.id} 不是模型图层`);
  const clips = list === 'motions' ? layer.motions : layer.expressions;
  if (!clips) fail(`图层 ${layer.id} 没有 ${list}`);
  if (index >= clips.length) fail(`图层 ${layer.id} 的 ${list} 下标越界：${index}`);
  return { clips: clips as unknown as unknown[], index, clip: clips[index]! };
}

function updateMotion(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer } = findLayer(ctx, cmd.layer ?? cmd.id);
  const { clips, index, clip } = clipAt(ctx, layer, 'motions', cmd.index);
  const patch = plainObject(cmd.patch, 'updateMotion.patch');
  const allowed = new Set(['motion', 'start', 'end', 'offset', 'speed', 'loop', 'weight', 'fadeIn', 'fadeOut', 'semantics', 'provenance']);
  for (const key of Object.keys(patch)) if (!allowed.has(key)) fail(`updateMotion 不允许修改字段：${key}`);
  const base: PathSeg[] = ['layers', ctx.scene.layers.indexOf(layer), 'motions', index];
  const record = clip as unknown as Record<string, unknown>;
  if ('motion' in patch) {
    const id = reqString(patch.motion, 'updateMotion.motion');
    if (!(ctx.scene.assets.motions ?? []).some((asset) => asset.id === id)) fail(`motion 资源不存在：${id}`);
    setProp(ctx, record, 'motion', id, base);
  }
  if ('start' in patch) setProp(ctx, record, 'start', frameNumber(patch.start, 'updateMotion.start'), base);
  if ('end' in patch) {
    if (patch.end === null) removeProp(ctx, record, 'end', base);
    else setProp(ctx, record, 'end', frameNumber(patch.end, 'updateMotion.end'), base);
  }
  if ('offset' in patch) {
    if (patch.offset === null) removeProp(ctx, record, 'offset', base);
    else setProp(ctx, record, 'offset', reqNumber(patch.offset, 'updateMotion.offset'), base);
  }
  if ('speed' in patch) {
    if (patch.speed === null) removeProp(ctx, record, 'speed', base);
    else setProp(ctx, record, 'speed', reqNumber(patch.speed, 'updateMotion.speed', { positive: true }), base);
  }
  if ('loop' in patch) {
    if (patch.loop === null) removeProp(ctx, record, 'loop', base);
    else {
      if (typeof patch.loop !== 'boolean') fail('updateMotion.loop 必须是布尔值或 null');
      setProp(ctx, record, 'loop', patch.loop, base);
    }
  }
  if ('weight' in patch) {
    if (patch.weight === null) removeProp(ctx, record, 'weight', base);
    else setProp(ctx, record, 'weight', animNumber(patch.weight, 'updateMotion.weight'), base);
  }
  if ('fadeIn' in patch) {
    if (patch.fadeIn === null) removeProp(ctx, record, 'fadeIn', base);
    else setProp(ctx, record, 'fadeIn', reqNumber(patch.fadeIn, 'updateMotion.fadeIn', { min: 0 }), base);
  }
  if ('fadeOut' in patch) {
    if (patch.fadeOut === null) removeProp(ctx, record, 'fadeOut', base);
    else setProp(ctx, record, 'fadeOut', reqNumber(patch.fadeOut, 'updateMotion.fadeOut', { min: 0 }), base);
  }
  if ('semantics' in patch) {
    if (patch.semantics === null) removeProp(ctx, record, 'semantics', base);
    else setProp(ctx, record, 'semantics', clone(plainObject(patch.semantics, 'updateMotion.semantics')), base);
  }
  if ('provenance' in patch) {
    if (patch.provenance === null) removeProp(ctx, record, 'provenance', base);
    else setProp(ctx, record, 'provenance', clone(plainObject(patch.provenance, 'updateMotion.provenance')), base);
  }
  const updated = clips[index] as unknown as MotionClip;
  if (updated.end !== undefined && !(updated.end > updated.start)) fail('motion.end 必须大于 start');
  addEffects(ctx, 'evaluator', 'render');
}

function updateExpression(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer } = findLayer(ctx, cmd.layer ?? cmd.id);
  const { clips, index, clip } = clipAt(ctx, layer, 'expressions', cmd.index);
  const patch = plainObject(cmd.patch, 'updateExpression.patch');
  const allowed = new Set(['expression', 'start', 'end', 'weight', 'fadeIn', 'fadeOut', 'semantics', 'provenance']);
  for (const key of Object.keys(patch)) if (!allowed.has(key)) fail(`updateExpression 不允许修改字段：${key}`);
  const base: PathSeg[] = ['layers', ctx.scene.layers.indexOf(layer), 'expressions', index];
  const record = clip as unknown as Record<string, unknown>;
  if ('expression' in patch) {
    const id = reqString(patch.expression, 'updateExpression.expression');
    if (!(ctx.scene.assets.expressions ?? []).some((asset) => asset.id === id)) fail(`expression 资源不存在：${id}`);
    setProp(ctx, record, 'expression', id, base);
  }
  if ('start' in patch) setProp(ctx, record, 'start', frameNumber(patch.start, 'updateExpression.start'), base);
  if ('end' in patch) {
    if (patch.end === null) removeProp(ctx, record, 'end', base);
    else setProp(ctx, record, 'end', frameNumber(patch.end, 'updateExpression.end'), base);
  }
  if ('weight' in patch) {
    if (patch.weight === null) removeProp(ctx, record, 'weight', base);
    else setProp(ctx, record, 'weight', animNumber(patch.weight, 'updateExpression.weight'), base);
  }
  if ('fadeIn' in patch) {
    if (patch.fadeIn === null) removeProp(ctx, record, 'fadeIn', base);
    else setProp(ctx, record, 'fadeIn', reqNumber(patch.fadeIn, 'updateExpression.fadeIn', { min: 0 }), base);
  }
  if ('fadeOut' in patch) {
    if (patch.fadeOut === null) removeProp(ctx, record, 'fadeOut', base);
    else setProp(ctx, record, 'fadeOut', reqNumber(patch.fadeOut, 'updateExpression.fadeOut', { min: 0 }), base);
  }
  if ('semantics' in patch) {
    if (patch.semantics === null) removeProp(ctx, record, 'semantics', base);
    else setProp(ctx, record, 'semantics', clone(plainObject(patch.semantics, 'updateExpression.semantics')), base);
  }
  if ('provenance' in patch) {
    if (patch.provenance === null) removeProp(ctx, record, 'provenance', base);
    else setProp(ctx, record, 'provenance', clone(plainObject(patch.provenance, 'updateExpression.provenance')), base);
  }
  const updated = clips[index] as unknown as ExpressionClip;
  if (updated.end !== undefined && !(updated.end > updated.start)) fail('expression.end 必须大于 start');
  addEffects(ctx, 'evaluator', 'render');
}

function removeClip(ctx: EditContext, cmd: Record<string, unknown>, list: 'motions' | 'expressions'): void {
  const { layer, index: layerIndex } = findLayer(ctx, cmd.layer ?? cmd.id);
  if (layer.kind !== 'model') fail(`图层 ${layer.id} 不是模型图层`);
  const clips = list === 'motions' ? layer.motions : layer.expressions;
  const index = reqNumber(cmd.index, `${list}.index`, { integer: true, min: 0 });
  if (!clips || index >= clips.length) fail(`图层 ${layer.id} 的 ${list} 下标越界：${index}`);
  removeProp(ctx, clips as unknown as unknown[], index, ['layers', layerIndex, list]);
  addEffects(ctx, 'evaluator', 'render');
}

// ---------- 音频 ----------

function addAudio(ctx: EditContext, cmd: Record<string, unknown>): void {
  const uri = reqString(cmd.uri, 'addAudio.uri');
  const assets = ensureArray(ctx, ctx.scene.assets as unknown as Record<string, unknown>, 'audio', ['assets']) as AudioAsset[];
  let asset = assets.find((item) => item.uri === uri);
  if (!asset) {
    const used = new Set(assets.map((item) => item.id));
    asset = { id: assetIdFor(uri, cmd.assetId, used), uri };
    insertArrayItem(ctx, assets as unknown as unknown[], assets.length, asset, ['assets', 'audio']);
  }
  const start = cmd.start === undefined ? 0 : frameNumber(cmd.start, 'addAudio.start');
  const clip: AudioClip = { audio: asset.id, start };
  if (cmd.offset !== undefined) clip.offset = reqNumber(cmd.offset, 'addAudio.offset', { min: 0 });
  if (cmd.duration !== undefined) clip.duration = reqNumber(cmd.duration, 'addAudio.duration', { min: 0 });
  if (cmd.gain !== undefined) clip.gain = reqNumber(cmd.gain, 'addAudio.gain', { min: 0 });
  const clips = ensureArray(ctx, ctx.scene as unknown as Record<string, unknown>, 'audio', []);
  insertArrayItem(ctx, clips, clips.length, clip, ['audio']);
  addEffects(ctx, 'audio', 'clock');
}

function updateAudio(ctx: EditContext, cmd: Record<string, unknown>): void {
  const clips = ctx.scene.audio;
  if (!clips) fail('场景没有 audio 片段');
  const index = reqNumber(cmd.index, 'updateAudio.index', { integer: true, min: 0 });
  if (index >= clips.length) fail(`audio 下标越界：${index}`);
  const patch = plainObject(cmd.patch, 'updateAudio.patch');
  const allowed = new Set(['audio', 'start', 'offset', 'duration', 'gain']);
  for (const key of Object.keys(patch)) if (!allowed.has(key)) fail(`updateAudio 不允许修改字段：${key}`);
  const record = clips[index] as unknown as Record<string, unknown>;
  const base: PathSeg[] = ['audio', index];
  if ('audio' in patch) {
    const id = reqString(patch.audio, 'updateAudio.audio');
    if (!(ctx.scene.assets.audio ?? []).some((asset) => asset.id === id)) fail(`音频资源不存在：${id}`);
    setProp(ctx, record, 'audio', id, base);
  }
  if ('start' in patch) setProp(ctx, record, 'start', frameNumber(patch.start, 'updateAudio.start'), base);
  if ('offset' in patch) {
    if (patch.offset === null) removeProp(ctx, record, 'offset', base);
    else setProp(ctx, record, 'offset', reqNumber(patch.offset, 'updateAudio.offset', { min: 0 }), base);
  }
  if ('duration' in patch) {
    if (patch.duration === null) removeProp(ctx, record, 'duration', base);
    else setProp(ctx, record, 'duration', reqNumber(patch.duration, 'updateAudio.duration', { min: 0 }), base);
  }
  if ('gain' in patch) {
    if (patch.gain === null) removeProp(ctx, record, 'gain', base);
    else setProp(ctx, record, 'gain', reqNumber(patch.gain, 'updateAudio.gain', { min: 0 }), base);
  }
  addEffects(ctx, 'audio', 'clock');
}

function removeAudio(ctx: EditContext, cmd: Record<string, unknown>): void {
  const clips = ctx.scene.audio;
  if (!clips) fail('场景没有 audio 片段');
  const index = reqNumber(cmd.index, 'removeAudio.index', { integer: true, min: 0 });
  if (index >= clips.length) fail(`audio 下标越界：${index}`);
  removeProp(ctx, clips as unknown as unknown[], index, ['audio']);
  addEffects(ctx, 'audio', 'clock');
}

// ---------- marker ----------

function addMarker(ctx: EditContext, cmd: Record<string, unknown>): void {
  const marker = clone(plainObject(cmd.marker, 'addMarker.marker')) as unknown as Marker;
  checkId(marker.id, 'marker.id');
  if ((ctx.scene.markers ?? []).some((item) => item.id === marker.id)) fail(`标记 ID 已存在：${marker.id}`);
  frameNumber(marker.start, 'marker.start');
  if (marker.end !== undefined) reqNumber(marker.end, 'marker.end');
  const markers = ensureArray(ctx, ctx.scene as unknown as Record<string, unknown>, 'markers', []);
  insertArrayItem(ctx, markers, markers.length, marker, ['markers']);
  addEffects(ctx, 'render');
}

function updateMarker(ctx: EditContext, cmd: Record<string, unknown>): void {
  const id = checkId(cmd.id, 'updateMarker.id');
  const markers = ctx.scene.markers;
  const index = markers?.findIndex((marker) => marker.id === id) ?? -1;
  if (!markers || index < 0) fail(`找不到标记：${id}`);
  const patch = plainObject(cmd.patch, 'updateMarker.patch');
  const allowed = new Set(['start', 'end', 'label', 'kind', 'semantics']);
  for (const key of Object.keys(patch)) if (!allowed.has(key)) fail(`updateMarker 不允许修改字段：${key}`);
  const record = markers[index]! as unknown as Record<string, unknown>;
  const base: PathSeg[] = ['markers', index];
  if ('start' in patch) setProp(ctx, record, 'start', frameNumber(patch.start, 'updateMarker.start'), base);
  if ('end' in patch) {
    if (patch.end === null) removeProp(ctx, record, 'end', base);
    else setProp(ctx, record, 'end', reqNumber(patch.end, 'updateMarker.end'), base);
  }
  if ('label' in patch) {
    if (patch.label === null) removeProp(ctx, record, 'label', base);
    else setProp(ctx, record, 'label', reqString(patch.label, 'updateMarker.label'), base);
  }
  if ('kind' in patch) {
    if (patch.kind === null) removeProp(ctx, record, 'kind', base);
    else setProp(ctx, record, 'kind', reqString(patch.kind, 'updateMarker.kind'), base);
  }
  if ('semantics' in patch) {
    if (patch.semantics === null) removeProp(ctx, record, 'semantics', base);
    else setProp(ctx, record, 'semantics', clone(plainObject(patch.semantics, 'updateMarker.semantics')), base);
  }
  addEffects(ctx, 'render');
}

function removeMarker(ctx: EditContext, cmd: Record<string, unknown>): void {
  const id = checkId(cmd.id, 'removeMarker.id');
  const markers = ctx.scene.markers;
  const index = markers?.findIndex((marker) => marker.id === id) ?? -1;
  if (!markers || index < 0) fail(`找不到标记：${id}`);
  removeProp(ctx, markers as unknown as unknown[], index, ['markers']);
  addEffects(ctx, 'render');
}

// ---------- camera ----------

function setCamera(ctx: EditContext, cmd: Record<string, unknown>): void {
  if (cmd.camera === null) {
    if (ctx.scene.camera !== undefined) {
      removeProp(ctx, ctx.scene as unknown as Record<string, unknown>, 'camera', []);
      addEffects(ctx, 'render');
    }
    return;
  }
  const camera = clone(plainObject(cmd.camera, 'setCamera.camera')) as unknown as CameraTransform;
  setProp(ctx, ctx.scene as unknown as Record<string, unknown>, 'camera', camera, []);
  addEffects(ctx, 'render');
}

function updateCamera(ctx: EditContext, cmd: Record<string, unknown>): void {
  if (!ctx.scene.camera) fail('场景没有 camera，请先 setCamera');
  const patch = plainObject(cmd.patch, 'updateCamera.patch');
  for (const key of Object.keys(patch)) if (!CAMERA_KEYS.has(key)) fail(`updateCamera 不允许修改字段：${key}`);
  const camera = ctx.scene.camera as unknown as Record<string, unknown>;
  for (const key of Object.keys(patch)) {
    const value = patch[key];
    if (value === null) {
      if (key !== 'rotation') fail(`camera.${key} 是必填属性，不能设为 null`);
      removeProp(ctx, camera, key, ['camera']);
    } else {
      setProp(ctx, camera, key, animNumber(value, `updateCamera.${key}`), ['camera']);
    }
  }
  addEffects(ctx, 'render');
}

// ---------- 关键帧 ----------

function applySetKeyframes(ctx: EditContext, cmd: Record<string, unknown>): void {
  const target = cmd.target;
  const defaultEase = cmd.ease === undefined || cmd.ease === null ? undefined : normalizeEase(cmd.ease, 'setKeyframes.ease');
  const keys = normalizeKeyframes(cmd.keys, defaultEase);
  if (keys.length === 0) {
    const slot = resolveTarget(ctx, target, { create: false });
    if (!slot) return;
    const t = plainObject(target, 'target');
    const isParam = t.kind === 'param' || (t.kind === undefined && t.param !== undefined);
    slot.reset();
    if (isParam) addEffects(ctx, 'render', 'workers', 'evaluator');
    else addEffects(ctx, 'render');
    return;
  }
  const slot = requireSlot(ctx, target, true);
  slot.set({ keys });
  addAnimTargetEffects(ctx, target);
}

function applyAddKeyframe(ctx: EditContext, cmd: Record<string, unknown>): void {
  const slot = requireSlot(ctx, cmd.target, true);
  const t = frameNumber(cmd.t ?? cmd.time, 'addKeyframe.t');
  const v = reqNumber(cmd.v ?? cmd.value, 'addKeyframe.v');
  const ease = cmd.ease === undefined || cmd.ease === null ? undefined : normalizeEase(cmd.ease, 'addKeyframe.ease');
  const current = slot.get();
  let keys: NumericKey[];
  if (current === undefined || typeof current === 'number') keys = [];
  else keys = editableKeys(current, slot.label);
  const index = findKeyAt(keys, t);
  if (index >= 0) {
    keys[index]!.v = v;
    if (ease !== undefined) keys[index]!.ease = ease;
  } else {
    keys.push(keyOf(t, v, ease));
    keys.sort((a, b) => a.t - b.t);
  }
  slot.set({ keys });
  addAnimTargetEffects(ctx, cmd.target);
}

function applyUpdateKeyframe(ctx: EditContext, cmd: Record<string, unknown>): void {
  const slot = requireSlot(ctx, cmd.target, false);
  const t = frameNumber(cmd.t ?? cmd.time, 'updateKeyframe.t');
  const current = slot.get();
  if (current === undefined) fail(`${slot.label} 没有关键帧`);
  const keys = editableKeys(current, slot.label);
  const index = findKeyAt(keys, t);
  if (index < 0) fail(`${slot.label} 在 t=${t} 没有关键帧`);
  const key = keys[index]!;
  if (cmd.v !== undefined) key.v = reqNumber(cmd.v, 'updateKeyframe.v');
  const newT = cmd.newT ?? cmd.newTime;
  if (newT !== undefined) {
    const nextT = frameNumber(newT, 'updateKeyframe.newT');
    if (nextT !== key.t && keys.some((item, i) => i !== index && Math.abs(item.t - nextT) <= EPS)) {
      fail(`${slot.label} 在 t=${nextT} 已有关键帧`);
    }
    key.t = nextT;
  }
  if (cmd.ease !== undefined && cmd.ease !== null) key.ease = normalizeEase(cmd.ease, 'updateKeyframe.ease');
  keys.sort((a, b) => a.t - b.t);
  slot.set({ keys });
  addAnimTargetEffects(ctx, cmd.target);
}

function applyRemoveKeyframe(ctx: EditContext, cmd: Record<string, unknown>): void {
  const slot = requireSlot(ctx, cmd.target, false);
  const t = frameNumber(cmd.t ?? cmd.time, 'removeKeyframe.t');
  const current = slot.get();
  if (current === undefined) fail(`${slot.label} 没有关键帧`);
  const sampled = sample(current, t, slot.fallback);
  const keys = editableKeys(current, slot.label);
  const index = findKeyAt(keys, t);
  if (index < 0) fail(`${slot.label} 在 t=${t} 没有关键帧`);
  keys.splice(index, 1);
  const flatten = cmd.flatten !== false;
  if (keys.length === 0) {
    if (flatten) slot.set(sampled);
    else slot.set({ keys: [keyOf(t, sampled, undefined)] });
  } else {
    slot.set({ keys });
  }
  addAnimTargetEffects(ctx, cmd.target);
}

function applySetEase(ctx: EditContext, cmd: Record<string, unknown>): void {
  const slot = requireSlot(ctx, cmd.target, false);
  const current = slot.get();
  if (current === undefined) fail(`${slot.label} 没有关键帧`);
  const keys = editableKeys(current, slot.label);
  const ease = normalizeEase(cmd.ease, 'setEase.ease');
  let times: number[] | null = null;
  if (cmd.times !== undefined) {
    if (!Array.isArray(cmd.times)) fail('setEase.times 必须是数组');
    times = cmd.times.map((time, i) => frameNumber(time, `setEase.times[${i}]`));
  }
  if (cmd.t !== undefined) {
    const time = frameNumber(cmd.t, 'setEase.t');
    times = [...(times ?? []), time];
  }
  if (times === null) {
    for (const key of keys) key.ease = Array.isArray(ease) ? ([...ease] as Ease) : ease;
  } else {
    for (const time of times) {
      const index = findKeyAt(keys, time);
      if (index < 0) fail(`${slot.label} 在 t=${time} 没有关键帧`);
      keys[index]!.ease = Array.isArray(ease) ? ([...ease] as Ease) : ease;
    }
  }
  slot.set({ keys });
  addAnimTargetEffects(ctx, cmd.target);
}

function generatePreset(
  preset: AnimationPreset,
  base: { x: number; y: number; scale: number; rotation: number; opacity: number },
  start: number,
  end: number,
  intensity: number,
  ease: Ease,
): Partial<Record<'x' | 'y' | 'scale' | 'rotation' | 'opacity', NumericKey[]>> {
  const at = (fraction: number) => start + (end - start) * fraction;
  const key = (fraction: number, v: number) => keyOf(at(fraction), v, ease);
  switch (preset) {
    case 'fadeIn':
      return { opacity: [key(0, 0), key(1, base.opacity)] };
    case 'fadeOut':
      return { opacity: [key(0, base.opacity), key(1, 0)] };
    case 'float':
      return { y: [key(0, base.y), key(0.5, base.y - intensity), key(1, base.y)] };
    case 'pulse':
      return { scale: [key(0, base.scale), key(0.5, base.scale * (1 + intensity)), key(1, base.scale)] };
    case 'spin':
      return { rotation: [key(0, base.rotation), key(1, base.rotation + intensity)] };
    case 'bounce':
      return {
        y: [
          key(0, base.y),
          key(0.2, base.y - intensity),
          key(0.4, base.y),
          key(0.55, base.y - intensity * 0.5),
          key(0.7, base.y),
          key(0.85, base.y - intensity * 0.25),
          key(1, base.y),
        ],
      };
    case 'blink':
      return {
        opacity: [
          key(0, base.opacity),
          key(0.42, base.opacity),
          key(0.48, 0),
          key(0.54, 0),
          key(0.6, base.opacity),
          key(1, base.opacity),
        ],
      };
    case 'slideIn':
      return { x: [key(0, base.x - intensity), key(1, base.x)] };
    case 'shake':
      return {
        x: [
          key(0, base.x),
          key(0.1, base.x + intensity),
          key(0.3, base.x - intensity),
          key(0.5, base.x + intensity),
          key(0.7, base.x - intensity),
          key(0.9, base.x + intensity * 0.5),
          key(1, base.x),
        ],
      };
    case 'zoomIn':
      return {
        scale: [key(0, base.scale * (1 - intensity)), key(1, base.scale)],
        opacity: [key(0, 0), key(1, base.opacity)],
      };
    default:
      fail(`未知动画预设：${String(preset)}`);
  }
}

function animateLayer(ctx: EditContext, cmd: Record<string, unknown>): void {
  const { layer } = findLayer(ctx, cmd.id);
  const preset = reqString(cmd.preset, 'animate.preset') as AnimationPreset;
  if (!Object.prototype.hasOwnProperty.call(DEFAULT_INTENSITY, preset)) fail(`未知动画预设：${preset}`);
  const start = cmd.start === undefined ? layer.in : frameNumber(cmd.start, 'animate.start');
  const end =
    cmd.end === undefined ? Math.min(layer.out, start + 60) : frameNumber(cmd.end, 'animate.end');
  if (!(end > start)) fail(`animate 的 end（${end}）必须大于 start（${start}）`);
  const intensity = cmd.intensity === undefined ? DEFAULT_INTENSITY[preset] : reqNumber(cmd.intensity, 'animate.intensity');
  const ease = cmd.ease === undefined || cmd.ease === null ? PRESET_EASE[preset] : normalizeEase(cmd.ease, 'animate.ease');
  const transform = layer.transform;
  const base = {
    x: sample(transform.x, start, 0),
    y: sample(transform.y, start, 0),
    scale: sample(transform.scale, start, 1),
    rotation: sample(transform.rotation, start, 0),
    opacity: sample(layer.opacity, start, 1),
  };
  const generated = generatePreset(preset, base, start, end, intensity, ease);
  const layerIndex = ctx.scene.layers.indexOf(layer);
  const transformRecord = transform as unknown as Record<string, unknown>;
  for (const prop of ['x', 'y', 'scale', 'rotation'] as const) {
    const keys = generated[prop];
    if (!keys) continue;
    const current = transform[prop] as AnimNumber | undefined;
    const merged = mergeAnimated(current, start, end, keys);
    setProp(ctx, transformRecord, prop, merged, ['layers', layerIndex, 'transform']);
  }
  if (generated.opacity) {
    const merged = mergeAnimated(layer.opacity, start, end, generated.opacity);
    setProp(ctx, layer as unknown as Record<string, unknown>, 'opacity', merged, ['layers', layerIndex]);
  }
  addEffects(ctx, 'render');
}

// ---------- 命令分发 ----------

const OP_ALIASES: Record<string, string> = {
  updateScene: 'setScene',
  addImage: 'addImageLayer',
  addDecoration: 'addImageLayer',
  addModel: 'addModelLayer',
  editLayer: 'updateLayer',
  updateDecoration: 'updateLayer',
  deleteLayer: 'removeLayer',
  removeDecoration: 'removeLayer',
  reorderLayer: 'moveLayer',
  cloneLayer: 'duplicateLayer',
  addMotionClip: 'addMotion',
  addExpressionClip: 'addExpression',
  updateMotionClip: 'updateMotion',
  updateExpressionClip: 'updateExpression',
  removeMotionClip: 'removeMotion',
  deleteMotion: 'removeMotion',
  removeExpressionClip: 'removeExpression',
  deleteExpression: 'removeExpression',
  addAudioClip: 'addAudio',
  updateAudioClip: 'updateAudio',
  removeAudioClip: 'removeAudio',
  deleteAudio: 'removeAudio',
  deleteMarker: 'removeMarker',
  setCurve: 'setKeyframes',
  setAnimation: 'setKeyframes',
  addKey: 'addKeyframe',
  updateKey: 'updateKeyframe',
  removeKey: 'removeKeyframe',
  deleteKeyframe: 'removeKeyframe',
  ease: 'setEase',
  autoAnimate: 'animate',
  nop: 'noop',
  'no-op': 'noop',
};

function canonicalOp(cmd: Record<string, unknown>): string {
  const raw = cmd.op;
  if (typeof raw !== 'string' || raw.length === 0) fail('命令缺少 op');
  if (raw === 'addLayer') {
    const layer = cmd.layer;
    return isRecord(layer) && layer.kind === 'image' ? 'addImageLayer' : 'addModelLayer';
  }
  return Object.prototype.hasOwnProperty.call(OP_ALIASES, raw) ? OP_ALIASES[raw]! : raw;
}

function applyOne(ctx: EditContext, cmd: unknown): void {
  const record = plainObject(cmd, '命令');
  const op = canonicalOp(record);
  switch (op) {
    case 'setScene':
      return applySetScene(ctx, record);
    case 'addImageLayer':
      return addImageLayer(ctx, record);
    case 'addModelLayer':
      return addModelLayer(ctx, record);
    case 'updateLayer':
      return updateLayer(ctx, record);
    case 'removeLayer':
      return removeLayer(ctx, record);
    case 'moveLayer':
      return moveLayer(ctx, record);
    case 'duplicateLayer':
      return duplicateLayer(ctx, record);
    case 'addMotion':
      return addMotion(ctx, record);
    case 'addExpression':
      return addExpression(ctx, record);
    case 'updateMotion':
      return updateMotion(ctx, record);
    case 'updateExpression':
      return updateExpression(ctx, record);
    case 'removeMotion':
      return removeClip(ctx, record, 'motions');
    case 'removeExpression':
      return removeClip(ctx, record, 'expressions');
    case 'addAudio':
      return addAudio(ctx, record);
    case 'updateAudio':
      return updateAudio(ctx, record);
    case 'removeAudio':
      return removeAudio(ctx, record);
    case 'addMarker':
      return addMarker(ctx, record);
    case 'updateMarker':
      return updateMarker(ctx, record);
    case 'removeMarker':
      return removeMarker(ctx, record);
    case 'setCamera':
      return setCamera(ctx, record);
    case 'updateCamera':
      return updateCamera(ctx, record);
    case 'setKeyframes':
      return applySetKeyframes(ctx, record);
    case 'addKeyframe':
      return applyAddKeyframe(ctx, record);
    case 'updateKeyframe':
      return applyUpdateKeyframe(ctx, record);
    case 'removeKeyframe':
      return applyRemoveKeyframe(ctx, record);
    case 'setEase':
      return applySetEase(ctx, record);
    case 'animate':
      return animateLayer(ctx, record);
    case 'preview':
      ctx.previewFrame = frameNumber(record.frame, 'preview.frame');
      return;
    case 'noop':
      return;
    default:
      fail(`未知编辑命令：${op}`);
  }
}

/**
 * 应用一条或多条编辑命令（原地修改 scene）。
 *
 * validate 缺省 'each'：每条真正改动了场景的命令之后跑 parseScene；
 * 'final' 只在整批结束时校验一次；'none' 不校验。
 * 任何异常都会按相反顺序撤销本批 changes，然后重新抛出。
 */
export function applyEditCommands(
  scene: Scene,
  commands: EditCommand | EditCommand[],
  options: EditOptions = {},
): EditResult {
  const list = Array.isArray(commands) ? commands : [commands];
  const validate = options.validate ?? 'each';
  if (validate !== 'each' && validate !== 'final' && validate !== 'none') fail(`未知 validate 模式：${String(validate)}`);
  if (options.frame !== undefined) frameNumber(options.frame, 'options.frame');

  const ctx: EditContext = { scene, changes: [], effects: new Set() };
  let preview: FramePreview | undefined;

  try {
    for (const command of list) {
      const before = ctx.changes.length;
      applyOne(ctx, command);
      if (validate === 'each' && ctx.changes.length > before) parseScene(scene);
    }
    if (validate === 'final' && ctx.changes.length > 0) parseScene(scene);
    // preview 命令的帧号在最后统一求值，保证看到的是整批命令后的场景。
    if (options.frame !== undefined) preview = previewAt(scene, options.frame);
    else if (ctx.previewFrame !== undefined) preview = previewAt(scene, ctx.previewFrame);
  } catch (error) {
    try {
      undoEditChanges(scene, ctx.changes);
    } catch (rollbackError) {
      throw new Error(`${String((error as Error)?.message ?? error)}\n回滚失败：${String((rollbackError as Error)?.message ?? rollbackError)}`);
    }
    throw error;
  }

  return {
    ok: true,
    changes: ctx.changes,
    effects: effectList(ctx.effects),
    ...(preview !== undefined ? { preview } : {}),
  };
}
