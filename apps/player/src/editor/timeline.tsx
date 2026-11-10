/*
 * 多轨时间线（编辑器底部面板）。
 *
 * 只依赖 contracts.ts 的 EditorStoreApi；所有修改都通过 store.apply 发送到场景命令层。
 * 布局：一个滚动容器；左侧 240px 轨道头用 position:sticky 固定，顶部标尺同样 sticky。
 * 轨道从上到下：镜头/标记 → 相机 → 音频 → 图层（可展开属性子轨）→ 固定参数。
 */
import {
  EASE_PRESETS,
  isCurve,
  sample,
  type Animatable,
  type Ease,
  type EasePreset,
  type EditCommand,
  type EditTarget,
  type Layer,
  type Scene,
} from '@charaforge/scene';
import {
  memo,
  useEffect,
  useReducer,
  useRef,
  useState,
  type CSSProperties,
  type JSX,
  type PointerEvent as ReactPointerEvent,
} from 'react';
import type { Catalog } from '../motionCatalog.ts';
import { timecode } from '../util.ts';
import type { EditorState, EditorStoreApi, KeySelection } from './contracts.ts';

const HEADER_WIDTH = 240;
const RULER_HEIGHT = 28;
const MIN_ZOOM = 0.25;
const MAX_ZOOM = 20;
const SUMMARY_TICK_LIMIT = 600;
const BAKED_KEY_LIMIT = 1500;

const TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 6000, 12000, 30000, 60000];
const EASE_ORDER: EasePreset[] = ['linear', 'smooth', 'easeIn', 'easeOut', 'easeInOut', 'step'];
const EASE_LABELS: Record<EasePreset, string> = {
  linear: '线性',
  smooth: '平滑',
  easeIn: '缓入',
  easeOut: '缓出',
  easeInOut: '缓入缓出',
  step: '阶梯',
  hold: '保持',
};

const FILTER_PROPS: Record<string, readonly string[]> = {
  blur: ['sigma'],
  motionBlur: ['angle', 'length'],
  resample: ['factor'],
  color: ['brightness', 'contrast', 'saturation'],
};
const FILTER_DEFAULTS: Record<string, Record<string, number>> = {
  blur: { sigma: 0 },
  motionBlur: { angle: 0, length: 0 },
  resample: { factor: 1 },
  color: { brightness: 0, contrast: 1, saturation: 1 },
};

type AnimNumber = Animatable<number>;

interface CurveKey {
  t: number;
  v: number;
  ease?: Ease;
}

interface PropDef {
  path: string;
  label: string;
  curve: AnimNumber | undefined;
  fallback: number;
}

interface TargetDef extends PropDef {
  target: EditTarget;
}

// ---------- 通用小工具 ----------

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function clampZoom(value: number): number {
  return clamp(Number.isFinite(value) ? value : 2, MIN_ZOOM, MAX_ZOOM);
}

function clampFrame(value: number, duration: number): number {
  return clamp(Number.isFinite(value) ? value : 0, 0, Math.max(0, duration - 1));
}

function roundSeconds(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return '—';
  const abs = Math.abs(value);
  let text: string;
  if (abs >= 1000) text = value.toFixed(1);
  else if (abs >= 10) text = value.toFixed(2);
  else if (abs >= 1) text = value.toFixed(2);
  else text = value.toFixed(3);
  return text.replace(/\.?0+$/, '') || '0';
}

function sampleEven<T>(items: readonly T[], limit: number): T[] {
  if (limit <= 0 || items.length === 0) return [];
  if (items.length <= limit) return [...items];
  if (limit === 1) return [items[0]!];
  const out: T[] = [];
  for (let i = 0; i < limit; i++) {
    const at = Math.round((i * (items.length - 1)) / (limit - 1));
    out.push(items[at]!);
  }
  return out;
}

function asAnim(value: unknown): AnimNumber | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (value && typeof value === 'object' && ('keys' in value || 'baked' in value)) return value as AnimNumber;
  return undefined;
}

/** 曲线 → 可渲染关键帧（烘焙曲线按每帧展开，并限制数量）。 */
function curveKeys(value: AnimNumber | undefined, limit = Infinity): CurveKey[] {
  if (value === undefined || typeof value === 'number') return [];
  if ('baked' in value) {
    const keys = value.baked.values.map((v, i) => ({ t: value.baked.start + i, v }));
    return keys.length > limit ? sampleEven(keys, limit) : keys;
  }
  return value.keys.map((key) => {
    const item: CurveKey = { t: key.t, v: key.v };
    if (key.ease !== undefined) item.ease = key.ease;
    return item;
  });
}

function findKeyAt(value: AnimNumber | undefined, t: number): CurveKey | undefined {
  if (value === undefined || typeof value === 'number') return undefined;
  return curveKeys(value, BAKED_KEY_LIMIT).find((key) => Math.abs(key.t - t) <= 1e-6);
}

function easeToPreset(ease: Ease | undefined): EasePreset | '' {
  if (ease === undefined || ease === 'linear') return 'linear';
  if (ease === 'step') return 'step';
  if (ease === 'smooth') return 'smooth';
  const near = (a: readonly number[], b: readonly number[]) => a.every((v, i) => Math.abs(v - b[i]!) <= 1e-6);
  if (near(ease, EASE_PRESETS.easeIn as number[])) return 'easeIn';
  if (near(ease, EASE_PRESETS.easeOut as number[])) return 'easeOut';
  if (near(ease, EASE_PRESETS.easeInOut as number[])) return 'easeInOut';
  return '';
}

function targetKey(target: EditTarget): string {
  if (target.kind === 'layer') return `layer:${target.id}:${target.path}`;
  if (target.kind === 'camera') return `camera:${target.path}`;
  return `param:${target.id}:${target.param}`;
}

function selectTargetOwner(store: EditorStoreApi, target: EditTarget): void {
  if (target.kind === 'camera') store.select({ kind: 'camera' });
  else store.select({ kind: 'layer', id: target.id });
}

function notifyError(store: EditorStoreApi, prefix: string, error: unknown): void {
  const message = String((error as Error)?.message ?? error);
  store.notify(`${prefix}：${message}`, 'error');
}

// ---------- 属性表 ----------

function layerPropDefs(layer: Layer): PropDef[] {
  const transform = layer.transform as unknown as Record<string, unknown>;
  const defs: PropDef[] = [
    { path: 'transform.x', label: 'X', curve: asAnim(transform.x), fallback: 0 },
    { path: 'transform.y', label: 'Y', curve: asAnim(transform.y), fallback: 0 },
    { path: 'transform.scale', label: '缩放', curve: asAnim(transform.scale), fallback: 1 },
    { path: 'transform.rotation', label: '旋转', curve: asAnim(transform.rotation), fallback: 0 },
  ];
  // 可选轴向缩放只在真的有动画时出现，避免给普通图层增加空行。
  const scaleX = asAnim(transform.scaleX);
  const scaleY = asAnim(transform.scaleY);
  if (isCurve(scaleX)) defs.push({ path: 'transform.scaleX', label: '横向缩放', curve: scaleX, fallback: 1 });
  if (isCurve(scaleY)) defs.push({ path: 'transform.scaleY', label: '纵向缩放', curve: scaleY, fallback: 1 });
  defs.push({ path: 'opacity', label: '不透明度', curve: asAnim(layer.opacity), fallback: 1 });

  if (layer.clip) {
    const clip = layer.clip as unknown as Record<string, unknown>;
    defs.push(
      { path: 'clip.x', label: '裁切 X', curve: asAnim(clip.x), fallback: 0 },
      { path: 'clip.y', label: '裁切 Y', curve: asAnim(clip.y), fallback: 0 },
      { path: 'clip.width', label: '裁切宽', curve: asAnim(clip.width), fallback: 0 },
      { path: 'clip.height', label: '裁切高', curve: asAnim(clip.height), fallback: 0 },
      { path: 'clip.radius', label: '圆角', curve: asAnim(clip.radius), fallback: 0 },
      { path: 'clip.feather', label: '羽化', curve: asAnim(clip.feather), fallback: 0 },
    );
  }

  (layer.filters ?? []).forEach((filter, filterIndex) => {
    for (const prop of FILTER_PROPS[filter.type] ?? []) {
      const record = filter as unknown as Record<string, unknown>;
      defs.push({
        path: `filters.${filterIndex}.${prop}`,
        label: `滤镜${filterIndex + 1} ${prop}`,
        curve: asAnim(record[prop]),
        fallback: FILTER_DEFAULTS[filter.type]?.[prop] ?? 0,
      });
    }
  });

  return defs;
}

function layerTargets(layer: Layer): TargetDef[] {
  return layerPropDefs(layer).map((def) => ({
    ...def,
    target: { kind: 'layer', id: layer.id, path: def.path },
  }));
}

function cameraTargets(camera: NonNullable<Scene['camera']>): TargetDef[] {
  const record = camera as unknown as Record<string, unknown>;
  return [
    { path: 'x', label: 'X', curve: asAnim(record.x), fallback: 0 },
    { path: 'y', label: 'Y', curve: asAnim(record.y), fallback: 0 },
    { path: 'zoom', label: '缩放', curve: asAnim(record.zoom), fallback: 1 },
    { path: 'rotation', label: '旋转', curve: asAnim(record.rotation), fallback: 0 },
  ].map((def) => ({ ...def, target: { kind: 'camera', path: def.path } }));
}

function collectTicks(defs: readonly PropDef[]): CurveKey[] {
  const byTime = new Map<string, number>();
  for (const def of defs) {
    for (const key of curveKeys(def.curve, BAKED_KEY_LIMIT)) {
      const slot = String(Math.round(key.t * 1e6));
      if (!byTime.has(slot)) byTime.set(slot, key.t);
    }
  }
  const sorted = [...byTime.values()].sort((a, b) => a - b).map((t) => ({ t, v: 0 }));
  return sampleEven(sorted, SUMMARY_TICK_LIMIT);
}

function collectCameraTicks(camera: NonNullable<Scene['camera']>): { t: number; v: number; target: EditTarget }[] {
  const byTime = new Map<string, { t: number; v: number; target: EditTarget }>();
  for (const def of cameraTargets(camera)) {
    for (const key of curveKeys(def.curve, BAKED_KEY_LIMIT)) {
      const slot = String(Math.round(key.t * 1e6));
      if (!byTime.has(slot)) byTime.set(slot, { t: key.t, v: key.v, target: def.target });
    }
  }
  return sampleEven([...byTime.values()].sort((a, b) => a.t - b.t), SUMMARY_TICK_LIMIT);
}

/** 图层摘要签名的用途：播放时 React.memo 跳过未变化的行；编辑后签名变化会重新渲染。 */
function layerSignature(layer: Layer): string {
  const parts: string[] = [`${layer.in}:${layer.out}`];
  for (const def of layerPropDefs(layer)) {
    const curve = def.curve;
    if (curve === undefined) {
      parts.push(`${def.path}:u`);
    } else if (typeof curve === 'number') {
      parts.push(`${def.path}:${curve}`);
    } else if ('baked' in curve) {
      parts.push(`${def.path}:b${curve.baked.start}:${curve.baked.values.length}`);
    } else {
      parts.push(
        `${def.path}:` +
          curve.keys
            .map((key) => `${key.t},${key.v},${key.ease === undefined ? '' : Array.isArray(key.ease) ? key.ease.join(',') : key.ease}`)
            .join(';'),
      );
    }
  }
  if (layer.kind === 'model') {
    parts.push(`mn:${layer.motions?.length ?? 0}`);
    for (const clip of layer.motions ?? []) {
      parts.push(`m:${clip.motion}:${clip.start}:${clip.end ?? '-'}:${clip.offset ?? 0}:${clip.speed ?? 1}:${clip.loop ? 1 : 0}`);
    }
    parts.push(`en:${layer.expressions?.length ?? 0}`);
    for (const clip of layer.expressions ?? []) {
      parts.push(`e:${clip.expression}:${clip.start}:${clip.end ?? '-'}`);
    }
  }
  return parts.join('|');
}

function findCurveForTarget(scene: Scene, target: EditTarget): AnimNumber | undefined {
  if (target.kind === 'camera') {
    if (!scene.camera) return undefined;
    const record = scene.camera as unknown as Record<string, unknown>;
    return asAnim(record[target.path]);
  }
  const layer = scene.layers.find((item) => item.id === target.id);
  if (!layer) return undefined;
  if (target.kind === 'param') {
    if (layer.kind !== 'model') return undefined;
    return layer.params?.find((track) => track.id === target.param)?.curve;
  }
  return layerPropDefs(layer).find((def) => def.path === target.path)?.curve;
}

interface EaseInfo {
  target: EditTarget;
  t: number;
  ease: Ease | undefined;
  keyId: string;
}

/** 工具栏缓动选择显示的目标：选中键 > 选中图层/相机在播放头正下方的键。 */
function findEaseInfo(scene: Scene, state: EditorState): EaseInfo | null {
  const first = state.selectedKeys[0];
  if (first) {
    const key = findKeyAt(findCurveForTarget(scene, first.target), first.t);
    if (key) return { target: first.target, t: first.t, ease: key.ease, keyId: first.key };
  }

  const scan = (defs: readonly TargetDef[]): EaseInfo | null => {
    for (const def of defs) {
      const key = findKeyAt(def.curve, state.frame);
      if (key) return { target: def.target, t: key.t, ease: key.ease, keyId: targetKey(def.target) };
    }
    return null;
  };

  const selection = state.selection;
  if (selection.kind === 'layer') {
    const layerId = selection.id;
    const layer = scene.layers.find((item) => item.id === layerId);
    if (layer) {
      const found = scan(layerTargets(layer));
      if (found) return found;
      if (state.showParams && layer.kind === 'model') {
        for (const pin of state.pinned) {
          const splitAt = pin.indexOf('::');
          if (splitAt < 0) continue;
          const pinLayerId = pin.slice(0, splitAt);
          const param = pin.slice(splitAt + 2);
          if (pinLayerId !== layer.id || !param) continue;
          const curve = layer.params?.find((track) => track.id === param)?.curve;
          const key = findKeyAt(curve, state.frame);
          if (key) {
            const target: EditTarget = { kind: 'param', id: layer.id, param };
            return { target, t: key.t, ease: key.ease, keyId: targetKey(target) };
          }
        }
      }
    }
  }
  if (selection.kind === 'camera' && scene.camera) {
    return scan(cameraTargets(scene.camera));
  }
  return null;
}

// ---------- 拖拽 ----------

interface KeyDragOptions {
  e: ReactPointerEvent<HTMLElement>;
  store: EditorStoreApi;
  target: EditTarget;
  keyId: string;
  t: number;
  v: number;
  zoom: number;
  snap: boolean;
  duration: number;
}

function beginKeyDrag(options: KeyDragOptions): void {
  const { e, store, target, keyId, t: initialT, v, zoom, snap, duration } = options;
  if (e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();

  const startX = e.clientX;
  const label = `drag-key:${keyId}:${initialT}`;
  const previousUserSelect = document.body.style.userSelect;
  document.body.style.userSelect = 'none';
  let currentT = initialT;
  let moved = false;
  let disposed = false;

  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onUp);
    document.body.style.userSelect = previousUserSelect;
  };

  const applyMove = (from: number, to: number) => {
    const command: EditCommand = { op: 'updateKeyframe', target, t: from, newT: to, v };
    try {
      void store.apply(command, { history: 'coalesce', label, validate: 'none' }).catch(() => {});
    } catch {
      // 冲突（例如同一时刻已有关键帧）时保持本地状态不动，下一次移动会重试。
    }
  };

  const onMove = (ev: PointerEvent) => {
    if ((ev.buttons & 1) === 0) {
      onUp(ev);
      return;
    }
    ev.preventDefault();
    const dx = ev.clientX - startX;
    if (!moved && Math.abs(dx) < 3) return;
    moved = true;
    const raw = initialT + dx / zoom;
    const stepped = snap && !ev.altKey ? Math.round(raw) : Math.round(raw * 1000) / 1000;
    const nextT = clampFrame(stepped, duration);
    if (Math.abs(nextT - currentT) < 1e-9) return;

    // 不允许把关键帧拖到同一曲线已有的时刻；否则命令会失败，后续拖动也会失去基准。
    const scene = store.getState().scene;
    if (scene && findKeyAt(findCurveForTarget(scene, target), nextT)) return;

    const from = currentT;
    currentT = nextT;
    applyMove(from, nextT);
  };

  const onUp = (ev: PointerEvent) => {
    if (disposed) return;
    cleanup();

    if (!moved) {
      const state = store.getState();
      const entry: KeySelection = { key: keyId, target, t: initialT };
      const same = (key: KeySelection) => key.key === keyId && Math.abs(key.t - initialT) <= 1e-6;
      const exists = state.selectedKeys.some(same);
      if (ev.shiftKey) {
        store.setSelectedKeys(exists ? state.selectedKeys.filter((key) => !same(key)) : [...state.selectedKeys, entry]);
      } else if (!(state.selectedKeys.length === 1 && same(state.selectedKeys[0]!))) {
        store.setSelectedKeys([entry]);
      }
      return;
    }

    if (Math.abs(currentT - initialT) > 1e-9) {
      // 最后一条命令走完整校验；值不变，但 replace 仍会记录 change，从而让场景跑一次 parseScene。
      const command: EditCommand = { op: 'updateKeyframe', target, t: currentT, newT: currentT, v };
      try {
        void store.apply(command, { history: 'coalesce', label, validate: 'each' }).catch(() => {});
      } catch {
        // 忽略：下一次编辑会重新校验。
      }
    }

    const state = store.getState();
    if (state.selectedKeys.some((key) => key.key === keyId && Math.abs(key.t - initialT) <= 1e-6)) {
      store.setSelectedKeys(
        state.selectedKeys.map((key) =>
          key.key === keyId && Math.abs(key.t - initialT) <= 1e-6 ? { ...key, t: currentT } : key,
        ),
      );
    }
  };

  window.addEventListener('pointermove', onMove, { passive: false });
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onUp);
}

type ClipKind = 'motion' | 'expression' | 'audio';
type ClipDragMode = 'move' | 'left' | 'right';

interface ClipDragOptions {
  e: ReactPointerEvent<HTMLElement>;
  store: EditorStoreApi;
  kind: ClipKind;
  layerId?: string;
  index: number;
  mode: ClipDragMode;
  initialStart: number;
  initialEnd: number;
  initialOffset: number;
  initialDurationSec: number;
  hasExplicitEnd: boolean;
  zoom: number;
  snap: boolean;
  fps: number;
}

function beginClipDrag(options: ClipDragOptions): void {
  const { e, store, kind, layerId, index, mode, initialStart, initialEnd, initialOffset, initialDurationSec, hasExplicitEnd, zoom, snap, fps } =
    options;
  if (e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();

  const startX = e.clientX;
  const label = `drag-${kind}:${layerId ?? 'scene'}:${index}:${mode}:${initialStart}`;
  const previousUserSelect = document.body.style.userSelect;
  document.body.style.userSelect = 'none';
  const span = Math.max(1, initialEnd - initialStart);
  const minDuration = 1 / Math.max(1, fps);
  let moved = false;
  let disposed = false;
  let lastStart = initialStart;
  let lastOffset = initialOffset;
  let lastEnd = initialEnd;
  let lastDurationSec = initialDurationSec;

  const makeCommand = (): EditCommand => {
    if (kind === 'audio') {
      if (mode === 'move') return { op: 'updateAudio', index, patch: { start: lastStart } };
      if (mode === 'left') {
        return {
          op: 'updateAudio',
          index,
          patch: { start: lastStart, offset: roundSeconds(lastOffset), duration: roundSeconds(lastDurationSec) },
        };
      }
      return { op: 'updateAudio', index, patch: { duration: roundSeconds(lastDurationSec) } };
    }
    if (kind === 'motion') {
      if (mode === 'move') {
        return hasExplicitEnd
          ? { op: 'updateMotion', layer: layerId!, index, patch: { start: lastStart, end: lastStart + span } }
          : { op: 'updateMotion', layer: layerId!, index, patch: { start: lastStart } };
      }
      if (mode === 'left') return { op: 'updateMotion', layer: layerId!, index, patch: { start: lastStart, end: lastEnd } };
      return { op: 'updateMotion', layer: layerId!, index, patch: { end: lastEnd } };
    }
    if (mode === 'move') {
      return hasExplicitEnd
        ? { op: 'updateExpression', layer: layerId!, index, patch: { start: lastStart, end: lastStart + span } }
        : { op: 'updateExpression', layer: layerId!, index, patch: { start: lastStart } };
    }
    if (mode === 'left') return { op: 'updateExpression', layer: layerId!, index, patch: { start: lastStart, end: lastEnd } };
    return { op: 'updateExpression', layer: layerId!, index, patch: { end: lastEnd } };
  };

  const apply = (validate: 'none' | 'each') => {
    try {
      void store.apply(makeCommand(), { history: 'coalesce', label, validate }).catch(() => {});
    } catch {
      // 冲突时保留上一次成功位置。
    }
  };

  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onUp);
    document.body.style.userSelect = previousUserSelect;
  };

  const onMove = (ev: PointerEvent) => {
    if ((ev.buttons & 1) === 0) {
      onUp();
      return;
    }
    ev.preventDefault();
    const dx = ev.clientX - startX;
    if (!moved && Math.abs(dx) < 3) return;
    moved = true;
    const raw = dx / zoom;
    const delta = snap && !ev.altKey ? Math.round(raw) : Math.round(raw * 1000) / 1000;

    if (mode === 'move') {
      const nextStart = Math.max(0, initialStart + delta);
      if (Math.abs(nextStart - lastStart) < 1e-9) return;
      lastStart = nextStart;
      apply('none');
      return;
    }

    if (mode === 'right') {
      if (kind === 'audio') {
        const nextDuration = Math.max(minDuration, initialDurationSec + delta / fps);
        if (Math.abs(nextDuration - lastDurationSec) < 1e-9) return;
        lastDurationSec = roundSeconds(nextDuration);
      } else {
        const nextEnd = Math.max(initialStart + 1, initialEnd + delta);
        if (Math.abs(nextEnd - lastEnd) < 1e-9) return;
        lastEnd = nextEnd;
      }
      apply('none');
      return;
    }

    // mode === 'left'
    if (kind === 'audio') {
      const minStart = Math.max(0, initialStart - initialOffset * fps);
      const nextStart = clamp(initialStart + delta, minStart, initialEnd - 1);
      if (Math.abs(nextStart - lastStart) < 1e-9) return;
      const deltaSeconds = (nextStart - initialStart) / fps;
      lastStart = nextStart;
      lastOffset = Math.max(0, initialOffset + deltaSeconds);
      lastDurationSec = Math.max(minDuration, initialDurationSec - deltaSeconds);
    } else {
      const nextStart = clamp(initialStart + delta, 0, initialEnd - 1);
      if (Math.abs(nextStart - lastStart) < 1e-9) return;
      lastStart = nextStart;
      lastEnd = initialEnd;
    }
    apply('none');
  };

  const onUp = () => {
    if (disposed) return;
    cleanup();
    if (!moved) return;
    // 最后一条命令完整校验一次。
    apply('each');
  };

  window.addEventListener('pointermove', onMove, { passive: false });
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onUp);
}

interface LayerRangeDragOptions {
  e: ReactPointerEvent<HTMLElement>;
  store: EditorStoreApi;
  layer: Layer;
  edge: 'in' | 'out';
  zoom: number;
  snap: boolean;
}

function beginLayerRangeDrag(options: LayerRangeDragOptions): void {
  const { e, store, layer, edge, zoom, snap } = options;
  if (e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();
  store.select({ kind: 'layer', id: layer.id });

  const startX = e.clientX;
  const initialIn = layer.in;
  const initialOut = layer.out;
  const label = `drag-layer-range:${layer.id}:${edge}:${edge === 'in' ? initialIn : initialOut}`;
  const previousUserSelect = document.body.style.userSelect;
  document.body.style.userSelect = 'none';
  let moved = false;
  let disposed = false;
  let lastIn = initialIn;
  let lastOut = initialOut;

  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onUp);
    document.body.style.userSelect = previousUserSelect;
  };

  const apply = (validate: 'none' | 'each') => {
    const command: EditCommand =
      edge === 'in'
        ? { op: 'updateLayer', id: layer.id, patch: { in: lastIn } }
        : { op: 'updateLayer', id: layer.id, patch: { out: lastOut } };
    try {
      void store.apply(command, { history: 'coalesce', label, validate }).catch(() => {});
    } catch {
      // 忽略。
    }
  };

  const onMove = (ev: PointerEvent) => {
    if ((ev.buttons & 1) === 0) {
      onUp();
      return;
    }
    ev.preventDefault();
    const dx = ev.clientX - startX;
    if (!moved && Math.abs(dx) < 3) return;
    moved = true;
    const raw = dx / zoom;
    const delta = snap && !ev.altKey ? Math.round(raw) : Math.round(raw * 1000) / 1000;
    if (edge === 'in') {
      const next = clamp(initialIn + delta, 0, initialOut - 1);
      if (Math.abs(next - lastIn) < 1e-9) return;
      lastIn = next;
    } else {
      const next = Math.max(initialIn + 1, initialOut + delta);
      if (Math.abs(next - lastOut) < 1e-9) return;
      lastOut = next;
    }
    apply('none');
  };

  const onUp = () => {
    if (disposed) return;
    cleanup();
    if (!moved) return;
    apply('each');
  };

  window.addEventListener('pointermove', onMove, { passive: false });
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onUp);
}

// ---------- 音频 / 动作片段辅助 ----------

function resolveMotionEnd(
  clip: { motion: string; start: number; end?: number; offset?: number; speed?: number },
  assets: Scene['assets'],
  catalog: Catalog | null,
  fps: number,
): number {
  if (clip.end !== undefined) return clip.end;
  const asset = assets.motions?.find((item) => item.id === clip.motion);
  const meta = catalog?.motions.find(
    (item) => item.uri === asset?.uri || (asset?.uri !== undefined && item.file !== undefined && asset.uri.endsWith(`/${item.file}`)),
  );
  if (meta) {
    const offset = clip.offset ?? 0;
    const speed = clip.speed ?? 1;
    const seconds = Math.max(0, (meta.meta.duration - offset) / speed);
    return clip.start + Math.max(1, Math.round(seconds * fps));
  }
  return clip.start + 120;
}

function resolveExpressionEnd(clip: { expression: string; start: number; end?: number }): number {
  return clip.end ?? clip.start + 120;
}

function findMotionMeta(
  clip: { motion: string },
  assets: Scene['assets'],
  catalog: Catalog | null,
): Catalog['motions'][number] | undefined {
  const asset = assets.motions?.find((item) => item.id === clip.motion);
  return catalog?.motions.find(
    (item) => item.uri === asset?.uri || (asset?.uri !== undefined && item.file !== undefined && asset.uri.endsWith(`/${item.file}`)),
  );
}

// ---------- 共用行组件 ----------

interface PropertyTrackRowProps {
  store: EditorStoreApi;
  label: string;
  sublabel?: string;
  target: EditTarget;
  keyId: string;
  curve: AnimNumber | undefined;
  fallback: number;
  frame: number;
  zoom: number;
  duration: number;
  snap: boolean;
  selectedKeys: KeySelection[];
  indent?: number;
  onUnpin?: () => void;
}

function PropertyTrackRow(props: PropertyTrackRowProps): JSX.Element {
  const { store, label, sublabel, target, keyId, curve, fallback, frame, zoom, duration, snap, selectedKeys } = props;
  const keys = curveKeys(curve, BAKED_KEY_LIMIT);
  const value = sample(curve, frame, fallback);

  let prevT: number | undefined;
  let nextT: number | undefined;
  for (const key of keys) {
    if (key.t < frame - 1e-6) prevT = key.t;
    else if (key.t > frame + 1e-6) {
      nextT = key.t;
      break;
    }
  }

  const addKey = () => {
    const command: EditCommand = { op: 'addKeyframe', target, t: frame, v: value };
    try {
      void store.apply(command, { history: true, label: `打关键帧：${label}` }).catch((error) => notifyError(store, '打关键帧失败', error));
    } catch (error) {
      notifyError(store, '打关键帧失败', error);
    }
  };

  const testId = `track-${keyId.replace(/[^A-Za-z0-9_-]+/g, '-')}`;

  return (
    <div className="tl-row tl-row-prop" data-testid={testId}>
      <div
        className="tl-head tl-prop-head"
        style={{ paddingLeft: props.indent ?? 30 }}
        title={`${label}${sublabel ? ` · ${sublabel}` : ''} · 当前值 ${formatNumber(value)}`}
        onClick={() => selectTargetOwner(store, target)}
      >
        <span className="tl-prop-label">{label}</span>
        {sublabel && <span className="tl-prop-sub">{sublabel}</span>}
        <span className="tl-prop-value">{formatNumber(value)}</span>
        <span className="tl-prop-actions">
          <button
            type="button"
            className="tl-icon-btn"
            disabled={prevT === undefined}
            title="上一个关键帧"
            onClick={(event) => {
              event.stopPropagation();
              if (prevT !== undefined) store.seek(prevT);
            }}
          >
            ‹
          </button>
          <button
            type="button"
            className="tl-icon-btn"
            disabled={nextT === undefined}
            title="下一个关键帧"
            onClick={(event) => {
              event.stopPropagation();
              if (nextT !== undefined) store.seek(nextT);
            }}
          >
            ›
          </button>
          <button
            type="button"
            className="tl-icon-btn add"
            title={`在 ${formatNumber(frame)} 帧打关键帧（当前值 ${formatNumber(value)}）`}
            onClick={(event) => {
              event.stopPropagation();
              addKey();
            }}
          >
            ◆
          </button>
          {props.onUnpin && (
            <button
              type="button"
              className="tl-icon-btn unpin"
              title="取消固定参数"
              onClick={(event) => {
                event.stopPropagation();
                props.onUnpin?.();
              }}
            >
              ×
            </button>
          )}
        </span>
      </div>
      <div className="tl-lane tl-prop-lane">
        {keys.map((key, index) => {
          const next = keys[index + 1];
          if (!next) return null;
          return (
            <span
              key={`seg-${key.t}-${next.t}`}
              className={`tl-seg${key.ease === 'step' ? ' step' : ''}`}
              style={{ left: key.t * zoom, width: Math.max(0, (next.t - key.t) * zoom) }}
            >
              {key.ease === 'step' && <span className="tl-seg-step" style={{ left: '100%' }} />}
            </span>
          );
        })}
        {keys.map((key) => {
          const selected = selectedKeys.some((item) => item.key === keyId && Math.abs(item.t - key.t) <= 1e-6);
          return (
            <button
              key={`key-${key.t}`}
              type="button"
              className={`tl-key tl-key-prop${selected ? ' selected' : ''}`}
              style={{ left: key.t * zoom }}
              data-keyframe={key.t}
              title={`${label} · ${formatNumber(key.t)} 帧 · 值 ${formatNumber(key.v)}${key.ease ? ` · ${easeToPreset(key.ease) || '自定义'}` : ''}`}
              onPointerDown={(event) => {
                selectTargetOwner(store, target);
                beginKeyDrag({ e: event, store, target, keyId, t: key.t, v: key.v, zoom, snap, duration });
              }}
              onDoubleClick={(event) => {
                event.stopPropagation();
                store.seek(key.t);
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

// ---------- 图层行 ----------

interface LayerRowProps {
  store: EditorStoreApi;
  layer: Layer;
  assets: Scene['assets'];
  catalog: Catalog | null;
  zoom: number;
  duration: number;
  fps: number;
  snap: boolean;
  selected: boolean;
  expanded: boolean;
  selectedKeys: KeySelection[];
  frame: number;
  sig: string;
}

function LayerDetails({
  store,
  layer,
  frame,
  zoom,
  duration,
  snap,
  selectedKeys,
}: Pick<LayerRowProps, 'store' | 'layer' | 'frame' | 'zoom' | 'duration' | 'snap' | 'selectedKeys'>): JSX.Element {
  const all = layerPropDefs(layer);
  const wanted = new Set([
    'transform.x',
    'transform.y',
    'transform.scale',
    'transform.scaleX',
    'transform.scaleY',
    'transform.rotation',
    'opacity',
  ]);
  const defs = all.filter((def) => {
    if (wanted.has(def.path)) return true;
    if (layer.clip && def.path.startsWith('clip.')) return true;
    return def.path.startsWith('filters.') && isCurve(def.curve);
  });

  return (
    <>
      {defs.map((def) => {
        const target: EditTarget = { kind: 'layer', id: layer.id, path: def.path };
        return (
          <PropertyTrackRow
            key={def.path}
            store={store}
            label={def.label}
            target={target}
            keyId={targetKey(target)}
            curve={def.curve}
            fallback={def.fallback}
            frame={frame}
            zoom={zoom}
            duration={duration}
            snap={snap}
            selectedKeys={selectedKeys}
            indent={34}
          />
        );
      })}
    </>
  );
}

const LayerRow = memo(
  function LayerRow(props: LayerRowProps): JSX.Element {
    const { store, layer, assets, catalog, zoom, duration, fps, snap, selected, expanded, selectedKeys, frame } = props;
    const targets = layerTargets(layer);
    const ticks = collectTicks(targets);
    const motionClips = layer.kind === 'model' ? layer.motions ?? [] : [];
    const expressionClips = layer.kind === 'model' ? layer.expressions ?? [] : [];
    const spanStart = layer.in * zoom;
    const spanEnd = Math.min(duration, layer.out) * zoom;
    const spanWidth = Math.max(1, spanEnd - spanStart);

    return (
      <>
        <div className={`tl-row tl-layer-row${selected ? ' tl-row-selected' : ''}`} data-testid={`track-layer-${layer.id}`}>
          <div
            className={`tl-head tl-layer-head${selected ? ' selected' : ''}`}
            title={`${layer.id} · ${layer.kind === 'model' ? '模型' : '图片'} · ${layer.in}–${layer.out}`}
            onClick={() => store.select({ kind: 'layer', id: layer.id })}
            onDoubleClick={(event) => {
              if ((event.target as HTMLElement).closest('button')) return;
              store.setExpanded(`layer:${layer.id}`);
            }}
          >
            <button
              type="button"
              className={`tl-chevron${expanded ? ' open' : ''}`}
              title={expanded ? '收起属性轨' : '展开属性轨'}
              onClick={(event) => {
                event.stopPropagation();
                store.setExpanded(`layer:${layer.id}`);
              }}
            >
              ▸
            </button>
            <span className={`tl-kind ${layer.kind}`} title={layer.kind === 'model' ? '模型图层' : '图片图层'}>
              {layer.kind === 'model' ? 'M' : 'I'}
            </span>
            <span className="tl-layer-id" title={layer.id}>
              {layer.id}
            </span>
            <span className="tl-range" title={`入点 ${layer.in} / 出点 ${layer.out}`}>
              {formatNumber(layer.in)}–{formatNumber(layer.out)}
            </span>
          </div>

          <div className="tl-lane tl-layer-lane">
            <div className={`tl-layer-span ${layer.kind}`} style={{ left: spanStart, width: spanWidth }} />
            <div
              className="tl-range-handle left"
              style={{ left: spanStart - 3 }}
              title={`拖动修改入点（当前 ${layer.in}）`}
              onPointerDown={(event) => beginLayerRangeDrag({ e: event, store, layer, edge: 'in', zoom, snap })}
            />
            <div
              className="tl-range-handle right"
              style={{ left: spanEnd - 4 }}
              title={`拖动修改出点（当前 ${layer.out}）`}
              onPointerDown={(event) => beginLayerRangeDrag({ e: event, store, layer, edge: 'out', zoom, snap })}
            />

            {motionClips.map((clip, index) => {
              const end = resolveMotionEnd(clip, assets, catalog, fps);
              if (clip.start >= duration) return null;
              const width = Math.max(2, (Math.min(duration, Math.max(clip.start + 1, end)) - clip.start) * zoom);
              const meta = findMotionMeta(clip, assets, catalog);
              return (
                <div
                  key={`motion-${index}`}
                  className="tl-clip motion"
                  data-testid={`motion-${layer.id}-${index}`}
                  style={{ left: clip.start * zoom, width }}
                  title={`motion ${clip.motion} · ${formatNumber(clip.start)}–${formatNumber(end)}${meta ? ` · ${meta.meta.duration.toFixed(2)}s` : ''}`}
                  onPointerDown={(event) => {
                    store.select({ kind: 'layer', id: layer.id });
                    beginClipDrag({
                      e: event,
                      store,
                      kind: 'motion',
                      layerId: layer.id,
                      index,
                      mode: 'move',
                      initialStart: clip.start,
                      initialEnd: end,
                      initialOffset: clip.offset ?? 0,
                      initialDurationSec: (end - clip.start) / fps,
                      hasExplicitEnd: clip.end !== undefined,
                      zoom,
                      snap,
                      fps,
                    });
                  }}
                  onDoubleClick={() => store.seek(clip.start)}
                >
                  <span className="tl-clip-label">motion · {clip.motion}</span>
                  <span
                    className="tl-clip-handle left"
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      store.select({ kind: 'layer', id: layer.id });
                      beginClipDrag({
                        e: event,
                        store,
                        kind: 'motion',
                        layerId: layer.id,
                        index,
                        mode: 'left',
                        initialStart: clip.start,
                        initialEnd: end,
                        initialOffset: clip.offset ?? 0,
                        initialDurationSec: (end - clip.start) / fps,
                        hasExplicitEnd: clip.end !== undefined,
                        zoom,
                        snap,
                        fps,
                      });
                    }}
                  />
                  <span
                    className="tl-clip-handle right"
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      store.select({ kind: 'layer', id: layer.id });
                      beginClipDrag({
                        e: event,
                        store,
                        kind: 'motion',
                        layerId: layer.id,
                        index,
                        mode: 'right',
                        initialStart: clip.start,
                        initialEnd: end,
                        initialOffset: clip.offset ?? 0,
                        initialDurationSec: (end - clip.start) / fps,
                        hasExplicitEnd: clip.end !== undefined,
                        zoom,
                        snap,
                        fps,
                      });
                    }}
                  />
                </div>
              );
            })}

            {expressionClips.map((clip, index) => {
              const end = resolveExpressionEnd(clip);
              if (clip.start >= duration) return null;
              const width = Math.max(2, (Math.min(duration, Math.max(clip.start + 1, end)) - clip.start) * zoom);
              return (
                <div
                  key={`expression-${index}`}
                  className="tl-clip expression"
                  data-testid={`expression-${layer.id}-${index}`}
                  style={{ left: clip.start * zoom, width }}
                  title={`expression ${clip.expression} · ${formatNumber(clip.start)}–${formatNumber(end)}`}
                  onPointerDown={(event) => {
                    store.select({ kind: 'layer', id: layer.id });
                    beginClipDrag({
                      e: event,
                      store,
                      kind: 'expression',
                      layerId: layer.id,
                      index,
                      mode: 'move',
                      initialStart: clip.start,
                      initialEnd: end,
                      initialOffset: 0,
                      initialDurationSec: (end - clip.start) / fps,
                      hasExplicitEnd: clip.end !== undefined,
                      zoom,
                      snap,
                      fps,
                    });
                  }}
                  onDoubleClick={() => store.seek(clip.start)}
                >
                  <span className="tl-clip-label">exp · {clip.expression}</span>
                  <span
                    className="tl-clip-handle left"
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      store.select({ kind: 'layer', id: layer.id });
                      beginClipDrag({
                        e: event,
                        store,
                        kind: 'expression',
                        layerId: layer.id,
                        index,
                        mode: 'left',
                        initialStart: clip.start,
                        initialEnd: end,
                        initialOffset: 0,
                        initialDurationSec: (end - clip.start) / fps,
                        hasExplicitEnd: clip.end !== undefined,
                        zoom,
                        snap,
                        fps,
                      });
                    }}
                  />
                  <span
                    className="tl-clip-handle right"
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      store.select({ kind: 'layer', id: layer.id });
                      beginClipDrag({
                        e: event,
                        store,
                        kind: 'expression',
                        layerId: layer.id,
                        index,
                        mode: 'right',
                        initialStart: clip.start,
                        initialEnd: end,
                        initialOffset: 0,
                        initialDurationSec: (end - clip.start) / fps,
                        hasExplicitEnd: clip.end !== undefined,
                        zoom,
                        snap,
                        fps,
                      });
                    }}
                  />
                </div>
              );
            })}

            {ticks.map((tick) => (
              <button
                key={`tick-${tick.t}`}
                type="button"
                className="tl-key tl-key-summary"
                style={{ left: tick.t * zoom }}
                data-keyframe={tick.t}
                title={`关键帧 · ${formatNumber(tick.t)} 帧 · ${timecode(tick.t, fps)}`}
                onPointerDown={(event) => event.stopPropagation()}
                onClick={() => {
                  store.seek(tick.t);
                  store.select({ kind: 'layer', id: layer.id });
                }}
              />
            ))}
          </div>
        </div>

        {expanded && (
          <LayerDetails
            store={store}
            layer={layer}
            frame={frame}
            zoom={zoom}
            duration={duration}
            snap={snap}
            selectedKeys={selectedKeys}
          />
        )}
      </>
    );
  },
  (previous, next) => {
    if (
      previous.store !== next.store ||
      previous.layer !== next.layer ||
      previous.assets !== next.assets ||
      previous.catalog !== next.catalog ||
      previous.zoom !== next.zoom ||
      previous.duration !== next.duration ||
      previous.fps !== next.fps ||
      previous.snap !== next.snap ||
      previous.selected !== next.selected ||
      previous.expanded !== next.expanded ||
      previous.sig !== next.sig
    ) {
      return false;
    }
    // 折叠行不显示播放头相关数值，因此播放时不因 frame 改变而重渲染。
    if (previous.expanded || next.expanded) {
      return previous.frame === next.frame && previous.selectedKeys === next.selectedKeys;
    }
    return true;
  },
);

// ---------- 时间线 ----------

export function Timeline({ store }: { store: EditorStoreApi }): JSX.Element {
  const [, forceRender] = useReducer((value: number) => value + 1, 0);
  useEffect(() => store.subscribe(forceRender), [store]);

  const state = store.getState();
  const scene = state.scene;
  const rawFps = scene?.timebase.fps;
  const fps = typeof rawFps === 'number' && Number.isFinite(rawFps) && rawFps > 0 ? rawFps : 60;
  const rawDuration = scene?.duration;
  const duration = typeof rawDuration === 'number' && Number.isFinite(rawDuration) && rawDuration >= 1 ? Math.round(rawDuration) : 1;
  const zoom = clampZoom(state.timelineZoom || 2);
  const playheadFrame = clampFrame(state.frame, duration);
  const laneWidth = Math.max(0.25, duration * zoom);
  const playheadX = playheadFrame * zoom;
  const badgeShift = playheadX < 24 ? 0 : playheadX > laneWidth - 24 ? -100 : -50;

  const [snap, setSnap] = useState(true);
  const [panMode, setPanMode] = useState(false);
  const [panning, setPanning] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const rulerLaneRef = useRef<HTMLDivElement>(null);
  const interactingRef = useRef(false);
  const panRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    scrollLeft: number;
    scrollTop: number;
  } | null>(null);
  const audioGainMemory = useRef(new Map<number, number>());

  const ensureVisible = (frame: number, zoomValue = zoom) => {
    const element = scrollRef.current;
    if (!element) return;
    const x = HEADER_WIDTH + frame * zoomValue;
    const left = element.scrollLeft + HEADER_WIDTH + 24;
    const right = element.scrollLeft + element.clientWidth - 80;
    if (x < left) element.scrollLeft = Math.max(0, x - HEADER_WIDTH - 24);
    else if (x > right) element.scrollLeft = Math.max(0, x - element.clientWidth + 80);
  };

  // 播放头移动时自动水平滚动；用户正在拖动标尺时暂停。
  useEffect(() => {
    if (interactingRef.current) return;
    ensureVisible(playheadFrame);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playheadFrame]);

  // Ctrl/Cmd + 滚轮围绕鼠标位置缩放；Shift + 滚轮水平滚动；普通滚轮交给浏览器纵向滚动。
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const rect = element.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        // 鼠标在固定轨道头上时不做锚点缩放，避免时间线无意义地跳动。
        if (mouseX < HEADER_WIDTH) return;
        const laneX = element.scrollLeft + mouseX - HEADER_WIDTH;
        const anchorFrame = Math.max(0, laneX / zoom);
        const nextZoom = clampZoom(zoom * Math.exp(-event.deltaY * 0.0015));
        if (Math.abs(nextZoom - zoom) < 1e-9) return;
        store.setTimelineZoom(nextZoom);
        requestAnimationFrame(() => {
          element.scrollLeft = Math.max(0, HEADER_WIDTH + anchorFrame * nextZoom - mouseX);
        });
      } else if (event.shiftKey) {
        event.preventDefault();
        const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
        element.scrollLeft += delta;
      }
    };
    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, [store, zoom]);

  const beginPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const element = scrollRef.current;
    if (!element) return;
    const middle = event.button === 1;
    const leftInPanMode = panMode && event.button === 0;
    if (!middle && !leftInPanMode) return;
    if (leftInPanMode) {
      const target = event.target as HTMLElement;
      // 平移模式下仍然让按钮、关键帧、片段、标记响应自己的点击。
      if (target.closest('button, input, select, textarea, .tl-key, .tl-clip, .tl-audio-bar, .tl-range-handle, .tl-marker')) return;
    }
    event.preventDefault();
    element.setPointerCapture(event.pointerId);
    panRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: element.scrollLeft,
      scrollTop: element.scrollTop,
    };
    setPanning(true);
  };

  const movePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    const element = scrollRef.current;
    if (!pan || !element) return;
    element.scrollLeft = Math.max(0, pan.scrollLeft - (event.clientX - pan.startX));
    element.scrollTop = Math.max(0, pan.scrollTop - (event.clientY - pan.startY));
  };

  const endPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    const element = scrollRef.current;
    if (!pan) return;
    panRef.current = null;
    setPanning(false);
    try {
      element?.releasePointerCapture(event.pointerId);
    } catch {
      /* 指针已经释放 */
    }
  };

  const beginScrub = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0 || panMode) return;
    const lane = rulerLaneRef.current;
    if (!lane) return;
    event.preventDefault();
    interactingRef.current = true;
    const seekAt = (clientX: number) => {
      const rect = lane.getBoundingClientRect();
      const raw = (clientX - rect.left) / zoom;
      const frame = snap ? Math.round(raw) : Math.round(raw * 10) / 10;
      store.seek(clampFrame(frame, duration));
    };
    seekAt(event.clientX);
    const onMove = (moveEvent: PointerEvent) => {
      moveEvent.preventDefault();
      seekAt(moveEvent.clientX);
    };
    const onUp = () => {
      interactingRef.current = false;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      ensureVisible(store.getState().frame);
    };
    window.addEventListener('pointermove', onMove, { passive: false });
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  };

  const onRowsPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest('.tl-key, .tl-clip, .tl-audio-bar, .tl-range-handle, .tl-marker, button')) return;
    if (!target.closest('.tl-lane')) return;
    beginScrub(event);
  };

  const setZoom = (nextZoom: number) => {
    const clamped = clampZoom(nextZoom);
    store.setTimelineZoom(clamped);
    requestAnimationFrame(() => ensureVisible(store.getState().frame, clamped));
  };

  // 缩放滑块用对数刻度：低倍率区间（整段全览）更容易微调，高倍率也不会一步跳太远。
  const zoomSliderValue = Math.round(
    ((Math.log(zoom) - Math.log(MIN_ZOOM)) / (Math.log(MAX_ZOOM) - Math.log(MIN_ZOOM))) * 1000,
  );
  const setZoomFromSlider = (value: number) => {
    const ratio = clamp(value, 0, 1000) / 1000;
    setZoom(Math.exp(Math.log(MIN_ZOOM) + ratio * (Math.log(MAX_ZOOM) - Math.log(MIN_ZOOM))));
  };

  const fit = () => {
    const element = scrollRef.current;
    const available = Math.max(120, (element?.clientWidth ?? 1200) - HEADER_WIDTH - 16);
    store.setTimelineZoom(clampZoom(available / duration));
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollLeft = 0;
    });
  };

  const removeSelectedKeys = () => {
    if (state.selectedKeys.length === 0) return;
    const commands: EditCommand[] = state.selectedKeys.map((key) => ({
      op: 'removeKeyframe',
      target: key.target,
      t: key.t,
    }));
    try {
      void store
        .apply(commands, { history: true, label: `删除关键帧（${commands.length}）` })
        .then(() => store.setSelectedKeys([]))
        .catch((error) => notifyError(store, '删除关键帧失败', error));
    } catch (error) {
      notifyError(store, '删除关键帧失败', error);
    }
  };

  const easeInfo = scene ? findEaseInfo(scene, state) : null;
  const easeValue = easeInfo ? easeToPreset(easeInfo.ease) : '';

  const applyEase = (preset: EasePreset) => {
    const keys: KeySelection[] =
      state.selectedKeys.length > 0
        ? state.selectedKeys
        : easeInfo
          ? [{ key: easeInfo.keyId, target: easeInfo.target, t: easeInfo.t }]
          : [];
    if (keys.length === 0) return;
    const groups = new Map<string, { target: EditTarget; times: number[] }>();
    for (const key of keys) {
      const id = JSON.stringify(key.target);
      const group = groups.get(id) ?? { target: key.target, times: [] };
      if (!group.times.some((time) => Math.abs(time - key.t) <= 1e-6)) group.times.push(key.t);
      groups.set(id, group);
    }
    const commands: EditCommand[] = [...groups.values()].map((group) => ({
      op: 'setEase',
      target: group.target,
      times: group.times,
      ease: preset,
    }));
    try {
      void store.apply(commands, { history: true, label: `缓动：${EASE_LABELS[preset]}` }).catch((error) => notifyError(store, '设置缓动失败', error));
    } catch (error) {
      notifyError(store, '设置缓动失败', error);
    }
  };

  const addKeyTargets: TargetDef[] = [];
  const currentSelection = state.selection;
  if (scene && currentSelection.kind === 'layer') {
    const layer = scene.layers.find((item) => item.id === currentSelection.id);
    if (layer) {
      for (const def of layerTargets(layer)) if (isCurve(def.curve)) addKeyTargets.push(def);
    }
  } else if (scene && currentSelection.kind === 'camera' && scene.camera) {
    for (const def of cameraTargets(scene.camera)) if (isCurve(def.curve)) addKeyTargets.push(def);
  }

  const addKeysAtPlayhead = () => {
    if (addKeyTargets.length === 0) return;
    const commands: EditCommand[] = addKeyTargets.map((def) => ({
      op: 'addKeyframe',
      target: def.target,
      t: state.frame,
      v: sample(def.curve, state.frame, def.fallback),
    }));
    try {
      void store.apply(commands, { history: true, label: `打关键帧（${commands.length}）` }).catch((error) => notifyError(store, '打关键帧失败', error));
    } catch (error) {
      notifyError(store, '打关键帧失败', error);
    }
  };

  const tickStep = (() => {
    for (const step of TICK_STEPS) if (step * zoom >= 58) return step;
    return TICK_STEPS[TICK_STEPS.length - 1]!;
  })();
  const labelStep = (() => {
    let step = TICK_STEPS[TICK_STEPS.length - 1]!;
    for (const candidate of TICK_STEPS) {
      if (candidate * zoom >= 170) {
        step = candidate;
        break;
      }
    }
    // 让主刻度一定落在普通刻度上。
    return Math.max(tickStep, Math.ceil(step / tickStep) * tickStep);
  })();

  const ticks: JSX.Element[] = [];
  if (scene) {
    let count = 0;
    for (let frame = 0; frame <= duration && count < 4000; frame += tickStep, count++) {
      const major = Math.round(frame) % labelStep === 0;
      ticks.push(
        <div key={`tick-${frame}`} className={`tl-tick${major ? ' major' : ''}`} style={{ left: frame * zoom }}>
          {major && (
            <span className="tl-tick-label">
              <span className="tl-tick-frame">{formatNumber(frame)}</span>
              <span className="tl-tick-time">{timecode(frame, fps)}</span>
            </span>
          )}
        </div>,
      );
    }
  }

  const contentStyle = {
    width: HEADER_WIDTH + laneWidth,
    '--tl-lane-w': `${laneWidth}px`,
    '--tl-grid': `${Math.max(1, tickStep * zoom)}px`,
  } as CSSProperties;

  const markers = scene?.markers ?? [];
  const audioClips = scene?.audio ?? [];
  const camera = scene?.camera;
  const cameraTicks = camera ? collectCameraTicks(camera) : [];
  const pinned = state.pinned;

  return (
    <div className="timeline" data-testid="timeline">
      <div className="tl-toolbar">
        <button
          type="button"
          className="tl-btn tl-btn-play"
          data-testid="timeline-play"
          disabled={!scene}
          title={state.playing ? '暂停' : '播放'}
          onClick={() => {
            void store.togglePlay().catch((error) => notifyError(store, state.playing ? '暂停失败' : '播放失败', error));
          }}
        >
          {state.playing ? '⏸' : '▶'}
        </button>
        <button type="button" className="tl-btn" disabled={!scene} title="后退 10 帧" data-testid="timeline-step-back-10" onClick={() => store.step(-10)}>
          -10
        </button>
        <button type="button" className="tl-btn" disabled={!scene} title="后退 1 帧" data-testid="timeline-step-back-1" onClick={() => store.step(-1)}>
          -1
        </button>
        <button type="button" className="tl-btn" disabled={!scene} title="前进 1 帧" data-testid="timeline-step-forward-1" onClick={() => store.step(1)}>
          +1
        </button>
        <button type="button" className="tl-btn" disabled={!scene} title="前进 10 帧" data-testid="timeline-step-forward-10" onClick={() => store.step(10)}>
          +10
        </button>
        <button type="button" className="tl-btn" disabled={!scene} title="跳到开头（Home）" data-testid="timeline-home" onClick={() => store.seek(0)}>
          ⏮
        </button>
        <button
          type="button"
          className="tl-btn"
          disabled={!scene}
          title="跳到结尾（End）"
          data-testid="timeline-end"
          onClick={() => store.seek(Math.max(0, duration - 1))}
        >
          ⏭
        </button>

        <span className="tl-sep" />
        <span className="tl-time" title="当前帧 / 时间码">
          <b className="tl-frame">{formatNumber(playheadFrame)}</b>
          <span className="tl-timecode">{timecode(playheadFrame, fps)}</span>
        </span>

        <span className="tl-sep" />
        <button type="button" className="tl-btn" title="缩小时间线（Ctrl+滚轮）" data-testid="timeline-zoom-out" onClick={() => setZoom(zoom / 1.5)}>
          −
        </button>
        <input
          className="tl-zoom-slider"
          type="range"
          min={0}
          max={1000}
          step={1}
          value={zoomSliderValue}
          data-testid="timeline-zoom-slider"
          title="时间线缩放（对数刻度；Ctrl+滚轮围绕鼠标缩放）"
          onChange={(event) => setZoomFromSlider(Number(event.target.value))}
        />
        <button type="button" className="tl-btn" title="放大时间线（Ctrl+滚轮）" data-testid="timeline-zoom-in" onClick={() => setZoom(zoom * 1.5)}>
          +
        </button>
        <span className="tl-zoom-read" title="每帧像素">
          {zoom.toFixed(2)} px/帧
        </span>
        <button type="button" className="tl-btn" data-testid="timeline-fit" title="缩放到完整时长" onClick={fit}>
          适应
        </button>
        <button
          type="button"
          className={`tl-btn${panMode ? ' active' : ''}`}
          data-testid="timeline-pan"
          title="平移模式：左键拖动时间线；中键拖动始终可用"
          onClick={() => setPanMode((value) => !value)}
        >
          平移
        </button>
        <button
          type="button"
          className={`tl-btn${snap ? ' active' : ''}`}
          data-testid="timeline-snap"
          title="吸附到整数帧（拖动时按住 Alt 临时关闭）"
          onClick={() => setSnap((value) => !value)}
        >
          {snap ? '吸附开' : '吸附关'}
        </button>

        <span className="tl-sep" />
        <select
          className="tl-select"
          data-testid="timeline-ease"
          value={easeValue}
          disabled={!easeInfo}
          title={easeInfo ? '为选中的关键帧设置缓动（作用于到下一帧的区间）' : '选中关键帧或在播放头下放置关键帧后可设置缓动'}
          onChange={(event) => {
            if (event.target.value) applyEase(event.target.value as EasePreset);
          }}
        >
          {easeValue === '' && <option value="">自定义</option>}
          {EASE_ORDER.map((preset) => (
            <option key={preset} value={preset}>
              {EASE_LABELS[preset]}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="tl-btn danger"
          data-testid="timeline-delete-keys"
          disabled={state.selectedKeys.length === 0}
          title={state.selectedKeys.length > 0 ? `删除选中的 ${state.selectedKeys.length} 个关键帧` : '先选中关键帧'}
          onClick={removeSelectedKeys}
        >
          删关键帧
        </button>
        <button
          type="button"
          className="tl-btn"
          data-testid="timeline-add-key"
          disabled={addKeyTargets.length === 0}
          title={
            addKeyTargets.length > 0
              ? `为选中对象当前帧的 ${addKeyTargets.length} 条动画属性打关键帧`
              : '选中含动画属性的图层或相机后可快速打关键帧'
          }
          onClick={addKeysAtPlayhead}
        >
          打关键帧
        </button>
        <span className="tl-hint" title="Ctrl+滚轮围绕鼠标缩放；中键拖动或在“平移”模式下左键拖动时间线">
          Ctrl+滚轮缩放 · 中键拖动平移 · 双击图层头展开
        </span>
      </div>

      {scene ? (
        <div
          className={`tl-scroll${panMode ? ' pan-mode' : ''}${panning ? ' panning' : ''}`}
          ref={scrollRef}
          onPointerDown={beginPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onPointerCancel={endPan}
          onMouseDown={(event) => {
            // 阻止 Chromium 在中键按下时进入自动滚动模式。
            if (event.button === 1) event.preventDefault();
          }}
          onAuxClick={(event) => {
            if (event.button === 1) event.preventDefault();
          }}
        >
          <div className="tl-content" style={contentStyle}>
            <div className="tl-ruler-row">
              <div className="tl-ruler-head">
                <span>时间线</span>
                <b>{formatNumber(fps)} fps</b>
                <span className="tl-ruler-duration">{formatNumber(duration)} 帧</span>
              </div>
              <div
                className="tl-ruler-lane"
                data-testid="timeline-ruler"
                ref={rulerLaneRef}
                title="点击 / 拖动跳转播放头"
                onPointerDown={beginScrub}
              >
                {ticks}
                <div className="tl-playhead-head" style={{ left: playheadX }}>
                  <span className="tl-playhead-badge" style={{ transform: `translateX(${badgeShift}%)` }}>
                    {Math.round(playheadFrame)}
                  </span>
                </div>
              </div>
            </div>

            <div className="tl-rows" onPointerDown={onRowsPointerDown}>
              <div className="tl-row tl-marker-row" data-testid="track-markers">
                <div className="tl-head tl-marker-head" title="镜头与标记">
                  <span>🎬 镜头 / 标记</span>
                  <span className="tl-count">{markers.length}</span>
                </div>
                <div className="tl-lane tl-marker-lane">
                  {markers.map((marker) => {
                    const selected = state.selection.kind === 'marker' && state.selection.id === marker.id;
                    const width = marker.end !== undefined ? Math.max(2, (marker.end - marker.start) * zoom) : 3;
                    return (
                      <button
                        key={marker.id}
                        type="button"
                        className={`tl-marker${marker.kind === 'shot' ? ' shot' : ''}${marker.end === undefined ? ' point' : ''}${
                          selected ? ' selected' : ''
                        }`}
                        style={{ left: marker.start * zoom, width }}
                        data-testid={`marker-${marker.id}`}
                        title={`${marker.label ?? marker.id}${marker.kind ? ` · ${marker.kind}` : ''} · ${formatNumber(marker.start)}${
                          marker.end !== undefined ? `–${formatNumber(marker.end)}` : ''
                        } · ${timecode(marker.start, fps)}`}
                        onPointerDown={(event) => event.stopPropagation()}
                        onClick={() => {
                          store.seek(marker.start);
                          store.select({ kind: 'marker', id: marker.id });
                        }}
                      >
                        <span className="tl-marker-label">{marker.label ?? marker.id}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="tl-row tl-camera-row" data-testid="track-camera">
                <div
                  className={`tl-head tl-camera-head${state.selection.kind === 'camera' ? ' selected' : ''}`}
                  title={camera ? '相机轨道（拖动关键帧修改相机动画）' : '场景没有相机'}
                  onClick={() => {
                    if (camera) store.select({ kind: 'camera' });
                  }}
                >
                  <span>🎥 相机</span>
                  {camera ? (
                    <span className="tl-range" title="当前相机缩放">
                      z {formatNumber(sample(camera.zoom, playheadFrame, 1))}
                    </span>
                  ) : (
                    <span className="tl-range">未启用</span>
                  )}
                </div>
                <div className="tl-lane tl-camera-lane">
                  {camera ? (
                    cameraTicks.map((tick) => {
                      const keyId = targetKey(tick.target);
                      const selected = state.selectedKeys.some(
                        (item) => item.key === keyId && Math.abs(item.t - tick.t) <= 1e-6,
                      );
                      return (
                        <button
                          key={`camera-key-${tick.t}`}
                          type="button"
                          className={`tl-key tl-key-camera${selected ? ' selected' : ''}`}
                          style={{ left: tick.t * zoom }}
                          data-keyframe={tick.t}
                          title={`相机关键帧 · ${formatNumber(tick.t)} 帧 · ${timecode(tick.t, fps)}`}
                          onPointerDown={(event) => {
                            store.select({ kind: 'camera' });
                            beginKeyDrag({
                              e: event,
                              store,
                              target: tick.target,
                              keyId,
                              t: tick.t,
                              v: tick.v,
                              zoom,
                              snap,
                              duration,
                            });
                          }}
                          onDoubleClick={(event) => {
                            event.stopPropagation();
                            store.seek(tick.t);
                          }}
                        />
                      );
                    })
                  ) : (
                    <button
                      type="button"
                      className="tl-inline-enable"
                      title="给场景添加一个居中、1 倍缩放的相机"
                      onPointerDown={(event) => event.stopPropagation()}
                      onClick={() => {
                        void store
                          .apply(
                            {
                              op: 'setCamera',
                              camera: {
                                x: scene.composition.width / 2,
                                y: scene.composition.height / 2,
                                zoom: 1,
                              },
                            },
                            { history: true, label: '启用相机' },
                          )
                          .catch((error) => notifyError(store, '启用相机失败', error));
                      }}
                    >
                      + 启用相机
                    </button>
                  )}
                </div>
              </div>

              <div className="tl-row tl-group-row" data-testid="track-audio-group">
                <div className="tl-head tl-group-head">
                  <span>🎵 音频 / 音乐轨</span>
                  <span className="tl-count">{audioClips.length}</span>
                  <span className="tl-group-hint">从左侧素材面板导入</span>
                </div>
                <div className="tl-lane tl-group-lane" />
              </div>

              {audioClips.map((clip, index) => {
                const rawDurationSec = clip.duration ?? state.audioDurations[clip.audio] ?? 5;
                const durationSec =
                  typeof rawDurationSec === 'number' && Number.isFinite(rawDurationSec) && rawDurationSec > 0 ? rawDurationSec : 5;
                const durationFrames = Math.max(1, Math.round(durationSec * fps));
                const end = clip.start + durationFrames;
                const width = Math.max(2, (Math.min(duration, end) - clip.start) * zoom);
                const gain = clip.gain ?? 1;
                const selected = state.selection.kind === 'audio' && state.selection.index === index;
                const drag = (mode: ClipDragMode) => (event: ReactPointerEvent<HTMLElement>) => {
                  event.stopPropagation();
                  store.select({ kind: 'audio', index });
                  beginClipDrag({
                    e: event,
                    store,
                    kind: 'audio',
                    index,
                    mode,
                    initialStart: clip.start,
                    initialEnd: end,
                    initialOffset: clip.offset ?? 0,
                    initialDurationSec: durationSec,
                    hasExplicitEnd: clip.duration !== undefined,
                    zoom,
                    snap,
                    fps,
                  });
                };
                return (
                  <div
                    key={`audio-${index}`}
                    className={`tl-row tl-audio-row${selected ? ' tl-row-selected' : ''}`}
                    data-testid={`track-audio-${index}`}
                  >
                    <div
                      className={`tl-head tl-audio-head${selected ? ' selected' : ''}`}
                      title={`${clip.audio} · ${formatNumber(clip.start)}–${formatNumber(end)} · ${durationSec.toFixed(2)}s`}
                      onClick={() => store.select({ kind: 'audio', index })}
                    >
                      <span className="tl-audio-id">🎵 {clip.audio}</span>
                      <span className="tl-range">{durationSec.toFixed(2)}s</span>
                      <button
                        type="button"
                        className={`tl-icon-btn${gain === 0 ? ' active' : ''}`}
                        title={gain === 0 ? '取消静音' : '静音'}
                        onClick={(event) => {
                          event.stopPropagation();
                          const remembered = audioGainMemory.current.get(index) ?? 1;
                          if (gain > 0) audioGainMemory.current.set(index, gain);
                          const nextGain = gain > 0 ? 0 : remembered > 0 ? remembered : 1;
                          void store
                            .apply({ op: 'updateAudio', index, patch: { gain: nextGain } }, { history: true, label: '音频静音' })
                            .catch((error) => notifyError(store, '修改音频失败', error));
                        }}
                      >
                        M
                      </button>
                      <button
                        type="button"
                        className="tl-icon-btn unpin"
                        title="移除音频片段"
                        onClick={(event) => {
                          event.stopPropagation();
                          try {
                            void store
                              .apply({ op: 'removeAudio', index }, { history: true, label: '移除音频' })
                              .then(() => store.select({ kind: 'scene' }))
                              .catch((error) => notifyError(store, '移除音频失败', error));
                          } catch (error) {
                            notifyError(store, '移除音频失败', error);
                          }
                        }}
                      >
                        ×
                      </button>
                    </div>
                    <div className="tl-lane tl-audio-lane">
                      <div
                        className={`tl-audio-bar${selected ? ' selected' : ''}`}
                        data-testid={`audio-${index}`}
                        style={{ left: clip.start * zoom, width }}
                        title={`${clip.audio} · 起点 ${formatNumber(clip.start)} · 时长 ${durationSec.toFixed(2)}s · 增益 ${gain}`}
                        onPointerDown={drag('move')}
                        onDoubleClick={() => store.seek(clip.start)}
                      >
                        <span
                          className="tl-audio-gain"
                          style={{ width: `${clamp(gain, 0, 1) * 100}%` }}
                        />
                        <span className="tl-audio-label">
                          {clip.audio}
                          {gain !== 1 ? ` · ${Math.round(gain * 100)}%` : ''}
                        </span>
                        <span className="tl-clip-handle left" onPointerDown={drag('left')} />
                        <span className="tl-clip-handle right" onPointerDown={drag('right')} />
                      </div>
                    </div>
                  </div>
                );
              })}

              <div className="tl-row tl-group-row" data-testid="track-layers-group">
                <div className="tl-head tl-group-head">
                  <span>图层</span>
                  <span className="tl-count">{scene.layers.length}</span>
                  <span className="tl-group-hint">双击展开属性</span>
                </div>
                <div className="tl-lane tl-group-lane" />
              </div>

              {scene.layers.map((layer) => (
                <LayerRow
                  key={layer.id}
                  store={store}
                  layer={layer}
                  assets={scene.assets}
                  catalog={state.catalog}
                  zoom={zoom}
                  duration={duration}
                  fps={fps}
                  snap={snap}
                  selected={state.selection.kind === 'layer' && state.selection.id === layer.id}
                  expanded={Boolean(state.expanded[`layer:${layer.id}`])}
                  selectedKeys={state.selectedKeys}
                  frame={state.frame}
                  sig={layerSignature(layer)}
                />
              ))}

              {state.showParams && pinned.length > 0 && (
                <>
                  <div className="tl-row tl-group-row tl-param-group" data-testid="track-params-group">
                    <div className="tl-head tl-group-head">
                      <span>🎛 参数</span>
                      <span className="tl-count">{pinned.length}</span>
                      <span className="tl-group-hint">属性面板固定</span>
                    </div>
                    <div className="tl-lane tl-group-lane" />
                  </div>
                  {pinned.map((entry) => {
                    const splitAt = entry.indexOf('::');
                    if (splitAt < 0) return null;
                    const layerId = entry.slice(0, splitAt);
                    const param = entry.slice(splitAt + 2);
                    if (!layerId || !param) return null;
                    const layer = scene.layers.find((item) => item.id === layerId);
                    const curve = layer?.kind === 'model' ? layer.params?.find((track) => track.id === param)?.curve : undefined;
                    const target: EditTarget = { kind: 'param', id: layerId, param };
                    return (
                      <PropertyTrackRow
                        key={entry}
                        store={store}
                        label={param}
                        sublabel={layerId}
                        target={target}
                        keyId={targetKey(target)}
                        curve={curve}
                        fallback={0}
                        frame={state.frame}
                        zoom={zoom}
                        duration={duration}
                        snap={snap}
                        selectedKeys={state.selectedKeys}
                        indent={26}
                        onUnpin={() => store.togglePinnedParam(layerId, param)}
                      />
                    );
                  })}
                </>
              )}
            </div>

            <div
              className="tl-playhead"
              data-frame={playheadFrame}
              style={{ left: HEADER_WIDTH + playheadFrame * zoom }}
              title={`播放头 · ${Math.round(playheadFrame)} 帧 · ${timecode(playheadFrame, fps)}`}
            />
          </div>
        </div>
      ) : (
        <div className="tl-empty">没有打开的场景</div>
      )}
    </div>
  );
}
