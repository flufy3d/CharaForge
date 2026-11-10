/*
 * 编辑器状态与命令层。
 *
 * 所有 UI（面板、检查器、时间线）都只通过 EditorStoreApi / 本文件的 EditorStore 修改场景；
 * 场景对象由 @charaforge/scene 的 applyEditCommands 原地修改，再把 effects 交给 PlayerEngine，
 * 让正在运行的运行时重建求值器 / 时钟 / Worker 副本，不刷新页面。
 */
import { asciiUrl, type CanvasInfo } from '@charaforge/runtime';
import {
  MAX_IMAGE_PIXELS,
  MAX_IMAGE_SIDE,
  applyEditChanges,
  applyEditCommands,
  parseScene,
  sample,
  undoEditChanges,
  type AnimationPreset,
  type EditCommand,
  type EditEffect,
  type EditResult,
  type ImageAsset,
  type ImageLayer,
  type Layer,
  type ModelLayer,
  type Scene,
} from '@charaforge/scene';
import { PlayerEngine, type EngineStatus } from '../engine.ts';
import {
  ANIMATION_DIR,
  EXP3_DIR,
  PREVIEW_MODEL_URI,
  assetUriFromDir,
  declaredAssetUri,
  declaredExpressionFiles,
  declaredMotionFiles,
  displayName,
  isExpressionFile,
  isMotionFile,
  parseExpressionInfo,
  parseMotionMeta,
  sortAssets,
  type AssetFileInfo,
  type AssetSource,
  type Catalog,
  type ExpressionAssetInfo,
  type MotionAssetInfo,
} from '../motionCatalog.ts';
import { bytesToBase64, pageUrl } from '../util.ts';
import type {
  ApplyOptions,
  EditorNotice,
  EditorState,
  EditorStoreApi,
  EditorTool,
  EditorWindowApi,
  ExportProgress,
  ImportImageOptions,
  KeySelection,
  Selection,
} from './contracts.ts';
import { exportSceneVideo, type ExportVideoResult } from './export.ts';

// 编辑器默认打开“带背景和小元素”的 MV；原始拟合场景仍是 scene.json。
export const DEFAULT_SCENE_PATH = 'scenes/deepseek-chan/scene.mv.json';

const MAX_HISTORY = 100;
const COALESCE_MS = 1500;
const NOTICE_LIMIT = 6;
const NOTICE_TTL = 8000;

interface HistoryEntry {
  changes: EditResult['changes'];
  effects: EditEffect[];
  label: string;
  time: number;
}

interface UploadedAsset {
  uri: string;
  sha256?: string;
}

interface EditorStoreOptions {
  search?: URLSearchParams;
}

function errorMessage(e: unknown): string {
  return String((e as Error)?.message ?? e);
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

/** 场景 ID 只允许 [A-Za-z0-9_.:-]；中文文件名会退化成 fallback，再由 uniqueId 编号。 */
export function sanitizeId(raw: string, fallback = 'item'): string {
  const cleaned = raw
    .replace(/[^A-Za-z0-9_.:-]+/g, '-')
    .replace(/^[-.:]+|[-.:]+$/g, '')
    .slice(0, 80);
  return cleaned || fallback;
}

function uniqueId(base: string, used: Set<string>): string {
  let id = base;
  let n = 2;
  while (used.has(id)) id = `${base}-${n++}`;
  return id;
}

function usedIds(scene: Scene): Set<string> {
  const used = new Set<string>();
  for (const layer of scene.layers) used.add(layer.id);
  for (const list of [scene.assets.models, scene.assets.motions, scene.assets.expressions, scene.assets.audio, scene.assets.images]) {
    for (const asset of list ?? []) used.add(asset.id);
  }
  return used;
}

function basename(path: string): string {
  const clean = path.split(/[?#]/, 1)[0] ?? path;
  return clean.split(/[\\/]/).pop() ?? clean;
}

function stem(name: string): string {
  return name.replace(/\.[^.]+$/, '');
}

function downloadBlob(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

/** 下载文件名允许中文，只去掉 Windows / URL 不友好的字符。 */
function safeFileName(name: string, fallback = 'charaforge'): string {
  const clean = name
    .replace(/[<>:"/\\|?*\0-\x1f]/g, '-')
    .replace(/\s+/g, ' ')
    .replace(/^[.\s]+|[.\s]+$/g, '')
    .slice(0, 160);
  return clean || fallback;
}

/** 与 server/editorApi.ts 的 sceneJsonText 保持一致：小场景可读，大场景紧凑。 */
function sceneJsonText(scene: Scene): string {
  const params = scene.layers.reduce((n, layer) => n + (layer.kind === 'model' ? layer.params?.length ?? 0 : 0), 0);
  const compact = scene.layers.length > 20 || params > 2000;
  return `${JSON.stringify(scene, null, compact ? 0 : 2)}\n`;
}

function isPng(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  );
}

function isWebp(bytes: Uint8Array): boolean {
  return (
    bytes.length >= 12 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  );
}

function effectSet(effects: EditEffect[]): EditEffect[] {
  const order: EditEffect[] = ['render', 'workers', 'evaluator', 'structure', 'audio', 'clock'];
  const set = new Set(effects);
  return order.filter((e) => set.has(e));
}

function commandLabel(commands: EditCommand[]): string {
  const first = commands[0];
  if (!first) return '编辑';
  const record = first as unknown as Record<string, unknown>;
  const op = typeof record.op === 'string' ? record.op : '编辑';
  const target = record.id ?? record.layer ?? record.uri;
  return typeof target === 'string' ? `${op}:${target}` : op;
}

function isAbort(e: unknown): boolean {
  return (e as { name?: unknown } | null)?.name === 'AbortError';
}

function defaultWorkers(params: URLSearchParams): number {
  if (params.has('workers')) {
    const n = Number(params.get('workers'));
    return Number.isFinite(n) ? clamp(Math.floor(n), 0, 4) : 0;
  }
  return Math.max(2, Math.min(4, Math.floor((navigator.hardwareConcurrency || 4) / 2) - 1));
}

export class EditorStore implements EditorStoreApi {
  readonly engine = new PlayerEngine();

  private state: EditorState;
  private readonly listeners = new Set<() => void>();
  private readonly params: URLSearchParams;
  private canvas: HTMLCanvasElement | null = null;
  private bootPromise: Promise<void> | null = null;
  private booted = false;
  private destroyed = false;

  private readonly undoStack: HistoryEntry[] = [];
  private readonly redoStack: HistoryEntry[] = [];
  private readonly noticeTimers = new Map<number, number>();
  private nextNoticeId = 1;

  private lastStatusAt = 0;
  private lastStatusSignature = '';
  private lastExportProgressAt = 0;
  private audioCache = new Map<string, number>();
  private catalogPromise: Promise<void> | null = null;
  private windowApi: EditorWindowApi | null = null;

  constructor(opts: EditorStoreOptions = {}) {
    this.params = opts.search ?? new URLSearchParams(typeof location !== 'undefined' ? location.search : '');
    const scenePath = this.params.get('scene') || DEFAULT_SCENE_PATH;
    this.state = {
      phase: 'booting',
      stage: '初始化',
      error: null,
      scene: null,
      scenePath,
      savePath: scenePath,
      frame: 0,
      playing: false,
      status: null,
      selection: { kind: 'scene' },
      expanded: {},
      selectedKeys: [],
      pinned: [],
      dirty: false,
      canUndo: false,
      canRedo: false,
      notices: [],
      exporting: null,
      catalog: null,
      catalogError: null,
      audioDurations: {},
      activeTool: 'layers',
      timelineZoom: 1,
      showParams: true,
      dragging: false,
    };
  }

  getState(): EditorState {
    return this.state;
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  private setState(patch: Partial<EditorState>): void {
    if (this.destroyed) return;
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }

  private get scene(): Scene | null {
    return this.state.scene;
  }

  private requireScene(): Scene {
    const scene = this.state.scene;
    if (!scene) throw new Error('场景尚未加载');
    return scene;
  }

  get frame(): number {
    return this.state.frame;
  }

  /** 画布上选中的图层（不是模型图层时也返回）。 */
  selectedLayer(): Layer | null {
    const selection = this.state.selection;
    if (selection.kind !== 'layer') return null;
    return this.state.scene?.layers.find((layer) => layer.id === selection.id) ?? null;
  }

  selectedModelLayer(): ModelLayer | null {
    const layer = this.selectedLayer();
    return layer?.kind === 'model' ? layer : null;
  }

  resolveAsset(uri: string): string {
    return this.engine.runtime?.resolveAsset(uri) ?? pageUrl(`assets/${uri.replace(/^\/+/, '')}`);
  }

  imageAsset(id: string): ImageAsset | undefined {
    return this.state.scene?.assets.images?.find((asset) => asset.id === id);
  }

  /** 图层本地坐标系的宽高（模型画布 / 图片像素）；给舞台包围盒用。 */
  layerLocalSize(layer: Layer): { width: number; height: number } | null {
    if (layer.kind === 'model') {
      const canvas = this.engine.runtime?.layerInstance(layer.id)?.canvas;
      return canvas ? { width: canvas.width, height: canvas.height } : null;
    }
    const asset = this.imageAsset(layer.image);
    return asset ? { width: asset.width, height: asset.height } : null;
  }

  modelCanvas(layerId?: string): CanvasInfo | null {
    const id = layerId ?? this.state.scene?.layers.find((layer) => layer.kind === 'model')?.id;
    if (!id) return null;
    return this.engine.runtime?.layerInstance(id)?.canvas ?? null;
  }

  /** 取当前帧某个动画目标的值（K 键 / 参数面板展示用）。 */
  sampleTarget(target: import('@charaforge/scene').EditTarget): { value: number; curve: import('@charaforge/scene').Animatable<number> | undefined } {
    const scene = this.state.scene;
    const frame = this.state.frame;
    if (!scene) return { value: 0, curve: undefined };
    if (target.kind === 'camera') {
      const camera = scene.camera;
      if (!camera) return { value: 0, curve: undefined };
      const key = target.path.replace(/^camera\./, '') as 'x' | 'y' | 'zoom' | 'rotation';
      const value = camera[key];
      return { value: sample(value, frame, key === 'zoom' ? 1 : 0), curve: value };
    }
    if (target.kind === 'param') {
      const layer = scene.layers.find((item): item is ModelLayer => item.kind === 'model' && item.id === target.id);
      const track = layer?.params?.find((item) => item.id === target.param);
      if (track) return { value: sample(track.curve, frame, 0), curve: track.curve };
      const live = this.engine.runtime?.layerInstance(target.id)?.getParam(target.param);
      return { value: Number.isFinite(live) ? (live as number) : 0, curve: undefined };
    }
    const layer = scene.layers.find((item) => item.id === target.id);
    if (!layer) return { value: 0, curve: undefined };
    const path = target.path.replace(/^layer\./, '');
    if (path === 'opacity') return { value: sample(layer.opacity, frame, 1), curve: layer.opacity };
    if (path.startsWith('transform.')) {
      const key = path.slice('transform.'.length) as 'x' | 'y' | 'scale' | 'scaleX' | 'scaleY' | 'rotation';
      const value = layer.transform[key];
      const fallback = key === 'scale' || key === 'scaleX' || key === 'scaleY' ? 1 : 0;
      return { value: sample(value, frame, fallback), curve: value };
    }
    if (path.startsWith('clip.')) {
      const key = path.slice('clip.'.length) as 'x' | 'y' | 'width' | 'height' | 'radius' | 'feather';
      const value = layer.clip?.[key];
      const fallback = key === 'x' || key === 'y' || key === 'width' || key === 'height' || key === 'radius' || key === 'feather' ? 0 : 0;
      return { value: sample(value, frame, fallback), curve: value };
    }
    const match = /^(?:filters?)\.(\d+)\.([A-Za-z]+)$/.exec(path);
    if (match) {
      const filter = layer.filters?.[Number(match[1])];
      if (filter) {
        const prop = match[2]!;
        const record = filter as unknown as Record<string, unknown>;
        const value = record[prop];
        if (typeof value === 'number' || (value && typeof value === 'object')) {
          return { value: sample(value as import('@charaforge/scene').Animatable<number>, frame, 0), curve: value as import('@charaforge/scene').Animatable<number> };
        }
      }
    }
    return { value: 0, curve: undefined };
  }

  async addKeyAtPlayhead(target: import('@charaforge/scene').EditTarget): Promise<void> {
    const { value } = this.sampleTarget(target);
    await this.applyUi([{ op: 'addKeyframe', target, t: this.state.frame, v: value, ease: 'smooth' }], {
      label: '添加关键帧',
    });
  }

  // ---------- 启动 ----------

  init(canvas: HTMLCanvasElement): Promise<void> {
    if (this.bootPromise) return this.bootPromise;
    this.canvas = canvas;
    this.bootPromise = this.doInit(canvas);
    return this.bootPromise;
  }

  private async doInit(canvas: HTMLCanvasElement): Promise<void> {
    try {
      this.setState({ phase: 'booting', stage: '检查 Cubism Core', error: null });
      if (typeof (globalThis as { Live2DCubismCore?: unknown }).Live2DCubismCore === 'undefined') {
        throw new Error('没有找到 Live2D Cubism Core：请先运行 pnpm run setup（或 pnpm setup:cubism）');
      }
      const workers = defaultWorkers(this.params);
      this.setState({ stage: `初始化 WebGL 与 Cubism Core（${workers} 个 Worker）` });
      await this.engine.init(canvas, workers);
      this.engine.onStatus = (status) => this.onEngineStatus(status);

      this.setState({ stage: '读取场景' });
      const scenePath = this.state.scenePath;
      const json = await this.fetchSceneJson(scenePath);
      const scene = await this.loadIntoEngine(json, (stage) => this.setState({ stage }));

      const initialFrame = this.params.has('frame') ? Number(this.params.get('frame')) : 0;
      const frame = Number.isFinite(initialFrame) ? clamp(Math.floor(initialFrame), 0, Math.max(0, scene.duration - 1)) : 0;
      this.engine.seek(frame);
      this.engine.start();
      try {
        this.engine.startPrewarm();
      } catch (e) {
        this.notify(`预热失败（不影响编辑）：${errorMessage(e)}`, 'warn');
      }

      this.setState({
        phase: 'ready',
        stage: '就绪',
        error: null,
        scene,
        scenePath,
        savePath: this.state.savePath || scenePath,
        frame,
        playing: false,
        selection: { kind: 'scene' },
        dirty: false,
        canUndo: false,
        canRedo: false,
      });
      this.booted = true;
      this.attachWindowApi();
      (window as unknown as { __player?: PlayerEngine }).__player = this.engine;
      void this.refreshAudioDurations().catch((e) => this.notify(`音频时长读取失败：${errorMessage(e)}`, 'warn'));
      void this.loadCatalog();
      this.notify('编辑器就绪');
    } catch (e) {
      const message = errorMessage(e);
      this.setState({ phase: 'error', stage: '启动失败', error: message });
      this.notify(`启动失败：${message}`, 'error');
    }
  }

  private attachWindowApi(): void {
    if (this.windowApi) return;
    const store = this;
    const api: EditorWindowApi = {
      ready: Promise.resolve(),
      state: () => ({ ...store.state }) as Record<string, unknown>,
      summary: () => store.summary(),
      scene: () => store.sceneSnapshot(),
      apply: (commands, opts) => {
        const unwrapped = !Array.isArray(commands) && 'commands' in commands ? commands.commands : commands;
        return store.apply(unwrapped, opts);
      },
      undo: () => store.undo(),
      redo: () => store.redo(),
      seek: (frame) => store.seek(frame),
      play: () => store.play(),
      pause: () => store.pause(),
      select: (selection) => store.select(selection),
      save: (path) => store.save(path),
      open: (path) => store.open(path),
      exportVideo: async (opts) => {
        const result = await store.exportVideo(opts as Parameters<EditorStoreApi['exportVideo']>[0]);
        const { data, ...rest } = result;
        return { ...rest, bytes: data.byteLength };
      },
      addMotion: (uri, name) => store.addMotion(uri, name),
      addExpression: (uri) => store.addExpression(uri),
      importImage: async (dataUri, opts) => {
        const res = await fetch(dataUri);
        if (!res.ok) throw new Error(`读取 data URI 失败：HTTP ${res.status}`);
        const blob = await res.blob();
        await store.addGeneratedImage(blob, opts?.name ?? 'ai-image', opts);
      },
      store: this,
    };
    this.windowApi = api;
    window.__editor = api;
  }

  destroy(): void {
    this.destroyed = true;
    for (const timer of this.noticeTimers.values()) window.clearTimeout(timer);
    this.noticeTimers.clear();
    this.listeners.clear();
    try {
      this.engine.stop();
      this.engine.clock?.dispose();
      this.engine.runtime?.dispose();
    } catch {
      /* 已释放 */
    }
  }

  // ---------- 引擎状态 ----------

  private onEngineStatus(status: EngineStatus): void {
    const sig = `${status.frame}|${status.playing}|${status.ready}|${status.muted}|${Math.floor(status.waitMs / 250)}`;
    const now = performance.now();
    // 引擎空闲时 rAF 也会持续上报相同状态：签名不变就只按 4Hz 刷新 React。
    // 播放中时间线/播放头需要比 4 Hz 更跟手；暂停时状态不变就跳过重渲染。
    const interval = status.playing ? 100 : 250;
    if (now - this.lastStatusAt < interval && sig === this.lastStatusSignature) return;
    this.lastStatusAt = now;
    this.lastStatusSignature = sig;
    this.setState({ frame: status.frame, playing: status.playing, status: { ...status } });
  }

  // ---------- 播放 / 时间 ----------

  seek(frame: number): void {
    if (!this.state.scene) return;
    const f = clamp(Math.floor(frame), 0, Math.max(0, this.state.scene.duration - 1));
    this.engine.seek(f);
    this.setState({ frame: f });
  }

  step(delta: number): void {
    this.seek(this.state.frame + delta);
  }

  async play(): Promise<void> {
    try {
      await this.engine.play();
      this.setState({ playing: true });
    } catch (e) {
      this.notify(`播放失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  pause(): void {
    this.engine.pause();
    this.setState({ playing: false, frame: this.engine.frame });
  }

  async togglePlay(): Promise<void> {
    if (this.engine.clock?.playing) this.pause();
    else await this.play();
  }

  // ---------- 选择 / 时间线 UI ----------

  select(selection: Selection): void {
    const scene = this.state.scene;
    if (selection.kind === 'layer') {
      if (!scene?.layers.some((layer) => layer.id === selection.id)) {
        this.notify(`找不到图层：${selection.id}`, 'warn');
        return;
      }
    } else if (selection.kind === 'marker') {
      if (!scene?.markers?.some((marker) => marker.id === selection.id)) {
        this.notify(`找不到标记：${selection.id}`, 'warn');
        return;
      }
    } else if (selection.kind === 'audio') {
      if (!scene || selection.index < 0 || selection.index >= (scene.audio?.length ?? 0)) {
        this.notify('找不到音频片段', 'warn');
        return;
      }
    }
    this.setState({ selection });
  }

  setExpanded(key: string, open?: boolean): void {
    const current = !!this.state.expanded[key];
    const next = open === undefined ? !current : open;
    if (next === current) return;
    this.setState({ expanded: { ...this.state.expanded, [key]: next } });
  }

  setSelectedKeys(keys: KeySelection[]): void {
    this.setState({ selectedKeys: keys });
  }

  setTimelineZoom(pxPerFrame: number): void {
    if (!Number.isFinite(pxPerFrame)) return;
    this.setState({ timelineZoom: clamp(pxPerFrame, 0.05, 40) });
  }

  setActiveTool(tool: EditorTool): void {
    this.setState({ activeTool: tool });
  }

  togglePinnedParam(layerId: string, param: string): void {
    const key = `${layerId}::${param}`;
    const pinned = this.state.pinned.includes(key)
      ? this.state.pinned.filter((item) => item !== key)
      : [...this.state.pinned, key];
    this.setState({ pinned });
  }

  isPinned(layerId: string, param: string): boolean {
    return this.state.pinned.includes(`${layerId}::${param}`);
  }

  setShowParams(show: boolean): void {
    this.setState({ showParams: show });
  }

  setDragging(dragging: boolean): void {
    this.setState({ dragging });
  }

  // ---------- 编辑 ----------

  async apply(commands: EditCommand | EditCommand[], opts: ApplyOptions = {}): Promise<EditResult> {
    const scene = this.requireScene();
    const list = Array.isArray(commands) ? commands : [commands];
    let result: EditResult;
    try {
      result = applyEditCommands(scene, list, {
        validate: opts.validate ?? 'final',
        frame: opts.frame ?? this.state.frame,
      });
    } catch (e) {
      this.notify(`编辑失败：${errorMessage(e)}`, 'error');
      throw e;
    }

    if (opts.history !== false && result.changes.length > 0) {
      const label = opts.label ?? commandLabel(list);
      const now = Date.now();
      const last = this.undoStack[this.undoStack.length - 1];
      if (opts.history === 'coalesce' && last && last.label === label && now - last.time <= COALESCE_MS) {
        last.changes.push(...result.changes);
        last.effects = effectSet([...last.effects, ...result.effects]);
        last.time = now;
      } else {
        this.undoStack.push({ changes: result.changes, effects: [...result.effects], label, time: now });
        if (this.undoStack.length > MAX_HISTORY) this.undoStack.shift();
      }
      this.redoStack.length = 0;
    }

    try {
      if (this.engine.scene) await this.engine.applySceneEdits(this.engine.scene, result.effects);
    } catch (e) {
      this.notify(`应用到预览失败：${errorMessage(e)}`, 'error');
      throw e;
    }

    const engineScene = this.engine.scene;
    const patch: Partial<EditorState> = {
      dirty: true,
      canUndo: this.undoStack.length > 0,
      canRedo: this.redoStack.length > 0,
    };
    if (engineScene && engineScene !== this.state.scene) patch.scene = engineScene;
    patch.frame = this.engine.frame;
    this.setState(patch);
    this.syncSelectionValidity();
    if (result.effects.includes('audio') || result.effects.includes('clock')) {
      void this.refreshAudioDurations().catch(() => {});
    }
    return result;
  }

  /** UI 用：失败已经写进 notices，不再让每个按钮都处理 rejected promise。 */
  async applyUi(commands: EditCommand | EditCommand[], opts: ApplyOptions = {}): Promise<EditResult | null> {
    try {
      return await this.apply(commands, opts);
    } catch {
      return null;
    }
  }

  async undo(): Promise<void> {
    const entry = this.undoStack.pop();
    if (!entry) return;
    const scene = this.requireScene();
    try {
      undoEditChanges(scene, entry.changes);
      if (this.engine.scene) await this.engine.applySceneEdits(this.engine.scene, entry.effects);
      this.redoStack.push(entry);
      this.setState({
        dirty: true,
        canUndo: this.undoStack.length > 0,
        canRedo: this.redoStack.length > 0,
        frame: this.engine.frame,
      });
      this.syncSelectionValidity();
      this.notify(`已撤销：${entry.label}`);
    } catch (e) {
      // 撤销失败时把场景恢复到撤销前，保证 UI 与场景一致。
      try {
        applyEditChanges(scene, entry.changes);
        if (this.engine.scene) await this.engine.applySceneEdits(this.engine.scene, entry.effects);
      } catch {
        /* 尽力恢复 */
      }
      this.undoStack.push(entry);
      this.notify(`撤销失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  async redo(): Promise<void> {
    const entry = this.redoStack.pop();
    if (!entry) return;
    const scene = this.requireScene();
    try {
      applyEditChanges(scene, entry.changes);
      if (this.engine.scene) await this.engine.applySceneEdits(this.engine.scene, entry.effects);
      this.undoStack.push(entry);
      this.setState({
        dirty: true,
        canUndo: this.undoStack.length > 0,
        canRedo: this.redoStack.length > 0,
        frame: this.engine.frame,
      });
      this.syncSelectionValidity();
      this.notify(`已重做：${entry.label}`);
    } catch (e) {
      try {
        undoEditChanges(scene, entry.changes);
        if (this.engine.scene) await this.engine.applySceneEdits(this.engine.scene, entry.effects);
      } catch {
        /* 尽力恢复 */
      }
      this.redoStack.push(entry);
      this.notify(`重做失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  private syncSelectionValidity(): void {
    const scene = this.state.scene;
    if (!scene) return;
    const selection = this.state.selection;
    let valid = true;
    if (selection.kind === 'layer') valid = scene.layers.some((layer) => layer.id === selection.id);
    else if (selection.kind === 'marker') valid = (scene.markers ?? []).some((marker) => marker.id === selection.id);
    else if (selection.kind === 'audio') valid = selection.index < (scene.audio?.length ?? 0);
    if (!valid) this.setState({ selection: { kind: 'scene' } });
    const selectedKeys = this.state.selectedKeys.filter((key) => key.t <= scene.duration);
    if (selectedKeys.length !== this.state.selectedKeys.length) this.setState({ selectedKeys });
  }

  setSavePath(path: string): void {
    this.setState({ savePath: path });
  }

  async save(path?: string): Promise<void> {
    const target = (path ?? this.state.savePath ?? this.state.scenePath).trim();
    if (!target) {
      this.notify('保存路径不能为空', 'warn');
      return;
    }
    const scene = this.requireScene();
    let parsed: Scene;
    try {
      parsed = parseScene(scene);
    } catch (e) {
      this.notify(`场景校验失败，未保存：${errorMessage(e)}`, 'error');
      return;
    }

    try {
      const res = await fetch(pageUrl('__charaforge/editor/save-scene'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: target, scene: parsed, backup: true }),
      });
      if (!res.ok) {
        let detail = '';
        try {
          const body = (await res.json()) as { error?: unknown };
          if (typeof body.error === 'string') detail = `：${body.error}`;
        } catch {
          /* 非 JSON 错误体 */
        }
        throw new Error(`HTTP ${res.status}${detail}`);
      }
      // 「另存为」后把当前文档路径也切到新文件，否则刷新会重新读回旧场景。
      this.setState({ savePath: target, scenePath: target, dirty: false });
      try {
        const q = new URLSearchParams(location.search);
        q.set('scene', target);
        history.replaceState(null, '', `${location.pathname}?${q.toString()}${location.hash}`);
      } catch {
        /* 沙箱不允许改地址栏 */
      }
      this.notify(`已保存 ${target}`);
    } catch (e) {
      // 没有本地编辑服务器（例如纯静态预览）时退化成下载。
      try {
        const text = sceneJsonText(parsed);
        const name = safeFileName(basename(target).endsWith('.json') ? basename(target) : `${basename(target)}.json`, 'scene.json');
        downloadBlob(name, new Blob([text], { type: 'application/json' }));
        this.setState({ savePath: target, dirty: false });
        this.notify(`保存接口不可用（${errorMessage(e)}），已下载 ${name}`, 'warn');
      } catch (downloadError) {
        this.notify(`保存失败：${errorMessage(e)}；下载也失败：${errorMessage(downloadError)}`, 'error');
        throw e;
      }
    }
  }

  async open(path: string): Promise<void> {
    const normalized = path.trim().replace(/\\/g, '/').replace(/^\/+/, '');
    if (!normalized) {
      this.notify('场景路径不能为空', 'warn');
      return;
    }
    const previousPath = this.state.scenePath;
    const previousSavePath = this.state.savePath;
    const keepSavePath = previousSavePath !== previousPath;
    // 打开另一个场景从第 0 帧开始；同一个场景的刷新请用 reload()（保留当前帧）。
    if (normalized !== previousPath) this.setState({ frame: 0 });
    await this.loadSceneAt(normalized, { keepSavePath });
  }

  async reload(): Promise<void> {
    await this.loadSceneAt(this.state.scenePath, { keepSavePath: true });
  }

  private async loadSceneAt(path: string, opts: { keepSavePath: boolean }): Promise<void> {
    this.setState({ phase: 'booting', stage: `读取场景 ${path}`, error: null, catalog: null, catalogError: null });
    try {
      this.engine.pause();
      this.engine.stop();
      const json = await this.fetchSceneJson(path);
      const scene = await this.loadIntoEngine(json, (stage) => this.setState({ stage }));
      const frame = clamp(this.state.frame, 0, Math.max(0, scene.duration - 1));
      this.engine.seek(frame);
      this.engine.start();
      try {
        this.engine.startPrewarm();
      } catch (e) {
        this.notify(`预热失败（不影响编辑）：${errorMessage(e)}`, 'warn');
      }
      this.undoStack.length = 0;
      this.redoStack.length = 0;
      this.setState({
        phase: 'ready',
        stage: '就绪',
        error: null,
        scene,
        scenePath: path,
        savePath: opts.keepSavePath ? this.state.savePath : path,
        frame,
        playing: false,
        selection: { kind: 'scene' },
        selectedKeys: [],
        dirty: false,
        canUndo: false,
        canRedo: false,
      });
      try {
        const q = new URLSearchParams(location.search);
        q.set('scene', path);
        q.delete('frame');
        history.replaceState(null, '', `${location.pathname}?${q.toString()}${location.hash}`);
      } catch {
        /* 某些沙箱不允许改地址栏 */
      }
      void this.refreshAudioDurations().catch(() => {});
      void this.loadCatalog();
      this.notify(`已打开 ${path}`);
    } catch (e) {
      this.setState({ phase: 'error', stage: '打开场景失败', error: errorMessage(e) });
      this.notify(`打开场景失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  private async loadIntoEngine(input: unknown, onStage: (stage: string) => void): Promise<Scene> {
    const scene = await this.engine.load(input as object, onStage);
    this.engine.redraw();
    return scene;
  }

  private async fetchSceneJson(path: string): Promise<unknown> {
    const res = await fetch(asciiUrl(pageUrl(path)), { cache: 'no-cache' });
    if (!res.ok) throw new Error(`场景加载失败 ${res.status}：${path}`);
    try {
      return (await res.json()) as unknown;
    } catch (e) {
      throw new Error(`场景 JSON 解析失败：${errorMessage(e)}`);
    }
  }

  // ---------- 素材 / 元素 / 动作 ----------

  async importImageFile(file: File, opts: ImportImageOptions = {}): Promise<void> {
    try {
      await this.importImageBlob(file, opts.name ?? stem(file.name), opts, file.name);
    } catch (e) {
      this.notify(`导入图片失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  async addGeneratedImage(blob: Blob, name: string, opts: ImportImageOptions = {}): Promise<void> {
    try {
      await this.importImageBlob(blob, opts.name ?? name, opts, name);
    } catch (e) {
      this.notify(`添加图片失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  private async importImageBlob(blob: Blob, name: string, opts: ImportImageOptions, uploadName: string): Promise<void> {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    if (!isPng(bytes) && !isWebp(bytes)) throw new Error('只支持 PNG / WebP 图片（按文件头判断）');
    const bitmap = await createImageBitmap(new Blob([bytes], { type: isPng(bytes) ? 'image/png' : 'image/webp' }));
    let width: number;
    let height: number;
    try {
      width = bitmap.width;
      height = bitmap.height;
    } finally {
      bitmap.close();
    }
    if (!(width > 0 && height > 0)) throw new Error('图片尺寸无效');
    if (width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE) throw new Error(`图片单边不能超过 ${MAX_IMAGE_SIDE} 像素（${width}×${height}）`);
    if (width * height > MAX_IMAGE_PIXELS) throw new Error(`图片总像素不能超过 ${MAX_IMAGE_PIXELS}（${width}×${height}）`);

    const mime = isPng(bytes) ? 'image/png' : 'image/webp';
    const uploaded = await this.uploadBlob(new Blob([bytes], { type: mime }), 'image', uploadName);
    let uri: string;
    let sha256: string | undefined;
    if (uploaded) {
      uri = uploaded.uri;
      sha256 = uploaded.sha256;
    } else {
      uri = `data:${mime};base64,${bytesToBase64(bytes)}`;
      this.notify('本地编辑接口不可用，图片将以 data URI 嵌入场景', 'warn');
    }

    const scene = this.requireScene();
    const used = usedIds(scene);
    // 资源 ID 用上传文件名（ASCII），图层显示名可以保留中文。
    const id = uniqueId(sanitizeId(stem(uploadName), 'image'), used);
    const asset: ImageAsset = { id, uri, width, height, ...(sha256 ? { sha256 } : {}) };
    const layer = this.buildImageLayer(asset, { ...opts, name: opts.name ?? name });
    const index = opts.asBackground ? 0 : clamp(opts.index ?? scene.layers.length, 0, scene.layers.length);
    const result = await this.applyUi([{ op: 'addImageLayer', asset, layer, index }], { label: `导入图片：${name}` });
    if (result) {
      this.select({ kind: 'layer', id: layer.id });
      this.notify(`已添加图片图层 ${layer.name ?? layer.id}（${width}×${height}）`);
    }
  }

  private buildImageLayer(asset: ImageAsset, opts: ImportImageOptions): ImageLayer {
    const scene = this.requireScene();
    const W = scene.composition.width;
    const H = scene.composition.height;
    const used = usedIds(scene);
    const id = uniqueId(sanitizeId(asset.id, 'image'), used);
    const name = opts.name ?? (opts.asBackground ? '背景' : asset.id);
    const anchorX = asset.width / 2;
    const anchorY = asset.height / 2;
    if (opts.asBackground) {
      return {
        id,
        name,
        kind: 'image',
        image: asset.id,
        in: 0,
        out: scene.duration,
        space: 'screen',
        transform: { x: W / 2, y: H / 2, scale: 1, anchorX, anchorY },
        opacity: 1,
        provenance: { source: 'import', tool: 'charaforge-editor' },
      };
    }

    const target = opts.attachToSelected ? this.selectedModelLayer() ?? this.firstModelLayer() : null;
    if (target) {
      return {
        id,
        name,
        kind: 'image',
        image: asset.id,
        in: 0,
        out: scene.duration,
        transform: { x: 0, y: 0, scale: 1, anchorX, anchorY },
        attach: { layer: target.id, anchor: [0.5, 0.2], offset: [0, 0], rotateWith: true, scaleWith: true },
        opacity: 1,
        provenance: { source: 'import', tool: 'charaforge-editor' },
      };
    }

    const scale = Math.min(W / asset.width, H / asset.height) * 0.5;
    return {
      id,
      name,
      kind: 'image',
      image: asset.id,
      in: 0,
      out: scene.duration,
      space: 'screen',
      transform: { x: W / 2, y: H / 2, scale, anchorX, anchorY },
      opacity: 1,
      provenance: { source: 'import', tool: 'charaforge-editor' },
    };
  }

  private firstModelLayer(): ModelLayer | null {
    return this.state.scene?.layers.find((layer): layer is ModelLayer => layer.kind === 'model') ?? null;
  }

  /** 往场景里加一个已存在的图片资源（素材面板“添加为图层”）。 */
  async addImageAssetLayer(assetId: string, opts: ImportImageOptions = {}): Promise<void> {
    const asset = this.imageAsset(assetId);
    if (!asset) {
      this.notify(`找不到图片资源：${assetId}`, 'warn');
      return;
    }
    const scene = this.requireScene();
    // addImageLayer 会同时插入资源；复用已有资源时复制一份同 uri 的新资源，避免 ID 冲突。
    const used = usedIds(scene);
    const copyId = uniqueId(sanitizeId(`${asset.id}-copy`, 'image'), used);
    const copy: ImageAsset = { ...asset, id: copyId };
    const layer = this.buildImageLayer(copy, { ...opts, name: opts.name ?? asset.id });
    const index = opts.asBackground ? 0 : clamp(opts.index ?? scene.layers.length, 0, scene.layers.length);
    const result = await this.applyUi([{ op: 'addImageLayer', asset: copy, layer, index }], { label: `添加图片：${asset.id}` });
    if (result) this.select({ kind: 'layer', id: layer.id });
  }

  async importAudioFile(file: File, startAt?: number): Promise<void> {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const uploaded = await this.uploadBlob(new Blob([bytes], { type: file.type || 'application/octet-stream' }), 'audio', file.name);
      const uri = uploaded?.uri ?? `data:${file.type || 'audio/mpeg'};base64,${bytesToBase64(bytes)}`;
      if (!uploaded) this.notify('本地编辑接口不可用，音频将以 data URI 嵌入场景', 'warn');
      const start = startAt ?? this.state.frame;
      const result = await this.applyUi([{ op: 'addAudio', uri, start }], { label: `导入音频：${file.name}` });
      if (result) this.notify(`已添加音频 ${stem(file.name)}（起始帧 ${start}）`);
    } catch (e) {
      this.notify(`导入音频失败：${errorMessage(e)}`, 'error');
      throw e;
    }
  }

  async addAudioAsset(assetId: string, startAt?: number): Promise<void> {
    const asset = this.state.scene?.assets.audio?.find((item) => item.id === assetId);
    if (!asset) {
      this.notify(`找不到音频资源：${assetId}`, 'warn');
      return;
    }
    const start = startAt ?? this.state.frame;
    const result = await this.applyUi([{ op: 'addAudio', uri: asset.uri, start }], { label: `添加音频：${asset.id}` });
    if (result) this.notify(`已添加音频片段（起始帧 ${start}）`);
  }

  private async uploadBlob(blob: Blob, kind: 'image' | 'audio', name: string): Promise<UploadedAsset | null> {
    try {
      const url = new URL(pageUrl('__charaforge/editor/upload'));
      url.searchParams.set('kind', kind);
      url.searchParams.set('name', name);
      const res = await fetch(url.href, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: blob,
      });
      if (!res.ok) return null;
      const body = (await res.json()) as { ok?: unknown; asset?: { uri?: unknown; sha256?: unknown } };
      if (body.ok !== true || !body.asset || typeof body.asset.uri !== 'string') return null;
      return {
        uri: body.asset.uri,
        ...(typeof body.asset.sha256 === 'string' ? { sha256: body.asset.sha256 } : {}),
      };
    } catch {
      return null;
    }
  }

  async addMotion(uri: string, name?: string): Promise<void> {
    const layer = this.selectedModelLayer() ?? this.firstModelLayer();
    if (!layer) {
      this.notify('场景里没有模型图层，无法添加动作', 'warn');
      return;
    }
    if (this.state.selection.kind !== 'layer' || this.state.selection.id !== layer.id) {
      this.select({ kind: 'layer', id: layer.id });
    }
    const start = this.state.frame;
    const result = await this.applyUi(
      [{ op: 'addMotion', layer: layer.id, uri, start, fadeIn: 0, loop: false }],
      { label: `添加动作：${name ?? uri}` },
    );
    if (result) this.notify(`已添加动作 ${name ?? basename(uri)}（起始帧 ${start}）`);
  }

  async addExpression(uri: string): Promise<void> {
    const layer = this.selectedModelLayer() ?? this.firstModelLayer();
    if (!layer) {
      this.notify('场景里没有模型图层，无法添加表情', 'warn');
      return;
    }
    if (this.state.selection.kind !== 'layer' || this.state.selection.id !== layer.id) {
      this.select({ kind: 'layer', id: layer.id });
    }
    const start = this.state.frame;
    const result = await this.applyUi(
      [{ op: 'addExpression', layer: layer.id, uri, start, fadeIn: 0 }],
      { label: `添加表情：${basename(uri)}` },
    );
    if (result) this.notify(`已添加表情 ${basename(uri)}（起始帧 ${start}）`);
  }

  async addAnimationPreset(preset: AnimationPreset, layerId?: string, durationFrames = 90): Promise<void> {
    const layer =
      (layerId ? this.state.scene?.layers.find((item) => item.id === layerId) : null) ??
      this.selectedLayer() ??
      this.firstModelLayer() ??
      this.state.scene?.layers[0];
    if (!layer) {
      this.notify('没有可应用动画的图层', 'warn');
      return;
    }
    const start = this.state.frame;
    const duration = Math.max(1, Math.round(durationFrames));
    const end = Math.max(start + 1, Math.min(layer.out, start + duration));
    const result = await this.applyUi([{ op: 'animate', id: layer.id, preset, start, end }], {
      label: `一键动画：${preset}`,
    });
    if (result) this.notify(`已对 ${layer.name ?? layer.id} 应用动画 ${preset}（${start}–${end}）`);
  }

  async setBackground(color: string | null): Promise<void> {
    await this.applyUi([{ op: 'setScene', patch: { background: color } }], { label: '设置背景' });
  }

  async addModelLayer(assetId?: string): Promise<void> {
    const scene = this.requireScene();
    const asset = (assetId ? scene.assets.models.find((item) => item.id === assetId) : scene.assets.models[0]) ?? scene.assets.models[0];
    if (!asset) {
      this.notify('场景没有模型资源', 'warn');
      return;
    }
    let canvas: CanvasInfo | null = null;
    try {
      canvas = await this.engine.runtime.loadModel(asset.uri).then((resource) => resource.canvas);
    } catch (e) {
      this.notify(`模型信息读取失败：${errorMessage(e)}`, 'warn');
    }
    const W = scene.composition.width;
    const H = scene.composition.height;
    const used = usedIds(scene);
    const id = uniqueId(sanitizeId(stem(basename(asset.uri)), 'model'), used);
    const scale = canvas ? Math.min(W / canvas.width, H / canvas.height) * 0.8 : 0.3;
    const layer: ModelLayer = {
      id,
      kind: 'model',
      model: asset.id,
      in: 0,
      out: scene.duration,
      transform: {
        x: W / 2,
        y: H / 2,
        scale,
        ...(canvas ? { anchorX: canvas.width / 2, anchorY: canvas.height / 2 } : {}),
      },
    };
    const result = await this.applyUi([{ op: 'addModelLayer', layer, index: scene.layers.length }], {
      label: `新建模型实例：${asset.id}`,
    });
    if (result) {
      this.select({ kind: 'layer', id: layer.id });
      this.notify(`已新建模型图层 ${id}`);
    }
  }

  async duplicateLayer(id: string): Promise<void> {
    const scene = this.requireScene();
    if (!scene.layers.some((layer) => layer.id === id)) return;
    const used = usedIds(scene);
    const newId = uniqueId(sanitizeId(`${id}-copy`, 'layer-copy'), used);
    const result = await this.applyUi([{ op: 'duplicateLayer', id, newId }], { label: `复制图层：${id}` });
    if (result) {
      this.select({ kind: 'layer', id: newId });
      this.notify(`已复制为 ${newId}`);
    }
  }

  async loadCatalog(): Promise<void> {
    if (this.catalogPromise) return this.catalogPromise;
    this.catalogPromise = this.doLoadCatalog().finally(() => {
      this.catalogPromise = null;
    });
    return this.catalogPromise;
  }

  private async doLoadCatalog(): Promise<void> {
    const scene = this.state.scene;
    if (!scene) return;
    const modelUri = scene.assets.models[0]?.uri ?? PREVIEW_MODEL_URI;
    const warnings: string[] = [];
    try {
      let model3: unknown = null;
      try {
        model3 = await this.fetchJson(this.assetUrl(modelUri));
      } catch (e) {
        warnings.push(`model3.json 读取失败，只依赖目录列表：${errorMessage(e)}`);
      }

      const declaredMotions = declaredMotionFiles(model3);
      const declaredExpressions = declaredExpressionFiles(model3);
      const motionSource: AssetSource = declaredMotions.length ? 'declared' : 'directory';
      const expressionSource: AssetSource = declaredExpressions.length ? 'declared' : 'directory';

      const modelDir = modelUri.includes('/') ? modelUri.slice(0, modelUri.lastIndexOf('/')) : '';
      const modelAssetDir = `assets/${modelDir}`.replace(/\/+/g, '/');
      const animationDir = modelAssetDir.startsWith('assets/models/') ? `${modelAssetDir}/Animation` : ANIMATION_DIR;
      const exp3Dir = modelAssetDir.startsWith('assets/models/') ? `${modelAssetDir}/Exp3` : EXP3_DIR;

      const motionRaw: RawAsset[] = declaredMotions.length
        ? dedupeRaw(declaredMotions.map((file) => rawDeclared(modelUri, file)))
        : await this.listRawAssets(animationDir, isMotionFile, warnings, '动作');
      const expressionRaw: RawAsset[] = declaredExpressions.length
        ? dedupeRaw(declaredExpressions.map((item) => rawDeclared(modelUri, item.file, item.name)))
        : await this.listRawAssets(exp3Dir, isExpressionFile, warnings, '表情');

      const motions = (
        await Promise.all(
          motionRaw.map(async (raw): Promise<MotionAssetInfo | null> => {
            try {
              const meta = parseMotionMeta(await this.fetchJson(this.assetUrl(raw.uri)), raw.file);
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
              return null;
            }
          }),
        )
      ).filter((item): item is MotionAssetInfo => item !== null);

      const expressions = (
        await Promise.all(
          expressionRaw.map(async (raw): Promise<ExpressionAssetInfo | null> => {
            try {
              const info = parseExpressionInfo(await this.fetchJson(this.assetUrl(raw.uri)), raw.file);
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
              return null;
            }
          }),
        )
      ).filter((item): item is ExpressionAssetInfo => item !== null);

      sortAssets(motions);
      sortAssets(expressions);
      const catalog: Catalog = {
        modelUri,
        motions,
        expressions,
        sources: { motions: motionSource, expressions: expressionSource },
        warnings,
      };
      if (this.state.scene?.assets.models[0]?.uri !== modelUri) return;
      this.setState({
        catalog,
        catalogError: !motions.length && !expressions.length && warnings.length ? warnings.join('\n') : null,
      });
    } catch (e) {
      this.setState({ catalogError: errorMessage(e) });
      this.notify(`动作 / 表情目录读取失败：${errorMessage(e)}`, 'warn');
    }
  }

  private assetUrl(uri: string): string {
    return new URL(uri.replace(/^\/+/, ''), pageUrl('assets/')).href;
  }

  private async fetchJson(url: string): Promise<unknown> {
    const res = await fetch(asciiUrl(url));
    if (!res.ok) throw new Error(`加载失败 ${res.status}：${url}`);
    return (await res.json()) as unknown;
  }

  private async listRawAssets(
    dir: string,
    accept: (name: string) => boolean,
    warnings: string[],
    label: string,
  ): Promise<RawAsset[]> {
    try {
      const url = new URL(pageUrl('__charaforge/list-assets'));
      url.searchParams.set('dir', dir);
      const res = await fetch(asciiUrl(url.href));
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
      const files = body.files as AssetFileInfo[];
      return files.filter((file) => accept(file.name)).map((file) => rawDirectory(dir, file));
    } catch (e) {
      warnings.push(`${label}目录读取失败：${errorMessage(e)}`);
      return [];
    }
  }

  async refreshAudioDurations(): Promise<void> {
    const scene = this.state.scene;
    const assets = scene?.assets.audio ?? [];
    if (!assets.length) {
      this.setState({ audioDurations: {} });
      return;
    }
    const out: Record<string, number> = {};
    for (const asset of assets) {
      const cached = this.audioCache.get(asset.uri);
      if (cached !== undefined) out[asset.id] = cached;
    }
    const missing = assets.filter((asset) => out[asset.id] === undefined);
    if (missing.length) {
      const ctx = new AudioContext();
      try {
        await Promise.all(
          missing.map(async (asset) => {
            try {
              const res = await fetch(asciiUrl(this.resolveAsset(asset.uri)));
              if (!res.ok) throw new Error(`HTTP ${res.status}`);
              const buffer = await ctx.decodeAudioData(await res.arrayBuffer());
              this.audioCache.set(asset.uri, buffer.duration);
              out[asset.id] = buffer.duration;
            } catch (e) {
              this.notify(`音频 ${asset.id} 解码失败：${errorMessage(e)}`, 'warn');
            }
          }),
        );
      } finally {
        await ctx.close().catch(() => {});
      }
    }
    if (this.state.scene !== scene) return;
    this.setState({ audioDurations: out });
  }

  // ---------- 导出 ----------

  async exportVideo(opts: Parameters<EditorStoreApi['exportVideo']>[0] = {}): Promise<ExportVideoResult> {
    if (this.state.exporting) throw new Error('已有导出任务正在运行');
    const scene = this.requireScene();
    const canvas = this.canvas;
    if (!canvas) throw new Error('画布尚未初始化');
    const from = clamp(Math.floor(opts.from ?? 0), 0, scene.duration - 1);
    const to = clamp(Math.floor(opts.to ?? scene.duration), from + 1, scene.duration);
    if (to <= from) throw new Error(`导出范围无效：[${from}, ${to})`);

    const wasPlaying = this.engine.clock?.playing ?? false;
    const restoreFrame = this.engine.frame;
    this.engine.pause();
    this.engine.stop();
    const startedAt = Date.now();
    const progress: ExportProgress = { phase: 'audio', frame: 0, total: to - from, startedAt };
    this.setState({ exporting: progress });
    this.lastExportProgressAt = 0;

    try {
      const result = await exportSceneVideo({
        scene,
        canvas,
        renderFrame: async (frame) => {
          await this.engine.runtime.renderFrameAsync(frame);
          this.engine.runtime.present();
        },
        resolveAsset: (uri) => this.engine.runtime.resolveAsset(uri),
        from,
        to,
        format: opts.format ?? 'auto',
        quality: opts.quality ?? 'high',
        background: opts.background ?? scene.composition.background ?? '#000',
        onProgress: (p) => {
          const now = performance.now();
          if (p.phase === 'video' && now - this.lastExportProgressAt < 120) return;
          this.lastExportProgressAt = now;
          this.setState({ exporting: { ...p, startedAt } });
        },
      });

      const title = opts.filename ? stem(basename(opts.filename)) : `${scene.meta.title || 'charaforge'}-${from}-${to}`;
      const rawName = opts.filename ? basename(opts.filename) : `${title}.${result.extension}`;
      const normalizedName = rawName.toLowerCase().endsWith(`.${result.extension}`)
        ? rawName
        : `${stem(rawName)}.${result.extension}`;
      const filename = safeFileName(normalizedName, `charaforge.${result.extension}`);
      downloadBlob(filename, new Blob([result.data as unknown as BlobPart], { type: result.mimeType }));
      this.notify(
        `已导出 ${filename}（${result.width}×${result.height}，${result.frames} 帧${result.hasAudio ? '，含音频' : '，无音频'}）`,
      );
      return result;
    } catch (e) {
      if (isAbort(e)) this.notify('导出已取消', 'warn');
      else this.notify(`导出失败：${errorMessage(e)}`, 'error');
      throw e;
    } finally {
      this.setState({ exporting: null, frame: restoreFrame });
      this.engine.start();
      this.engine.seek(restoreFrame);
      if (wasPlaying) {
        void this.engine.play().catch((e) => this.notify(`恢复播放失败：${errorMessage(e)}`, 'warn'));
      }
    }
  }

  // ---------- 快照 ----------

  sceneSnapshot(): Scene {
    return JSON.parse(JSON.stringify(this.requireScene())) as Scene;
  }

  summary(): Record<string, unknown> {
    const scene = this.state.scene;
    return {
      scenePath: this.state.scenePath,
      title: scene?.meta.title ?? '',
      fps: scene?.timebase.fps ?? 0,
      duration: scene?.duration ?? 0,
      composition: scene?.composition ? { ...scene.composition } : null,
      frame: this.state.frame,
      playing: this.state.playing,
      layers: scene?.layers.length ?? 0,
      markers: scene?.markers?.length ?? 0,
      audio: scene?.audio?.length ?? 0,
      dirty: this.state.dirty,
      canUndo: this.state.canUndo,
      canRedo: this.state.canRedo,
      selection: this.state.selection,
    };
  }

  // ---------- 通知 ----------

  notify(text: string, kind: EditorNotice['kind'] = 'info'): void {
    const id = this.nextNoticeId++;
    const notice: EditorNotice = { id, text, kind };
    let notices = [...this.state.notices, notice];
    if (notices.length > NOTICE_LIMIT) notices = notices.slice(notices.length - NOTICE_LIMIT);
    this.setState({ notices });
    const timer = window.setTimeout(() => {
      this.noticeTimers.delete(id);
      this.setState({ notices: this.state.notices.filter((item) => item.id !== id) });
    }, NOTICE_TTL);
    this.noticeTimers.set(id, timer);
  }

  dismissNotice(id: number): void {
    const timer = this.noticeTimers.get(id);
    if (timer !== undefined) window.clearTimeout(timer);
    this.noticeTimers.delete(id);
    this.setState({ notices: this.state.notices.filter((item) => item.id !== id) });
  }
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

function rawDirectory(dir: string, file: AssetFileInfo): RawAsset {
  return {
    file: file.name,
    uri: assetUriFromDir(dir, file.name),
    dir,
    size: file.size,
    mtime: file.mtime,
    source: 'directory',
  };
}

function rawDeclared(modelUri: string, file: string, declaredName?: string): RawAsset {
  const uri = declaredAssetUri(modelUri, file);
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
  return assets.filter((asset) => {
    if (seen.has(asset.uri)) return false;
    seen.add(asset.uri);
    return true;
  });
}

export function createEditorStore(opts?: EditorStoreOptions): EditorStore {
  return new EditorStore(opts);
}
