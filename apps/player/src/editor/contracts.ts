/*
 * 编辑器 UI 的共享契约：store、时间线、属性面板和 window.__editor AI 接口都只依赖这里。
 * 具体实现见 store.ts；时间线可以只依赖 EditorStoreApi 的一个子集。
 */
import type {
  EditCommand,
  EditResult,
  EasePreset,
  Scene,
} from '@charaforge/scene';
import type { EngineStatus } from '../engine.ts';
import type { Catalog } from '../motionCatalog.ts';
import type { ExportVideoResult } from './export.ts';

export type EditorTool = 'layers' | 'assets' | 'motions' | 'elements';

export type Selection =
  | { kind: 'scene' }
  | { kind: 'camera' }
  | { kind: 'layer'; id: string }
  | { kind: 'audio'; index: number }
  | { kind: 'marker'; id: string };

export interface KeySelection {
  /** 稳定的目标键，用于 React key：例如 layer:s04.main:transform.x / camera:x / param:s04.main:ParamAngleX */
  key: string;
  /** 目标属性路径（与 EditTarget 一致） */
  target: import('@charaforge/scene').EditTarget;
  t: number;
}

export interface ExportProgress {
  phase: 'audio' | 'video' | 'finalizing';
  frame: number;
  total: number;
  startedAt: number;
}

export interface EditorNotice {
  id: number;
  text: string;
  kind: 'info' | 'warn' | 'error';
}

export interface EditorState {
  phase: 'booting' | 'ready' | 'error';
  stage: string;
  error: string | null;
  /** 当前场景对象（原地编辑；不要整体替换引用，除非来自 open/reload） */
  scene: Scene | null;
  /** 当前场景的仓库相对路径 */
  scenePath: string;
  /** 保存目标（可能与 scenePath 不同） */
  savePath: string;
  frame: number;
  playing: boolean;
  status: EngineStatus | null;
  selection: Selection;
  /** 时间线展开的图层 / 分组：layer:<id> / camera / audio / params:<layerId> */
  expanded: Record<string, boolean>;
  /** 时间线中选中的关键帧（可多选） */
  selectedKeys: KeySelection[];
  /** 时间线中固定的参数轨：`<layerId>::<paramId>` */
  pinned: string[];
  dirty: boolean;
  canUndo: boolean;
  canRedo: boolean;
  notices: EditorNotice[];
  exporting: ExportProgress | null;
  /** 动作 / 表情素材目录（异步加载，可能为 null） */
  catalog: Catalog | null;
  catalogError: string | null;
  /** audio asset id → 时长（秒）；时间线画音频条用 */
  audioDurations: Record<string, number>;
  activeTool: EditorTool;
  /** 时间线缩放：每帧多少像素 */
  timelineZoom: number;
  /** 时间线是否隐藏参数轨 */
  showParams: boolean;
  /** 画布上选中的图层包围盒（用于直接拖动） */
  dragging: boolean;
}

export interface ApplyOptions {
  /** true（缺省）= 新历史条目；false = 不记历史；'coalesce' = 与同 label 的上一条合并 */
  history?: boolean | 'coalesce';
  label?: string;
  validate?: 'each' | 'final' | 'none';
  frame?: number;
}

export interface ImportImageOptions {
  /** 插入到 layers 的位置；缺省追加到最上层 */
  index?: number;
  /** 生成元素时给一个可读名字 */
  name?: string;
  /** 直接作为背景铺满合成画面（screen space） */
  asBackground?: boolean;
  /** 绑定到当前选中的模型图层 */
  attachToSelected?: boolean;
}

/**
 * 编辑器 store 的稳定接口。时间线只需要 getState/subscribe/seek/select/apply/undo/redo/
 * setExpanded/setSelectedKeys/setTimelineZoom/togglePinnedParam。
 */
export interface EditorStoreApi {
  getState(): EditorState;
  subscribe(fn: () => void): () => void;
  init(canvas: HTMLCanvasElement): Promise<void>;
  destroy(): void;

  // 播放 / 时间
  seek(frame: number): void;
  step(delta: number): void;
  play(): Promise<void>;
  pause(): void;
  togglePlay(): Promise<void>;

  // 选择 / 时间线 UI
  select(selection: Selection): void;
  setExpanded(key: string, open?: boolean): void;
  setSelectedKeys(keys: KeySelection[]): void;
  setTimelineZoom(pxPerFrame: number): void;
  setActiveTool(tool: EditorTool): void;
  togglePinnedParam(layerId: string, param: string): void;
  isPinned(layerId: string, param: string): boolean;
  setShowParams(show: boolean): void;

  // 编辑
  apply(commands: EditCommand | EditCommand[], opts?: ApplyOptions): Promise<EditResult>;
  undo(): Promise<void>;
  redo(): Promise<void>;
  setSavePath(path: string): void;
  save(path?: string): Promise<void>;
  open(path: string): Promise<void>;
  reload(): Promise<void>;

  // 素材 / 元素 / 动作
  importImageFile(file: File, opts?: ImportImageOptions): Promise<void>;
  importAudioFile(file: File, startAt?: number): Promise<void>;
  addGeneratedImage(blob: Blob, name: string, opts?: ImportImageOptions): Promise<void>;
  addMotion(uri: string, name?: string): Promise<void>;
  addExpression(uri: string): Promise<void>;
  addAnimationPreset(preset: import('@charaforge/scene').AnimationPreset, layerId?: string): Promise<void>;
  setBackground(color: string | null): Promise<void>;
  loadCatalog(): Promise<void>;
  refreshAudioDurations(): Promise<void>;
  notify(text: string, kind?: EditorNotice['kind']): void;

  // 导出
  exportVideo(opts?: {
    from?: number;
    to?: number;
    format?: 'mp4' | 'webm' | 'auto';
    quality?: 'high' | 'medium' | 'low';
    background?: string;
    filename?: string;
  }): Promise<ExportVideoResult>;

  // 给 AI / 测试用的只读快照
  sceneSnapshot(): Scene;
  summary(): Record<string, unknown>;
}

/** window.__editor 的稳定接口（Playwright / AI 在浏览器里调用）。 */
export interface EditorWindowApi {
  ready: Promise<void>;
  state(): Record<string, unknown>;
  summary(): Record<string, unknown>;
  scene(): Scene;
  apply(commands: EditCommand | EditCommand[] | { commands: EditCommand[] }, opts?: ApplyOptions): Promise<EditResult>;
  undo(): Promise<void>;
  redo(): Promise<void>;
  seek(frame: number): void;
  play(): Promise<void>;
  pause(): void;
  select(selection: Selection): void;
  save(path?: string): Promise<void>;
  open(path: string): Promise<void>;
  exportVideo(opts?: Record<string, unknown>): Promise<Record<string, unknown>>;
  addMotion(uri: string, name?: string): Promise<void>;
  addExpression(uri: string): Promise<void>;
  importImage(dataUri: string, opts?: ImportImageOptions): Promise<void>;
  readonly store: EditorStoreApi | null;
}

declare global {
  interface Window {
    __editor?: EditorWindowApi;
  }
}

export type { EditCommand, EditResult, EasePreset, Scene, Catalog, ExportVideoResult };
