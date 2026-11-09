/*
 * 右侧属性检查器。
 *
 * 所有字段都是受控视图：值来自 store.state.scene（原地修改），提交时走 store.apply 的编辑命令。
 * AnimField 同时覆盖常量属性与曲线属性：曲线字段编辑时写当前帧关键帧，常量字段（有 onConstant）写回常量。
 */
import {
  EASE_PRESETS,
  sample,
  type Animatable,
  type AnimationPreset,
  type AudioClip,
  type Ease,
  type EasePreset,
  type EditTarget,
  type Filter,
  type Layer,
  type Marker,
  type ModelLayer,
} from '@charaforge/scene';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { EditorState } from './contracts.ts';
import type { EditorStore } from './store.ts';

const EASE_OPTIONS: { value: EasePreset; label: string }[] = [
  { value: 'linear', label: '线性' },
  { value: 'smooth', label: '平滑' },
  { value: 'easeIn', label: '缓入' },
  { value: 'easeOut', label: '缓出' },
  { value: 'easeInOut', label: '缓入缓出' },
  { value: 'step', label: '步进' },
  { value: 'hold', label: '保持' },
];

const PRESET_LABELS: Record<AnimationPreset, string> = {
  fadeIn: '淡入',
  fadeOut: '淡出',
  float: '漂浮',
  pulse: '脉冲',
  spin: '旋转',
  bounce: '弹跳',
  blink: '闪烁',
  slideIn: '滑入',
  shake: '抖动',
  zoomIn: '放大',
};

const PRESET_VALUES = Object.keys(PRESET_LABELS) as AnimationPreset[];

export function formatNumber(value: number, integer = false): string {
  if (!Number.isFinite(value)) return '0';
  if (integer) return String(Math.round(value));
  const rounded = Math.abs(value) < 1e-6 ? 0 : Math.round(value * 10000) / 10000;
  return String(rounded);
}

export function easePresetOf(ease: Ease | undefined): EasePreset {
  if (!ease) return 'linear';
  if (typeof ease === 'string') return ease;
  for (const [name, preset] of Object.entries(EASE_PRESETS) as [EasePreset, Ease][]) {
    if (
      Array.isArray(preset) &&
      Array.isArray(ease) &&
      preset.length === 4 &&
      preset.every((v, i) => Math.abs(v - ease[i]!) < 1e-6)
    ) {
      return name;
    }
  }
  return 'linear';
}

function keyTimes(curve: Animatable<number> | undefined): number[] | null {
  if (!curve || typeof curve === 'number') return null;
  if ('keys' in curve) return curve.keys.map((key) => key.t);
  return null;
}

function keyAt(curve: Animatable<number> | undefined, frame: number): boolean {
  if (!curve || typeof curve === 'number') return false;
  if ('keys' in curve) return curve.keys.some((key) => Math.abs(key.t - frame) <= 1e-6);
  const baked = curve.baked;
  return frame >= baked.start && frame < baked.start + baked.values.length;
}

function easeAt(curve: Animatable<number> | undefined, frame: number): Ease | undefined {
  if (!curve || typeof curve === 'number' || !('keys' in curve)) return undefined;
  return curve.keys.find((key) => Math.abs(key.t - frame) <= 1e-6)?.ease;
}

interface CommitNumberProps {
  value: number;
  onCommit: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  disabled?: boolean;
  title?: string;
  className?: string;
  /** false = 只在失焦 / 回车时提交（适合会弹确认框或重建结构的字段）。 */
  live?: boolean;
}

export function CommitNumber({ value, onCommit, min, max, step, integer, disabled, title, className, live = true }: CommitNumberProps) {
  const [text, setText] = useState(formatNumber(value, integer));
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (document.activeElement !== ref.current) setText(formatNumber(value, integer));
  }, [value, integer]);

  const commit = (raw: string) => {
    if (raw.trim() === '') return;
    let n = Number(raw);
    if (!Number.isFinite(n)) return;
    if (integer) n = Math.round(n);
    if (min !== undefined) n = Math.max(min, n);
    if (max !== undefined) n = Math.min(max, n);
    onCommit(n);
  };

  return (
    <input
      ref={ref}
      type="number"
      className={className ?? 'num-input'}
      value={text}
      min={min}
      max={max}
      step={step ?? (integer ? 1 : 0.01)}
      disabled={disabled}
      title={title}
      onChange={(e) => {
        setText(e.target.value);
        if (live) commit(e.target.value);
      }}
      onBlur={(e) => {
        if (!live) commit(e.target.value);
        setText(formatNumber(value, integer));
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur();
      }}
    />
  );
}

interface CommitTextProps {
  value: string;
  onCommit: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  multiline?: boolean;
  className?: string;
}

function CommitText({ value, onCommit, placeholder, disabled, multiline, className }: CommitTextProps) {
  const [text, setText] = useState(value);
  const ref = useRef<HTMLInputElement & HTMLTextAreaElement>(null);
  useEffect(() => {
    if (document.activeElement !== ref.current) setText(value);
  }, [value]);
  const commit = () => {
    if (text !== value) onCommit(text);
  };
  if (multiline) {
    return (
      <textarea
        ref={ref}
        className={className ?? 'text-input textarea'}
        value={text}
        placeholder={placeholder}
        disabled={disabled}
        rows={2}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
      />
    );
  }
  return (
    <input
      ref={ref}
      className={className ?? 'text-input'}
      value={text}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur();
      }}
    />
  );
}

interface AnimFieldProps {
  label: string;
  target: EditTarget;
  value: Animatable<number> | undefined;
  fallback?: number;
  min?: number;
  max?: number;
  step?: number;
  range?: boolean;
  disabled?: boolean;
  integer?: boolean;
  ease: EasePreset;
  onEaseChange: (ease: EasePreset) => void;
  store: EditorStore;
  onConstant?: (value: number) => void;
  title?: string;
  compact?: boolean;
}

export function AnimField(props: AnimFieldProps) {
  const { label, target, value, fallback = 0, min, max, step, range, disabled, integer, ease, onEaseChange, store, onConstant, title } = props;
  const frame = store.getState().frame;
  const sampled = sample(value, frame, fallback);
  const curve = value !== undefined && typeof value === 'object';
  const hasKey = keyAt(value, frame);
  const times = keyTimes(value);
  const baked = curve && value !== undefined && typeof value === 'object' && 'baked' in value;
  const currentEase = easeAt(value, frame);
  const displayEase = hasKey ? easePresetOf(currentEase) : ease;

  const apply = (commands: Parameters<EditorStore['apply']>[0], labelText: string, coalesce = false) => {
    void store.applyUi(commands, { label: labelText, ...(coalesce ? { history: 'coalesce' as const } : {}) });
  };

  const writeValue = (next: number) => {
    if (curve || target.kind === 'param') {
      apply([{ op: 'addKeyframe', target, t: frame, v: next, ease }], `设置关键帧：${label}`, true);
    } else if (onConstant) {
      onConstant(next);
    } else {
      apply([{ op: 'addKeyframe', target, t: frame, v: next, ease }], `设置关键帧：${label}`, true);
    }
  };

  const toggleKey = () => {
    if (hasKey) {
      if (target.kind === 'param' && times && times.length <= 1) {
        apply([{ op: 'setKeyframes', target, keys: [] }], `删除参数轨：${label}`);
      } else {
        apply([{ op: 'removeKeyframe', target, t: frame }], `删除关键帧：${label}`);
      }
      return;
    }
    apply([{ op: 'addKeyframe', target, t: frame, v: sampled, ease }], `添加关键帧：${label}`);
  };

  const jumpKey = (direction: -1 | 1) => {
    if (!times || times.length === 0) return;
    const next =
      direction > 0
        ? times.find((t) => t > frame + 1e-6)
        : [...times].reverse().find((t) => t < frame - 1e-6);
    if (next !== undefined) store.seek(next);
  };

  const setKeyEase = (preset: EasePreset) => {
    if (hasKey) apply([{ op: 'setEase', target, t: frame, ease: preset }], `设置缓动：${label}`, true);
    else onEaseChange(preset);
  };

  return (
    <div className={`anim-field${disabled ? ' disabled' : ''}`} title={title}>
      <div className="anim-head">
        <button
          type="button"
          className={`key-diamond${hasKey ? ' active' : ''}`}
          data-testid="anim-diamond"
          disabled={disabled}
          title={hasKey ? '删除当前帧关键帧' : '在当前帧添加关键帧'}
          onClick={toggleKey}
        >
          ◆
        </button>
        <span className="anim-label">{label}</span>
        {curve ? (
          <span className="key-count" title={baked ? '烘焙曲线（逐帧）' : `${times?.length ?? 0} 个关键帧`}>
            {baked ? '烘焙' : `${times?.length ?? 0}K`}
          </span>
        ) : (
          <span className="key-count muted">常量</span>
        )}
        {target.kind === 'param' && (
          <button type="button" className="mini" title="固定到时间线参数轨" onClick={() => store.togglePinnedParam(target.id, target.param)}>
            {store.isPinned(target.id, target.param) ? '已固定' : '固定'}
          </button>
        )}
        <span className="spacer" />
        {curve && !baked && (
          <>
            <button type="button" className="mini" disabled={disabled || !times?.length} title="上一个关键帧" onClick={() => jumpKey(-1)}>
              ‹
            </button>
            <button type="button" className="mini" disabled={disabled || !times?.length} title="下一个关键帧" onClick={() => jumpKey(1)}>
              ›
            </button>
          </>
        )}
      </div>
      <div className="anim-body">
        <CommitNumber
          value={sampled}
          integer={integer}
          min={min}
          max={max}
          step={step ?? (integer ? 1 : 0.01)}
          disabled={disabled}
          title={title ?? label}
          onCommit={writeValue}
        />
        {range && (
          <input
            className="anim-range"
            type="range"
            min={min ?? 0}
            max={max ?? 100}
            step={step ?? 0.01}
            value={sampled}
            disabled={disabled}
            onChange={(e) => writeValue(Number(e.target.value))}
          />
        )}
        <select
          className="ease-select"
          value={displayEase}
          disabled={disabled}
          title={hasKey ? '当前关键帧的缓动' : '新关键帧的缓动'}
          onChange={(e) => setKeyEase(e.target.value as EasePreset)}
        >
          {EASE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

interface InspectorProps {
  store: EditorStore;
  state: EditorState;
  newKeyEase: EasePreset;
  setNewKeyEase: (ease: EasePreset) => void;
}

export function Inspector({ store, state, newKeyEase, setNewKeyEase }: InspectorProps) {
  const scene = state.scene;
  if (!scene) return null;
  const selection = state.selection;
  let title = '场景';
  let body: ReactNode;
  if (selection.kind === 'scene' || selection.kind === 'camera') {
    title = '场景';
    body = <SceneInspector store={store} state={state} newKeyEase={newKeyEase} setNewKeyEase={setNewKeyEase} />;
  } else if (selection.kind === 'layer') {
    const layer = scene.layers.find((item) => item.id === selection.id);
    if (!layer) {
      title = '图层';
      body = <div className="empty">图层已不存在</div>;
    } else {
      title = `图层 · ${layer.name ?? layer.id}`;
      body = (
        <LayerInspector
          store={store}
          state={state}
          layer={layer}
          newKeyEase={newKeyEase}
          setNewKeyEase={setNewKeyEase}
        />
      );
    }
  } else if (selection.kind === 'audio') {
    title = '音频片段';
    body = <AudioInspector store={store} state={state} index={selection.index} />;
  } else {
    title = '标记';
    body = <MarkerInspector store={store} state={state} id={selection.id} />;
  }

  return (
    <aside className="editor-panel editor-inspector" data-testid="inspector">
      <div className="panel-head">
        <strong>{title}</strong>
        <span className="spacer" />
        <label className="inline small" title="新建关键帧使用的缓动">
          新键
          <select className="ease-select" value={newKeyEase} onChange={(e) => setNewKeyEase(e.target.value as EasePreset)}>
            {EASE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="panel-scroll inspector-body">{body}</div>
    </aside>
  );
}

// ---------- 场景 ----------

function SceneInspector({
  store,
  state,
  newKeyEase,
  setNewKeyEase,
}: {
  store: EditorStore;
  state: EditorState;
  newKeyEase: EasePreset;
  setNewKeyEase: (ease: EasePreset) => void;
}) {
  const scene = state.scene!;
  const setScene = (patch: Record<string, unknown>, label: string) => {
    void store.applyUi([{ op: 'setScene', patch } as unknown as import('@charaforge/scene').EditCommand], { label });
  };

  const changeFps = (fps: number) => {
    if (!(fps > 0) || fps === scene.timebase.fps) return;
    if (!window.confirm(`把场景帧率改为 ${fps} fps？所有时间仍按帧保存，不改变已有关键帧的帧号。`)) return;
    setScene({ fps }, '修改帧率');
  };

  const changeSize = (key: 'width' | 'height', value: number) => {
    if (!(value > 0) || value === scene.composition[key]) return;
    if (!window.confirm(`把合成${key === 'width' ? '宽度' : '高度'}改为 ${value} px？这是结构修改，会重建合成缓冲。`)) return;
    setScene({ [key]: value }, `修改合成${key === 'width' ? '宽度' : '高度'}`);
  };

  const camera = scene.camera;
  const W = scene.composition.width;
  const H = scene.composition.height;
  const background = scene.composition.background ?? null;

  return (
    <div className="section-stack">
      <section className="inspector-section">
        <h4>基本信息</h4>
        <div className="field-row">
          <label>标题</label>
          <CommitText
            value={scene.meta.title}
            onCommit={(value) => {
              if (value.trim()) setScene({ title: value.trim() }, '修改标题');
            }}
          />
        </div>
        <div className="field-row">
          <label>描述</label>
          <CommitText
            value={scene.meta.description ?? ''}
            multiline
            onCommit={(value) => setScene({ description: value.trim() ? value : null }, '修改描述')}
          />
        </div>
        <div className="field-grid">
          <label>
            帧率
            <CommitNumber value={scene.timebase.fps} min={1} max={240} step={1} integer live={false} onCommit={changeFps} />
          </label>
          <label>
            总帧数
            <CommitNumber
              value={scene.duration}
              min={1}
              step={1}
              integer
              live={false}
              onCommit={(value) => setScene({ duration: Math.max(1, Math.round(value)) }, '修改总帧数')}
            />
          </label>
          <label>
            宽度
            <CommitNumber value={W} min={1} step={1} integer live={false} onCommit={(value) => changeSize('width', value)} />
          </label>
          <label>
            高度
            <CommitNumber value={H} min={1} step={1} integer live={false} onCommit={(value) => changeSize('height', value)} />
          </label>
        </div>
        <div className="field-row">
          <label>背景</label>
          <input
            type="color"
            className="color-input"
            value={background && /^#[0-9a-f]{6}$/i.test(background) ? background : '#000000'}
            disabled={background === null}
            onChange={(e) => void store.setBackground(e.target.value)}
          />
          <label className="inline">
            <input
              type="checkbox"
              checked={background === null}
              onChange={(e) => void store.setBackground(e.target.checked ? null : '#000000')}
            />
            透明
          </label>
          <code className="small muted">{background ?? 'transparent'}</code>
        </div>
      </section>

      <section className="inspector-section">
        <h4>
          相机
          <span className="spacer" />
          {camera ? (
            <button type="button" className="mini danger" onClick={() => void store.applyUi([{ op: 'setCamera', camera: null }], { label: '删除相机' })}>
              删除
            </button>
          ) : (
            <button
              type="button"
              className="mini"
              onClick={() =>
                void store.applyUi(
                  [{ op: 'setCamera', camera: { x: W / 2, y: H / 2, zoom: 1, rotation: 0 } }],
                  { label: '启用相机' },
                )
              }
            >
              启用
            </button>
          )}
        </h4>
        {camera ? (
          <>
            <AnimField
              label="相机 X"
              target={{ kind: 'camera', path: 'x' }}
              value={camera.x}
              store={store}
              ease={newKeyEase}
              onEaseChange={setNewKeyEase}
              onConstant={(value) => void store.applyUi([{ op: 'updateCamera', patch: { x: value } }], { label: '相机 X', history: 'coalesce' })}
              range
              min={0}
              max={W}
              step={1}
            />
            <AnimField
              label="相机 Y"
              target={{ kind: 'camera', path: 'y' }}
              value={camera.y}
              store={store}
              ease={newKeyEase}
              onEaseChange={setNewKeyEase}
              onConstant={(value) => void store.applyUi([{ op: 'updateCamera', patch: { y: value } }], { label: '相机 Y', history: 'coalesce' })}
              range
              min={0}
              max={H}
              step={1}
            />
            <AnimField
              label="缩放"
              target={{ kind: 'camera', path: 'zoom' }}
              value={camera.zoom}
              fallback={1}
              min={0.05}
              max={10}
              step={0.01}
              range
              store={store}
              ease={newKeyEase}
              onEaseChange={setNewKeyEase}
              onConstant={(value) => void store.applyUi([{ op: 'updateCamera', patch: { zoom: value } }], { label: '相机缩放', history: 'coalesce' })}
            />
            <AnimField
              label="旋转"
              target={{ kind: 'camera', path: 'rotation' }}
              value={camera.rotation}
              fallback={0}
              min={-180}
              max={180}
              step={0.1}
              range
              store={store}
              ease={newKeyEase}
              onEaseChange={setNewKeyEase}
              onConstant={(value) => void store.applyUi([{ op: 'updateCamera', patch: { rotation: value } }], { label: '相机旋转', history: 'coalesce' })}
            />
          </>
        ) : (
          <div className="empty small">没有相机。启用后可以为 x/y/zoom/rotation 打关键帧。</div>
        )}
        <div className="small muted">镜头标记：{scene.markers?.length ?? 0} 个</div>
      </section>

      <section className="inspector-section">
        <h4>统计</h4>
        <div className="kv-grid">
          <span>图层</span>
          <span>{scene.layers.length}</span>
          <span>图片资源</span>
          <span>{scene.assets.images?.length ?? 0}</span>
          <span>音频片段</span>
          <span>{scene.audio?.length ?? 0}</span>
          <span>时长</span>
          <span>{(scene.duration / scene.timebase.fps).toFixed(2)} s</span>
        </div>
      </section>
    </div>
  );
}

// ---------- 图层 ----------

interface LayerInspectorProps {
  store: EditorStore;
  state: EditorState;
  layer: Layer;
  newKeyEase: EasePreset;
  setNewKeyEase: (ease: EasePreset) => void;
}

function LayerInspector({ store, state, layer, newKeyEase, setNewKeyEase }: LayerInspectorProps) {
  const scene = state.scene!;
  const frame = state.frame;
  const W = scene.composition.width;
  const H = scene.composition.height;
  const canvas = layer.kind === 'model' ? store.modelCanvas(layer.id) : null;
  const localWidth = layer.kind === 'model' ? canvas?.width ?? 0 : store.imageAsset(layer.image)?.width ?? 0;
  const localHeight = layer.kind === 'model' ? canvas?.height ?? 0 : store.imageAsset(layer.image)?.height ?? 0;
  const defaultAnchorX = layer.transform.anchorX ?? localWidth / 2;
  const defaultAnchorY = layer.transform.anchorY ?? localHeight / 2;

  const patch = (value: Record<string, unknown>, label: string) => {
    void store.applyUi([{ op: 'updateLayer', id: layer.id, patch: value } as unknown as import('@charaforge/scene').EditCommand], {
      label,
      history: 'coalesce',
    });
  };

  const layerTarget = (path: string): EditTarget => ({ kind: 'layer', id: layer.id, path });

  const setClipConstant = (key: string, value: number) => {
    if (!layer.clip) return;
    patch({ clip: { ...layer.clip, [key]: value } as unknown }, `裁剪 ${key}`);
  };

  const setFilterConstant = (index: number, key: string, value: number) => {
    const filters = (layer.filters ?? []).map((filter, i) =>
      i === index ? ({ ...filter, [key]: value } as Filter) : filter,
    );
    patch({ filters }, `滤镜 ${key}`);
  };

  const transform = layer.transform;

  return (
    <div className="section-stack">
      <section className="inspector-section">
        <h4>图层</h4>
        <div className="field-row">
          <label>名称</label>
          <CommitText value={layer.name ?? ''} onCommit={(value) => patch({ name: value.trim() ? value.trim() : null }, '修改图层名称')} />
        </div>
        <div className="kv-grid mono small">
          <span>ID</span>
          <span>{layer.id}</span>
          <span>类型</span>
          <span>{layer.kind === 'model' ? '模型' : '图片'}</span>
          {layer.kind === 'model' ? (
            <>
              <span>模型</span>
              <span>{scene.assets.models.find((asset) => asset.id === layer.model)?.name ?? layer.model}</span>
            </>
          ) : (
            <>
              <span>图片</span>
              <span>{layer.image}</span>
            </>
          )}
        </div>
        <div className="field-grid">
          <label>
            入点
            <CommitNumber
              value={layer.in}
              min={0}
              step={1}
              integer
              onCommit={(value) => {
                if (value >= layer.out) return store.notify('入点必须小于出点', 'warn');
                patch({ in: Math.max(0, Math.round(value)) }, '修改入点');
              }}
            />
          </label>
          <label>
            出点
            <CommitNumber
              value={layer.out}
              min={1}
              step={1}
              integer
              onCommit={(value) => {
                if (value <= layer.in) return store.notify('出点必须大于入点', 'warn');
                patch({ out: Math.max(layer.in + 1, Math.round(value)) }, '修改出点');
              }}
            />
          </label>
          <label>
            空间
            <select
              value={layer.space ?? 'world'}
              disabled={layer.kind === 'image' && !!layer.attach}
              onChange={(e) => patch({ space: e.target.value as 'world' | 'screen' }, '修改空间')}
            >
              <option value="world">world</option>
              <option value="screen">screen</option>
            </select>
          </label>
          <label>
            混合
            <select
              value={layer.blendMode ?? 'normal'}
              onChange={(e) =>
                patch({ blendMode: e.target.value === 'normal' ? null : (e.target.value as 'add' | 'multiply' | 'screen') }, '修改混合模式')
              }
            >
              <option value="normal">normal</option>
              <option value="add">add</option>
              <option value="multiply">multiply</option>
              <option value="screen">screen</option>
            </select>
          </label>
        </div>
        {layer.kind === 'image' && layer.attach && <div className="small muted">已绑定到 {layer.attach.layer}，space 被忽略。</div>}
        <AnimField
          label="不透明度"
          target={layerTarget('opacity')}
          value={layer.opacity}
          fallback={1}
          min={0}
          max={1}
          step={0.01}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ opacity: value }, '不透明度')}
        />
      </section>

      <section className="inspector-section">
        <h4>变换</h4>
        <AnimField
          label="X"
          target={layerTarget('transform.x')}
          value={transform.x}
          min={-10000}
          max={10000}
          step={1}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ transform: { x: value } }, '变换 X')}
        />
        <AnimField
          label="Y"
          target={layerTarget('transform.y')}
          value={transform.y}
          min={-10000}
          max={10000}
          step={1}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ transform: { y: value } }, '变换 Y')}
        />
        <AnimField
          label="缩放"
          target={layerTarget('transform.scale')}
          value={transform.scale}
          fallback={1}
          min={0.001}
          max={20}
          step={0.001}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ transform: { scale: value } }, '缩放')}
        />
        <AnimField
          label="缩放 X"
          target={layerTarget('transform.scaleX')}
          value={transform.scaleX}
          fallback={1}
          min={0.001}
          max={20}
          step={0.001}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ transform: { scaleX: value } }, '缩放 X')}
        />
        <AnimField
          label="缩放 Y"
          target={layerTarget('transform.scaleY')}
          value={transform.scaleY}
          fallback={1}
          min={0.001}
          max={20}
          step={0.001}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ transform: { scaleY: value } }, '缩放 Y')}
        />
        <AnimField
          label="旋转"
          target={layerTarget('transform.rotation')}
          value={transform.rotation}
          fallback={0}
          min={-360}
          max={360}
          step={0.1}
          range
          store={store}
          ease={newKeyEase}
          onEaseChange={setNewKeyEase}
          onConstant={(value) => patch({ transform: { rotation: value } }, '旋转')}
        />
        <div className="field-grid">
          <label>
            锚点 X
            <CommitNumber
              value={defaultAnchorX}
              min={-localWidth * 2}
              max={localWidth * 2}
              step={1}
              title={`图片 / 模型画布像素，缺省 ${localWidth / 2}`}
              onCommit={(value) => patch({ transform: { anchorX: value } }, '锚点 X')}
            />
          </label>
          <label>
            锚点 Y
            <CommitNumber
              value={defaultAnchorY}
              min={-localHeight * 2}
              max={localHeight * 2}
              step={1}
              onCommit={(value) => patch({ transform: { anchorY: value } }, '锚点 Y')}
            />
          </label>
        </div>
        <div className="field-row">
          <label className="inline">
            <input type="checkbox" checked={!!transform.flipX} onChange={(e) => patch({ transform: { flipX: e.target.checked } }, '水平翻转')} />
            水平翻转
          </label>
          <button
            type="button"
            className="mini"
            title="锚点回到画布 / 图片中心"
            onClick={() => patch({ transform: { anchorX: localWidth / 2, anchorY: localHeight / 2 } }, '锚点居中')}
          >
            锚点居中
          </button>
        </div>
        <div className="small muted">
          本地尺寸：{localWidth || '?'}×{localHeight || '?'}　当前帧 {frame}
        </div>
      </section>

      <section className="inspector-section">
        <h4>
          裁剪
          <span className="spacer" />
          {layer.clip ? (
            <button type="button" className="mini danger" onClick={() => patch({ clip: null }, '移除裁剪')}>
              移除
            </button>
          ) : (
            <button
              type="button"
              className="mini"
              onClick={() => patch({ clip: { x: 0, y: 0, width: W, height: H, radius: 0, feather: 0 } }, '启用裁剪')}
            >
              启用
            </button>
          )}
        </h4>
        {layer.clip ? (
          <>
            {(
              [
                ['x', 'X', 0, W, 1],
                ['y', 'Y', 0, H, 1],
                ['width', '宽', 0, W, 1],
                ['height', '高', 0, H, 1],
                ['radius', '圆角', 0, 1000, 1],
                ['feather', '羽化', 0, 500, 1],
              ] as const
            ).map(([key, label, min, max, step]) => (
              <AnimField
                key={key}
                label={label}
                target={layerTarget(`clip.${key}`)}
                value={layer.clip?.[key]}
                fallback={0}
                min={min}
                max={max}
                step={step}
                range
                store={store}
                ease={newKeyEase}
                onEaseChange={setNewKeyEase}
                onConstant={(value) => setClipConstant(key, value)}
              />
            ))}
          </>
        ) : (
          <div className="empty small">没有裁剪。</div>
        )}
      </section>

      <section className="inspector-section">
        <h4>
          滤镜
          <span className="spacer" />
          <select
            className="mini-select"
            value=""
            onChange={(e) => {
              const type = e.target.value;
              if (!type) return;
              const filter: Filter =
                type === 'blur'
                  ? { type: 'blur', sigma: 4 }
                  : type === 'motionBlur'
                    ? { type: 'motionBlur', angle: 0, length: 8 }
                    : type === 'resample'
                      ? { type: 'resample', factor: 0.75 }
                      : { type: 'color', brightness: 0, contrast: 1, saturation: 1 };
              patch({ filters: [...(layer.filters ?? []), filter] }, '添加滤镜');
            }}
          >
            <option value="">+ 添加</option>
            <option value="blur">高斯模糊</option>
            <option value="motionBlur">方向模糊</option>
            <option value="resample">降采样</option>
            <option value="color">颜色调整</option>
          </select>
        </h4>
        {(layer.filters ?? []).length === 0 && <div className="empty small">没有滤镜。</div>}
        {(layer.filters ?? []).map((filter, index) => (
          <div key={index} className="filter-block">
            <div className="filter-head">
              <label className="inline">
                <input
                  type="checkbox"
                  checked={filter.enabled !== false}
                  onChange={(e) => {
                    const filters = (layer.filters ?? []).map((item, i) =>
                      i === index ? ({ ...item, enabled: e.target.checked } as Filter) : item,
                    );
                    patch({ filters }, '切换滤镜');
                  }}
                />
                {filter.type === 'blur'
                  ? '高斯模糊'
                  : filter.type === 'motionBlur'
                    ? '方向模糊'
                    : filter.type === 'resample'
                      ? '降采样'
                      : '颜色调整'}
              </label>
              <span className="spacer" />
              <button
                type="button"
                className="mini danger"
                onClick={() => patch({ filters: (layer.filters ?? []).filter((_, i) => i !== index) }, '删除滤镜')}
              >
                删除
              </button>
            </div>
            {filter.type === 'blur' && (
              <AnimField
                label="sigma"
                target={layerTarget(`filters.${index}.sigma`)}
                value={filter.sigma}
                fallback={0}
                min={0}
                max={100}
                step={0.1}
                range
                store={store}
                ease={newKeyEase}
                onEaseChange={setNewKeyEase}
                onConstant={(value) => setFilterConstant(index, 'sigma', value)}
              />
            )}
            {filter.type === 'motionBlur' && (
              <>
                <AnimField
                  label="角度"
                  target={layerTarget(`filters.${index}.angle`)}
                  value={filter.angle}
                  fallback={0}
                  min={-180}
                  max={180}
                  step={0.5}
                  range
                  store={store}
                  ease={newKeyEase}
                  onEaseChange={setNewKeyEase}
                  onConstant={(value) => setFilterConstant(index, 'angle', value)}
                />
                <AnimField
                  label="长度"
                  target={layerTarget(`filters.${index}.length`)}
                  value={filter.length}
                  fallback={0}
                  min={0}
                  max={200}
                  step={0.5}
                  range
                  store={store}
                  ease={newKeyEase}
                  onEaseChange={setNewKeyEase}
                  onConstant={(value) => setFilterConstant(index, 'length', value)}
                />
              </>
            )}
            {filter.type === 'resample' && (
              <AnimField
                label="factor"
                target={layerTarget(`filters.${index}.factor`)}
                value={filter.factor}
                fallback={1}
                min={0.02}
                max={1}
                step={0.01}
                range
                store={store}
                ease={newKeyEase}
                onEaseChange={setNewKeyEase}
                onConstant={(value) => setFilterConstant(index, 'factor', value)}
              />
            )}
            {filter.type === 'color' && (
              <>
                <AnimField
                  label="亮度"
                  target={layerTarget(`filters.${index}.brightness`)}
                  value={filter.brightness}
                  fallback={0}
                  min={-1}
                  max={1}
                  step={0.01}
                  range
                  store={store}
                  ease={newKeyEase}
                  onEaseChange={setNewKeyEase}
                  onConstant={(value) => setFilterConstant(index, 'brightness', value)}
                />
                <AnimField
                  label="对比度"
                  target={layerTarget(`filters.${index}.contrast`)}
                  value={filter.contrast}
                  fallback={1}
                  min={0}
                  max={3}
                  step={0.01}
                  range
                  store={store}
                  ease={newKeyEase}
                  onEaseChange={setNewKeyEase}
                  onConstant={(value) => setFilterConstant(index, 'contrast', value)}
                />
                <AnimField
                  label="饱和度"
                  target={layerTarget(`filters.${index}.saturation`)}
                  value={filter.saturation}
                  fallback={1}
                  min={0}
                  max={3}
                  step={0.01}
                  range
                  store={store}
                  ease={newKeyEase}
                  onEaseChange={setNewKeyEase}
                  onConstant={(value) => setFilterConstant(index, 'saturation', value)}
                />
              </>
            )}
          </div>
        ))}
      </section>

      {layer.kind === 'model' && (
        <section className="inspector-section">
          <h4>物理</h4>
          <div className="field-grid">
            <label className="inline">
              <input
                type="checkbox"
                checked={layer.physics?.enabled !== false}
                onChange={(e) => patch({ physics: { ...(layer.physics ?? {}), enabled: e.target.checked } }, '物理开关')}
              />
              启用物理
            </label>
            <label>
              预热帧
              <CommitNumber
                value={layer.physics?.warmup ?? 0}
                min={0}
                max={600}
                step={1}
                integer
                onCommit={(value) => patch({ physics: { ...(layer.physics ?? {}), warmup: Math.round(value) } }, '物理预热')}
              />
            </label>
            <label className="inline">
              <input
                type="checkbox"
                checked={layer.physics?.stabilize === true}
                onChange={(e) => patch({ physics: { ...(layer.physics ?? {}), stabilize: e.target.checked } }, '物理稳定')}
              />
              预热前稳定
            </label>
          </div>
        </section>
      )}

      {layer.kind === 'image' && (
        <section className="inspector-section">
          <h4>绑定</h4>
          <div className="field-row">
            <label>目标图层</label>
            <select
              value={layer.attach?.layer ?? ''}
              onChange={(e) => {
                const id = e.target.value;
                if (!id) {
                  patch({ attach: null, space: 'screen', transform: { x: W / 2, y: H / 2 } }, '解除绑定');
                } else {
                  patch(
                    {
                      attach: {
                        layer: id,
                        anchor: layer.attach?.anchor ?? [0.5, 0.2],
                        offset: layer.attach?.offset ?? [0, 0],
                        rotateWith: layer.attach?.rotateWith ?? true,
                        scaleWith: layer.attach?.scaleWith ?? true,
                      },
                      transform: { x: 0, y: 0 },
                    },
                    '绑定到模型',
                  );
                }
              }}
            >
              <option value="">不绑定（独立图片）</option>
              {scene.layers
                .filter((item): item is ModelLayer => item.kind === 'model')
                .map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name ?? item.id}
                  </option>
                ))}
            </select>
          </div>
          {layer.attach && (
            <>
              <div className="field-grid">
                <label>
                  anchor X
                  <CommitNumber
                    value={layer.attach.anchor[0]}
                    min={0}
                    max={1}
                    step={0.001}
                    onCommit={(value) =>
                      patch(
                        { attach: { ...layer.attach!, anchor: [value, layer.attach!.anchor[1]] } },
                        '绑定锚点 X',
                      )
                    }
                  />
                </label>
                <label>
                  anchor Y
                  <CommitNumber
                    value={layer.attach.anchor[1]}
                    min={0}
                    max={1}
                    step={0.001}
                    onCommit={(value) =>
                      patch(
                        { attach: { ...layer.attach!, anchor: [layer.attach!.anchor[0], value] } },
                        '绑定锚点 Y',
                      )
                    }
                  />
                </label>
                <label>
                  offset X
                  <CommitNumber
                    value={layer.attach.offset?.[0] ?? 0}
                    step={1}
                    onCommit={(value) =>
                      patch(
                        { attach: { ...layer.attach!, offset: [value, layer.attach!.offset?.[1] ?? 0] } },
                        '绑定偏移 X',
                      )
                    }
                  />
                </label>
                <label>
                  offset Y
                  <CommitNumber
                    value={layer.attach.offset?.[1] ?? 0}
                    step={1}
                    onCommit={(value) =>
                      patch(
                        { attach: { ...layer.attach!, offset: [layer.attach!.offset?.[0] ?? 0, value] } },
                        '绑定偏移 Y',
                      )
                    }
                  />
                </label>
              </div>
              <div className="field-row">
                <label className="inline">
                  <input
                    type="checkbox"
                    checked={layer.attach.rotateWith ?? true}
                    onChange={(e) => patch({ attach: { ...layer.attach!, rotateWith: e.target.checked } }, '跟随旋转')}
                  />
                  跟随旋转
                </label>
                <label className="inline">
                  <input
                    type="checkbox"
                    checked={layer.attach.scaleWith ?? true}
                    onChange={(e) => patch({ attach: { ...layer.attach!, scaleWith: e.target.checked } }, '跟随缩放')}
                  />
                  跟随缩放
                </label>
              </div>
            </>
          )}
        </section>
      )}

      {layer.kind === 'model' && (
        <ClipEditors store={store} state={state} layer={layer} newKeyEase={newKeyEase} setNewKeyEase={setNewKeyEase} />
      )}

      <section className="inspector-section">
        <h4>一键动画</h4>
        <div className="preset-buttons">
          {PRESET_VALUES.map((preset) => (
            <button key={preset} type="button" className="mini" onClick={() => void store.addAnimationPreset(preset, layer.id)}>
              {PRESET_LABELS[preset]}
            </button>
          ))}
        </div>
        <div className="small muted">从当前帧开始，默认持续 90 帧（不超过图层出点）。</div>
      </section>

      {layer.kind === 'model' && <ParamEditor store={store} state={state} layer={layer} newKeyEase={newKeyEase} />}
    </div>
  );
}

// ---------- 动作 / 表情片段 ----------

function ClipEditors({
  store,
  state,
  layer,
  newKeyEase: _newKeyEase,
  setNewKeyEase: _setNewKeyEase,
}: {
  store: EditorStore;
  state: EditorState;
  layer: ModelLayer;
  newKeyEase: EasePreset;
  setNewKeyEase: (ease: EasePreset) => void;
}) {
  const scene = state.scene!;
  const catalog = state.catalog;
  const motionAsset = (id: string) => scene.assets.motions?.find((asset) => asset.id === id);
  const expressionAsset = (id: string) => scene.assets.expressions?.find((asset) => asset.id === id);

  return (
    <section className="inspector-section">
      <h4>动作</h4>
      <div className="clip-list">
        {(layer.motions ?? []).length === 0 && <div className="empty small">没有动作片段。</div>}
        {(layer.motions ?? []).map((clip, index) => (
          <div key={index} className="clip-card">
            <div className="clip-head">
              <span className="mono">{motionAsset(clip.motion)?.uri.split('/').pop() ?? clip.motion}</span>
              <span className="spacer" />
              <button
                type="button"
                className="mini danger"
                onClick={() =>
                  void store.applyUi([{ op: 'removeMotion', layer: layer.id, index }], { label: '删除动作片段' })
                }
              >
                删除
              </button>
            </div>
            <div className="field-grid">
              <label>
                开始
                <CommitNumber
                  value={clip.start}
                  min={0}
                  step={1}
                  integer
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { start: value } }], {
                      label: '动作开始帧',
                    })
                  }
                />
              </label>
              <label>
                结束
                <CommitNumber
                  value={clip.end ?? layer.out}
                  min={clip.start + 1}
                  step={1}
                  integer
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { end: value } }], {
                      label: '动作结束帧',
                    })
                  }
                />
                <button
                  type="button"
                  className="mini"
                  title="删除 end，使用动作自然结束"
                  onClick={() =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { end: null } }], {
                      label: '动作自然结束',
                    })
                  }
                >
                  自动
                </button>
              </label>
              <label>
                offset(s)
                <CommitNumber
                  value={clip.offset ?? 0}
                  min={0}
                  step={0.01}
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { offset: value } }], {
                      label: '动作 offset',
                    })
                  }
                />
              </label>
              <label>
                speed
                <CommitNumber
                  value={clip.speed ?? 1}
                  min={0.05}
                  max={8}
                  step={0.01}
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { speed: value } }], {
                      label: '动作速度',
                    })
                  }
                />
              </label>
              <label>
                淡入(s)
                <CommitNumber
                  value={clip.fadeIn ?? 0}
                  min={0}
                  step={0.01}
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { fadeIn: value } }], {
                      label: '动作淡入',
                    })
                  }
                />
              </label>
              <label>
                淡出(s)
                <CommitNumber
                  value={clip.fadeOut ?? 0}
                  min={0}
                  step={0.01}
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { fadeOut: value } }], {
                      label: '动作淡出',
                    })
                  }
                />
              </label>
            </div>
            <label className="inline">
              <input
                type="checkbox"
                checked={clip.loop === true}
                onChange={(e) =>
                  void store.applyUi([{ op: 'updateMotion', layer: layer.id, index, patch: { loop: e.target.checked } }], {
                    label: '动作循环',
                  })
                }
              />
              循环
            </label>
          </div>
        ))}
      </div>
      <div className="field-row">
        <select
          className="grow"
          value=""
          disabled={!catalog?.motions.length}
          onChange={(e) => {
            const uri = e.target.value;
            if (uri) void store.addMotion(uri);
          }}
        >
          <option value="">{catalog?.motions.length ? '+ 添加动作…' : '目录未加载 / 无动作'}</option>
          {(catalog?.motions ?? []).map((asset) => (
            <option key={asset.uri} value={asset.uri}>
              {asset.name}（{asset.meta.duration.toFixed(2)}s）
            </option>
          ))}
        </select>
      </div>

      <h4>表情</h4>
      <div className="clip-list">
        {(layer.expressions ?? []).length === 0 && <div className="empty small">没有表情片段。</div>}
        {(layer.expressions ?? []).map((clip, index) => (
          <div key={index} className="clip-card">
            <div className="clip-head">
              <span className="mono">{expressionAsset(clip.expression)?.uri.split('/').pop() ?? clip.expression}</span>
              <span className="spacer" />
              <button
                type="button"
                className="mini danger"
                onClick={() =>
                  void store.applyUi([{ op: 'removeExpression', layer: layer.id, index }], { label: '删除表情片段' })
                }
              >
                删除
              </button>
            </div>
            <div className="field-grid">
              <label>
                开始
                <CommitNumber
                  value={clip.start}
                  min={0}
                  step={1}
                  integer
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateExpression', layer: layer.id, index, patch: { start: value } }], {
                      label: '表情开始帧',
                    })
                  }
                />
              </label>
              <label>
                结束
                <CommitNumber
                  value={clip.end ?? layer.out}
                  min={clip.start + 1}
                  step={1}
                  integer
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateExpression', layer: layer.id, index, patch: { end: value } }], {
                      label: '表情结束帧',
                    })
                  }
                />
                <button
                  type="button"
                  className="mini"
                  onClick={() =>
                    void store.applyUi([{ op: 'updateExpression', layer: layer.id, index, patch: { end: null } }], {
                      label: '表情持续到图层结束',
                    })
                  }
                >
                  自动
                </button>
              </label>
              <label>
                淡入(s)
                <CommitNumber
                  value={clip.fadeIn ?? 0}
                  min={0}
                  step={0.01}
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateExpression', layer: layer.id, index, patch: { fadeIn: value } }], {
                      label: '表情淡入',
                    })
                  }
                />
              </label>
              <label>
                淡出(s)
                <CommitNumber
                  value={clip.fadeOut ?? 0}
                  min={0}
                  step={0.01}
                  onCommit={(value) =>
                    void store.applyUi([{ op: 'updateExpression', layer: layer.id, index, patch: { fadeOut: value } }], {
                      label: '表情淡出',
                    })
                  }
                />
              </label>
            </div>
          </div>
        ))}
      </div>
      <div className="field-row">
        <select
          className="grow"
          value=""
          disabled={!catalog?.expressions.length}
          onChange={(e) => {
            const uri = e.target.value;
            if (uri) void store.addExpression(uri);
          }}
        >
          <option value="">{catalog?.expressions.length ? '+ 添加表情…' : '目录未加载 / 无表情'}</option>
          {(catalog?.expressions ?? []).map((asset) => (
            <option key={asset.uri} value={asset.uri}>
              {asset.name}（{asset.info.parameters.length} 参数）
            </option>
          ))}
        </select>
      </div>
    </section>
  );
}

// ---------- 参数轨 ----------

function ParamEditor({
  store,
  state,
  layer,
  newKeyEase,
}: {
  store: EditorStore;
  state: EditorState;
  layer: ModelLayer;
  newKeyEase: EasePreset;
}) {
  const [query, setQuery] = useState('');
  const instance = store.engine.runtime?.layerInstance(layer.id);
  const tracks = layer.params ?? [];
  const trackMap = new Map(tracks.map((track) => [track.id, track]));
  const ids = new Set<string>([...tracks.map((track) => track.id), ...(instance?.resource.parameters.map((param) => param.id) ?? [])]);
  const normalized = query.trim().toLowerCase();
  const rows = [...ids]
    .filter((id) => !normalized || id.toLowerCase().includes(normalized))
    .slice(0, 120)
    .map((id) => {
      const info = instance?.resource.parameters.find((param) => param.id === id);
      return { id, info, track: trackMap.get(id) };
    });

  return (
    <section className="inspector-section">
      <h4>
        参数
        <span className="spacer" />
        <span className="small muted">
          {tracks.length} 轨 / {instance?.resource.parameters.length ?? 0} 参数
        </span>
      </h4>
      <input
        className="text-input search"
        type="search"
        placeholder="搜索参数 ID…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="param-list">
        {rows.length === 0 && <div className="empty small">没有匹配的参数。</div>}
        {rows.map(({ id, info, track }) => {
          const frame = state.frame;
          const value = track ? sample(track.curve, frame, 0) : instance?.getParam(id) ?? info?.default ?? 0;
          const times = track ? keyTimes(track.curve) : null;
          const hasKey = track ? keyAt(track.curve, frame) : false;
          const pinned = store.isPinned(layer.id, id);
          return (
            <div key={id} className={`param-row${track ? ' has-track' : ''}`}>
              <button
                type="button"
                className={`key-diamond${hasKey ? ' active' : ''}`}
                data-testid="anim-diamond"
                title={hasKey ? '删除当前帧关键帧' : '添加关键帧'}
                onClick={() => {
                  if (!track) {
                    void store.applyUi(
                      [{ op: 'addKeyframe', target: { kind: 'param', id: layer.id, param: id }, t: frame, v: value, ease: newKeyEase }],
                      { label: `参数关键帧：${id}` },
                    );
                  } else if (hasKey) {
                    if (times && times.length <= 1) {
                      void store.applyUi([{ op: 'setKeyframes', target: { kind: 'param', id: layer.id, param: id }, keys: [] }], {
                        label: `删除参数轨：${id}`,
                      });
                    } else {
                      void store.applyUi(
                        [{ op: 'removeKeyframe', target: { kind: 'param', id: layer.id, param: id }, t: frame }],
                        { label: `删除参数关键帧：${id}` },
                      );
                    }
                  } else {
                    void store.applyUi(
                      [{ op: 'addKeyframe', target: { kind: 'param', id: layer.id, param: id }, t: frame, v: value, ease: newKeyEase }],
                      { label: `参数关键帧：${id}` },
                    );
                  }
                }}
              >
                ◆
              </button>
              <span className="param-name mono" title={info?.name ?? id}>
                {id}
              </span>
              <CommitNumber
                value={value}
                min={info?.min ?? -100}
                max={info?.max ?? 100}
                step={0.01}
                onCommit={(next) =>
                  void store.applyUi(
                    [{ op: 'addKeyframe', target: { kind: 'param', id: layer.id, param: id }, t: frame, v: next, ease: newKeyEase }],
                    { label: `参数关键帧：${id}`, history: 'coalesce' },
                  )
                }
              />
              <button
                type="button"
                className={`mini${pinned ? ' active' : ''}`}
                title="固定到时间线参数轨"
                onClick={() => store.togglePinnedParam(layer.id, id)}
              >
                {pinned ? '已固定' : '固定'}
              </button>
              {track && (
                <>
                  <button
                    type="button"
                    className="mini"
                    title="上一个关键帧"
                    disabled={!times?.length}
                    onClick={() => {
                      const prev = times ? [...times].reverse().find((t) => t < frame - 1e-6) : undefined;
                      if (prev !== undefined) store.seek(prev);
                    }}
                  >
                    ‹
                  </button>
                  <button
                    type="button"
                    className="mini"
                    title="下一个关键帧"
                    disabled={!times?.length}
                    onClick={() => {
                      const next = times?.find((t) => t > frame + 1e-6);
                      if (next !== undefined) store.seek(next);
                    }}
                  >
                    ›
                  </button>
                  <span className="param-keys small muted">{times?.length ?? 0}K</span>
                </>
              )}
            </div>
          );
        })}
      </div>
      {ids.size > rows.length && <div className="small muted">只显示前 {rows.length} 个匹配参数，请用搜索缩小范围。</div>}
    </section>
  );
}

// ---------- 音频 ----------

function AudioInspector({ store, state, index }: { store: EditorStore; state: EditorState; index: number }) {
  const scene = state.scene!;
  const clip: AudioClip | undefined = scene.audio?.[index];
  if (!clip) return <div className="empty">音频片段已不存在。</div>;
  const assets = scene.assets.audio ?? [];
  const patch = (value: Record<string, unknown>, label: string) =>
    void store.applyUi([{ op: 'updateAudio', index, patch: value } as unknown as import('@charaforge/scene').EditCommand], { label });

  return (
    <div className="section-stack">
      <section className="inspector-section">
        <h4>音频片段</h4>
        <div className="field-row">
          <label>资源</label>
          <select value={clip.audio} onChange={(e) => patch({ audio: e.target.value }, '选择音频资源')}>
            {assets.map((asset) => (
              <option key={asset.id} value={asset.id}>
                {asset.uri}（{state.audioDurations[asset.id]?.toFixed(2) ?? '?'}s）
              </option>
            ))}
          </select>
        </div>
        <div className="field-grid">
          <label>
            开始帧
            <CommitNumber value={clip.start} min={0} step={1} integer onCommit={(value) => patch({ start: value }, '音频开始帧')} />
          </label>
          <label>
            offset(s)
            <CommitNumber value={clip.offset ?? 0} min={0} step={0.01} onCommit={(value) => patch({ offset: value }, '音频 offset')} />
          </label>
          <label>
            时长(s)
            <CommitNumber
              value={clip.duration ?? state.audioDurations[clip.audio] ?? 0}
              min={0}
              step={0.01}
              onCommit={(value) => patch({ duration: value > 0 ? value : null }, '音频时长')}
            />
          </label>
          <label>
            增益
            <CommitNumber value={clip.gain ?? 1} min={0} max={4} step={0.01} onCommit={(value) => patch({ gain: value }, '音频增益')} />
          </label>
        </div>
        <div className="field-row">
          <button type="button" className="mini" onClick={() => store.seek(clip.start)}>
            跳到开始
          </button>
          <button
            type="button"
            className="mini danger"
            onClick={() =>
              void store.applyUi([{ op: 'removeAudio', index }], { label: '删除音频片段' }).then((result) => {
                if (result) store.select({ kind: 'scene' });
              })
            }
          >
            删除片段
          </button>
        </div>
        <div className="small muted">片段时长为实际音频长度时会在时间线按素材时长绘制。</div>
      </section>
    </div>
  );
}

// ---------- 标记 ----------

function MarkerInspector({ store, state, id }: { store: EditorStore; state: EditorState; id: string }) {
  const scene = state.scene!;
  const marker: Marker | undefined = scene.markers?.find((item) => item.id === id);
  if (!marker) return <div className="empty">标记已不存在。</div>;
  const patch = (value: Record<string, unknown>, label: string) =>
    void store.applyUi([{ op: 'updateMarker', id, patch: value } as unknown as import('@charaforge/scene').EditCommand], { label });

  return (
    <div className="section-stack">
      <section className="inspector-section">
        <h4>标记</h4>
        <div className="kv-grid mono small">
          <span>ID</span>
          <span>{marker.id}</span>
        </div>
        <div className="field-row">
          <label>标签</label>
          <CommitText value={marker.label ?? ''} onCommit={(value) => patch({ label: value.trim() ? value.trim() : null }, '标记标签')} />
        </div>
        <div className="field-row">
          <label>类型</label>
          <CommitText value={marker.kind ?? ''} onCommit={(value) => patch({ kind: value.trim() ? value.trim() : null }, '标记类型')} />
        </div>
        <div className="field-grid">
          <label>
            开始
            <CommitNumber value={marker.start} min={0} step={1} integer onCommit={(value) => patch({ start: value }, '标记开始帧')} />
          </label>
          <label>
            结束
            <CommitNumber
              value={marker.end ?? marker.start + 1}
              min={marker.start}
              step={1}
              integer
              onCommit={(value) => patch({ end: value > marker.start ? value : null }, '标记结束帧')}
            />
          </label>
        </div>
        <div className="field-row">
          <button type="button" className="mini" onClick={() => store.seek(marker.start)}>
            跳到标记
          </button>
          <button
            type="button"
            className="mini danger"
            onClick={() =>
              void store.applyUi([{ op: 'removeMarker', id }], { label: '删除标记' }).then((result) => {
                if (result) store.select({ kind: 'scene' });
              })
            }
          >
            删除标记
          </button>
        </div>
      </section>
    </div>
  );
}
