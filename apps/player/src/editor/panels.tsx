/*
 * 左侧面板：图层 / 素材 / 动作 / 元素。
 * 所有操作通过 EditorStore 的命令接口完成，不直接改 scene 对象。
 */
import type { AnimationPreset } from '@charaforge/scene';
import { useRef, useState } from 'react';
import type { EditorState } from './contracts.ts';
import type { EditorStore } from './store.ts';

const TOOL_LABELS: { id: 'layers' | 'assets' | 'motions' | 'elements'; label: string }[] = [
  { id: 'layers', label: '图层' },
  { id: 'assets', label: '素材' },
  { id: 'motions', label: '动作' },
  { id: 'elements', label: '元素' },
];

export function Panels({ store, state }: { store: EditorStore; state: EditorState }) {
  return (
    <aside className="editor-panel editor-left">
      <div className="tool-tabs" role="tablist">
        {TOOL_LABELS.map((tool) => (
          <button
            key={tool.id}
            type="button"
            role="tab"
            aria-selected={state.activeTool === tool.id}
            className={state.activeTool === tool.id ? 'active' : ''}
            onClick={() => store.setActiveTool(tool.id)}
          >
            {tool.label}
          </button>
        ))}
      </div>
      <div className="panel-scroll">
        {state.activeTool === 'layers' && <LayersPanel store={store} state={state} />}
        {state.activeTool === 'assets' && <AssetsPanel store={store} state={state} />}
        {state.activeTool === 'motions' && <MotionsPanel store={store} state={state} />}
        {state.activeTool === 'elements' && <ElementsPanel store={store} state={state} />}
      </div>
    </aside>
  );
}

// ---------- 图层 ----------

function LayersPanel({ store, state }: { store: EditorStore; state: EditorState }) {
  const scene = state.scene!;
  const [modelId, setModelId] = useState(scene.assets.models[0]?.id ?? '');
  const effectiveModelId = scene.assets.models.some((asset) => asset.id === modelId) ? modelId : scene.assets.models[0]?.id ?? '';
  const imageInput = useRef<HTMLInputElement>(null);
  const audioInput = useRef<HTMLInputElement>(null);
  const hiddenOpacity = useRef(new Map<string, number>());
  const selectedId = state.selection.kind === 'layer' ? state.selection.id : null;
  // 顶部的图层最靠前：scene.layers 末尾是最上层。
  const layers = [...scene.layers].reverse();

  const toggleVisible = (id: string, opacity: number) => {
    if (opacity > 0) {
      hiddenOpacity.current.set(id, opacity);
      void store.applyUi([{ op: 'updateLayer', id, patch: { opacity: 0 } }], { label: `隐藏图层：${id}` });
    } else {
      const restored = hiddenOpacity.current.get(id) ?? 1;
      void store.applyUi([{ op: 'updateLayer', id, patch: { opacity: restored } }], { label: `显示图层：${id}` });
    }
  };

  return (
    <div className="panel-section">
      <div className="panel-actions">
        <select value={effectiveModelId} onChange={(e) => setModelId(e.target.value)} title="选择模型资源">
          {scene.assets.models.map((asset) => (
            <option key={asset.id} value={asset.id}>
              {asset.name ?? asset.id}
            </option>
          ))}
        </select>
        <button type="button" className="primary" onClick={() => void store.addModelLayer(effectiveModelId)} disabled={!scene.assets.models.length}>
          新建模型实例
        </button>
        <button type="button" onClick={() => imageInput.current?.click()}>
          导入图片
        </button>
        <button type="button" onClick={() => audioInput.current?.click()}>
          添加音频
        </button>
        <button
          type="button"
          onClick={() => {
            const color = window.prompt('背景色（CSS 颜色，留空取消）', '#14202b');
            if (color?.trim()) void store.setBackground(color.trim());
          }}
        >
          添加背景色
        </button>
        <input
          ref={imageInput}
          type="file"
          accept="image/png,image/webp"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.currentTarget.value = '';
            if (file) void store.importImageFile(file).catch(() => {});
          }}
        />
        <input
          ref={audioInput}
          type="file"
          accept="audio/wav,audio/mpeg,audio/ogg,audio/mp4,.mp3,.wav,.ogg,.m4a"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.currentTarget.value = '';
            if (file) void store.importAudioFile(file).catch(() => {});
          }}
        />
      </div>

      <div className="layer-list">
        {layers.length === 0 && <div className="empty">场景没有图层。</div>}
        {layers.map((layer) => {
          const z = scene.layers.indexOf(layer);
          const opacity = store.sampleTarget({ kind: 'layer', id: layer.id, path: 'opacity' }).value;
          return (
            <div
              key={layer.id}
              data-testid={`layer-row-${layer.id}`}
              className={`layer-row${selectedId === layer.id ? ' active' : ''}`}
              onClick={() => store.select({ kind: 'layer', id: layer.id })}
            >
              <button
                type="button"
                className="mini icon"
                title={opacity > 0 ? '隐藏（opacity = 0）' : '显示'}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleVisible(layer.id, opacity);
                }}
              >
                {opacity > 0 ? '◉' : '○'}
              </button>
              <span className={`layer-kind ${layer.kind}`} title={layer.kind === 'model' ? '模型' : '图片'}>
                {layer.kind === 'model' ? '模' : '图'}
              </span>
              <span className="layer-text">
                <span className="layer-name">{layer.name ?? layer.id}</span>
                <span className="layer-sub mono">
                  z{z}　{layer.in}–{layer.out}
                  {layer.kind === 'image' && layer.attach ? `　→ ${layer.attach.layer}` : ''}
                </span>
              </span>
              <span className="layer-actions" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="mini icon"
                  title="上移一层（更靠前）"
                  disabled={z >= scene.layers.length - 1}
                  onClick={() =>
                    void store.applyUi([{ op: 'moveLayer', id: layer.id, index: z + 1 }], { label: `上移图层：${layer.id}` })
                  }
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="mini icon"
                  title="下移一层（更靠后）"
                  disabled={z <= 0}
                  onClick={() =>
                    void store.applyUi([{ op: 'moveLayer', id: layer.id, index: z - 1 }], { label: `下移图层：${layer.id}` })
                  }
                >
                  ↓
                </button>
                <button type="button" className="mini icon" title="复制" onClick={() => void store.duplicateLayer(layer.id)}>
                  ⧉
                </button>
                <button
                  type="button"
                  className="mini icon danger"
                  title="删除"
                  onClick={() => {
                    if (window.confirm(`删除图层 ${layer.name ?? layer.id}？`)) {
                      void store
                        .applyUi([{ op: 'removeLayer', id: layer.id }], { label: `删除图层：${layer.id}` })
                        .then((result) => {
                          if (result) store.select({ kind: 'scene' });
                        });
                    }
                  }}
                >
                  ×
                </button>
              </span>
            </div>
          );
        })}
      </div>
      <div className="small muted panel-note">
        列表按 z 从高到低排列（顶部最靠前）。模型图层共享同一个 moc，实例各自独立求值。
      </div>
    </div>
  );
}

// ---------- 素材 ----------

function AssetsPanel({ store, state }: { store: EditorStore; state: EditorState }) {
  const scene = state.scene!;
  const images = scene.assets.images ?? [];
  const audio = scene.assets.audio ?? [];
  const imageInput = useRef<HTMLInputElement>(null);
  const audioInput = useRef<HTMLInputElement>(null);

  return (
    <div className="panel-section">
      <div className="panel-actions">
        <button type="button" onClick={() => imageInput.current?.click()}>
          导入图片
        </button>
        <button type="button" onClick={() => audioInput.current?.click()}>
          导入音频
        </button>
        <input
          ref={imageInput}
          type="file"
          accept="image/png,image/webp"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.currentTarget.value = '';
            if (file) void store.importImageFile(file).catch(() => {});
          }}
        />
        <input
          ref={audioInput}
          type="file"
          accept="audio/wav,audio/mpeg,audio/ogg,audio/mp4,.mp3,.wav,.ogg,.m4a"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.currentTarget.value = '';
            if (file) void store.importAudioFile(file).catch(() => {});
          }}
        />
      </div>

      <h4>图片（{images.length}）</h4>
      {images.length === 0 && <div className="empty small">还没有图片资源。导入 PNG / WebP 后会出现在这里。</div>}
      <div className="asset-list">
        {images.map((asset) => (
          <div key={asset.id} className="asset-row">
            <img className="asset-thumb" src={store.resolveAsset(asset.uri)} alt="" loading="lazy" />
            <span className="asset-text">
              <span className="asset-name">{asset.id}</span>
              <span className="small muted mono">
                {asset.width}×{asset.height}
                {asset.uri.startsWith('data:') ? ' · 内嵌' : ''}
              </span>
            </span>
            <span className="asset-actions">
              <button type="button" className="mini" onClick={() => void store.addImageAssetLayer(asset.id)}>
                加为图层
              </button>
              <button type="button" className="mini" onClick={() => void store.addImageAssetLayer(asset.id, { asBackground: true })}>
                设为背景
              </button>
            </span>
          </div>
        ))}
      </div>

      <h4>音频（{audio.length}）</h4>
      {audio.length === 0 && <div className="empty small">还没有音频资源。</div>}
      <div className="asset-list">
        {audio.map((asset) => (
          <div key={asset.id} className="asset-row">
            <span className="asset-audio-icon">♪</span>
            <span className="asset-text">
              <span className="asset-name">{asset.id}</span>
              <span className="small muted mono">
                {state.audioDurations[asset.id] !== undefined
                  ? `${state.audioDurations[asset.id]!.toFixed(2)} s`
                  : '时长读取中…'}{' '}
                · {asset.uri}
              </span>
            </span>
            <span className="asset-actions">
              <button type="button" className="mini" onClick={() => void store.addAudioAsset(asset.id)}>
                插入播放头
              </button>
            </span>
          </div>
        ))}
      </div>
      <div className="small muted panel-note">
        导入的素材会上传到仓库 <code>assets/editor/</code>（本地编辑接口可用时）；否则以 data URI 嵌入场景。
      </div>
    </div>
  );
}

// ---------- 动作 ----------

function MotionsPanel({ store, state }: { store: EditorStore; state: EditorState }) {
  const [tab, setTab] = useState<'motion' | 'expression'>('motion');
  const [query, setQuery] = useState('');
  const catalog = state.catalog;
  const hasModel = state.scene?.layers.some((layer) => layer.kind === 'model') ?? false;
  const normalized = query.trim().toLowerCase();
  const motions = (catalog?.motions ?? []).filter((asset) => !normalized || asset.name.toLowerCase().includes(normalized));
  const expressions = (catalog?.expressions ?? []).filter((asset) => !normalized || asset.name.toLowerCase().includes(normalized));
  const list = tab === 'motion' ? motions : expressions;

  return (
    <div className="panel-section">
      <div className="tool-tabs sub">
        <button type="button" className={tab === 'motion' ? 'active' : ''} onClick={() => setTab('motion')}>
          动作 {catalog?.motions.length ?? 0}
        </button>
        <button type="button" className={tab === 'expression' ? 'active' : ''} onClick={() => setTab('expression')}>
          表情 {catalog?.expressions.length ?? 0}
        </button>
      </div>
      <input
        className="text-input search"
        type="search"
        placeholder={`搜索${tab === 'motion' ? '动作' : '表情'}名称…`}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="small muted">
        {catalog
          ? `来源：${tab === 'motion' ? (catalog.sources.motions === 'declared' ? 'model3.json 声明' : 'Animation/ 目录') : catalog.sources.expressions === 'declared' ? 'model3.json 声明' : 'Exp3/ 目录'}`
          : state.catalogError
            ? '目录读取失败'
            : '读取动作 / 表情目录…'}
      </div>
      {!hasModel && <div className="notice-inline">场景没有模型图层，添加前请先新建模型实例。</div>}
      {state.catalogError && <pre className="panel-error">{state.catalogError}</pre>}
      {catalog?.warnings.map((warning) => (
        <div key={warning} className="panel-warning">
          {warning}
        </div>
      ))}
      {!catalog ? (
        <div className="empty small">读取素材目录…</div>
      ) : list.length === 0 ? (
        <div className="empty small">没有匹配的{tab === 'motion' ? '动作' : '表情'}。</div>
      ) : (
        <div className="motion-list">
          {list.map((asset) => (
            <div key={asset.uri} className="motion-row">
              <span className="asset-text">
                <span className="asset-name">{asset.name}</span>
                <span className="small muted">
                  {asset.kind === 'motion'
                    ? `${asset.meta.duration.toFixed(3)} s · ${asset.meta.loop ? 'loop' : 'once'} · ${asset.meta.curveCount} 曲线`
                    : `${asset.info.parameters.length} 参数 · ${[...new Set(asset.info.parameters.map((p) => p.blend))].join('/') || '—'}`}
                </span>
              </span>
              <button
                type="button"
                className="mini"
                disabled={!hasModel}
                title={hasModel ? '添加到当前选中模型图层（没有选中时用第一个模型图层）' : '没有模型图层'}
                onClick={() => {
                  if (asset.kind === 'motion') void store.addMotion(asset.uri, asset.name);
                  else void store.addExpression(asset.uri);
                }}
              >
                添加到选中图层
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="small muted panel-note">
        动作从当前播放头开始，默认不循环、淡入 0；表情从当前帧开始，持续到图层出点。片段参数可在右侧检查器里细调。
      </div>
    </div>
  );
}

// ---------- 元素 ----------

type ShapeKind = 'circle' | 'star' | 'heart' | 'sparkle' | 'bubble' | 'text';

const SHAPE_LABELS: { value: ShapeKind; label: string }[] = [
  { value: 'circle', label: '圆形' },
  { value: 'star', label: '星星' },
  { value: 'heart', label: '爱心' },
  { value: 'sparkle', label: '闪光' },
  { value: 'bubble', label: '气泡' },
  { value: 'text', label: '文字' },
];

const QUICK_PRESETS: { value: AnimationPreset; label: string }[] = [
  { value: 'fadeIn', label: '淡入' },
  { value: 'fadeOut', label: '淡出' },
  { value: 'float', label: '漂浮' },
  { value: 'pulse', label: '脉冲' },
  { value: 'spin', label: '旋转' },
  { value: 'bounce', label: '弹跳' },
  { value: 'blink', label: '闪烁' },
  { value: 'slideIn', label: '滑入' },
  { value: 'shake', label: '抖动' },
  { value: 'zoomIn', label: '放大' },
];

function starPath(ctx: CanvasRenderingContext2D, cx: number, cy: number, outer: number, inner: number, points = 5, rotation = -Math.PI / 2): void {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const radius = i % 2 === 0 ? outer : inner;
    const angle = rotation + (i * Math.PI) / points;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

/** 生成透明 PNG 元素（元素面板 / AI 预览用）。 */
export async function generateShapePng(options: {
  shape: ShapeKind;
  color: string;
  size: number;
  text?: string;
}): Promise<Blob> {
  const size = Math.max(16, Math.min(2048, Math.round(options.size)));
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('无法创建 2D 画布');
  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.42;
  ctx.fillStyle = options.color;
  ctx.strokeStyle = options.color;
  ctx.lineWidth = Math.max(1, size * 0.03);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';

  switch (options.shape) {
    case 'circle':
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      break;
    case 'star':
      starPath(ctx, cx, cy, r, r * 0.45, 5);
      ctx.fill();
      break;
    case 'heart': {
      const s = size / 100;
      ctx.save();
      ctx.translate(cx, cy + 4 * s);
      ctx.scale(s, s);
      ctx.beginPath();
      ctx.moveTo(0, 28);
      ctx.bezierCurveTo(-45, -8, -30, -42, 0, -22);
      ctx.bezierCurveTo(30, -42, 45, -8, 0, 28);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      break;
    }
    case 'sparkle': {
      ctx.beginPath();
      ctx.moveTo(cx, cy - r);
      ctx.quadraticCurveTo(cx + r * 0.14, cy - r * 0.14, cx + r, cy);
      ctx.quadraticCurveTo(cx + r * 0.14, cy + r * 0.14, cx, cy + r);
      ctx.quadraticCurveTo(cx - r * 0.14, cy + r * 0.14, cx - r, cy);
      ctx.quadraticCurveTo(cx - r * 0.14, cy - r * 0.14, cx, cy - r);
      ctx.closePath();
      ctx.fill();
          // 小十字高光
      starPath(ctx, cx, cy, r * 0.22, r * 0.05, 4, 0);
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fill();
      break;
    }
    case 'bubble': {
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.globalAlpha = 0.72;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = Math.max(1, size * 0.018);
      ctx.beginPath();
      ctx.arc(cx - r * 0.28, cy - r * 0.32, r * 0.42, Math.PI * 0.85, Math.PI * 1.65);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx - r * 0.38, cy - r * 0.18, r * 0.1, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.fill();
      break;
    }
    case 'text': {
      const content = options.text?.trim() || '你好';
      const fontSize = Math.max(12, size * 0.48);
      ctx.font = `700 ${fontSize}px "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(content, cx, cy, size * 0.92);
      break;
    }
  }

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('canvas.toBlob 返回空'));
    }, 'image/png');
  });
}

function ElementsPanel({ store, state }: { store: EditorStore; state: EditorState }) {
  const scene = state.scene!;
  const [shape, setShape] = useState<ShapeKind>('star');
  const [color, setColor] = useState('#ffd45e');
  const [size, setSize] = useState(256);
  const [text, setText] = useState('你好');
  const [attach, setAttach] = useState(false);
  const [gradientA, setGradientA] = useState('#2a3d6b');
  const [gradientB, setGradientB] = useState('#b44c8c');
  const [preset, setPreset] = useState<AnimationPreset>('float');
  const [presetFrames, setPresetFrames] = useState(90);
  const [busy, setBusy] = useState(false);
  const selected = store.selectedLayer();

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      store.notify(String((e as Error)?.message ?? e), 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel-section">
      <h4>生成元素</h4>
      <div className="field-grid">
        <label>
          形状
          <select value={shape} onChange={(e) => setShape(e.target.value as ShapeKind)}>
            {SHAPE_LABELS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          颜色
          <input type="color" className="color-input" value={color} onChange={(e) => setColor(e.target.value)} />
        </label>
        <label>
          尺寸(px)
          <input type="number" className="num-input" min={16} max={2048} step={1} value={size} onChange={(e) => setSize(Number(e.target.value))} />
        </label>
        {shape === 'text' && (
          <label>
            文字
            <input className="text-input" value={text} onChange={(e) => setText(e.target.value)} />
          </label>
        )}
      </div>
      <label className="inline">
        <input type="checkbox" checked={attach} onChange={(e) => setAttach(e.target.checked)} />
        跟随选中角色（绑定到模型图层）
      </label>
      <div className="field-row">
        <button
          type="button"
          className="primary"
          disabled={busy}
          onClick={() =>
            void run(async () => {
              const blob = await generateShapePng({ shape, color, size, text });
              const label = SHAPE_LABELS.find((s) => s.value === shape)?.label ?? shape;
              // 上传文件名用 ASCII，图层显示名保留中文（避免素材目录出现一堆 asset.png）。
              await store.addGeneratedImage(blob, `element-${shape}`, {
                attachToSelected: attach,
                name: `元素-${label}`,
              });
            })
          }
        >
          生成并添加
        </button>
      </div>

      <h4>渐变背景</h4>
      <div className="field-grid">
        <label>
          起始色
          <input type="color" className="color-input" value={gradientA} onChange={(e) => setGradientA(e.target.value)} />
        </label>
        <label>
          结束色
          <input type="color" className="color-input" value={gradientB} onChange={(e) => setGradientB(e.target.value)} />
        </label>
      </div>
      <button
        type="button"
        disabled={busy}
        onClick={() =>
          void run(async () => {
            const W = scene.composition.width;
            const H = scene.composition.height;
            const canvas = document.createElement('canvas');
            canvas.width = W;
            canvas.height = H;
            const ctx = canvas.getContext('2d');
            if (!ctx) throw new Error('无法创建 2D 画布');
            const gradient = ctx.createLinearGradient(0, 0, W, H);
            gradient.addColorStop(0, gradientA);
            gradient.addColorStop(1, gradientB);
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, W, H);
            const blob = await new Promise<Blob>((resolve, reject) => {
              canvas.toBlob((result) => (result ? resolve(result) : reject(new Error('canvas.toBlob 返回空'))), 'image/png');
            });
            await store.addGeneratedImage(blob, 'gradient-background', { asBackground: true, name: '渐变背景' });
          })
        }
      >
        添加渐变背景
      </button>
      <div className="small muted">生成 {scene.composition.width}×{scene.composition.height} 的渐变 PNG，插入到图层最底部作为背景。</div>

      <h4>快速动画</h4>
      <div className="field-grid">
        <label>
          预设
          <select value={preset} onChange={(e) => setPreset(e.target.value as AnimationPreset)}>
            {QUICK_PRESETS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          时长(帧)
          <input
            type="number"
            className="num-input"
            min={1}
            max={600}
            step={1}
            value={presetFrames}
            onChange={(e) => setPresetFrames(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="field-row">
        <button
          type="button"
          disabled={!selected || busy}
          title={selected ? `应用到 ${selected.name ?? selected.id}` : '先选中一个图层'}
          onClick={() =>
            selected &&
            void store.addAnimationPreset(preset, selected.id, Math.max(1, Math.round(presetFrames)))
          }
        >
          应用到选中图层
        </button>
      </div>
      <div className="small muted">
        {selected ? `当前选中：${selected.name ?? selected.id}` : '当前没有选中图层。先在左侧图层列表或舞台上点选。'}
      </div>
      <div className="small muted panel-note">
        生成元素是普通 ImageLayer，可以在右侧逐项调变换、透明度和关键帧；保存时素材随场景一起写入。
      </div>
    </div>
  );
}
