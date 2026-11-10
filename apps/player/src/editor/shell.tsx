/*
 * 编辑器主布局：顶栏、左侧面板、中央舞台、右侧检查器、底部时间线。
 *
 * 舞台上的包围盒/拖动都基于场景规范里的变换公式（与 runtime 的 sampleTransform / cameraMatrix 一致）：
 *   world = [x,y] + R(rotation)·diag(scale·scaleX·flip, scale·scaleY)·(p − anchor)
 *   screen = [W/2,H/2] + zoom·R(rot)·(world − [x,y])
 * 拖动只改 transform.x/y，写入一条可合并的历史记录（drag-layer:<id>）。
 */
import { sample } from '@charaforge/scene';
import type { CanvasInfo } from '@charaforge/runtime';
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type PointerEvent as ReactPointerEvent } from 'react';
import { timecode } from '../util.ts';
import { Timeline } from './timeline.tsx';
import type { EditorState } from './contracts.ts';
import { Panels } from './panels.tsx';
import { Inspector } from './inspector.tsx';
import type { EditorStore } from './store.ts';
import type { EasePreset, ImageLayer, Layer, ModelLayer, Scene } from '@charaforge/scene';

type Mat = readonly [number, number, number, number, number, number];
const IDENTITY: Mat = [1, 0, 0, 1, 0, 0];

function mulMat(m1: Mat, m2: Mat): Mat {
  return [
    m1[0] * m2[0] + m1[2] * m2[1],
    m1[1] * m2[0] + m1[3] * m2[1],
    m1[0] * m2[2] + m1[2] * m2[3],
    m1[1] * m2[2] + m1[3] * m2[3],
    m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
    m1[1] * m2[4] + m1[3] * m2[5] + m1[5],
  ];
}

function applyMat(m: Mat, x: number, y: number): [number, number] {
  return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
}

function invertMat(m: Mat): Mat {
  const det = m[0] * m[3] - m[1] * m[2];
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return IDENTITY;
  const a = m[3] / det;
  const b = -m[1] / det;
  const c = -m[2] / det;
  const d = m[0] / det;
  return [a, b, c, d, -(a * m[4] + c * m[5]), -(b * m[4] + d * m[5])];
}

function translateMat(x: number, y: number): Mat {
  return [1, 0, 0, 1, x, y];
}

function scaleMat(sx: number, sy: number): Mat {
  return [sx, 0, 0, sy, 0, 0];
}

function rotateMat(deg: number): Mat {
  if (deg === 0) return IDENTITY;
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  return [c, s, -s, c, 0, 0];
}

function transformMatrix(
  t: {
    x: number;
    y: number;
    scale: number;
    scaleX: number;
    scaleY: number;
    rotation: number;
    anchorX: number;
    anchorY: number;
    flipX: boolean;
  },
): Mat {
  const sx = t.scale * t.scaleX * (t.flipX ? -1 : 1);
  const sy = t.scale * t.scaleY;
  return mulMat(
    translateMat(t.x, t.y),
    mulMat(rotateMat(t.rotation), mulMat(scaleMat(sx, sy), translateMat(-t.anchorX, -t.anchorY))),
  );
}

function cameraMatrix(camera: { x: number; y: number; zoom: number; rotation: number }, W: number, H: number): Mat {
  return mulMat(
    translateMat(W / 2, H / 2),
    mulMat(rotateMat(camera.rotation), mulMat(scaleMat(camera.zoom, camera.zoom), translateMat(-camera.x, -camera.y))),
  );
}

function sampleTransformMatrix(
  transform: Layer['transform'],
  frame: number,
  localWidth: number,
  localHeight: number,
): Mat {
  return transformMatrix({
    x: sample(transform.x, frame, 0),
    y: sample(transform.y, frame, 0),
    scale: sample(transform.scale, frame, 1),
    scaleX: sample(transform.scaleX, frame, 1),
    scaleY: sample(transform.scaleY, frame, 1),
    rotation: sample(transform.rotation, frame, 0),
    anchorX: transform.anchorX ?? localWidth / 2,
    anchorY: transform.anchorY ?? localHeight / 2,
    flipX: transform.flipX === true,
  });
}

function sampledCamera(scene: Scene, frame: number): { x: number; y: number; zoom: number; rotation: number } | null {
  if (!scene.camera) return null;
  return {
    x: sample(scene.camera.x, frame, 0),
    y: sample(scene.camera.y, frame, 0),
    zoom: sample(scene.camera.zoom, frame, 1),
    rotation: sample(scene.camera.rotation, frame, 0),
  };
}

interface Geometry {
  matrix: Mat;
  width: number;
  height: number;
}

function imageLocalSize(scene: Scene, layer: ImageLayer): { width: number; height: number } | null {
  const asset = scene.assets.images?.find((item) => item.id === layer.image);
  return asset ? { width: asset.width, height: asset.height } : null;
}

function layerGeometry(
  layer: Layer,
  scene: Scene,
  frame: number,
  store: EditorStore,
): Geometry | null {
  if (!(layer.in <= frame && frame < layer.out)) return null;
  if (sample(layer.opacity, frame, 1) <= 0) return null;
  const W = scene.composition.width;
  const H = scene.composition.height;
  const cam = sampledCamera(scene, frame);
  const camM = cam ? cameraMatrix(cam, W, H) : IDENTITY;
  if (layer.kind === 'model') {
    const canvas: CanvasInfo | null = store.modelCanvas(layer.id);
    if (!canvas) return null;
    const local = sampleTransformMatrix(layer.transform, frame, canvas.width, canvas.height);
    return { matrix: (layer.space ?? 'world') === 'screen' ? local : mulMat(camM, local), width: canvas.width, height: canvas.height };
  }
  const size = imageLocalSize(scene, layer);
  if (!size) return null;
  const own = sampleTransformMatrix(layer.transform, frame, size.width, size.height);
  if (layer.attach) {
    const target = scene.layers.find((item): item is ModelLayer => item.kind === 'model' && item.id === layer.attach!.layer);
    if (!target) return null;
    const targetCanvas = store.modelCanvas(target.id);
    if (!targetCanvas) return null;
    if (!(target.in <= frame && frame < target.out)) return null;
    const targetLocal = sampleTransformMatrix(target.transform, frame, targetCanvas.width, targetCanvas.height);
    const targetScreen = (target.space ?? 'world') === 'screen' ? targetLocal : mulMat(camM, targetLocal);
    const rotateWith = layer.attach.rotateWith ?? true;
    const scaleWith = layer.attach.scaleWith ?? true;
    let mt: Mat;
    if (rotateWith && scaleWith) {
      mt = targetScreen;
    } else {
      const t = {
        x: sample(target.transform.x, frame, 0),
        y: sample(target.transform.y, frame, 0),
        scale: 1,
        scaleX: scaleWith ? sample(target.transform.scale, frame, 1) * sample(target.transform.scaleX, frame, 1) * (target.transform.flipX ? -1 : 1) : 1,
        scaleY: scaleWith ? sample(target.transform.scale, frame, 1) * sample(target.transform.scaleY, frame, 1) : 1,
        rotation: rotateWith ? sample(target.transform.rotation, frame, 0) : 0,
        anchorX: target.transform.anchorX ?? targetCanvas.width / 2,
        anchorY: target.transform.anchorY ?? targetCanvas.height / 2,
        flipX: false,
      };
      const localTarget = transformMatrix(t);
      mt = (target.space ?? 'world') === 'screen' ? localTarget : mulMat(camM, localTarget);
    }
    const anchor = layer.attach.anchor;
    const offset = layer.attach.offset ?? [0, 0];
    return {
      matrix: mulMat(mulMat(mt, translateMat(anchor[0] * targetCanvas.width + offset[0], anchor[1] * targetCanvas.height + offset[1])), own),
      width: size.width,
      height: size.height,
    };
  }
  return {
    matrix: (layer.space ?? 'world') === 'screen' ? own : mulMat(camM, own),
    width: size.width,
    height: size.height,
  };
}

function polygonPoints(geometry: Geometry): string {
  const corners: [number, number][] = [
    [0, 0],
    [geometry.width, 0],
    [geometry.width, geometry.height],
    [0, geometry.height],
  ];
  return corners.map(([x, y]) => applyMat(geometry.matrix, x, y).join(',')).join(' ');
}

function useEditorState(store: EditorStore): EditorState {
  return useSyncExternalStore(
    (listener) => store.subscribe(listener),
    () => store.getState(),
    () => store.getState(),
  );
}

interface ShellProps {
  store: EditorStore;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export function EditorShell({ store, canvasRef }: ShellProps) {
  const state = useEditorState(store);
  const [scenePath, setScenePath] = useState(state.scenePath);
  const [savePath, setSavePath] = useState(state.savePath);
  const [newKeyEase, setNewKeyEase] = useState<EasePreset>('smooth');
  const [exportOpen, setExportOpen] = useState(false);
  const stageAreaRef = useRef<HTMLDivElement>(null);
  const stageFrameRef = useRef<HTMLDivElement>(null);
  const [stageSize, setStageSize] = useState({ width: 800, height: 450 });
  const dragRef = useRef<{
    layerId: string;
    startWorld: { x: number; y: number };
    startX: number;
    startY: number;
    pending: { x: number; y: number } | null;
    raf: number | null;
    pointerId: number;
  } | null>(null);

  useEffect(() => {
    setScenePath(state.scenePath);
    setSavePath(state.savePath);
  }, [state.scenePath, state.savePath]);

  useEffect(() => {
    const element = stageAreaRef.current;
    if (!element) return;
    const measure = () => {
      const rect = element.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) setStageSize({ width: rect.width, height: rect.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const scene = state.scene;
  const W = scene?.composition.width ?? 1280;
  const H = scene?.composition.height ?? 720;
  const duration = scene?.duration ?? 1;
  const fps = scene?.timebase.fps ?? 60;
  const frame = Math.min(state.frame, Math.max(0, duration - 1));
  const aspect = W / H;
  const maxWidth = Math.max(80, stageSize.width - 24);
  const maxHeight = Math.max(60, stageSize.height - 24);
  const frameWidth = Math.min(maxWidth, maxHeight * aspect);
  const frameHeight = frameWidth / aspect;

  const selectedGeometry = useMemo(() => {
    const selection = state.selection;
    if (!scene || selection.kind !== 'layer') return null;
    const layer = scene.layers.find((item) => item.id === selection.id);
    return layer ? layerGeometry(layer, scene, frame, store) : null;
    // state.scene 原地修改；state 对象每次 notify 都会变，因此这里能拿到最新曲线。
  }, [scene, state, frame, store]);

  const selection = state.selection;
  const selectedLayer = selection.kind === 'layer' ? scene?.layers.find((item) => item.id === selection.id) ?? null : null;
  const selectedAnchor =
    selectedGeometry && selectedLayer
      ? applyMat(
          selectedGeometry.matrix,
          selectedLayer.transform.anchorX ?? selectedGeometry.width / 2,
          selectedLayer.transform.anchorY ?? selectedGeometry.height / 2,
        )
      : null;
  const loading = state.phase !== 'ready' || !state.status?.ready;

  function toComposition(event: ReactPointerEvent): { x: number; y: number; rect: DOMRect } | null {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: ((event.clientX - rect.left) / rect.width) * W,
      y: ((event.clientY - rect.top) / rect.height) * H,
      rect,
    };
  }

  function hitTest(pointX: number, pointY: number): Layer | null {
    if (!scene) return null;
    for (let i = scene.layers.length - 1; i >= 0; i--) {
      const layer = scene.layers[i]!;
      const geometry = layerGeometry(layer, scene, frame, store);
      if (!geometry) continue;
      const [lx, ly] = applyMat(invertMat(geometry.matrix), pointX, pointY);
      if (lx >= 0 && ly >= 0 && lx <= geometry.width && ly <= geometry.height) return layer;
    }
    return null;
  }

  function worldPoint(layer: Layer, pointX: number, pointY: number): { x: number; y: number } {
    if ((layer.space ?? 'world') === 'screen') return { x: pointX, y: pointY };
    const cam = scene ? sampledCamera(scene, frame) : null;
    if (!cam) return { x: pointX, y: pointY };
    const inv = invertMat(cameraMatrix(cam, W, H));
    const [x, y] = applyMat(inv, pointX, pointY);
    return { x, y };
  }

  function dragCommands(layerId: string, x: number, y: number): Parameters<EditorStore['apply']>[0] {
    const layer = scene?.layers.find((item) => item.id === layerId);
    if (!layer) return [];
    const target = (path: string): { kind: 'layer'; id: string; path: string } => ({ kind: 'layer', id: layerId, path });
    const xCurve = typeof layer.transform.x !== 'number';
    const yCurve = typeof layer.transform.y !== 'number';
    if (!xCurve && !yCurve) {
      return [{ op: 'updateLayer', id: layerId, patch: { transform: { x, y } } }];
    }
    const commands: Parameters<EditorStore['apply']>[0] = [];
    if (Array.isArray(commands)) {
      if (xCurve) commands.push({ op: 'addKeyframe', target: target('transform.x'), t: store.getState().frame, v: x, ease: 'smooth' });
      if (yCurve) commands.push({ op: 'addKeyframe', target: target('transform.y'), t: store.getState().frame, v: y, ease: 'smooth' });
    }
    return commands;
  }

  function applyDragPosition(layerId: string, x: number, y: number) {
    const commands = dragCommands(layerId, x, y);
    if (!Array.isArray(commands) || commands.length === 0) return;
    void store.applyUi(commands, {
      history: 'coalesce',
      label: `drag-layer:${layerId}`,
      validate: 'none',
      frame: store.getState().frame,
    });
  }

  function scheduleDrag() {
    const drag = dragRef.current;
    if (!drag || drag.raf !== null) return;
    drag.raf = requestAnimationFrame(() => {
      const current = dragRef.current;
      if (!current) return;
      current.raf = null;
      const pending = current.pending;
      if (!pending) return;
      current.pending = null;
      applyDragPosition(current.layerId, pending.x, pending.y);
    });
  }

  function onStagePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!scene || event.button !== 0) return;
    const point = toComposition(event);
    if (!point) return;
    const layer = hitTest(point.x, point.y);
    if (!layer) {
      store.select({ kind: 'scene' });
      return;
    }
    store.select({ kind: 'layer', id: layer.id });
    if (store.getState().playing) store.pause();
    if (layer.kind === 'image' && layer.attach) return;
    const startX = sample(layer.transform.x, frame, 0);
    const startY = sample(layer.transform.y, frame, 0);
    dragRef.current = {
      layerId: layer.id,
      startWorld: worldPoint(layer, point.x, point.y),
      startX,
      startY,
      pending: null,
      raf: null,
      pointerId: event.pointerId,
    };
    stageFrameRef.current?.setPointerCapture(event.pointerId);
    store.setDragging(true);
    event.preventDefault();
  }

  function onStagePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || !scene) return;
    const point = toComposition(event);
    const layer = scene.layers.find((item) => item.id === drag.layerId);
    if (!point || !layer) return;
    const world = worldPoint(layer, point.x, point.y);
    drag.pending = {
      x: drag.startX + (world.x - drag.startWorld.x),
      y: drag.startY + (world.y - drag.startWorld.y),
    };
    scheduleDrag();
  }

  function finishDrag(event?: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.raf !== null) cancelAnimationFrame(drag.raf);
    drag.raf = null;
    const pending = drag.pending;
    if (pending) {
      drag.pending = null;
      applyDragPosition(drag.layerId, pending.x, pending.y);
    }
    dragRef.current = null;
    if (event) {
      try {
        stageFrameRef.current?.releasePointerCapture(event.pointerId);
      } catch {
        /* 指针已经释放 */
      }
    }
    store.setDragging(false);
  }

  // ---------- 键盘 ----------

  useEffect(() => {
    const isTyping = (target: EventTarget | null): boolean => {
      if (!(target instanceof HTMLElement)) return false;
      if (target.isContentEditable) return true;
      if (target instanceof HTMLTextAreaElement) return true;
      if (target instanceof HTMLSelectElement) return true;
      if (target instanceof HTMLInputElement) {
        return !['range', 'checkbox', 'radio', 'button', 'submit', 'color'].includes(target.type);
      }
      return false;
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const current = store.getState();
      const typing = isTyping(event.target);
      const mod = event.ctrlKey || event.metaKey;
      if (mod) {
        if (event.key.toLowerCase() === 's') {
          event.preventDefault();
          void store.save();
          return;
        }
        if (typing) return;
        if (event.key.toLowerCase() === 'z') {
          event.preventDefault();
          if (event.shiftKey) void store.redo().catch(() => {});
          else void store.undo().catch(() => {});
          return;
        }
        if (event.key.toLowerCase() === 'y') {
          event.preventDefault();
          void store.redo().catch(() => {});
          return;
        }
        return;
      }
      if (typing) return;
      const step = event.shiftKey ? 10 : 1;
      if (event.key === ' ') {
        event.preventDefault();
        void store.togglePlay().catch(() => {});
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        store.step(step);
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault();
        store.step(-step);
      } else if (event.key === 'Home') {
        event.preventDefault();
        store.seek(0);
      } else if (event.key === 'End') {
        event.preventDefault();
        store.seek(current.scene ? current.scene.duration - 1 : 0);
      } else if (event.key === 'Delete' || event.key === 'Backspace') {
        if (!current.selectedKeys.length) return;
        event.preventDefault();
        const seen = new Set<string>();
        const commands: Parameters<EditorStore['apply']>[0] = [];
        for (const key of current.selectedKeys) {
          const target = key.target;
          const id =
            target.kind === 'layer'
              ? `layer:${target.id}:${target.path}:${key.t}`
              : target.kind === 'camera'
                ? `camera:${target.path}:${key.t}`
                : `param:${target.id}:${target.param}:${key.t}`;
          if (seen.has(id)) continue;
          seen.add(id);
          if (Array.isArray(commands)) commands.push({ op: 'removeKeyframe', target, t: key.t });
          else break;
        }
        if (Array.isArray(commands) && commands.length) {
          void store.applyUi(commands, { label: '删除关键帧' }).then((result) => {
            if (result) store.setSelectedKeys([]);
          });
        }
      } else if (event.key.toLowerCase() === 'k') {
        const first = current.selectedKeys[0];
        if (first) {
          event.preventDefault();
          void store.addKeyAtPlayhead(first.target);
        }
      } else if (event.key.toLowerCase() === 'e') {
        if (current.scene) {
          event.preventDefault();
          setExportOpen(true);
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [store]);

  const canInteract = state.phase === 'ready' && !!scene;
  const playing = state.playing;
  const confirmDiscard = (): boolean => !state.dirty || window.confirm('当前有未保存修改，继续会丢失这些修改。确定继续？');
  const openScene = (): void => {
    if (!confirmDiscard()) return;
    void store.open(scenePath).catch(() => {});
  };
  const reloadScene = (): void => {
    if (!confirmDiscard()) return;
    void store.reload().catch(() => {});
  };
  // 独立播放器打开当前场景/帧；未保存修改不会带过去（先保存或导出）。
  const playerHref = scene
    ? `player.html?scene=${encodeURIComponent(scenePath)}&frame=${Math.max(0, Math.floor(state.frame))}`
    : 'player.html';

  return (
    <div className="editor-app" data-testid="editor-app">
      <header className="editor-topbar">
        <div className="topbar-row main">
          <div className="brand">
            <strong>CharaForge 场景编辑器</strong>
            <span className="muted small scene-title">{scene?.meta.title ?? '未加载场景'}</span>
          </div>
          <span className={`dirty-dot${state.dirty ? ' active' : ''}`} title={state.dirty ? '有未保存修改' : '已保存'}>
            {state.dirty ? '● 未保存' : '已保存'}
          </span>
          <span className="spacer" />
          <label className="path-field" title="场景仓库相对路径">
            <span>场景</span>
            <input
              data-testid="scene-path-input"
              value={scenePath}
              spellCheck={false}
              onChange={(e) => setScenePath(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') openScene();
              }}
            />
          </label>
          <button type="button" onClick={openScene} disabled={!canInteract}>
            打开
          </button>
          <button type="button" onClick={reloadScene} disabled={!canInteract}>
            刷新
          </button>
        </div>
        <div className="topbar-row sub">
          <label className="path-field" title="保存目标（仓库相对路径）">
            <span>保存到</span>
            <input
              data-testid="save-path-input"
              value={savePath}
              spellCheck={false}
              onChange={(e) => {
                setSavePath(e.target.value);
                store.setSavePath(e.target.value);
              }}
            />
          </label>
          <button
            type="button"
            className="primary"
            data-testid="save-button"
            disabled={!canInteract}
            title="Ctrl/Cmd+S"
            onClick={() => void store.save(savePath)}
          >
            保存
          </button>
          <button type="button" disabled={!state.canUndo} title="Ctrl/Cmd+Z" onClick={() => void store.undo().catch(() => {})}>
            撤销
          </button>
          <button type="button" disabled={!state.canRedo} title="Ctrl/Cmd+Shift+Z / Ctrl/Cmd+Y" onClick={() => void store.redo().catch(() => {})}>
            重做
          </button>
          <button type="button" className="primary" data-testid="export-button" disabled={!canInteract} title="E" onClick={() => setExportOpen(true)}>
            导出视频…
          </button>
          <a
            className="toolbar-link"
            href={playerHref}
            target="_blank"
            rel="noreferrer"
            title="在独立播放器中打开当前场景/帧（未保存修改不会带过去）"
          >
            独立播放器 ↗
          </a>
          <a
            className="toolbar-link"
            href="motions.html"
            target="_blank"
            rel="noreferrer"
            title="独立打开动作 / 表情素材查看器（含缩略图、图集、循环预览）"
          >
            素材 / 动作查看器 ↗
          </a>
          <span className="ai-badge" title="在浏览器控制台或 Playwright 里调用 window.__editor 操作场景">
            AI: window.__editor
          </span>
          <span className="spacer" />
          <span className="topbar-keys small muted">空格 播放 · Ctrl+S 保存 · Ctrl+Z 撤销 · E 导出</span>
        </div>
      </header>

      <div className="editor-body">
        {scene ? (
          <Panels store={store} state={state} />
        ) : (
          <aside className="editor-panel editor-left">
            <div className="tool-tabs">
              <button type="button" className="active">
                图层
              </button>
            </div>
            <div className="panel-scroll">
              <div className="empty">正在加载场景与图层…</div>
            </div>
          </aside>
        )}

        <main className="editor-center">
          <div className="stage-area" ref={stageAreaRef}>
            <div
              className={`stage-frame${state.dragging ? ' dragging' : ''}`}
              ref={stageFrameRef}
              style={{ width: `${Math.round(frameWidth)}px`, height: `${Math.round(frameHeight)}px` }}
              onPointerDown={onStagePointerDown}
              onPointerMove={onStagePointerMove}
              onPointerUp={finishDrag}
              onPointerCancel={finishDrag}
            >
              <canvas ref={canvasRef} data-testid="stage-canvas" className="stage-canvas" />
              {selectedGeometry && selectedLayer && (
                <svg className="selection-overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
                  <polygon points={polygonPoints(selectedGeometry)} />
                  {selectedAnchor && <circle cx={selectedAnchor[0]} cy={selectedAnchor[1]} r={6} />}
                </svg>
              )}
              {selectedLayer && (
                <div className="stage-selection-label">
                  {selectedLayer.name ?? selectedLayer.id}
                  {selectedLayer.kind === 'image' && selectedLayer.attach ? '（跟随）' : ''}
                </div>
              )}
              {(loading || state.error) && (
                <div className="stage-overlay">
                  {state.error ? (
                    <div className="error-box" role="alert">
                      <strong>编辑器启动 / 加载失败</strong>
                      <pre>{state.error}</pre>
                      <div className="small">检查 pnpm run setup、控制台网络请求和场景 JSON。</div>
                    </div>
                  ) : (
                    <>
                      <div className="spinner" />
                      <div>{state.stage}…</div>
                      <div className="small muted">
                        {state.status?.prewarm && state.status.prewarm.total > 0
                          ? `预热 ${state.status.prewarm.done}/${state.status.prewarm.total}`
                          : state.status?.waitMs
                            ? `补算物理 ${(state.status.waitMs / 1000).toFixed(1)} s`
                            : '首次加载大型场景可能需要数秒'}
                      </div>
                    </>
                  )}
                </div>
              )}
              <div className="stage-badge">
                {W}×{H} · {fps} fps · workers {store.engine.workers}
                {state.status && !loading ? ` · render ${state.status.renderMs.toFixed(1)} ms` : ''}
              </div>
            </div>
          </div>

          <div className="transport">
            <button
              type="button"
              className="primary"
              disabled={!canInteract}
              title="空格"
              onClick={() => void store.togglePlay().catch(() => {})}
            >
              {playing ? '❚❚ 暂停' : '▶ 播放'}
            </button>
            <button type="button" disabled={!canInteract} title="Home" onClick={() => store.seek(0)}>
              |◀
            </button>
            <button type="button" disabled={!canInteract} title="Shift+←" onClick={() => store.step(-10)}>
              −10
            </button>
            <button type="button" disabled={!canInteract} title="←" onClick={() => store.step(-1)}>
              −1
            </button>
            <button type="button" disabled={!canInteract} title="→" onClick={() => store.step(1)}>
              +1
            </button>
            <button type="button" disabled={!canInteract} title="Shift+→" onClick={() => store.step(10)}>
              +10
            </button>
            <button type="button" disabled={!canInteract} title="End" onClick={() => store.seek(duration - 1)}>
              ▶|
            </button>
            <input
              className="scrub"
              type="range"
              min={0}
              max={Math.max(0, duration - 1)}
              step={1}
              value={frame}
              disabled={!canInteract}
              onChange={(e) => store.seek(Number(e.target.value))}
            />
            <span className="mono frame-readout">
              帧 {frame} / {duration - 1}　{timecode(frame, fps)}
            </span>
            <button type="button" disabled={!canInteract} title="M" onClick={() => store.engine.setMuted(!(state.status?.muted ?? false))}>
              {state.status?.muted ? '🔇 静音' : '🔊 声音'}
            </button>
            <label className="inline small" title="时间线显示参数轨">
              <input type="checkbox" checked={state.showParams} onChange={(e) => store.setShowParams(e.target.checked)} />
              参数轨
            </label>
            <span className="small muted keys-hint">
              空格 播放 · ←/→ ±1 · Shift ±10 · Home/End · Ctrl+Z 撤销 · Ctrl+S 保存 · Del 删除选中键 · K 加键 · E 导出
            </span>
          </div>
        </main>

        <Inspector store={store} state={state} newKeyEase={newKeyEase} setNewKeyEase={setNewKeyEase} />
      </div>

      <div className="editor-timeline">
        {scene ? <Timeline store={store} /> : <div className="timeline-placeholder">时间线：等待场景加载…</div>}
      </div>

      <NoticeStack store={store} state={state} />

      {exportOpen && scene && (
        <ExportDialog
          store={store}
          state={state}
          duration={duration}
          onClose={() => setExportOpen(false)}
        />
      )}
    </div>
  );
}

// ---------- 通知 ----------

function NoticeStack({ store, state }: { store: EditorStore; state: EditorState }) {
  return (
    <div className="notice-stack" aria-live="polite">
      {state.notices.map((notice) => (
        <div key={notice.id} className={`notice ${notice.kind}`} role={notice.kind === 'error' ? 'alert' : 'status'}>
          <span>{notice.text}</span>
          <button type="button" className="mini" onClick={() => store.dismissNotice(notice.id)}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

// ---------- 导出 ----------

function ExportDialog({
  store,
  state,
  duration,
  onClose,
}: {
  store: EditorStore;
  state: EditorState;
  duration: number;
  onClose: () => void;
}) {
  const scene = state.scene!;
  const [from, setFrom] = useState(0);
  const [to, setTo] = useState(duration);
  const [format, setFormat] = useState<'auto' | 'mp4' | 'webm'>('auto');
  const [quality, setQuality] = useState<'high' | 'medium' | 'low'>('high');
  const [filename, setFilename] = useState(`${scene.meta.title || 'charaforge'}.mp4`);
  const [background, setBackground] = useState(
    scene.composition.background && /^#[0-9a-f]{6}$/i.test(scene.composition.background) ? scene.composition.background : '#000000',
  );
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now());
  const exporting = state.exporting;
  useEffect(() => {
    if (!exporting) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [exporting]);

  const start = async () => {
    setBusy(true);
    try {
      await store.exportVideo({
        from: Number(from) || 0,
        to: Number(to) || duration,
        format,
        quality,
        filename: filename.trim() || undefined,
        background,
      });
      onClose();
    } catch {
      /* 错误已进 notices */
    } finally {
      setBusy(false);
    }
  };

  const elapsed = exporting ? Math.max(0, (now - exporting.startedAt) / 1000) : 0;
  const pct = exporting && exporting.total > 0 ? Math.min(100, Math.floor((exporting.frame / exporting.total) * 100)) : 0;
  const phaseLabel = exporting ? (exporting.phase === 'audio' ? '混音 / 准备' : exporting.phase === 'video' ? '逐帧渲染' : '封装') : '';

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={() => !exporting && !busy && onClose()}>
      <div className="modal export-modal" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <strong>导出视频</strong>
          <span className="spacer" />
          <button type="button" className="mini" disabled={busy || !!exporting} onClick={onClose}>
            关闭
          </button>
        </header>
        <div className="modal-body">
          <div className="field-grid">
            <label>
              起始帧
              <input
                type="number"
                className="num-input"
                min={0}
                max={duration - 1}
                value={from}
                disabled={busy || !!exporting}
                onChange={(e) => setFrom(Number(e.target.value))}
              />
            </label>
            <label>
              结束帧（不含）
              <input
                type="number"
                className="num-input"
                min={1}
                max={duration}
                value={to}
                disabled={busy || !!exporting}
                onChange={(e) => setTo(Number(e.target.value))}
              />
            </label>
            <label>
              容器
              <select value={format} disabled={busy || !!exporting} onChange={(e) => setFormat(e.target.value as typeof format)}>
                <option value="auto">自动（MP4 优先）</option>
                <option value="mp4">MP4</option>
                <option value="webm">WebM</option>
              </select>
            </label>
            <label>
              质量
              <select value={quality} disabled={busy || !!exporting} onChange={(e) => setQuality(e.target.value as typeof quality)}>
                <option value="high">高</option>
                <option value="medium">中</option>
                <option value="low">低</option>
              </select>
            </label>
            <label className="wide">
              文件名
              <input className="text-input" value={filename} disabled={busy || !!exporting} onChange={(e) => setFilename(e.target.value)} />
            </label>
            <label>
              背景
              <input type="color" className="color-input" value={background} disabled={busy || !!exporting} onChange={(e) => setBackground(e.target.value)} />
            </label>
          </div>

          {exporting && (
            <div className="export-progress">
              <div className="progress-head">
                <strong>{phaseLabel}</strong>
                <span className="spacer" />
                <span className="mono">
                  {exporting.frame}/{exporting.total} · {pct}% · {elapsed.toFixed(1)} s
                </span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
              <div className="small muted">
                导出期间会暂停播放并逐帧渲染；完成前请不要关闭页面。输出尺寸 {scene.composition.width}×{scene.composition.height}。
              </div>
            </div>
          )}

          <div className="small muted">
            导出使用浏览器 WebCodecs（mediabunny），没有可用编码器时会明确报错。范围外不会写入音频/视频。
          </div>
        </div>
        <footer className="modal-foot">
          <span className="small muted">
            帧 {from}–{Math.max(from + 1, to)} / {duration}
          </span>
          <span className="spacer" />
          <button type="button" disabled={busy || !!exporting} onClick={onClose}>
            取消
          </button>
          <button type="button" className="primary" disabled={busy || !!exporting || to <= from} onClick={() => void start()}>
            {exporting ? '导出中…' : '开始导出'}
          </button>
        </footer>
      </div>
    </div>
  );
}

