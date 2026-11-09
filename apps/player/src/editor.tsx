/*
 * 最小饰品编辑器（?edit=1）。
 *
 * 复用播放器的 runtime / renderFrame：所有修改都直接改当前 scene 里的 ImageLayer 对象，
 * 调 engine.redraw() 让下一帧重新合成，不重载模型。
 *
 * 保存：
 *   - 优先 File System Access API：用户选择仓库根目录，写 scene 指定的 json + assets/accessories/；
 *   - 不支持时降级为下载 scene.json 和图片文件。
 * 图片导入先用 blob URL 预览；保存时换成 assets/accessories/ 下的相对路径，刷新即可重新加载。
 */
import { MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, parseScene, type ImageAsset, type ImageLayer, type Scene } from '@charaforge/scene';
import { useReducer, useRef, useState } from 'react';
import type { PlayerEngine } from './engine.ts';

interface Props {
  engine: PlayerEngine;
  /** 当前场景的仓库相对路径；缺省为示例场景 */
  defaultSavePath: string;
}

interface FsWritable {
  write(data: Blob | string): Promise<void>;
  close(): Promise<void>;
}
interface FsFileHandle {
  createWritable(): Promise<FsWritable>;
}
interface FsDirHandle {
  getDirectoryHandle(name: string, opts?: { create?: boolean }): Promise<FsDirHandle>;
  getFileHandle(name: string, opts?: { create?: boolean }): Promise<FsFileHandle>;
}
type DirectoryPicker = (opts?: { mode?: 'read' | 'readwrite' }) => Promise<FsDirHandle>;

function sanitizeId(name: string): string {
  const out = name.replace(/[^A-Za-z0-9_.:-]+/g, '-').replace(/^-+|-+$/g, '');
  return out || 'accessory';
}

function safeFileName(id: string): string {
  const out = id.replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^-+|-+$/g, '');
  return out || 'accessory';
}

function uniqueId(base: string, scene: Scene): string {
  const used = new Set([...(scene.assets.images ?? []).map((a) => a.id), ...scene.layers.map((l) => l.id)]);
  let id = sanitizeId(base);
  let n = 2;
  while (used.has(id)) id = `${sanitizeId(base)}-${n++}`;
  return id;
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function isPng(b: Uint8Array): boolean {
  return b.length >= 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47 && b[4] === 0x0d && b[5] === 0x0a && b[6] === 0x1a && b[7] === 0x0a;
}
function isWebp(b: Uint8Array): boolean {
  return b.length >= 12 && b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46 && b[8] === 0x57 && b[9] === 0x45 && b[10] === 0x42 && b[11] === 0x50;
}

/** 只解码、不执行；返回尺寸和 alpha 统计，并拒绝没有透明像素的图。 */
async function inspectAccessory(file: File): Promise<{ width: number; height: number; nonOpaque: number; visible: number }> {
  const head = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  if (!isPng(head) && !isWebp(head)) throw new Error('只支持 PNG / WebP（按文件头判断）');
  const bmp = await createImageBitmap(file);
  try {
    if (bmp.width < 1 || bmp.height < 1 || bmp.width > MAX_IMAGE_SIDE || bmp.height > MAX_IMAGE_SIDE || bmp.width * bmp.height > MAX_IMAGE_PIXELS) {
      throw new Error(`图片尺寸超出限制：${bmp.width}×${bmp.height}`);
    }
    const c = document.createElement('canvas');
    c.width = bmp.width;
    c.height = bmp.height;
    const g = c.getContext('2d', { willReadFrequently: true });
    if (!g) throw new Error('无法创建 canvas 2d 上下文');
    g.clearRect(0, 0, c.width, c.height);
    g.drawImage(bmp, 0, 0);
    const data = g.getImageData(0, 0, c.width, c.height).data;
    let nonOpaque = 0;
    let visible = 0;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i]! < 255) nonOpaque++;
      if (data[i]! > 0) visible++;
    }
    if (visible === 0) throw new Error('图片没有任何可见像素');
    if (nonOpaque === 0) throw new Error('图片没有透明/半透明像素（alpha 全为 255）');
    return { width: bmp.width, height: bmp.height, nonOpaque, visible };
  } finally {
    bmp.close();
  }
}

async function sha256Hex(blob: Blob): Promise<string | undefined> {
  if (!globalThis.crypto?.subtle) return undefined;
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function downloadFile(name: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function normalizeScenePath(p: string): string {
  const s = p.trim().replace(/\\/g, '/').replace(/^\/+/, '');
  if (!s || s.split('/').some((x) => x === '' || x === '..')) throw new Error('保存路径非法');
  if (!s.endsWith('.json')) throw new Error('保存路径必须以 .json 结尾');
  return s;
}

async function writeRepoFile(root: FsDirHandle, rel: string, data: Blob | string): Promise<void> {
  const parts = rel.split('/').filter(Boolean);
  if (!parts.length || parts.some((x) => x === '..')) throw new Error(`写入路径非法：${rel}`);
  let dir = root;
  for (let i = 0; i < parts.length - 1; i++) dir = await dir.getDirectoryHandle(parts[i]!, { create: true });
  const handle = await dir.getFileHandle(parts[parts.length - 1]!, { create: true });
  const w = await handle.createWritable();
  await w.write(data);
  await w.close();
}

function range(min: number, max: number, step: number, value: number, onChange: (v: number) => void) {
  return (
    <>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <input type="number" min={min} max={max} step={step} value={+value.toFixed(3)} onChange={(e) => Number.isFinite(Number(e.target.value)) && onChange(Number(e.target.value))} />
    </>
  );
}

export function AccessoryEditor({ engine, defaultSavePath }: Props) {
  const sceneNow = engine.runtime.currentScene;
  const [, bump] = useReducer((x: number) => x + 1, 0);
  const firstModel = sceneNow?.layers.find((l) => l.kind === 'model')?.id ?? '';
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [attachTarget, setAttachTarget] = useState(firstModel);
  const [savePath, setSavePath] = useState(defaultSavePath);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const files = useRef(new Map<string, File>());
  const hiddenOpacity = useRef(new Map<string, number>());
  const dirHandle = useRef<FsDirHandle | null>(null);

  if (!sceneNow) return null;
  const scene = sceneNow;
  const imageLayers = scene.layers.filter((l): l is ImageLayer => l.kind === 'image');
  const modelLayers = scene.layers.filter((l) => l.kind === 'model');
  const assets = scene.assets.images ?? [];
  const assetById = new Map(assets.map((a) => [a.id, a]));
  const layer = imageLayers.find((l) => l.id === selectedId) ?? imageLayers[0];
  const asset = layer ? assetById.get(layer.image) : undefined;
  const targetCanvas = layer?.attach ? engine.runtime.layerInstance(layer.attach.layer)?.canvas : undefined;
  const xRange = Math.max(64, targetCanvas?.width ?? scene.composition.width);
  const yRange = Math.max(64, targetCanvas?.height ?? scene.composition.height);

  const touch = (fn: () => void) => {
    fn();
    bump();
    engine.redraw();
  };

  function setNumber(key: 'x' | 'y' | 'scale' | 'rotation', v: number) {
    if (!layer || !Number.isFinite(v)) return;
    touch(() => {
      (layer.transform as unknown as Record<string, number>)[key] = v;
    });
  }

  function setOpacity(v: number) {
    if (!layer || !Number.isFinite(v)) return;
    touch(() => {
      layer.opacity = Math.max(0, Math.min(1, v));
    });
  }

  function toggleVisible() {
    if (!layer) return;
    const cur = num(layer.opacity, 1);
    touch(() => {
      if (cur > 0) {
        hiddenOpacity.current.set(layer.id, cur);
        layer.opacity = 0;
      } else layer.opacity = hiddenOpacity.current.get(layer.id) ?? 1;
    });
  }

  function setAttach(id: string) {
    if (!layer) return;
    touch(() => {
      if (!id) {
        delete layer.attach;
        layer.space = 'screen';
        (layer.transform as unknown as Record<string, number>).x = scene.composition.width / 2;
        (layer.transform as unknown as Record<string, number>).y = scene.composition.height / 2;
      } else {
        layer.attach = {
          layer: id,
          anchor: layer.attach?.anchor ?? [0.5, 0.15],
          offset: layer.attach?.offset ?? [0, 0],
          rotateWith: layer.attach?.rotateWith ?? true,
          scaleWith: layer.attach?.scaleWith ?? true,
        };
        (layer.transform as unknown as Record<string, number>).x = 0;
        (layer.transform as unknown as Record<string, number>).y = 0;
      }
    });
  }

  function setAttachNumber(key: 'anchorX' | 'anchorY' | 'offsetX' | 'offsetY', v: number) {
    if (!layer?.attach || !Number.isFinite(v)) return;
    touch(() => {
      const a = layer.attach!;
      if (key === 'anchorX') a.anchor = [Math.max(0, Math.min(1, v)), a.anchor[1]];
      else if (key === 'anchorY') a.anchor = [a.anchor[0], Math.max(0, Math.min(1, v))];
      else if (key === 'offsetX') a.offset = [v, a.offset?.[1] ?? 0];
      else a.offset = [a.offset?.[0] ?? 0, v];
    });
  }

  async function onFile(file: File) {
    setError('');
    setStatus('');
    let previewUrl = '';
    try {
      const info = await inspectAccessory(file);
      const id = uniqueId(file.name.replace(/\.[^.]+$/, ''), scene);
      previewUrl = URL.createObjectURL(file);
      const sha256 = await sha256Hex(file);
      const a: ImageAsset = { id, uri: previewUrl, width: info.width, height: info.height, ...(sha256 ? { sha256 } : {}) };
      const defaultScale = Math.min(2, 1024 / Math.max(info.width, info.height));
      const tf = { x: 0, y: 0, scale: defaultScale, anchorX: info.width / 2, anchorY: info.height / 2 };
      const l: ImageLayer = {
        id,
        name: file.name.replace(/\.[^.]+$/, ''),
        kind: 'image',
        image: id,
        in: 0,
        out: scene.duration,
        ...(attachTarget
          ? { transform: tf, attach: { layer: attachTarget, anchor: [0.5, 0.15], offset: [0, 0], rotateWith: true, scaleWith: true } }
          : { space: 'screen' as const, transform: { ...tf, x: scene.composition.width / 2, y: scene.composition.height / 2 } }),
        opacity: 1,
        provenance: { source: 'import', tool: 'player-edit', detail: { file: file.name } },
      };
      await engine.runtime.addImageLayer(a, l);
      files.current.set(id, file);
      setSelectedId(id);
      engine.pause();
      engine.redraw();
      bump();
      setStatus(`已导入 ${file.name}（${info.width}×${info.height}）`);
    } catch (e) {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setError(String((e as Error)?.message ?? e));
    }
  }

  async function save() {
    setError('');
    setStatus('保存中…');
    try {
      const clone = structuredClone(scene) as Scene;
      const writes: { assetId: string; rel: string; uri: string; blob: File; sha256?: string }[] = [];
      for (const a of clone.assets.images ?? []) {
        const file = files.current.get(a.id);
        if (!file) continue;
        const ext = file.type === 'image/webp' || file.name.toLowerCase().endsWith('.webp') ? 'webp' : 'png';
        const name = `${safeFileName(a.id)}.${ext}`;
        const uri = `accessories/${name}`;
        const sha = await sha256Hex(file);
        a.uri = uri;
        if (sha) a.sha256 = sha;
        writes.push({ assetId: a.id, rel: `assets/accessories/${name}`, uri, blob: file, sha256: sha });
      }
      const rel = normalizeScenePath(savePath);
      const text = JSON.stringify(parseScene(clone), null, 2) + '\n';
      const picker = (window as unknown as { showDirectoryPicker?: DirectoryPicker }).showDirectoryPicker;
      if (picker) {
        let dir = dirHandle.current;
        if (!dir) {
          dir = await picker.call(window, { mode: 'readwrite' });
          dirHandle.current = dir;
        }
        try {
          for (const w of writes) await writeRepoFile(dir, w.rel, w.blob);
          await writeRepoFile(dir, rel, text);
        } catch (e) {
          dirHandle.current = null;
          throw e;
        }
        for (const w of writes) {
          const live = (scene.assets.images ?? []).find((a) => a.id === w.assetId);
          if (live) {
            live.uri = w.uri;
            if (w.sha256) live.sha256 = w.sha256;
          }
        }
        setStatus(`已保存 ${rel} 和 ${writes.length} 张图片`);
      } else {
        for (const w of writes) downloadFile(w.rel.split('/').pop()!, w.blob);
        downloadFile('scene.json', new Blob([text], { type: 'application/json' }));
        setStatus(`已下载 scene.json 和 ${writes.length} 张图片；请放到仓库对应目录后刷新`);
      }
      engine.redraw();
      bump();
    } catch (e) {
      const msg = String((e as Error)?.message ?? e);
      if (/AbortError|aborted|取消/i.test(msg)) setStatus('已取消保存');
      else setError(`保存失败：${msg}`);
    }
  }

  return (
    <div className="editor">
      <div className="row">
        <strong>饰品编辑</strong>
        <button onClick={() => fileInput.current?.click()}>导入图片</button>
        <input
          ref={fileInput}
          type="file"
          accept="image/png,image/webp"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.currentTarget.value = '';
            if (f) void onFile(f);
          }}
        />
        <span className="muted small">PNG/WebP，需透明背景；默认挂到第一个模型图层</span>
        <span className="muted small">保存到</span>
        <input className="save-path" value={savePath} onChange={(e) => setSavePath(e.target.value)} spellCheck={false} />
        <button className="primary" onClick={() => void save()}>
          保存
        </button>
      </div>
      {status && <div className="small ok">{status}</div>}
      {error && <div className="small warn">{error}</div>}
      {imageLayers.length > 0 ? (
        <div className="editor-body">
          <div className="editor-list">
            {imageLayers.map((l) => {
              const a = assetById.get(l.image);
              const thumb = a ? engine.runtime.resolveAsset(a.uri) : '';
              return (
                <button key={l.id} className={l.id === layer?.id ? 'item active' : 'item'} onClick={() => { setSelectedId(l.id); engine.pause(); engine.redraw(); }}>
                  {thumb && <img src={thumb} alt="" />}
                  <span>
                    <span>{l.name ?? l.id}</span>
                    <span className="muted small">
                      {l.id}
                      {l.attach ? ` → ${l.attach.layer}` : '（独立）'}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
          {layer && (
            <div className="editor-fields">
              <label>
                可见
                <input type="checkbox" checked={num(layer.opacity, 1) > 0} onChange={toggleVisible} />
              </label>
              <label>
                X
                {range(-xRange, xRange, 1, num((layer.transform as unknown as Record<string, unknown>).x, 0), (v) => setNumber('x', v))}
              </label>
              <label>
                Y
                {range(-yRange, yRange, 1, num((layer.transform as unknown as Record<string, unknown>).y, 0), (v) => setNumber('y', v))}
              </label>
              <label>
                缩放
                {range(0.01, 20, 0.01, num((layer.transform as unknown as Record<string, unknown>).scale, 1), (v) => setNumber('scale', v))}
              </label>
              <label>
                旋转
                {range(-180, 180, 1, num((layer.transform as unknown as Record<string, unknown>).rotation, 0), (v) => setNumber('rotation', v))}
              </label>
              <label>
                不透明度
                {range(0, 1, 0.01, num(layer.opacity, 1), setOpacity)}
              </label>
              <label>
                绑定
                <select value={layer.attach?.layer ?? ''} onChange={(e) => setAttach(e.target.value)}>
                  <option value="">不绑定（屏幕图层）</option>
                  {modelLayers.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name ?? m.id}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                导入默认绑定
                <select value={attachTarget} onChange={(e) => setAttachTarget(e.target.value)}>
                  <option value="">不绑定</option>
                  {modelLayers.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name ?? m.id}
                    </option>
                  ))}
                </select>
              </label>
              {layer.attach && (
                <>
                  <label>
                    anchor X
                    <input type="number" min={0} max={1} step={0.001} value={+layer.attach.anchor[0].toFixed(3)} onChange={(e) => setAttachNumber('anchorX', Number(e.target.value))} />
                  </label>
                  <label>
                    anchor Y
                    <input type="number" min={0} max={1} step={0.001} value={+layer.attach.anchor[1].toFixed(3)} onChange={(e) => setAttachNumber('anchorY', Number(e.target.value))} />
                  </label>
                  <label>
                    offset X
                    <input type="number" step={1} value={+(layer.attach.offset?.[0] ?? 0).toFixed(1)} onChange={(e) => setAttachNumber('offsetX', Number(e.target.value))} />
                  </label>
                  <label>
                    offset Y
                    <input type="number" step={1} value={+(layer.attach.offset?.[1] ?? 0).toFixed(1)} onChange={(e) => setAttachNumber('offsetY', Number(e.target.value))} />
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={layer.attach.rotateWith ?? true}
                      onChange={(e) => touch(() => { layer.attach!.rotateWith = e.target.checked; })}
                    />
                    跟随旋转
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={layer.attach.scaleWith ?? true}
                      onChange={(e) => touch(() => { layer.attach!.scaleWith = e.target.checked; })}
                    />
                    跟随缩放
                  </label>
                </>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="muted small">还没有图片图层。点「导入图片」添加。</div>
      )}
      <div className="small muted">
        保存会写入上面路径和 assets/accessories/；首次保存请选择仓库根目录。编辑器修改只影响当前页面，保存后刷新同一 URL 即可恢复。
        {asset?.sha256 ? ` sha256 ${asset.sha256.slice(0, 12)}…` : ''}
      </div>
    </div>
  );
}
