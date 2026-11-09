/*
 * 导出测试页：把 exportSceneVideo 暴露给 Playwright（scripts/export-video.ts）。
 * 与播放器使用同一份运行时；场景在页面里加载，逐帧渲染 + 上屏，再交给 mediabunny 编码。
 *
 * window.__charaforgeExport:
 *   ready                 Promise<void>
 *   render(opts)          渲染并导出，返回元数据（视频字节留在页面里）
 *   readResult(offset,n)  按 base64 分块读取导出结果
 *   reset()               丢弃结果 / 中止当前任务
 *   status()              当前状态（给 CLI 轮询进度）
 *   abort()               中止当前导出
 *   info()                GPU / MSAA / Worker 信息
 */
import { CharaForgeRuntime } from '@charaforge/runtime';
import type { Scene } from '@charaforge/scene';
import { exportSceneVideo, type ExportVideoResult } from './editor/export.ts';
import { bytesToBase64, CORE_PATH, pageUrl } from './util.ts';

export interface ExportPageRenderOptions {
  /** 场景对象，或场景 URL（相对页面地址） */
  scene: object | string;
  from?: number;
  to?: number;
  format?: 'mp4' | 'webm' | 'auto';
  quality?: 'high' | 'medium' | 'low';
  width?: number;
  height?: number;
  background?: string;
  /** 覆盖场景帧率（缺省用场景 timebase.fps） */
  fps?: number;
  /** 仅用于文档；运行时在页面加载时按 ?workers= 创建，导出中不能改变 */
  workers?: number;
}

export interface ExportPageResult {
  width: number;
  height: number;
  fps: number;
  frames: number;
  duration: number;
  hasAudio: boolean;
  mimeType: string;
  extension: 'mp4' | 'webm';
  codecs: { video: string; audio?: string };
  bytes: number;
}

export interface CharaForgeExportApi {
  ready: Promise<void>;
  render(opts: ExportPageRenderOptions): Promise<ExportPageResult>;
  readResult(offset: number, length: number): { data: string; total: number };
  reset(): void;
  status(): string;
  abort(): void;
  info(): { vendor: string; renderer: string; msaa: number; workers: number };
}

const params = new URLSearchParams(location.search);
const workers = params.has('workers') ? Number(params.get('workers')) : 0;
const msaa = params.has('msaa') ? Number(params.get('msaa')) : undefined;

const root = document.getElementById('root');
const statusEl = document.getElementById('status');
if (!root) throw new Error('缺少 #root');

let runtime!: CharaForgeRuntime;
let canvas!: HTMLCanvasElement;
let result: ExportVideoResult | null = null;
let controller: AbortController | null = null;
let status = 'booting';

function setStatus(text: string): void {
  status = text;
  if (statusEl) statusEl.textContent = `[export] ${text}`;
}

const ready: Promise<void> = (async () => {
  if (typeof (globalThis as { Live2DCubismCore?: unknown }).Live2DCubismCore === 'undefined') {
    throw new Error(`没有找到 Live2DCubismCore（${CORE_PATH}）`);
  }
  canvas = document.createElement('canvas');
  canvas.id = 'export-canvas';
  root.appendChild(canvas);
  runtime = await CharaForgeRuntime.create({
    assetBase: pageUrl('assets/'),
    coreUrl: pageUrl(CORE_PATH),
    canvas,
    msaa,
    workers,
  });
  setStatus(`ready (workers ${workers}, msaa ${runtime.compositor.msaaSamples})`);
})().catch((e) => {
  setStatus(`init failed: ${e instanceof Error ? e.message : String(e)}`);
  throw e;
});

const api: CharaForgeExportApi = {
  ready,

  async render(opts) {
    if (controller) throw new Error('已有导出任务在运行');
    controller = new AbortController();
    result = null;
    const t0 = performance.now();
    try {
      setStatus('加载场景');
      const scene: Scene = await runtime.loadScene(opts.scene);
      setStatus(`场景已加载（${scene.composition.width}×${scene.composition.height}，${scene.duration} 帧）`);
      const out = await exportSceneVideo({
        scene,
        canvas,
        renderFrame: async (f) => {
          await runtime.renderFrameAsync(f);
          runtime.present();
        },
        resolveAsset: (uri) => runtime.resolveAsset(uri),
        from: opts.from,
        to: opts.to,
        fps: opts.fps,
        format: opts.format,
        quality: opts.quality,
        width: opts.width,
        height: opts.height,
        background: opts.background,
        onProgress: (p) => {
          const pct = p.total > 0 ? Math.floor((p.frame / p.total) * 100) : 0;
          setStatus(`${p.phase} ${p.frame}/${p.total} (${pct}%)`);
        },
        signal: controller.signal,
      });
      result = out;
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      setStatus(
        `done ${out.extension}/${out.codecs.video}${out.codecs.audio ? `+${out.codecs.audio}` : ''} ` +
          `${out.frames} 帧 ${out.width}×${out.height} ${out.data.byteLength} 字节（${secs}s）`,
      );
      return {
        width: out.width,
        height: out.height,
        fps: out.fps,
        frames: out.frames,
        duration: out.duration,
        hasAudio: out.hasAudio,
        mimeType: out.mimeType,
        extension: out.extension,
        codecs: out.codecs,
        bytes: out.data.byteLength,
      };
    } catch (e) {
      setStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
      throw e;
    } finally {
      controller = null;
    }
  },

  readResult(offset, length) {
    if (!result) return { data: '', total: 0 };
    const bytes = result.data;
    const total = bytes.byteLength;
    const start = Math.max(0, Math.min(total, Math.floor(offset)));
    const end = Math.max(start, Math.min(total, start + Math.max(0, Math.floor(length))));
    return { data: bytesToBase64(bytes.subarray(start, end)), total };
  },

  reset() {
    controller?.abort();
    controller = null;
    result = null;
    setStatus('idle');
  },

  status() {
    return status;
  },

  abort() {
    controller?.abort();
  },

  info() {
    return { ...runtime.compositor.rendererInfo, msaa: runtime.compositor.msaaSamples, workers };
  },
};

declare global {
  interface Window {
    __charaforgeExport: CharaForgeExportApi;
  }
}

window.__charaforgeExport = api;
