/*
 * Harness 页面：把运行时以 window.__charaforge 的形式暴露给 Playwright（接口见 harness-api.ts）。
 * 播放器与 harness 用的是同一份 @charaforge/runtime 渲染代码。
 */
import { CharaForgeRuntime, type InstanceState, type Pixels, type RuntimeState } from '@charaforge/runtime';
import type { CharaForgeHarness, PixelsResult, ReadOptions, SceneSummary } from './harness-api.ts';
import { bytesToBase64, CORE_PATH, pageUrl, sha256Hex } from './util.ts';

const params = new URLSearchParams(location.search);
const msaa = params.has('msaa') ? Number(params.get('msaa')) : 4;
/** ?workers=N：用 Worker 求值路径（拟合缺省用同步路径 0，以便 saveState / restoreState） */
const workers = params.has('workers') ? Number(params.get('workers')) : 0;

let runtime!: CharaForgeRuntime;
const states = new Map<number, RuntimeState>();
const instStates = new Map<number, InstanceState>();
let nextHandle = 1;

async function encode(p: Pixels, fmt: ReadOptions['format']): Promise<PixelsResult> {
  if (fmt === 'sha256') return { width: p.width, height: p.height, sha256: await sha256Hex(p.data) };
  return { width: p.width, height: p.height, data: bytesToBase64(p.data) };
}

function summary(): SceneSummary {
  const s = runtime.currentScene!;
  return {
    title: s.meta.title,
    fps: s.timebase.fps,
    duration: s.duration,
    width: s.composition.width,
    height: s.composition.height,
    layers: s.layers.map((l) =>
      l.kind === 'model'
        ? { id: l.id, kind: l.kind, model: l.model, in: l.in, out: l.out }
        : { id: l.id, kind: l.kind, image: l.image, attach: l.attach?.layer, in: l.in, out: l.out },
    ),
    markers: (s.markers ?? []).map((m) => ({ id: m.id, start: m.start, end: m.end, label: m.label, kind: m.kind })),
    credits: [...(s.meta.credits ?? []), ...s.assets.models.flatMap((m) => (m.credits ? [m.credits] : []))],
  };
}

const api: CharaForgeHarness = {
  ready: (async () => {
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    runtime = await CharaForgeRuntime.create({ assetBase: pageUrl('assets/'), coreUrl: pageUrl(CORE_PATH), canvas, msaa, workers });
    // 调试用：直接访问运行时对象（非稳定接口）
    (window as unknown as { __cfRuntime: CharaForgeRuntime }).__cfRuntime = runtime;
  })(),
  info() {
    return { ...runtime.compositor.rendererInfo, msaa: runtime.compositor.msaaSamples, workers };
  },
  async loadScene(scene) {
    await runtime.loadScene(typeof scene === 'string' ? pageUrl(scene) : scene);
    return summary();
  },
  async renderFrame(frame, opts = {}) {
    const st = await runtime.renderFrameAsync(frame, { overrides: opts.overrides });
    const out = { frame: st.frame, steps: st.steps, ms: st.ms };
    if (!opts.read) return out;
    return { ...out, ...(await encode(runtime.readPixels(opts.read.region, opts.read.scale ?? 1), opts.read.format)) };
  },
  async readPixels(o = {}) {
    return encode(runtime.readPixels(o.region, o.scale ?? 1), o.format);
  },
  saveState() {
    const h = nextHandle++;
    states.set(h, runtime.saveState());
    return h;
  },
  restoreState(h) {
    const s = states.get(h);
    if (!s) throw new Error(`无效状态句柄 ${h}`);
    runtime.restoreState(s);
  },
  dropState(h) {
    states.delete(h);
    instStates.delete(h);
  },
  layerParams(layerId) {
    const inst = runtime.layerInstance(layerId);
    if (!inst) throw new Error(`没有图层 ${layerId}`);
    return inst.getParams();
  },
  createInstance(uri) {
    return runtime.createInstance(uri);
  },
  destroyInstance(id) {
    runtime.destroyInstance(id);
  },
  instanceInfo(id) {
    const inst = runtime.instance(id);
    return { canvas: inst.canvas, parameters: inst.resource.parameters, parts: inst.resource.parts };
  },
  stepInstance(id, p) {
    runtime.stepInstance(id, p);
  },
  resetInstancePhysics(id) {
    runtime.instance(id).resetPhysics();
  },
  instanceParams(id) {
    return runtime.instance(id).getParams();
  },
  async renderInstance(id, o) {
    const px = runtime.renderInstance(id, { transform: o.transform, width: o.width, height: o.height, region: o.region, scale: o.scale })!;
    return encode(px, o.format);
  },
  saveInstanceState(id) {
    const h = nextHandle++;
    instStates.set(h, runtime.saveInstanceState(id));
    return h;
  },
  restoreInstanceState(id, h) {
    const s = instStates.get(h);
    if (!s) throw new Error(`无效状态句柄 ${h}`);
    runtime.restoreInstanceState(id, s);
  },
};

window.__charaforge = api;

// ---- 参考视频逐帧定位（调试 / 验证用，scripts/verify-video.ts 调用）----
import { loadVideo, seekVideoFrame } from './videoSync.ts';
const videos = new Map<string, HTMLVideoElement>();
(window as unknown as { __charaforgeVideo: unknown }).__charaforgeVideo = {
  /** 把参考视频 seek 到帧 f，返回解码后的 RGB（base64，行从上到下） */
  async frameRGB(uri: string, frame: number, fps: number): Promise<{ width: number; height: number; data: string }> {
    let v = videos.get(uri);
    if (!v) { v = await loadVideo(pageUrl(uri)); videos.set(uri, v); }
    await seekVideoFrame(v, frame, fps);
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    const g = c.getContext('2d', { willReadFrequently: true })!;
    g.drawImage(v, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const rgb = new Uint8Array(c.width * c.height * 3);
    for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]!; rgb[j + 1] = d[i + 1]!; rgb[j + 2] = d[i + 2]!; }
    return { width: c.width, height: c.height, data: bytesToBase64(rgb) };
  },
};
