/**
 * Harness API —— harness.html 在 `window.__charaforge` 上暴露给 Playwright 的接口。
 *
 * 用法（Node 端）：
 *   const page = await openHarness();                       // 见 scripts/browser.ts
 *   await page.evaluate(() => window.__charaforge.ready);
 *   await page.evaluate((p) => window.__charaforge.loadScene(p), 'packages/scene/examples/minimal.scene.json');
 *   const img = await page.evaluate(() => window.__charaforge.renderFrame(42, { read: {} }));
 *   // img.data 为 base64 的 RGBA8（预乘 alpha，行从上到下），img.width × img.height
 *
 * 约定：
 *   - 所有路径都是「仓库相对路径」（与 dev 服务器 URL 一致），如 'scenes/deepseek-chan/scene.json'；
 *     模型 uri 相对于资源根目录 assets/，如 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json'。
 *   - 渲染与读回都在同一个 WebGL2 上下文（Pixi v8 + Cubism）里，确定性：同输入同输出，逐位相同。
 *   - 一切都是帧号的函数：renderFrame(f) 不依赖调用历史（内部会按需重放物理，带检查点缓存）。
 *   - harness.html?workers=N 使用 Worker 求值路径（与同步路径逐位相同，但不支持 saveState/restoreState）；
 *     缺省 workers=0（同步路径），?msaa=N 覆盖 MSAA 采样数。
 */

export interface Region {
  /** 合成（或单实例输出）像素坐标，原点左上 */
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ReadOptions {
  /** 读回区域，缺省整幅 */
  region?: Region;
  /** 输出缩放（<1 时做区域平均下采样），缺省 1。输出尺寸 = round(region × scale) */
  scale?: number;
  /** 'rgba' = base64 RGBA8（缺省）；'sha256' = 只返回像素的 SHA-256（十六进制），用于确定性比对 */
  format?: 'rgba' | 'sha256';
}

export interface PixelsResult {
  width: number;
  height: number;
  /** format='rgba' 时为 base64 编码的 RGBA8（预乘 alpha，行从上到下） */
  data?: string;
  /** format='sha256' 时的十六进制摘要 */
  sha256?: string;
}

/** 外部参数覆盖（仅作用于被请求的那一帧，不进入后续帧的物理状态链）。 */
export interface ParamOverrides {
  /** 规范 d 步之后、e 步（物理）之前写入，set 语义 */
  pre?: Record<string, number>;
  /** 规范 f 步之后写入，set 语义（覆盖物理输出） */
  post?: Record<string, number>;
  /** 部件不透明度（g 步之后） */
  parts?: Record<string, number>;
}

export interface SceneSummary {
  title: string;
  fps: number;
  duration: number;
  width: number;
  height: number;
  layers: { id: string; kind: 'model' | 'image'; model?: string; image?: string; attach?: string; in: number; out: number }[];
  markers: { id: string; start: number; end?: number; label?: string; kind?: string }[];
  credits: string[];
}

export interface FrameResult extends Partial<PixelsResult> {
  frame: number;
  /** 本次求值实际推进的帧数（顺序播放 = 活动图层数；seek 时含重放） */
  steps: number;
  ms: number;
}

export interface ParameterInfo {
  id: string;
  name?: string;
  group?: string;
  min: number;
  max: number;
  default: number;
}

export interface InstanceInfo {
  /** moc3 画布：模型空间 = 画布像素（原点左上，y 向下） */
  canvas: { width: number; height: number; originX: number; originY: number; pixelsPerUnit: number };
  parameters: ParameterInfo[];
  parts: { id: string; name?: string; defaultOpacity: number }[];
}

/**
 * 与场景图层 transform 同语义：out = [x,y] + R(rotation)·diag(sx, sy)·(p − anchor)，
 * sx = scale·scaleX·(flipX ? −1 : 1)，sy = scale·scaleY；anchor 缺省画布中心。
 */
export interface InstanceTransform {
  x: number;
  y: number;
  scale: number;
  /** 缺省 1 */
  scaleX?: number;
  /** 缺省 1 */
  scaleY?: number;
  rotation?: number;
  anchorX?: number;
  anchorY?: number;
  flipX?: boolean;
}

export interface CharaForgeHarness {
  /** 页面初始化完成（Core/Framework/Pixi 就绪）后 resolve */
  ready: Promise<void>;
  /** WebGL 信息；renderer 应包含 "NVIDIA"（不是 SwiftShader） */
  info(): { vendor: string; renderer: string; msaa: number; workers: number };

  // ---- 场景 ----
  /** 加载场景：仓库相对路径或场景对象。返回摘要。 */
  loadScene(scene: string | object): Promise<SceneSummary>;
  /** 渲染帧 f（可带按图层 ID 的覆盖），可选同时读回像素。 */
  renderFrame(frame: number, opts?: { overrides?: Record<string, ParamOverrides>; read?: ReadOptions }): Promise<FrameResult>;
  /** 读回最近一次 renderFrame 的合成结果 */
  readPixels(opts?: ReadOptions): Promise<PixelsResult>;
  /** 保存全部图层状态（物理 + 求值游标），返回句柄 */
  saveState(): number;
  restoreState(handle: number): void;
  dropState(handle: number): void;
  /** 某图层在最近一次求值后的全部参数值 */
  layerParams(layerId: string): Record<string, number>;

  // ---- 单实例（不依赖场景，用于拟合）----
  createInstance(modelUri: string): Promise<number>;
  destroyInstance(id: number): void;
  instanceInfo(id: number): InstanceInfo;
  /** 规范 a～g 的单实例版本：复位 → pre → （physics 时）物理前进 dt（缺省 1/60）→ post → parts → update */
  stepInstance(id: number, p: ParamOverrides & { physics?: boolean; dt?: number }): void;
  /** 物理复位到初始状态 */
  resetInstancePhysics(id: number): void;
  /** 当前参数值（stepInstance 之后，含物理输出） */
  instanceParams(id: number): Record<string, number>;
  /** 渲染到 width×height 的透明离屏缓冲并读回 */
  renderInstance(id: number, o: { transform: InstanceTransform; width: number; height: number } & ReadOptions): Promise<PixelsResult>;
  saveInstanceState(id: number): number;
  restoreInstanceState(id: number, handle: number): void;
}

declare global {
  interface Window {
    __charaforge: CharaForgeHarness;
  }
}
