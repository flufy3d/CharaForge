/*
 * 模型实例：一个 CubismModel + 一个 WebGL 渲染器 + 一份物理状态。
 *
 * 只提供「原子操作」，不含任何隐式行为（无自动眨眼/呼吸/口型/pose）：
 *   resetPose()          规范 a：参数、部件不透明度、模型不透明度复位为默认值
 *   setParam / ...       写参数（按 ID，带 set/add/multiply 语义与权重）
 *   stepPhysics(dt)      规范 e：物理前进 dt 秒（Framework 内部按 physics3 的 Fps 细分）
 *   update()             Core 更新顶点
 *   draw(...)            绘制到指定 framebuffer
 *   saveState/restoreState  物理状态的逐位精确快照
 */
import { CubismMatrix44 } from '@cubism/framework/math/cubismmatrix44';
import type { CubismModel } from '@cubism/framework/model/cubismmodel';
import { CubismPhysics } from '@cubism/framework/physics/cubismphysics';
import { CubismRenderer_WebGL } from '@cubism/framework/rendering/cubismrenderer_webgl';
import { prepareCubismShaders } from './cubism.ts';
import { cubismGlProxy, type ProxyContext } from './glProxy.ts';
import type { Affine } from './math.ts';
import { restorePhysics, savePhysics, type PhysicsSnapshot } from './physicsState.ts';
import type { CanvasInfo, ModelResource } from './resources.ts';

export type ParamMode = 'set' | 'add' | 'multiply';

/**
 * 一帧求值后 Core 的全部动态输出（参数、部件不透明度、drawable 顶点/不透明度/颜色/绘制顺序/动态标志……），
 * 按固定顺序逐字节打包。Worker 里求值 + Core update 之后导出，主线程导入到自己的 Core 模型里直接绘制，
 * 渲染器读到的数据与「在主线程上 update」逐位相同。
 */
/** Core 的全部输入：Core update 是它们的纯函数。 */
export interface CoreInputs {
  params: Float32Array;
  parts: Float32Array;
  modelOpacity: number;
}

export interface DynamicsPacket {
  buf: ArrayBuffer;
  modelOpacity: number;
}

/** 实例状态快照（物理是唯一跨帧状态：motion/expression 由帧号无状态求值，见 motion.ts）。 */
export interface InstanceState {
  physics: PhysicsSnapshot | null;
}

let warnedUnknown = new Set<string>();
export class ModelInstance {
  readonly resource: ModelResource;
  readonly model: CubismModel;
  /** 无 GL 的实例（Worker 里）为 null */
  readonly renderer: CubismRenderer_WebGL | null;
  readonly physics: CubismPhysics | null;
  private readonly initialPhysics: PhysicsSnapshot | null;
  private readonly paramIndex = new Map<string, number>();
  private readonly partIndex = new Map<string, number>();
  private readonly defaults: Float32Array;
  private readonly partDefaults: Float32Array;
  private readonly values: Float32Array;
  private readonly partOpacities: Float32Array;
  private readonly mins: Float32Array;
  private readonly maxs: Float32Array;
  private readonly mvp = new CubismMatrix44();
  private readonly mvpArray = new Float32Array(16);
  /** 模型里是否有开启背面剔除的 drawable（决定翻转绘制时是否需要修正正面朝向）。 */
  readonly usesCulling: boolean;
  private invertWinding = false;
  private proxyCtx: ProxyContext | null = null;
  private released = false;

  constructor(resource: ModelResource, opts: { maskBufferSize?: number } = {}) {
    this.resource = resource;
    const model = resource.moc.createModel();
    if (!model) throw new Error('moc.createModel 失败');
    this.model = model;
    const core = model.getModel();
    this.values = core.parameters.values;
    this.partOpacities = core.parts.opacities;
    this.defaults = core.parameters.defaultValues.slice();
    this.mins = core.parameters.minimumValues;
    this.maxs = core.parameters.maximumValues;
    this.partDefaults = core.parts.opacities.slice();
    for (let i = 0; i < core.parameters.count; i++) this.paramIndex.set(core.parameters.ids[i]!, i);
    for (let i = 0; i < core.parts.count; i++) this.partIndex.set(core.parts.ids[i]!, i);

    let culling = false;
    for (let i = 0; i < model.getDrawableCount(); i++) if (model.getDrawableCulling(i)) { culling = true; break; }
    this.usesCulling = culling;

    if (resource.physicsJson) {
      this.physics = CubismPhysics.create(resource.physicsJson, resource.physicsJson.byteLength);
      this.initialPhysics = savePhysics(this.physics);
    } else {
      this.physics = null;
      this.initialPhysics = null;
    }

    // 渲染器：遮罩缓冲数量按模型实际使用的遮罩组合数确定（每张缓冲最多 36/32 个）。
    const gl = resource.gl;
    if (gl) {
      const renderer = new CubismRenderer_WebGL(resource.canvas.width, resource.canvas.height);
      renderer.initialize(model, maskBufferCountFor(model));
      renderer.setIsPremultipliedAlpha(true);
      // 必须在 startUp 之前设置：setClippingMaskBufferSize 会重建遮罩管理器，而 GL 是在 startUp 里注入的。
      if (opts.maskBufferSize) renderer.setClippingMaskBufferSize(opts.maskBufferSize);
      // 所有实例共享一个 GL 代理（静态 UV/索引缓冲缓存 + 绕序修正），着色器针对代理只编译一次
      const shared = cubismGlProxy(gl);
      // UV / 索引是 moc 的静态数据：同一模型的所有实例按 drawable 序号共用一份 GL 缓冲（见 glProxy.ts）
      for (let i = 0; i < model.getDrawableCount(); i++) {
        shared.staticKeys.set(model.getDrawableVertexUvs(i), `${resource.url}|${i}|uv`);
        shared.staticKeys.set(model.getDrawableVertexIndices(i), `${resource.url}|${i}|idx`);
      }
      this.proxyCtx = shared.ctx;
      prepareCubismShaders(shared.proxy);
      renderer.startUp(shared.proxy);
      resource.textures.forEach((t, i) => renderer.bindTexture(i, t));
      this.renderer = renderer;
    } else {
      this.renderer = null;
    }
    this.resetPose();
    this.model.update();
  }

  get canvas(): CanvasInfo {
    return this.resource.canvas;
  }

  hasParam(id: string): boolean {
    return this.paramIndex.has(id);
  }

  paramIndexOf(id: string): number {
    const i = this.paramIndex.get(id);
    if (i === undefined) {
      if (!warnedUnknown.has(id)) {
        warnedUnknown.add(id);
        console.warn(`[runtime] 模型没有参数 ${id}，已忽略`);
      }
      return -1;
    }
    return i;
  }

  partIndexOf(id: string): number {
    const i = this.partIndex.get(id);
    if (i === undefined) {
      if (!warnedUnknown.has(`part:${id}`)) {
        warnedUnknown.add(`part:${id}`);
        console.warn(`[runtime] 模型没有部件 ${id}，已忽略`);
      }
      return -1;
    }
    return i;
  }

  /** 规范 a：所有参数、部件不透明度、模型不透明度复位为默认值。 */
  resetPose(): void {
    this.values.set(this.defaults);
    this.partOpacities.set(this.partDefaults);
    this.model.setModelOapcity(1);
  }

  /**
   * 写参数。mode：
   *   set      v ← v + (x − v)·w
   *   add      v ← v + x·w
   *   multiply v ← v·(1 + (x − 1)·w)（与 exp3 Multiply 相同）
   * 结果按 Framework 的规则夹到 [min, max]（重复参数按周期折回）。
   */
  setParam(id: string, x: number, mode: ParamMode = 'set', w = 1): void {
    const i = this.paramIndexOf(id);
    if (i >= 0) this.setParamByIndex(i, x, mode, w);
  }

  setParamByIndex(i: number, x: number, mode: ParamMode = 'set', w = 1): void {
    const v = this.values[i]!;
    let r: number;
    if (mode === 'add') r = v + x * w;
    else if (mode === 'multiply') r = v * (1 + (x - 1) * w);
    else r = w === 1 ? x : v + (x - v) * w;
    // 走 Framework 的 setter，保证夹取/重复语义与 motion/exp3 一致。
    this.model.setParameterValueByIndex(i, r, 1);
  }

  /** 批量 set（权重 1）。 */
  setParams(values: Record<string, number>): void {
    for (const id in values) this.setParam(id, values[id]!);
  }

  getParam(id: string): number {
    const i = this.paramIndex.get(id);
    return i === undefined ? NaN : this.values[i]!;
  }

  /** 读出全部参数（按 ID）。 */
  getParams(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [id, i] of this.paramIndex) out[id] = this.values[i]!;
    return out;
  }

  paramRange(id: string): { min: number; max: number; default: number } | null {
    const i = this.paramIndex.get(id);
    return i === undefined ? null : { min: this.mins[i]!, max: this.maxs[i]!, default: this.defaults[i]! };
  }

  setPartOpacity(id: string, v: number): void {
    const i = this.partIndexOf(id);
    if (i >= 0) this.partOpacities[i] = v;
  }

  getPartOpacity(id: string): number {
    const i = this.partIndex.get(id);
    return i === undefined ? NaN : this.partOpacities[i]!;
  }

  /** 规范 e：物理前进 dt 秒。 */
  stepPhysics(dt: number): void {
    if (this.physics) this.physics.evaluate(this.model, dt);
  }

  /** 用当前参数作为输入，把物理置于静止平衡态（Cubism stabilization）。 */
  stabilizePhysics(): void {
    if (this.physics) this.physics.stabilization(this.model);
  }

  /** 物理复位到加载时的初始状态（不做 stabilization，见规范第 4 条）。 */
  resetPhysics(): void {
    if (this.physics && this.initialPhysics) restorePhysics(this.physics, this.initialPhysics);
  }

  saveState(): InstanceState {
    return { physics: this.physics ? savePhysics(this.physics) : null };
  }

  restoreState(s: InstanceState): void {
    if (this.physics && s.physics) restorePhysics(this.physics, s.physics);
  }

  private dynViews: ArrayBufferView[] | null = null;

  /** Core 动态数据的实时视图（固定顺序；Core 堆不增长，视图长期有效）。 */
  private dynamicViews(): ArrayBufferView[] {
    if (this.dynViews) return this.dynViews;
    const core = this.model.getModel();
    const d = core.drawables;
    const o = core.offscreens;
    const views: ArrayBufferView[] = [
      core.parameters.values,
      core.parts.opacities,
      d.opacities,
      d.multiplyColors,
      d.screenColors,
      ...d.vertexPositions,
      d.drawOrders,
      core.getRenderOrders(),
      d.dynamicFlags,
    ];
    if (o && o.count > 0) views.push(o.opacities, o.multiplyColors, o.screenColors);
    this.dynViews = views;
    return views;
  }

  /** 导出 Core 的输入（参数值、部件不透明度、模型不透明度），交给 Worker 做 update。 */
  exportInputs(): CoreInputs {
    const core = this.model.getModel();
    return { params: core.parameters.values.slice(), parts: core.parts.opacities.slice(), modelOpacity: this.model.getModelOapcity() };
  }

  /** 写入 Core 输入（原样逐位写入，不做夹取——它们本来就是求值结果）并 update。 */
  updateFromInputs(i: CoreInputs): void {
    const core = this.model.getModel();
    core.parameters.values.set(i.params);
    core.parts.opacities.set(i.parts);
    this.model.setModelOapcity(i.modelOpacity);
    this.update();
  }

  /** 导出当前 Core 动态数据（应在 update() 之后调用）。 */
  exportDynamics(reuse?: ArrayBuffer): DynamicsPacket {
    const views = this.dynamicViews();
    let n = 0;
    for (const v of views) n += (v.byteLength + 3) & ~3;
    const buf = reuse && reuse.byteLength === n ? reuse : new ArrayBuffer(n);
    const out = new Uint8Array(buf);
    let off = 0;
    for (const v of views) {
      out.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), off);
      off += (v.byteLength + 3) & ~3;
    }
    return { buf, modelOpacity: this.model.getModelOapcity() };
  }

  /** 把另一个实例（同一 moc）导出的动态数据写进本实例的 Core，之后可直接 draw（不要再 update）。 */
  importDynamics(p: DynamicsPacket): void {
    const views = this.dynamicViews();
    const src = new Uint8Array(p.buf);
    let off = 0;
    for (const v of views) {
      new Uint8Array(v.buffer, v.byteOffset, v.byteLength).set(src.subarray(off, off + v.byteLength));
      off += (v.byteLength + 3) & ~3;
    }
    if (off !== src.length) throw new Error('动态数据包与实例不匹配');
    this.model.setModelOapcity(p.modelOpacity);
  }

  /** Core 更新（顶点、drawable 不透明度等）。 */
  update(): void {
    this.model.update();
    this.canonicalizeStaleVertices();
  }

  /**
   * 让 Core 的输出只取决于本次输入（确定性，规范：画面只是帧号的函数）。
   *
   * Core 不重算被跳过的 drawable（不可见、且本次没有 VertexPositionsDidChange）的顶点，数组里留着上一次算到它时的值
   * （或从没算过的 0）。Framework 的 CubismClippingManager.calcClippedDrawableTotalBounds 计算遮罩布局时会把
   * 被裁切 drawable 的顶点全部算进外接框，不看可见性，于是遮罩的布局、进而遮罩边缘几个像素取决于这个 Core 模型
   * 之前 update 过哪些帧：seek 与顺序播放、隔帧渲染与逐帧渲染、Worker 的调度顺序都会让结果差 ±1～3（P5 实测，见 docs/handoff.md）。
   * 这里把这些「本次未更新」的顶点置为 NaN：外接框计算里 NaN 比较恒为假，Framework 按「没有有效点」跳过它
   * （与它的注释一致）；绘制与遮罩绘制本来就跳过这些 drawable（IsVisible / VertexPositionsDidChange 判断），所以只影响外接框。
   */
  private canonicalizeStaleVertices(): void {
    const d = this.model.getModel().drawables;
    const flags = d.dynamicFlags;
    const U = Live2DCubismCore.Utils;
    for (let i = 0; i < d.count; i++) {
      const fl = flags[i]!;
      if (!U.hasIsVisibleBit(fl) && !U.hasVertexPositionsDidChangeBit(fl)) d.vertexPositions[i]!.fill(NaN);
    }
  }

  /**
   * 绘制到当前已绑定的 framebuffer 之外的指定 fbo。
   * @param toTarget 模型画布像素 → 目标像素（y 向下）的仿射变换
   * @param width/height 目标尺寸（像素）
   * @param rowZeroIsTop 目标的 GL 第 0 行是否对应图像顶部（Pixi RenderTexture 为 true；画布默认帧缓冲为 false）
   */
  draw(fbo: WebGLFramebuffer | null, toTarget: Affine, width: number, height: number, rowZeroIsTop: boolean): void {
    const { originX, originY, pixelsPerUnit: ppu } = this.resource.canvas;
    // Cubism 单位（y 向上） → 画布像素（y 向下）
    const [a, b, c, d, e, f] = toTarget;
    // 合成 unit → target px
    const A = a * ppu, B = b * ppu, C = -c * ppu, D = -d * ppu;
    const E = a * originX + c * originY + e, F = b * originX + d * originY + f;
    // target px → NDC
    const sx = 2 / width;
    const sy = rowZeroIsTop ? 2 / height : -2 / height;
    const ty = rowZeroIsTop ? -1 : 1;
    const m = this.mvpArray;
    m.fill(0);
    m[0] = A * sx; m[1] = B * sy;
    m[4] = C * sx; m[5] = D * sy;
    m[10] = 1;
    m[12] = E * sx - 1; m[13] = F * sy + ty;
    m[15] = 1;
    this.mvp.setMatrix(m);
    // 行列式 < 0 说明绕序相对 Cubism 预期（y 向上 NDC、det > 0）被翻转了。
    this.invertWinding = (m[0]! * m[5]! - m[1]! * m[4]!) < 0;
    const r = this.renderer;
    if (!r) throw new Error('该实例没有渲染器（无 GL）');
    if (this.proxyCtx) {
      this.proxyCtx.renderer = r;
      this.proxyCtx.invertWinding = this.usesCulling && this.invertWinding;
    }
    r.setMvpMatrix(this.mvp);
    r.setRenderTargetSize(width, height);
    r.setRenderState(fbo as WebGLFramebuffer, [0, 0, width, height]);
    r.drawModel();
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    this.renderer?.release();
    if (this.physics) CubismPhysics.delete(this.physics);
    this.resource.moc.deleteModel(this.model);
  }
}

/** 统计模型实际使用的遮罩组合数，决定渲染器需要几张遮罩缓冲。 */
function maskBufferCountFor(model: CubismModel): number {
  const masks = model.getDrawableMasks();
  const counts = model.getDrawableMaskCounts();
  const keys = new Set<string>();
  for (let i = 0; i < counts.length; i++) {
    if (counts[i]! > 0) keys.add(Array.from(masks[i]!.subarray(0, counts[i]!)).sort((x, y) => x - y).join(','));
  }
  const n = keys.size;
  if (n <= 36) return 1;
  return Math.ceil(n / 32);
}

export function resetUnknownWarnings(): void {
  warnedUnknown = new Set();
}
