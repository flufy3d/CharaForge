/*
 * CharaForge 运行时：场景加载、按帧求值、合成、读回；以及给拟合器用的单实例接口。
 *
 * 渲染结果只是 (场景, 帧号, 覆盖参数) 的函数：
 *   - 无墙钟时间、无随机、无隐式自动行为；
 *   - 物理固定步长 1/fps，seek 时从 in − warmup 重放（带检查点），顺序播放增量推进，二者逐位相同。
 */
import { parseScene, sample, type CameraTransform, type ImageAsset, type ImageLayer, type ModelLayer, type Scene, type Transform2D } from '@charaforge/scene';
import { asciiUrl } from './asciiUrl.ts';
import { waitForCubismCore } from './coreReady.ts';
import { Compositor, type LayerDrawParams, type Pixels, type Region, type ResolvedFilter } from './compositor.ts';
import { LayerEvaluator, type LayerStateToken, type ParamOverrides } from './evaluator.ts';
import type { InstanceState, ModelInstance } from './instance.ts';
import { cameraMatrix, IDENTITY, layerMatrix, mul, translate, type Affine } from './math.ts';
import { apply } from './math.ts';
import { defaultFetcher, ImageResource, ModelResource, type CanvasInfo, type Fetcher } from './resources.ts';
import { loadSceneAssets, resolver, type SceneAssets } from './sceneAssets.ts';
import { WorkerPool, type UpdateResult } from './workerPool.ts';

export interface RuntimeOptions {
  /** 资源根目录（assetRoot）的绝对 URL，场景里的 uri 相对于它。 */
  assetBase: string;
  /** 合成分辨率；加载场景时会改为场景的 composition 尺寸。 */
  width?: number;
  height?: number;
  canvas?: HTMLCanvasElement;
  /** Cubism 绘制的 MSAA 采样数，缺省 4；0 关闭。 */
  msaa?: number;
  /** Cubism 遮罩缓冲尺寸，缺省 2048。 */
  maskBufferSize?: number;
  fetcher?: Fetcher;
  /**
   * 求值 Worker 数（0 = 同步路径，全部在主线程求值；缺省 0）。
   * >0 时每个 Worker 一份 Core；图层的求值（物理状态链）在其主场 Worker，Core update 分给最空闲的 Worker，
   * 主线程只做导入与合成。必须用 renderFrameAsync；不支持 saveState/restoreState（拟合请用同步路径）。
   * 两条路径的像素逐位相同（测试 worker 一致性）。
   */
  workers?: number;
  /** Worker 里加载 Core 的 URL（缺省相对页面地址的 vendor/cubism/core/live2dcubismcore.min.js；建议调用方传绝对 URL） */
  coreUrl?: string;
}

/** 单实例绘制用的变换（与场景图层 transform 相同的语义，但不可动画）。 */
export interface InstanceTransform {
  x: number;
  y: number;
  scale: number;
  /** 缺省 1：sx = scale·scaleX·(flipX ? −1 : 1) */
  scaleX?: number;
  /** 缺省 1：sy = scale·scaleY */
  scaleY?: number;
  rotation?: number;
  anchorX?: number;
  anchorY?: number;
  flipX?: boolean;
}

export interface RenderFrameOptions {
  /** 按图层 ID 的一次性参数覆盖，仅作用于本帧。 */
  overrides?: Record<string, ParamOverrides>;
  /** （Worker 路径）拿到本帧结果后立即让 Worker 开始算这些帧，与主线程绘制重叠。 */
  prefetch?: number[];
}

interface LayerPlanBase {
  layer: ModelLayer | ImageLayer;
  visible: boolean;
  toScreen: Affine;
  draw: LayerDrawParams;
}

interface ModelLayerPlan extends LayerPlanBase {
  kind: 'model';
  layer: ModelLayer;
  ev: LayerEvaluator;
  /** 保持帧（规范 4b），所有属性在 g 求值 */
  g: number;
  antialias: boolean;
  overrides?: ParamOverrides;
  /** frameHold 段内 g 已经画过：直接复用缓存的图层缓冲 */
  cacheHit: boolean;
}

interface ImageLayerPlan extends LayerPlanBase {
  kind: 'image';
  layer: ImageLayer;
  image: ImageResource;
}

type LayerPlan = ModelLayerPlan | ImageLayerPlan;

const WORKER_CORE_MEMORY = 256 * 1024 * 1024;

interface PendingJob {
  key: string;
  plan: ModelLayerPlan;
  resolve: (r: UpdateResult) => void;
  reject: (e: Error) => void;
}

export interface RenderStats {
  frame: number;
  /** 本次求值各图层实际 step 的帧数之和（顺序播放时 = 活动图层数） */
  steps: number;
  activeLayers: number;
  ms: number;
  /** Worker 路径：等待 Worker 结果的时间（ms） */
  waitMs?: number;
  /** Worker 路径：Worker 内 Core update + 导出的最长耗时（ms） */
  workerMs?: number;
}

export interface RuntimeState {
  layers: Record<string, LayerStateToken>;
}

export class CharaForgeRuntime {
  readonly compositor: Compositor;
  readonly assetBase: string;
  private readonly fetcher: Fetcher;
  private readonly maskBufferSize: number;
  private readonly resources = new Map<string, Promise<ModelResource>>();
  private readonly images = new Map<string, Promise<ImageResource>>();
  private scene: Scene | null = null;
  private layers: { layer: ModelLayer; ev: LayerEvaluator }[] = [];
  private imageLayers: { layer: ImageLayer; image: ImageResource }[] = [];
  /** 最近一次 loadScene / syncScene 的资源表；rebuildEvaluators 复用它，避免重新加载模型与 motion。 */
  private sceneAssets: SceneAssets | null = null;
  /** 每个模型资源上空闲的实例（换场景时复用，避免反复创建渲染器） */
  private readonly instancePool = new Map<ModelResource, ModelInstance[]>();
  private readonly looseInstances = new Map<number, ModelInstance>();
  private nextInstanceId = 1;
  /** frameHold 缓存：图层 ID → 缓冲里是哪个 g 的画面 */
  private readonly holdCache = new Map<string, number>();
  readonly pool: WorkerPool | null;
  /** Worker 路径：已发出的 Core update 请求，key = 图层|g|覆盖 */
  private readonly inflight = new Map<string, Promise<UpdateResult>>();
  private queue: PendingJob[] = [];
  /** 图层 → 不再需要早于它的帧（播放器向前推进时提高） */
  private readonly watermark = new Map<string, number>();
  private readonly workerInit: Promise<void> | null;
  /** 统计：最近一次渲染做了几次 Core update（同步路径）/ 收到几个动态数据包（Worker 路径） */
  lastUpdates = 0;

  private constructor(compositor: Compositor, opts: RuntimeOptions) {
    this.compositor = compositor;
    const n = opts.workers ?? 0;
    this.pool = n > 0 ? new WorkerPool(n) : null;
    this.assetBase = new URL(opts.assetBase, typeof location !== 'undefined' ? location.href : undefined).href;
    this.fetcher = opts.fetcher ?? defaultFetcher;
    this.maskBufferSize = opts.maskBufferSize ?? 2048;
    this.workerInit = this.pool
      ? this.pool.init(
          // 缺省相对页面地址解析（页面可能在反向代理的路径前缀下）；Worker 拿到的是解析好的绝对 URL
          new URL(opts.coreUrl ?? 'vendor/cubism/core/live2dcubismcore.min.js', typeof document !== 'undefined' ? document.baseURI : location.href).href,
          this.assetBase,
          WORKER_CORE_MEMORY,
        )
      : null;
  }

  static async create(opts: RuntimeOptions): Promise<CharaForgeRuntime> {
    // 主线程同样等 Core 的 WASM 实例化完成（见 coreReady.ts）；没加载 Core 时保持原行为，到加载模型时再报错
    if ('Live2DCubismCore' in globalThis) await waitForCubismCore();
    const comp = await Compositor.create(opts.width ?? 1280, opts.height ?? 720, { canvas: opts.canvas, msaa: opts.msaa });
    const rt = new CharaForgeRuntime(comp, opts);
    await rt.workerInit;
    return rt;
  }

  get gl(): WebGL2RenderingContext {
    return this.compositor.gl;
  }

  get currentScene(): Scene | null {
    return this.scene;
  }

  resolveAsset(uri: string): string {
    return new URL(uri, this.assetBase).href;
  }

  /** 加载（并缓存）模型资源。uri 相对于 assetBase。 */
  loadModel(uri: string): Promise<ModelResource> {
    const url = this.resolveAsset(uri);
    let p = this.resources.get(url);
    if (!p) {
      p = ModelResource.load(this.gl, url, this.fetcher);
      this.resources.set(url, p);
      p.catch(() => this.resources.delete(url));
    }
    return p;
  }

  /** 加载（并缓存）图片资源。同一 URL 被多个 scene 资产引用时共享纹理，但尺寸声明必须一致。 */
  private loadImage(asset: ImageAsset): Promise<ImageResource> {
    const url = this.resolveAsset(asset.uri);
    let p = this.images.get(url);
    if (!p) {
      p = ImageResource.load(url, this.fetcher, asset);
      this.images.set(url, p);
      p.catch(() => this.images.delete(url));
    } else {
      p = p.then((r) => {
        if (r.width !== asset.width || r.height !== asset.height) {
          throw new Error(`图片尺寸与 scene 声明不一致：${url}（实际 ${r.width}×${r.height}，声明 ${asset.width}×${asset.height}）`);
        }
        return r;
      });
    }
    return p;
  }

  /** 加载场景（对象或 URL）。会做结构 + 语义校验。 */
  async loadScene(input: Scene | unknown | string): Promise<Scene> {
    return this.syncScene(input);
  }

  /**
   * 把当前场景同步为新的场景（编辑器用），不刷新页面：
   *   - 资源按 URL 缓存（loadModel / loadImage），只有新增的模型 / 图片才真正加载；
   *   - 模型实例按「图层 ID + 模型资源」复用，删除或换了模型的图层把实例还回 instancePool；
   *   - 所有 LayerEvaluator 重建，参数轨 / motion / expression 等求值输入的修改都会生效；
   *   - 结构变化后重新给 Worker 发送场景，并清空保持帧缓存、在途请求与检查点。
   * 对内容相同的场景，渲染结果与重新 loadScene 逐位相同。
   */
  async syncScene(input: Scene | unknown): Promise<Scene> {
    let raw: unknown = input;
    if (typeof input === 'string') {
      const res = await fetch(asciiUrl(input, typeof document !== 'undefined' ? document.baseURI : undefined));
      if (!res.ok) throw new Error(`场景加载失败 ${res.status}：${input}`);
      raw = await res.json();
    }
    const scene = parseScene(raw);
    const assets = await loadSceneAssets(scene, (u) => this.loadModel(u), (u) => this.resolveAsset(u));
    const modelLayers = scene.layers.filter((l): l is ModelLayer => l.kind === 'model');
    const imageLayers = scene.layers.filter((l): l is ImageLayer => l.kind === 'image');
    const imageAssets = new Map<string, ImageResource>();
    await Promise.all((scene.assets.images ?? []).map(async (a) => imageAssets.set(a.id, await this.loadImage(a))));

    // 旧场景的 Worker 任务已经无效；先丢弃再整体换场景，避免旧结果落到新图层上。
    this.holdCache.clear();
    this.dropQueue();
    if (this.pool) await this.pool.loadScene(scene, modelLayers.map((l) => l.id));

    // 按 ID 复用模型实例；删除 / 换模型的旧实例还回池子。
    const prev = new Map<string, { layer: ModelLayer; instance: ModelInstance }>();
    for (const { layer, ev } of this.layers) prev.set(layer.id, { layer, instance: ev.instance });
    const reused = new Set<ModelInstance>();
    const fps = scene.timebase.fps;
    const motion = resolver(assets.motions, 'motion');
    const expression = resolver(assets.expressions, 'expression');
    const nextLayers = modelLayers.map((layer) => {
      const res = assets.models.get(layer.model);
      if (!res) throw new Error(`模型资源未加载：${layer.model}`);
      const old = prev.get(layer.id);
      let instance: ModelInstance;
      if (old && old.layer.model === layer.model && old.instance.resource === res) {
        instance = old.instance;
        // 与 acquireInstance 一致：从头开始求值（LayerEvaluator 的 initialState 也在这里确定）。
        instance.resetPhysics();
        instance.resetPose();
      } else {
        instance = this.acquireInstance(res);
      }
      reused.add(instance);
      return { layer, ev: new LayerEvaluator(layer, instance, fps, motion, expression) };
    });
    for (const { instance } of prev.values()) if (!reused.has(instance)) this.releaseInstance(instance);

    const nextImages = imageLayers.map((layer) => {
      const image = imageAssets.get(layer.image);
      if (!image) throw new Error(`图片资源未加载：${layer.image}`);
      return { layer, image };
    });

    // 全部加载成功后才替换当前状态（失败时旧场景仍可用）。
    this.layers = nextLayers;
    this.imageLayers = nextImages;
    this.sceneAssets = assets;
    this.scene = scene;
    this.compositor.dropCaches();
    this.compositor.resize(scene.composition.width, scene.composition.height);
    this.invalidate();
    return scene;
  }

  /**
   * 重建主线程的 LayerEvaluator（编辑器修改了参数轨 / motion / expression 等求值输入之后）。
   * 复用现有模型实例与已加载资源，不重新加载模型/图片；最后刷新 Worker 端的场景副本。
   */
  async rebuildEvaluators(): Promise<void> {
    const scene = this.scene;
    if (!scene) throw new Error('尚未加载场景');
    const assets = this.sceneAssets;
    if (!assets) throw new Error('场景资源尚未加载');

    const prev = new Map<string, { layer: ModelLayer; instance: ModelInstance }>();
    for (const { layer, ev } of this.layers) prev.set(layer.id, { layer, instance: ev.instance });
    const reused = new Set<ModelInstance>();
    const fps = scene.timebase.fps;
    const motion = resolver(assets.motions, 'motion');
    const expression = resolver(assets.expressions, 'expression');
    const nextLayers = scene.layers
      .filter((l): l is ModelLayer => l.kind === 'model')
      .map((layer) => {
        const res = assets.models.get(layer.model);
        if (!res) throw new Error(`模型资源未加载：${layer.model}`);
        const old = prev.get(layer.id);
        let instance: ModelInstance;
        if (old && old.instance.resource === res) {
          instance = old.instance;
          instance.resetPhysics();
          instance.resetPose();
        } else {
          instance = this.acquireInstance(res);
        }
        reused.add(instance);
        return { layer, ev: new LayerEvaluator(layer, instance, fps, motion, expression) };
      });
    for (const { instance } of prev.values()) if (!reused.has(instance)) this.releaseInstance(instance);
    this.layers = nextLayers;

    this.invalidate();
    await this.reloadWorkers();
  }

  /**
   * 重新把当前场景发给 Worker（求值输入改变后调用；同步路径下是空操作）。
   * 丢弃在途请求与水位，旧场景算出的结果不再有效。
   */
  async reloadWorkers(): Promise<void> {
    this.holdCache.clear();
    this.dropQueue();
    if (!this.pool || !this.scene) return;
    const layerIds = this.scene.layers.filter((l): l is ModelLayer => l.kind === 'model').map((l) => l.id);
    await this.pool.loadScene(this.scene, layerIds);
    this.holdCache.clear();
  }

  private acquireInstance(res: ModelResource): ModelInstance {
    const pool = this.instancePool.get(res);
    const inst = pool?.pop() ?? res.createInstance({ maskBufferSize: this.maskBufferSize });
    inst.resetPhysics();
    inst.resetPose();
    return inst;
  }

  private releaseInstance(instance: ModelInstance): void {
    const res = instance.resource;
    const pool = this.instancePool.get(res) ?? [];
    pool.push(instance);
    this.instancePool.set(res, pool);
  }

  /** 编辑器用：移除模型或图片图层；模型实例还回池子。Worker 副本需调用方按需 reloadWorkers / syncScene。 */
  removeLayer(id: string): void {
    const mi = this.layers.findIndex((l) => l.layer.id === id);
    if (mi >= 0) {
      const entry = this.layers.splice(mi, 1)[0];
      if (entry) this.releaseInstance(entry.ev.instance);
    }
    const ii = this.imageLayers.findIndex((l) => l.layer.id === id);
    if (ii >= 0) this.imageLayers.splice(ii, 1);
    if (this.scene) {
      const si = this.scene.layers.findIndex((l) => l.id === id);
      if (si >= 0) this.scene.layers.splice(si, 1);
    }
    this.invalidate();
  }

  /** 编辑器用：把图层移动到 scene.layers 的 index 位置（z 顺序；不重建求值器）。 */
  moveLayer(id: string, index: number): void {
    if (!this.scene) return;
    const from = this.scene.layers.findIndex((l) => l.id === id);
    if (from < 0) return;
    const layer = this.scene.layers.splice(from, 1)[0];
    if (!layer) return;
    const to = Math.max(0, Math.min(this.scene.layers.length, Math.floor(index)));
    this.scene.layers.splice(to, 0, layer);
    const order = new Map(this.scene.layers.map((l, i) => [l.id, i]));
    this.layers.sort((a, b) => (order.get(a.layer.id) ?? 0) - (order.get(b.layer.id) ?? 0));
    this.imageLayers.sort((a, b) => (order.get(a.layer.id) ?? 0) - (order.get(b.layer.id) ?? 0));
  }

  /** 场景内图层的模型实例（只读访问参数、画布信息等）。 */
  layerInstance(layerId: string): ModelInstance | undefined {
    return this.layers.find((l) => l.layer.id === layerId)?.ev.instance;
  }

  /**
   * 编辑器用：向当前场景追加一个图片资产和图片图层（追加到 layers 末尾，即最上层）。
   * 只支持运行时新增；保存前由编辑器把 blob/data URI 换成 assets/accessories/ 下的相对路径。
   */
  async addImageLayer(asset: ImageAsset, layer: ImageLayer): Promise<void> {
    if (!this.scene) throw new Error('尚未加载场景');
    if (this.scene.layers.some((l) => l.id === layer.id)) throw new Error(`图层 ID 重复：${layer.id}`);
    if ((this.scene.assets.images ?? []).some((a) => a.id === asset.id)) throw new Error(`图片资源 ID 重复：${asset.id}`);
    const image = await this.loadImage(asset);
    if (!this.scene.assets.images) this.scene.assets.images = [];
    this.scene.assets.images.push(asset);
    this.scene.layers.push(layer);
    this.imageLayers.push({ layer, image });
    this.invalidate();
  }

  /** 帧 f 上每个存在的图层要做什么；返回顺序 = scene.layers 顺序（z 顺序）。 */
  private plan(f: number, overrides?: Record<string, ParamOverrides>): LayerPlan[] {
    const scene = this.scene!;
    const W = scene.composition.width;
    const H = scene.composition.height;
    const antialias = scene.composition.antialias ?? true;
    const modelPlans = new Map<string, ModelLayerPlan>();
    for (const { layer, ev } of this.layers) {
      const p = this.planModel(layer, ev, f, overrides, W, H, antialias);
      if (p) modelPlans.set(layer.id, p);
    }
    const imagePlans = new Map<string, ImageLayerPlan>();
    for (const { layer, image } of this.imageLayers) {
      const p = this.planImage(layer, image, f, modelPlans, W, H);
      if (p) imagePlans.set(layer.id, p);
    }
    const plans: LayerPlan[] = [];
    for (const layer of scene.layers) {
      const p = layer.kind === 'model' ? modelPlans.get(layer.id) : imagePlans.get(layer.id);
      if (p) plans.push(p);
    }
    return plans;
  }

  private planModel(
    layer: ModelLayer,
    ev: LayerEvaluator,
    f: number,
    overrides: Record<string, ParamOverrides> | undefined,
    W: number,
    H: number,
    antialias: boolean,
  ): ModelLayerPlan | null {
    // 图层是否存在按显示帧 f 判断；画面（状态与全部属性，含相机）在保持帧 g 求值（规范 4b）
    if (!ev.isActive(f)) return null;
    const g = holdFrame(layer, f);
    if (!ev.isActive(g)) return null;
    const ov = overrides?.[layer.id];
    const hasOv = !!ov && !!(ov.pre || ov.post || ov.parts);
    const opacity = sample(layer.opacity, g, 1);
    const toWorld = layerMatrix(sampleTransform(layer.transform, g, ev.instance.canvas));
    const cam = this.scene!.camera ? cameraMatrix(sampleCamera(this.scene!.camera, g), W, H) : IDENTITY;
    const toScreen = (layer.space ?? 'world') === 'screen' ? toWorld : mul(cam, toWorld);
    const draw = resolveLayerDraw(layer, g, opacity);
    const visible = opacity > 0 && layerOnScreen(toScreen, canvasBounds(ev.instance.canvas), draw, W, H);
    const cacheHit = visible && !hasOv && !!layer.frameHold?.length && this.holdCache.get(layer.id) === g;
    return { kind: 'model', layer, ev, g, visible, toScreen, draw, antialias, overrides: hasOv ? ov : undefined, cacheHit };
  }

  private planImage(
    layer: ImageLayer,
    image: ImageResource,
    f: number,
    modelPlans: Map<string, ModelLayerPlan>,
    W: number,
    H: number,
  ): ImageLayerPlan | null {
    const scene = this.scene!;
    // 图片图层同样遵守 in/out 活跃区间（与模型图层一致）。旧实现漏掉了这个判断，
    // 会让背景/元素图层在整个场景里一直可见。
    if (f < layer.in || f >= layer.out) return null;
    const opacity = sample(layer.opacity, f, 1);
    const own = layerMatrix(sampleTransform(layer.transform, f, imageCanvas(image)));
    let toScreen: Affine;
    if (layer.attach) {
      const target = modelPlans.get(layer.attach.layer);
      // 目标在 f 不活跃：图片也不绘制（attach 没有可用的整体变换）
      if (!target) return null;
      const targetCanvas = target.ev.instance.canvas;
      const attach = layer.attach;
      let mt: Affine;
      if ((attach.rotateWith ?? true) && (attach.scaleWith ?? true)) {
        mt = target.toScreen;
      } else {
        // 分别剥离目标旋转/缩放（含 flipX），平移与相机始终保留
        const t = sampleTransform(target.layer.transform, target.g, targetCanvas);
        const sx = t.scale * (t.scaleX ?? 1) * (t.flipX ? -1 : 1);
        const sy = t.scale * (t.scaleY ?? 1);
        const local = layerMatrix({
          x: t.x,
          y: t.y,
          scale: 1,
          scaleX: (attach.scaleWith ?? true) ? sx : 1,
          scaleY: (attach.scaleWith ?? true) ? sy : 1,
          rotation: (attach.rotateWith ?? true) ? t.rotation : 0,
          anchorX: t.anchorX,
          anchorY: t.anchorY,
          flipX: false,
        });
        const cam = scene.camera ? cameraMatrix(sampleCamera(scene.camera, target.g), W, H) : IDENTITY;
        mt = (target.layer.space ?? 'world') === 'screen' ? local : mul(cam, local);
      }
      const a = attach.anchor;
      const o = attach.offset ?? [0, 0];
      toScreen = mul(mul(mt, translate(a[0] * targetCanvas.width + o[0], a[1] * targetCanvas.height + o[1])), own);
    } else {
      const cam = scene.camera ? cameraMatrix(sampleCamera(scene.camera, f), W, H) : IDENTITY;
      toScreen = (layer.space ?? 'world') === 'screen' ? own : mul(cam, own);
    }
    const draw = resolveLayerDraw(layer, f, opacity);
    const visible = opacity > 0 && layerOnScreen(toScreen, { x0: 0, y0: 0, x1: image.width, y1: image.height }, draw, W, H);
    return { kind: 'image', layer, image, visible, toScreen, draw };
  }

  /** 画一个已求值好的图层（模型/图片 → filters → 缓存 → 输出）。 */
  private drawPlanned(p: LayerPlan): void {
    const comp = this.compositor;
    if (p.kind === 'image') {
      comp.outputLayer(comp.layerSourceImage(p.image.texture, p.toScreen, p.draw.filters), p.draw);
      return;
    }
    if (p.cacheHit) {
      comp.outputLayer(comp.cacheRT(p.layer.id), p.draw);
      return;
    }
    const src = comp.layerSource(p.ev.instance, p.toScreen, p.draw.filters, p.antialias);
    if (p.layer.frameHold?.length && !p.overrides) {
      comp.copy(src, comp.cacheRT(p.layer.id));
      this.holdCache.set(p.layer.id, p.g);
    }
    comp.outputLayer(src, p.draw);
  }

  /**
   * 场景求值 + 合成到合成缓冲（不上屏）。同步路径（workers = 0）。
   * 不可见的图层（opacity = 0、clip 面积为 0、完全出画）只推进物理，不做 Core update 和绘制；
   * frameHold 段内 g 已画过时直接复用图层缓冲。
   */
  renderFrame(frame: number, opts: RenderFrameOptions = {}): RenderStats {
    if (this.pool) throw new Error('Worker 模式请用 renderFrameAsync');
    const t0 = performance.now();
    if (!this.scene) throw new Error('尚未加载场景');
    const f = Math.floor(frame);
    this.compositor.begin(this.scene.composition.background);
    let steps = 0;
    let updates = 0;
    const plans = this.plan(f, opts.overrides);
    for (const p of plans) {
      if (p.kind === 'image') {
        if (p.visible) this.drawPlanned(p);
        continue;
      }
      if (p.cacheHit) {
        this.drawPlanned(p);
        continue;
      }
      p.ev.prepare(p.g, p.overrides, p.visible);
      steps += p.ev.lastSteps;
      if (p.ev.lastUpdated) updates++;
      if (p.visible) this.drawPlanned(p);
    }
    this.lastUpdates = updates;
    return { frame: f, steps, activeLayers: plans.length, ms: performance.now() - t0 };
  }

  /**
   * 渲染帧（Worker 路径；workers = 0 时等价于 renderFrame）。
   *
   * 图层的主场 Worker 按规范 a～g 求值（持有物理状态链），Core 输入交给最空闲的 Worker 做 Core update
   * （约 10 ms/实例/帧，最贵的部分），动态数据回到主线程后导入主线程的 Core 模型直接绘制。
   * 主线程只做导入、绘制与合成。prefetch 帧在等待本帧结果的同时就发出去，Worker 计算与主线程绘制重叠。
   * Core update 是输入的纯函数、求值路径无关，所以与同步路径逐位相同。
   */
  async renderFrameAsync(frame: number, opts: RenderFrameOptions = {}): Promise<RenderStats> {
    if (!this.pool) return this.renderFrame(frame, opts);
    const t0 = performance.now();
    if (!this.scene) throw new Error('尚未加载场景');
    const f = Math.floor(frame);
    const plans = this.plan(f, opts.overrides);
    // 本帧需要的 g 永远不算过期（向后 seek 时水位要降下来）；图片图层不参与 Worker 求值
    for (const p of plans) if (p.kind === 'model') this.watermark.set(p.layer.id, p.g);
    let steps = 0;
    const waits = plans.map((p) => {
      if (p.kind !== 'model') return null;
      if (p.cacheHit) return null;
      if (!p.visible) {
        void this.pool!.advance(p.layer.id, p.g, p.overrides).catch((e) => console.error(e));
        return null;
      }
      return this.requestUpdate(p);
    });
    for (const pf of opts.prefetch ?? []) {
      if (pf < 0 || pf >= this.scene.duration || pf === f) continue;
      for (const p of this.plan(pf)) if (p.kind === 'model' && p.visible && !p.cacheHit) void this.requestUpdate(p).catch(() => {});
    }
    const results = await Promise.all(waits);
    const t1 = performance.now();
    this.compositor.begin(this.scene.composition.background);
    let updates = 0;
    plans.forEach((p, i) => {
      const r = results[i];
      if (p.kind === 'model' && r) {
        steps += r.steps;
        p.ev.instance.importDynamics(r.packet);
        updates++;
      }
      if (p.visible) this.drawPlanned(p);
    });
    // 清掉已经用过的（≤ 当前 g）请求，并提高水位：更早的预取任务不再需要
    for (const p of plans) {
      if (p.kind !== 'model') continue;
      this.watermark.set(p.layer.id, p.g + 1);
      const prefix = `${p.layer.id}|`;
      for (const [k, q] of this.inflight) {
        if (!k.startsWith(prefix) || Number(k.split('|')[1]) > p.g) continue;
        this.inflight.delete(k);
        // 结果已导入（或已过期），缓冲还给 Worker 复用
        void q.then((r) => this.pool?.recycle(r), () => {});
      }
    }
    this.lastUpdates = updates;
    const workerMs = Math.max(0, ...results.map((r) => r?.ms ?? 0));
    return { frame: f, steps, activeLayers: plans.length, ms: performance.now() - t0, waitMs: t1 - t0, workerMs };
  }

  /**
   * 只预取、不渲染（Worker 路径；同步路径下什么也不做）：把这些帧上各可见图层的求值 + Core update 交给 Worker。
   * 用于提前做新图层的物理预热（从 in − warmup 起逐帧推进，约 0.05～0.3 秒）：结果留在 inflight 里，
   * 真正渲染到该帧时直接取用；求值器的状态链与按顺序播放完全相同，所以不改变任何像素。
   * 任务按帧号从小到大排队，不会抢在播放所需的帧前面。返回全部完成（或作为过期任务被丢弃）时兑现的 Promise。
   */
  prefetch(frames: number[]): Promise<void> {
    if (!this.pool || !this.scene) return Promise.resolve();
    const jobs: Promise<unknown>[] = [];
    for (const f of frames) {
      if (f < 0 || f >= this.scene.duration) continue;
      for (const p of this.plan(f)) if (p.kind === 'model' && p.visible && !p.cacheHit) jobs.push(this.requestUpdate(p).catch(() => {}));
    }
    return Promise.all(jobs).then(() => {});
  }

  /**
   * 请求 (图层, g, 覆盖) 的 Core update 结果；同一请求只发一次。
   * 任务先进入主线程的等待队列，有空闲 Worker 时按帧号从小到大发出（发出时才在主线程求值到「参数就绪」）。
   */
  private requestUpdate(p: ModelLayerPlan): Promise<UpdateResult> {
    const key = `${p.layer.id}|${p.g}|${p.overrides ? JSON.stringify(p.overrides) : ''}`;
    const q = this.inflight.get(key);
    if (q) return q;
    let job!: PendingJob;
    const promise = new Promise<UpdateResult>((resolve, reject) => {
      job = { key, plan: p, resolve, reject };
    });
    promise.catch(() => {});
    this.inflight.set(key, promise);
    this.queue.push(job);
    this.pump();
    return promise;
  }

  /** 把等待队列里的任务发给空闲 Worker；早于该图层水位的预取任务直接丢弃。 */
  private pump(): void {
    const pool = this.pool!;
    while (this.queue.length && pool.hasIdle(1)) {
      // 丢弃过期任务
      this.queue = this.queue.filter((j) => {
        const stale = j.plan.g < (this.watermark.get(j.plan.layer.id) ?? -Infinity);
        if (stale) {
          this.inflight.delete(j.key);
          j.reject(new Error('stale'));
        }
        return !stale;
      });
      if (!this.queue.length) break;
      let bi = 0;
      for (let i = 1; i < this.queue.length; i++) if (this.queue[i]!.plan.g < this.queue[bi]!.plan.g) bi = i;
      const job = this.queue.splice(bi, 1)[0]!;
      const { plan } = job;
      pool.evalUpdate(plan.layer.id, plan.g, plan.overrides).then(
        (r) => { job.resolve(r); this.pump(); },
        (e) => { this.inflight.delete(job.key); job.reject(e); this.pump(); },
      );
    }
  }

  /** 合成结果上屏（画到 canvas）。 */
  present(): void {
    this.compositor.present();
  }

  dispose(): void {
    this.pool?.dispose();
    for (const p of this.images.values()) void p.then((r) => r.dispose(), () => {});
    this.images.clear();
  }

  readPixels(region?: Region, scale = 1): Pixels {
    return this.compositor.readPixels(region, scale);
  }

  /** 保存所有图层的完整状态（物理 + 求值游标）。 */
  saveState(): RuntimeState {
    if (this.pool) throw new Error('Worker 模式不支持 saveState，请用同步路径（workers = 0）');
    const layers: Record<string, LayerStateToken> = {};
    for (const { layer, ev } of this.layers) layers[layer.id] = ev.saveToken();
    return { layers };
  }

  restoreState(s: RuntimeState): void {
    if (this.pool) throw new Error('Worker 模式不支持 restoreState，请用同步路径（workers = 0）');
    for (const { layer, ev } of this.layers) {
      const t = s.layers[layer.id];
      if (t) ev.restoreToken(t);
    }
  }

  /** 丢弃所有检查点（例如外部修改了 scene 对象之后）。 */
  invalidate(): void {
    for (const { ev } of this.layers) ev.invalidate();
    this.holdCache.clear();
    this.dropQueue();
  }

  private dropQueue(): void {
    for (const j of this.queue) j.reject(new Error('cancelled'));
    this.queue = [];
    this.inflight.clear();
    this.watermark.clear();
  }

  // ---------- 单实例接口（不依赖场景，用于拟合） ----------

  /** 创建一个独立实例，返回句柄。 */
  async createInstance(modelUri: string): Promise<number> {
    const res = await this.loadModel(modelUri);
    const inst = res.createInstance({ maskBufferSize: this.maskBufferSize });
    const id = this.nextInstanceId++;
    this.looseInstances.set(id, inst);
    return id;
  }

  instance(id: number): ModelInstance {
    const inst = this.looseInstances.get(id);
    if (!inst) throw new Error(`没有实例 ${id}`);
    return inst;
  }

  destroyInstance(id: number): void {
    this.instance(id).release();
    this.looseInstances.delete(id);
  }

  /**
   * 按规范 a～g 对单实例求值一帧（无 motion/expression/轨道，只有显式参数）：
   * a 复位 → pre 参数 → （physics 时）物理前进 dt → post 参数 → 部件不透明度 → update。
   */
  stepInstance(id: number, p: { pre?: Record<string, number>; post?: Record<string, number>; parts?: Record<string, number>; physics?: boolean; dt?: number }): void {
    const inst = this.instance(id);
    inst.resetPose();
    if (p.pre) inst.setParams(p.pre);
    if (p.physics) inst.stepPhysics(p.dt ?? 1 / 60);
    if (p.post) inst.setParams(p.post);
    if (p.parts) for (const k in p.parts) inst.setPartOpacity(k, p.parts[k]!);
    inst.update();
  }

  /**
   * 把单实例渲染到 width×height 的离屏缓冲并读回（不影响场景合成缓冲）。
   * transform 把模型画布像素映射到输出像素（与图层 transform 同语义；anchor 缺省画布中心）。
   * 输出为透明背景上的预乘 RGBA（便于拟合端自行与背景做 alpha 合成或当作遮罩）。
   */
  renderInstance(
    id: number,
    o: { transform: InstanceTransform; width: number; height: number; region?: Region; scale?: number; read?: boolean },
  ): Pixels | null {
    const inst = this.instance(id);
    const comp = this.compositor;
    const rt = comp.pooledRT(o.width, o.height, ':inst');
    const m = layerMatrix(fillTransform(o.transform, inst.canvas));
    comp.drawModel(inst, m, rt);
    if (o.read === false) return null;
    return comp.readPixels(o.region, o.scale ?? 1, rt);
  }

  saveInstanceState(id: number): InstanceState {
    return this.instance(id).saveState();
  }

  restoreInstanceState(id: number, s: InstanceState): void {
    this.instance(id).restoreState(s);
  }

  canvasInfo(id: number): CanvasInfo {
    return this.instance(id).canvas;
  }
}

// ---------- 采样工具 ----------

/**
 * 图层是否可能出现在画面上：opacity 已在外面判断；这里看 clip 面积、以及图层本地矩形
 * （模型画布或图片，经 toScreen 变换、按模糊半径外扩）与合成画面 / clip 矩形是否相交。保守判断：只会多画，不会漏画。
 */
interface LayerBounds {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

function canvasBounds(c: CanvasInfo): LayerBounds {
  return { x0: 0, y0: 0, x1: c.width, y1: c.height };
}

/** 图片图层的本地坐标系 = 图片像素；origin/ppu 不参与 2D 变换，仅为满足 CanvasInfo 类型。 */
function imageCanvas(image: ImageResource): CanvasInfo {
  return { width: image.width, height: image.height, originX: 0, originY: 0, pixelsPerUnit: 1 };
}

function layerOnScreen(toScreen: Affine, b: LayerBounds, d: LayerDrawParams, W: number, H: number): boolean {
  let pad = 1;
  for (const f of d.filters) {
    if (f.type === 'blur') pad += Math.ceil(3 * Math.max(0, f.sigma));
    else if (f.type === 'motionBlur') pad += Math.ceil(Math.max(0, f.length) / 2) + 1;
    else if (f.type === 'resample' && f.factor > 0 && f.factor < 1) pad += Math.ceil(3 / f.factor);
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [px, py] of [[b.x0, b.y0], [b.x1, b.y0], [b.x0, b.y1], [b.x1, b.y1]] as const) {
    const [x, y] = apply(toScreen, px, py);
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  let cx0 = 0, cy0 = 0, cx1 = W, cy1 = H;
  if (d.clip) {
    if (!(d.clip.width > 0 && d.clip.height > 0)) return false;
    const fe = Math.max(1, d.clip.feather) / 2;
    cx0 = Math.max(cx0, d.clip.x - fe); cy0 = Math.max(cy0, d.clip.y - fe);
    cx1 = Math.min(cx1, d.clip.x + d.clip.width + fe); cy1 = Math.min(cy1, d.clip.y + d.clip.height + fe);
  }
  return x1 > cx0 && x0 < cx1 && y1 > cy0 && y0 < cy1 && Number.isFinite(x0 + y0 + x1 + y1);
}

/**
 * 帧保持（规范 4b）：f 落在某段 [start, end) 内时返回 start + floor((f − start)/every)·every，否则返回 f。
 * 物理仍逐帧推进：求值器从 S(g−1) 推进到 g，与无保持时同一帧的状态逐位相同。
 */
export function holdFrame(layer: ModelLayer, f: number): number {
  for (const h of layer.frameHold ?? []) {
    if (f >= h.start && f < h.end) return h.start + Math.floor((f - h.start) / h.every) * h.every;
  }
  return f;
}

function fillTransform(t: InstanceTransform, c: CanvasInfo) {
  return {
    x: t.x,
    y: t.y,
    scale: t.scale,
    scaleX: t.scaleX ?? 1,
    scaleY: t.scaleY ?? 1,
    rotation: t.rotation ?? 0,
    anchorX: t.anchorX ?? c.width / 2,
    anchorY: t.anchorY ?? c.height / 2,
    flipX: t.flipX ?? false,
  };
}

export function sampleTransform(t: Transform2D, f: number, c: CanvasInfo) {
  return fillTransform(
    {
      x: sample(t.x, f, 0),
      y: sample(t.y, f, 0),
      scale: sample(t.scale, f, 1),
      scaleX: sample(t.scaleX, f, 1),
      scaleY: sample(t.scaleY, f, 1),
      rotation: sample(t.rotation, f, 0),
      anchorX: t.anchorX,
      anchorY: t.anchorY,
      flipX: t.flipX,
    },
    c,
  );
}

export function sampleCamera(c: CameraTransform, f: number) {
  return { x: sample(c.x, f, 0), y: sample(c.y, f, 0), zoom: sample(c.zoom, f, 1), rotation: sample(c.rotation, f, 0) };
}

function resolveLayerDraw(layer: ModelLayer | ImageLayer, f: number, opacity: number): LayerDrawParams {
  const filters: ResolvedFilter[] = [];
  for (const fl of layer.filters ?? []) {
    if (fl.enabled === false) continue;
    if (fl.type === 'blur') filters.push({ type: 'blur', sigma: sample(fl.sigma, f, 0) });
    else if (fl.type === 'motionBlur') filters.push({ type: 'motionBlur', angle: sample(fl.angle, f, 0), length: sample(fl.length, f, 0) });
    else if (fl.type === 'resample') filters.push({ type: 'resample', factor: sample(fl.factor, f, 1), method: fl.method ?? 'bilinear' });
    else
      filters.push({
        type: 'color',
        brightness: sample(fl.brightness, f, 0),
        contrast: sample(fl.contrast, f, 1),
        saturation: sample(fl.saturation, f, 1),
        multiply: sample(fl.multiply, f, [1, 1, 1]) as [number, number, number],
        offset: sample(fl.offset, f, [0, 0, 0]) as [number, number, number],
      });
  }
  const clip = layer.clip
    ? {
        x: sample(layer.clip.x, f, 0),
        y: sample(layer.clip.y, f, 0),
        width: sample(layer.clip.width, f, 0),
        height: sample(layer.clip.height, f, 0),
        radius: sample(layer.clip.radius, f, 0),
        feather: sample(layer.clip.feather, f, 0),
      }
    : null;
  return { filters, clip, opacity, blendMode: layer.blendMode ?? 'normal' };
}

export type { Affine };
