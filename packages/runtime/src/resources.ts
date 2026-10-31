/*
 * 模型资源：多实例共享的只读部分（moc、贴图、physics3 原始数据、motion、exp3、cdi3 显示名）。
 * 实例（ModelInstance）各自持有 CubismModel / 渲染器 / 物理状态。
 *
 * 资源来自不可信下载：这里只做 JSON 解析与二进制读取，不执行任何内容。
 */
import { CubismMoc } from '@cubism/framework/model/cubismmoc';
import { CubismMotion } from '@cubism/framework/motion/cubismmotion';
import { CubismExpressionMotion } from '@cubism/framework/motion/cubismexpressionmotion';
import { ensureCubism } from './cubism.ts';
import { ModelInstance } from './instance.ts';

export type Fetcher = (url: string) => Promise<ArrayBuffer>;

export const defaultFetcher: Fetcher = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`加载失败 ${res.status}：${url}`);
  return res.arrayBuffer();
};

/** moc3 画布信息。模型空间 = 画布像素：px = origin + unit·PPU（x），py = originY − unit·PPU（y 向下）。 */
export interface CanvasInfo {
  width: number;
  height: number;
  originX: number;
  originY: number;
  pixelsPerUnit: number;
}

export interface ParameterInfo {
  id: string;
  name?: string;
  group?: string;
  min: number;
  max: number;
  default: number;
}

export interface PartInfo {
  id: string;
  name?: string;
  defaultOpacity: number;
}

interface Model3Json {
  FileReferences: {
    Moc: string;
    Textures: string[];
    Physics?: string;
    DisplayInfo?: string;
    Pose?: string;
    Expressions?: { Name: string; File: string }[];
    Motions?: Record<string, { File: string }[]>;
  };
}

export interface ResourceOptions {
  /** 贴图各向异性过滤；0 = 关闭。 */
  anisotropy?: number;
}

export class ModelResource {
  readonly url: string;
  readonly baseUrl: string;
  readonly setting: Model3Json;
  readonly moc: CubismMoc;
  readonly mocBytes: ArrayBuffer;
  readonly textures: WebGLTexture[];
  readonly physicsJson: ArrayBuffer | null;
  readonly canvas: CanvasInfo;
  readonly parameters: ParameterInfo[];
  readonly parts: PartInfo[];
  /** null = 无 GL（Worker 里只做求值，不加载贴图） */
  readonly gl: WebGL2RenderingContext | null;
  private readonly fetcher: Fetcher;
  private readonly motions = new Map<string, Promise<CubismMotion>>();
  private readonly expressions = new Map<string, Promise<CubismExpressionMotion>>();
  private instanceCount = 0;

  private constructor(init: {
    url: string;
    setting: Model3Json;
    moc: CubismMoc;
    mocBytes: ArrayBuffer;
    textures: WebGLTexture[];
    physicsJson: ArrayBuffer | null;
    canvas: CanvasInfo;
    parameters: ParameterInfo[];
    parts: PartInfo[];
    gl: WebGL2RenderingContext | null;
    fetcher: Fetcher;
  }) {
    this.url = init.url;
    this.baseUrl = new URL('.', init.url).href;
    this.setting = init.setting;
    this.moc = init.moc;
    this.mocBytes = init.mocBytes;
    this.textures = init.textures;
    this.physicsJson = init.physicsJson;
    this.canvas = init.canvas;
    this.parameters = init.parameters;
    this.parts = init.parts;
    this.gl = init.gl;
    this.fetcher = init.fetcher;
  }

  /** 加载 model3.json 及其引用的 moc / 贴图 / physics / cdi3。url 必须是绝对 URL。 */
  static async load(gl: WebGL2RenderingContext | null, url: string, fetcher: Fetcher = defaultFetcher, opts: ResourceOptions = {}): Promise<ModelResource> {
    ensureCubism();
    const abs = (rel: string) => new URL(rel, url).href;
    const setting = JSON.parse(new TextDecoder().decode(await fetcher(url))) as Model3Json;
    const refs = setting.FileReferences;
    if (!refs?.Moc) throw new Error(`model3.json 缺少 Moc：${url}`);
    const [mocBytes, physicsJson, cdiBytes, texBytes] = await Promise.all([
      fetcher(abs(refs.Moc)),
      refs.Physics ? fetcher(abs(refs.Physics)) : Promise.resolve(null),
      refs.DisplayInfo ? fetcher(abs(refs.DisplayInfo)).catch(() => null) : Promise.resolve(null),
      gl ? Promise.all(refs.Textures.map((t) => fetcher(abs(t)))) : Promise.resolve([] as ArrayBuffer[]),
    ]);
    if (refs.Pose) console.warn('[runtime] 模型带 pose3，本运行时不做隐式 pose 处理（规范：无隐式行为）');
    const moc = CubismMoc.create(mocBytes, true);
    if (!moc) throw new Error(`moc3 加载失败（版本不兼容或数据损坏）：${refs.Moc}`);

    const textures = gl ? await Promise.all(texBytes.map((b) => uploadTexture(gl, b, opts.anisotropy ?? 0))) : [];

    // 用一个临时 Core 模型读取参数/部件表与画布信息（Core 的初始值即默认值）。
    const core = Live2DCubismCore.Model.fromMoc((moc as unknown as { _moc: Live2DCubismCore.Moc })._moc);
    if (!core) throw new Error('Core 无法从 moc 创建模型');
    const ci = core.canvasinfo;
    const canvas: CanvasInfo = {
      width: ci.CanvasWidth,
      height: ci.CanvasHeight,
      originX: ci.CanvasOriginX,
      originY: ci.CanvasOriginY,
      pixelsPerUnit: ci.PixelsPerUnit,
    };
    const names = new Map<string, { name?: string; group?: string }>();
    const partNames = new Map<string, string>();
    if (cdiBytes) {
      try {
        const cdi = JSON.parse(new TextDecoder().decode(cdiBytes)) as {
          Parameters?: { Id: string; Name?: string; GroupId?: string }[];
          Parts?: { Id: string; Name?: string }[];
        };
        for (const p of cdi.Parameters ?? []) names.set(p.Id, { name: p.Name, group: p.GroupId });
        for (const p of cdi.Parts ?? []) if (p.Name) partNames.set(p.Id, p.Name);
      } catch {
        console.warn('[runtime] cdi3.json 解析失败，忽略显示名');
      }
    }
    const P = core.parameters;
    const parameters: ParameterInfo[] = [];
    for (let i = 0; i < P.count; i++) {
      const id = P.ids[i]!;
      parameters.push({
        id,
        name: names.get(id)?.name,
        group: names.get(id)?.group,
        min: P.minimumValues[i]!,
        max: P.maximumValues[i]!,
        default: P.defaultValues[i]!,
      });
    }
    const parts: PartInfo[] = [];
    for (let i = 0; i < core.parts.count; i++) {
      const id = core.parts.ids[i]!;
      parts.push({ id, name: partNames.get(id), defaultOpacity: core.parts.opacities[i]! });
    }
    core.release();

    return new ModelResource({ url, setting, moc, mocBytes, textures, physicsJson, canvas, parameters, parts, gl, fetcher });
  }

  /** 加载并缓存 motion3（按绝对 URL 共享）。 */
  loadMotion(url: string): Promise<CubismMotion> {
    let p = this.motions.get(url);
    if (!p) {
      p = this.fetcher(url).then((buf) => {
        const m = CubismMotion.create(buf, buf.byteLength, undefined, undefined, true);
        if (!m) throw new Error(`motion3 解析失败：${url}`);
        // 不把 EyeBlink / LipSync 组绑定到 motion 的效果曲线：本运行时没有隐式眨眼/口型。
        m.setEffectIds([], []);
        return m;
      });
      this.motions.set(url, p);
    }
    return p;
  }

  /** 加载并缓存 exp3（按绝对 URL 共享）。 */
  loadExpression(url: string): Promise<CubismExpressionMotion> {
    let p = this.expressions.get(url);
    if (!p) {
      p = this.fetcher(url).then((buf) => {
        const e = CubismExpressionMotion.create(buf, buf.byteLength);
        if (!e) throw new Error(`exp3 解析失败：${url}`);
        return e;
      });
      this.expressions.set(url, p);
    }
    return p;
  }

  /** model3.json 中声明的 motion / expression（相对模型目录的绝对 URL）。 */
  listDeclared(): { motions: { group: string; url: string }[]; expressions: { name: string; url: string }[] } {
    const refs = this.setting.FileReferences;
    const motions: { group: string; url: string }[] = [];
    for (const [group, list] of Object.entries(refs.Motions ?? {})) for (const m of list) motions.push({ group, url: new URL(m.File, this.url).href });
    const expressions = (refs.Expressions ?? []).map((e) => ({ name: e.Name, url: new URL(e.File, this.url).href }));
    return { motions, expressions };
  }

  createInstance(opts: { maskBufferSize?: number } = {}): ModelInstance {
    this.instanceCount++;
    return new ModelInstance(this, opts);
  }
}

async function uploadTexture(gl: WebGL2RenderingContext, bytes: ArrayBuffer, anisotropy: number): Promise<WebGLTexture> {
  // ImageBitmap 在创建时就做预乘（WebGL 对 ImageBitmap 忽略 UNPACK_* 标志），与 Cubism 的预乘 alpha 管线一致。
  const bmp = await createImageBitmap(new Blob([bytes]), {
    premultiplyAlpha: 'premultiply',
    colorSpaceConversion: 'none',
    imageOrientation: 'none',
  });
  const tex = gl.createTexture();
  if (!tex) throw new Error('createTexture 失败');
  const prevActive = gl.getParameter(gl.ACTIVE_TEXTURE) as number;
  gl.activeTexture(gl.TEXTURE0);
  const prev = gl.getParameter(gl.TEXTURE_BINDING_2D) as WebGLTexture | null;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (anisotropy > 0) {
    const ext = gl.getExtension('EXT_texture_filter_anisotropic');
    if (ext) gl.texParameterf(gl.TEXTURE_2D, ext.TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);
  }
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.bindTexture(gl.TEXTURE_2D, prev);
  gl.activeTexture(prevActive);
  bmp.close();
  return tex;
}
