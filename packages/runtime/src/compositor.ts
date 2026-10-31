/*
 * 合成器（Pixi.js v8 + Cubism WebGL 渲染器共享同一个 WebGL2 上下文）。
 *
 * 每个图层：
 *   1. Cubism 把模型画进独立的预乘 alpha 缓冲（合成分辨率；可选 MSAA，之后 resolve 到 Pixi RenderTexture）
 *   2. 依次应用 filters（blur → 两遍可分离高斯；motionBlur；color）
 *   3. 输出 pass：圆角羽化 clip × opacity，按 blendMode 混合进合成缓冲
 *
 * GL 状态隔离：Cubism 绘制前后各做一次，前：绑定独立 VAO；后：解绑 VAO 并调用
 * renderer.resetState() 让 Pixi 丢弃所有缓存的 GL 绑定（纹理、程序、帧缓冲、混合、剔除……）。
 * 所有 Pixi pass 都通过 renderer.render({ container, target }) 完成，不手写 GL 状态。
 * 「Cubism 绘制用独立 VAO + 绘制后重置 Pixi 状态缓存」这一思路参考了 easy-live2d
 * （https://github.com/Panzer-Jack/easy-live2d ，MIT，Copyright (c) 2025 老船长PZ_Jack），未复制其代码。
 */
import {
  autoDetectRenderer,
  Container,
  GlProgram,
  Mesh,
  MeshGeometry,
  RenderTexture,
  Shader,
  Sprite,
  Texture,
  UniformGroup,
  type BLEND_MODES,
  type WebGLRenderer,
} from 'pixi.js';
import type { ModelInstance } from './instance.ts';
import type { Affine } from './math.ts';
import { BLUR_FRAG, COLOR_FRAG, COPY_FRAG, DEBUG_FRAG, MOTION_BLUR_FRAG, OUTPUT_FRAG, PASS_VERT, RESAMPLE_FRAG } from './shaders.ts';

export type ResolvedFilter =
  | { type: 'blur'; sigma: number }
  | { type: 'motionBlur'; angle: number; length: number }
  | {
      type: 'color';
      brightness: number;
      contrast: number;
      saturation: number;
      multiply: [number, number, number];
      offset: [number, number, number];
    };

export interface ResolvedClip {
  x: number;
  y: number;
  width: number;
  height: number;
  radius: number;
  feather: number;
}

export interface LayerDrawParams {
  filters: ResolvedFilter[];
  clip: ResolvedClip | null;
  opacity: number;
  blendMode: 'normal' | 'add' | 'multiply' | 'screen';
}

export interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Pixels {
  width: number;
  height: number;
  /** RGBA8，预乘 alpha，行从上到下 */
  data: Uint8Array;
}

export interface CompositorOptions {
  /** MSAA 采样数；0/1 = 关闭。缺省 4。 */
  msaa?: number;
  canvas?: HTMLCanvasElement | OffscreenCanvas;
}

interface PassShader {
  shader: Shader;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  uniforms: Record<string, any>;
}

export class Compositor {
  readonly renderer: WebGLRenderer;
  readonly gl: WebGL2RenderingContext;
  width = 0;
  height = 0;
  readonly msaaSamples: number;
  /** 合成结果（预乘 alpha，合成分辨率） */
  comp!: RenderTexture;
  private layerRT!: RenderTexture;
  private tmpA!: RenderTexture;
  private tmpB!: RenderTexture;
  private readonly vao: WebGLVertexArrayObject;
  private msaa: { fbo: WebGLFramebuffer; rb: WebGLRenderbuffer; w: number; h: number } | null = null;
  private readonly stage = new Container();
  private readonly quad: Mesh<MeshGeometry, Shader>;
  private readonly blur: PassShader;
  private readonly motionBlur: PassShader;
  private readonly color: PassShader;
  private readonly output: PassShader;
  private readonly resample: PassShader;
  private readonly debug: PassShader;
  private readonly copyPass: PassShader;
  private readonly rtPool = new Map<string, RenderTexture>();
  private readonly presentSprite = new Sprite();
  private readbackBuf: Uint8Array = new Uint8Array(0);

  static async create(width: number, height: number, opts: CompositorOptions = {}): Promise<Compositor> {
    const renderer = (await autoDetectRenderer({
      preference: 'webgl',
      preferWebGLVersion: 2,
      width,
      height,
      canvas: opts.canvas as HTMLCanvasElement | undefined,
      antialias: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: true,
      backgroundAlpha: 0,
      resolution: 1,
      autoDensity: false,
      hello: false,
      powerPreference: 'high-performance',
    })) as WebGLRenderer;
    if (!('gl' in renderer) || !(renderer.gl instanceof WebGL2RenderingContext)) {
      throw new Error('需要 WebGL2（Cubism R5 渲染器与本合成器都依赖 WebGL2）');
    }
    return new Compositor(renderer, width, height, opts);
  }

  private constructor(renderer: WebGLRenderer, width: number, height: number, opts: CompositorOptions) {
    this.renderer = renderer;
    this.gl = renderer.gl as WebGL2RenderingContext;
    const maxSamples = this.gl.getParameter(this.gl.MAX_SAMPLES) as number;
    this.msaaSamples = Math.min(opts.msaa ?? 4, maxSamples);
    const vao = this.gl.createVertexArray();
    if (!vao) throw new Error('createVertexArray 失败');
    this.vao = vao;

    const geometry = new MeshGeometry({
      positions: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
      uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const mk = (name: string, frag: string, u: Record<string, { value: unknown; type: string }>, extra: Record<string, unknown> = {}): PassShader => {
      const group = new UniformGroup(u as never);
      const shader = new Shader({
        glProgram: GlProgram.from({ vertex: PASS_VERT, fragment: frag, name: `charaforge-${name}` }),
        resources: { uTexture: Texture.EMPTY.source, passUniforms: group, ...extra },
      });
      return { shader, uniforms: group.uniforms };
    };
    this.blur = mk('blur', BLUR_FRAG, { uDir: { value: new Float32Array([1, 0]), type: 'vec2<f32>' }, uSigma: { value: 1, type: 'f32' } });
    this.motionBlur = mk('motionblur', MOTION_BLUR_FRAG, { uDir: { value: new Float32Array([1, 0]), type: 'vec2<f32>' }, uLength: { value: 0, type: 'f32' } });
    this.color = mk('color', COLOR_FRAG, {
      uBrightness: { value: 0, type: 'f32' },
      uContrast: { value: 1, type: 'f32' },
      uSaturation: { value: 1, type: 'f32' },
      uMultiply: { value: new Float32Array([1, 1, 1]), type: 'vec3<f32>' },
      uOffset: { value: new Float32Array([0, 0, 0]), type: 'vec3<f32>' },
    });
    this.output = mk('output', OUTPUT_FRAG, {
      uOpacity: { value: 1, type: 'f32' },
      uClipOn: { value: 0, type: 'f32' },
      uClipRect: { value: new Float32Array(4), type: 'vec4<f32>' },
      uClipShape: { value: new Float32Array(2), type: 'vec2<f32>' },
      uOrigin: { value: new Float32Array(2), type: 'vec2<f32>' },
    });
    this.resample = mk('resample', RESAMPLE_FRAG, {
      uRegion: { value: new Float32Array(4), type: 'vec4<f32>' },
      uOutSize: { value: new Float32Array(2), type: 'vec2<f32>' },
      uTaps: { value: 1, type: 'f32' },
    });
    this.debug = mk('debug', DEBUG_FRAG, {
      uMode: { value: 0, type: 'f32' },
      uMix: { value: 0.5, type: 'f32' },
      uGain: { value: 1, type: 'f32' },
    }, { uVideo: Texture.EMPTY.source });
    this.copyPass = mk('copy', COPY_FRAG, { uUnused: { value: 0, type: 'f32' } });
    this.quad = new Mesh<MeshGeometry, Shader>({ geometry, shader: this.blur.shader });
    this.stage.addChild(this.quad);
    this.resize(width, height);
  }

  /** 改变合成分辨率（重建图层缓冲与合成缓冲）。 */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    for (const rt of [this.comp, this.layerRT, this.tmpA, this.tmpB]) rt?.destroy(true);
    const mk = () => RenderTexture.create({ width, height, resolution: 1, antialias: false, scaleMode: 'linear' });
    this.comp = mk();
    this.layerRT = mk();
    this.tmpA = mk();
    this.tmpB = mk();
    this.renderer.resize(width, height);
    this.presentSprite.texture = this.comp;
  }

  /** 清空合成缓冲。background 为 CSS 颜色或 null（透明）。 */
  begin(background: string | null | undefined): void {
    const rgba = parseColor(background);
    this.clearRT(this.comp, rgba);
  }

  /** 用原生 GL 清空一张 RenderTexture（不依赖 Pixi 渲染通道的延迟清屏语义）。 */
  clearRT(rt: RenderTexture, rgba: [number, number, number, number]): void {
    const gl = this.gl;
    this.renderer.renderTarget.bind({ target: rt, clear: false });
    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.viewport(0, 0, rt.width, rt.height);
    gl.clearColor(rgba[0], rgba[1], rgba[2], rgba[3]);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this.renderer.resetState();
  }

  /**
   * 把一个模型实例画进图层缓冲（清空为透明），返回该缓冲。
   * toScreen：模型画布像素 → 合成像素。
   */
  drawModel(inst: ModelInstance, toScreen: Affine, target: RenderTexture = this.layerRT, antialias = true): RenderTexture {
    const gl = this.gl;
    const r = this.renderer;
    const w = target.width;
    const h = target.height;
    r.renderTarget.bind({ target, clear: true, clearColor: [0, 0, 0, 0] });
    const fbo = gl.getParameter(gl.FRAMEBUFFER_BINDING) as WebGLFramebuffer;
    gl.bindVertexArray(this.vao);
    try {
      if (antialias && this.msaaSamples > 1) {
        const m = this.ensureMsaa(w, h);
        gl.bindFramebuffer(gl.FRAMEBUFFER, m.fbo);
        gl.viewport(0, 0, w, h);
        gl.disable(gl.SCISSOR_TEST);
        gl.colorMask(true, true, true, true);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        inst.draw(m.fbo, toScreen, w, h, true);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, m.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, fbo);
        gl.disable(gl.SCISSOR_TEST);
        gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      } else {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.viewport(0, 0, w, h);
        gl.disable(gl.SCISSOR_TEST);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        inst.draw(fbo, toScreen, w, h, true);
      }
    } finally {
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      r.resetState();
    }
    return target;
  }

  /** 图层：模型 → filters → clip/opacity → 按 blendMode 混合进合成缓冲。 */
  drawLayer(inst: ModelInstance, toScreen: Affine, p: LayerDrawParams, antialias = true): void {
    this.outputLayer(this.layerSource(inst, toScreen, p.filters, antialias), p);
  }

  /** 模型 + filters，返回结果所在的缓冲（三张图层级缓冲之一，下一次调用前有效）。 */
  layerSource(inst: ModelInstance, toScreen: Affine, filters: ResolvedFilter[], antialias = true): RenderTexture {
    let src = this.drawModel(inst, toScreen, this.layerRT, antialias);
    for (const f of filters) src = this.applyFilter(src, f);
    return src;
  }

  /** 输出 pass：clip × opacity，按 blendMode 混合进合成缓冲。 */
  outputLayer(src: RenderTexture, p: LayerDrawParams): void {
    const u = this.output.uniforms as Record<string, unknown> & { uClipRect: Float32Array; uClipShape: Float32Array; uOrigin: Float32Array };
    u.uOpacity = p.opacity;
    u.uClipOn = p.clip ? 1 : 0;
    if (p.clip) {
      u.uClipRect.set([p.clip.x, p.clip.y, p.clip.width, p.clip.height]);
      u.uClipShape.set([p.clip.radius, p.clip.feather]);
    }
    u.uOrigin.set([0, 0]);
    this.pass(this.output, src, this.comp, false, p.blendMode);
  }

  /** 逐像素精确拷贝（texelFetch），用于 frameHold 的图层缓存。 */
  copy(src: RenderTexture, dst: RenderTexture): void {
    this.pass(this.copyPass, src, dst, true);
  }

  /** 每个 key 一张合成分辨率的持久缓冲（frameHold 缓存用）。 */
  cacheRT(key: string): RenderTexture {
    return this.pooledRT(this.width, this.height, `:cache:${key}`);
  }

  /** 丢弃全部持久缓存缓冲。 */
  dropCaches(): void {
    for (const [k, rt] of this.rtPool) {
      if (k.includes(':cache:')) {
        rt.destroy(true);
        this.rtPool.delete(k);
      }
    }
  }

  /** 三个图层级缓冲里除 src 以外的两个。 */
  private others(src: RenderTexture): [RenderTexture, RenderTexture] {
    const all = [this.layerRT, this.tmpA, this.tmpB].filter((t) => t !== src);
    return [all[0]!, all[1]!];
  }

  private applyFilter(src: RenderTexture, f: ResolvedFilter): RenderTexture {
    const [dst, dst2] = this.others(src);
    if (f.type === 'blur') {
      if (!(f.sigma > 0)) return src;
      const u = this.blur.uniforms as Record<string, unknown> & { uDir: Float32Array };
      u.uSigma = f.sigma;
      u.uDir.set([1, 0]);
      this.pass(this.blur, src, dst, true);
      u.uDir.set([0, 1]);
      this.pass(this.blur, dst, dst2, true);
      return dst2;
    }
    if (f.type === 'motionBlur') {
      if (!(f.length > 0)) return src;
      const u = this.motionBlur.uniforms as Record<string, unknown> & { uDir: Float32Array };
      const a = (f.angle * Math.PI) / 180;
      u.uDir.set([Math.cos(a), Math.sin(a)]);
      u.uLength = f.length;
      this.pass(this.motionBlur, src, dst, true);
      return dst;
    }
    const u = this.color.uniforms as Record<string, unknown> & { uMultiply: Float32Array; uOffset: Float32Array };
    u.uBrightness = f.brightness;
    u.uContrast = f.contrast;
    u.uSaturation = f.saturation;
    u.uMultiply.set(f.multiply);
    u.uOffset.set(f.offset);
    this.pass(this.color, src, dst, true);
    return dst;
  }

  /** 用全屏四边形把 src 经 shader 画到 dst。 */
  private pass(ps: PassShader, src: RenderTexture, dst: RenderTexture, clear: boolean, blend: BLEND_MODES = 'normal'): void {
    ps.shader.resources.uTexture = src.source;
    const q = this.quad;
    q.shader = ps.shader;
    q.blendMode = blend;
    q.scale.set(dst.width, dst.height);
    q.position.set(0, 0);
    this.renderer.render({ container: this.stage, target: dst, clear, clearColor: [0, 0, 0, 0] });
  }

  /**
   * 读回像素。region 为合成像素区域（缺省全画面），scale 为输出缩放（缺省 1）。
   * 输出尺寸 = round(region.w·scale) × round(region.h·scale)。缩小时做区域平均。
   */
  readPixels(region?: Region, scale = 1, src: RenderTexture = this.comp): Pixels {
    const reg = region ?? { x: 0, y: 0, width: src.width, height: src.height };
    const ow = Math.max(1, Math.round(reg.width * scale));
    const oh = Math.max(1, Math.round(reg.height * scale));
    let target = src;
    let rx = reg.x, ry = reg.y;
    const identity = scale === 1 && Number.isInteger(reg.x) && Number.isInteger(reg.y) && ow === reg.width && oh === reg.height;
    if (!identity) {
      target = this.pooledRT(ow, oh);
      const u = this.resample.uniforms as Record<string, unknown> & { uRegion: Float32Array; uOutSize: Float32Array };
      u.uRegion.set([reg.x, reg.y, reg.width, reg.height]);
      u.uOutSize.set([ow, oh]);
      u.uTaps = scale >= 1 ? 1 : Math.min(8, Math.ceil(1 / scale - 1e-9));
      this.pass(this.resample, src, target, true);
      rx = 0;
      ry = 0;
    }
    return this.readRT(target, rx, ry, ow, oh);
  }

  /** 直接读 RenderTexture 的一块（无缩放）。 */
  readRT(rt: RenderTexture, x: number, y: number, w: number, h: number, out?: Uint8Array): Pixels {
    const gl = this.gl;
    this.renderer.renderTarget.bind({ target: rt, clear: false });
    const n = w * h * 4;
    const data = out && out.length >= n ? out : new Uint8Array(n);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    gl.readPixels(x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
    this.renderer.resetState();
    return { width: w, height: h, data: data.length === n ? data : data.subarray(0, n) };
  }

  /** 尺寸为 w×h 的临时 RenderTexture（按尺寸复用）。 */
  pooledRT(w: number, h: number, tag = ''): RenderTexture {
    const key = `${w}x${h}${tag}`;
    let rt = this.rtPool.get(key);
    if (!rt) {
      rt = RenderTexture.create({ width: w, height: h, resolution: 1, antialias: false, scaleMode: 'linear' });
      this.rtPool.set(key, rt);
    }
    return rt;
  }

  /** 把合成结果画到画布（renderer 的根目标）。 */
  present(): void {
    this.presentSprite.texture = this.comp;
    this.renderer.render({ container: this.presentSprite, clear: true, clearColor: [0, 0, 0, 0] });
  }

  /** 调试视图：叠加（mode=overlay, mix）或差异（mode=diff, gain）。video 为参考帧纹理。 */
  presentDebug(video: Texture, mode: 'overlay' | 'diff', mix: number, gain: number): void {
    const u = this.debug.uniforms as Record<string, unknown>;
    u.uMode = mode === 'diff' ? 1 : 0;
    u.uMix = mix;
    u.uGain = gain;
    this.debug.shader.resources.uVideo = video.source;
    this.debug.shader.resources.uTexture = this.comp.source;
    const q = this.quad;
    q.shader = this.debug.shader;
    q.blendMode = 'normal';
    q.scale.set(this.width, this.height);
    this.renderer.render({ container: this.stage, clear: true, clearColor: [0, 0, 0, 1] });
  }

  private ensureMsaa(w: number, h: number) {
    const gl = this.gl;
    if (this.msaa && this.msaa.w === w && this.msaa.h === h) return this.msaa;
    if (this.msaa) {
      gl.deleteFramebuffer(this.msaa.fbo);
      gl.deleteRenderbuffer(this.msaa.rb);
    }
    const rb = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.msaaSamples, gl.RGBA8, w, h);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, rb);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('MSAA framebuffer 不完整');
    gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    this.msaa = { fbo, rb, w, h };
    return this.msaa;
  }

  /** 读一个字节缓冲（复用） */
  scratch(n: number): Uint8Array {
    if (this.readbackBuf.length < n) this.readbackBuf = new Uint8Array(n);
    return this.readbackBuf;
  }

  get rendererInfo(): { vendor: string; renderer: string } {
    const gl = this.gl;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return {
      vendor: String(gl.getParameter(ext ? ext.UNMASKED_VENDOR_WEBGL : gl.VENDOR)),
      renderer: String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER)),
    };
  }
}

/** CSS 颜色 → 预乘 RGBA（0..1）。支持 #rgb/#rrggbb/#rrggbbaa、rgb()/rgba()、transparent、null。 */
export function parseColor(c: string | null | undefined): [number, number, number, number] {
  if (c === null || c === undefined || c === 'transparent') return [0, 0, 0, 0];
  let r = 0, g = 0, b = 0, a = 1;
  const s = c.trim();
  if (s.startsWith('#')) {
    const h = s.slice(1);
    const x = (i: number, n: number) => parseInt(h.slice(i, i + n), 16) / (n === 1 ? 15 : 255);
    if (h.length === 3 || h.length === 4) { r = x(0, 1); g = x(1, 1); b = x(2, 1); if (h.length === 4) a = x(3, 1); }
    else { r = x(0, 2); g = x(2, 2); b = x(4, 2); if (h.length === 8) a = x(6, 2); }
  } else {
    const m = /rgba?\(([^)]+)\)/.exec(s);
    if (!m) throw new Error(`不支持的颜色：${c}`);
    const v = m[1]!.split(/[,\s/]+/).filter(Boolean).map(Number);
    r = v[0]! / 255; g = v[1]! / 255; b = v[2]! / 255; a = v[3] ?? 1;
  }
  return [r * a, g * a, b * a, a];
}
