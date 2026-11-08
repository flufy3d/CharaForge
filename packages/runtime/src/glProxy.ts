/*
 * 给 Cubism 渲染器用的 GL 包装对象（每个原生 GL 上下文一个，所有实例共享，因此着色器只编译一次）。
 *
 * 1. 静态缓冲缓存：Cubism R5 渲染器每画一个 drawable 都把 UV 与索引用 bufferData 重新上传到同一个共享缓冲
 *    （CPU 剖析中 bufferData 占绘制时间约 24%）。UV / 索引在模型生命周期内不变，这里拦截对这些
 *    Core 视图的 bufferData：第一次为该视图建一个 STATIC_DRAW 缓冲，之后直接绑定它，跳过上传。
 *    数据与绘制调用完全相同，像素逐位不变。
 *    缓存按「模型 URL + drawable 序号 + 种类」共享给同一模型的所有实例（它们的 UV / 索引逐字节相同）。
 *    以前按实例各建一份，每个新实例第一次绘制要建约 1000 个缓冲；WSL 的 Mesa d3d12 下 GPU 进程执行这些
 *    命令要 0.1～0.7 秒，渲染进程在 WaitForAvailableEntries 上干等，全片播放到每个新图层都会卡（P5 实测，见 docs/handoff.md）。
 * 2. 绕序修正：带背面剔除的模型在翻转绘制（det < 0）时，主绘制阶段把 Cubism 设置的 frontFace(CCW) 换成 CW。
 */
import type { CubismRenderer_WebGL } from '@cubism/framework/rendering/cubismrenderer_webgl';

export interface ProxyContext {
  /** 当前正在绘制的渲染器（用于判断是否在生成遮罩） */
  renderer: CubismRenderer_WebGL | null;
  invertWinding: boolean;
}

interface Shared {
  proxy: WebGL2RenderingContext;
  ctx: ProxyContext;
  /**
   * 静态视图 → 共享键。同一个 moc 的所有实例，第 i 个 drawable 的 UV / 索引内容完全相同，
   * 键 = 模型 URL + drawable 序号 + 种类，所以第二个及以后的实例直接复用第一个实例建好的缓冲。
   */
  staticKeys: WeakMap<ArrayBufferView, string>;
  cache: Map<string, WebGLBuffer>;
}

const shared = new WeakMap<WebGL2RenderingContext, Shared>();

export function cubismGlProxy(gl: WebGL2RenderingContext): Shared {
  let s = shared.get(gl);
  if (s) return s;
  const ctx: ProxyContext = { renderer: null, invertWinding: false };
  const staticKeys = new WeakMap<ArrayBufferView, string>();
  const cache = new Map<string, WebGLBuffer>();
  const overrides: Record<string, unknown> = {
    frontFace(mode: number) {
      const flip = ctx.invertWinding && !!ctx.renderer && !ctx.renderer.isGeneratingMask();
      gl.frontFace(flip ? (mode === gl.CCW ? gl.CW : gl.CCW) : mode);
    },
    bufferData(target: number, data: unknown, usage: number, ...rest: unknown[]) {
      const key = ArrayBuffer.isView(data) && rest.length === 0 ? staticKeys.get(data) : undefined;
      if (key !== undefined) {
        let buf = cache.get(key);
        if (buf) {
          gl.bindBuffer(target, buf);
          return;
        }
        buf = gl.createBuffer()!;
        cache.set(key, buf);
        gl.bindBuffer(target, buf);
        gl.bufferData(target, data as ArrayBufferView, gl.STATIC_DRAW);
        return;
      }
      (gl.bufferData as (...a: unknown[]) => void)(target, data, usage, ...rest);
    },
  };
  // 不用 Proxy（每次属性访问都走 trap，开销比省下的上传还大）：建一个普通对象，
  // 方法预先 bind 到原生上下文，常量直接拷贝，其余属性（drawingBufferWidth 等）用 getter 转发。
  const proxy: Record<string, unknown> = {};
  const seen = new Set<string>();
  for (let o: object | null = gl; o && o !== Object.prototype; o = Object.getPrototypeOf(o)) {
    for (const key of Object.getOwnPropertyNames(o)) {
      if (seen.has(key) || key === 'constructor') continue;
      seen.add(key);
      const desc = Object.getOwnPropertyDescriptor(o, key)!;
      if (key in overrides) {
        proxy[key] = overrides[key];
      } else if (typeof desc.value === 'function') {
        proxy[key] = (desc.value as (...a: unknown[]) => unknown).bind(gl);
      } else if ('value' in desc) {
        proxy[key] = desc.value;
      } else {
        Object.defineProperty(proxy, key, { get: () => (gl as unknown as Record<string, unknown>)[key], enumerable: true });
      }
    }
  }
  s = { proxy: proxy as unknown as WebGL2RenderingContext, ctx, staticKeys, cache };
  shared.set(gl, s);
  return s;
}
