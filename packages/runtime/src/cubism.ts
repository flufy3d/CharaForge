/*
 * Cubism Framework 的启动与着色器注入。
 *
 * Framework 源码来自 Live2D Cubism SDK for Web R5（vendor/cubism/framework/src，Live2D Open Software License），
 * 由 scripts/build-framework.mjs 编译到 .cache/cubism-framework/ 后在这里引用；本仓库不包含其源码。
 * Core（live2dcubismcore.min.js）必须由页面用 <script> 先行加载，提供全局 Live2DCubismCore。
 */
import { CubismFramework, LogLevel, Option } from '@cubism/framework/live2dcubismframework';
import { CubismShaderManager_WebGL, CubismShaderSet } from '@cubism/framework/rendering/cubismshader_webgl';
import { CUBISM_SHADERS } from '@cubism/framework/shaders';

let started = false;

/**
 * Core 的 WASM 堆大小。Core 的堆不会自动增长，本模型 moc3 就有 16.5MB，
 * 默认 16MB 会导致加载或多实例更新失败；这里默认 512MB（4 个实例 + 余量）。
 */
export const CORE_MEMORY_BYTES = 512 * 1024 * 1024;

/** 启动 Framework（全局一次）。Core 未加载时抛错。 */
export function ensureCubism(memoryBytes = CORE_MEMORY_BYTES): void {
  if (started) return;
  if (typeof Live2DCubismCore === 'undefined') {
    throw new Error('未找到 Live2DCubismCore：请先用 <script> 加载 vendor/cubism/core/live2dcubismcore.min.js');
  }
  const option = new Option();
  option.logFunction = (msg: string) => console.warn(`[cubism] ${msg}`);
  option.loggingLevel = LogLevel.LogLevel_Warning;
  CubismFramework.startUp(option);
  CubismFramework.initialize(memoryBytes);
  started = true;
}

// 着色器文件名 → CubismShader_WebGL 上的属性名（与 Framework R5 cubismshader_webgl.ts 的 loadShaders 一致）。
const SHADER_PROPS: Record<string, string> = {
  'vertshadersrc.vert': '_vertShaderSrc',
  'vertshadersrcmasked.vert': '_vertShaderSrcMasked',
  'vertshadersrcsetupmask.vert': '_vertShaderSrcSetupMask',
  'fragshadersrcsetupmask.frag': '_fragShaderSrcSetupMask',
  'fragshadersrcpremultipliedalpha.frag': '_fragShaderSrcPremultipliedAlpha',
  'fragshadersrcmaskpremultipliedalpha.frag': '_fragShaderSrcMaskPremultipliedAlpha',
  'fragshadersrcmaskinvertedpremultipliedalpha.frag': '_fragShaderSrcMaskInvertedPremultipliedAlpha',
  'vertshadersrccopy.vert': '_vertShaderSrcCopy',
  'fragshadersrccopy.frag': '_fragShaderSrcCopy',
  'fragshadersrccolorblend.frag': '_fragShaderSrcColorBlend',
  'fragshadersrcalphablend.frag': '_fragShaderSrcAlphaBlend',
  'vertshadersrcblend.vert': '_vertShaderSrcBlend',
  'fragshadersrcpremultipliedalphablend.frag': '_fragShaderSrcBlend',
};

const preparedContexts = new WeakSet<WebGLRenderingContext>();

/**
 * R5 渲染器默认在首次绘制时用 fetch 异步加载着色器，加载完成前画不出任何东西。
 * 这里改为同步注入源码并立即编译，保证「第一次 drawModel 就是完整结果」。
 */
export function prepareCubismShaders(gl: WebGL2RenderingContext): void {
  if (preparedContexts.has(gl)) return;
  const mgr = CubismShaderManager_WebGL.getInstance();
  mgr.setGlContext(gl);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const shader = mgr.getShader(gl) as any;
  shader._shaderSets.length = shader._shaderCount;
  for (let i = 0; i < shader._shaderCount; i++) shader._shaderSets[i] = new CubismShaderSet();
  for (const [file, prop] of Object.entries(SHADER_PROPS)) {
    const src = CUBISM_SHADERS[file];
    if (src === undefined) throw new Error(`缺少 Cubism 着色器 ${file}`);
    shader[prop] = src;
  }
  shader.registerShader();
  shader.registerBlendShader();
  shader._isShaderLoading = false;
  shader._isShaderLoaded = true;
  preparedContexts.add(gl);
}
