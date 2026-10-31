/*
 * 在 Node（worker_threads）里加载 Cubism Core 与 runtime 的无 GL 部分（ModelResource / ModelInstance / LayerEvaluator），
 * 用于几何（顶点）求值与雅可比。参数求值、物理推进完全复用 packages/runtime 的代码；不做任何渲染。
 * 需要以 `--import tools/fit/src/geom/register.mjs` 启动（解析 @cubism/framework 别名）。
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { runInThisContext } from 'node:vm';
import { createRequire } from 'node:module';
import { REPO } from '../paths.ts';

let loaded: Promise<typeof import('../../../../packages/runtime/src/index.ts')> | null = null;

export function loadRuntime() {
  if (!loaded) {
    // 官方 Core 脚本在全局作用域定义 Live2DCubismCore（与浏览器 <script> 加载等价）
    // Core 是 Emscripten 产物，检测到 Node 环境会用 __dirname / require：包一层函数作用域提供它们
    const src = readFileSync(join(REPO, 'vendor/cubism/core/live2dcubismcore.min.js'), 'utf8');
    const fn = runInThisContext(`(function (__dirname, __filename, require, module, exports) {${src}\n;return Live2DCubismCore;})`, { filename: 'live2dcubismcore.min.js' });
    const mod = { exports: {} };
    (globalThis as any).Live2DCubismCore = fn(join(REPO, 'vendor/cubism/core'), join(REPO, 'vendor/cubism/core/live2dcubismcore.min.js'), createRequire(import.meta.url), mod, mod.exports);
    loaded = import('../../../../packages/runtime/src/index.ts');
  }
  return loaded;
}

/** 文件读取的 fetcher（URL 为 file://） */
export async function fileFetcher(url: string): Promise<ArrayBuffer> {
  const p = decodeURIComponent(new URL(url).pathname);
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
