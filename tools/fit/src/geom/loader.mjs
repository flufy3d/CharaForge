// Node 模块解析钩子：让 packages/runtime 的源码能在 Node（worker_threads）里直接运行。
//   - '@cubism/framework/<path>' → .cache/cubism-framework/<path>.js（与 Vite 别名一致，由 pnpm framework 生成）；
//   - Framework 内部无扩展名的相对导入补上 .js。
// 不修改 runtime，只是模块解析。
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve as presolve } from 'node:path';

const REPO = presolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const FW = presolve(REPO, '.cache/cubism-framework');

export async function resolve(spec, ctx, next) {
  if (spec.startsWith('@cubism/framework/')) {
    return { url: pathToFileURL(presolve(FW, spec.slice('@cubism/framework/'.length) + '.js')).href, shortCircuit: true };
  }
  if ((spec.startsWith('./') || spec.startsWith('../')) && ctx.parentURL?.includes('/.cache/cubism-framework/') && !spec.endsWith('.js')) {
    const p = presolve(dirname(fileURLToPath(ctx.parentURL)), spec);
    if (existsSync(p + '.js')) return { url: pathToFileURL(p + '.js').href, shortCircuit: true };
    if (existsSync(presolve(p, 'index.js'))) return { url: pathToFileURL(presolve(p, 'index.js')).href, shortCircuit: true };
  }
  return next(spec, ctx);
}
