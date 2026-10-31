// 把 vendor/cubism/framework/src（官方 Cubism Framework R5 的 TS 源码）编译成 ESM + .d.ts，
// 输出到 .cache/cubism-framework/（不进 git）。
//
// 为什么不直接 import 源码：Framework 不是按本仓库的 strict / noUncheckedIndexedAccess 写的，
// 直接引用会让 `tsc --noEmit` 把它一起严格检查而失败。编译成 JS + d.ts 后，d.ts 属于 lib 文件，
// 被 skipLibCheck 跳过；Vite 和 tsc 用的是同一份产物，保证渲染代码只有一份。
//
// 输入没变化时（按文件大小 + mtime 计算指纹）直接跳过，开销可以忽略。
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const SRC = join(ROOT, 'vendor', 'cubism', 'framework', 'src');
const CORE_DTS = join(ROOT, 'vendor', 'cubism', 'core', 'live2dcubismcore.d.ts');
const OUT = join(ROOT, '.cache', 'cubism-framework');
// R5 的 WebGL 渲染器在运行时用 fetch 加载 Framework/Shaders/WebGL/*.vert|frag。
// setup:cubism 没有把它们装进 vendor/，这里从已缓存的 SDK 压缩包里只读地取出，生成一个同步可用的模块，
// 避免异步加载导致「首帧着色器尚未就绪」这种不确定性。
const SDK_ZIP = join(ROOT, '.cache', 'downloads', 'CubismSdkForWeb-5-r.5.zip');
const STAMP = join(OUT, '.stamp');
const VERSION = '2';

if (!existsSync(SRC) || !existsSync(CORE_DTS) || !existsSync(SDK_ZIP)) {
  console.error('[build-framework] 找不到 vendor/cubism，请先运行 pnpm setup:cubism');
  process.exit(1);
}

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, out);
    else if (name.endsWith('.ts')) out.push(p);
  }
  return out;
}

const files = walk(SRC).sort();
const h = createHash('sha256').update(VERSION);
for (const f of [...files, CORE_DTS, SDK_ZIP]) {
  const st = statSync(f);
  h.update(`${relative(ROOT, f)}:${st.size}:${st.mtimeMs}\n`);
}
const fingerprint = h.digest('hex');
if (existsSync(STAMP) && readFileSync(STAMP, 'utf8') === fingerprint) process.exit(0);

const require = createRequire(join(ROOT, 'packages', 'runtime', 'package.json'));
const ts = require('typescript');

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
const options = {
  target: ts.ScriptTarget.ES2020,
  module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler,
  lib: ['lib.es2020.d.ts', 'lib.dom.d.ts'],
  declaration: true,
  sourceMap: true,
  strict: false,
  noImplicitAny: false,
  skipLibCheck: true,
  rootDir: SRC,
  outDir: OUT,
  removeComments: false,
  types: [],
};
const program = ts.createProgram([...files, CORE_DTS], options);
const result = program.emit();
const diags = ts.getPreEmitDiagnostics(program).concat(result.diagnostics)
  .filter((d) => d.category === ts.DiagnosticCategory.Error);
if (diags.length) {
  console.error(ts.formatDiagnosticsWithColorAndContext(diags.slice(0, 20), {
    getCanonicalFileName: (f) => f, getCurrentDirectory: () => ROOT, getNewLine: () => '\n',
  }));
  process.exit(1);
}
// 着色器源码 → shaders.js（键为文件名）。
const { unzipSync, strFromU8 } = require('fflate');
const shaderEntries = unzipSync(readFileSync(SDK_ZIP), {
  filter: (f) => /\/Framework\/Shaders\/WebGL\/[a-z]+\.(vert|frag)$/.test(f.name),
});
const shaders = {};
for (const [name, data] of Object.entries(shaderEntries)) shaders[name.split('/').pop()] = strFromU8(data);
if (Object.keys(shaders).length < 13) {
  console.error('[build-framework] SDK 压缩包里的 WebGL 着色器不完整');
  process.exit(1);
}
writeFileSync(join(OUT, 'shaders.js'), `// 由 build-framework.mjs 生成，内容来自 Cubism SDK for Web R5 的 Framework/Shaders/WebGL/\nexport const CUBISM_SHADERS = ${JSON.stringify(shaders, null, 1)};\n`);
writeFileSync(join(OUT, 'shaders.d.ts'), 'export declare const CUBISM_SHADERS: Record<string, string>;\n');
// 让 d.ts 能看到 Core 的全局命名空间 Live2DCubismCore。
writeFileSync(join(OUT, 'live2dcubismcore.d.ts'), readFileSync(CORE_DTS));
writeFileSync(STAMP, fingerprint);
console.log(`[build-framework] 编译 ${files.length} 个文件 → ${relative(ROOT, OUT)}`);
