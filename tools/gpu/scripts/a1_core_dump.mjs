/*
 * A1 对拍：在 Node 里用官方 Cubism Core（vendor/cubism/core/live2dcubismcore.js）加载 moc3，
 * 对给定的参数集逐个 csmUpdateModel，dump 各 drawable 的动态输出，供 tools/gpu/l2d/deform.py 对拍。
 *
 * 用法：node tools/gpu/scripts/a1_core_dump.mjs <job.json>
 *   job = {
 *     moc: moc3 路径,
 *     params: float32 二进制 [N, P]（P = Core 参数个数，顺序同 Core），
 *     parts?: float32 二进制 [N, Q]（部件不透明度；缺省全 1）,
 *     patches?: [{offset, hex}]（仅在内存里改 moc 字节，用于探测；不写回文件）,
 *     out: 输出前缀 → <out>.json（元信息与布局）+ <out>.bin（数据）,
 *     static?: true 时额外输出 <out>.static.json / <out>.static.bin（UV、索引、参数 key 等静态信息）
 *   }
 * 输出的顶点是 Core 原始单位（float32，y 向上）；换成画布像素：x_px = originX + x·ppu，y_px = originY − y·ppu。
 */
import { readFileSync, writeFileSync, openSync, writeSync, closeSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runInThisContext } from 'node:vm';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');

async function loadCore() {
  // Core 是 Emscripten 产物，检测到 Node 会用 __dirname / require：包一层函数作用域提供它们（同 tools/fit/src/geom/core.ts）
  const file = join(REPO, 'vendor/cubism/core/live2dcubismcore.js');
  const src = readFileSync(file, 'utf8');
  const fn = runInThisContext(`(function (__dirname, __filename, require, module, exports) {${src}\n;return Live2DCubismCore;})`, { filename: file });
  const mod = { exports: {} };
  const core = fn(dirname(file), file, createRequire(import.meta.url), mod, mod.exports);
  // Emscripten 模块是异步初始化的：等到能调用为止
  // Core 堆不会自动增长；16.5MB 的 moc 需要扩大（同 packages/runtime 的 CORE_MEMORY_BYTES 思路）
  for (let i = 0; ; i++) {
    try {
      core.Memory.initializeAmountOfMemory(256 * 1024 * 1024);
      return core;
    } catch (e) {
      if (i > 500) throw e;
      await new Promise((r) => setTimeout(r, 10));
    }
  }
}

function readF32(path, n) {
  const b = readFileSync(path);
  if (b.byteLength !== n * 4) throw new Error(`${path}: 期望 ${n * 4} 字节，实际 ${b.byteLength}`);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

const job = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const Core = await loadCore();
const raw = readFileSync(job.moc);
const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
for (const p of job.patches ?? []) {
  const bytes = Buffer.from(p.hex, 'hex');
  if (p.offset < 0 || p.offset + bytes.length > buf.byteLength) throw new Error(`patch 越界：${p.offset}`);
  new Uint8Array(buf, p.offset, bytes.length).set(bytes);
}
const moc = Core.Moc.fromArrayBuffer(buf);
if (!moc) throw new Error('Moc.fromArrayBuffer 失败');
const model = Core.Model.fromMoc(moc);
if (!model) throw new Error('Model.fromMoc 失败');

const pr = model.parameters, pa = model.parts, dr = model.drawables, ci = model.canvasinfo;
const canvasMeta = { width: ci.CanvasWidth, height: ci.CanvasHeight, originX: ci.CanvasOriginX, originY: ci.CanvasOriginY, ppu: ci.PixelsPerUnit };
const P = pr.count, Q = pa.count, D = dr.count;
const vcount = Array.from(dr.vertexCounts);
const vofs = [0];
for (const c of vcount) vofs.push(vofs[vofs.length - 1] + c);
const V = vofs[D];

const N = job.n;
const params = readF32(job.params, N * P);
const parts = job.parts ? readF32(job.parts, N * Q) : null;

// 输出布局：各段按样本连续存放
const layout = [
  ['verts', 'f4', 2 * V],
  ['opacity', 'f4', D],
  ['drawOrder', 'i4', D],
  ['renderOrder', 'i4', D],
  ['multiply', 'f4', 4 * D],
  ['screen', 'f4', 4 * D],
  ['dynFlags', 'u1', D],
];
const arrays = {};
for (const [name, dt, n] of layout) {
  const T = dt === 'f4' ? Float32Array : dt === 'i4' ? Int32Array : Uint8Array;
  arrays[name] = new T(N * n);
}

if (job.static) {
  // 静态信息：用于校验解析器（UV、索引、mask、标志、参数 key 等）
  const chunks = [];
  let so = 0;
  const put = (a) => { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); chunks.push(u); const o = so; so += u.byteLength; return o; };
  const uv = new Float32Array(2 * V);
  for (let d = 0; d < D; d++) uv.set(dr.vertexUvs[d], 2 * vofs[d]);
  const icount = Array.from(dr.indexCounts);
  const idx = new Int32Array(icount.reduce((a, b) => a + b, 0));
  for (let d = 0, o = 0; d < D; d++) { idx.set(dr.indices[d], o); o += icount[d]; }
  const st = {
    canvas: canvasMeta,
    parameters: {
      ids: Array.from(pr.ids), min: Array.from(pr.minimumValues), max: Array.from(pr.maximumValues), default: Array.from(pr.defaultValues),
      types: Array.from(pr.types ?? []), repeats: Array.from(pr.repeats ?? []), keys: Array.from(pr.keyValues ?? [], (k) => Array.from(k)),
    },
    parts: { ids: Array.from(pa.ids), parents: Array.from(pa.parentIndices), offscreens: Array.from(pa.offscreenIndices ?? []) },
    drawables: {
      ids: Array.from(dr.ids), constantFlags: Array.from(dr.constantFlags), textureIndices: Array.from(dr.textureIndices),
      masks: Array.from(dr.masks, (m) => Array.from(m)), parentParts: Array.from(dr.parentPartIndices), blendModes: Array.from(dr.blendModes ?? []),
      vertexCounts: vcount, indexCounts: icount,
    },
    offscreens: model.offscreens ? { count: model.offscreens.count } : null,
    uvOffset: put(uv), idxOffset: put(idx),
  };
  writeFileSync(`${job.out}.static.bin`, Buffer.concat(chunks));
  writeFileSync(`${job.out}.static.json`, JSON.stringify(st));
}
// Core 对「不可见」（参数超出 keyform 绑定的 key 范围，或父变形器不可见）的 drawable 不更新顶点 / 不透明度 / draw order，
// 保留上一次的值。为了让每个样本的输出只取决于它自己的参数，缺省每个样本都新建一个 Model（初始值全 0）。
const fresh = job.fresh ?? true;
let model_ = model;
let tUpdate = 0;
for (let s = 0; s < N; s++) {
  if (fresh && s > 0) {
    model_.release();
    model_ = Core.Model.fromMoc(moc);
    if (!model_) throw new Error('Model.fromMoc 失败');
  }
  const pr = model_.parameters, pa = model_.parts, dr = model_.drawables, model = model_;
  pr.values.set(params.subarray(s * P, (s + 1) * P));
  if (parts) pa.opacities.set(parts.subarray(s * Q, (s + 1) * Q));
  else pa.opacities.fill(1);
  dr.resetDynamicFlags();
  const tu = performance.now();
  model.update();
  tUpdate += performance.now() - tu;
  const vv = arrays.verts.subarray(s * 2 * V, (s + 1) * 2 * V);
  for (let d = 0; d < D; d++) vv.set(dr.vertexPositions[d], 2 * vofs[d]);
  arrays.opacity.set(dr.opacities, s * D);
  arrays.drawOrder.set(dr.drawOrders, s * D);
  arrays.renderOrder.set(model.getRenderOrders().subarray(0, D), s * D);
  arrays.multiply.set(dr.multiplyColors, s * 4 * D);
  arrays.screen.set(dr.screenColors, s * 4 * D);
  arrays.dynFlags.set(dr.dynamicFlags, s * D);
}
const ms = tUpdate;

const fd = openSync(`${job.out}.bin`, 'w');
let off = 0;
const meta = { n: N, P, Q, D, V, vertexCounts: vcount, sections: {}, msPerUpdate: ms / Math.max(1, N), mocVersion: Core.Version.csmGetMocVersion(moc, buf), coreVersion: Core.Version.csmGetVersion() };
for (const [name, dt, n] of layout) {
  const a = arrays[name];
  writeSync(fd, new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
  meta.sections[name] = { offset: off, dtype: dt, shape: [N, n] };
  off += a.byteLength;
}
closeSync(fd);
meta.canvas = canvasMeta;
writeFileSync(`${job.out}.json`, JSON.stringify(meta));

console.log(JSON.stringify({ n: N, D, V, msPerUpdate: meta.msPerUpdate }));
