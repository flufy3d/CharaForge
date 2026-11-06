/*
 * A5：VTS 前向模型的 TS 参考输出（tools/fit/src/vts.ts），给 tools/gpu/tests/test_a5_vts.py 对拍。
 *   node --experimental-strip-types tools/gpu/scripts/a5_vts_ref.ts <in.json> <out.json>
 * in.json：{ defaults: {参数: 缺省值}, inputs: {输入: number[]}, n, f0, smoothC, breathPeriod, breathPhase }
 * decouple 由环境变量 FIT_MODEL='{"decouple":true}' 控制（vts.ts 读 MODELX）。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { forward, loadVts } from '../../fit/src/vts.ts';

const [inPath, outPath] = process.argv.slice(2);
const q = JSON.parse(readFileSync(inPath!, 'utf8'));
const repo = join(import.meta.dirname, '../../..');
const m = loadVts(join(repo, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.vtube.json'), q.defaults);
const inputs: Record<string, Float64Array> = {};
for (const [k, v] of Object.entries(q.inputs as Record<string, number[]>)) inputs[k] = Float64Array.from(v);
const out = forward(m, inputs, q.n, q.f0, { smoothC: q.smoothC, breathPeriod: q.breathPeriod, breathPhase: q.breathPhase });
writeFileSync(outPath!, JSON.stringify({
  neutral: m.neutral,
  inputs: m.inputs,
  outputs: Object.fromEntries(Object.entries(out).map(([k, v]) => [k, Array.from(v)])),
}));
