// A2 物理对拍参考数据：用 packages/runtime 同一份代码（LayerEvaluator / ModelInstance + Framework 的 CubismPhysics）
// 逐帧记录「物理前的参数」与「物理后的参数」，供 tools/gpu/l2d/physics.py 对拍。
//
//   node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types tools/gpu/scripts/a2_ref_dump.ts [输出目录]
//
// 输出（缺省 .data/gpu/a2/ref/）：
//   model_params.json          模型参数表 { ids, min, max, default }（Core 的 Float32 值）
//   <name>.json                { name, kind, P, T, events: [{ kind: 'step'|'stab', dt }], frames?, layer? }
//   <name>.in.f32 / .out.f32   [T, P] Float32（小端），第 t 个事件之前 / 之后的全部参数值
//   <name>.rig.f64             [T, NO] Float64，事件之后 Framework 的 _currentRigOutputs（按 setting、output 顺序展平）
//   <name>.prev.f64            [T, NO] Float64，事件之后的 _previousRigOutputs
//   <name>.remain.f64          [T]      Float64，事件之后的 _currentRemainTime
// 序列：
//   场景图层：s24.main（含预热，帧 2278–2441）、s15.left、s30.full、s06.main（均 ≥ 300 帧，含预热）；
//   合成输入（直接驱动 ModelInstance，dt = 1/60，从加载时的初始物理状态开始）：
//     step 阶跃、sweep 正弦扫频、walk 随机游走、stab_walk（先 stabilization 再随机游走），
//     以及固定步长内插的额外检查：walk_dt50（dt = 1/50）、walk_dt144（dt = 1/144）、jitter（每帧随机 dt）。
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileFetcher, loadRuntime } from '../../fit/src/geom/core.ts';
import { MODEL_URI, REPO } from '../../fit/src/paths.ts';

const OUT = resolve(process.argv[2] ?? join(REPO, '.data/gpu/a2/ref'));
mkdirSync(OUT, { recursive: true });

const rt = await loadRuntime();
const res = await rt.ModelResource.load(null, pathToFileURL(join(REPO, 'assets', MODEL_URI)).href, fileFetcher);
const inst = res.createInstance();
const core = inst.model.getModel();
const values = core.parameters.values as Float32Array;
const P = core.parameters.count;
const ids: string[] = Array.from(core.parameters.ids as ArrayLike<string>);
const mins = Array.from(core.parameters.minimumValues as Float32Array);
const maxs = Array.from(core.parameters.maximumValues as Float32Array);
const defs = Array.from(core.parameters.defaultValues as Float32Array);
writeFileSync(join(OUT, 'model_params.json'), JSON.stringify({ ids, min: mins, max: maxs, default: defs }));

// 物理 JSON 的输入 / 输出参数 ID（检查是否都存在于模型）
const physJson = JSON.parse(readFileSync(join(REPO, 'assets', MODEL_URI.replace(/model3\.json$/, 'physics3.json')), 'utf8'));
const srcIds = new Set<string>(), dstIds = new Set<string>();
for (const s of physJson.PhysicsSettings) {
  for (const i of s.Input) srcIds.add(i.Source.Id);
  for (const o of s.Output) dstIds.add(o.Destination.Id);
}
const idSet = new Set(ids);
const missing = [...srcIds, ...dstIds].filter((x) => !idSet.has(x));
console.log(`参数 ${P} 个；物理输入 ${srcIds.size} 个、输出 ${dstIds.size} 个；模型里不存在的 ID：${missing.length ? missing.join(', ') : '无'}`);

// ---------- 记录 ----------
interface Internals {
  _currentRigOutputs: { outputs: number[] }[];
  _previousRigOutputs: { outputs: number[] }[];
  _currentRemainTime: number;
}
const phys = inst.physics as unknown as Internals;
const NO = phys._currentRigOutputs.reduce((a, o) => a + o.outputs.length, 0);

class Recorder {
  ins: Float32Array[] = [];
  outs: Float32Array[] = [];
  rig: number[][] = [];
  prev: number[][] = [];
  remain: number[] = [];
  events: { kind: 'step' | 'stab'; dt: number }[] = [];
  private after() {
    this.outs.push(values.slice());
    this.rig.push(phys._currentRigOutputs.flatMap((o) => o.outputs.slice()));
    this.prev.push(phys._previousRigOutputs.flatMap((o) => o.outputs.slice()));
    this.remain.push(phys._currentRemainTime);
  }
  step(dt: number, orig: (dt: number) => void) {
    this.ins.push(values.slice());
    orig(dt);
    this.after();
    this.events.push({ kind: 'step', dt });
  }
  stab(orig: () => void) {
    this.ins.push(values.slice());
    orig();
    this.after();
    this.events.push({ kind: 'stab', dt: 0 });
  }
  save(name: string, extra: Record<string, unknown>) {
    const T = this.ins.length;
    const cat32 = (a: Float32Array[]) => { const o = new Float32Array(T * P); a.forEach((r, t) => o.set(r, t * P)); return o; };
    const cat64 = (a: number[][], n: number) => { const o = new Float64Array(T * n); a.forEach((r, t) => o.set(r, t * n)); return o; };
    const w = (f: string, a: ArrayBufferView) => writeFileSync(join(OUT, f), Buffer.from(a.buffer, a.byteOffset, a.byteLength));
    w(`${name}.in.f32`, cat32(this.ins));
    w(`${name}.out.f32`, cat32(this.outs));
    w(`${name}.rig.f64`, cat64(this.rig, NO));
    w(`${name}.prev.f64`, cat64(this.prev, NO));
    w(`${name}.remain.f64`, Float64Array.from(this.remain));
    writeFileSync(join(OUT, `${name}.json`), JSON.stringify({ name, P, NO, T, events: this.events, ...extra }));
    console.log(`${name}: ${T} 个事件`);
  }
}

const origStep = inst.stepPhysics.bind(inst);
const origStab = inst.stabilizePhysics.bind(inst);
let rec: Recorder | null = null;
inst.stepPhysics = (dt: number) => (rec ? rec.step(dt, origStep) : origStep(dt));
inst.stabilizePhysics = () => (rec ? rec.stab(origStab) : origStab());

// ---------- 场景图层（LayerEvaluator，逐帧顺序推进，含预热） ----------
const scene = JSON.parse(readFileSync(join(REPO, 'scenes/deepseek-chan/scene.json'), 'utf8'));
const noClip = (id: string): never => { throw new Error(`不支持 motion/expression：${id}`); };
for (const layerId of ['s24.main', 's15.left', 's30.full', 's06.main']) {
  const layer = scene.layers.find((l: any) => l.id === layerId);
  rec = new Recorder();
  const ev = new rt.LayerEvaluator(layer, inst, scene.timebase.fps, noClip, noClip);
  for (let f = layer.in; f < layer.out; f++) ev.prepare(f, undefined, 'advance');
  const warm = layer.physics?.warmup ?? 0;
  rec.save(layerId.replace('.', '_'), { kind: 'layer', layer: layerId, frames: [layer.in - warm, layer.out - 1], stabilize: !!layer.physics?.stabilize });
  rec = null;
}

// ---------- 合成输入 ----------
let seed = 0x9e3779b9;
const rand = () => { // mulberry32
  seed = (seed + 0x6d2b79f5) | 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
const inputs = [...srcIds].filter((x) => idSet.has(x)).map((id) => ({ id, ...inst.paramRange(id)! }));

type Gen = (t: number) => Record<string, number>;
function synth(name: string, T: number, gen: Gen, opts: { dt?: number | ((t: number) => number); stabilize?: boolean } = {}) {
  inst.resetPhysics();
  rec = new Recorder();
  const dtOf = typeof opts.dt === 'function' ? opts.dt : () => (opts.dt as number | undefined) ?? 1 / 60;
  if (opts.stabilize) {
    inst.resetPose();
    inst.setParams(gen(0));
    inst.stabilizePhysics();
  }
  for (let t = 0; t < T; t++) {
    inst.resetPose();
    inst.setParams(gen(t));
    inst.stepPhysics(dtOf(t));
  }
  rec.save(name, { kind: 'synth', stabilize: !!opts.stabilize });
  rec = null;
}

// 阶跃：每个输入各自每隔 20～90 帧跳到一个随机值（1/4 概率跳到端点）
function stepGen(T: number): Gen {
  const tab = inputs.map((p) => {
    const vals = new Float64Array(T);
    let v = p.default, next = 0;
    for (let t = 0; t < T; t++) {
      if (t >= next) {
        const r = rand();
        v = r < 0.125 ? p.min : r < 0.25 ? p.max : p.min + (p.max - p.min) * rand();
        next = t + 20 + Math.floor(rand() * 70);
      }
      vals[t] = v;
    }
    return vals;
  });
  return (t) => Object.fromEntries(inputs.map((p, i) => [p.id, tab[i]![t]!]));
}
// 正弦扫频：频率 0.1 Hz → 8 Hz 指数扫，振幅 0.5～1.1 倍半量程（超出部分被 setParam 夹取）
function sweepGen(T: number): Gen {
  const ps = inputs.map(() => ({ amp: 0.5 + 0.6 * rand(), ph: 2 * Math.PI * rand(), k: 0.7 + 0.6 * rand() }));
  const f0 = 0.1, f1 = 8, dur = T / 60;
  return (t) => {
    const s = t / 60;
    const r = Math.log(f1 / f0) / dur;
    return Object.fromEntries(inputs.map((p, i) => {
      const q = ps[i]!;
      const phase = 2 * Math.PI * f0 * (Math.exp(r * q.k * s) - 1) / (r * q.k);
      const mid = (p.min + p.max) / 2, half = (p.max - p.min) / 2;
      return [p.id, mid + q.amp * half * Math.sin(phase + q.ph)];
    }));
  };
}
// 随机游走：带动量的布朗运动，碰到边界反弹
function walkGen(T: number, sigma = 0.04): Gen {
  const tab = inputs.map((p) => {
    const vals = new Float64Array(T);
    const R = p.max - p.min;
    let v = p.default, vel = 0;
    for (let t = 0; t < T; t++) {
      vel = 0.85 * vel + sigma * R * gauss();
      v += vel;
      if (v > p.max) { v = 2 * p.max - v; vel = -vel; }
      if (v < p.min) { v = 2 * p.min - v; vel = -vel; }
      v = Math.min(p.max, Math.max(p.min, v));
      vals[t] = v;
    }
    return vals;
  });
  return (t) => Object.fromEntries(inputs.map((p, i) => [p.id, tab[i]![t]!]));
}

synth('step', 600, stepGen(600));
synth('sweep', 720, sweepGen(720));
synth('walk', 900, walkGen(900));
synth('stab_walk', 600, walkGen(600), { stabilize: true });
synth('walk_dt50', 600, walkGen(600), { dt: 1 / 50 });
synth('walk_dt144', 600, walkGen(600), { dt: 1 / 144 });
{
  const dts = Array.from({ length: 600 }, () => 1 / 240 + (1 / 20 - 1 / 240) * rand());
  synth('jitter', 600, walkGen(600), { dt: (t) => dts[t]! });
}
console.log(`完成：${OUT}`);
