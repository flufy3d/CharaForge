/*
 * 几何 Worker（worker_threads）：用 packages/runtime 的 LayerEvaluator / ModelInstance（无 GL）按场景规范逐帧求值，
 * 在指定帧上输出采样顶点的画布坐标，以及「姿态/表情维度 → 顶点」的雅可比（有限差分，对物理之后的参数值做扰动）。
 *
 * 雅可比是「瞬时」的：物理状态与物理输出固定在当前轨迹的前向推进结果上，只扰动画面直接取用的参数
 * （姿态输出 postPhysics、表情参数、头发组叠加）。物理随轨迹的变化由外层迭代（重新前向推进）处理。
 */
import { parentPort, workerData } from 'node:worker_threads';
import { pathToFileURL } from 'node:url';
import { fileFetcher, loadRuntime } from './core.ts';

export interface DimSpec {
  key: string;
  /** 扰动时改哪些参数：[参数 ID, 系数]（参数变化 = 系数 × delta） */
  params: [string, number][];
  delta: number;
}
export interface JacRequest {
  layer: any;
  frames: number[];
  dims: DimSpec[];
  /** 采样顶点：[drawable, vertex] 交错 */
  sample: Int32Array;
}
export interface JacFrame {
  frame: number;
  /** 画布像素坐标，交错 x,y（长度 2K） */
  pos: Float32Array;
  /** drawable 有效不透明度（visible 且 opacity） */
  opa: Float32Array;
  /** 雅可比（行主序 2K × D，单位：画布像素 / 维度单位） */
  J: Float32Array;
}

const rt = await loadRuntime();
const res = await rt.ModelResource.load(null, pathToFileURL(workerData.modelPath).href, fileFetcher);
const inst = res.createInstance();
const model = inst.model;
const c = inst.canvas;
const noClip = (id: string): never => { throw new Error(`几何 Worker 不支持 motion/expression：${id}`); };

function readSample(sample: Int32Array, pos: Float32Array, opa: Float32Array) {
  const K = sample.length / 2;
  let lastD = -1;
  let v: Float32Array | null = null;
  let o = 0;
  for (let k = 0; k < K; k++) {
    const d = sample[2 * k]!, i = sample[2 * k + 1]!;
    if (d !== lastD) {
      v = model.getDrawableVertices(d) as Float32Array;
      o = model.getDrawableDynamicFlagIsVisible(d) ? model.getDrawableOpacity(d) : 0;
      lastD = d;
    }
    pos[2 * k] = c.originX + v![2 * i]! * c.pixelsPerUnit;
    pos[2 * k + 1] = c.originY - v![2 * i + 1]! * c.pixelsPerUnit;
    opa[k] = o;
  }
}

function jac(req: JacRequest): JacFrame[] {
  const ev = new rt.LayerEvaluator(req.layer, inst, 60, noClip, noClip);
  const K = req.sample.length / 2, D = req.dims.length;
  const out: JacFrame[] = [];
  const tmp = new Float32Array(2 * K), tmpO = new Float32Array(K);
  const idx = req.dims.map((d) => d.params.map(([id, coef]) => [inst.paramIndexOf(id), coef] as [number, number]));
  const values = model.getModel().parameters.values as Float32Array;
  for (const f of [...req.frames].sort((a, b) => a - b)) {
    ev.prepare(f, undefined, 'draw');
    const pos = new Float32Array(2 * K), opa = new Float32Array(K);
    readSample(req.sample, pos, opa);
    const base = values.slice();
    const J = new Float32Array(2 * K * D);
    for (let j = 0; j < D; j++) {
      const dlt = req.dims[j]!.delta;
      for (const [pi, coef] of idx[j]!) if (pi >= 0) inst.setParamByIndex(pi, base[pi]! + coef * dlt, 'set', 1);
      inst.update();
      readSample(req.sample, tmp, tmpO);
      for (let r = 0; r < 2 * K; r++) J[r * D + j] = (tmp[r]! - pos[r]!) / dlt;
      values.set(base);
    }
    inst.update();
    out.push({ frame: f, pos, opa, J });
  }
  return out;
}

/**
 * 物理脉冲响应（线性化）：在静止平衡态（base 参数）附近，物理输入 q 在第 0 帧起加 Δ 的阶跃，
 * 记录输出参数 r 的阶跃响应，差分得到脉冲响应 h_rq[τ]（τ = 0..taps−1）。物理用 runtime 的 ModelInstance。
 */
function impulse(req: { base: Record<string, number>; inputs: string[]; outputs: string[]; taps: number; delta: number }) {
  const run = (bump: string | null) => {
    inst.resetPhysics();
    inst.resetPose();
    inst.setParams(req.base);
    inst.stabilizePhysics();
    const rows: number[][] = [];
    for (let t = 0; t < req.taps; t++) {
      inst.resetPose();
      inst.setParams(req.base);
      if (bump) inst.setParam(bump, (req.base[bump] ?? 0) + req.delta);
      inst.stepPhysics(1 / 60);
      const v = inst.getParams();
      rows.push(req.outputs.map((o) => v[o] ?? 0));
    }
    return rows;
  };
  const ref = run(null);
  const out: Record<string, Record<string, number[]>> = {};
  for (const q of req.inputs) {
    const st = run(q);
    out[q] = {};
    req.outputs.forEach((r, ri) => {
      const step = st.map((row, t) => (row[ri]! - ref[t]![ri]!) / req.delta);
      out[q]![r] = step.map((v, t) => v - (t ? step[t - 1]! : 0));
    });
  }
  return out;
}

parentPort!.on('message', (msg: { id: number; req: JacRequest; counts?: boolean; impulse?: any }) => {
  if (msg.impulse) {
    parentPort!.postMessage({ id: msg.id, r: impulse(msg.impulse) });
    return;
  }
  if ((msg as any).ids) {
    parentPort!.postMessage({ id: msg.id, r: Array.from(model.getModel().drawables.ids as string[]) });
    return;
  }
  if (msg.counts) {
    const counts = Array.from({ length: model.getDrawableCount() }, (_, d) => model.getDrawableVertexCount(d));
    parentPort!.postMessage({ id: msg.id, r: counts });
    return;
  }
  try {
    const r = jac(msg.req);
    parentPort!.postMessage({ id: msg.id, r }, r.flatMap((x) => [x.pos.buffer, x.opa.buffer, x.J.buffer]) as ArrayBuffer[]);
  } catch (e) {
    parentPort!.postMessage({ id: msg.id, error: String((e as Error)?.stack ?? e) });
  }
});
parentPort!.postMessage({ ready: true, drawables: model.getDrawableCount() });
