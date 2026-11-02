/*
 * 第三轮续：在「面捕输入空间」里做整段轨迹拟合。
 *
 * 前向模型：面捕输入（每个新画面帧一个节点，节点间线性插值）→ VTS 映射与平滑（vts.ts，读 vtube.json）
 *           → 模型参数（prePhysics 逐帧轨）→ runtime 的物理与求值 → 顶点 / 渲染。
 * 图层变换：VTS 里模型位置固定，平移/缩放/旋转只来自剪辑推拉 → 稀疏关键帧（缺省每 24 帧，zoom 镜头每 12 帧）线性插值。
 * 镜头常量：前倾 LeanInput、体型 Clothes2（多假设各自收敛，按 SSIM@360 挑）、呼吸相位/周期、外观（blur/resample/color）。
 * 全局常量：VTS 平滑系数 c（全片共用）。
 *
 * 一次 GN 迭代：渲染（harness）→ 光流（flow.py）→ 顶点雅可比（GeomPool，对物理之后的可见参数做有限差分）
 *   → 链式到输入节点：直接参数 = 映射斜率 × 指数平滑；姿态输出（ParamAngle*、ParamBodyAngle*）再卷积物理脉冲响应
 *     （runtime 物理在本镜头静止态附近线性化，worker 里测）→ 带平滑与 LM 阻尼的正规方程 → 更新。
 * 之后像素精修：逐节点小维度 CMA，损失为页面里的 SSIM 类损失。
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Shot } from '../../../packages/scene/src/index.ts';
import { CMAES } from './cmaes.ts';
import { exportLayer, type LayerPlan, type ModelCalib } from './fitshot.ts';
import type { GeomPool } from './geom/gpool.ts';
import { FIT_DIR, FIT_TAG, MODEL_URI, REPO } from './paths.ts';
import type { Pool } from './pool.ts';
import { RefFrames } from './refvideo.ts';
import { detectNewFrames, exclusionsAt, holdSegments, trackAt } from './shotsutil.ts';
import { bandSolve } from './traj.ts';
import { INPUT_DIMS, forward, linearLinks, type VtsModel } from './vts.ts';

const W = 1280, H = 720;
export const POSE_OUT = ['ParamAngleX', 'ParamAngleY', 'ParamAngleZ', 'ParamBodyAngleX', 'ParamBodyAngleY', 'ParamBodyAngleZ'];
export const POSE_IN = ['AngleX', 'AngleY', 'AngleZ', 'PositionX', 'PositionY', 'PositionZ'];
export const DIRECT = ['ParamEyeLOpen', 'ParamEyeROpen', 'EyeSmile', 'ParamEyeBallX', 'ParamEyeBallY', 'ParamBrowLY', 'ParamBrowRY', 'ParamMouthOpenY', 'ParamMouthForm'];
/**
 * 次级物理输出（头发、耳朵、飘带、袖子、胸、花边、头饰、马尾、帽子）：它们由面捕输入经物理驱动（如耳朵受眉毛与眼睛开合影响），
 * 在可见画面里占大块面积，所以也进雅可比链（线性化物理的脉冲响应）。
 */
const PHYS_JSON = JSON.parse(readFileSync(join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.physics3.json'), 'utf8'));
const PHYS_SOURCES = new Set<string>(PHYS_JSON.PhysicsSettings.flatMap((s: any) => s.Input.map((i: any) => i.Source.Id)));
export const SECONDARY: string[] = [...new Set<string>(PHYS_JSON.PhysicsSettings.flatMap((s: any) => s.Output.map((o: any) => o.Destination.Id)))]
  .filter((id) => /^(FrontHair|SideHair|LateHair|LateSideHair|ShortPonytail|LEar|REar|LFlip|RFlip|LControl|RControl|LRibbon|RRibbon|ribbon|LSleeve|RSleeve|Chest|Lace|RBow|LBow|LowX|Hat_|ParamBrowRAngle|ParamBrowLAngle|AngleXbs|AngleYbs|PupilX|PupilY|REyeball|LEyeball|REyes|LEyes|REyeLash|LEyeLash)/.test(id))
  .filter((id) => !POSE_OUT.includes(id));
const VIS = [...POSE_OUT, ...DIRECT, ...SECONDARY];
const TF_KEYS = ['x', 'y', 'ls', 'rot'] as const;
const TF_SD: Record<string, number> = { x: 4, y: 4, ls: 0.015, rot: 1 };
const TF_TAU: Record<string, number> = { x: 60, y: 60, ls: 0.1, rot: 4 };

export interface ShotConsts {
  lean: number;
  clothes2: number;
  breathPhase: number;
  breathPeriod: number;
  blur: number;
  resample: number;
  color: { brightness: number; contrast: number; saturation: number; multiply: [number, number, number] } | null;
}

/** 一个图层的完整状态：输入节点、变换关键帧、镜头常量 */
export interface LayerState {
  nodes: number[];
  inputs: Record<string, number>[];
  keys: number[];
  tf: { x: number; y: number; ls: number; rot: number }[];
  consts: ShotConsts;
  /** 圆形开窗 clip 的关键帧（clipRange 内；含不透明度 a） */
  clip?: ClipKey[];
  /** 景深模糊倍率（DOF 段 blur = dofK × shots 效果强度曲线） */
  dofK?: number;
}

const interp = (xs: number[], f: number): [number, number, number] => {
  // 返回 (左下标, 右下标, 右权重)；区间外保持端点
  if (f <= xs[0]!) return [0, 0, 0];
  if (f >= xs[xs.length - 1]!) return [xs.length - 1, xs.length - 1, 0];
  let i = 0;
  while (xs[i + 1]! <= f) i++;
  return [i, i + 1, (f - xs[i]!) / (xs[i + 1]! - xs[i]!)];
};

/** 逐帧的面捕输入（节点间线性插值） */
export function inputSeries(vts: VtsModel, st: LayerState, fin: number, fout: number): Record<string, Float64Array> {
  const n = fout - fin;
  const out: Record<string, Float64Array> = {};
  for (const name of vts.inputs) {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const [l, r, w] = interp(st.nodes, fin + i);
      a[i] = st.inputs[l]![name]! * (1 - w) + st.inputs[r]![name]! * w;
    }
    out[name] = a;
  }
  return out;
}

export function tfAt(st: LayerState, f: number) {
  const [l, r, w] = interp(st.keys, f);
  const a = st.tf[l]!, b = st.tf[r]!;
  return { x: a.x * (1 - w) + b.x * w, y: a.y * (1 - w) + b.y * w, ls: a.ls * (1 - w) + b.ls * w, rot: a.rot * (1 - w) + b.rot * w };
}

/** 图层 JSON（场景格式）：所有 VTS 输出参数为逐帧 prePhysics set 轨，变换为逐帧 baked */
export function buildLayer(p: LayerPlan, vts: VtsModel, st: LayerState, calib: ModelCalib, smoothC: number) {
  const n = p.out - p.in;
  const params = forward(vts, inputSeries(vts, st, p.in, p.out), n, p.in, { smoothC, breathPeriod: st.consts.breathPeriod, breathPhase: st.consts.breathPhase });
  const ramp = (sw: LayerPlan['switches'][number], g: number) => {
    if (g < sw.est) return sw.from;
    if (sw.fade <= 0 || g >= sw.est + sw.fade) return sw.to;
    return sw.from + (sw.to - sw.from) * (0.5 - 0.5 * Math.cos((Math.PI * (g - sw.est)) / sw.fade));
  };
  const tf = Array.from({ length: n }, (_, i) => tfAt(st, p.in + i));
  const ax = (calib.eyes[0][0] + calib.eyes[1][0]) / 2, ay = (calib.eyes[0][1] + calib.eyes[1][1]) / 2;
  const bk = (vals: ArrayLike<number>) => ({ baked: { start: p.in, values: Array.from(vals) } });
  const filters: unknown[] = [];
  if (st.consts.color) filters.push({ type: 'color', ...st.consts.color });
  filters.push({ type: 'resample', factor: st.consts.resample, method: 'bilinear' });
  // 景深：blur = 常量（DOF 段外）或 dofK × 效果强度（DOF 段内，强度曲线来自 shots 的 depthOfField 效果）
  const dofAt = (g: number) => {
    if (!p.blurRanges.some(([a, b]) => g >= a && g < b)) return null;
    return (st.dofK ?? 1) * (p.dofStrength ? p.dofStrength(g) : 8);
  };
  filters.push({ type: 'blur', sigma: bk(Array.from({ length: n }, (_, i) => dofAt(p.in + i) ?? st.consts.blur)) });
  // 圆形开窗 clip 与淡入（关键帧线性插值；区间外为覆盖全画面的大矩形、不透明）
  const clipAt = (g: number) => {
    const ks = st.clip;
    if (!ks || !p.clipRange || g < p.clipRange[0] || g >= p.clipRange[1]) return null;
    const [l, r, w] = interp(ks.map((k) => k.f), g);
    const A = ks[l]!, B = ks[r]!;
    const m = (k: 'cx' | 'cy' | 'r' | 'a') => A[k] * (1 - w) + B[k] * w;
    return { cx: m('cx'), cy: m('cy'), r: Math.max(0, m('r')), a: Math.min(1, Math.max(0, m('a'))) };
  };
  const clips = Array.from({ length: n }, (_, i) => clipAt(p.in + i));
  const scaleX = p.flipRamp ? Array.from({ length: n }, (_, i) => {
    const g = p.in + i, [a, b] = p.flipRamp!;
    return g >= b ? 1 : Math.max(0.02, (g - a) / (b - a));
  }) : null;
  const constants = Object.entries(p.constants).filter(([k]) => k !== 'Clothes2' && k !== 'LeanInput');
  return {
    id: p.id, kind: 'model', model: 'dsc', in: p.in, out: p.out,
    transform: { x: bk(tf.map((t) => t.x)), y: bk(tf.map((t) => t.y)), scale: bk(tf.map((t) => Math.exp(t.ls))), ...(scaleX ? { scaleX: bk(scaleX) } : {}), rotation: bk(tf.map((t) => t.rot)), anchorX: ax, anchorY: ay },
    ...(p.clipRange ? {
      clip: {
        x: bk(clips.map((c) => (c ? c.cx - c.r : -10000))), y: bk(clips.map((c) => (c ? c.cy - c.r : -10000))),
        width: bk(clips.map((c) => (c ? 2 * c.r : 30000))), height: bk(clips.map((c) => (c ? 2 * c.r : 30000))),
        radius: bk(clips.map((c) => (c ? c.r : 0))), feather: 3,
      },
      opacity: bk(clips.map((c) => (c ? c.a : 1))),
    } : {}),
    filters,
    // 没有 continues 的实例：起点不做 stabilize，从更早 90 帧开始用首帧输入预热（规范 4：预热帧用帧 in 的输入）
    physics: { warmup: 90, stabilize: false },
    params: [
      ...constants.map(([id, v]) => ({ id, curve: { keys: [{ t: p.in, v }] } })),
      { id: 'LeanInput', curve: bk(Array(n).fill(st.consts.lean)) },
      { id: 'Clothes2', curve: bk(Array(n).fill(st.consts.clothes2)) },
      ...p.switches.map((sw) => ({ id: sw.param, curve: bk(Array.from({ length: n }, (_, i) => ramp(sw, p.in + i))) })),
      ...Object.entries(params).map(([id, v]) => ({ id, curve: bk(v) })),
    ],
  };
}

export interface ClipKey { f: number; cx: number; cy: number; r: number; a: number }

export interface Fit2Options {
  /** 其他（已拟合或待拟合）图层：参与渲染与合成 */
  others?: () => unknown[];
  /** 本图层拟合期间隐藏的图层 */
  hidden?: string[];
  /** 只拟合这些面捕输入（其余保持中性值），例如强虚化的背景实例只解头部角度 */
  inputs?: string[];
  log?: (s: string) => void;
  smoothC: number;
  /** 镜头常量的固定值（跳过搜索），用于诊断 */
  fixedConsts?: Partial<ShotConsts>;
  itersA?: number;
  itersB?: number;
  refineRounds?: number;
  keyEvery?: number;
  samples?: number;
}

export async function fitInputs(pool: Pool, gpool: GeomPool, shot: Shot, p: LayerPlan, calib: ModelCalib, vts: VtsModel, opts: Fit2Options) {
  const log = opts.log ?? ((s: string) => console.log(s));
  const t0 = Date.now();
  const outDir = join(FIT_DIR, shot.id + FIT_TAG);
  const work = join(outDir, 'work');
  mkdirSync(work, { recursive: true });
  const ref = new RefFrames(shot.start, shot.end);
  const timing: Record<string, number> = {};
  const tick = (k: string, t: number) => (timing[k] = (timing[k] ?? 0) + (Date.now() - t) / 1000);
  const zOf = (id: string) => shot.instances.find((i) => i.id === id)?.z ?? 0;
  // 只拟合的输入子集（其余固定中性值）
  const fitInputs = opts.inputs ?? vts.inputs;
  const vtsFit: VtsModel = { ...vts, inputs: vts.inputs };
  void vtsFit;
  const Din = vts.inputs.length;
  const inputMask = vts.inputs.map((k) => fitInputs.includes(k));

  // ---- 节点、关键帧、初值 ----
  const { isNew } = detectNewFrames(ref, shot, p.inst, p.in, p.out);
  const nodes = [...isNew.entries()].filter(([, v]) => v).map(([f]) => f).sort((a, b) => a - b);
  const zoom = shot.effects.some((e) => e.kind === 'zoom');
  const every = opts.keyEvery ?? (zoom ? 12 : 24);
  const keys: number[] = [];
  for (let f = p.in; f < p.out - 1; f += every) keys.push(f);
  keys.push(p.out - 1);
  const mdx = calib.eyes[1][0] - calib.eyes[0][0];
  const eyeTf = (f: number) => {
    const e = trackAt(p.inst, 'eyes', f) as number[][] | null;
    if (!e) return { x: 640, y: 360, ls: Math.log(0.3), rot: 0 };
    return {
      x: (e[0]![0]! + e[1]![0]!) / 2, y: (e[0]![1]! + e[1]![1]!) / 2,
      ls: Math.log(Math.hypot(e[1]![0]! - e[0]![0]!, e[1]![1]! - e[0]![1]!) / mdx),
      rot: (Math.atan2(e[1]![1]! - e[0]![1]!, e[1]![0]! - e[0]![0]!) * 180) / Math.PI,
    };
  };
  const st: LayerState = {
    nodes,
    inputs: nodes.map(() => Object.fromEntries(vts.inputs.map((k) => [k, vts.neutral[k] ?? 0]))),
    keys,
    tf: keys.map(eyeTf),
    consts: {
      lean: 0, clothes2: p.constants.Clothes2 ?? 0, breathPhase: 0, breathPeriod: 3.5, blur: 1, resample: 1,
      color: p.colorFilter ? { brightness: 0, contrast: 1, saturation: 1, multiply: [1, 1, 1] } : null,
      ...(opts.fixedConsts ?? {}),
    },
  };
  if (p.clipRange) {
    const [a, b] = p.clipRange;
    st.clip = [];
    for (let f = a; f <= b; f += 8) {
      const t = tfAt(st, Math.min(f, p.out - 1));
      const u = (f - a) / (b - a);
      st.clip.push({ f, cx: t.x, cy: t.y + 80, r: 60 + 700 * u, a: Math.min(1, 0.2 + 1.6 * u) });
    }
  }
  st.dofK = 1;
  log(`[${shot.id}] ${p.id}: ${nodes.length} 个输入节点 × ${fitInputs.length} 个面捕输入，${keys.length} 个变换关键帧（每 ${every} 帧）`);

  // ---- 渲染 / 光流 / 打分 ----
  let layer = buildLayer(p, vts, st, calib, opts.smoothC);
  const setupPages = async () => {
    layer = buildLayer(p, vts, st, calib, opts.smoothC);
    const others = opts.others ? (opts.others() as any[]).filter((l) => l.id !== p.id) : [];
    const all = [...others, layer].sort((a: any, b: any) => (zOf(a.id) - zOf(b.id)));
    await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [all, MODEL_URI, calib.irisOffset] as const);
    await pool.all((h) => (window as any).__fit.setHidden(h), opts.hidden ?? []);
  };
  const renderFrames = async (frames: number[]) => {
    const t = Date.now();
    await setupPages();
    const n = pool.size;
    await Promise.all(pool.pages.map(async (pg, k) => {
      for (const f of frames.filter((_, i) => i % n === k)) {
        const img = (await pg.evaluate((f) => (window as any).__fit.render(f, 1), f)) as { data: string };
        writeFileSync(join(work, `ours_${f}.rgba`), Buffer.from(img.data, 'base64'));
      }
    }));
    tick('render', t);
  };
  const manifest = (frames: number[]) => {
    writeFileSync(join(work, 'manifest.json'), JSON.stringify({
      out: work,
      frames: frames.map((f) => ({ frame: f, ours: join(work, `ours_${f}.rgba`), ref: ref.file, refIndex: f - shot.start, excl: exclusionsAt(shot, f) })),
    }));
    return join(work, 'manifest.json');
  };
  const py = (script: string, man: string) => {
    const r = spawnSync(join(REPO, 'tools/.venv/bin/python'), ['-I', join(REPO, script), man], { encoding: 'utf8', maxBuffer: 1 << 26 });
    if (r.status !== 0) throw new Error(`${script} 失败：${r.stderr}`);
    return r.stdout;
  };
  const flows = (frames: number[]) => {
    const t = Date.now();
    py('tools/fit/flow.py', manifest(frames));
    const out = new Map<number, Float32Array>();
    for (const f of frames) {
      const b = readFileSync(join(work, `flow_${f}.f32`));
      out.set(f, new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4));
    }
    tick('flow', t);
    return out;
  };
  /** SSIM@360（与 metric.py 同定义）；渲染当前状态的这些帧 */
  const ssim360 = async (frames: number[]) => {
    await renderFrames(frames);
    const t = Date.now();
    const r = JSON.parse(py('tools/fit/score360.py', manifest(frames))) as Record<string, number | null>;
    tick('score', t);
    const vals = Object.values(r).filter((v): v is number => v !== null);
    return { mean: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0, min: vals.length ? Math.min(...vals) : 0, per: r };
  };

  // ---- 物理脉冲响应（本镜头静止态附近） ----
  let H_ = new Map<string, Map<string, number[]>>();
  const measureImpulse = async () => {
    jacCache = null;
    const base: Record<string, number> = { LeanInput: st.consts.lean, Clothes2: st.consts.clothes2 };
    // 物理输入：VTS 输出里同时是物理输入的参数（头身角度、眼开合、眉、嘴……）
    const ins = [...new Set(links.map((l) => l.output))].filter((o) => PHYS_SOURCES.has(o));
    const outs = [...POSE_OUT, ...SECONDARY];
    const deltaOf = (q: string) => (POSE_IN.includes(q) ? 2 : 0.1);
    const r: Record<string, Record<string, number[]>> = {};
    for (const q of ins) Object.assign(r, await gpool.impulse({ base, inputs: [q], outputs: outs, taps: 50, delta: deltaOf(q) }));
    H_ = new Map(ins.map((q) => [q, new Map(outs.filter((o) => r[q]![o]!.some((v) => Math.abs(v) > 1e-5)).map((o) => [o, r[q]![o]!]))]));
  };

  // ---- 灵敏度：可见参数（节点帧）对输入节点 ----
  const links = linearLinks(vts, opts.smoothC).filter((l) => vts.inputs.includes(l.input));
  const sens = () => {
    // S[fi] = { n0, r → Float64Array((n1−n0)·Din) }：只保留对帧 f 有影响的节点窗口（f 前 HORIZON 帧到 f 后一个节点）
    const N = nodes.length, n = p.out - p.in;
    const HORIZON = 90;
    const win = nodes.map((f) => {
      let n0 = 0;
      while (n0 < N - 1 && nodes[n0]! < f - HORIZON) n0++;
      let n1 = 0;
      while (n1 < N && nodes[n1]! <= f) n1++;
      n1 = Math.min(N, n1 + 1);
      return { n0, n1 };
    });
    const S = nodes.map((_, fi) => ({ n0: win[fi]!.n0, rows: VIS.map(() => new Float64Array((win[fi]!.n1 - win[fi]!.n0) * Din)) }));
    for (let nn = 0; nn < N; nn++) {
      // 节点 nn 的插值权重序列（只在 [前一节点, 后一节点] 非零）
      const w = new Float64Array(n);
      const ja = Math.max(0, (nn > 0 ? nodes[nn - 1]! : p.in) - p.in), jb = Math.min(n - 1, (nn + 1 < N ? nodes[nn + 1]! : p.out - 1) - p.in);
      for (let j = (nn === 0 ? 0 : ja); j <= (nn === N - 1 ? n - 1 : jb); j++) {
        const [l, r, ww] = interp(nodes, p.in + j);
        w[j] = (l === nn ? 1 - ww : 0) + (r === nn ? ww : 0) - (l === nn && r === nn ? 1 - ww : 0);
      }
      const fis = nodes.map((f, fi) => [f, fi] as const).filter(([, fi]) => nn >= win[fi]!.n0 && nn < win[fi]!.n1);
      for (let i = 0; i < Din; i++) {
        const name = vts.inputs[i]!;
        for (const lk of links.filter((x) => x.input === name)) {
          const q = new Float64Array(n);
          let prev = w[0]!;
          for (let j = 0; j < n; j++) { prev = prev + lk.alpha * (w[j]! - prev); q[j] = lk.slope * prev; }
          const vi = VIS.indexOf(lk.output);
          if (vi >= 0 && DIRECT.includes(lk.output)) for (const [f, fi] of fis) S[fi]!.rows[vi]![(nn - S[fi]!.n0) * Din + i] += q[f - p.in]!;
          const hq = H_.get(lk.output);
          if (hq) for (const [o, h] of hq) {
            const ri = VIS.indexOf(o);
            for (const [f, fi] of fis) {
              const j0 = f - p.in;
              let acc = 0;
              for (let tau = 0; tau < h.length && tau <= j0; tau++) acc += h[tau]! * q[j0 - tau]!;
              S[fi]!.rows[ri]![(nn - S[fi]!.n0) * Din + i] += acc;
            }
          }
        }
      }
    }
    return S;
  };

  // ---- 采样顶点 ----
  const counts = await gpool.vertexCounts();
  const total = counts.reduce((a, b) => a + b, 0);
  const step = Math.max(1, Math.round(total / (opts.samples ?? 4000)));
  const smp: number[] = [];
  { let k = 0; counts.forEach((cnt, d) => { for (let i = 0; i < cnt; i++, k++) if (k % step === 0) smp.push(d, i); }); }
  const sample = new Int32Array(smp);
  const dimSpecs = VIS.map((r) => ({ key: r, params: [[r, 1]] as [string, number][], delta: POSE_OUT.includes(r) ? 1 : 0.05 }));
  log(`[${shot.id}] ${p.id}: 可见参数 ${VIS.length} 个（姿态 ${POSE_OUT.length}、直接 ${DIRECT.length}、次级物理 ${SECONDARY.length}）`);

  // ---- GN ----
  const K = keys.length, N = nodes.length;
  const NT = K * 4 + N * Din;
  const sdOf = (c: number) => (c < K * 4 ? TF_SD[TF_KEYS[c % 4]!]! : INPUT_DIMS[vts.inputs[(c - K * 4) % Din]!]!.sd);
  let lastMedian = 20;
  const frameResid = new Map<number, number>();
  const framePts = new Map<number, number>();
  let jacCache: Map<number, import('./geom/worker.ts').JacFrame> | null = null;
  let gnCount = 0;
  const gn = async (lam: number, useInputs: boolean) => {
    const frames = nodes;
    await renderFrames(frames);
    const fl = flows(frames);
    let t = Date.now();
    // 雅可比每 3 次迭代全量重算一次，其余迭代只重算顶点位置（雅可比变化慢；全量重算 174 维很贵）
    let jac: Map<number, import('./geom/worker.ts').JacFrame>;
    if (!jacCache || gnCount % 3 === 0) {
      jac = await gpool.jacobians(layer, frames, dimSpecs, sample);
      jacCache = jac;
    } else {
      const pos = await gpool.jacobians(layer, frames, [], sample);
      jac = new Map([...pos].map(([f, x]) => [f, { ...x, J: jacCache.get(f)!.J }]));
    }
    gnCount++;
    tick('jac', t);
    t = Date.now();
    const S = sens();
    const Hm = new Float64Array(NT * NT), g = new Float64Array(NT);
    const stats: number[] = [];
    const cauchy = Math.max(3, 1.5 * lastMedian);
    const L = layer as any;
    const ax = L.transform.anchorX, ay = L.transform.anchorY;
    nodes.forEach((f, fi) => {
      const T = tfAt(st, f), j = jac.get(f)!, F = fl.get(f)!;
      const Sc = Math.exp(T.ls), rr = (T.rot * Math.PI) / 180, cs = Math.cos(rr), sn = Math.sin(rr);
      const V = VIS.length, Kp = j.opa.length;
      const M = new Float64Array((4 + V) * (4 + V)), b = new Float64Array(4 + V);
      let cnt = 0;
      const rows: { J: Float64Array; e: [number, number] }[] = [];
      for (let k = 0; k < Kp; k++) {
        if (j.opa[k]! < 0.5) continue;
        const dx = j.pos[2 * k]! - ax, dy = j.pos[2 * k + 1]! - ay;
        const px = T.x + Sc * (cs * dx - sn * dy), py = T.y + Sc * (sn * dx + cs * dy);
        const u = px / 2 - 0.25, v = py / 2 - 0.25;
        const x0 = Math.floor(u), y0 = Math.floor(v);
        if (x0 < 0 || y0 < 0 || x0 >= 639 || y0 >= 359) continue;
        const fx = u - x0, fy = v - y0;
        let vx = 0, vy = 0, vv = 1;
        for (const [ox, oy, wgt] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]] as const) {
          const o = ((y0 + oy) * 640 + x0 + ox) * 3;
          vx += wgt * F[o]!; vy += wgt * F[o + 1]!; vv = Math.min(vv, F[o + 2]!);
        }
        if (vv < 0.5) continue;
        const J = new Float64Array(2 * (4 + V));
        // 变换（x, y, ls, rot）
        J[0] = 1; J[4 + V + 1] = 1;
        J[2] = px - T.x; J[4 + V + 2] = py - T.y;
        J[3] = (-(py - T.y) * Math.PI) / 180; J[4 + V + 3] = ((px - T.x) * Math.PI) / 180;
        for (let r = 0; r < V; r++) {
          const cx = j.J[2 * k * V + r]!, cy = j.J[(2 * k + 1) * V + r]!;
          J[4 + r] = Sc * (cs * cx - sn * cy); J[4 + V + 4 + r] = Sc * (sn * cx + cs * cy);
        }
        rows.push({ J, e: [vx, vy] });
        stats.push(Math.hypot(vx, vy));
        cnt++;
      }
      if (!cnt) return;
      // 边界帧（镜头首末两个节点、排除区刚结束的帧）数据项加权：它们的观测窗口短，只靠平滑会被拖偏
      const prevEx = exclusionsAt(shot, f - 2).length > exclusionsAt(shot, f).length;
      const boundary = fi < 2 || fi >= nodes.length - 2 || prevEx;
      const norm = (600 / cnt) * (boundary ? 3 : 1);
      const fm = rows.map((r) => Math.hypot(r.e[0], r.e[1])).sort((x, y) => x - y);
      frameResid.set(f, fm[fm.length >> 1]!);
      framePts.set(f, cnt);
      const D = 4 + V;
      for (const rw of rows) {
        const m = Math.hypot(rw.e[0], rw.e[1]);
        const wt = norm / (1 + (m / cauchy) ** 2);
        for (let a = 0; a < D; a++) {
          const ja = rw.J[a]!, jb = rw.J[D + a]!;
          b[a] += wt * (ja * rw.e[0] + jb * rw.e[1]);
          for (let c = 0; c <= a; c++) M[a * D + c] += wt * (ja * rw.J[c]! + jb * rw.J[D + c]!);
        }
      }
      for (let a = 0; a < D; a++) for (let c = 0; c < a; c++) M[c * D + a] = M[a * D + c]!;
      // 链式矩阵 C（D × NT，归一化单位）：变换行 → 关键帧插值权重；可见参数行 → 灵敏度
      const [kl, kr, kw] = interp(keys, f);
      const cols: { c: number; row: Float64Array }[] = [];
      const colMap = new Map<number, Float64Array>();
      const addC = (c: number, row: number, v: number) => {
        let col = colMap.get(c);
        if (!col) { col = new Float64Array(D); colMap.set(c, col); }
        col[row] += v * sdOf(c);
      };
      for (let a = 0; a < 4; a++) {
        addC(kl * 4 + a, a, kr === kl ? 1 : 1 - kw);
        if (kr !== kl) addC(kr * 4 + a, a, kw);
      }
      if (useInputs) for (let r = 0; r < V; r++) {
        const sr = S[fi]!.rows[r]!, off = S[fi]!.n0 * Din;
        for (let c = 0; c < sr.length; c++) if (Math.abs(sr[c]!) > 1e-7 && inputMask[c % Din]) addC(K * 4 + off + c, 4 + r, sr[c]!);
      }
      for (const [c, row] of colMap) cols.push({ c, row });
      // H += Cᵀ M C，g += Cᵀ b
      const MC = cols.map(({ row }) => {
        const out = new Float64Array(D);
        for (let a = 0; a < D; a++) { let s = 0; for (let c2 = 0; c2 < D; c2++) s += M[a * D + c2]! * row[c2]!; out[a] = s; }
        return out;
      });
      cols.forEach(({ c, row }, ci) => {
        let s = 0;
        for (let a = 0; a < D; a++) s += row[a]! * b[a]!;
        g[c] += s;
        cols.forEach(({ c: c2 }, cj) => {
          let v = 0;
          const mc = MC[cj]!;
          for (let a = 0; a < D; a++) v += row[a]! * mc[a]!;
          Hm[c * NT + c2] += v;
        });
      });
    });
    // 平滑：输入节点二阶差分；变换关键帧二阶差分
    const smooth = (idx: (m: number) => number, cnt: number, sd: number, tau: number, cur: (m: number) => number, lw: number) => {
      const sc = sd / tau;
      for (let m = 1; m + 1 < cnt; m++) {
        const ids = [idx(m - 1), idx(m), idx(m + 1)], a = [1, -2, 1];
        const r0 = (cur(m - 1) - 2 * cur(m) + cur(m + 1)) / tau;
        for (let i1 = 0; i1 < 3; i1++) {
          g[ids[i1]!] -= lw * sc * a[i1]! * r0;
          for (let i2 = 0; i2 < 3; i2++) Hm[ids[i1]! * NT + ids[i2]!] += lw * sc * sc * a[i1]! * a[i2]!;
        }
      }
    };
    TF_KEYS.forEach((key, a) => smooth((m) => m * 4 + a, K, TF_SD[key]!, TF_TAU[key]!, (m) => st.tf[m]![key], lam));
    if (useInputs) vts.inputs.forEach((name, i) => smooth((m) => K * 4 + m * Din + i, N, INPUT_DIMS[name]!.sd, INPUT_DIMS[name]!.tau, (m) => st.inputs[m]![name]!, lam));
    let tr = 0;
    for (let i = 0; i < NT; i++) tr += Hm[i * NT + i]!;
    for (let i = 0; i < NT; i++) Hm[i * NT + i] = Hm[i * NT + i]! * 1.05 + 1e-6 * (tr / NT) + 1e-3;
    const du = bandSolve(Hm, g, NT, NT);
    // 信赖域：变换的位移与输入的变化按 sd 单位截断
    stats.sort((a, b) => a - b);
    const med = stats.length ? stats[stats.length >> 1]! : 0;
    let mx = 0;
    for (let i = 0; i < NT; i++) mx = Math.max(mx, Math.abs(du[i]!));
    const lim = Math.max(2, 0.7 * med) * 2; // 以 sd 计（变换 sd = 4px），粗略对应位移
    const scale = mx > lim ? lim / mx : 1;
    for (let k = 0; k < K; k++) TF_KEYS.forEach((key, a) => (st.tf[k]![key] += du[k * 4 + a]! * scale * TF_SD[key]!));
    if (useInputs) for (let m = 0; m < N; m++) vts.inputs.forEach((name, i) => {
      if (!inputMask[i]) return;
      const d = INPUT_DIMS[name]!;
      st.inputs[m]![name] = Math.min(d.hi, Math.max(d.lo, st.inputs[m]![name]! + du[K * 4 + m * Din + i]! * scale * d.sd));
    });
    tick('solve', t);
    lastMedian = med;
    return { median: med, p90: stats.length ? stats[Math.floor(stats.length * 0.9)]! : NaN, points: stats.length };
  };

  const runGN = async (iters: number, tag: string, lam: number, useInputs = true) => {
    let r = { median: NaN, p90: NaN, points: 0 };
    for (let it = 0; it < iters; it++) {
      r = await gn(lam, useInputs);
      log(`[${shot.id}] ${p.id} ${tag} #${it}: 光流残差中位 ${r.median.toFixed(2)}px、p90 ${r.p90.toFixed(2)}px（${r.points} 点）`);
    }
    return r;
  };

  // ---- (1) 只解变换，几轮 ----
  await measureImpulse();
  await runGN(3, '变换', 20, false);
  // ---- (2) 镜头常量：多假设各自完整收敛，按 SSIM@360 挑 ----
  const probe = nodes.filter((_, i) => i % Math.max(1, Math.floor(nodes.length / 6)) === 0);
  const snapshot = () => JSON.parse(JSON.stringify(st)) as LayerState;
  const restore = (s: LayerState) => { st.inputs = s.inputs; st.tf = s.tf; st.consts = s.consts; };
  // 先体型（差别大，各 2 轮即可分开），再前倾（与身体俯仰简并，各自收敛 4 轮）；都按 SSIM@360 挑
  const base = snapshot();
  let best: { s: LayerState; score: number; h: Partial<ShotConsts> } | null = null;
  const trial = async (h: Partial<ShotConsts>, iters: number, from: LayerState) => {
    restore(JSON.parse(JSON.stringify(from)));
    Object.assign(st.consts, h);
    lastMedian = 20;
    await measureImpulse();
    await runGN(iters, `假设 ${JSON.stringify(h)}`, 10);
    const sc = await ssim360(probe);
    log(`[${shot.id}] ${p.id} 假设 ${JSON.stringify(h)} → SSIM@360（抽样 ${probe.length} 帧）均值 ${sc.mean.toFixed(3)}、最小 ${sc.min.toFixed(3)}`);
    if (!best || sc.mean > best.score) best = { s: snapshot(), score: sc.mean, h: { ...(best?.h ?? {}), ...h } };
  };
  const c2s = opts.fixedConsts?.clothes2 !== undefined ? [opts.fixedConsts.clothes2] : [...new Set([st.consts.clothes2, 0, 0.4, 1])];
  const leans = opts.fixedConsts?.lean !== undefined ? [opts.fixedConsts.lean] : [0, -15, -30];
  if (c2s.length > 1) for (const c2 of c2s) await trial({ clothes2: c2 }, 2, base);
  const afterC2 = best ? (best as { s: LayerState }).s : base;
  const c2best = best ? (best as { h: Partial<ShotConsts> }).h.clothes2 : c2s[0];
  best = null;
  for (const le of leans) await trial({ clothes2: c2best, lean: le }, leans.length > 1 ? 4 : 0, afterC2);
  // 前倾细化：在最优档两侧再试 ±7.5、±3.75（从当前最优状态出发）
  if (leans.length > 1) for (const stepL of [7.5, 3.75]) {
    const b0 = (best as unknown as { s: LayerState; h: Partial<ShotConsts> });
    const center = b0.h.lean!;
    for (const le of [center - stepL, center + stepL].filter((v) => v >= -30 && v <= 0)) await trial({ clothes2: c2best, lean: le }, 3, b0.s);
  }
  restore(best!.s);
  log(`[${shot.id}] ${p.id} 选定镜头常量 ${JSON.stringify(best!.h)}`);
  await measureImpulse();
  // ---- (3) 继续 GN ----
  await runGN(opts.itersB ?? 6, '细', 5);
  // ---- (4) 外观常量 + 呼吸（像素损失）----
  // 外观常量只在几何残差最低的 30% 帧上拟合（防止用模糊掩盖几何误差），最多 5 帧
  let appFrames: number[] = [];
  const appearance = async () => {
    // 只在观测充分（有效光流点数 ≥ 最大值 60%）的帧里挑残差最低的
    const maxPts = Math.max(...framePts.values());
    const ranked = [...frameResid.entries()].filter(([f]) => nodes.includes(f) && (framePts.get(f) ?? 0) >= 0.6 * maxPts).sort((a, b) => a[1] - b[1]);
    appFrames = ranked.slice(0, Math.max(3, Math.min(5, Math.ceil(ranked.length * 0.3)))).map(([f]) => f);
    log(`[${shot.id}] ${p.id} 外观拟合帧（光流残差最低）：${appFrames.join(',')}（残差 ${appFrames.map((f) => frameResid.get(f)!.toFixed(2)).join('/')} px）`);
    // 清晰度（blur、resample）用与几何无关的梯度能量匹配估计（tools/fit/sharp.py）：先渲染不加这两个滤镜的版本
    if (!p.blurRanges.length) {
      const keep = { blur: st.consts.blur, resample: st.consts.resample };
      st.consts.blur = 0; st.consts.resample = 1;
      await renderFrames(appFrames);
      const r = JSON.parse(py('tools/fit/sharp.py', manifest(appFrames))) as { blur: number; resample: number; dist: number; frames: number };
      if (r.frames > 0) { st.consts.blur = r.blur; st.consts.resample = r.resample; } else Object.assign(st.consts, keep);
      log(`[${shot.id}] ${p.id} 清晰度（梯度能量匹配，${r.frames} 帧）：blur ${st.consts.blur}、resample ${st.consts.resample}（距离 ${r.dist.toFixed(4)}）`);
    }
    await setupPages();
    const dims: { key: string; lo: number; hi: number; sd: number }[] = [
      // VTS 自动呼吸的相位与周期（经物理影响身体与头发；候选改写 [in, f] 的 ParamBreath，物理从头重放）
      { key: 'bph', lo: -Math.PI, hi: Math.PI, sd: 0.8 }, { key: 'bper', lo: 2.5, hi: 6, sd: 0.5 },
      ...(p.blurRanges.length ? [{ key: 'dofK', lo: 0.4, hi: 1.6, sd: 0.15 }] : []),
      ...(st.consts.color ? [
        { key: 'brightness', lo: -0.3, hi: 0.3, sd: 0.03 }, { key: 'contrast', lo: 0.6, hi: 1.5, sd: 0.05 }, { key: 'saturation', lo: 0.5, hi: 1.6, sd: 0.06 },
        { key: 'mr', lo: 0.7, hi: 1.3, sd: 0.03 }, { key: 'mg', lo: 0.7, hi: 1.3, sd: 0.03 }, { key: 'mb', lo: 0.7, hi: 1.3, sd: 0.03 },
      ] : []),
    ];
    const c = st.consts;
    const x0 = dims.map((d) => ({ dofK: st.dofK ?? 1, blur: c.blur, resample: c.resample, bph: c.breathPhase, bper: c.breathPeriod, brightness: c.color?.brightness ?? 0, contrast: c.color?.contrast ?? 1, saturation: c.color?.saturation ?? 1, mr: c.color?.multiply[0] ?? 1, mg: c.color?.multiply[1] ?? 1, mb: c.color?.multiply[2] ?? 1 } as Record<string, number>)[d.key]!);
    const vals = (x: number[]) => Object.fromEntries(dims.map((d, i) => [d.key, Math.min(d.hi, Math.max(d.lo, x[i]!))]));
    const cand = (v: Record<string, number>) => ({ blur: v.blur ?? st.consts.blur, resample: v.resample ?? st.consts.resample, ...(st.consts.color ? { color: { brightness: v.brightness, contrast: v.contrast, saturation: v.saturation, multiply: [v.mr, v.mg, v.mb] } } : {}) });
    const breathSet = (v: Record<string, number>, f: number) => {
      const set: [number, Record<string, number>][] = [];
      for (let g = p.in; g <= f; g++) set.push([g, { ParamBreath: 0.5 + 0.5 * Math.sin((2 * Math.PI * g) / 60 / v.bper! + v.bph!) }]);
      return set;
    };
    const loss = async (cs: Record<string, number>[]) => {
      const acc = cs.map(() => 0);
      for (const f of appFrames) {
        await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
        const inDof = p.blurRanges.some(([a, b]) => f >= a && f < b);
        const rs = await pool.evalBatch<{ loss: number }>(f, cs.map((v) => ({ [p.id]: { ...cand(v), ...(inDof ? { blur: v.dofK! * (p.dofStrength ? p.dofStrength(f) : 8) } : {}), set: breathSet(v, f) } })), { scale: 0.5, r: 3, wcol: 1, weye: 0, wseg: 1, wgrad: 0.5, eyesOf: [] });
        rs.forEach((r, i) => (acc[i] += r.loss));
      }
      return acc;
    };
    const es = new CMAES(x0, { sigma: 1, scales: dims.map((d) => d.sd), lambda: 12, seed: 3 });
    let bestA = { c: (await loss([vals(x0)]))[0]!, v: vals(x0) };
    for (let gen = 0; gen < 15; gen++) {
      const xs = es.ask();
      const cs = await loss(xs.map(vals));
      es.tell(cs);
      cs.forEach((cc, i) => { if (cc < bestA.c) bestA = { c: cc, v: vals(xs[i]!) }; });
    }
    if (bestA.v.blur !== undefined) st.consts.blur = bestA.v.blur;
    if (bestA.v.resample !== undefined) st.consts.resample = bestA.v.resample;
    st.consts.breathPhase = bestA.v.bph!; st.consts.breathPeriod = bestA.v.bper!;
    if (bestA.v.dofK !== undefined) st.dofK = bestA.v.dofK;
    if (st.consts.color) st.consts.color = { brightness: bestA.v.brightness!, contrast: bestA.v.contrast!, saturation: bestA.v.saturation!, multiply: [bestA.v.mr!, bestA.v.mg!, bestA.v.mb!] };
    log(`[${shot.id}] ${p.id} 外观常量 ${JSON.stringify(Object.fromEntries(Object.entries(bestA.v).map(([k, v]) => [k, +v.toFixed(3)])))}`);
  };
  const t4 = Date.now();
  await appearance();
  tick('appearance', t4);
  await runGN(2, '收尾', 5);
  // ---- (4b) 开窗 clip 与淡入：逐关键帧 CMA（圆心、半径、不透明度），页面损失 ----
  if (st.clip) {
    const t = Date.now();
    await setupPages();
    for (const k of st.clip) {
      const f = Math.min(k.f, p.out - 1);
      await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
      const x0 = [k.cx, k.cy, k.r, k.a];
      const es = new CMAES(x0, { sigma: 1, scales: [20, 20, 40, 0.15], lambda: 10, seed: f });
      const cand = (x: number[]) => ({ clip: [x[0]!, x[1]!, Math.max(0, x[2]!)], opacity: Math.min(1, Math.max(0, x[3]!)) });
      const lossX = async (xs: number[][]) => (await pool.evalBatch<{ loss: number }>(f, xs.map((x) => ({ [p.id]: cand(x) })), { scale: 0.5, r: 3, wcol: 1, weye: 0, wseg: 1, wgrad: 0.3, eyesOf: [] })).map((r) => r.loss);
      let bestK = { c: (await lossX([x0]))[0]!, x: x0 };
      for (let gen = 0; gen < 10; gen++) {
        const xs = es.ask();
        const cs = await lossX(xs);
        es.tell(cs);
        cs.forEach((c, i) => { if (c < bestK.c) bestK = { c, x: xs[i]! }; });
      }
      Object.assign(k, { cx: bestK.x[0], cy: bestK.x[1], r: Math.max(0, bestK.x[2]!), a: Math.min(1, Math.max(0, bestK.x[3]!)) });
    }
    log(`[${shot.id}] ${p.id} 开窗 clip：${st.clip.map((k) => `${k.f}:(${k.cx.toFixed(0)},${k.cy.toFixed(0)},r${k.r.toFixed(0)},a${k.a.toFixed(2)})`).join(' ')}`);
    tick('clip', t);
  }

  // ---- (5) 像素精修：逐节点（时间正序）小维度 CMA，页面里的 SSIM 类损失在该节点帧上打分 ----
  //   候选只改本节点的面捕输入；前向模型（映射 + 平滑）重算后，把 [上一节点, 本节点] 的参数写进页面（之后的帧下一个节点再处理）
  if ((opts.refineRounds ?? 0) > 0) {
    const t = Date.now();
    const before = await ssim360(nodes);
    log(`[${shot.id}] ${p.id} 像素精修前 SSIM@360 均值 ${before.mean.toFixed(3)}、最小 ${before.min.toFixed(3)}`);
    await setupPages();
    const rdims = ['FaceAngleX', 'FaceAngleY', 'FaceAngleZ', 'EyeOpenLeft', 'EyeOpenRight', 'EyeRightX', 'EyeRightY'];
    const writesFor = (n: number) => {
      const out = forward(vts, inputSeries(vts, st, p.in, p.out), p.out - p.in, p.in, { smoothC: opts.smoothC, breathPeriod: st.consts.breathPeriod, breathPhase: st.consts.breathPhase });
      const a = n > 0 ? nodes[n - 1]! : p.in, b = Math.min(p.out - 1, (nodes[n + 1] ?? p.out) - 1);
      const set: [number, Record<string, number>][] = [];
      for (let g = a; g <= b; g++) set.push([g, Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v[g - p.in]!]))]);
      return set;
    };
    for (let round = 0; round < opts.refineRounds!; round++) {
      for (let n = 0; n < nodes.length; n++) {
        const f = nodes[n]!;
        if (!exclusionsAt(shot, f).length || true) {
          await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
        }
        const x0 = rdims.map((k) => st.inputs[n]![k]!);
        const es = new CMAES(x0, { sigma: 0.7, scales: rdims.map((k) => INPUT_DIMS[k]!.sd), lambda: 8, seed: 11 + n });
        const evalX = async (xs: number[][]) => {
          const cands = xs.map((x) => {
            const keep = { ...st.inputs[n]! };
            rdims.forEach((k, i) => (st.inputs[n]![k] = Math.min(INPUT_DIMS[k]!.hi, Math.max(INPUT_DIMS[k]!.lo, x[i]!))));
            const set = writesFor(n);
            st.inputs[n] = keep;
            return { [p.id]: { set } };
          });
          const rs = await pool.evalBatch<{ loss: number }>(f, cands, { scale: 0.5, r: 3, wcol: 1, weye: 0, wseg: 1, wgrad: 0.3, eyesOf: [] });
          return rs.map((r) => r.loss);
        };
        let bestN = { c: (await evalX([x0]))[0]!, x: x0 };
        for (let gen = 0; gen < 6; gen++) {
          const xs = es.ask();
          const cs = await evalX(xs);
          es.tell(cs);
          cs.forEach((c, i) => { if (c < bestN.c) bestN = { c, x: xs[i]! }; });
        }
        rdims.forEach((k, i) => (st.inputs[n]![k] = Math.min(INPUT_DIMS[k]!.hi, Math.max(INPUT_DIMS[k]!.lo, bestN.x[i]!))));
        await pool.all(([lid, f, v]) => (window as any).__fit.commit(f, { [lid]: v }), [p.id, f, { set: writesFor(n) }] as const);
      }
      const after = await ssim360(nodes);
      log(`[${shot.id}] ${p.id} 像素精修第 ${round + 1} 轮后 SSIM@360 均值 ${after.mean.toFixed(3)}、最小 ${after.min.toFixed(3)}`);
    }
    tick('refine', t);
  }

  const endScore = await ssim360(nodes);
  log(`[${shot.id}] ${p.id} 结束：SSIM@360（全部节点帧，拟合内部打分）均值 ${endScore.mean.toFixed(3)}、最小 ${endScore.min.toFixed(3)}`);

  // ---- 导出 ----
  const final = buildLayer(p, vts, st, calib, opts.smoothC) as any;
  const params: Record<string, number[]> = {};
  for (const t of final.params) if (t.curve.baked) params[t.id] = t.curve.baked.values;
  const d = {
    tf: { x: final.transform.x.baked.values, y: final.transform.y.baked.values, scale: final.transform.scale.baked.values, rotation: final.transform.rotation.baked.values },
    opacity: null, blur: final.filters.find((f: any) => f.type === 'blur').sigma.baked.values, clip: null,
    color: st.consts.color, resample: st.consts.resample, sx: null, params,
  };
  const base2 = { layers: [{ id: p.id, transform: final.transform, params: final.params }] } as any;
  const holds = holdSegments(isNew, p.in, p.out);
  const out = exportLayer(base2, p, d as any, { log: nodes } as any, holds, shot) as any;
  for (const t of out.params) if (t.provenance?.detail) t.provenance.detail.what = 'VTS forward model: face-tracking inputs (fitted, per new frame) → mapping + smoothing';
  out.provenance.detail.method = 'face-input-space trajectory GN on optical-flow correspondences + pixel refinement (round 3b)';
  out.provenance.detail.vts = { smoothC: opts.smoothC, consts: st.consts };
  const elapsed = (Date.now() - t0) / 1000;
  log(`[${shot.id}] ${p.id} 完成：${elapsed.toFixed(0)}s；分项 ${JSON.stringify(Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v)])))}`);
  ref.close();
  return { layer: out, state: st, elapsed, timing, score: endScore, built: final };
}
