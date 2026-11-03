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
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
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
import { MODELX } from './modelx.ts';
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
  .filter((id) => !POSE_OUT.includes(id))
  // MODELX.bodyJac：身体链的次级物理输出（肩、盆骨、身体位移、前倾）
  .concat(MODELX.bodyJac ? [...new Set<string>(PHYS_JSON.PhysicsSettings.flatMap((s: any) => s.Output.map((o: any) => o.Destination.Id)))]
    .filter((id) => /^(Shoulder|HipAngle|ParamHipAngle|HipZ|ParamBodyAngleRX|ParamBodyAngleX2|ParamBodyAngleY2|Param5[034]$|LEGAngle)/.test(id)) : []);
// MODELX.consts：镜头常量（前倾 LeanInput、体型 Clothes2）作为连续的全局未知量进 GN（直接形变 + 物理稳态增益）
const CONST_PARAMS = MODELX.consts ? ['LeanInput', 'Clothes2'] : [];
const CONST_SD: Record<string, number> = { LeanInput: 3, Clothes2: 0.03 };
const CONST_BOX: Record<string, [number, number]> = { LeanInput: [-30, 30], Clothes2: [0, 1] };
const VIS = [...POSE_OUT, ...DIRECT, ...SECONDARY, ...CONST_PARAMS];
const TF_KEYS = ['x', 'y', 'ls', 'rot'] as const;
const TF_SD: Record<string, number> = { x: 4, y: 4, ls: 0.015, rot: 1 };
const TF_TAU: Record<string, number> = { x: 60, y: 60, ls: 0.1, rot: 4 };

/** 内存阈值：软（降低批大小）8.5GB、硬（报错退出）10.3GB；cgroup 统计含 Chromium 与几何 Worker */
const MEM_SOFT = 8.5 * 2 ** 30, MEM_HARD = 10.3 * 2 ** 30;
function cgroupBytes(): number {
  try {
    const rel = readFileSync('/proc/self/cgroup', 'utf8').split('\n').find((l) => l.startsWith('0::'))?.slice(3).trim();
    if (rel) {
      // memory.current 含页缓存（参考帧缓存文件等，可回收）：减去 inactive_file
      const cur = Number(readFileSync(`/sys/fs/cgroup${rel}/memory.current`, 'utf8'));
      const ina = Number(/^inactive_file (\d+)/m.exec(readFileSync(`/sys/fs/cgroup${rel}/memory.stat`, 'utf8'))?.[1] ?? 0);
      return cur - ina;
    }
  } catch { /* 无 cgroup v2 */ }
  return process.memoryUsage().rss;
}

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
  /** 诊断（MODELX.upper）：逐节点帧的自由偏移——可见参数 postPhysics add 与变换 [dx, dy, dls, drot]，用来测表达能力上限 */
  free?: { f: number; vis: Record<string, number>; tf: [number, number, number, number] }[];
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
  // 逐帧精修偏移：节点帧上的值，节点之间线性插值（区间外保持端点）
  const fr = [...(st.free ?? [])].sort((a, b) => a.f - b.f);
  const freeAt = new Map<number, { vis: Record<string, number>; tf: number[] }>();
  if (fr.length) for (let i = 0; i < n; i++) {
    const [l, r, w] = interp(fr.map((x) => x.f), p.in + i);
    const A = fr[l]!, B = fr[r]!;
    freeAt.set(p.in + i, { vis: Object.fromEntries(Object.keys(A.vis).map((k) => [k, A.vis[k]! * (1 - w) + B.vis[k]! * w])), tf: A.tf.map((v, k) => v * (1 - w) + B.tf[k]! * w) });
  }
  const tf = Array.from({ length: n }, (_, i) => {
    const t = tfAt(st, p.in + i), q = freeAt.get(p.in + i);
    return q ? { x: t.x + q.tf[0], y: t.y + q.tf[1], ls: t.ls + q.tf[2], rot: t.rot + q.tf[3] } : t;
  });
  const freeIds = [...new Set((st.free ?? []).flatMap((x) => Object.keys(x.vis)))];
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
  const scaleX = p.flipCurve ? Array.from({ length: n }, (_, i) => Math.max(0.02, p.flipCurve!(p.in + i))) : p.flipRamp ? Array.from({ length: n }, (_, i) => {
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
      ...freeIds.map((id) => ({ id, stage: 'postPhysics', mode: 'add', curve: bk(Array.from({ length: n }, (_, i) => freeAt.get(p.in + i)?.vis[id] ?? 0)) })),
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
  /** 内存保护：记录本进程 RSS / heapUsed 与 cgroup（含 Chromium、几何 Worker）的总量；超过硬上限主动报错退出，不等内核 OOM */
  const memGuard = (stage: string, quiet = false) => {
    if (!quiet) (globalThis as any).gc?.();
    const m = process.memoryUsage();
    const cg = cgroupBytes();
    if (cg > MEM_HARD) throw new Error(`内存 ${(cg / 2 ** 30).toFixed(2)}G 超过上限 ${(MEM_HARD / 2 ** 30).toFixed(1)}G（${stage}），主动退出`);
    if (quiet) return cg;
    log(`[${shot.id}] ${p.id} 内存（${stage}）：RSS ${(m.rss / 2 ** 30).toFixed(2)}G、heapUsed ${(m.heapUsed / 2 ** 30).toFixed(2)}G、cgroup ${(cg / 2 ** 30).toFixed(2)}G`);
    if (cg > MEM_HARD) throw new Error(`内存 ${(cg / 2 ** 30).toFixed(2)}G 超过上限 ${(MEM_HARD / 2 ** 30).toFixed(1)}G（${stage}），主动退出`);
    return cg;
  };
  memGuard('开始');
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
  const every = opts.keyEvery ?? MODELX.keyEvery ?? (zoom ? 12 : 24);
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
  if (p.clipKeys) st.clip = p.clipKeys.map((k) => ({ ...k }));
  else if (p.clipRange) {
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
    const ins = [...new Set([...links.map((l) => l.output), ...CONST_PARAMS])].filter((o) => PHYS_SOURCES.has(o));
    const outs = [...POSE_OUT, ...SECONDARY];
    const deltaOf = (q: string) => (POSE_IN.includes(q) ? 2 : 0.1);
    const r: Record<string, Record<string, number[]>> = {};
    for (const q of ins) Object.assign(r, await gpool.impulse({ base, inputs: [q], outputs: outs, taps: 50, delta: deltaOf(q) }));
    constsAtImpulse = { lean: st.consts.lean, clothes2: st.consts.clothes2 };
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
  // 采样点数：缺省 4000，但雅可比缓存（节点数 × 2K × 可见参数 × 4 字节）不超过约 1GB（s15 左实例 280 个节点时降到约 2600）
  const nSamples = Math.min(opts.samples ?? 4000, Math.floor(1.0e9 / (Math.max(1, nodes.length) * 2 * (VIS.length + 4) * 4)));
  const step = Math.max(1, Math.round(total / nSamples));
  const smp: number[] = [];
  { let k = 0; counts.forEach((cnt, d) => { for (let i = 0; i < cnt; i++, k++) if (k % step === 0) smp.push(d, i); }); }
  // 虹膜主网格（与页面 irisScreen 同一组 drawable）的全部顶点追加在末尾：锚点先验用它们的质心
  const ids = await gpool.drawableIds();
  const irisD = ['ArtMesh236', 'ArtMesh134'].map((id) => ids.indexOf(id));
  const irisRange: [number, number][] = [];
  for (const d of irisD) { const a = smp.length / 2; for (let i = 0; i < (counts[d] ?? 0); i++) smp.push(d, i); irisRange.push([a, smp.length / 2]); }
  const sample = new Int32Array(smp);
  // 锚点先验：shots 里 confidence ≥ 0.4 的 eyes 锚点（落在节点帧 ±1 上），作为软约束进 GN 数据项（鲁棒加权）
  const anchors = new Map<number, [[number, number], [number, number]]>();
  for (const t of (p.inst as any).track as any[]) {
    if (!t.eyes || (t.confidence ?? 1) < 0.4 || t.frame < p.in || t.frame >= p.out) continue;
    const nf = nodes.find((f) => Math.abs(f - t.frame) <= 1);
    if (nf !== undefined) anchors.set(nf, t.eyes);
  }
  log(`[${shot.id}] ${p.id}: 采样 ${sample.length / 2} 个顶点（含虹膜 ${irisRange.map(([a, b]) => b - a).join('+')}），锚点 ${anchors.size} 个`);
  const dimSpecs = VIS.map((r) => ({ key: r, params: [[r, 1]] as [string, number][], delta: POSE_OUT.includes(r) || r === 'LeanInput' ? 1 : r === 'Clothes2' ? 0.02 : 0.05 }));
  log(`[${shot.id}] ${p.id}: 可见参数 ${VIS.length} 个（姿态 ${POSE_OUT.length}、直接 ${DIRECT.length}、次级物理 ${SECONDARY.length}）`);

  // ---- GN ----
  const K = keys.length, N = nodes.length;
  const NT0 = K * 4 + N * Din, NG = CONST_PARAMS.length;
  const NT = NT0 + NG;
  const sdOf = (c: number) => (c >= NT0 ? CONST_SD[CONST_PARAMS[c - NT0]!]! : c < K * 4 ? TF_SD[TF_KEYS[c % 4]!]! : INPUT_DIMS[vts.inputs[(c - K * 4) % Din]!]!.sd);
  /** 常量 q 对可见参数 r 的稳态灵敏度：直接（r = q）+ 物理脉冲响应之和（常量从预热起就不变） */
  const constGain = (q: string, r: string) => (r === q ? 1 : 0) + (H_.get(q)?.get(r)?.reduce((a, b) => a + b, 0) ?? 0);
  let constsFree = false;
  let constsAtImpulse: Record<string, number> = {};
  let lastMedian = 20;
  let lastStepScale = 1;
  /** 每个锚点帧：拟合虹膜到锚点的最大偏差 / 眼距（gn 里更新，用于发散检查） */
  const anchorDev = new Map<number, number>();
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
    // MODELX.anneal：鲁棒核尺度退火（前几轮让大误差区——头发、肩、下巴——也参与，否则 Cauchy 权重把它们当离群点忽略）
    const cauchy = Math.max(3, 1.5 * lastMedian, MODELX.anneal ? MODELX.anneal * 0.85 ** gnCount : 0);
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
      // 锚点先验：两只眼的虹膜质心（屏幕）对 shots 锚点；Cauchy 尺度 max(10px, 0.15 × 眼距)，权重约等于 80 个光流点
      const anc = anchors.get(f);
      if (anc) {
        const ed = Math.hypot(anc[1][0] - anc[0][0], anc[1][1] - anc[0][1]);
        const sig = Math.max(10, 0.15 * ed);
        irisRange.forEach(([a0, a1], e) => {
          const J = new Float64Array(2 * D);
          let px = 0, py = 0;
          const nn = a1 - a0;
          for (let k = a0; k < a1; k++) {
            const dx = j.pos[2 * k]! - ax, dy = j.pos[2 * k + 1]! - ay;
            const qx = T.x + Sc * (cs * dx - sn * dy), qy = T.y + Sc * (sn * dx + cs * dy);
            px += qx / nn; py += qy / nn;
            J[0] += 1 / nn; J[D + 1] += 1 / nn;
            J[2] += (qx - T.x) / nn; J[D + 2] += (qy - T.y) / nn;
            J[3] += (-(qy - T.y) * Math.PI) / 180 / nn; J[D + 3] += ((qx - T.x) * Math.PI) / 180 / nn;
            for (let r = 0; r < V; r++) {
              const cx = j.J[2 * k * V + r]!, cy = j.J[(2 * k + 1) * V + r]!;
              J[4 + r] += (Sc * (cs * cx - sn * cy)) / nn; J[D + 4 + r] += (Sc * (sn * cx + cs * cy)) / nn;
            }
          }
          const ex = anc[e]![0] - px, ey = anc[e]![1] - py;
          anchorDev.set(f, Math.max(anchorDev.get(f) ?? 0, Math.hypot(ex, ey) / Math.max(1, ed)));
          // 自适应：眼距 ≤ 60px（小全身）全权重，≥ 120px（特写，光流约 1px 精度、人工锚点 ±5～10px）为 0，只做发散检查
          const fa = MODELX.anchorW ?? Math.min(1, Math.max(0, (120 - ed) / 60));
          if (fa <= 0) return;
          const wt = (fa * norm * 80) / (1 + (Math.hypot(ex, ey) / sig) ** 2);
          for (let a = 0; a < D; a++) {
            b[a] += wt * (J[a]! * ex + J[D + a]! * ey);
            for (let c = 0; c < D; c++) M[a * D + c] += wt * (J[a]! * J[c]! + J[D + a]! * J[D + c]!);
          }
        });
      }
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
      if (constsFree) CONST_PARAMS.forEach((q, gi) => VIS.forEach((r, ri) => { const v = constGain(q, r); if (Math.abs(v) > 1e-7) addC(NT0 + gi, 4 + ri, v); }));
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
    // MODELX.clamp：逐分量截断（缺省是整步按最大分量等比缩小——一个病态分量会把所有分量一起拖慢）
    if (MODELX.clamp) for (let i = 0; i < NT; i++) du[i] = Math.max(-lim, Math.min(lim, du[i]!));
    const scale = MODELX.clamp ? 1 : mx > lim ? lim / mx : 1;
    lastStepScale = mx > lim ? lim / mx : 1;
    for (let k = 0; k < K; k++) TF_KEYS.forEach((key, a) => (st.tf[k]![key] += du[k * 4 + a]! * scale * TF_SD[key]!));
    if (useInputs) for (let m = 0; m < N; m++) vts.inputs.forEach((name, i) => {
      if (!inputMask[i]) return;
      const d = INPUT_DIMS[name]!;
      st.inputs[m]![name] = Math.min(d.hi, Math.max(d.lo, st.inputs[m]![name]! + du[K * 4 + m * Din + i]! * scale * d.sd));
    });
    if (constsFree) {
      CONST_PARAMS.forEach((q, gi) => {
        const key = q === 'LeanInput' ? 'lean' : 'clothes2';
        const [lo, hi] = CONST_BOX[q]!;
        st.consts[key] = Math.min(hi, Math.max(lo, st.consts[key] + du[NT0 + gi]! * scale * CONST_SD[q]!));
      });
      // 常量变化较大时重测物理脉冲响应（线性化点变了）
      if (Math.abs(st.consts.lean - (constsAtImpulse.lean ?? 0)) > 3 || Math.abs(st.consts.clothes2 - (constsAtImpulse.clothes2 ?? 0)) > 0.06) await measureImpulse();
    }
    tick('solve', t);
    lastMedian = med;
    return { median: med, p90: stats.length ? stats[Math.floor(stats.length * 0.9)]! : NaN, points: stats.length };
  };

  const reanchor = () => {
    const bad = [...anchorDev.entries()].filter(([, d]) => d > 1.5).map(([f]) => f);
    const ks = new Set(bad.map((f) => { const [l, r, w] = interp(keys, f); return w < 0.5 ? l : r; }));
    for (const k of ks) st.tf[k] = eyeTf(keys[k]!);
    if (ks.size) log(`[${shot.id}] ${p.id} 发散检查：${bad.length} 个锚点帧偏差 > 1.5 眼距（${bad.join(',')}），${ks.size} 个变换关键帧重锚`);
    anchorDev.clear();
  };
  const runGN = async (iters: number, tag: string, lam: number, useInputs = true) => {
    let r = { median: NaN, p90: NaN, points: 0 };
    for (let it = 0; it < iters; it++) {
      if (it > 0) reanchor();
      r = await gn(lam, useInputs);
      log(`[${shot.id}] ${p.id} ${tag} #${it}: 光流残差中位 ${r.median.toFixed(2)}px、p90 ${r.p90.toFixed(2)}px（${r.points} 点，步长系数 ${lastStepScale.toFixed(2)}）${constsFree ? `；lean ${st.consts.lean.toFixed(2)}、clothes2 ${st.consts.clothes2.toFixed(3)}` : ''}`);
    }
    return r;
  };

  // ---- 阶段断点（work/ckpt_<图层>.json）：traj（轨迹 + 外观 + 开窗）→ perframe → eyes。指纹不符（参数、FIT_MODEL、shots 改过）则忽略 ----
  const ckFile = join(work, `ckpt_${p.id}.json`);
  const fingerprint = createHash('sha256').update(JSON.stringify({
    m: process.env.FIT_MODEL ?? '', c: opts.smoothC, a: opts.itersA, b: opts.itersB, x: opts.fixedConsts, in: p.in, out: p.out, nodes,
    shots: createHash('sha256').update(readFileSync(join(REPO, 'scenes/deepseek-chan/shots.json'))).digest('hex'),
  })).digest('hex').slice(0, 16);
  const RANK: Record<string, number> = { traj: 1, perframe: 2, eyes: 3 };
  let ck: { stage: string; fp: string; st: LayerState } | null = null;
  if (process.env.FIT_RESUME !== '0' && existsSync(ckFile)) {
    const c = JSON.parse(readFileSync(ckFile, 'utf8'));
    if (c.fp === fingerprint) { ck = c; Object.assign(st, c.st); log(`[${shot.id}] ${p.id} 从断点继续：阶段「${c.stage}」已完成（${ckFile}）`); }
    else log(`[${shot.id}] ${p.id} 断点指纹不符，忽略（${ckFile}）`);
  }
  const done = (stage: string) => (ck ? (RANK[ck.stage] ?? 0) >= RANK[stage]! : false);
  const saveCk = (stage: string) => {
    writeFileSync(`${ckFile}.tmp`, JSON.stringify({ stage, fp: fingerprint, st }));
    renameSync(`${ckFile}.tmp`, ckFile);
    log(`[${shot.id}] ${p.id} 断点已存：${stage}`);
  };
  if (!done('traj')) {
  // ---- (0) 变换初值的粗配准（图层规划 reg: true 时开启，目前只有 s15；FIT_MODEL='{"reg":true|false}' 强制）----
  //   s24 实测有害（0.687 → 0.524：快速运动 + 划过特效的前几个关键帧给出错误尺度），s43 小幅有益（27 → 28/30 帧 ≥ 0.7）。
  //   shots 的眼睛轨迹多为人工锚点插值（s15 的尺度可差 2 倍），光流追不回来。逐变换关键帧：只渲染本图层（中性姿态），
  //   tools/fit/reg.py 做多尺度模板匹配（梯度幅值的掩码 NCC，与掩码面积无关），相关明显更高才采纳。
  if ((MODELX.reg ?? p.reg) === true) {
    const t = Date.now();
    await setupPages();
    const others = opts.others ? (opts.others() as any[]).map((l) => l.id).filter((id) => id !== p.id) : [];
    await pool.all((h) => (window as any).__fit.setHidden(h), [...(opts.hidden ?? []), ...others]);
    const n = pool.size;
    const regEyes = new Map<number, [number, number][]>();
    await Promise.all(pool.pages.map(async (pg, k) => {
      for (const f of keys.filter((_, i) => i % n === k)) {
        const img = (await pg.evaluate((f) => (window as any).__fit.render(f, 1), f)) as { data: string };
        writeFileSync(join(work, `reg_${f}.rgba`), Buffer.from(img.data, 'base64'));
        regEyes.set(f, (await pg.evaluate(([lid, f]) => (window as any).__fit.irisScreen(lid, f), [p.id, f] as const)) as [number, number][]);
      }
    }));
    writeFileSync(join(work, 'reg.json'), JSON.stringify({
      search: 200, logScale: 0.8,
      frames: keys.map((f, k) => ({ frame: f, ours: join(work, `reg_${f}.rgba`), ref: ref.file, refIndex: f - shot.start, excl: exclusionsAt(shot, f), anchor: regEyes.get(f) ? [(regEyes.get(f)![0]![0] + regEyes.get(f)![1]![0]) / 2, (regEyes.get(f)![0]![1] + regEyes.get(f)![1]![1]) / 2] : [st.tf[k]!.x, st.tf[k]!.y],
        eyeDist: regEyes.get(f) ? Math.hypot(regEyes.get(f)![1]![0] - regEyes.get(f)![0]![0], regEyes.get(f)![1]![1] - regEyes.get(f)![0]![1]) : mdx * Math.exp(st.tf[k]!.ls) })),
    }));
    const r = JSON.parse(py('tools/fit/reg.py', join(work, 'reg.json'))) as Record<string, { dx: number; dy: number; dls: number; drot: number; ncc0: number; ncc: number } | null>;
    const accepted: string[] = [];
    keys.forEach((f, k) => {
      const q = r[String(f)];
      if (!q || q.ncc < 0.15 || q.ncc - q.ncc0 < 0.03) return;
      // 模板绕渲染出的眼睛中点 c 缩放 e^dls、旋转 drot，再平移 (dx, dy)；图层锚点的屏幕位置 (x, y) 做同样的变换
      const e = regEyes.get(f);
      const cx = e ? (e[0]![0] + e[1]![0]) / 2 : st.tf[k]!.x, cy = e ? (e[0]![1] + e[1]![1]) / 2 : st.tf[k]!.y;
      const kk = Math.exp(q.dls), th = (q.drot * Math.PI) / 180;
      const ux = st.tf[k]!.x - cx, uy = st.tf[k]!.y - cy;
      st.tf[k] = { x: cx + kk * (Math.cos(th) * ux - Math.sin(th) * uy) + q.dx, y: cy + kk * (Math.sin(th) * ux + Math.cos(th) * uy) + q.dy, ls: st.tf[k]!.ls + q.dls, rot: st.tf[k]!.rot + q.drot };
      accepted.push(`${f}:(${q.dx.toFixed(0)},${q.dy.toFixed(0)},${q.dls.toFixed(2)},${q.drot}°) ${q.ncc0.toFixed(2)}→${q.ncc.toFixed(2)}`);
    });
    log(`[${shot.id}] ${p.id} 粗配准：${accepted.length}/${keys.length} 个关键帧改动 ${accepted.join(' ')}`);
    tick('reg', t);
  }
  memGuard('粗配准后');
  // ---- (1) 只解变换，几轮 ----
  await measureImpulse();
  await runGN(3, '变换', 20, false);
  // ---- (2) 镜头常量：多假设各自完整收敛，按 SSIM@360 挑（MODELX.consts 时各假设只是初值，常量随 GN 连续更新）----
  constsFree = CONST_PARAMS.length > 0;
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
  const c2best = best ? (constsFree ? (best as { s: LayerState }).s.consts.clothes2 : (best as { h: Partial<ShotConsts> }).h.clothes2) : c2s[0];
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
  if (st.clip && !p.clipKeys) {
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

  saveCk('traj');
  } // !done('traj')
  // ---- (6) 逐帧精修：每个节点帧独立放开全部可见参数（postPhysics add）与变换偏移，光流 GN 逐帧求解（鲁棒核 20px 退火到 6px）----
  // MODELX.perframe（缺省 12）：逐帧精修，作为正式阶段导出（postPhysics add 轨 + 变换偏移）；MODELX.upper：同一算法只做诊断、不导出
  const pfIters = MODELX.upper ?? MODELX.perframe ?? 12;
  if (pfIters > 0) {
    jacCache = null; memGuard('轨迹阶段结束'); // 逐帧精修自己算雅可比；释放轨迹阶段的缓存（s15 左实例约 1GB）
    const before = await ssim360(nodes);
    log(`[${shot.id}] ${p.id} 逐帧精修前 SSIM@360 均值 ${before.mean.toFixed(3)}`);
    const visSd = (r: string) => (POSE_OUT.includes(r) || r === 'LeanInput' ? 3 : r === 'Clothes2' ? 0.05 : 0.3);
    const tPf = Date.now();
    if (!done('perframe')) {
    st.free = nodes.map((f) => ({ f, vis: Object.fromEntries(VIS.map((r) => [r, 0])), tf: [0, 0, 0, 0] as [number, number, number, number] }));
    // 分块处理节点（每块渲染 + 光流 + 雅可比，解完即释放）：整段 280 个节点的雅可比约 1GB、光流约 0.8GB，
    // 加上上一轮未回收的副本会把 s15 推到 OOM。内存超过 8.5GB 时块再减半。
    let pfChunk = 40;
    for (let it = 0; it < pfIters; it++) {
      const meds: number[] = [];
      for (let c0 = 0; c0 < nodes.length; c0 += pfChunk) {
      const chunk = nodes.slice(c0, c0 + pfChunk);
      layer = buildLayer(p, vts, st, calib, opts.smoothC);
      await renderFrames(chunk);
      const fl = flows(chunk);
      const jac = await gpool.jacobians(layer, chunk, dimSpecs, sample);
      const L = layer as any;
      const ax = L.transform.anchorX, ay = L.transform.anchorY;
      chunk.forEach((f, ci) => {
        const fi = c0 + ci;
        const q = st.free![fi]!;
        const j = jac.get(f)!, F = fl.get(f)!;
        const i0 = f - p.in;
        const T = { x: L.transform.x.baked.values[i0], y: L.transform.y.baked.values[i0], ls: Math.log(L.transform.scale.baked.values[i0]), rot: L.transform.rotation.baked.values[i0] };
        const Sc = Math.exp(T.ls), rr = (T.rot * Math.PI) / 180, cs = Math.cos(rr), sn = Math.sin(rr);
        const V = VIS.length, D = 4 + V, Kp = j.opa.length;
        const M = new Float64Array(D * D), b = new Float64Array(D);
        const es: number[] = [];
        for (let k = 0; k < Kp; k++) {
          if (j.opa[k]! < 0.5) continue;
          const dx = j.pos[2 * k]! - ax, dy = j.pos[2 * k + 1]! - ay;
          const px = T.x + Sc * (cs * dx - sn * dy), py = T.y + Sc * (sn * dx + cs * dy);
          const u = px / 2 - 0.25, v = py / 2 - 0.25;
          const x0 = Math.floor(u), y0 = Math.floor(v);
          if (x0 < 0 || y0 < 0 || x0 >= 639 || y0 >= 359) continue;
          const o = (y0 * 640 + x0) * 3;
          if (F[o + 2]! < 0.5) continue;
          const ex = F[o]!, ey = F[o + 1]!;
          const J = new Float64Array(2 * D);
          J[0] = 1; J[D + 1] = 1; J[2] = px - T.x; J[D + 2] = py - T.y;
          J[3] = (-(py - T.y) * Math.PI) / 180; J[D + 3] = ((px - T.x) * Math.PI) / 180;
          for (let r = 0; r < V; r++) {
            const cx = j.J[2 * k * V + r]! * visSd(VIS[r]!), cy = j.J[(2 * k + 1) * V + r]! * visSd(VIS[r]!);
            J[4 + r] = Sc * (cs * cx - sn * cy); J[D + 4 + r] = Sc * (sn * cx + cs * cy);
          }
          const m = Math.hypot(ex, ey);
          es.push(m);
          const wt = 1 / (1 + (m / Math.max(6, 20 * 0.8 ** it)) ** 2); // 鲁棒核尺度从 20px 退火到 6px：大误差区（耳、后发）也参与
          for (let a = 0; a < D; a++) {
            const ja = J[a]!, jb = J[D + a]!;
            if (ja === 0 && jb === 0) continue;
            b[a] += wt * (ja * ex + jb * ey);
            for (let c2 = 0; c2 < D; c2++) M[a * D + c2] += wt * (ja * J[c2]! + jb * J[D + c2]!);
          }
        }
        es.sort((x, y) => x - y);
        meds.push(es[es.length >> 1] ?? 0);
        // LM：对角放大 + 单位先验（可见参数以 visSd 为单位，偏离 0 有弱惩罚）
        let tr = 0;
        for (let a = 0; a < D; a++) tr += M[a * D + a]!;
        for (let a = 0; a < D; a++) {
          M[a * D + a] = M[a * D + a]! * 1.2 + 1e-3 * (tr / D) + 1e-6;
          if (a >= 4) { const r = VIS[a - 4]!; b[a] -= 1e-3 * (tr / D) * (q.vis[r]! / visSd(r)); }
        }
        const du = bandSolve(M, b, D, D);
        let mx = 0;
        for (let a = 4; a < D; a++) mx = Math.max(mx, Math.abs(du[a]!));
        const sc = mx > 2 ? 2 / mx : 1;
        q.tf[0] += du[0]! * sc; q.tf[1] += du[1]! * sc; q.tf[2] += du[2]! * sc; q.tf[3] += du[3]! * sc;
        for (let a = 4; a < D; a++) q.vis[VIS[a - 4]!] += du[a]! * sc * visSd(VIS[a - 4]!);
      });
      fl.clear(); jac.clear();
      if (memGuard(`逐帧精修 #${it} 块 ${c0}`) > MEM_SOFT && pfChunk > 8) { pfChunk = Math.max(8, pfChunk >> 1); log(`[${shot.id}] ${p.id} 内存超过 8.5GB：逐帧精修块大小降到 ${pfChunk}`); }
      }
      meds.sort((x, y) => x - y);
      log(`[${shot.id}] ${p.id} 逐帧精修 #${it}: 逐帧光流残差中位数的中位 ${meds[meds.length >> 1]!.toFixed(2)}px`);
    }
    saveCk('perframe');
    } // !done('perframe')
    // 眼睛开合：每个节点帧对左右眼 ParamEyeL/ROpen 的偏移做 5×5 网格搜索（两眼独立；像素损失），用于 wink 等单眼动作
    if (!done('eyes')) {
      const tE = Date.now();
      await setupPages();
      memGuard('眼睛开合前');
      const grid = [-1.2, -0.8, -0.4, 0, 0.3];
      let changed = 0;
      for (const [fi, f] of nodes.entries()) {
        // 页面里每个候选都要读回整幅缓冲：V8 外部内存回收滞后，chrome-headless 会涨到 2GB+ 被 OOM（cf-run14）。
        // 每 10 帧强制页面 GC（CDP HeapProfiler.collectGarbage）；每 5 帧查实时 cgroup，超过软阈值立即 GC，超过硬阈值报错。
        if (fi % 10 === 0) await pool.gc();
        if (fi % 5 === 0 && memGuard(`眼睛开合 ${fi}/${nodes.length}`, fi % 50 !== 0) > MEM_SOFT) await pool.gc();
        if (!exclusionsAt(shot, f).length || true) await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
        const q = st.free![fi]!;
        const bL = q.vis.ParamEyeLOpen ?? 0, bR = q.vis.ParamEyeROpen ?? 0;
        const cands = grid.flatMap((dl) => grid.map((dr) => [dl, dr] as const));
        const rs = await pool.evalBatch<{ loss: number }>(f, cands.map(([dl, dr]) => ({ [p.id]: { set: [[f, { 'post:ParamEyeLOpen': bL + dl, 'post:ParamEyeROpen': bR + dr }]] } })), { scale: 0.5, r: 3, wcol: 1, weye: 0, wseg: 0, wgrad: 0.3, eyesOf: [] });
        let bi = cands.findIndex(([dl, dr]) => dl === 0 && dr === 0);
        rs.forEach((r, i) => { if (r.loss < rs[bi]!.loss - 1e-4) bi = i; });
        const [dl, dr] = cands[bi]!;
        if (dl || dr) changed++;
        q.vis.ParamEyeLOpen = bL + dl; q.vis.ParamEyeROpen = bR + dr;
        await pool.all(([lid, f, v]) => (window as any).__fit.commit(f, { [lid]: v }), [p.id, f, { set: [[f, { 'post:ParamEyeLOpen': bL + dl, 'post:ParamEyeROpen': bR + dr }]] }] as const);
      }
      log(`[${shot.id}] ${p.id} 眼睛开合网格搜索：${changed}/${nodes.length} 个节点帧改动`);
      memGuard('眼睛开合后');
      saveCk('eyes');
      tick('eyes', tE);
    }
    const after = await ssim360(nodes);
    log(`[${shot.id}] ${p.id} 逐帧精修后 SSIM@360 均值 ${after.mean.toFixed(3)}、最小 ${after.min.toFixed(3)}（${before.mean.toFixed(3)} → ${after.mean.toFixed(3)}）`);
    // 最常被用到的可见参数（|偏移|/sd 的均值）
    const use = VIS.map((r) => [r, st.free!.reduce((a, x) => a + Math.abs(x.vis[r]!) / visSd(r), 0) / nodes.length] as const).sort((a, b) => b[1] - a[1]).slice(0, 15);
    log(`[${shot.id}] ${p.id} 逐帧精修偏移最大的可见参数（sd 单位）：${use.map(([r, v]) => `${r}:${v.toFixed(2)}`).join(' ')}`);
    log(`[${shot.id}] ${p.id} 逐帧精修变换偏移（均值 |dx|,|dy|,|dls|,|drot|）：${[0, 1, 2, 3].map((k) => (st.free!.reduce((a, x) => a + Math.abs(x.tf[k]!), 0) / nodes.length).toFixed(3)).join(',')}`);
    writeFileSync(join(work, 'upper_layer.json'), JSON.stringify(buildLayer(p, vts, st, calib, opts.smoothC)));
    if (MODELX.upper) st.free = undefined; // 诊断模式：导出正常拟合结果
    tick('perframe', tPf);
  }
  const endScore = await ssim360(nodes);
  log(`[${shot.id}] ${p.id} 结束：SSIM@360（全部节点帧，拟合内部打分）均值 ${endScore.mean.toFixed(3)}、最小 ${endScore.min.toFixed(3)}`);

  // ---- 导出 ----
  const final = buildLayer(p, vts, st, calib, opts.smoothC) as any;
  const params: Record<string, number[]> = {};
  for (const t of final.params) if (t.curve.baked) params[t.stage === 'postPhysics' ? `post:${t.id}` : t.id] = t.curve.baked.values;
  const d = {
    tf: { x: final.transform.x.baked.values, y: final.transform.y.baked.values, scale: final.transform.scale.baked.values, rotation: final.transform.rotation.baked.values },
    opacity: final.opacity?.baked?.values ?? null, blur: final.filters.find((f: any) => f.type === 'blur').sigma.baked.values,
    clip: final.clip ? { x: final.clip.x.baked.values, y: final.clip.y.baked.values, width: final.clip.width.baked.values, height: final.clip.height.baked.values, radius: final.clip.radius.baked.values } : null,
    color: st.consts.color, resample: st.consts.resample, sx: final.transform.scaleX?.baked?.values ?? null, params,
  };
  const base2 = { layers: [{ id: p.id, transform: final.transform, params: final.params }] } as any;
  const holds = holdSegments(isNew, p.in, p.out);
  const out = exportLayer(base2, p, d as any, { log: nodes } as any, holds, shot) as any;
  // 检查：翻牌段导出的 scaleX 必须等于 shots 的 cardFlip 强度
  if (p.flipCurve) {
    const sx = out.transform.scaleX?.baked;
    if (!sx) throw new Error(`${p.id}：导出缺 transform.scaleX（cardFlip）`);
    for (let g = p.in; g < Math.min(p.out, p.in + 12); g++) {
      const want = Math.max(0.02, p.flipCurve(g)), got = sx.values[g - sx.start];
      if (Math.abs(got - want) > 0.01) throw new Error(`${p.id}：帧 ${g} 导出 scaleX ${got} ≠ cardFlip ${want}`);
    }
    log(`[${shot.id}] ${p.id} 检查通过：导出 scaleX 与 cardFlip 强度一致（${p.in}～${p.in + 11}）`);
  }
  for (const t of out.params) if (t.provenance?.detail) t.provenance.detail.what = t.stage === 'postPhysics'
    ? 'per-frame refinement: offset added to the physics/pose output (optical-flow GN per new frame, linear between frames)'
    : 'VTS forward model: face-tracking inputs (fitted, per new frame) → mapping + smoothing';
  out.physics = final.physics; // 与拟合时同一物理起点（exportLayer 的缺省是旧引擎的 stabilize）
  out.provenance.detail.method = 'face-input-space trajectory GN on optical-flow correspondences + pixel refinement (round 3b)';
  out.provenance.detail.vts = { smoothC: opts.smoothC, consts: st.consts };
  const elapsed = (Date.now() - t0) / 1000;
  log(`[${shot.id}] ${p.id} 完成：${elapsed.toFixed(0)}s；分项 ${JSON.stringify(Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v)])))}`);
  ref.close();
  return { layer: out, state: st, elapsed, timing, score: endScore, built: final };
}
