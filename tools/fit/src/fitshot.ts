/*
 * 逐镜头拟合（第二轮）。
 *
 * 每个镜头：
 *   1. 解码参考帧；按实例在角色区域内检测帧重复，只拟合新画面帧，输出 frameHold；
 *   2. 构造只含本镜头图层的拟合场景（逐帧 baked 曲线），在多个 harness 页面（Pool）里建立同样的状态；
 *   3. 按 order 逐图层拟合，未拟合的图层隐藏：
 *      a) 锚点帧（可见、虹膜可检测、shots 置信度高）：金字塔多起点全局搜索（1/8 → 1/4 → 1/2 分辨率），
 *         有虹膜时变换由眼睛对齐解析求出；同时拟合镜头常量（体型 Clothes2、前倾、呼吸相位/周期、清晰度 blur）；
 *         候选写满 [in, f]，物理从 stabilize + 预热重新推进；
 *      b) 锚点之前时间倒序、之后时间正序逐帧：粗（1/4，变换由眼睛对齐 + 头/身/眼）→ 细（1/2，全部维度）；
 *         物理输入写 f−1 与 f（Cubism 物理输出滞后一帧），直接参数写 f；
 *         发散检测（损失突增或虹膜误差过大）→ 在该帧重新做多起点搜索（重锚）；
 *   4. 开关：shots 的切换按画面检测逐帧权重（如爱心眼的粉色像素占比）再拟合斜坡；
 *   5. 导出图层 JSON（逐帧曲线、frameHold、clip、滤镜、provenance）。
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Shot, ShotInstance } from '../../../packages/scene/src/index.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { CMAES } from './cmaes.ts';
import {
  BLUR_DIM, BODY_DIMS, DEFAULT_STATE, FACE_DIMS, FIT_PARAM_IDS, HAIR_DIMS, HAIR_GROUPS, HEAD_DIMS, OPACITY_DIM, PHYS_INPUT_IDS, POSE_POST_IDS,
  SHOT_CONST_DIMS, TF_DIMS, hairParams, toParams, type Dim,
} from './model.ts';
import { FIT_DIR, FIT_TAG, MODEL_URI } from './paths.ts';
import type { Pool } from './pool.ts';
import { RefFrames } from './refvideo.ts';
import { detectNewFrames, detectRefEyes, exclusionsAt, heartEyeWeight, holdSegments, trackAt, visibleFraction } from './shotsutil.ts';

export interface ModelCalib {
  eyes: [[number, number], [number, number]];
  canvas: { width: number; height: number };
  /** 青色质心 − 虹膜网格顶点质心（画布像素） */
  irisOffset: [[number, number], [number, number]];
}

export interface SwitchPlan {
  param: string;
  from: number;
  to: number;
  /** 画面检测窗口 [a, b) */
  window: [number, number];
  est: number;
  fade: number;
  /** 画面检测器：返回帧 f 的权重估计（null = 无法判断） */
  detect?: 'heartEyes';
}

export interface LayerPlan {
  id: string;
  inst: ShotInstance;
  in: number;
  out: number;
  /** 开关常量（shots 的 variant 中不被镜头常量覆盖的部分） */
  constants: Record<string, number>;
  switches: SwitchPlan[];
  /** 逐帧拟合 blur sigma 的区间（景深）；区间外用镜头常量 c:blur */
  blurRanges: [number, number][];
  /** 逐帧拟合不透明度的区间 */
  opacityRanges: [number, number][];
  /** 逐帧拟合水平缩放 scaleX 的区间（翻牌） */
  flipRange?: [number, number];
  /** 翻牌：scaleX 在 [a, b) 从 0 线性展开到 1（第三轮续，按画面估计） */
  flipRamp?: [number, number];
  /** 翻牌：逐帧 scaleX 曲线（shots 的 cardFlip 效果强度）；优先于 flipRamp */
  flipCurve?: (f: number) => number;
  /** 圆形开窗的关键帧（shots 的实例 clip + opacity 轨）：给出时直接使用，不再逐关键帧拟合 */
  clipKeys?: { f: number; cx: number; cy: number; r: number; a: number }[];
  /** 变换初值粗配准（traj2 的 reg） */
  reg?: boolean;
  /** 景深强度曲线（sigma，像素）：blurRanges 内 blur = dofK × 它 */
  dofStrength?: (f: number) => number;
  /** 逐帧拟合圆形 clip 的区间（区间外不裁切） */
  clipRange?: [number, number];
  /** 镜头常量（SHOT_CONST_DIMS 里的 key） */
  consts: string[];
  /** 镜头常量的初值（如 shots 的 variant 给出的 Clothes2） */
  constInit?: Record<string, number>;
  colorFilter: boolean;
  /** 头发 postPhysics 叠加（按发束分组、每组 1 个自由度） */
  hairBake: boolean;
  z: number;
  firstFitFrom?: number;
}

export interface FitOptions {
  until?: number;
  start?: number;
  previewEvery?: number;
  fitIgnoreExclusions?: string[];
  order?: string[];
  only?: string[];
  /** 逐帧粗/细两级的代数缩放（速度/质量折中），缺省 1 */
  effort?: number;
  log?: (s: string) => void;
}

type State = Record<string, number>;
type Res = { loss: number; area: number; ssim: number; eye: number; seg: number; grad?: number; tf?: number[] };
type Eyes = [[number, number], [number, number]];
/** frame：物理输入写 f−1 与 f、直接参数写 f；win：写满 [f−BACK_WINDOW, f]；all：写满 [in, f] */
type Mode = 'frame' | 'win' | 'all';
const BACK_WINDOW = 8;

interface LayerRuntime {
  plan: LayerPlan;
  newFrames: Set<number>;
  lastFit: number | null;
  color: { brightness: number; contrast: number; saturation: number; multiply: [number, number, number] };
  lastArea: number;
  log: { frame: number; stage: string; loss: number; ssim: number; area: number; eye: number; evals: number; ms: number; reanchored?: boolean }[];
}

const COLOR_DIMS: Dim[] = [
  { key: 'brightness', lo: -0.3, hi: 0.3, sd: 0.03, tau: 1 },
  { key: 'contrast', lo: 0.6, hi: 1.5, sd: 0.05, tau: 1 },
  { key: 'saturation', lo: 0.5, hi: 1.6, sd: 0.06, tau: 1 },
  { key: 'mr', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
  { key: 'mg', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
  { key: 'mb', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
];
const CLIP_DIMS: Dim[] = [
  { key: 'cx', lo: -500, hi: 1800, sd: 6, tau: 30 },
  { key: 'cy', lo: -500, hi: 1300, sd: 6, tau: 30 },
  { key: 'cr', lo: 0, hi: 1500, sd: 8, tau: 40 },
];
const inRanges = (rs: [number, number][], f: number) => rs.some(([a, b]) => f >= a && f < b);
const PARAM_CONSTS = ['LeanInput', 'Clothes2'];

export function makeScene(shot: Shot, plans: LayerPlan[], calib: ModelCalib, init: Map<string, State>) {
  const ax = (calib.eyes[0][0] + calib.eyes[1][0]) / 2;
  const ay = (calib.eyes[0][1] + calib.eyes[1][1]) / 2;
  const sorted = [...plans].sort((a, b) => a.z - b.z);
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: { title: `fit ${shot.id}` },
    timebase: { fps: 60 },
    duration: shot.end,
    composition: { width: 1280, height: 720, background: null },
    assets: { models: [{ id: 'dsc', uri: MODEL_URI }] },
    layers: sorted.map((p) => {
      const s = init.get(p.id)!;
      const n = p.out - p.in;
      const baked = (v: number) => ({ baked: { start: p.in, values: Array(n).fill(v) } });
      const filters: unknown[] = [];
      if (p.colorFilter) filters.push({ type: 'color', brightness: 0, contrast: 1, saturation: 1, multiply: [1, 1, 1] });
      // 低分辨率素材放大的软化（resample，镜头常量；factor 1 = 不做）在模糊之前
      filters.push({ type: 'resample', factor: 1, method: 'bilinear' });
      filters.push({ type: 'blur', sigma: baked(s['c:blur'] ?? 0) });
      const pre = toParams(s, p.in);
      const paramConsts = PARAM_CONSTS.filter((k) => p.consts.includes(`c:${k}`));
      return {
        id: p.id,
        kind: 'model',
        model: 'dsc',
        in: p.in,
        out: p.out,
        transform: { x: baked(s.x!), y: baked(s.y!), scale: baked(Math.exp(s.ls!)), ...(p.flipRange ? { scaleX: baked(1) } : {}), rotation: baked(s.rot!), anchorX: ax, anchorY: ay },
        ...(p.opacityRanges.length ? { opacity: baked(1) } : {}),
        // 区间外 clip 为覆盖全画面的大矩形（等价于不裁切）
        ...(p.clipRange ? { clip: { x: baked(-10000), y: baked(-10000), width: baked(30000), height: baked(30000), radius: baked(0), feather: 2 } } : {}),
        filters,
        physics: { warmup: 60, stabilize: true },
        params: [
          ...Object.entries(p.constants).filter(([id]) => !paramConsts.includes(id)).map(([id, v]) => ({ id, curve: { keys: [{ t: p.in, v }] } })),
          ...p.switches.map((sw) => ({ id: sw.param, curve: baked(sw.from) })),
          ...paramConsts.map((id) => ({ id, curve: baked(s[`c:${id}`] ?? 0) })),
          ...FIT_PARAM_IDS.map((id) => ({ id, curve: baked(pre[id]!) })),
          ...POSE_POST_IDS.map((id) => ({ id, stage: 'postPhysics', curve: baked(pre[`post:${id}`]!) })),
          ...(p.hairBake ? HAIR_GROUPS.flatMap((g) => g.params).map((id) => ({ id, stage: 'postPhysics', mode: 'add', curve: baked(0) })) : []),
        ],
      };
    }),
  };
}

export function initialState(plan: LayerPlan, calib: ModelCalib, f: number): State {
  const eyes = trackAt(plan.inst, 'eyes', f) as [[number, number], [number, number]] | null;
  const mdx = calib.eyes[1][0] - calib.eyes[0][0];
  let x = 640, y = 360, scale = 0.3, rot = 0;
  if (eyes) {
    x = (eyes[0][0] + eyes[1][0]) / 2;
    y = (eyes[0][1] + eyes[1][1]) / 2;
    scale = Math.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1]) / mdx;
    rot = (Math.atan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0]) * 180) / Math.PI;
  } else {
    const head = trackAt(plan.inst, 'head', f) as number[] | null;
    if (head) { x = head[0]! + head[2]! / 2; y = head[1]! + head[3]! * 0.6; scale = head[2]! / 1100; }
  }
  const blur = (trackAt(plan.inst, 'blurSigma', f) as number | null) ?? 0;
  return { ...DEFAULT_STATE, ...(plan.constInit ?? {}), x, y, ls: Math.log(scale), rot, blur };
}

export async function fitShot(pool: Pool, shot: Shot, plans: LayerPlan[], calib: ModelCalib, opts: FitOptions = {}) {
  const log = opts.log ?? ((s: string) => console.log(s));
  const outDir = join(FIT_DIR, shot.id + FIT_TAG);
  mkdirSync(join(outDir, 'preview'), { recursive: true });
  const t0 = Date.now();
  const ref = new RefFrames(shot.start, shot.end);
  const effort = opts.effort ?? 1;

  // ---- 1. 帧重复检测 ----
  const L = new Map<string, LayerRuntime>();
  const cadenceLog: Record<string, unknown> = {};
  for (const p of plans) {
    const { isNew, d } = detectNewFrames(ref, shot, p.inst, p.in, p.out);
    const nf = new Set([...isNew.entries()].filter(([, v]) => v).map(([f]) => f));
    cadenceLog[p.id] = { newCount: nf.size, total: p.out - p.in, d: Object.fromEntries([...d].map(([f, v]) => [f, +v.toFixed(2)])) };
    L.set(p.id, { plan: p, newFrames: nf, lastFit: null, color: { brightness: 0, contrast: 1, saturation: 1, multiply: [1, 1, 1] }, lastArea: 0, log: [] });
    log(`[${shot.id}] ${p.id}: 新画面 ${nf.size}/${p.out - p.in} 帧`);
  }

  // ---- 2. 拟合场景 ----
  const init = new Map(plans.map((p) => [p.id, initialState(p, calib, p.in)]));
  const scene = makeScene(shot, plans, calib, init);
  const reused = new Map<string, any>();
  if (opts.only) {
    const prev = JSON.parse(readFileSync(join(outDir, 'result.json'), 'utf8'));
    for (const l of prev.layers) if (!opts.only.includes(l.id)) reused.set(l.id, l);
    scene.layers = scene.layers.map((l) => reused.get(l.id) ?? l) as typeof scene.layers;
    log(`[${shot.id}] 沿用上次结果：${[...reused.keys()].join(', ')}`);
  }
  await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [scene.layers, MODEL_URI, calib.irisOffset] as const);

  const frameExcl = (f: number) => exclusionsAt(shot, f, { skipReasons: opts.fitIgnoreExclusions });
  let curRefFrame = -1;
  const setFrame = async (f: number) => {
    if (curRefFrame === f) return;
    await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), frameExcl(f)] as const);
    curRefFrame = f;
  };

  /** 开关权重：画面检测得到的斜坡（拟合前确定），没有检测器时用 shots 估计 */
  const switchRamps = new Map<string, { start: number; fade: number; rms: number | null; samples: Record<number, number> }>();
  for (const p of plans) {
    for (const sw of p.switches) {
      const samples: Record<number, number> = {};
      if (sw.detect === 'heartEyes') {
        for (let f = sw.window[0]; f < sw.window[1]; f++) {
          const e = trackAt(p.inst, 'eyes', f) as number[][] | null;
          const w = e ? heartEyeWeight(ref.frame(f), e) : null;
          if (w !== null) samples[f] = w;
        }
      }
      const ramp = fitSwitchRamp(new Map(Object.entries(samples).map(([k, v]) => [Number(k), v])), sw);
      switchRamps.set(`${p.id}:${sw.param}`, { ...ramp, samples });
      log(`[${shot.id}] ${p.id} 开关 ${sw.param}: shots 估计 ${sw.est}（fade ${sw.fade}）→ 画面检测 起点 ${ramp.start}、fade ${ramp.fade}（rms ${ramp.rms?.toFixed(3) ?? '-'}，${Object.keys(samples).length} 个样本）`);
    }
  }
  const switchValue = (p: LayerPlan, sw: SwitchPlan, g: number) => {
    const r = switchRamps.get(`${p.id}:${sw.param}`)!;
    return rampValue(sw.from, sw.to, r.start, r.fade, g);
  };

  /** 帧 g 的全部参数（物理输入 + 直接参数 + 开关 + 镜头常量 + 头发叠加） */
  const paramsAt = (p: LayerPlan, s: State, g: number) => {
    const out: Record<string, number> = { ...toParams(s, g) };
    for (const sw of p.switches) out[sw.param] = switchValue(p, sw, g);
    for (const k of PARAM_CONSTS) if (p.consts.includes(`c:${k}`)) out[k] = s[`c:${k}`] ?? 0;
    if (p.hairBake) for (const [k, v] of Object.entries(hairParams(s))) out[`post:${k}`] = v;
    return out;
  };
  const PHYS = new Set(PHYS_INPUT_IDS);

  const valsOf = (lr: LayerRuntime, s: State, f: number, mode: Mode, withColor = false) => {
    const p = lr.plan;
    const set: [number, Record<string, number>][] = [];
    if (mode === 'all') for (let g = p.in; g <= f; g++) set.push([g, paramsAt(p, s, g)]);
    else if (mode === 'win') for (let g = Math.max(p.in, f - BACK_WINDOW); g <= f; g++) set.push([g, paramsAt(p, s, g)]);
    else {
      const prevShown = lr.newFrames.has(f - 1) && lr.lastFit === f - 1;
      if (f - 1 >= p.in) {
        const pr = paramsAt(p, s, f - 1);
        set.push([f - 1, prevShown ? Object.fromEntries(Object.entries(pr).filter(([k]) => PHYS.has(k))) : pr]);
      }
      set.push([f, paramsAt(p, s, f)]);
    }
    const v: Record<string, unknown> = { tf: [s.x, s.y, Math.exp(s.ls!), s.rot], set };
    v.blur = inRanges(p.blurRanges, f) ? s.blur ?? 0 : s['c:blur'] ?? 0;
    if (p.opacityRanges.length) v.opacity = inRanges(p.opacityRanges, f) ? s.opacity ?? 1 : 1;
    if (p.clipRange) v.clip = f >= p.clipRange[0] && f < p.clipRange[1] ? [s.cx, s.cy, s.cr] : null;
    if (withColor) v.color = lr.color;
    return v;
  };

  let totalEvals = 0;
  const optimize = async (
    lr: LayerRuntime, f: number, base: State, dims: Dim[],
    o: { scale: number; r: number; gens: number; lambda: number; sigmaMul?: number; prev: State | null; mode?: Mode; color?: boolean; align?: Eyes | null; wgrad?: number },
  ) => {
    const x0 = dims.map((d) => {
      const v = base[d.key];
      return v !== undefined && Number.isFinite(v) ? v : (d.lo + d.hi) / 2;
    });
    const es = new CMAES(x0, { sigma: o.sigmaMul ?? 1, scales: dims.map((d) => d.sd), lambda: o.lambda, seed: f * 7919 + dims.length });
    const opt = { scale: o.scale, r: o.r, wcol: 1, weye: o.color ? 0 : 0.08, wseg: 1, wgrad: o.wgrad ?? 0.5, eyesOf: [lr.plan.id] };
    const toState = (x: number[]): State => {
      const s = { ...base };
      dims.forEach((d, i) => (s[d.key] = Math.min(d.hi, Math.max(d.lo, x[i]!))));
      return s;
    };
    const cost = (s: State, x: number[], res: Res) => {
      let c = res.loss;
      dims.forEach((d, i) => { const ex = Math.max(0, d.lo - x[i]!, x[i]! - d.hi); c += (ex / d.sd) ** 2 * 0.01; });
      if (o.prev) for (const d of dims) { const p = o.prev[d.key]; if (p !== undefined && !d.key.startsWith('c:')) c += 0.006 * ((s[d.key]! - p) / d.tau) ** 2; }
      if (lr.lastArea > 0 && res.area < 0.6 * lr.lastArea) c += (1 - res.area / (0.6 * lr.lastArea)) * 0.5;
      return c;
    };
    const evalStates = async (xs: number[][]) => {
      const states = xs.map(toState);
      const cands = states.map((s) => {
        const v = valsOf(lr, s, f, o.mode ?? 'frame', o.color);
        if (o.color) v.color = { brightness: s.brightness, contrast: s.contrast, saturation: s.saturation, multiply: [s.mr, s.mg, s.mb] };
        if (o.align) v.alignEyes = o.align;
        return { [lr.plan.id]: v };
      });
      const res = await pool.evalBatch<Res>(f, cands, opt);
      totalEvals += xs.length;
      res.forEach((r, i) => { if (r.tf) Object.assign(states[i]!, { x: r.tf[0], y: r.tf[1], ls: Math.log(r.tf[2]!), rot: r.tf[3] }); });
      return res.map((r, i) => ({ c: cost(states[i]!, xs[i]!, r), r, s: states[i]! }));
    };
    let best = (await evalStates([x0]))[0]!;
    let evals = 1;
    for (let g = 0; g < o.gens; g++) {
      const xs = es.ask();
      const rs = await evalStates(xs);
      evals += xs.length;
      es.tell(rs.map((r) => r.c));
      for (const r of rs) if (r.c < best.c) best = r;
    }
    return { state: best.s, loss: best.r.loss, ssim: best.r.ssim, area: best.r.area, eye: best.r.eye, cost: best.c, evals };
  };

  const perFrameDims = (lr: LayerRuntime, f: number) => {
    const p = lr.plan;
    const ds = [...TF_DIMS, ...HEAD_DIMS, ...BODY_DIMS, ...FACE_DIMS];
    if (inRanges(p.blurRanges, f)) ds.push(BLUR_DIM);
    if (inRanges(p.opacityRanges, f)) ds.push(OPACITY_DIM);
    if (p.clipRange && f >= p.clipRange[0] && f < p.clipRange[1]) ds.push(...CLIP_DIMS);
    if (p.hairBake) ds.push(...HAIR_DIMS);
    return ds;
  };
  const constDimsOf = (p: LayerPlan) => SHOT_CONST_DIMS.filter((d) => p.consts.includes(d.key));

  const commit = async (lr: LayerRuntime, f: number, s: State, mode: Mode) =>
    pool.all(([lid, f, v]) => (window as any).__fit.commit(f, { [lid]: v }), [lr.plan.id, f, valsOf(lr, s, f, mode, lr.plan.colorFilter)] as const);
  const copyFrames = async (lid: string, src: number, a: number, b: number) =>
    pool.all(([lid, src, a, b]) => (window as any).__fit.copyFrames(lid, src, a, b), [lid, src, a, b] as const);

  const preview = async (f: number, tag: string) => {
    const img = (await pool.one((f) => (window as any).__fit.render(f, 0.5), f)) as { width: number; height: number; data: string };
    const P = Buffer.from(img.data, 'base64');
    const R = ref.frame(f);
    const w = img.width, h = img.height;
    const out = new Uint8Array(w * 2 * h * 4);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const ro = (y * 2 * 1280 + x * 2) * 3, po = (y * w + x) * 4;
        const a = P[po + 3]! / 255;
        for (let k = 0; k < 3; k++) {
          out[(y * 2 * w + x) * 4 + k] = R[ro + k]!;
          out[(y * 2 * w + w + x) * 4 + k] = Math.round(P[po + k]! + (1 - a) * R[ro + k]!);
        }
        out[(y * 2 * w + x) * 4 + 3] = 255;
        out[(y * 2 * w + w + x) * 4 + 3] = 255;
      }
    writeFileSync(join(outDir, 'preview', `${tag}_${f}.png`), encodePng(w * 2, h, out, false));
  };

  // ---- 3. 逐图层拟合 ----
  const until = Math.min(opts.until ?? shot.end, shot.end);
  let fitted = 0;
  const lastEyes = new Map<string, Eyes>();
  const refEyes = (lr: LayerRuntime, f: number): Eyes | null => {
    const guesses: number[][][] = [];
    const prev = lastEyes.get(lr.plan.id);
    if (prev) guesses.push(prev);
    const sh = trackAt(lr.plan.inst, 'eyes', f) as number[][] | null;
    if (sh) guesses.push(sh);
    for (const g of guesses) {
      const e = detectRefEyes(ref.frame(f), g, frameExcl(f));
      if (e) { lastEyes.set(lr.plan.id, e); return e; }
    }
    return null;
  };

  const pickAnchor = (lr: LayerRuntime): number | undefined => {
    let best: number | undefined, bestScore = -1;
    const lo = Math.max(lr.plan.in, opts.start ?? 0, lr.plan.firstFitFrom ?? 0);
    for (let f = lo; f < Math.min(lr.plan.out, until); f++) {
      if (!lr.newFrames.has(f)) continue;
      const bb = trackAt(lr.plan.inst, 'bbox', f) as number[] | null;
      if (!bb) continue;
      const vis = visibleFraction(bb, frameExcl(f));
      if (vis < 0.5) continue;
      const smp = lr.plan.inst.track.filter((t) => t.eyes && Math.abs(t.frame - f) <= 1 && (t.source ?? '').includes('iris'));
      const conf = smp.length ? Math.max(...smp.map((t) => t.confidence ?? 0)) : 0.2;
      const g = trackAt(lr.plan.inst, 'eyes', f) as number[][] | null;
      const det = g ? detectRefEyes(ref.frame(f), g, frameExcl(f)) : null;
      // 裁切/淡入段里的帧不适合当锚点
      const clipped = (lr.plan.clipRange && f < lr.plan.clipRange[1]) || inRanges(lr.plan.opacityRanges, f);
      const score = vis * (0.3 + conf) + (det ? 0.5 : 0) - (clipped ? 1 : 0) - 0.0005 * (f - lo);
      if (score > bestScore + 1e-9) { bestScore = score; best = f; }
    }
    return best;
  };

  const logFrame = (lr: LayerRuntime, f: number) => {
    const last = lr.log[lr.log.length - 1]!;
    log(`[${shot.id}] ${lr.plan.id} ${last.stage} f=${f} loss ${last.loss.toFixed(4)} ssim≈${last.ssim.toFixed(3)} 眼 ${last.eye.toFixed(1)}px 面积 ${last.area} ${last.evals} 次 ${(last.ms / 1000).toFixed(1)}s${last.reanchored ? ' [重锚]' : ''}`);
  };

  /** 多起点搜索（锚点与重锚共用）：头部/身体角度网格，变换由眼睛对齐求出（有虹膜时） */
  const globalSearch = async (lr: LayerRuntime, f: number, s0: State, mode: Mode, withConsts: boolean, light = false) => {
    const p = lr.plan;
    const align = refEyes(lr, f);
    const cdims = withConsts ? constDimsOf(p) : [];
    const poseDims = [...(align ? [] : TF_DIMS), ...HEAD_DIMS, ...BODY_DIMS];
    const quick = !!process.env.FIT_QUICK;
    // 镜头常量的网格：体型（Clothes2）与前倾各取几档
    const cgrid: State[] = [{}];
    if (withConsts && p.consts.includes('c:Clothes2')) cgrid.splice(0, 1, ...[0, 0.4, 1].map((v) => ({ 'c:Clothes2': v })));
    if (withConsts && p.consts.includes('c:LeanInput')) {
      const g2: State[] = [];
      for (const c of cgrid) for (const v of [0, -15, -30]) g2.push({ ...c, 'c:LeanInput': v });
      cgrid.splice(0, cgrid.length, ...g2);
    }
    let evals = 0;
    const starts: { cost: number; state: State }[] = [];
    for (const cg of quick ? [cgrid[0]!] : cgrid)
      for (const hx of quick ? [0] : [-15, 0, 15])
        for (const hz of quick ? [0] : [-10, 0, 10]) {
          const r = await optimize(lr, f, { ...s0, ...cg, hx, hz, bx: hx * 0.5, bz: hz * 0.5 }, poseDims,
            { scale: 0.125, r: 1, gens: 3, lambda: 10, sigmaMul: 1.5, prev: null, mode, align, wgrad: 1 });
          evals += r.evals;
          starts.push({ cost: r.cost, state: r.state });
        }
    starts.sort((a, b) => a.cost - b.cost);
    // 最好的 3 个起点在 1/4 分辨率上继续
    let best = null as Awaited<ReturnType<typeof optimize>> | null;
    for (const st of starts.slice(0, quick || light ? 1 : 3)) {
      const r = await optimize(lr, f, st.state, [...poseDims, ...cdims], { scale: 0.25, r: 2, gens: 15, lambda: 14, sigmaMul: 1, prev: null, mode, align, wgrad: 0.7 });
      evals += r.evals;
      if (!best || r.cost < best.cost) best = r;
    }
    // 分块坐标下降：维度一多（20+）CMA-ES 在有限代数内收敛不了（合成数据实验：真值损失 0.28、单次联合优化停在 1.1），
    // 所以按「变换+头」「身体+体型」「表情」「其余」轮流优化，每块维度少、收敛快
    let cur = best!;
    const blocks = refineBlocks(lr, f, cdims, !!align);
    for (let round = 0; round < (quick || light ? 1 : 3); round++)
      for (const b of blocks) {
        const r = await optimize(lr, f, cur.state, b.dims, { scale: 0.5, r: 3, gens: b.gens, lambda: 12, sigmaMul: round ? 0.6 : 1, prev: null, mode, align: b.align ? align : null, wgrad: 0.5 });
        evals += r.evals;
        if (r.cost <= cur.cost) cur = r;
      }
    return { ...cur, evals, align };
  };

  /** 分块：每块的维度与代数；align 表示该块里变换由眼睛对齐求出 */
  const refineBlocks = (lr: LayerRuntime, f: number, cdims: Dim[], hasAlign: boolean) => {
    const p = lr.plan;
    const extra = perFrameDims(lr, f).filter((d) => ![...TF_DIMS, ...HEAD_DIMS, ...BODY_DIMS, ...FACE_DIMS].includes(d));
    const shape = cdims.filter((d) => d.key === 'c:LeanInput' || d.key === 'c:Clothes2');
    const misc = cdims.filter((d) => !shape.includes(d));
    const blocks: { dims: Dim[]; gens: number; align: boolean }[] = [
      { dims: [...(hasAlign ? [] : TF_DIMS), ...HEAD_DIMS], gens: 15, align: true },
      { dims: [...BODY_DIMS, ...shape], gens: 15, align: true },
      { dims: FACE_DIMS, gens: 12, align: true },
      { dims: [...TF_DIMS, ...extra, ...misc], gens: 12, align: false },
    ];
    void p;
    return blocks.filter((b) => b.dims.length);
  };

  const fitFrame = async (lr: LayerRuntime, f: number, init: State, prev: State, mode: Mode, gm = 1) => {
    const align = refEyes(lr, f);
    const ga = Math.max(3, Math.round(6 * gm * effort)), gb = Math.max(4, Math.round(10 * gm * effort));
    const a = align
      ? await optimize(lr, f, init, [...HEAD_DIMS, ...BODY_DIMS, ...FACE_DIMS.filter((d) => d.key.startsWith('eye') || d.key.startsWith('eb'))], { scale: 0.25, r: 2, gens: ga, lambda: 10, prev, mode, align, wgrad: 0.7 })
      : await optimize(lr, f, init, [...TF_DIMS, ...HEAD_DIMS, ...BODY_DIMS], { scale: 0.25, r: 2, gens: ga, lambda: 10, prev, mode, wgrad: 0.7 });
    const b = await optimize(lr, f, a.state, perFrameDims(lr, f), { scale: 0.5, r: 3, gens: gb, lambda: 14, prev, mode, wgrad: 0.5 });
    return { ...b, evals: a.evals + b.evals, align };
  };

  /** 发散检测：损失相对上一帧突增，或虹膜误差超过眼距的 15% */
  const diverged = (r: { loss: number; eye: number; align: Eyes | null }, prevLoss: number) => {
    const ed = r.align ? Math.hypot(r.align[1][0] - r.align[0][0], r.align[1][1] - r.align[0][1]) : 0;
    return r.loss > prevLoss + 0.25 || (ed > 0 && r.eye > 0.15 * ed);
  };

  const extrapolate = (cur: State, older: State | null, k: number): State => {
    if (!older) return cur;
    const s = { ...cur };
    for (const d of TF_DIMS) s[d.key] = cur[d.key]! + (cur[d.key]! - older[d.key]!) * k;
    return s;
  };

  const order = opts.order ? opts.order.map((id) => L.get(id)!) : [...L.values()].sort((a, b) => b.plan.z - a.plan.z);
  const done = new Set<string>();
  for (const lr of order) {
    const p = lr.plan;
    if (reused.has(p.id)) { done.add(p.id); continue; }
    await pool.all((h) => (window as any).__fit.setHidden(h), order.filter((o) => o !== lr && !done.has(o.plan.id) && !reused.has(o.plan.id)).map((o) => o.plan.id));
    curRefFrame = -1;
    lastEyes.delete(p.id);
    const fA = pickAnchor(lr);
    if (fA === undefined) { log(`[${shot.id}] ${p.id}: 没有可拟合帧`); continue; }
    const news = [...lr.newFrames].filter((f) => f >= p.in && f < Math.min(p.out, until)).sort((a, b) => a - b);
    log(`[${shot.id}] ${p.id}: 锚点帧 ${fA}（新画面 ${news.length} 帧）`);
    const states = new Map<number, State>();
    const losses = new Map<number, number>();

    // (a) 锚点
    {
      const f = fA;
      const tf = Date.now();
      await setFrame(f);
      let s = initialState(p, calib, f);
      if (p.clipRange) Object.assign(s, { cx: 640, cy: 360, cr: 400 });
      const g = await globalSearch(lr, f, s, 'all', true);
      s = g.state;
      log(`[${shot.id}] ${p.id} 锚点 ${f}：虹膜 ${g.align ? '已检测' : '未检测到'}，常量 ${JSON.stringify(Object.fromEntries(Object.entries(s).filter(([k]) => k.startsWith('c:')).map(([k, v]) => [k, +v.toFixed(3)])))}`);
      let colEvals = 0;
      if (p.colorFilter) {
        const cs: State = { ...s, brightness: 0, contrast: 1, saturation: 1, mr: 1, mg: 1, mb: 1 };
        const col = await optimize(lr, f, cs, COLOR_DIMS, { scale: 0.5, r: 3, gens: 20, lambda: 12, sigmaMul: 1.5, prev: null, color: true, mode: 'all' });
        const c = col.state;
        lr.color = { brightness: c.brightness!, contrast: c.contrast!, saturation: c.saturation!, multiply: [c.mr!, c.mg!, c.mb!] };
        colEvals = col.evals;
        log(`[${shot.id}] ${p.id} 颜色 ${JSON.stringify(Object.fromEntries(Object.entries(lr.color).map(([k, v]) => [k, Array.isArray(v) ? v.map((x) => +x.toFixed(3)) : +v.toFixed(3)])))}`);
      }
      await commit(lr, f, s, 'all');
      await copyFrames(p.id, f, p.in, f); // 锚点之前的帧（变换、滤镜等）先等于锚点
      states.set(f, s);
      losses.set(f, g.loss);
      lr.lastArea = g.area;
      lr.log.push({ frame: f, stage: '锚点', loss: g.loss, ssim: g.ssim, area: g.area, eye: g.eye, evals: g.evals + colEvals, ms: Date.now() - tf });
      fitted++;
      logFrame(lr, f);
      if (opts.previewEvery) await preview(f, p.id);
    }

    // (b) 锚点之前：时间倒序
    {
      let next = fA;
      let nextOlder: number | null = null;
      for (const f of news.filter((g) => g < fA).reverse()) {
        const tf = Date.now();
        await setFrame(f);
        const bb = trackAt(p.inst, 'bbox', f) as number[] | null;
        const vis = bb ? visibleFraction(bb, frameExcl(f)) : 0;
        const ns = states.get(next)!;
        const init = extrapolate(ns, nextOlder !== null ? states.get(nextOlder)! : null, (next - f) / Math.max(1, (nextOlder ?? next) - next || 1));
        let s = ns;
        if (vis >= 0.15) {
          let r = await fitFrame(lr, f, init, ns, 'win', 0.8);
          let re = false;
          if (diverged(r, losses.get(next) ?? r.loss)) {
            const g = await globalSearch(lr, f, ns, 'win', false, true);
            if (g.cost < r.cost) { r = { ...g, evals: g.evals + r.evals }; re = true; }
          }
          s = r.state;
          losses.set(f, r.loss);
          lr.log.push({ frame: f, stage: '倒序', loss: r.loss, ssim: r.ssim, area: r.area, eye: r.eye, evals: r.evals, ms: Date.now() - tf, reanchored: re });
          fitted++;
          logFrame(lr, f);
        }
        await commit(lr, f, s, 'win');
        await copyFrames(p.id, f, p.in, Math.max(p.in, f - BACK_WINDOW));
        await copyFrames(p.id, f, f + 1, next - 1);
        states.set(f, s);
        nextOlder = next;
        next = f;
      }
    }

    // (c) 锚点重拟合（真实物理历史）→ 时间正序
    let lastFit = fA;
    let prevFit: number | null = null;
    lr.lastFit = null;
    const before = news.filter((g) => g < fA);
    if (before.length) {
      const tf = Date.now();
      await setFrame(fA);
      const r = await fitFrame(lr, fA, states.get(fA)!, states.get(fA)!, 'frame', 0.7);
      if (r.loss < (losses.get(fA) ?? Infinity) + 0.1) {
        states.set(fA, r.state);
        losses.set(fA, r.loss);
      }
      await commit(lr, fA, states.get(fA)!, 'frame');
      lr.log.push({ frame: fA, stage: '锚点重拟合', loss: r.loss, ssim: r.ssim, area: r.area, eye: r.eye, evals: r.evals, ms: Date.now() - tf });
      logFrame(lr, fA);
      prevFit = before[before.length - 1]!;
    }
    lr.lastFit = fA;
    for (const f of news.filter((g) => g > fA)) {
      const tf = Date.now();
      await copyFrames(p.id, lastFit, lastFit + 1, f + 1);
      await setFrame(f);
      const bb = trackAt(p.inst, 'bbox', f) as number[] | null;
      const vis = bb ? visibleFraction(bb, frameExcl(f)) : 0;
      const cur = states.get(lastFit)!;
      if (vis < 0.15) {
        await commit(lr, f, cur, 'frame');
        states.set(f, cur);
        lr.lastFit = lastFit = f;
        continue;
      }
      const init = extrapolate(cur, prevFit !== null ? states.get(prevFit)! : null, (f - lastFit) / Math.max(1, lastFit - (prevFit ?? lastFit)));
      let r = await fitFrame(lr, f, init, cur, 'frame');
      let re = false;
      if (diverged(r, losses.get(lastFit) ?? r.loss)) {
        const g = await globalSearch(lr, f, cur, 'frame', false, true);
        if (g.cost < r.cost) { r = { ...g, evals: g.evals + r.evals }; re = true; }
      }
      states.set(f, r.state);
      losses.set(f, r.loss);
      lr.lastArea = 0.5 * lr.lastArea + 0.5 * r.area;
      await commit(lr, f, r.state, 'frame');
      lr.log.push({ frame: f, stage: '正序', loss: r.loss, ssim: r.ssim, area: r.area, eye: r.eye, evals: r.evals, ms: Date.now() - tf, reanchored: re });
      fitted++;
      logFrame(lr, f);
      if (opts.previewEvery && fitted % opts.previewEvery === 0) await preview(f, p.id);
      prevFit = lastFit;
      lr.lastFit = lastFit = f;
    }
    await copyFrames(p.id, lastFit, lastFit + 1, p.out);
    done.add(p.id);
  }
  await pool.all((h) => (window as any).__fit.setHidden(h), [] as string[]);

  // ---- 4b. 开关的画面检测（拟合后）：用拟合出的虹膜屏幕位置（比 shots 的 eyes 准得多）重新检测，覆盖斜坡 ----
  const dumps = new Map<string, ReturnType<typeof dumpType>>();
  for (const p of plans) dumps.set(p.id, (await pool.one((lid) => (window as any).__fit.dump(lid), p.id)) as ReturnType<typeof dumpType>);
  for (const p of plans) {
    if (reused.has(p.id)) continue;
    for (const sw of p.switches) {
      if (!sw.detect) continue;
      const r = await redetectSwitch(pool, ref, p, sw);
      switchRamps.set(`${p.id}:${sw.param}`, r);
      const vals = dumps.get(p.id)!.params[sw.param]!;
      for (let f = p.in; f < p.out; f++) vals[f - p.in] = rampValue(sw.from, sw.to, r.start, r.fade, f);
      log(`[${shot.id}] ${p.id} 开关 ${sw.param}（拟合后用拟合的虹膜位置重测）：起点 ${r.start}、fade ${r.fade}（rms ${r.rms?.toFixed(3) ?? '-'}，${Object.keys(r.samples).length} 个样本）`);
    }
  }

  // ---- 5. 导出 ----
  const layers: unknown[] = [];
  for (const p of [...plans].sort((a, b) => a.z - b.z)) {
    const lr = L.get(p.id)!;
    const isNew = new Map<number, boolean>();
    for (let f = p.in; f < p.out; f++) isNew.set(f, lr.newFrames.has(f));
    const holds = holdSegments(isNew, p.in, p.out);
    layers.push(reused.get(p.id) ?? exportLayer(scene, p, dumps.get(p.id)!, lr, holds, shot));
  }
  const elapsed = (Date.now() - t0) / 1000;
  const timing = await pool.timing();
  const result = {
    shot: shot.id,
    range: [shot.start, shot.end],
    elapsedSec: elapsed,
    workers: pool.size,
    evals: totalEvals,
    fittedFrames: fitted,
    timing,
    layers,
    switches: Object.fromEntries(switchRamps),
    perFrame: Object.fromEntries([...L.values()].map((lr) => [lr.plan.id, lr.log])),
    cadence: cadenceLog,
  };
  writeFileSync(join(outDir, 'result.json'), JSON.stringify(result));
  log(`[${shot.id}] 完成：拟合 ${fitted} 帧，${totalEvals} 次渲染，${elapsed.toFixed(0)}s（${pool.size} 个页面）；页面计时 ${JSON.stringify(Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v)])))}`);
  ref.close();
  return result;
}

function dumpType() {
  return null as unknown as {
    tf: { x: number[]; y: number[]; scale: number[]; rotation: number[] };
    opacity: number[] | null;
    blur: number[] | null;
    clip: { x: number[]; y: number[]; width: number[]; height: number[]; radius: number[] } | null;
    color: Record<string, unknown> | null;
    params: Record<string, number[]>;
  };
}

/** 用页面里拟合结果的虹膜位置逐帧检测开关权重，并拟合斜坡（不影响物理：key* 开关不是物理输入） */
export async function redetectSwitch(pool: Pool, ref: RefFrames, p: LayerPlan, sw: SwitchPlan) {
  const samples: Record<number, number> = {};
  for (let f = Math.max(p.in, sw.window[0] - 40); f < Math.min(p.out, sw.window[1]); f++) {
    const e = (await pool.one(([lid, f]) => (window as any).__fit.irisScreen(lid, f), [p.id, f] as const)) as number[][];
    const w = heartEyeWeight(ref.frame(f), e);
    if (w !== null) samples[f] = w;
  }
  const ramp = fitSwitchRamp(new Map(Object.entries(samples).map(([k, v]) => [Number(k), v])), { ...sw, window: [sw.window[0] - 40, sw.window[1]] });
  return { ...ramp, samples };
}

export function rampValue(from: number, to: number, start: number, fade: number, f: number): number {
  if (f < start) return from;
  if (fade <= 0 || f >= start + fade) return to;
  const u = (f - start) / fade;
  return from + (to - from) * (0.5 - 0.5 * Math.cos(Math.PI * u));
}

/** 逐帧权重样本 → (起点, fade) 网格搜索最小平方误差；样本不足时用 shots 估计 */
function fitSwitchRamp(ws: Map<number, number>, sw: SwitchPlan) {
  if (ws.size < 3) return { start: sw.est, fade: sw.fade, rms: null as number | null };
  let best = { start: sw.est, fade: sw.fade, rms: Infinity as number | null };
  for (let s = sw.window[0] - 30; s < sw.window[1]; s++)
    for (let fd = 0; fd <= 40; fd++) {
      let e = 0;
      for (const [f, w] of ws) e += (rampValue(sw.from, sw.to, s, fd, f) - w) ** 2;
      const rms = Math.sqrt(e / ws.size);
      if (rms < best.rms! - 1e-9) best = { start: s, fade: fd, rms };
    }
  return best;
}

export function exportLayer(
  scene: ReturnType<typeof makeScene>, p: LayerPlan, d: ReturnType<typeof dumpType>, lr: LayerRuntime,
  holds: { start: number; end: number; every: number }[], shot: Shot,
) {
  const base = scene.layers.find((x) => x.id === p.id)! as any;
  const r6 = (v: number) => Math.round(v * 1e6) / 1e6;
  const bk = (vals: number[]) => ({ baked: { start: p.in, values: vals.map(r6) } });
  const prov = (what: string) => ({ source: 'fit' as const, tool: '@charaforge/fit', detail: { shot: shot.id, what } });
  const filters: unknown[] = [];
  const c = d.color as any;
  if (c) filters.push({ type: 'color', brightness: r6(c.brightness), contrast: r6(c.contrast), saturation: r6(c.saturation), multiply: (c.multiply as number[]).map(r6) });
  const rs = (d as any).resample as number | null | undefined;
  if (rs !== undefined && rs !== null && rs < 1) filters.push({ type: 'resample', factor: r6(rs), method: 'bilinear' });
  if (d.blur) filters.push({ type: 'blur', sigma: bk(d.blur) });
  const clip = d.clip ? { x: bk(d.clip.x), y: bk(d.clip.y), width: bk(d.clip.width), height: bk(d.clip.height), radius: bk(d.clip.radius), feather: 2 } : undefined;
  return {
    id: p.id,
    name: `${shot.id} ${p.inst.role}`,
    kind: 'model',
    model: 'dsc',
    in: p.in,
    out: p.out,
    transform: { x: bk(d.tf.x), y: bk(d.tf.y), scale: bk(d.tf.scale), ...((d as any).sx ? { scaleX: bk((d as any).sx) } : {}), rotation: bk(d.tf.rotation), anchorX: base.transform.anchorX, anchorY: base.transform.anchorY },
    ...(d.opacity ? { opacity: bk(d.opacity) } : {}),
    ...(clip ? { clip } : {}),
    filters,
    ...(holds.length ? { frameHold: holds } : {}),
    physics: { warmup: 60, stabilize: true },
    params: base.params.map((t: any) => {
      if (t.curve.keys) return { id: t.id, curve: t.curve, semantics: { tags: ['switch'] }, provenance: { source: 'fit', tool: '@charaforge/fit', detail: { shot: shot.id, from: 'shots.variant' } } };
      const key = t.stage === 'postPhysics' ? `post:${t.id}` : t.id;
      const what = p.switches.some((s) => s.param === t.id) ? 'switch-ramp (image-detected)'
        : PARAM_CONSTS.includes(t.id) ? 'shot-constant'
        : t.stage === 'postPhysics' && t.mode === 'add' ? 'hair-bake (per-group offset added to physics output)'
        : t.stage === 'postPhysics' ? 'pose output baked (overrides lagged physics output; inputs still drive secondary physics)'
        : t.id === 'ParamBreath' ? 'breathing sine (shot-constant phase/period)'
        : PHYS_INPUT_IDS.includes(t.id) ? 'per-frame (physics input, value at f drives frame f+1)' : 'per-frame';
      const out: Record<string, unknown> = { id: t.id, curve: bk(d.params[key]!), provenance: prov(what) };
      if (t.stage) out.stage = t.stage;
      if (t.mode) out.mode = t.mode;
      return out;
    }),
    provenance: { source: 'fit', tool: '@charaforge/fit', detail: { shot: shot.id, instance: p.inst.id, fittedFrames: lr.log.length } },
  };
}
