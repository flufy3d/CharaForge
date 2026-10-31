/*
 * 逐镜头拟合。
 *
 * 流程（每个镜头）：
 *   1. 解码参考帧；按实例在角色区域内检测帧重复，只拟合「新画面」帧，其余帧沿用上一新帧的值并输出 frameHold；
 *   2. 构造只含本镜头图层的拟合场景（transform / opacity / blur / 参数全部是逐帧 baked 曲线），交给页面里的拟合引擎；
 *   3. 首个可拟合帧 f0：用 shots 的 eyes 求初始变换（锚点 = 模型参考姿态下的双眼中点），
 *      多起点（头部角度网格 × 前倾）短程 CMA-ES → 粗（1/4 分辨率）→ 细（1/2 分辨率）；f0 阶段的候选写满 [in, f0]，
 *      每次从 stabilize + 预热重新推进，所以首帧的物理状态与最终场景完全一致；
 *   4. 之后逐个新帧：以上一拟合帧为初值，粗 → 细两级 CMA-ES。物理输入（AngleX/Y/Z、PositionX/Y/Z）写在 f−1 和 f
 *      （Cubism 物理输出滞后一帧），直接参数（眼、嘴、眉）写在 f；每个候选都从缓存的规范状态 S(f−2) 推进，
 *      物理状态逐帧正向推进；
 *   5. 导出图层 JSON（逐帧曲线 + frameHold + provenance）与日志。
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Page } from 'playwright';
import type { Shot, ShotInstance } from '../../../packages/scene/src/index.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { CMAES } from './cmaes.ts';
import { BLUR_DIM, DEFAULT_STATE, FACE_DIMS, FIT_PARAM_IDS, HEAD_DIMS, OPACITY_DIM, PHYS_INPUT_IDS, TF_DIMS, toParams, type Dim } from './model.ts';
import { FIT_DIR, MODEL_URI } from './paths.ts';
import { RefFrames } from './refvideo.ts';
import { detectNewFrames, detectRefEyes, exclusionsAt, holdSegments, trackAt, visibleFraction } from './shotsutil.ts';

export interface ModelCalib {
  /** 参考姿态下左右眼（画面左、画面右）虹膜中心，模型画布像素 */
  eyes: [[number, number], [number, number]];
  canvas: { width: number; height: number };
  /** 青色质心 − 虹膜网格顶点质心（画布像素） */
  irisOffset: [[number, number], [number, number]];
}

export interface LayerPlan {
  /** 图层 ID（= shots 实例 ID） */
  id: string;
  inst: ShotInstance;
  in: number;
  out: number;
  /** 开关参数：常量（shots 的 variant） */
  constants: Record<string, number>;
  /** 开关切换：在窗口内把权重作为额外维度逐帧拟合，结束后拟合成斜坡（精确到帧） */
  switches: { param: string; from: number; to: number; window: [number, number]; est: number; fade: number }[];
  /** 在这些帧区间内拟合 blur sigma */
  blurRanges: [number, number][];
  fitOpacity: boolean;
  /** 镜头常量维度（整段同值、经物理慢响应的输入，如前倾 LeanInput），只在首帧全局搜索里拟合 */
  shotConsts: Dim[];
  /** 是否拟合颜色滤镜（镜头内常量） */
  colorFilter: boolean;
  z: number;
  /** 首个拟合帧不早于此帧 */
  firstFitFrom?: number;
  /** 拟合本图层时，把这些实例（shots 的 bbox）当作额外排除区（只影响拟合损失），用于被前景实例遮挡的背景实例 */
  fitExcludeInstances?: string[];
}

export interface FitOptions {
  until?: number;
  start?: number;
  previewEvery?: number;
  gensA?: number;
  gensB?: number;
  lambdaA?: number;
  lambdaB?: number;
  /** 计入损失时忽略这些原因的排除区（只影响拟合，不影响指标） */
  fitIgnoreExclusions?: string[];
  /** 图层拟合顺序（图层 ID）；尚未拟合的图层在拟合时隐藏。缺省按 z 从前到后 */
  order?: string[];
  /** 只拟合这些图层，其余图层沿用上一次 result.json 的结果 */
  only?: string[];
  log?: (s: string) => void;
}

type State = Record<string, number>;
/** frame：物理输入写 f−1 与 f、直接参数写 f；win：写满 [f−BACK_WINDOW, f]（倒序拟合用）；all：写满 [in, f]（锚点用） */
type Mode = 'frame' | 'win' | 'all';
const BACK_WINDOW = 8;
type Res = { loss: number; area: number; ssim: number; eye: number; tf?: number[] };
type Eyes = [[number, number], [number, number]];

interface LayerRuntime {
  plan: LayerPlan;
  newFrames: Set<number>;
  state: State | null;
  lastFit: number | null;
  color: { brightness: number; contrast: number; saturation: number; multiply: [number, number, number] };
  lastArea: number;
  log: { frame: number; loss: number; ssim: number; area: number; eye: number; evals: number; ms: number }[];
  switchW: Map<string, Map<number, number>>;
}

const COLOR_DIMS: Dim[] = [
  { key: 'brightness', lo: -0.3, hi: 0.3, sd: 0.03, tau: 1 },
  { key: 'contrast', lo: 0.6, hi: 1.5, sd: 0.05, tau: 1 },
  { key: 'saturation', lo: 0.5, hi: 1.6, sd: 0.06, tau: 1 },
  { key: 'mr', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
  { key: 'mg', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
  { key: 'mb', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
];

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
      const filters: unknown[] = p.colorFilter ? [{ type: 'color', brightness: 0, contrast: 1, saturation: 1, multiply: [1, 1, 1] }] : [];
      if (p.blurRanges.length) filters.push({ type: 'blur', sigma: baked(s.blur ?? 0) });
      const pre = toParams(s);
      return {
        id: p.id,
        kind: 'model',
        model: 'dsc',
        in: p.in,
        out: p.out,
        transform: { x: baked(s.x!), y: baked(s.y!), scale: baked(Math.exp(s.ls!)), rotation: baked(s.rot!), anchorX: ax, anchorY: ay },
        ...(p.fitOpacity ? { opacity: baked(s.opacity ?? 1) } : {}),
        filters,
        physics: { warmup: 60, stabilize: true },
        params: [
          ...Object.entries(p.constants).map(([id, v]) => ({ id, curve: { keys: [{ t: p.in, v }] } })),
          ...p.switches.map((sw) => ({ id: sw.param, curve: baked(sw.from) })),
          ...p.shotConsts.map((d) => ({ id: d.key, curve: baked(s[`c:${d.key}`] ?? 0) })),
          ...FIT_PARAM_IDS.map((id) => ({ id, curve: baked(pre[id]!) })),
        ],
      };
    }),
  };
}

function initialState(plan: LayerPlan, calib: ModelCalib, f: number): State {
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
    if (head) {
      x = head[0]! + head[2]! / 2;
      y = head[1]! + head[3]! * 0.6;
      scale = head[2]! / 1100;
    }
  }
  const consts = Object.fromEntries(plan.shotConsts.map((d) => [`c:${d.key}`, Math.min(d.hi, Math.max(d.lo, 0))]));
  return { ...DEFAULT_STATE, ...consts, x, y, ls: Math.log(scale), rot, blur: (trackAt(plan.inst, 'blurSigma', f) as number | null) ?? 0 };
}

export async function fitShot(page: Page, shot: Shot, plans: LayerPlan[], calib: ModelCalib, opts: FitOptions = {}) {
  const log = opts.log ?? ((s: string) => console.log(s));
  const outDir = join(FIT_DIR, shot.id);
  mkdirSync(join(outDir, 'preview'), { recursive: true });
  const t0 = Date.now();
  const ref = new RefFrames(shot.start, shot.end);

  // ---- 1. 帧重复检测 ----
  const L = new Map<string, LayerRuntime>();
  const cadenceLog: Record<string, unknown> = {};
  for (const p of plans) {
    const { isNew, d } = detectNewFrames(ref, shot, p.inst, p.in, p.out);
    const nf = new Set([...isNew.entries()].filter(([, v]) => v).map(([f]) => f));
    cadenceLog[p.id] = { newCount: nf.size, total: p.out - p.in, d: Object.fromEntries([...d].map(([f, v]) => [f, +v.toFixed(2)])) };
    L.set(p.id, {
      plan: p, newFrames: nf, state: null, lastFit: null,
      color: { brightness: 0, contrast: 1, saturation: 1, multiply: [1, 1, 1] },
      lastArea: 0, log: [], switchW: new Map(p.switches.map((s) => [s.param, new Map()])),
    });
    log(`[${shot.id}] ${p.id}: 新画面 ${nf.size}/${p.out - p.in} 帧`);
  }

  // ---- 2. 拟合场景 ----
  const init = new Map(plans.map((p) => [p.id, initialState(p, calib, p.in)]));
  const scene = makeScene(shot, plans, calib, init);
  // 沿用上一次的图层（--only）：拟合时直接用导出的图层对象（逐帧曲线、滤镜、开关），导出时原样写回
  const reused = new Map<string, any>();
  if (opts.only) {
    const prev = JSON.parse(readFileSync(join(outDir, 'result.json'), 'utf8'));
    for (const l of prev.layers) if (!opts.only.includes(l.id)) reused.set(l.id, l);
    scene.layers = scene.layers.map((l) => reused.get(l.id) ?? l) as typeof scene.layers;
    log(`[${shot.id}] 沿用上次结果：${[...reused.keys()].join(', ')}`);
  }
  await page.evaluate(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [scene.layers, MODEL_URI, calib.irisOffset] as const);

  let extraExcl: string[] = [];
  const frameExcl = (f: number) => {
    const ex = exclusionsAt(shot, f, { skipReasons: opts.fitIgnoreExclusions });
    for (const id of extraExcl) {
      const inst = shot.instances.find((i) => i.id === id);
      const bb = inst ? (trackAt(inst, 'bbox', f) as number[] | null) : null;
      if (bb) ex.push(bb);
    }
    return ex;
  };
  let curRefFrame = -1;
  const setFrame = async (f: number) => {
    if (curRefFrame === f) return;
    await page.evaluate(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), frameExcl(f)] as const);
    curRefFrame = f;
  };

  /** 状态 → 页面写入（参数按帧写：物理输入 f−1 与 f，直接参数 f；mode='all' 时写满 [in, f]） */
  const valsOf = (lr: LayerRuntime, s: State, f: number, mode: Mode, withColor = false) => {
    const p = lr.plan;
    const params: Record<string, number> = { ...toParams(s) };
    for (const sw of p.switches) params[sw.param] = s[`sw:${sw.param}`] ?? (f >= sw.est ? sw.to : sw.from);
    for (const d of p.shotConsts) params[d.key] = s[`c:${d.key}`] ?? 0;
    const phys: Record<string, number> = {};
    for (const id of [...PHYS_INPUT_IDS, ...p.shotConsts.map((d) => d.key)]) phys[id] = params[id]!;
    const set: [number, Record<string, number>][] = [];
    if (mode === 'all') for (let g = p.in; g <= f; g++) set.push([g, params]);
    else if (mode === 'win') for (let g = Math.max(p.in, f - BACK_WINDOW); g <= f; g++) set.push([g, params]);
    else {
      // f−1 若不是显示出来的帧（帧保持段内），直接参数也一起写，物理的次级输入（眼球、眉毛联动等）更连贯
      const prevShown = lr.newFrames.has(f - 1) && lr.lastFit === f - 1;
      if (f - 1 >= p.in) set.push([f - 1, prevShown ? phys : params]);
      set.push([f, params]);
    }
    const v: Record<string, unknown> = { tf: [s.x, s.y, Math.exp(s.ls!), s.rot], set };
    if (p.blurRanges.length) v.blur = s.blur ?? 0;
    if (p.fitOpacity) v.opacity = s.opacity ?? 1;
    if (withColor) v.color = lr.color;
    return v;
  };

  let totalEvals = 0;
  /** 用 CMA-ES 优化图层在帧 f 的若干维度 */
  const optimize = async (
    lr: LayerRuntime, f: number, base: State, dims: Dim[],
    o: { scale: number; r: number; gens: number; lambda: number; sigmaMul?: number; prev: State | null; mode?: Mode; color?: boolean; align?: Eyes | null },
  ) => {
    // 新出现的维度（如开关权重 sw:*）在初值里还没有：开关取「估计切换帧之前/之后」的值，其余取区间中点
    const x0 = dims.map((d) => {
      const v = base[d.key];
      if (v !== undefined && Number.isFinite(v)) return v;
      const sw = lr.plan.switches.find((w) => `sw:${w.param}` === d.key);
      return sw ? (f >= sw.est ? sw.to : sw.from) : (d.lo + d.hi) / 2;
    });
    const es = new CMAES(x0, { sigma: o.sigmaMul ?? 1, scales: dims.map((d) => d.sd), lambda: o.lambda, seed: f * 7919 + dims.length });
    const opt = { scale: o.scale, r: o.r, wcol: 1, weye: o.color ? 0 : 0.08, eyesOf: [lr.plan.id] };
    const toState = (x: number[]): State => {
      const s = { ...base };
      dims.forEach((d, i) => (s[d.key] = Math.min(d.hi, Math.max(d.lo, x[i]!))));
      return s;
    };
    const cost = (s: State, x: number[], res: Res) => {
      let c = res.loss;
      dims.forEach((d, i) => { const ex = Math.max(0, d.lo - x[i]!, x[i]! - d.hi); c += (ex / d.sd) ** 2 * 0.01; });
      if (o.prev) for (const d of dims) { const p = o.prev[d.key]; if (p !== undefined) c += 0.006 * ((s[d.key]! - p) / d.tau) ** 2; }
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
      const res = (await page.evaluate(([f, c, op]) => (window as any).__fit.evalBatch(f, c, op), [f, cands, opt] as const)) as Res[];
      totalEvals += xs.length;
      // 眼睛对齐时，变换由页面按虹膜求出，写回状态
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

  const dimsFor = (lr: LayerRuntime, f: number) => {
    const ds = [...TF_DIMS, ...HEAD_DIMS, ...FACE_DIMS];
    if (lr.plan.blurRanges.some(([a, b]) => f >= a && f < b)) ds.push(BLUR_DIM);
    if (lr.plan.fitOpacity) ds.push(OPACITY_DIM);
    for (const sw of lr.plan.switches)
      if (f >= sw.window[0] && f < sw.window[1]) ds.push({ key: `sw:${sw.param}`, lo: Math.min(sw.from, sw.to), hi: Math.max(sw.from, sw.to), sd: 0.25, tau: 2 });
    return ds;
  };

  const commit = async (lr: LayerRuntime, f: number, s: State, mode: Mode) =>
    page.evaluate(([lid, f, v]) => (window as any).__fit.commit(f, { [lid]: v }), [lr.plan.id, f, valsOf(lr, s, f, mode, lr.plan.colorFilter)] as const);

  const preview = async (f: number, tag: string) => {
    const img = (await page.evaluate((f) => (window as any).__fit.render(f, 0.5), f)) as { width: number; height: number; data: string };
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

  // ---- 3/4. 逐图层拟合：锚点帧全局搜索 → 向前（时间倒序）→ 锚点重拟合 → 向后 ----
  const until = Math.min(opts.until ?? shot.end, shot.end);
  const firstFit = new Map<string, number>();
  let fitted = 0;

  /** 锚点：可见比例高、shots 的眼睛是虹膜细化/人工高置信的新画面帧（同分取最早） */
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
      const score = vis * (0.3 + conf) + (det ? 0.5 : 0) - 0.0005 * (f - lo);
      if (score > bestScore + 1e-9) { bestScore = score; best = f; }
    }
    return best;
  };

  const logFrame = (lr: LayerRuntime, f: number, tag: string) => {
    const last = lr.log[lr.log.length - 1]!;
    log(`[${shot.id}] ${lr.plan.id} ${tag} f=${f} loss ${last.loss.toFixed(4)} ssim≈${last.ssim.toFixed(3)} 眼 ${last.eye.toFixed(1)}px 面积 ${last.area} ${last.evals} 次 ${(last.ms / 1000).toFixed(1)}s`);
  };

  /** 单帧两级优化（粗：变换 + 头部角度；细：全部维度） */
  /** 参考帧虹膜（以上一帧检测结果或 shots 的 eyes 为搜索中心）；检测不到返回 null */
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

  const fitFrame = async (lr: LayerRuntime, f: number, init: State, prev: State, mode: Mode, gm = 1) => {
    const align = refEyes(lr, f);
    const a = align
      ? await optimize(lr, f, init, [...HEAD_DIMS, ...FACE_DIMS.filter((d) => d.key.startsWith('eye') || d.key.startsWith('eb'))], { scale: 0.25, r: 2, gens: Math.round((opts.gensA ?? 8) * gm), lambda: opts.lambdaA ?? 10, prev, mode, align })
      : await optimize(lr, f, init, [...TF_DIMS, ...HEAD_DIMS], { scale: 0.25, r: 2, gens: Math.round((opts.gensA ?? 8) * gm), lambda: opts.lambdaA ?? 10, prev, mode });
    const b = await optimize(lr, f, a.state, dimsFor(lr, f), { scale: 0.5, r: 3, gens: Math.round((opts.gensB ?? 12) * gm), lambda: opts.lambdaB ?? 14, prev, mode });
    return { ...b, evals: a.evals + b.evals };
  };

  /** 匀速外推变换维度作为初值（推拉镜头时比直接用上一帧好得多） */
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
    // 尚未拟合的其他图层先隐藏；本图层的额外排除区
    await page.evaluate((h) => (window as any).__fit.setHidden(h), order.filter((o) => o !== lr && !done.has(o.plan.id)).map((o) => o.plan.id));
    extraExcl = p.fitExcludeInstances ?? [];
    curRefFrame = -1;
    const fA = pickAnchor(lr);
    if (fA === undefined) { log(`[${shot.id}] ${p.id}: 没有可拟合帧`); continue; }
    firstFit.set(p.id, p.in);
    const news = [...lr.newFrames].filter((f) => f >= p.in && f < Math.min(p.out, until)).sort((a, b) => a - b);
    log(`[${shot.id}] ${p.id}: 锚点帧 ${fA}（新画面 ${news.length} 帧）`);
    const states = new Map<number, State>();

    // (a) 锚点：多起点全局搜索（候选写满 [in, fA]，物理从 stabilize + 预热重新推进）
    {
      const f = fA;
      const tf = Date.now();
      await setFrame(f);
      let s = initialState(p, calib, f);
      const constDims = p.shotConsts.map((d) => ({ ...d, key: `c:${d.key}` }));
      const constGrid = constDims.length ? [0, 0.5, 1].map((u) => Object.fromEntries(constDims.map((d) => [d.key, d.hi + (d.lo - d.hi) * u]))) : [{}];
      const quick = !!process.env.FIT_QUICK;
      const align = refEyes(lr, f);
      log(`[${shot.id}] ${p.id} 锚点 ${f} 参考帧虹膜 ${align ? JSON.stringify(align.map((e) => e.map((v) => +v.toFixed(1)))) : '未检测到'}`);
      // 有虹膜时：变换由眼睛对齐解析求出，只搜索姿态；否则连同变换一起搜索
      const poseDims = align ? [...HEAD_DIMS] : [...TF_DIMS, ...HEAD_DIMS];
      let coarse = null as Awaited<ReturnType<typeof optimize>> | null;
      let gridEvals = 0;
      for (const cg of constGrid)
        for (const hx of quick ? [0] : [-15, 0, 15])
          for (const hy of quick ? [0] : [-10, 0, 10])
            for (const hz of quick || !align ? [0] : [-10, 0, 10]) {
              const r = await optimize(lr, f, { ...s, ...cg, hx, hy, hz }, poseDims, { scale: 0.25, r: 2, gens: 3, lambda: 10, sigmaMul: 1.5, prev: null, mode: 'all', align });
              gridEvals += r.evals;
              if (!coarse || r.cost < coarse.cost) coarse = r;
            }
      log(`[${shot.id}] ${p.id} 网格最优 ${JSON.stringify(Object.fromEntries(Object.entries(coarse!.state).map(([k, v]) => [k, +v.toFixed(2)])))} cost ${coarse!.cost.toFixed(4)}`);
      coarse = await optimize(lr, f, coarse!.state, [...poseDims, ...constDims], { scale: 0.25, r: 2, gens: 25, lambda: 14, sigmaMul: 1.5, prev: null, mode: 'all', align });
      const mid = align
        ? await optimize(lr, f, coarse.state, [...HEAD_DIMS, ...FACE_DIMS, ...constDims], { scale: 0.5, r: 3, gens: 25, lambda: 16, sigmaMul: 1.2, prev: null, mode: 'all', align })
        : coarse;
      const fine = await optimize(lr, f, mid.state, [...dimsFor(lr, f), ...constDims], { scale: 0.5, r: 3, gens: align ? 15 : 35, lambda: 16, sigmaMul: align ? 0.7 : 1.5, prev: null, mode: 'all' });
      if (mid !== coarse) coarse.evals += mid.evals;
      s = fine.state;
      log(`[${shot.id}] ${p.id} 锚点 ${f} 粗 ${coarse.loss.toFixed(4)} → 细 ${fine.loss.toFixed(4)}（ssim ${fine.ssim.toFixed(3)}，眼 ${fine.eye.toFixed(1)}px）`);
      let colEvals = 0;
      if (p.colorFilter) {
        const cs: State = { ...s, brightness: 0, contrast: 1, saturation: 1, mr: 1, mg: 1, mb: 1 };
        const col = await optimize(lr, f, cs, COLOR_DIMS, { scale: 0.5, r: 3, gens: 25, lambda: 12, sigmaMul: 2, prev: null, color: true, mode: 'all' });
        const c = col.state;
        lr.color = { brightness: c.brightness!, contrast: c.contrast!, saturation: c.saturation!, multiply: [c.mr!, c.mg!, c.mb!] };
        colEvals = col.evals;
        log(`[${shot.id}] ${p.id} 颜色 ${JSON.stringify(lr.color)} → ${col.loss.toFixed(4)}`);
      }
      await commit(lr, f, s, 'all');
      states.set(f, s);
      lr.lastArea = fine.area;
      lr.log.push({ frame: f, loss: fine.loss, ssim: fine.ssim, area: fine.area, eye: fine.eye, evals: gridEvals + coarse.evals + fine.evals + colEvals, ms: Date.now() - tf });
      fitted++;
      logFrame(lr, f, '锚点');
      if (opts.previewEvery) await preview(f, p.id);
    }

    // (b) 锚点之前：时间倒序逐帧，候选写满 [in, f]（更早的帧随后会被覆盖），之后把 f 的值顺延到下一拟合帧之前
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
          const r = await fitFrame(lr, f, init, ns, 'win', 0.8);
          s = r.state;
          lr.log.push({ frame: f, loss: r.loss, ssim: r.ssim, area: r.area, eye: r.eye, evals: r.evals, ms: Date.now() - tf });
          fitted++;
          logFrame(lr, f, '倒序');
        }
        await commit(lr, f, s, 'win');
        // 更早的帧暂时等于 f 的值（随后被覆盖）；(f, next−1) 用 f 的值；next−1 保留 next 写入的物理输入
        await page.evaluate(([lid, src, a, b]) => (window as any).__fit.copyFrames(lid, src, a, b), [p.id, f, p.in, Math.max(p.in, f - BACK_WINDOW)] as const);
        await page.evaluate(([lid, src, a, b]) => (window as any).__fit.copyFrames(lid, src, a, b), [p.id, f, f + 1, next - 1] as const);
        states.set(f, s);
        nextOlder = next;
        next = f;
      }
    }

    // (c) 锚点用真实的物理历史重拟合（短程），然后时间正序
    let lastFit = fA;
    let prevFit: number | null = null;
    lr.lastFit = null;
    {
      const tf = Date.now();
      await setFrame(fA);
      const before = news.filter((g) => g < fA);
      if (before.length) {
        const r = await fitFrame(lr, fA, states.get(fA)!, states.get(fA)!, 'frame', 0.7);
        states.set(fA, r.state);
        await commit(lr, fA, r.state, 'frame');
        lr.log.push({ frame: fA, loss: r.loss, ssim: r.ssim, area: r.area, eye: r.eye, evals: r.evals, ms: Date.now() - tf });
        logFrame(lr, fA, '锚点重拟合');
        prevFit = before[before.length - 1]!;
      }
    }
    lr.lastFit = fA;
    for (const f of news.filter((g) => g > fA)) {
      const tf = Date.now();
      await page.evaluate(([lid, src, a, b]) => (window as any).__fit.copyFrames(lid, src, a, b), [p.id, lastFit, lastFit + 1, f + 1] as const);
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
      const r = await fitFrame(lr, f, init, cur, 'frame');
      states.set(f, r.state);
      lr.lastArea = 0.5 * lr.lastArea + 0.5 * r.area;
      await commit(lr, f, r.state, 'frame');
      lr.log.push({ frame: f, loss: r.loss, ssim: r.ssim, area: r.area, eye: r.eye, evals: r.evals, ms: Date.now() - tf });
      for (const sw of p.switches) if (r.state[`sw:${sw.param}`] !== undefined) lr.switchW.get(sw.param)!.set(f, r.state[`sw:${sw.param}`]!);
      fitted++;
      logFrame(lr, f, '正序');
      if (opts.previewEvery && fitted % opts.previewEvery === 0) await preview(f, p.id);
      prevFit = lastFit;
      lr.lastFit = lastFit = f;
    }
    await page.evaluate(([lid, src, a, b]) => (window as any).__fit.copyFrames(lid, src, a, b), [p.id, lastFit, lastFit + 1, p.out] as const);
    lr.state = states.get(lastFit)!;
    done.add(p.id);
  }
  await page.evaluate(() => (window as any).__fit.setHidden([]));
  extraExcl = [];

  // ---- 开关切换：逐帧权重 → 斜坡（起点精确到帧） ----
  const switchResults: Record<string, unknown>[] = [];
  const dumps = new Map<string, ReturnType<typeof dumpType>>();
  for (const p of plans) dumps.set(p.id, (await page.evaluate((lid) => (window as any).__fit.dump(lid), p.id)) as ReturnType<typeof dumpType>);
  for (const lr of L.values()) {
    if (reused.has(lr.plan.id)) continue;
    for (const sw of lr.plan.switches) {
      const ws = lr.switchW.get(sw.param)!;
      const ramp = fitSwitchRamp(ws, sw);
      switchResults.push({ layer: lr.plan.id, ...sw, ...ramp, samples: Object.fromEntries(ws) });
      const vals = dumps.get(lr.plan.id)!.params[sw.param]!;
      for (let f = lr.plan.in; f < lr.plan.out; f++) vals[f - lr.plan.in] = rampValue(sw.from, sw.to, ramp.start, ramp.fade, f);
      log(`[${shot.id}] ${lr.plan.id} 开关 ${sw.param}: shots 估计 ${sw.est}（fade ${sw.fade}）→ 拟合 起点 ${ramp.start}、fade ${ramp.fade}（rms ${ramp.rms?.toFixed(3)}）`);
    }
  }

  // ---- 5. 导出 ----
  const layers: unknown[] = [];
  for (const p of [...plans].sort((a, b) => a.z - b.z)) {
    const lr = L.get(p.id)!;
    const isNew = new Map<number, boolean>();
    for (let f = p.in; f < p.out; f++) isNew.set(f, lr.newFrames.has(f));
    const f0 = firstFit.get(p.id) ?? p.in;
    for (let f = p.in + 1; f <= f0 && f < p.out; f++) isNew.set(f, false); // f0 之前都等于 f0 的值
    if (f0 < p.out) isNew.set(f0, true);
    const holds = holdSegments(isNew, p.in, p.out).filter((h) => h.start >= f0 || h.end <= f0);
    layers.push(reused.get(p.id) ?? exportLayer(scene, p, dumps.get(p.id)!, lr, holds, shot));
  }
  const elapsed = (Date.now() - t0) / 1000;
  const timing = await page.evaluate(() => (window as any).__fit.timing());
  const result = {
    shot: shot.id,
    range: [shot.start, shot.end],
    elapsedSec: elapsed,
    evals: totalEvals,
    fittedFrames: fitted,
    timing,
    layers,
    switches: switchResults,
    perFrame: Object.fromEntries([...L.values()].map((lr) => [lr.plan.id, lr.log])),
    cadence: cadenceLog,
  };
  writeFileSync(join(outDir, 'result.json'), JSON.stringify(result));
  log(`[${shot.id}] 完成：拟合 ${fitted} 帧，${totalEvals} 次渲染，${elapsed.toFixed(0)}s；页面计时 ${JSON.stringify(timing)}`);
  ref.close();
  return result;
}

function dumpType() {
  return null as unknown as {
    tf: { x: number[]; y: number[]; scale: number[]; rotation: number[] };
    opacity: number[] | null;
    blur: number[] | null;
    color: Record<string, unknown> | null;
    params: Record<string, number[]>;
  };
}

function rampValue(from: number, to: number, start: number, fade: number, f: number): number {
  if (f < start) return from;
  if (fade <= 0 || f >= start + fade) return to;
  const u = (f - start) / fade;
  return from + (to - from) * (0.5 - 0.5 * Math.cos(Math.PI * u));
}

/** 用逐帧拟合出的权重样本，网格搜索 (起点, fade) 使平方误差最小；样本不足时回退到 shots 估计 */
function fitSwitchRamp(ws: Map<number, number>, sw: LayerPlan['switches'][number]) {
  if (ws.size < 3) return { start: sw.est, fade: sw.fade, rms: null as number | null };
  let best = { start: sw.est, fade: sw.fade, rms: Infinity as number | null };
  for (let s = sw.window[0]; s < sw.window[1]; s++)
    for (let fd = 0; fd <= Math.max(2 * sw.fade, 8); fd++) {
      let e = 0;
      for (const [f, w] of ws) e += (rampValue(sw.from, sw.to, s, fd, f) - w) ** 2;
      const rms = Math.sqrt(e / ws.size);
      if (rms < best.rms!) best = { start: s, fade: fd, rms };
    }
  return best;
}

function exportLayer(
  scene: ReturnType<typeof makeScene>, p: LayerPlan, d: ReturnType<typeof dumpType>, lr: LayerRuntime,
  holds: { start: number; end: number; every: number }[], shot: Shot,
) {
  const base = scene.layers.find((x) => x.id === p.id)! as any;
  const r4 = (v: number) => Math.round(v * 1e6) / 1e6;
  const bk = (vals: number[]) => ({ baked: { start: p.in, values: vals.map(r4) } });
  const prov = (what: string) => ({ source: 'fit' as const, tool: '@charaforge/fit', detail: { shot: shot.id, what } });
  const filters: unknown[] = [];
  const c = d.color as any;
  if (c) filters.push({ type: 'color', brightness: r4(c.brightness), contrast: r4(c.contrast), saturation: r4(c.saturation), multiply: (c.multiply as number[]).map(r4) });
  if (d.blur) filters.push({ type: 'blur', sigma: bk(d.blur) });
  return {
    id: p.id,
    name: `${shot.id} ${p.inst.role}`,
    kind: 'model',
    model: 'dsc',
    in: p.in,
    out: p.out,
    transform: {
      x: bk(d.tf.x), y: bk(d.tf.y),
      scale: { baked: { start: p.in, values: d.tf.scale.map((v) => Math.round(v * 1e6) / 1e6) } },
      rotation: bk(d.tf.rotation), anchorX: base.transform.anchorX, anchorY: base.transform.anchorY,
    },
    ...(d.opacity ? { opacity: bk(d.opacity) } : {}),
    filters,
    ...(holds.length ? { frameHold: holds } : {}),
    physics: { warmup: 60, stabilize: true },
    params: base.params.map((t: any) => {
      if (t.curve.keys) return { id: t.id, curve: t.curve, semantics: { tags: ['switch'] }, provenance: { source: 'fit', tool: '@charaforge/fit', detail: { shot: shot.id, from: 'shots.variant' } } };
      const what = p.switches.some((s) => s.param === t.id) ? 'switch-ramp' : p.shotConsts.some((s) => s.key === t.id) ? 'shot-constant' : PHYS_INPUT_IDS.includes(t.id) ? 'per-frame (physics input, value at f drives frame f+1)' : 'per-frame';
      return { id: t.id, curve: bk(d.params[t.id]!), provenance: prov(what) };
    }),
    provenance: { source: 'fit', tool: '@charaforge/fit', detail: { shot: shot.id, instance: p.inst.id, fittedFrames: lr.log.length } },
  };
}
