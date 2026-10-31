/*
 * 第三轮：基于对应关系的分析-合成 + 整段轨迹联合优化（Gauss-Newton / LM）。
 *
 * 每个图层（按 order 逐个，其余图层用当前值参与渲染）：
 *   状态：每个「新画面」帧一个节点 θ_n（变换 x,y,log 尺度,旋转 + 头/身角度 + 表情 + 头发组），保持帧沿用节点值；
 *   迭代：
 *     1. 轨迹 → 逐帧参数轨（物理输入、postPhysics 姿态输出、头发叠加……），在 harness 页面（runtime）里渲染各节点帧；
 *     2. 光流（tools/fit/flow.py，OpenCV DIS，前后向一致性，只取我们的角色遮罩内、排除区外）：我们的渲染 → 参考帧；
 *     3. 几何雅可比（GeomPool：worker_threads 里 runtime 的 LayerEvaluator + Cubism Core，物理按真实历史前向推进，
 *        有限差分扰动物理之后的参数）：采样顶点的画布坐标 ∂c/∂θ；变换部分解析求导；
 *     4. 采样顶点投到屏幕 p，光流给出目标 p + F(p)；Cauchy 稳健权重；每帧归一化；
 *        加二阶差分平滑（每维以 tau 为尺度）和 LM 阻尼，整段节点一起解带状正规方程；
 *   由粗到细：先只解变换 + 头/身，再加表情、头发组。
 *   镜头常量（体型 Clothes2、前倾）：几档候选各跑两轮粗迭代，用像素损失挑；外观常量（blur、color）最后用像素损失拟合。
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Shot } from '../../../packages/scene/src/index.ts';
import { CMAES } from './cmaes.ts';
import { exportLayer, initialState, makeScene, rampValue, type LayerPlan, type ModelCalib } from './fitshot.ts';
import type { GeomPool } from './geom/gpool.ts';
import type { DimSpec, JacFrame } from './geom/worker.ts';
import { BODY_DIMS, BREATH_DIM, FACE_DIMS, HAIR_DIMS, HAIR_GROUPS, HEAD_DIMS, PHYS_INPUT_IDS, TF_DIMS, hairParams, toParams, type Dim } from './model.ts';
import { FIT_DIR, FIT_TAG, MODEL_URI, REPO } from './paths.ts';
import type { Pool } from './pool.ts';
import { RefFrames } from './refvideo.ts';
import { detectNewFrames, exclusionsAt, holdSegments, trackAt } from './shotsutil.ts';

type State = Record<string, number>;
const W = 1280, H = 720;
const PARAM_CONSTS = ['LeanInput', 'Clothes2'];
const PHYS = new Set(PHYS_INPUT_IDS);

/** 维度 → 物理之后直接取用的参数（雅可比扰动用；与 model.ts 的 toParams 一致） */
function dimSpecs(dims: Dim[]): DimSpec[] {
  const k = 28 / 30;
  const m: Record<string, [string, number][]> = {
    hx: [['ParamAngleX', k]], hy: [['ParamAngleY', 1.5 * k]], hz: [['ParamAngleZ', k]],
    bx: [['ParamBodyAngleX', 1]], by: [['ParamBodyAngleY', 1]], bz: [['ParamBodyAngleZ', -0.7]],
    eyeL: [['ParamEyeLOpen', 1]], eyeR: [['ParamEyeROpen', 1]], es: [['EyeSmile', 1]],
    ebx: [['ParamEyeBallX', 1]], eby: [['ParamEyeBallY', 1]], brow: [['ParamBrowLY', 1], ['ParamBrowRY', 1]],
    mo: [['ParamMouthOpenY', 1]], mf: [['ParamMouthForm', 1]],
    br: [['ParamBreath', 1]],
  };
  for (const g of HAIR_GROUPS) m[g.key] = g.params.map((p) => [p, g.gain]);
  return dims.map((d) => ({ key: d.key, params: m[d.key]!, delta: ['hx', 'hy', 'hz', 'bx', 'by', 'bz'].includes(d.key) ? 1 : 0.05 }));
}

export interface TrajOptions {
  log?: (s: string) => void;
  /** 粗阶段 / 全维度阶段的迭代次数 */
  itersA?: number;
  itersB?: number;
  /** 镜头常量网格（缺省 Clothes2 × LeanInput 各 3 档） */
  constGrid?: boolean;
  order?: string[];
  /** 样本顶点数（每帧） */
  samples?: number;
}

export async function fitTrajectory(pool: Pool, gpool: GeomPool, shot: Shot, plans: LayerPlan[], calib: ModelCalib, opts: TrajOptions = {}) {
  const log = opts.log ?? ((s: string) => console.log(s));
  const t0 = Date.now();
  const outDir = join(FIT_DIR, shot.id + FIT_TAG);
  const work = join(outDir, 'work');
  mkdirSync(work, { recursive: true });
  const ref = new RefFrames(shot.start, shot.end);
  const timing: Record<string, number> = { render: 0, flow: 0, jac: 0, solve: 0, pixel: 0 };
  const tick = (k: string, t: number) => (timing[k] = (timing[k] ?? 0) + (Date.now() - t) / 1000);

  // ---- 节点（新画面帧）与初始状态 ----
  const nodesOf = new Map<string, number[]>();
  const newSets = new Map<string, Map<number, boolean>>();
  for (const p of plans) {
    const { isNew } = detectNewFrames(ref, shot, p.inst, p.in, p.out);
    newSets.set(p.id, isNew);
    nodesOf.set(p.id, [...isNew.entries()].filter(([, v]) => v).map(([f]) => f).sort((a, b) => a - b));
  }
  const mdx = calib.eyes[1][0] - calib.eyes[0][0];
  const theta = new Map<string, State[]>();
  for (const p of plans) {
    const base = initialState(p, calib, p.in);
    theta.set(p.id, nodesOf.get(p.id)!.map((f) => {
      // 变换初值：shots 的 eyes（锚点 = 模型参考姿态下的双眼中点，所以眼睛中点 ≈ 图层位置）
      const e = trackAt(p.inst, 'eyes', f) as number[][] | null;
      const s = { ...base };
      if (e) {
        s.x = (e[0]![0]! + e[1]![0]!) / 2; s.y = (e[0]![1]! + e[1]![1]!) / 2;
        s.ls = Math.log(Math.hypot(e[1]![0]! - e[0]![0]!, e[1]![1]! - e[0]![1]!) / mdx);
        s.rot = (Math.atan2(e[1]![1]! - e[0]![1]!, e[1]![0]! - e[0]![0]!) * 180) / Math.PI;
      }
      s.blur = (trackAt(p.inst, 'blurSigma', f) as number | null) ?? 0;
      s.br = 0.5;
      // 调试：FIT_INIT_JSON 覆盖初值（如诊断实验里给定前倾/身体俯仰的起点）
      if (process.env.FIT_INIT_JSON) Object.assign(s, JSON.parse(process.env.FIT_INIT_JSON));
      return s;
    }));
  }
  const scene = makeScene(shot, plans, calib, new Map(plans.map((p) => [p.id, theta.get(p.id)![0]!])));
  const layerOf = (id: string) => scene.layers.find((l) => l.id === id)! as any;
  const colors = new Map<string, { brightness: number; contrast: number; saturation: number; multiply: [number, number, number] }>();

  /** 帧 g 的全部参数（与 fitshot 的 paramsAt 一致：物理输入、姿态输出、开关斜坡、镜头常量、头发叠加） */
  const paramsAt = (p: LayerPlan, s: State, g: number) => {
    const out: Record<string, number> = { ...toParams(s, g) };
    for (const sw of p.switches) out[sw.param] = rampValue(sw.from, sw.to, sw.est, sw.fade, g);
    for (const k of PARAM_CONSTS) if (p.consts.includes(`c:${k}`)) out[k] = s[`c:${k}`] ?? 0;
    if (p.hairBake) for (const [k, v] of Object.entries(hairParams(s))) out[`post:${k}`] = v;
    return out;
  };

  /** 节点状态 → 图层的逐帧数组（保持帧用所在节点的值；物理输入取下一帧所在节点，使帧 f 的画面由 θ(f) 决定） */
  const apply = (p: LayerPlan) => {
    const L = layerOf(p.id);
    const nodes = nodesOf.get(p.id)!, th = theta.get(p.id)!;
    const nodeIdx = (g: number) => { let n = 0; while (n + 1 < nodes.length && nodes[n + 1]! <= g) n++; return n; };
    const n = p.out - p.in;
    const tracks = new Map<string, number[]>();
    for (const t of L.params) if (t.curve.baked) tracks.set(t.stage === 'postPhysics' ? `post:${t.id}` : t.id, t.curve.baked.values);
    const blurF = L.filters.find((f: any) => f.type === 'blur');
    for (let i = 0; i < n; i++) {
      const g = p.in + i;
      const s = th[nodeIdx(g)]!, sn = th[nodeIdx(Math.min(g + 1, p.out - 1))]!;
      const pr = paramsAt(p, s, g), pn = paramsAt(p, sn, g);
      for (const [k, arr] of tracks) {
        const v = PHYS.has(k) ? pn[k] : pr[k];
        if (v !== undefined) arr[i] = v;
      }
      L.transform.x.baked.values[i] = s.x; L.transform.y.baked.values[i] = s.y;
      L.transform.scale.baked.values[i] = Math.exp(s.ls!); L.transform.rotation.baked.values[i] = s.rot;
      if (L.transform.scaleX) L.transform.scaleX.baked.values[i] = p.flipRange && g >= p.flipRange[0] && g < p.flipRange[1] ? Math.exp(s.lsx ?? 0) : 1;
      if (blurF) blurF.sigma.baked.values[i] = p.blurRanges.some(([a, b]) => g >= a && g < b) ? s.blur ?? 0 : s['c:blur'] ?? 0;
    }
    const rf = L.filters.find((f: any) => f.type === 'resample');
    if (rf) rf.factor = th[0]!['c:resample'] ?? 1;
    const col = colors.get(p.id);
    const cf = L.filters.find((f: any) => f.type === 'color');
    if (col && cf) Object.assign(cf, col);
  };

  // ---- 渲染（runtime / harness，多页面分摊帧）----
  const renderFrames = async (frames: number[], hidden: string[] = []) => {
    const t = Date.now();
    await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [scene.layers, MODEL_URI, calib.irisOffset] as const);
    await pool.all((h) => (window as any).__fit.setHidden(h), hidden);
    const n = pool.size;
    await Promise.all(pool.pages.map(async (pg, k) => {
      for (const f of frames.filter((_, i) => i % n === k)) {
        const img = (await pg.evaluate((f) => (window as any).__fit.render(f, 1), f)) as { data: string };
        writeFileSync(join(work, `ours_${f}.rgba`), Buffer.from(img.data, 'base64'));
      }
    }));
    tick('render', t);
  };

  const flows = (frames: number[]) => {
    const t = Date.now();
    const man = {
      out: work,
      frames: frames.map((f) => ({ frame: f, ours: join(work, `ours_${f}.rgba`), ref: ref.file, refIndex: f - shot.start, excl: exclusionsAt(shot, f) })),
    };
    writeFileSync(join(work, 'manifest.json'), JSON.stringify(man));
    const r = spawnSync(join(REPO, 'tools/.venv/bin/python'), ['-I', join(REPO, 'tools/fit/flow.py'), join(work, 'manifest.json')], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error(`flow.py 失败：${r.stderr}`);
    const out = new Map<number, Float32Array>();
    for (const f of frames) {
      const b = readFileSync(join(work, `flow_${f}.f32`));
      out.set(f, new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4));
    }
    tick('flow', t);
    return out;
  };

  /** 像素损失（页面里的拟合损失，均值；用于常量挑选与外观拟合） */
  const pixelLoss = async (frames: number[], lid: string, cands: Record<string, unknown>[]) => {
    const t = Date.now();
    const acc = cands.map(() => 0);
    for (const f of frames) {
      await pool.all(([b64, ex]) => { (window as any).__fit.setRef(b64); (window as any).__fit.setExcl(ex); }, [ref.frame(f).toString('base64'), exclusionsAt(shot, f)] as const);
      const rs = await pool.evalBatch<{ loss: number; ssim: number }>(f, cands.map((c) => ({ [lid]: c })), { scale: 0.5, r: 3, wcol: 1, weye: 0, wseg: 1, wgrad: 0.5, eyesOf: [] });
      rs.forEach((r, i) => (acc[i] += r.loss / frames.length));
    }
    tick('pixel', t);
    return acc;
  };

  // ---- 一次 GN 迭代 ----
  const gnStep = async (p: LayerPlan, dims: Dim[], lamSmooth: number, cauchy: number, hidden: string[], sample: Int32Array) => {
    apply(p);
    const nodes = nodesOf.get(p.id)!, th = theta.get(p.id)!;
    const L = layerOf(p.id);
    await renderFrames(nodes, hidden);
    const fl = flows(nodes);
    let t = Date.now();
    const geoDims = dims.filter((d) => !TF_DIMS.includes(d));
    const jac = await gpool.jacobians(L, nodes, dimSpecs(geoDims), sample);
    tick('jac', t);
    t = Date.now();
    const D = dims.length, N = nodes.length;
    const n = N * D, band = 2 * D;
    const Hm = new Float64Array(n * n), g = new Float64Array(n);
    const ax = L.transform.anchorX, ay = L.transform.anchorY;
    const stats: number[] = [];
    let used = 0;
    const allRows: { ni: number; J: Float64Array }[] = [];
    nodes.forEach((f, ni) => {
      const s = th[ni]!, j = jac.get(f)!, F = fl.get(f)!;
      const S = Math.exp(s.ls!), rr = (s.rot! * Math.PI) / 180, cs = Math.cos(rr), sn = Math.sin(rr);
      const K = j.opa.length, Dg = geoDims.length;
      const rows: { J: Float64Array; e: [number, number] }[] = [];
      for (let k = 0; k < K; k++) {
        if (j.opa[k]! < 0.5) continue;
        const SX = Math.exp(s.lsx ?? 0) * (p.flipRange && f >= p.flipRange[0] && f < p.flipRange[1] ? 1 : 0) || 1;
        const dx = (j.pos[2 * k]! - ax) * SX, dy = j.pos[2 * k + 1]! - ay;
        const px = s.x! + S * (cs * dx - sn * dy), py = s.y! + S * (sn * dx + cs * dy);
        const u = px / 2 - 0.25, v = py / 2 - 0.25; // 光流采样在偶数像素（2i, 2j）
        const x0 = Math.floor(u), y0 = Math.floor(v);
        if (x0 < 0 || y0 < 0 || x0 >= 639 || y0 >= 359) continue;
        const fx = u - x0, fy = v - y0;
        let vx = 0, vy = 0, vv = 1;
        for (const [ox, oy, wgt] of [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]] as const) {
          const o = ((y0 + oy) * 640 + x0 + ox) * 3;
          vx += wgt * F[o]!; vy += wgt * F[o + 1]!; vv = Math.min(vv, F[o + 2]!);
        }
        if (vv < 0.5) continue;
        const Jr = new Float64Array(2 * D);
        dims.forEach((d, c) => {
          let jx = 0, jy = 0;
          if (d.key === 'x') jx = 1;
          else if (d.key === 'y') jy = 1;
          else if (d.key === 'ls') { jx = px - s.x!; jy = py - s.y!; }
          else if (d.key === 'rot') { jx = (-(py - s.y!) * Math.PI) / 180; jy = ((px - s.x!) * Math.PI) / 180; }
          else if (d.key === 'lsx') { jx = S * cs * dx; jy = S * sn * dx; }
          else {
            const gi = geoDims.indexOf(d);
            const cx = j.J[2 * k * Dg + gi]!, cy = j.J[(2 * k + 1) * Dg + gi]!;
            jx = S * (cs * cx * SX - sn * cy); jy = S * (sn * cx * SX + cs * cy);
          }
          Jr[c] = jx * d.sd; Jr[D + c] = jy * d.sd;
        });
        rows.push({ J: Jr, e: [vx, vy] });
        if ((k & 7) === 0) allRows.push({ ni, J: Jr });
        stats.push(Math.hypot(vx, vy));
      }
      if (!rows.length) return;
      used++;
      const norm = 600 / rows.length; // 每帧归一化到约 600 个点的权重
      const o = ni * D;
      for (const r of rows) {
        const m = Math.hypot(r.e[0], r.e[1]);
        const w = (norm / (1 + (m / cauchy) ** 2));
        for (let a = 0; a < D; a++) {
          const ja = r.J[a]!, jb = r.J[D + a]!;
          g[o + a] += w * (ja * r.e[0] + jb * r.e[1]);
          for (let b = 0; b <= a; b++) Hm[(o + a) * n + o + b] += w * (ja * r.J[b]! + jb * r.J[D + b]!);
        }
      }
    });
    // 对称补全
    for (let i = 0; i < n; i++) for (let k = Math.max(0, i - band); k < i; k++) Hm[k * n + i] = Hm[i * n + k];
    // 二阶差分平滑（以 tau 为尺度）+ 阻尼
    dims.forEach((d, c) => {
      const sc = d.sd / d.tau;
      for (let m = 1; m + 1 < N; m++) {
        const ids = [(m - 1) * D + c, m * D + c, (m + 1) * D + c];
        const a = [1, -2, 1];
        const r0 = (th[m - 1]![d.key]! - 2 * th[m]![d.key]! + th[m + 1]![d.key]!) / d.tau;
        for (let i1 = 0; i1 < 3; i1++) {
          g[ids[i1]!] -= lamSmooth * sc * a[i1]! * r0;
          for (let i2 = 0; i2 < 3; i2++) Hm[ids[i1]! * n + ids[i2]!] += lamSmooth * sc * sc * a[i1]! * a[i2]!;
        }
      }
    });
    // LM（Marquardt 缩放）：对角按自身放大，病态方向（如身体俯仰与平移的耦合）不会被统一阻尼压住
    let tr = 0;
    for (let i = 0; i < n; i++) tr += Hm[i * n + i]!;
    for (let i = 0; i < n; i++) Hm[i * n + i] = Hm[i * n + i]! * 1.05 + 1e-6 * (tr / n) + 1e-4;
    const du = bandSolve(Hm, g, n, band);
    // 信赖域：预测位移的中位数不超过 max(2px, 0.7 × 当前光流残差中位数)，防止线性化失效时越走越远
    stats.sort((a, b) => a - b);
    const med = stats.length ? stats[stats.length >> 1]! : 0;
    const pred = allRows.map((r) => {
      let dx = 0, dy = 0;
      for (let c = 0; c < D; c++) { dx += r.J[c]! * du[r.ni * D + c]!; dy += r.J[D + c]! * du[r.ni * D + c]!; }
      return Math.hypot(dx, dy);
    }).sort((a, b) => a - b);
    const pm = pred.length ? pred[pred.length >> 1]! : 0;
    const lim = Math.max(2, 0.7 * med);
    const scale = pm > lim ? lim / pm : 1;
    for (let i = 0; i < n; i++) du[i] *= scale;
    nodes.forEach((_, ni) => {
      dims.forEach((d, c) => {
        const s = th[ni]!;
        s[d.key] = Math.min(d.hi, Math.max(d.lo, s[d.key]! + d.sd * du[ni * D + c]!));
      });
    });
    tick('solve', t);
    return { median: stats.length ? stats[stats.length >> 1]! : NaN, p90: stats.length ? stats[Math.floor(stats.length * 0.9)]! : NaN, points: stats.length, framesUsed: used };
  };

  // ---- 逐图层 ----
  const order = (opts.order ?? [...plans].sort((a, b) => b.z - a.z).map((p) => p.id)).map((id) => plans.find((p) => p.id === id)!);
  const done = new Set<string>();
  const report: Record<string, unknown> = {};
  for (const p of order) {
    const hidden = order.filter((o) => o !== p && !done.has(o.id)).map((o) => o.id);
    const nodes = nodesOf.get(p.id)!;
    const sample = await makeSample(gpool, layerOf(p.id), opts.samples ?? 4000);
    const flipDims: Dim[] = p.flipRange ? [{ key: 'lsx', lo: Math.log(0.02), hi: 0, sd: 0.1, tau: 0.5 }] : [];
    const dimsA = [...TF_DIMS, ...flipDims, ...HEAD_DIMS, ...BODY_DIMS];
    const dimsB = [...dimsA, BREATH_DIM, ...FACE_DIMS, ...(p.hairBake ? HAIR_DIMS : [])];
    const iterLog: unknown[] = [];
    let lastMedian = 10;
    const run = async (dims: Dim[], iters: number, lam: number, tag: string) => {
      for (let it = 0; it < iters; it++) {
        const cauchy = Math.max(3, 1.5 * lastMedian);
        const r = await gnStep(p, dims, lam, cauchy, hidden, sample);
        iterLog.push({ tag, it, ...r });
        lastMedian = r.median;
        log(`[${shot.id}] ${p.id} ${tag} #${it}: 光流残差中位 ${r.median.toFixed(2)}px、p90 ${r.p90.toFixed(2)}px（${r.points} 点，${r.framesUsed}/${nodes.length} 帧）`);
      }
    };
    // (1) 粗：变换 + 头/身（缺省常量）先收敛；再从收敛点出发比较镜头常量候选（各跑两轮），用像素损失挑
    await run(dimsA, 4, 20, '粗·预收敛');
    // 分两步：先体型 Clothes2（差别大，两轮就分得开），再前倾 LeanInput（与身体俯仰/平移简并，每个候选要收敛几轮才好比）。
    // 用光流残差（中位 + 0.3×p90）比较：它是几何量，不受外观常量（blur/color 还没拟合）影响。
    const score = (r: { median: number; p90: number }) => r.median + 0.3 * r.p90;
    const tryConsts = async (cands: State[], iters: number, tag: string) => {
      const init = theta.get(p.id)!.map((s) => ({ ...s }));
      const initMedian = lastMedian;
      let best: { sc: number; th: State[]; c: State; med: number } | null = null;
      for (const c of cands) {
        theta.set(p.id, init.map((s) => ({ ...s, ...c, ...(c['c:LeanInput'] !== undefined ? { by: (s.by ?? 0) + 0.77 * (c['c:LeanInput']! - (s['c:LeanInput'] ?? 0)) } : {}) })));
        lastMedian = initMedian;
        let last = { median: Infinity, p90: Infinity };
        for (let it = 0; it < iters; it++) {
          const r = await gnStep(p, dimsA, 20, Math.max(3, 1.5 * lastMedian), hidden, sample);
          lastMedian = r.median;
          last = r;
        }
        const sc = score(last);
        log(`[${shot.id}] ${p.id} ${tag} 候选 ${JSON.stringify(c)}：${iters} 轮后残差中位 ${last.median.toFixed(2)}px、p90 ${last.p90.toFixed(2)}px → 分数 ${sc.toFixed(2)}`);
        if (!best || sc < best.sc) best = { sc, th: theta.get(p.id)!.map((s) => ({ ...s })), c, med: lastMedian };
      }
      theta.set(p.id, best!.th);
      lastMedian = best!.med;
      return best!.c;
    };
    const chosen: State = {};
    if (opts.constGrid !== false && p.consts.includes('c:Clothes2')) Object.assign(chosen, await tryConsts([0, 0.4, 1].map((v) => ({ 'c:Clothes2': v })), 2, '体型'));
    if (opts.constGrid !== false && p.consts.includes('c:LeanInput')) Object.assign(chosen, await tryConsts([0, -15, -30].map((v) => ({ 'c:LeanInput': v })), 5, '前倾'));
    const best = { c: chosen };
    log(`[${shot.id}] ${p.id} 选定常量 ${JSON.stringify(best!.c)}`);
    await run(dimsA, opts.itersA ?? 3, 20, '粗');
    // (2) 全维度
    await run(dimsB, opts.itersB ?? 5, 10, '细');
    // (3) 外观常量：blur（清晰度）与颜色滤镜，像素损失
    apply(p);
    await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [scene.layers, MODEL_URI, calib.irisOffset] as const);
    await pool.all((h) => (window as any).__fit.setHidden(h), hidden);
    const probe = nodes.filter((_, i) => i % Math.max(1, Math.floor(nodes.length / 4)) === 0).slice(0, 4);
    const appDims: Dim[] = [
      ...(p.blurRanges.length ? [] : [{ key: 'blur', lo: 0, hi: 3, sd: 0.4, tau: 1 }, { key: 'resample', lo: 0.35, hi: 1, sd: 0.1, tau: 1 }]),
      ...(p.colorFilter ? [
        { key: 'brightness', lo: -0.3, hi: 0.3, sd: 0.03, tau: 1 }, { key: 'contrast', lo: 0.6, hi: 1.5, sd: 0.05, tau: 1 },
        { key: 'saturation', lo: 0.5, hi: 1.6, sd: 0.06, tau: 1 }, { key: 'mr', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
        { key: 'mg', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 }, { key: 'mb', lo: 0.7, hi: 1.3, sd: 0.03, tau: 1 },
      ] : []),
    ];
    if (appDims.length) {
      const x0 = appDims.map((d) => (d.key === 'blur' ? theta.get(p.id)![0]!['c:blur'] ?? 0.6 : d.key === 'resample' ? 0.7 : ['contrast', 'saturation', 'mr', 'mg', 'mb'].includes(d.key) ? 1 : 0));
      const es = new CMAES(x0, { sigma: 1, scales: appDims.map((d) => d.sd), lambda: 12, seed: 7 });
      const toVals = (x: number[]) => {
        const v: Record<string, number> = {};
        appDims.forEach((d, i) => (v[d.key] = Math.min(d.hi, Math.max(d.lo, x[i]!))));
        return v;
      };
      const cand = (v: Record<string, number>) => ({
        ...(v.blur !== undefined ? { blur: v.blur } : {}),
        ...(v.resample !== undefined ? { resample: v.resample } : {}),
        ...(p.colorFilter ? { color: { brightness: v.brightness, contrast: v.contrast, saturation: v.saturation, multiply: [v.mr, v.mg, v.mb] } } : {}),
      });
      let bestA = { c: (await pixelLoss(probe, p.id, [cand(toVals(x0))]))[0]!, v: toVals(x0) };
      for (let gen = 0; gen < 15; gen++) {
        const xs = es.ask();
        const cs = await pixelLoss(probe, p.id, xs.map((x) => cand(toVals(x))));
        es.tell(cs);
        cs.forEach((c, i) => { if (c < bestA.c) bestA = { c, v: toVals(xs[i]!) }; });
      }
      if (bestA.v.blur !== undefined) for (const s of theta.get(p.id)!) s['c:blur'] = bestA.v.blur;
      if (bestA.v.resample !== undefined) for (const s of theta.get(p.id)!) s['c:resample'] = bestA.v.resample;
      if (p.colorFilter) colors.set(p.id, { brightness: bestA.v.brightness!, contrast: bestA.v.contrast!, saturation: bestA.v.saturation!, multiply: [bestA.v.mr!, bestA.v.mg!, bestA.v.mb!] });
      log(`[${shot.id}] ${p.id} 外观常量 ${JSON.stringify(Object.fromEntries(Object.entries(bestA.v).map(([k, v]) => [k, +v.toFixed(3)])))} → 像素损失 ${bestA.c.toFixed(4)}`);
    }
    // (4) 外观更新后再做两轮全维度（光流在新外观下更准）
    await run(dimsB, 2, 10, '收尾');
    apply(p);
    done.add(p.id);
    report[p.id] = { nodes: nodes.length, consts: best!.c, iterations: iterLog };
  }

  // ---- 导出 ----
  const layers: unknown[] = [];
  for (const p of [...plans].sort((a, b) => a.z - b.z)) {
    apply(p);
    const L = layerOf(p.id);
    const params: Record<string, number[]> = {};
    for (const t of L.params) if (t.curve.baked) params[t.stage === 'postPhysics' ? `post:${t.id}` : t.id] = t.curve.baked.values;
    const blurF = L.filters.find((f: any) => f.type === 'blur');
    const d = {
      tf: { x: L.transform.x.baked.values, y: L.transform.y.baked.values, scale: L.transform.scale.baked.values, rotation: L.transform.rotation.baked.values },
      opacity: L.opacity?.baked?.values ?? null,
      blur: blurF ? blurF.sigma.baked.values : null,
      clip: L.clip ? { x: L.clip.x.baked.values, y: L.clip.y.baked.values, width: L.clip.width.baked.values, height: L.clip.height.baked.values, radius: L.clip.radius.baked.values } : null,
      color: colors.get(p.id) ?? null,
      resample: L.filters.find((f: any) => f.type === 'resample')?.factor ?? null,
      sx: L.transform.scaleX?.baked?.values ?? null,
      params,
    };
    const holds = holdSegments(newSets.get(p.id)!, p.in, p.out);
    const layer = exportLayer(scene, p, d as any, { log: nodesOf.get(p.id)! } as any, holds, shot) as any;
    layer.provenance.detail.method = 'trajectory Gauss-Newton on dense optical-flow correspondences (round 3)';
    layers.push(layer);
  }
  const elapsed = (Date.now() - t0) / 1000;
  const result = { shot: shot.id, range: [shot.start, shot.end], method: 'traj-gn', elapsedSec: elapsed, timing, layers, report };
  writeFileSync(join(outDir, 'result.json'), JSON.stringify(result));
  log(`[${shot.id}] 完成：${elapsed.toFixed(0)}s；分项 ${JSON.stringify(Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v)])))}`);
  ref.close();
  return result;
}

/** 采样顶点：所有 drawable 的顶点按步长均匀抽取，总数约 target */
async function makeSample(gpool: GeomPool, layer: any, target: number): Promise<Int32Array> {
  void layer;
  const counts = await gpool.vertexCounts();
  const total = counts.reduce((a, b) => a + b, 0);
  const step = Math.max(1, Math.round(total / target));
  const out: number[] = [];
  let k = 0;
  counts.forEach((cnt, d) => { for (let i = 0; i < cnt; i++, k++) if (k % step === 0) out.push(d, i); });
  return new Int32Array(out);
}

/** 带状对称正定方程组 A x = b（A 为 n×n 稠密存储，只用半带宽 band 内的元素），Cholesky */
export function bandSolve(A: Float64Array, b: Float64Array, n: number, band: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const j0 = Math.max(0, i - band);
    for (let j = j0; j <= i; j++) {
      let sum = A[i * n + j]!;
      const k0 = Math.max(j0, j - band);
      for (let k = k0; k < j; k++) sum -= L[i * n + k]! * L[j * n + k]!;
      if (i === j) L[i * n + i] = Math.sqrt(Math.max(sum, 1e-12));
      else L[i * n + j] = sum / L[j * n + j]!;
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i]!;
    for (let k = Math.max(0, i - band); k < i; k++) s -= L[i * n + k]! * y[k]!;
    y[i] = s / L[i * n + i]!;
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]!;
    for (let k = i + 1; k <= Math.min(n - 1, i + band); k++) s -= L[k * n + i]! * x[k]!;
    x[i] = s / L[i * n + i]!;
  }
  return x;
}

export type { JacFrame };
