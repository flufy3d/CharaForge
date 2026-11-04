/*
 * 拟合命令行。
 *
 *   node --experimental-strip-types tools/fit/src/cli.ts fit --shot s43 [--until 3700] [--preview 4]
 *   node --experimental-strip-types tools/fit/src/cli.ts assemble          # 合成 scenes/deepseek-chan/scene.json
 *
 * 详见 tools/fit/README.md。
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { assembleScene } from './assemble.ts';
import { fitShot, redetectSwitch, type ModelCalib } from './fitshot.ts';
import { RefFrames } from './refvideo.ts';
import { readFileSync as rf } from 'node:fs';
import { Pool } from './pool.ts';
import { GeomPool } from './geom/gpool.ts';
import { fitTrajectory } from './traj.ts';
import { buildLayer, fitInputs } from './traj2.ts';
import { loadVts } from './vts.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { FIT_DIR, FIT_TAG, MODEL_URI, REPO } from './paths.ts';
import { fitUnits, LEGACY_SHOTS, planAuto, planFor, planS15v3 } from './plans.ts';
import { loadShots } from './shotsutil.ts';

/** 模型参考姿态（全部默认参数、物理静止）下的双眼虹膜中心（画布像素），用于由 shots 的 eyes 求初始变换 */
async function calibrate(page: import('playwright').Page): Promise<ModelCalib> {
  const file = join(FIT_DIR, 'calib.json');
  if (existsSync(file)) return JSON.parse(readFileSync(file, 'utf8'));
  const r = await page.evaluate(async (m) => {
    const H = window.__charaforge;
    const id = await H.createInstance(m);
    const info = H.instanceInfo(id);
    for (let i = 0; i < 180; i++) H.stepInstance(id, { physics: true });
    // 画布 [1000, 2750] × [300, 2050] → 1750×1750（scale 1）
    const px = await H.renderInstance(id, { transform: { x: 0, y: 0, scale: 1, anchorX: 1000, anchorY: 300 }, width: 1750, height: 1750 });
    // 虹膜主网格（ArtMesh236 左、ArtMesh134 右）的顶点质心，画布像素
    const inst = (window as any).__cfRuntime.instance(id);
    const mdl = inst.model, c = inst.canvas;
    const mesh = ['ArtMesh236', 'ArtMesh134'].map((name) => {
      for (let d = 0; d < mdl.getDrawableCount(); d++) {
        const did = mdl.getDrawableId(d);
        if ((did.getString().s ?? String(did.getString())) !== name) continue;
        const v = mdl.getDrawableVertices(d);
        let x = 0, y = 0;
        for (let k = 0; k < v.length / 2; k++) { x += v[2 * k]; y += v[2 * k + 1]; }
        const n = v.length / 2;
        return [c.originX + (x / n) * c.pixelsPerUnit, c.originY - (y / n) * c.pixelsPerUnit];
      }
      throw new Error(name);
    });
    H.destroyInstance(id);
    return { px, canvas: info.canvas, mesh };
  }, MODEL_URI);
  const d = Buffer.from(r.px.data!, 'base64');
  const W = r.px.width;
  // 虹膜：青色（B、G 都高，R 低）。从画布中线两侧的眼部窗口出发，在半径 70px 内迭代求青色像素质心（排除发梢上的青色高光）
  const cyan = (x: number, y: number) => {
    const o = (y * W + x) * 4;
    const R = d[o]!, G = d[o + 1]!, B = d[o + 2]!;
    return d[o + 3]! > 200 && B > 170 && G > 130 && R < 130 && G > 0.65 * B;
  };
  const centers: [number, number][] = [[705, 1285], [1045, 1285]]; // 输出缓冲坐标（= 画布 − [1000, 300]）的初值
  const acc = [[0, 0, 0], [0, 0, 0]];
  for (let it = 0; it < 4; it++)
    centers.forEach(([cx, cy], k) => {
      let sx = 0, sy = 0, n = 0;
      for (let y = Math.round(cy - 70); y <= cy + 70; y++)
        for (let x = Math.round(cx - 70); x <= cx + 70; x++)
          if ((x - cx) ** 2 + (y - cy) ** 2 <= 4900 && cyan(x, y)) { sx += x; sy += y; n++; }
      if (n) centers[k] = [sx / n, sy / n];
      acc[k] = [centers[k]![0] + 1000.5, centers[k]![1] + 300.5, n]; // 像素中心
    });
  if (acc[0]![2]! < 200 || acc[1]![2]! < 200) throw new Error(`眼睛定位失败：${JSON.stringify(acc)}`);
  const calib: ModelCalib = {
    eyes: [[acc[0]![0]!, acc[0]![1]!], [acc[1]![0]!, acc[1]![1]!]],
    canvas: { width: r.canvas.width, height: r.canvas.height },
    irisOffset: [0, 1].map((k) => [acc[k]![0]! - r.mesh[k]![0]!, acc[k]![1]! - r.mesh[k]![1]!]) as [[number, number], [number, number]],
  };
  mkdirSync(FIT_DIR, { recursive: true });
  writeFileSync(file, JSON.stringify(calib, null, 2));
  // 校验图：标出检测到的眼睛中心
  const out = Buffer.from(d);
  for (const [ex, ey] of calib.eyes)
    for (let k = -12; k <= 12; k++)
      for (const [x, y] of [[Math.round(ex - 1000) + k, Math.round(ey - 300)], [Math.round(ex - 1000), Math.round(ey - 300) + k]] as const) {
        const o = (y * W + x) * 4;
        out[o] = 255; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 255;
      }
  writeFileSync(join(FIT_DIR, 'calib.png'), encodePng(W, r.px.height, out));
  return calib;
}

async function main() {
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const cmd = argv[0];
  const { values } = parseArgs({
    args: argv.slice(1),
    options: { shot: { type: 'string' }, until: { type: 'string' }, start: { type: 'string' }, geom: { type: 'string' }, itersA: { type: 'string' }, itersB: { type: 'string' }, noconst: { type: 'boolean' }, smoothC: { type: 'string' }, fixed: { type: 'string' }, refine: { type: 'string' }, workers: { type: 'string' }, effort: { type: 'string' }, hair: { type: 'boolean' }, only: { type: 'string' }, scene: { type: 'string' }, frames: { type: 'string' }, preview: { type: 'string' }, out: { type: 'string' } },
  });
  if (cmd === 'assemble') {
    // --only s08,s17：只合成覆盖这些镜头的结果（批量驱动逐镜头打分用的临时场景）
    assembleScene(values.out, values.only ? String(values.only).split(',') : undefined);
    return;
  }
  if (cmd === 'units') {
    // P3：列出全片拟合单元与通用规划（不拟合）；--out 写 JSON（含 shots 问题清单）
    const shots = loadShots();
    const rows = fitUnits(shots.shots).map((u) => {
      if (LEGACY_SHOTS.has(u.id)) return { unit: u.id, shots: u.shots, range: [u.shot.start, u.shot.end], legacy: true, layers: [], skipped: [], issues: [] };
      const a = planAuto(u.shot);
      return {
        unit: u.id, shots: u.shots, range: [u.shot.start, u.shot.end],
        layers: a.order.map((l) => { const q = a.plans.find((x) => x.id === l)!; return { id: l, in: q.in, out: q.out, z: q.z, switches: q.switches.map((w) => `${w.param}:${w.from}->${w.to}@${w.est}/${w.fade}`), clip: q.clipKeys ? (q.clipKeys[0]!.w !== undefined ? 'rect' : 'circle') : null, blur: q.blurRanges, flip: !!q.flipCurve, opacity: !!q.opacityCurve, color: q.colorFilter, fixed: q.fixed }; }),
        skipped: a.skipped, issues: [...u.issues, ...a.issues],
      };
    });
    for (const r of rows) console.log(`${r.unit}\t${r.shots.join('+')}\t[${r.range.join(',')})\t${(r as any).legacy ? 'LEGACY' : r.layers.map((l: any) => `${l.id}[${l.in},${l.out})`).join(' ')}${r.skipped.length ? `\t跳过 ${r.skipped.map((k: any) => k.id).join(',')}` : ''}`);
    if (values.out) writeFileSync(String(values.out), JSON.stringify(rows, null, 1));
    return;
  }
  if (cmd === 'switches' && values.shot) {
    // 对已有 result.json 重新做开关的画面检测（用拟合结果的虹膜位置），改写开关轨
    await redetectSwitches(values.shot);
    return;
  }
  if ((cmd === 'fit4' || cmd === 'synth2') && values.shot) {
    await runInputs(cmd, values);
    return;
  }
  if (cmd === 'fit3' && values.shot) {
    await runTraj(values);
    return;
  }
  if (cmd !== 'fit' || !values.shot) {
    console.error('用法：cli.ts fit --shot s43[,s24,...] [--until F] [--preview N] | cli.ts assemble [--out path]');
    process.exit(2);
  }
  const shots = loadShots();
  const pool = await Pool.open(values.workers ? Number(values.workers) : 4);
  try {
    console.log(`[fit] GPU: ${pool.renderer}，${pool.size} 个页面`);
    const calib = await calibrate(pool.pages[0]!);
    console.log(`[fit] 模型眼睛（画布像素）：${JSON.stringify(calib.eyes)}`);
    for (const id of values.shot.split(',')) {
      const shot = shots.shots.find((s) => s.id === id);
      if (!shot) throw new Error(`没有镜头 ${id}`);
      const { plans, options } = planFor(shot, { hairBake: !!values.hair });
      const logFile = join(FIT_DIR, id + FIT_TAG, 'fit.log');
      mkdirSync(join(FIT_DIR, id + FIT_TAG), { recursive: true });
      writeFileSync(logFile, '');
      await fitShot(pool, shot, plans, calib, {
        ...options,
        until: values.until ? Number(values.until) : undefined,
        start: values.start ? Number(values.start) : undefined,
        only: values.only ? values.only.split(',') : undefined,
        effort: values.effort ? Number(values.effort) : undefined,
        previewEvery: values.preview ? Number(values.preview) : options.previewEvery,
        log: (s) => {
          console.log(s);
          appendFileSync(logFile, `${s}\n`);
        },
      });
    }
  } finally {
    await pool.close();
  }
}

/**
 * 第三轮续：面捕输入空间的整段轨迹拟合（traj2.ts）；synth2 = 生成符合 VTS 结构的合成基准真值。
 */
async function runInputs(cmd: string, values: Record<string, unknown>) {
  const shots = loadShots();
  // 内存：每个几何 Worker 一份 Core + 模型约 600MB，每个页面一个 Chromium；fit4 缺省 3 个页面 + 5 个几何 Worker（峰值目标 ≤ 7GB）
  const pool = await Pool.open(values.workers ? Number(values.workers) : cmd === 'fit4' ? 3 : 4);
  const gpool = cmd === 'fit4' ? await GeomPool.open(join(REPO, 'assets', MODEL_URI), values.geom ? Number(values.geom) : 5) : null;
  try {
    const calib = await calibrate(pool.pages[0]!);
    const defaults = await pool.pages[0]!.evaluate(async (m) => {
      const H = window.__charaforge;
      const id = await H.createInstance(m);
      const info = H.instanceInfo(id);
      H.destroyInstance(id);
      return Object.fromEntries(info.parameters.map((p) => [p.id, p.default]));
    }, MODEL_URI);
    const vts = loadVts(join(REPO, 'assets/models/deepseek-chan/DeepSeek/大肥鱼.vtube.json'), defaults);
    const smoothC = values.smoothC ? Number(values.smoothC) : 0.15;
    const units = fitUnits(shots.shots);
    const doneUnits = new Set<string>();
    for (const sid of String(values.shot).split(',')) {
      // P3：非打样镜头按拟合单元（continues 连起来的镜头合成一个虚拟镜头）跑，结果写 .data/fit/<单元首镜头>/
      const unit = LEGACY_SHOTS.has(sid) ? null : units.find((u) => u.shots.includes(sid));
      if (!LEGACY_SHOTS.has(sid) && !unit) throw new Error(`镜头 ${sid} 没有角色或不存在`);
      if (unit && doneUnits.has(unit.id)) continue;
      if (unit) doneUnits.add(unit.id);
      const id = unit ? unit.id : sid;
      const shot = unit ? unit.shot : shots.shots.find((s) => s.id === sid)!;
      const auto = unit ? planAuto(shot) : null;
      const multi = auto ?? (id === 's15' ? planS15v3(shot) : null);
      const plans = multi ? multi.plans : planFor(shot).plans;
      if (cmd === 'synth2') { synthTruth(shot, plans[0]!, vts, calib, smoothC); continue; }
      const dir = join(FIT_DIR, id + FIT_TAG);
      mkdirSync(dir, { recursive: true });
      const logFile = join(dir, 'fit.log');
      appendFileSync(logFile, `==== ${new Date().toISOString()} fit4 ${id}（断点续跑时保留之前的日志）\n`);
      const log = (s: string) => { console.log(s); appendFileSync(logFile, `${s}\n`); };
      if (auto) {
        log(`[${id}] 拟合单元 ${unit!.shots.join('+')} [${shot.start}, ${shot.end})：图层 ${auto.order.map((l) => { const q = plans.find((x) => x.id === l)!; return `${l}[${q.in},${q.out})`; }).join('、')}`);
        for (const k of auto.skipped) log(`[${id}] 跳过实例 ${k.id}：${k.reason}`);
        if (!auto.order.length) throw new Error(`单元 ${id} 没有可拟合的实例`);
      }
      const fixed = values.fixed ? JSON.parse(String(values.fixed)) : undefined;
      const order = multi ? multi.order : [plans[0]!.id];
      const built = new Map<string, unknown>();
      const layers: unknown[] = [];
      const states: Record<string, unknown> = {};
      const scores: Record<string, unknown> = {};
      let elapsed = 0;
      for (const lid of order) {
        const p = plans.find((x) => x.id === lid)!;
        // 图层级断点：整个图层拟合完成后存盘（done_<图层>.json），重跑时直接复用（指纹 = FIT_MODEL + 参数 + shots 内容；FIT_RESUME=0 关闭）
        const fp = createHash('sha256').update(JSON.stringify({ m: process.env.FIT_MODEL ?? '', smoothC, fixed, a: values.itersA, b: values.itersB, refine: values.refine, shots: createHash('sha256').update(readFileSync(join(REPO, 'scenes/deepseek-chan/shots.json'))).digest('hex') })).digest('hex').slice(0, 16);
        const doneFile = join(dir, `done_${lid}.json`);
        if (process.env.FIT_RESUME !== '0' && existsSync(doneFile)) {
          const d = JSON.parse(readFileSync(doneFile, 'utf8'));
          if (d.fp === fp) {
            log(`[${id}] ${lid} 图层断点：直接复用 ${doneFile}`);
            built.set(lid, d.r.built); layers.push(d.r.layer); states[lid] = d.r.state; scores[lid] = d.r.score; elapsed += d.r.elapsed;
            continue;
          }
        }
        const r = await fitInputs(pool, gpool!, shot, p, calib, vts, {
          smoothC, log, fixedConsts: auto ? { ...(p.fixed ?? {}), ...(fixed ?? {}) } : multi ? { clothes2: 0, ...(fixed ?? {}) } : fixed,
          others: () => [...built.values()],
          itersA: values.itersA ? Number(values.itersA) : undefined,
          itersB: values.itersB ? Number(values.itersB) : undefined,
          refineRounds: values.refine ? Number(values.refine) : 0,
        });
        writeFileSync(`${doneFile}.tmp`, JSON.stringify({ fp, r: { built: r.built, layer: r.layer, state: r.state, score: r.score, elapsed: r.elapsed } }));
        renameSync(`${doneFile}.tmp`, doneFile);
        built.set(lid, r.built);
        layers.push(r.layer);
        states[lid] = r.state;
        scores[lid] = r.score;
        elapsed += r.elapsed;
      }
      const zOf = (lid: string) => shot.instances.find((i) => i.id === lid)?.z ?? 0;
      layers.sort((a: any, b: any) => zOf(a.id) - zOf(b.id));
      writeFileSync(join(dir, 'result.json'), JSON.stringify({ shot: id, shots: unit ? unit.shots : [id], skipped: auto?.skipped ?? [], issues: [...(unit?.issues ?? []), ...(auto?.issues ?? [])], range: [shot.start, shot.end], method: 'vts-input-gn', smoothC, elapsedSec: elapsed, layers, state: order.length === 1 ? states[order[0]!] : states, scores }));
    }
  } finally {
    await gpool?.close();
    await pool.close();
  }
}

/** 合成基准真值：把第二轮结果投影到面捕输入空间（逆映射 + 平滑），变换取稀疏关键帧，再经同一前向模型生成图层 */
function synthTruth(shot: import('../../../packages/scene/src/index.ts').Shot, p: import('./fitshot.ts').LayerPlan, vts: import('./vts.ts').VtsModel, calib: ModelCalib, smoothC: number) {
  const r2 = JSON.parse(readFileSync(join(FIT_DIR, 'synth', `truth_r2_${shot.id}.json`), 'utf8')).layers[0];
  const n = r2.out - r2.in;
  const track = (id: string) => {
    const t = r2.params.find((x: any) => x.id === id && !x.stage);
    return t?.curve.baked ? (t.curve.baked.values as number[]) : t ? Array(n).fill(t.curve.keys[0].v) : Array(n).fill(0);
  };
  const ma = (a: number[], w: number) => a.map((_, i) => { let s = 0, c = 0; for (let k = -w; k <= w; k++) { const j = i + k; if (j >= 0 && j < n) { s += a[j]!; c++; } } return s / c; });
  const inv: Record<string, string> = { FaceAngleX: 'AngleX', FaceAngleY: 'AngleY', FaceAngleZ: 'AngleZ', EyeOpenLeft: 'ParamEyeLOpen', EyeOpenRight: 'ParamEyeROpen', EyeRightX: 'ParamEyeBallX', EyeRightY: 'ParamEyeBallY', MouthSmile: 'ParamMouthForm', MouthOpen: 'ParamMouthOpenY', Brows: 'ParamBrowLY' };
  const series: Record<string, number[]> = {};
  for (const [input, out] of Object.entries(inv)) {
    const s = vts.settings.find((x) => x.input === input && x.output === out)!;
    const vals = track(out).map((v) => s.inLo + ((v - s.outLo) / (s.outHi - s.outLo)) * (s.inHi - s.inLo));
    series[input] = ma(vals, 3);
  }
  const frames = Array.from({ length: n }, (_, i) => r2.in + i);
  const zoom = shot.effects.some((e) => e.kind === 'zoom');
  const every = zoom ? 12 : 24;
  const keys: number[] = [];
  for (let f = r2.in; f < r2.out - 1; f += every) keys.push(f);
  keys.push(r2.out - 1);
  const tv = (k: string) => ma(r2.transform[k].baked.values as number[], 6);
  const tx = tv('x'), ty = tv('y'), ts = tv('scale'), tr = tv('rotation');
  const breath: Record<string, [number, number]> = { s24: [0.888, 3.865], s43: [-2.594, 4.838] };
  const blurF = r2.filters.find((f: any) => f.type === 'blur');
  const colF = r2.filters.find((f: any) => f.type === 'color');
  const st = {
    nodes: frames,
    inputs: frames.map((_, i) => Object.fromEntries(vts.inputs.map((k) => [k, series[k] ? series[k]![i]! : vts.neutral[k] ?? 0]))),
    keys,
    tf: keys.map((f) => ({ x: tx[f - r2.in]!, y: ty[f - r2.in]!, ls: Math.log(ts[f - r2.in]!), rot: tr[f - r2.in]! })),
    consts: {
      lean: track('LeanInput')[0]!, clothes2: track('Clothes2')[0]!, breathPhase: breath[shot.id]?.[0] ?? 0, breathPeriod: breath[shot.id]?.[1] ?? 3.5,
      blur: blurF ? blurF.sigma.baked.values[0] : 1, resample: 1,
      color: colF ? { brightness: colF.brightness, contrast: colF.contrast, saturation: colF.saturation, multiply: colF.multiply } : null,
    },
  };
  const layer = buildLayer(p, vts, st as any, calib, smoothC) as any;
  if (r2.frameHold) layer.frameHold = r2.frameHold;
  const scene = {
    format: 'charaforge.scene', version: 1, meta: { title: `合成基准真值（面捕输入空间，${shot.id}）` }, timebase: { fps: 60 }, duration: 3868,
    composition: { width: 1280, height: 720, background: null, antialias: true },
    assets: { models: [{ id: 'dsc', uri: MODEL_URI }] }, layers: [layer],
  };
  mkdirSync(join(FIT_DIR, 'synth'), { recursive: true });
  writeFileSync(join(FIT_DIR, 'synth', `truth2_${shot.id}.json`), JSON.stringify(scene));
  writeFileSync(join(FIT_DIR, 'synth', `truth2_${shot.id}_state.json`), JSON.stringify({ smoothC, state: st }));
  console.log(`[synth2] ${shot.id}：真值场景 → .data/fit/synth/truth2_${shot.id}.json（${n} 帧，${keys.length} 个变换关键帧，常量 ${JSON.stringify({ lean: st.consts.lean, clothes2: st.consts.clothes2 })}）`);
}

/** 第三轮：光流对应 + 整段轨迹 Gauss-Newton（traj.ts） */
async function runTraj(values: Record<string, unknown>) {
  const shots = loadShots();
  const pool = await Pool.open(values.workers ? Number(values.workers) : 4);
  const gpool = await GeomPool.open(join(REPO, 'assets', MODEL_URI), values.geom ? Number(values.geom) : undefined);
  try {
    console.log(`[fit3] GPU: ${pool.renderer}，${pool.size} 个渲染页面，${gpool.size} 个几何 Worker`);
    const calib = await calibrate(pool.pages[0]!);
    for (const id of String(values.shot).split(',')) {
      const shot = shots.shots.find((s) => s.id === id)!;
      const { plans, options } = planFor(shot, { hairBake: !!values.hair });
      const logFile = join(FIT_DIR, id + FIT_TAG, 'fit.log');
      mkdirSync(join(FIT_DIR, id + FIT_TAG), { recursive: true });
      writeFileSync(logFile, '');
      await fitTrajectory(pool, gpool, shot, plans, calib, {
        order: options.order,
        itersA: values.itersA ? Number(values.itersA) : undefined,
        itersB: values.itersB ? Number(values.itersB) : undefined,
        constGrid: !values.noconst,
        log: (s) => { console.log(s); appendFileSync(logFile, `${s}\n`); },
      });
    }
  } finally {
    await gpool.close();
    await pool.close();
  }
}

async function redetectSwitches(shotId: string) {
  const shot = loadShots().shots.find((s) => s.id === shotId)!;
  const { plans } = planFor(shot);
  const file = join(FIT_DIR, shotId + FIT_TAG, 'result.json');
  const res = JSON.parse(rf(file, 'utf8'));
  const calib = JSON.parse(rf(join(FIT_DIR, 'calib.json'), 'utf8'));
  const pool = await Pool.open(1);
  const ref = new RefFrames(shot.start, shot.end);
  try {
    await pool.all(([ls, uri, off]) => (window as any).__fit.setup(ls, uri, off), [res.layers, MODEL_URI, calib.irisOffset] as const);
    for (const p of plans)
      for (const sw of p.switches) {
        if (!sw.detect) continue;
        const r = await redetectSwitch(pool, ref, p, sw);
        const layer = res.layers.find((l: any) => l.id === p.id);
        const t = layer.params.find((x: any) => x.id === sw.param);
        const vals = t.curve.baked.values as number[];
        for (let f = p.in; f < p.out; f++) {
          const u = f < r.start ? 0 : r.fade <= 0 || f >= r.start + r.fade ? 1 : 0.5 - 0.5 * Math.cos((Math.PI * (f - r.start)) / r.fade);
          vals[f - p.in] = Math.round((sw.from + (sw.to - sw.from) * u) * 1e6) / 1e6;
        }
        res.switches = { ...(res.switches ?? {}), [`${p.id}:${sw.param}`]: r };
        console.log(`[switches] ${p.id} ${sw.param}: 起点 ${r.start}、fade ${r.fade}（rms ${r.rms?.toFixed(3)}，${Object.keys(r.samples).length} 个样本）`);
      }
    writeFileSync(file, JSON.stringify(res));
  } finally {
    ref.close();
    await pool.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
