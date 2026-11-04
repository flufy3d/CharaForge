/*
 * 每个镜头的图层规划：哪些实例成为图层、图层区间、开关常量/切换、哪些段逐帧拟合模糊/不透明度/裁切。
 * 由 shots.json 推导；shots 里不准的地方（实例出现时间、开窗裁切等）按画面观察修正，并在汇报里列出。
 */
import type { Shot } from '../../../packages/scene/src/index.ts';
import type { FitOptions, LayerPlan } from './fitshot.ts';

const ALL_CONSTS = ['c:LeanInput', 'c:Clothes2', 'c:breathPhase', 'c:breathPeriod', 'c:blur'];

export function planFor(shot: Shot, opts: { hairBake?: boolean } = {}): { plans: LayerPlan[]; options: FitOptions } {
  const base = (id: string, over: Partial<LayerPlan> = {}): LayerPlan => {
    const inst = shot.instances.find((i) => i.id === id)!;
    const constants = { ...inst.variant };
    // Clothes2（体型）作为镜头常量拟合；shots 的 variant 只当初值
    const constInit: Record<string, number> = {};
    if (constants.Clothes2 !== undefined) constInit['c:Clothes2'] = constants.Clothes2;
    return {
      id: inst.id,
      inst,
      in: shot.start,
      out: shot.end,
      constants,
      switches: [],
      blurRanges: [],
      opacityRanges: [],
      consts: ALL_CONSTS,
      constInit,
      colorFilter: false,
      hairBake: !!opts.hairBake,
      z: inst.z,
      ...over,
    };
  };
  switch (shot.id) {
    case 's43':
    case 's24':
      // 角色有整体调色（偏亮、低对比、偏蓝）：拟合镜头内常量的颜色滤镜
      return { plans: [base(shot.instances[0]!.id, { colorFilter: true })], options: { previewEvery: 6 } };
    case 's15': {
      // [1300, 1509)：1300 起左、右两个实例完整可见（尚未虚化）；约 1340~1360 两者虚化到 σ≈8；
      // 中间实例约 1352 起在圆形开窗里淡入（shots 没有这个 clip），约 1400 开窗消失。
      // 1187~1299（泼溅擦除、细线框全身、放大 wink、翻牌出现 Q 版）见汇报。
      const S = 1300;
      const center = base('s15.center', {
        in: 1352,
        clipRange: [1352, 1400],
        opacityRanges: [[1352, 1368]],
        // key4 爱心眼：画面上 1404~1429 是闭眼笑，1430 睁眼时已经是爱心眼（高饱和品红像素从 1430 开始出现），
        // 所以切换发生在闭眼期间、精确帧不可观测；按 VTS 表情热键的 0.5 秒淡入取 [1400, 1430)。
        // （shots 的 1450 偏晚；用拟合虹膜位置做的自动检测会把头发青色高光当成虹膜，见汇报。）
        switches: [{ param: 'key4', from: 0, to: 1, window: [1400, 1431], est: 1400, fade: 30 }],
      });
      const per = ['c:LeanInput', 'c:Clothes2', 'c:breathPhase', 'c:breathPeriod'];
      const left = base('s15.left', { in: S, blurRanges: [[S, shot.end]], consts: per });
      const chibi = base('s15.chibi', { in: S, blurRanges: [[S, shot.end]], consts: per, constInit: { 'c:Clothes2': 1 } });
      return { plans: [center, left, chibi], options: { previewEvery: 8, order: ['s15.center', 's15.left', 's15.chibi'] } };
    }
    default:
      throw new Error(`镜头 ${shot.id} 还没有拟合规划`);
  }
}

/** shots 效果强度曲线的线性插值（区间外保持端点） */
const strengthAt = (ks: { frame: number; value: number }[], f: number) => {
  if (f <= ks[0]!.frame) return ks[0]!.value;
  if (f >= ks[ks.length - 1]!.frame) return ks[ks.length - 1]!.value;
  let i = 0;
  while (ks[i + 1]!.frame <= f) i++;
  const t = (f - ks[i]!.frame) / (ks[i + 1]!.frame - ks[i]!.frame);
  return ks[i]!.value + (ks[i + 1]!.value - ks[i]!.value) * t;
};

/**
 * 第三轮续（fit4）的 s15 规划：全部从 shots 读取（recon 重标后的结构），不再硬编码帧号。
 *   图层起点 = 实例轨迹里第一个有 bbox 的帧（翻牌实例取 cardFlip 效果起点）；
 *   开关 = 实例 switchEvents（frame、fade、value）；景深 = depthOfField 效果（instances、strength 曲线）；
 *   翻牌 = cardFlip 效果（strength 即 scaleX）；圆形开窗 = 实例 clip 关键帧 + opacity 轨（之后不裁切）。
 *   变换由 shots 眼睛锚点先验约束（traj2），不做 reg 粗配准。
 */
export function planS15v3(shot: Shot): { plans: LayerPlan[]; order: string[] } {
  const inst = (id: string) => shot.instances.find((i) => i.id === id)!;
  const fx = (kind: string, id: string) => shot.effects.find((e: any) => e.kind === kind && (!e.instances || e.instances.includes(id))) as any;
  const firstSeen = (id: string) => (inst(id) as any).track.find((t: any) => t.bbox)?.frame ?? shot.start;
  const plan = (id: string): LayerPlan => {
    const I = inst(id) as any;
    const flip = fx('cardFlip', id);
    const dof = fx('depthOfField', id);
    const start = flip ? flip.start : firstSeen(id);
    const switches = (I.switchEvents ?? []).map((e: any) => ({ param: e.param, from: 0, to: e.value, window: [e.frame - 8, e.frame + 8] as [number, number], est: e.frame, fade: e.fade ?? 0 }));
    let clipKeys: LayerPlan['clipKeys'];
    let clipRange: [number, number] | undefined;
    if (I.clip?.length) {
      const op = I.track.filter((t: any) => t.opacity !== undefined && t.opacity !== null).map((t: any) => ({ frame: t.frame, value: t.opacity }));
      clipKeys = I.clip.map((c: any) => ({ f: c.frame, cx: c.rect[0] + c.rect[2] / 2, cy: c.rect[1] + c.rect[3] / 2, r: c.radius, a: op.length ? strengthAt(op, c.frame) : 1 }));
      clipRange = [I.clip[0].frame, I.clip[I.clip.length - 1].frame + 1];
    }
    return {
      id, inst: I, in: start, out: shot.end, constants: { ...I.variant }, switches, blurRanges: dof ? [[dof.start, shot.end]] : [], opacityRanges: [],
      consts: [], colorFilter: false, hairBake: false, z: I.z,
      ...(dof ? { dofStrength: (f: number) => (dof.strength?.length ? strengthAt(dof.strength, f) : 8) } : {}),
      ...(flip ? { flipCurve: (f: number) => strengthAt(flip.strength, f) } : {}),
      ...(clipKeys ? { clipKeys, clipRange } : {}),
      reg: false, // reg 在 s15（c3）上把左实例推偏约 150px、s24 上也有害：关闭，改用 shots 锚点先验（traj2）
    } as LayerPlan;
  };
  const left = plan('s15.left'), chibi = plan('s15.chibi'), center = plan('s15.center');
  return { plans: [left, chibi, center], order: ['s15.left', 's15.chibi', 's15.center'] };
}

// ======================================================================================
// P3：全片通用规划（全部从 shots.json 推出）
// ======================================================================================

/** 已用专门规划正式拟合过的镜头（结果直接复用，不重跑）；跨镜头 continues 链在这些镜头处断开 */
export const LEGACY_SHOTS = new Set(['s15', 's24', 's43']);

/** shots 问题清单的一行（汇报用；不直接改 shots.json） */
export interface ShotIssue { shot: string; inst: string; frame: number | null; field: string; problem: string; suggest: string }

/** 拟合单元：一组由 continues 连起来的镜头（同一实例跨镜头 = 一个图层，物理不重置） */
export interface FitUnit {
  /** 单元 ID = 首个镜头 ID（结果写 .data/fit/<id>/result.json） */
  id: string;
  shots: string[];
  /** 合并后的虚拟镜头：区间 = 各镜头之并，实例 = continues 链合并，排除区/效果取并 */
  shot: Shot;
  issues: ShotIssue[];
}

/** 按镜头顺序把 continues 连接的镜头并成拟合单元；无角色镜头不成单元；LEGACY 镜头单独成单元 */
export function fitUnits(shots: Shot[]): FitUnit[] {
  const byInst = new Map<string, Shot>();
  for (const s of shots) for (const i of s.instances) byInst.set(i.id, s);
  const parent = new Map<string, string>(shots.map((s) => [s.id, s.id]));
  const find = (x: string): string => (parent.get(x) === x ? x : find(parent.get(x)!));
  for (const s of shots) {
    if (!s.hasCharacter || LEGACY_SHOTS.has(s.id)) continue;
    for (const i of s.instances) {
      const prev = i.continues ? byInst.get(i.continues) : undefined;
      if (!prev || LEGACY_SHOTS.has(prev.id) || !prev.hasCharacter) continue;
      parent.set(find(s.id), find(prev.id));
    }
  }
  const groups = new Map<string, Shot[]>();
  for (const s of shots) {
    if (!s.hasCharacter) continue;
    const r = find(s.id);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r)!.push(s);
  }
  return [...groups.values()]
    .map((g) => g.sort((a, b) => a.start - b.start))
    .sort((a, b) => a[0]!.start - b[0]!.start)
    .map((g) => mergeUnit(g, byInst));
}

/** 合并一个单元内的镜头（单镜头单元原样返回镜头对象，但实例会做眼睛初值补全） */
function mergeUnit(g: Shot[], byInst: Map<string, Shot>): FitUnit {
  const issues: ShotIssue[] = [];
  const ids = new Set(g.map((s) => s.id));
  // continues 链：实例 → 链根（只在单元内追溯；指向单元外的实例视为链根，并记下前驱）
  const rootOf = (i: ShotInstance): string => {
    let cur = i;
    for (;;) {
      const prevShot = cur.continues ? byInst.get(cur.continues) : undefined;
      if (!prevShot || !ids.has(prevShot.id)) return cur.id;
      cur = prevShot.instances.find((x) => x.id === cur.continues)!;
    }
  };
  const chains = new Map<string, { shot: Shot; inst: ShotInstance }[]>();
  for (const s of g) for (const i of s.instances) {
    const r = rootOf(i);
    if (!chains.has(r)) chains.set(r, []);
    chains.get(r)!.push({ shot: s, inst: i });
  }
  const instances: ShotInstance[] = [];
  for (const [rid, segs] of chains) {
    const head = segs[0]!.inst;
    // 开关：链根 variant + 各段 switchEvents；后续段 variant 与累计状态不一致时补一个阶跃切换并记问题
    const state: Record<string, number> = { ...head.variant };
    const events: ShotInstance['switchEvents'] = [];
    segs.forEach(({ shot, inst }, k) => {
      if (k > 0) {
        for (const [key, v] of Object.entries(inst.variant)) {
          if ((state[key] ?? 0) !== v) {
            issues.push({ shot: shot.id, inst: inst.id, frame: shot.start, field: `variant.${key}`, problem: `continues 链上前一段结束时 ${key}=${state[key] ?? 0}，本段 variant 为 ${v}，前一段没有对应 switchEvent`, suggest: `在前一段补 switchEvent（param ${key}, value ${v}），或改本段 variant` });
            events.push({ frame: shot.start, param: key, value: v, fade: 0 });
            state[key] = v;
          }
        }
        for (const [key, v] of Object.entries(state)) {
          if (v !== 0 && inst.variant[key] === undefined) issues.push({ shot: shot.id, inst: inst.id, frame: shot.start, field: `variant.${key}`, problem: `continues 链上累计 ${key}=${v}，本段 variant 未列出（按规范即默认值）`, suggest: `本段 variant 补 ${key}: ${v}（若确实切回默认则在前一段补 switchEvent）` });
        }
      }
      for (const e of inst.switchEvents) { events.push(e); state[e.param] = e.value; }
    });
    const track = segs.flatMap(({ inst }) => inst.track).sort((a, b) => a.frame - b.frame);
    const clipSegs = segs.filter(({ inst }) => inst.clip?.length);
    const clip = clipSegs.length ? clipSegs.flatMap(({ inst }) => inst.clip!).sort((a, b) => a.frame - b.frame) : null;
    // z：取可见帧最多的那段（同一时刻各实例的相对前后与 shots 一致即可）
    const span = (i: ShotInstance) => i.track.filter((t) => t.bbox).length;
    const zSeg = segs.reduce((a, b) => (span(b.inst) > span(a.inst) ? b : a));
    instances.push({
      ...head, id: rid, z: zSeg.inst.z, track, switchEvents: events, clip,
      colorGrade: segs.map(({ inst }) => inst.colorGrade).filter(Boolean).join('；') || undefined,
      notes: segs.map(({ inst }) => `${inst.id}: ${inst.notes ?? ''}`).join(' | '),
      // continues 指向单元外（已拟合镜头）的实例：保留，供汇报
      continues: head.continues,
    } as ShotInstance);
  }
  const first = g[0]!, last = g[g.length - 1]!;
  const shot: Shot = g.length === 1 ? { ...first, instances } : {
    ...first,
    end: last.end,
    category: g.map((s) => s.category).join('+'),
    instances,
    exclusions: g.flatMap((s) => s.exclusions),
    effects: g.flatMap((s) => s.effects),
    notes: g.map((s) => `${s.id}: ${s.notes ?? ''}`).join(' | '),
  };
  return { id: first.id, shots: g.map((s) => s.id), shot, issues };
}

/** shots 轨迹某字段的分段线性插值（只用非空采样；区间外保持端点） */
const sampleAt = (inst: ShotInstance, key: 'opacity' | 'blurSigma', f: number): number | null => {
  const ks = inst.track.filter((t) => typeof t[key] === 'number').map((t) => ({ frame: t.frame, value: t[key] as number }));
  return ks.length ? strengthAt(ks, f) : null;
};

/**
 * 通用图层规划（P3）：从（合并后的）镜头推出全部图层，规则与 planS15v3 一致并推广：
 *   - in = 轨迹第一个有 bbox 的采样（翻牌实例取 cardFlip 起点），out = 最后一个 bbox 之后第一个 bbox=null 的采样帧（无则镜头末）；
 *   - 开关：variant 为常量，switchEvents（含 fade）为斜坡，from 取切换前的累计值；
 *   - clip：circleWindow 效果作用的实例为圆形开窗（关键帧区间内裁切，之后不裁切）；其余为画中画圆角矩形（整层裁切，区间外保持端点）；
 *     不透明度取轨迹 opacity（无 clip 时也作为图层不透明度轨）；
 *   - 景深 depthOfField → 逐帧 blur = dofK × 强度曲线；其他实例轨迹 blurSigma ≥ 1.5 的区间同样处理（强度 = blurSigma）；
 *   - cardFlip → scaleX 曲线；颜色滤镜：有 colorGrade（实例或效果）、单实例镜头或 main 实例时拟合；
 *   - 镜头常量：Clothes2 取 variant（缺省 0）固定；前倾只对单元内头最大、头高 ≥ 300px 的实例搜索，其余固定 0；
 *   - 没有任何 eyes 的实例用 head 框估一个眼睛初值（confidence 0，不进锚点先验）；head、eyes 都没有的实例跳过（记问题）。
 */
export function planAuto(shot: Shot): { plans: LayerPlan[]; order: string[]; skipped: { id: string; reason: string }[]; issues: ShotIssue[] } {
  const issues: ShotIssue[] = [];
  const skipped: { id: string; reason: string }[] = [];
  const fxOf = (kind: string, id: string) => shot.effects.find((e) => e.kind === kind && (!e.instances || e.instances.includes(id))) as any;
  const headH = (i: ShotInstance) => Math.max(0, ...i.track.filter((t) => t.head).map((t) => t.head![3]));
  const shotGrade = shot.effects.some((e) => e.kind === 'colorGrade');
  const plans: LayerPlan[] = [];
  for (const I0 of shot.instances) {
    const I = { ...I0, track: I0.track.map((t) => ({ ...t })) } as ShotInstance;
    const seen = I.track.filter((t) => t.bbox);
    if (!seen.length) { skipped.push({ id: I.id, reason: '轨迹里没有 bbox（不可见）' }); continue; }
    // continues 指向单元外（已正式拟合的打样镜头）的短尾段：前驱图层已单独导出，这里不再另起一个物理重置的图层
    if (I.continues && !shot.instances.some((x) => x.id === I.continues) && seen[seen.length - 1]!.frame - seen[0]!.frame <= 30) {
      skipped.push({ id: I.id, reason: `转场尾段（continues ${I.continues}，前驱已在打样镜头中拟合），不单独起图层` });
      issues.push({ shot: shot.id, inst: I.id, frame: seen[0]!.frame, field: 'continues', problem: `continues ${I.continues} 指向已拟合的打样镜头；本段 ${seen[0]!.frame}～${seen[seen.length - 1]!.frame} 没有图层（转场特效帧）`, suggest: '若要连续，需重跑前驱镜头并把本段并入（全片阶段再定）' });
      continue;
    }
    if (!I.track.some((t) => t.eyes)) {
      const heads = I.track.filter((t) => t.head);
      if (!heads.length) {
        skipped.push({ id: I.id, reason: '没有 eyes 也没有 head，无法给变换初值' });
        issues.push({ shot: shot.id, inst: I.id, frame: seen[0]!.frame, field: 'track.eyes/head', problem: '可见（有 bbox）但没有任何 eyes/head 采样，拟合无法初始化', suggest: '补至少首尾两个 eyes 锚点（或 head 框）；若角色只露身体，标 eyesState hidden 并给 head 的画面外估计' });
        continue;
      }
      // 用 head 框估眼睛（参考姿态下两眼约在头框 55% 高度、水平 ±18%）
      for (const t of I.track) if (t.head && !t.eyes) {
        const [x, y, w, h] = t.head;
        t.eyes = [[x + w * 0.32, y + h * 0.55], [x + w * 0.68, y + h * 0.55]];
        t.confidence = 0;
      }
      issues.push({ shot: shot.id, inst: I.id, frame: heads[0]!.frame, field: 'track.eyes', problem: '没有任何 eyes 采样（拟合用 head 框估初值，不进锚点先验）', suggest: '补 eyes 锚点（至少首尾与转身处）' });
    }
    const flip = fxOf('cardFlip', I.id);
    const dof = fxOf('depthOfField', I.id);
    const circ = fxOf('circleWindow', I.id);
    const lastSeen = seen[seen.length - 1]!.frame;
    const after = I.track.find((t) => t.frame > lastSeen && t.bbox === null);
    const start = flip ? flip.start : seen[0]!.frame;
    const end = Math.min(shot.end, after ? after.frame : shot.end);
    if (end - start < 2) { skipped.push({ id: I.id, reason: `可见区间过短 [${start}, ${end})` }); continue; }
    // 开关：from = 切换前累计值
    const cur: Record<string, number> = { ...I.variant };
    const switches = [...I.switchEvents].sort((a, b) => a.frame - b.frame).map((e) => {
      const from = cur[e.param] ?? 0;
      cur[e.param] = e.value;
      return { param: e.param, from, to: e.value, window: [e.frame - 8, e.frame + 8] as [number, number], est: e.frame, fade: e.fade ?? 0 };
    });
    for (const sw of switches) if (sw.est < start || sw.est >= end) issues.push({ shot: shot.id, inst: I.id, frame: sw.est, field: `switchEvents.${sw.param}`, problem: `切换帧 ${sw.est} 不在可见区间 [${start}, ${end}) 内`, suggest: '核对切换帧或 bbox 可见区间' });
    // 不透明度轨
    const opS = I.track.filter((t) => typeof t.opacity === 'number');
    const opCurve = opS.some((t) => t.opacity! < 0.99) ? (f: number) => sampleAt(I, 'opacity', f)! : undefined;
    // clip
    let clipKeys: LayerPlan['clipKeys'];
    let clipRange: [number, number] | undefined;
    if (I.clip?.length) {
      const a = (f: number) => (opCurve ? opCurve(f) : 1);
      if (circ) {
        clipKeys = I.clip.map((c) => ({ f: c.frame, cx: c.rect[0] + c.rect[2] / 2, cy: c.rect[1] + c.rect[3] / 2, r: c.radius ?? Math.min(c.rect[2], c.rect[3]) / 2, a: a(c.frame) }));
        clipRange = [I.clip[0]!.frame, I.clip[I.clip.length - 1]!.frame + 1];
      } else {
        clipKeys = I.clip.map((c) => ({ f: c.frame, cx: c.rect[0] + c.rect[2] / 2, cy: c.rect[1] + c.rect[3] / 2, w: c.rect[2], h: c.rect[3], r: c.radius ?? 0, a: a(c.frame) }));
        clipRange = [start, end];
      }
      if (I.clip[0]!.frame > start + 2 && !circ) issues.push({ shot: shot.id, inst: I.id, frame: start, field: 'clip', problem: `首个 clip 关键帧 ${I.clip[0]!.frame} 晚于可见起点 ${start}（之前按首帧窗口保持）`, suggest: `在 ${start} 补 clip 关键帧` });
    }
    // 模糊：景深效果，或轨迹 blurSigma ≥ 1.5 的区间
    let blurRanges: [number, number][] = [];
    let dofStrength: ((f: number) => number) | undefined;
    if (dof) {
      blurRanges = [[Math.max(start, dof.start), end]];
      dofStrength = (f: number) => (dof.strength?.length ? strengthAt(dof.strength, f) : 8);
    } else if (I.track.some((t) => (t.blurSigma ?? 0) >= 1.5)) {
      const fs: number[] = [];
      for (let f = start; f < end; f++) if ((sampleAt(I, 'blurSigma', f) ?? 0) >= 1.5) fs.push(f);
      if (fs.length) {
        blurRanges = [[fs[0]!, fs[fs.length - 1]! + 1]];
        dofStrength = (f: number) => Math.max(0.3, sampleAt(I, 'blurSigma', f)!);
      }
    }
    plans.push({
      id: I.id, inst: I, in: start, out: end, constants: { ...I.variant }, switches, blurRanges, opacityRanges: [],
      consts: [], colorFilter: !!I.colorGrade || shotGrade || shot.instances.length === 1 || I.role === 'main', hairBake: false, z: I.z,
      ...(dofStrength ? { dofStrength } : {}),
      ...(flip ? { flipCurve: (f: number) => strengthAt(flip.strength, f) } : {}),
      ...(clipKeys ? { clipKeys, clipRange } : {}),
      ...(opCurve && !clipKeys ? { opacityCurve: opCurve } : {}),
      fixed: { clothes2: I.variant.Clothes2 ?? 0, lean: 0 },
      reg: false,
    } as LayerPlan);
  }
  // 前倾搜索：只对（未跳过的）头最大、头高 ≥ 300px 的实例
  const big = [...plans].sort((a, b) => headH(b.inst) - headH(a.inst))[0];
  if (big && headH(big.inst) >= 300) delete big.fixed!.lean;
  // 拟合顺序：先靠后（z 小）的图层，已拟合的图层参与后续图层的渲染；同 z 先出现的先拟合
  const order = [...plans].sort((a, b) => a.z - b.z || a.in - b.in).map((p) => p.id);
  return { plans, order, skipped, issues };
}
