/*
 * A5：把 tools/fit 的拟合单元与通用图层规划（plans.ts 的 fitUnits + planAuto）导出成 JSON，给 GPU 拟合用，
 * 保证规划语义与 fit4 完全一致（同一份代码）。逐帧量（景深 blur、翻牌 scaleX、clip、不透明度）按 traj2.ts buildLayer 的规则采样好。
 *   node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_plan.ts <镜头ID> <out.json>
 * 镜头可以是单元内任一镜头；打样镜头（s15、s24、s43）自成一个单元，同样走 planAuto。
 */
import { writeFileSync } from 'node:fs';
import { fitUnits, planAuto } from '../../fit/src/plans.ts';
import { loadShots } from '../../fit/src/shotsutil.ts';

const [sid, out] = process.argv.slice(2);
const shots = loadShots();
const unit = fitUnits(shots.shots).find((u) => u.shots.includes(sid!));
if (!unit) throw new Error(`镜头 ${sid} 没有拟合单元（无角色？）`);
const shot = unit.shot;
const a = planAuto(shot);

// traj2.ts 的 interp：返回 (左, 右, 右权重)，区间外保持端点
const interp = (xs: number[], f: number): [number, number, number] => {
  if (f <= xs[0]!) return [0, 0, 0];
  if (f >= xs[xs.length - 1]!) return [xs.length - 1, xs.length - 1, 0];
  let i = 0;
  while (xs[i + 1]! <= f) i++;
  return [i, i + 1, (f - xs[i]!) / (xs[i + 1]! - xs[i]!)];
};

const layers = a.order.map((lid) => {
  const p = a.plans.find((x) => x.id === lid)!;
  const n = p.out - p.in;
  const fr = Array.from({ length: n }, (_, i) => p.in + i);
  // 景深 / 模糊段（traj2 dofAt）：段内 blur = dofK × 强度；段外为 null（用常量 blur）
  const dof = fr.map((g) => (p.blurRanges.some(([s, e]) => g >= s && g < e) ? (p.dofStrength ? p.dofStrength(g) : 8) : null));
  // 翻牌（traj2 scaleX）
  const scaleX = p.flipCurve ? fr.map((g) => Math.max(0.02, p.flipCurve!(g))) : null;
  // clip（traj2 clipAt + buildLayer 的 clip 段）
  const ks = p.clipKeys;
  const clipAt = (g: number) => {
    if (!ks || !p.clipRange || g < p.clipRange[0] || g >= p.clipRange[1]) return null;
    const [l, r, w] = interp(ks.map((k) => k.f), g);
    const A = ks[l]!, B = ks[r]!;
    const m = (k: 'cx' | 'cy' | 'r' | 'a') => A[k] * (1 - w) + B[k] * w;
    const rw = A.w !== undefined && B.w !== undefined ? A.w * (1 - w) + B.w * w : undefined;
    const rh = A.h !== undefined && B.h !== undefined ? A.h * (1 - w) + B.h * w : undefined;
    return { cx: m('cx'), cy: m('cy'), r: Math.max(0, m('r')), a: Math.min(1, Math.max(0, m('a'))), w: rw, h: rh };
  };
  let clip = null as null | Record<string, number[] | number>;
  let opacity: number[] | null = null;
  if (p.clipRange) {
    const cs = fr.map(clipAt);
    clip = {
      x: cs.map((c) => (c ? c.cx - (c.w ?? 2 * c.r) / 2 : -10000)), y: cs.map((c) => (c ? c.cy - (c.h ?? 2 * c.r) / 2 : -10000)),
      width: cs.map((c) => (c ? c.w ?? 2 * c.r : 30000)), height: cs.map((c) => (c ? c.h ?? 2 * c.r : 30000)),
      radius: cs.map((c) => (c ? c.r : 0)), feather: cs.some((c) => c?.w !== undefined) ? 1 : 3,
    };
    opacity = cs.map((c, i) => (c ? c.a : p.opacityCurve ? p.opacityCurve(p.in + i) : 1));
  } else if (p.opacityCurve) {
    opacity = fr.map((g) => Math.min(1, Math.max(0, p.opacityCurve!(g))));
  }
  return {
    id: p.id, in: p.in, out: p.out, z: p.z, constants: p.constants,
    switches: p.switches.map((s) => ({ param: s.param, from: s.from, to: s.to, est: s.est, fade: s.fade, window: s.window })),
    fixed: p.fixed ?? {}, colorFilter: p.colorFilter, dof, scaleX, clip, opacity, track: p.inst.track,
  };
});

writeFileSync(out!, JSON.stringify({
  unit: unit.id, shots: unit.shots, start: shot.start, end: shot.end,
  effects: shot.effects.map((e: any) => ({ kind: e.kind, start: e.start, end: e.end })),
  layers, skipped: a.skipped, issues: [...unit.issues, ...a.issues],
}));
console.log(`[a5_plan] 单元 ${unit.id}（${unit.shots.join('+')}）[${shot.start}, ${shot.end})：${layers.map((l) => `${l.id}[${l.in},${l.out})`).join(' ')} → ${out}`);
