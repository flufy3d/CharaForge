import type { Animatable, Curve, Ease, Keyframe } from './common.ts';

/** 归一化三次贝塞尔 (0,0)-(x1,y1)-(x2,y2)-(1,1)，给定 x 求 y（与 CSS cubic-bezier 一致）。 */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bx = (t: number) => ((1 - 3 * x2 + 3 * x1) * t + (3 * x2 - 6 * x1)) * t * t + 3 * x1 * t;
  const by = (t: number) => ((1 - 3 * y2 + 3 * y1) * t + (3 * y2 - 6 * y1)) * t * t + 3 * y1 * t;
  // 二分法：单调、确定、无需导数，40 次迭代精度远超 float32。
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (bx(mid) < x) lo = mid;
    else hi = mid;
  }
  return by((lo + hi) / 2);
}

/** 缓动函数：u ∈ [0,1] → 插值权重。 */
export function applyEase(ease: Ease | undefined, u: number): number {
  if (ease === undefined || ease === 'linear') return u;
  if (ease === 'step') return 0;
  if (ease === 'smooth') return cubicBezier(0.42, 0, 0.58, 1, u);
  return cubicBezier(ease[0], ease[1], ease[2], ease[3], u);
}

function lerp<V>(a: V, b: V, w: number): V {
  if (typeof a === 'number') return (a + ((b as number) - a) * w) as V;
  if (Array.isArray(a)) return a.map((ai, i) => ai + ((b as number[])[i]! - ai) * w) as V;
  return w < 1 ? a : b;
}

/** 二分查找最后一个 t <= frame 的关键帧下标；frame 早于首帧时返回 -1。 */
function findKey<V>(keys: Keyframe<V>[], frame: number): number {
  let lo = 0;
  let hi = keys.length - 1;
  if (frame < keys[0]!.t) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (keys[mid]!.t <= frame) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export function isCurve<V>(a: Animatable<V>): a is Curve<V> {
  return typeof a === 'object' && a !== null && !Array.isArray(a) && ('keys' in a || 'baked' in a);
}

/** 在帧 frame 上求曲线的值。规则见 common.ts：区间外取首/尾值。 */
export function sampleCurve<V>(curve: Curve<V>, frame: number): V {
  if ('baked' in curve) {
    const { start, values } = curve.baked;
    const i = Math.min(Math.max(Math.round(frame - start), 0), values.length - 1);
    return values[i]!;
  }
  const keys = curve.keys;
  const i = findKey(keys, frame);
  if (i < 0) return keys[0]!.v;
  const k0 = keys[i]!;
  const k1 = keys[i + 1];
  if (!k1 || k1.t === k0.t) return k0.v;
  const u = (frame - k0.t) / (k1.t - k0.t);
  return lerp(k0.v, k1.v, applyEase(k0.ease, u));
}

/** 可动画属性求值；undefined 时返回缺省值。 */
export function sample<V>(a: Animatable<V> | undefined, frame: number, fallback: V): V {
  if (a === undefined) return fallback;
  return isCurve(a) ? sampleCurve(a, frame) : a;
}
