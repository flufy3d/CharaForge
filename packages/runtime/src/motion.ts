/*
 * motion / expression 片段的无状态求值。
 *
 * Cubism 的 MotionManager / ExpressionManager 是「按墙钟时间推进的队列」，状态随调用历史变化，
 * 无法按帧号随机访问。这里改为：片段在帧 f 的效果只由 (片段定义, f) 决定——
 * 每次求值都新建一个 CubismMotionQueueEntry，把开始/淡入/结束时刻设成由帧号算出的值，
 * 再调用 Framework 自己的 doUpdateParameters 做曲线求值与混合（因此曲线插值、
 * 循环修正（MotionBehavior V2）、按参数淡入淡出、Add/Multiply/Overwrite 语义都与官方实现一致）。
 *
 * 由此 motion/expression 没有跨帧状态，实例的完整状态 = 物理状态（见 instance.ts）。
 *
 * 时间约定（片段时间 = (f − start)/fps 秒）：
 *   - motion 内部时间 = offset + 片段时间 × speed；
 *   - 淡入：片段开始起 fadeIn 秒（缺省用 motion3/exp3 自带值，文件未写时 Cubism 缺省 1 秒）；
 *   - 淡出：在结束时刻前 fadeOut 秒内淡到 0（与 Cubism 设置 endTime 后的行为一致）；
 *     结束时刻 = end（若给出）；否则非循环 motion 为自然结束 (duration − offset)/speed；
 *     循环 motion / expression 无结束（持续到图层 out，无淡出）。
 *   - 循环时每圈不重新淡入（等价于 Cubism 的 isLoopFadeIn = false）。
 */
import { CubismMath } from '@cubism/framework/math/cubismmath';
import type { CubismModel } from '@cubism/framework/model/cubismmodel';
import type { CubismExpressionMotion } from '@cubism/framework/motion/cubismexpressionmotion';
import type { CubismMotion } from '@cubism/framework/motion/cubismmotion';
import { CubismMotionQueueEntry } from '@cubism/framework/motion/cubismmotionqueueentry';

interface MotionInternals {
  _fadeInSeconds: number;
  _fadeOutSeconds: number;
  _weight: number;
  _isLoop: boolean;
  _isLoopFadeIn: boolean;
  _previousLoopState: boolean;
  _motionData: { duration: number; fps: number };
}

const mi = (m: CubismMotion | CubismExpressionMotion) => m as unknown as MotionInternals;

export function motionDuration(m: CubismMotion): number {
  return mi(m)._motionData.duration;
}
export function fileFadeIn(m: CubismMotion | CubismExpressionMotion): number {
  return mi(m)._fadeInSeconds;
}
export function fileFadeOut(m: CubismMotion | CubismExpressionMotion): number {
  return mi(m)._fadeOutSeconds;
}

export interface ClipTiming {
  /** 片段时间（秒），≥ 0 */
  elapsed: number;
  /** 结束时刻（片段时间，秒）；undefined 表示无结束 */
  endTime?: number;
  fadeIn: number;
  fadeOut: number;
  /** 片段权重（已按帧采样） */
  weight: number;
}

/** 片段在当前时刻的总权重 = weight · fadeIn · fadeOut（Cubism 的 sin 缓动）。 */
export function clipFadeWeight(t: ClipTiming): number {
  const fin = t.fadeIn <= 0 ? 1 : CubismMath.getEasingSine(t.elapsed / t.fadeIn);
  const fout = t.fadeOut <= 0 || t.endTime === undefined ? 1 : CubismMath.getEasingSine((t.endTime - t.elapsed) / t.fadeOut);
  return t.weight * fin * fout;
}

/**
 * 叠加一个 motion 片段。motionTime 为 motion 内部时间（秒，未取模；循环由 Framework 处理）。
 */
export function applyMotionClip(model: CubismModel, motion: CubismMotion, t: ClipTiming, motionTime: number, loop: boolean): void {
  const m = mi(motion);
  // motion 对象在多实例间共享；每次调用前把会被读取的字段全部写一遍（同步调用，无竞争）。
  m._isLoop = loop;
  m._previousLoopState = loop; // 防止 V2 分支里 adjustEndTime 改写我们设定的结束时刻
  m._isLoopFadeIn = false;
  m._fadeInSeconds = t.fadeIn;
  m._fadeOutSeconds = t.fadeOut;
  m._weight = t.weight;
  const entry = new CubismMotionQueueEntry();
  entry.setIsStarted(true);
  entry.setStartTime(t.elapsed - motionTime);
  entry.setFadeInStartTime(0);
  entry.setEndTime(t.endTime ?? -1);
  motion.doUpdateParameters(model, t.elapsed, clipFadeWeight(t), entry);
}

/** 叠加一个 expression 片段（exp3 的 Add / Multiply / Overwrite，按权重）。 */
export function applyExpressionClip(model: CubismModel, expr: CubismExpressionMotion, t: ClipTiming): void {
  const w = clipFadeWeight(t);
  if (w <= 0) return;
  expr.doUpdateParameters(model, t.elapsed, w, null as unknown as CubismMotionQueueEntry);
}
