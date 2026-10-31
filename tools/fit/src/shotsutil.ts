/*
 * shots.json 的读取与插值：轨迹采样、排除区、帧重复（cadence）检测。
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseShots, type Shot, type ShotInstance, type Shots, type TrackSample } from '../../../packages/scene/src/index.ts';
import { REPO } from './paths.ts';
import type { RefFrames } from './refvideo.ts';

export function loadShots(): Shots {
  return parseShots(JSON.parse(readFileSync(join(REPO, 'scenes/deepseek-chan/shots.json'), 'utf8')));
}

type Num = number;
const lerp = (a: Num, b: Num, t: Num) => a + (b - a) * t;

/** 轨迹某个字段在帧 f 的线性插值（只用该字段非空的采样；区间外取首/尾值） */
export function trackAt<K extends 'bbox' | 'head' | 'eyes' | 'rotation' | 'blurSigma' | 'opacity'>(
  inst: ShotInstance,
  key: K,
  f: number,
): NonNullable<TrackSample[K]> | null {
  const ks = inst.track.filter((t) => t[key] !== undefined && t[key] !== null);
  if (!ks.length) return null;
  if (f <= ks[0]!.frame) return ks[0]![key] as NonNullable<TrackSample[K]>;
  if (f >= ks[ks.length - 1]!.frame) return ks[ks.length - 1]![key] as NonNullable<TrackSample[K]>;
  let i = 0;
  while (ks[i + 1]!.frame <= f) i++;
  const a = ks[i]!, b = ks[i + 1]!;
  const t = (f - a.frame) / (b.frame - a.frame);
  const va = a[key] as unknown, vb = b[key] as unknown;
  if (typeof va === 'number') return lerp(va, vb as number, t) as NonNullable<TrackSample[K]>;
  const flat = (v: unknown): number[] => (Array.isArray(v) ? (v as unknown[]).flatMap(flat) : [v as number]);
  const fa = flat(va), fb = flat(vb);
  const out = fa.map((x, j) => lerp(x, fb[j]!, t));
  if (key === 'eyes') return [[out[0]!, out[1]!], [out[2]!, out[3]!]] as NonNullable<TrackSample[K]>;
  return out as NonNullable<TrackSample[K]>;
}

/** 帧 f 上的排除矩形（视频像素 [x,y,w,h]）。关键帧之间线性插值，区间外取首/尾。 */
export function exclusionsAt(shot: Shot, f: number, opts: { skipReasons?: string[] } = {}): number[][] {
  const out: number[][] = [];
  for (const e of shot.exclusions) {
    if (f < e.start || f >= e.end) continue;
    if (opts.skipReasons?.includes(e.reason)) continue;
    const ks = e.rects;
    let r: number[];
    if (f <= ks[0]!.frame) r = ks[0]!.rect;
    else if (f >= ks[ks.length - 1]!.frame) r = ks[ks.length - 1]!.rect;
    else {
      let i = 0;
      while (ks[i + 1]!.frame <= f) i++;
      const t = (f - ks[i]!.frame) / (ks[i + 1]!.frame - ks[i]!.frame);
      r = ks[i]!.rect.map((v, j) => lerp(v, ks[i + 1]!.rect[j]!, t));
    }
    out.push(r);
  }
  return out;
}

/** 矩形 r 中不被排除区覆盖的面积比例（按 8px 网格近似） */
export function visibleFraction(r: number[], excl: number[][]): number {
  const [x, y, w, h] = r as [number, number, number, number];
  const x0 = Math.max(0, x), y0 = Math.max(0, y), x1 = Math.min(1280, x + w), y1 = Math.min(720, y + h);
  let tot = 0, vis = 0;
  for (let py = y0 + 4; py < y1; py += 8)
    for (let px = x0 + 4; px < x1; px += 8) {
      tot++;
      if (!excl.some(([ex, ey, ew, eh]) => px >= ex! && px < ex! + ew! && py >= ey! && py < ey! + eh!)) vis++;
    }
  return tot ? vis / tot : 0;
}

/**
 * 帧重复检测（在角色区域内）：取实例头部框中央（无头部框时用 bbox）作为角色区域，
 * d(f) = 与 f−1 的平均绝对差。判为重复：d < 0.35，或 d < 0.6·min(d(f−1), d(f+1))（交替模式里的低谷）；
 * 再做一次奇偶一致性修补：f−2、f+2 都是重复且 f±1 都是新帧时，f 也判为重复。
 * 返回逐帧的「是否新画面」与 d 值，供日志核对。
 */
export function detectNewFrames(ref: RefFrames, shot: Shot, inst: ShotInstance, a: number, b: number) {
  const d = new Map<number, number>();
  let prev = ref.frame(a);
  for (let f = a + 1; f < b; f++) {
    const cur = ref.frame(f);
    const box = trackAt(inst, 'head', f) ?? trackAt(inst, 'bbox', f);
    let x0 = 0, y0 = 62, x1 = 1280, y1 = 652;
    if (box) {
      const [x, y, w, h] = box as number[];
      x0 = Math.max(0, Math.round(x! + w! * 0.2)); x1 = Math.min(1280, Math.round(x! + w! * 0.8));
      y0 = Math.max(62, Math.round(y! + h! * 0.25)); y1 = Math.min(652, Math.round(y! + h! * 0.9));
    }
    let s = 0, n = 0;
    for (let y = y0; y < y1; y += 2)
      for (let x = x0; x < x1; x += 2) {
        const o = (y * 1280 + x) * 3;
        s += Math.abs(cur[o]! - prev[o]!) + Math.abs(cur[o + 1]! - prev[o + 1]!) + Math.abs(cur[o + 2]! - prev[o + 2]!);
        n += 3;
      }
    d.set(f, n ? s / n : 0);
    prev = cur;
  }
  const isNew = new Map<number, boolean>();
  isNew.set(a, true);
  for (let f = a + 1; f < b; f++) {
    const v = d.get(f)!;
    const nb = Math.min(d.get(f - 1) ?? Infinity, d.get(f + 1) ?? Infinity);
    isNew.set(f, !(v < 0.35 || v < 0.6 * nb));
  }
  for (let f = a + 2; f < b - 2; f++) {
    if (isNew.get(f) && !isNew.get(f - 2) && !isNew.get(f + 2) && isNew.get(f - 1) && isNew.get(f + 1)) isNew.set(f, false);
  }
  return { isNew, d };
}

/** 由「新画面」标记生成 frameHold 段：相邻新帧间隔 k≥2 的连续段合并成 every=k 的段 */
export function holdSegments(isNew: Map<number, boolean>, a: number, b: number): { start: number; end: number; every: number }[] {
  const news = [...isNew.entries()].filter(([f, v]) => v && f >= a && f < b).map(([f]) => f).sort((x, y) => x - y);
  if (!news.length || news[0] !== a) news.unshift(a);
  const runs: { g: number; k: number }[] = news.map((g, i) => ({ g, k: (news[i + 1] ?? b) - g }));
  const segs: { start: number; end: number; every: number }[] = [];
  for (const r of runs) {
    if (r.k < 2) continue;
    const last = segs[segs.length - 1];
    if (last && last.end === r.g && last.every === r.k) last.end = r.g + r.k;
    else segs.push({ start: r.g, end: r.g + r.k, every: r.k });
  }
  return segs;
}

/**
 * 参考帧里的虹膜青色质心（全分辨率），以 shots 的 eyes 为搜索中心，半径 0.35×眼距，迭代两次。
 * 与页面里损失用的青色判据相同。任一只眼像素数不足时返回 null（闭眼、特殊眼、被遮挡）。
 */
export function detectRefEyes(buf: Buffer, guess: number[][], excl: number[][]): [[number, number], [number, number]] | null {
  const ed = Math.hypot(guess[1]![0]! - guess[0]![0]!, guess[1]![1]! - guess[0]![1]!);
  if (!(ed > 8)) return null;
  const out: [number, number][] = [];
  const inExcl = (x: number, y: number) => excl.some(([ex, ey, ew, eh]) => x >= ex! && x < ex! + ew! && y >= ey! && y < ey! + eh!);
  for (const g of guess) {
    let cx = g[0]!, cy = g[1]!, n = 0;
    for (let it = 0; it < 3; it++) {
      const rad = (it === 0 ? 0.35 : 0.25) * ed;
      let sx = 0, sy = 0;
      n = 0;
      for (let y = Math.max(0, Math.floor(cy - rad)); y <= Math.min(719, Math.ceil(cy + rad)); y++)
        for (let x = Math.max(0, Math.floor(cx - rad)); x <= Math.min(1279, Math.ceil(cx + rad)); x++) {
          if ((x - cx) ** 2 + (y - cy) ** 2 > rad * rad) continue;
          const o = (y * 1280 + x) * 3;
          const R = buf[o]!, G = buf[o + 1]!, B = buf[o + 2]!;
          if (G - R > 45 && B - R > 55 && G > 0.6 * B && B > 110 && !inExcl(x, y)) { sx += x + 0.5; sy += y + 0.5; n++; }
        }
      if (n < 0.008 * ed * ed) return null;
      cx = sx / n; cy = sy / n;
    }
    out.push([cx, cy]);
  }
  return out as [[number, number], [number, number]];
}
