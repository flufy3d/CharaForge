/*
 * 验证浏览器里参考视频的逐帧定位精度：
 *   对每个检查帧 f，在无头 Chromium 中 seek 到帧 f，读回 RGB；再用 ffmpeg 解码出 f−1、f、f+1 三帧，
 *   计算平均绝对误差（MAE）。正确时与 f 的 MAE 最小（仅有解码器/色彩转换的微小差异），且明显小于相邻帧。
 *
 *   node --experimental-strip-types apps/player/scripts/verify-video.ts [video-uri] [frames...]
 * 缺省视频 assets/video/DeepSeek-chan.mp4；检查帧包括片头片尾和 ffmpeg 场景检测得到的快切附近。
 */
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { openHarness, REPO } from './harness-node.ts';

const uri = process.argv[2] ?? 'assets/video/DeepSeek-chan.mp4';
const CUTS = [742, 1011, 1045, 1707, 1726, 2304, 2368, 2446, 2639, 3097, 3145, 3662, 3722];
const frames = process.argv.length > 3 ? process.argv.slice(3).map(Number) : [0, 1, 2, 30, 59, 60, 61, 1999, 3000, 3866, 3867, ...CUTS.flatMap((c) => [c - 1, c, c + 1])];
const FPS = 60;
const W = 1280, H = 720;

function ffmpegFrames(ns: number[]): Map<number, Buffer> {
  const sorted = [...new Set(ns)].filter((n) => n >= 0 && n < 3868).sort((a, b) => a - b);
  const expr = sorted.map((n) => `eq(n\\,${n})`).join('+');
  const out = execFileSync('ffmpeg', ['-v', 'error', '-i', join(REPO, 'assets/video/DeepSeek-chan.mp4'), '-vf', `select=${expr}`, '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 30 });
  const m = new Map<number, Buffer>();
  sorted.forEach((n, i) => m.set(n, out.subarray(i * W * H * 3, (i + 1) * W * H * 3)));
  return m;
}

const mae = (a: Uint8Array, b: Uint8Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i]! - b[i]!); return s / a.length; };

const ref = ffmpegFrames(frames.flatMap((f) => [f - 1, f, f + 1]));
const h = await openHarness({ expectGpu: null });
let bad = 0;
try {
  console.log(`视频：${uri}`);
  console.log('帧     MAE(f-1)  MAE(f)  MAE(f+1)  结果');
  for (const f of frames) {
    const r = await h.page.evaluate(([u, f, fps]) => (window as any).__charaforgeVideo.frameRGB(u, f, fps), [uri, f, FPS] as const);
    const got = Buffer.from(r.data, 'base64');
    const m = [f - 1, f, f + 1].map((n) => (ref.has(n) ? mae(got, ref.get(n)!) : NaN));
    const best = [f - 1, f, f + 1][m.reduce((bi, v, i) => (v < (m[bi] ?? Infinity) || Number.isNaN(m[bi]!) ? i : bi), 1)];
    // 静止镜头中相邻帧可能完全相同：只要 f 的误差不大于最小值（并列也算对）即为正确
    const ok = !(m[1]! > Math.min(...m.filter((x) => !Number.isNaN(x))) + 1e-9);
    if (!ok) bad++;
    console.log(`${String(f).padEnd(6)} ${m.map((x) => (Number.isNaN(x) ? '   -    ' : x.toFixed(3).padStart(8))).join('  ')}  ${ok ? 'OK' : `错位（最接近 ${best}）`}`);
  }
} finally {
  await h.close();
}
console.log(bad ? `失败：${bad}/${frames.length} 帧错位` : `全部 ${frames.length} 帧逐帧精确`);
process.exit(bad ? 1 : 0);
