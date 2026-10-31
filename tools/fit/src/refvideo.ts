/*
 * 参考视频解码：用 ffmpeg 把 [a, b) 帧解成 RGB24 原始数据缓存到 .data/fit/ref/，按偏移随机读取。
 * 颜色转换参数必须与 tools/qa/metric.py 完全一致（BT.709、TV range → 全范围 RGB、精确取整），
 * 这样拟合损失和验收指标看到的是同一份参考像素。
 */
import { spawnSync } from 'node:child_process';
import { closeSync, existsSync, mkdirSync, openSync, readSync, renameSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { REPO } from './paths.ts';

/** 参考视频；上限实验用 FIT_VIDEO 换成合成参考，FIT_VIDEO_OFFSET = 该视频第 0 帧对应的场景帧号 */
export const VIDEO = process.env.FIT_VIDEO ? process.env.FIT_VIDEO : join(REPO, 'assets/video/DeepSeek-chan.mp4');
const OFFSET = Number(process.env.FIT_VIDEO_OFFSET ?? 0);
export const W = 1280;
export const H = 720;
export const FRAME_BYTES = W * H * 3;

/** 与 metric.py 的 FFMPEG_VF 保持逐字一致 */
export function ffmpegArgs(a: number, b: number): string[] {
  a -= OFFSET;
  b -= OFFSET;
  const vf = `select='between(n\\,${a}\\,${b - 1})',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgb24`;
  return ['-v', 'error', '-nostdin', '-i', VIDEO, '-vf', vf, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'];
}

export class RefFrames {
  readonly a: number;
  readonly b: number;
  private fd: number;
  /** 缓存文件路径（RGB24 连续帧，第 0 帧 = 场景帧 a） */
  readonly file: string;
  constructor(a: number, b: number) {
    this.a = a;
    this.b = b;
    const dir = join(REPO, '.data/fit/ref');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, `${process.env.FIT_VIDEO ? 'synth_' : ''}${a}_${b}.rgb`);
    const want = (b - a) * FRAME_BYTES;
    if (!existsSync(file) || statSync(file).size !== want) {
      const tmp = `${file}.tmp`;
      const fd = openSync(tmp, 'w');
      const r = spawnSync('ffmpeg', ffmpegArgs(a, b), { stdio: ['ignore', fd, 'inherit'] });
      closeSync(fd);
      if (r.status !== 0) throw new Error(`ffmpeg 解码失败（${a}..${b}）`);
      if (statSync(tmp).size !== want) throw new Error(`解码帧数不对：${statSync(tmp).size / FRAME_BYTES} ≠ ${b - a}`);
      renameSync(tmp, file);
    }
    this.file = file;
    this.fd = openSync(file, 'r');
  }
  /** 帧 f 的 RGB24（行从上到下） */
  frame(f: number): Buffer {
    if (f < this.a || f >= this.b) throw new Error(`帧 ${f} 不在缓存区间 [${this.a}, ${this.b})`);
    const buf = Buffer.allocUnsafe(FRAME_BYTES);
    readSync(this.fd, buf, 0, FRAME_BYTES, (f - this.a) * FRAME_BYTES);
    return buf;
  }
  close(): void {
    closeSync(this.fd);
  }
}
