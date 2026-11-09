/*
 * 播放时钟：以 Web Audio 的音频时间为准，当前帧 = floor(t × fps)。
 * 场景 audio 片段按 (start 帧, offset, duration, gain) 调度到 AudioContext 上。
 * 没有音频时退化为 performance.now()。
 */
import { asciiUrl } from '@charaforge/runtime';
import type { Scene } from '@charaforge/scene';

export class PlaybackClock {
  readonly fps: number;
  private ctx: AudioContext | null = null;
  private buffers = new Map<string, AudioBuffer>();
  private sources: AudioBufferSourceNode[] = [];
  private gains: { node: GainNode; gain: number }[] = [];
  private startCtxTime = 0;
  private startPerf = 0;
  private startSeconds = 0;
  playing = false;
  muted = false;
  private lastNow = 0;

  constructor(private readonly scene: Scene, private readonly resolve: (uri: string) => string) {
    this.fps = scene.timebase.fps;
  }

  get hasAudio(): boolean {
    return this.buffers.size > 0;
  }

  /** 解码场景用到的音频（第一次播放前调用；需要用户手势之后才能真正出声）。 */
  async load(): Promise<void> {
    const assets = this.scene.assets.audio ?? [];
    if (!assets.length || !(this.scene.audio ?? []).length) return;
    this.ctx = new AudioContext({ latencyHint: 'interactive' });
    await Promise.all(
      assets.map(async (a) => {
        const res = await fetch(asciiUrl(this.resolve(a.uri)));
        if (!res.ok) throw new Error(`音频加载失败：${a.uri}`);
        this.buffers.set(a.id, await this.ctx!.decodeAudioData(await res.arrayBuffer()));
      }),
    );
  }

  /** 当前场景时间（秒）。 */
  now(): number {
    if (!this.playing) return this.startSeconds;
    let t: number;
    if (this.ctx && this.hasAudio) {
      // AudioContext.currentTime 按音频缓冲块跳变（实测 10.7/21.3 ms 交替），直接用会让帧号不均匀前进、
      // 一部分 rAF 无新帧可画。getOutputTimestamp() 给出「此刻正被听到的音频时间」与对应的 performance 时刻，
      // 用它按 performance.now() 线性外推，得到平滑且已扣除输出延迟的音频时钟。
      const ts = this.ctx.getOutputTimestamp();
      let heard: number;
      if (ts.contextTime !== undefined && ts.performanceTime !== undefined && ts.contextTime > 0) {
        heard = ts.contextTime + (performance.now() - ts.performanceTime) / 1000;
      } else {
        heard = this.ctx.currentTime - (this.ctx.outputLatency || 0) - (this.ctx.baseLatency || 0);
      }
      t = this.startSeconds + Math.max(0, heard - this.startCtxTime);
    } else {
      t = this.startSeconds + (performance.now() - this.startPerf) / 1000;
    }
    // 单调：外推误差不能让时间倒退
    if (t < this.lastNow) t = this.lastNow;
    this.lastNow = t;
    return t;
  }

  frame(): number {
    return Math.floor(this.now() * this.fps + 1e-6);
  }

  async play(fromFrame: number): Promise<void> {
    this.stopSources();
    this.startSeconds = fromFrame / this.fps;
    this.lastNow = this.startSeconds;
    this.startPerf = performance.now();
    if (this.ctx && this.hasAudio) {
      await this.ctx.resume();
      // 预留一点调度余量，保证所有片段在同一时刻起播
      this.startCtxTime = this.ctx.currentTime + 0.05;
      // 片段在上下文时间 startCtxTime 开始；now() 用「正被听到的上下文时间」计算，所以画面与听到的声音对齐。
      for (const clip of this.scene.audio ?? []) {
        const buf = this.buffers.get(clip.audio);
        if (!buf) continue;
        const clipStart = clip.start / this.fps;
        const offset0 = clip.offset ?? 0;
        const dur0 = clip.duration ?? buf.duration - offset0;
        const into = this.startSeconds - clipStart; // 已经进入片段多少秒
        const when = this.startCtxTime + Math.max(0, -into);
        const offset = offset0 + Math.max(0, into);
        const dur = dur0 - Math.max(0, into);
        if (dur <= 0) continue;
        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        const gain = this.ctx.createGain();
        gain.gain.value = this.muted ? 0 : clip.gain ?? 1;
        src.connect(gain).connect(this.ctx.destination);
        src.start(when, offset, dur);
        this.sources.push(src);
        this.gains.push({ node: gain, gain: clip.gain ?? 1 });
      }
    }
    this.playing = true;
  }

  /** 静音只改增益，不影响时钟。 */
  setMuted(m: boolean): void {
    this.muted = m;
    for (const g of this.gains) g.node.gain.value = m ? 0 : g.gain;
  }

  pause(): number {
    const t = this.now();
    this.stopSources();
    this.playing = false;
    this.startSeconds = t;
    this.lastNow = t;
    return Math.floor(t * this.fps + 1e-6);
  }

  seek(frame: number): void {
    this.startSeconds = frame / this.fps;
    this.lastNow = this.startSeconds;
  }

  private stopSources(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* 已经停止 */
      }
    }
    this.sources = [];
    this.gains = [];
  }

  dispose(): void {
    this.stopSources();
    void this.ctx?.close();
  }
}
