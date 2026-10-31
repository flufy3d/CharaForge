/*
 * 播放器引擎（与 UI 无关）：运行时 + 时钟 + 参考视频 + 渲染循环。
 *
 * 画面只是帧号的函数：每个 rAF 取 clock.frame()，若与上一帧不同就 renderFrame(frame)。
 * 掉帧（渲染慢于 60fps）时会跳过若干帧的绘制，但 LayerEvaluator 会把中间帧的物理逐帧补上
 * （只做 a～f 步，不做 Core update 和绘制），所以画面与「逐帧渲染」时的同一帧逐位相同。
 */
import { CharaForgeRuntime } from '@charaforge/runtime';
import type { Marker, Scene } from '@charaforge/scene';
import { Texture, VideoSource } from 'pixi.js';
import { PlaybackClock } from './clock.ts';
import { loadVideo, seekVideoFrame, timeToFrame } from './videoSync.ts';

export type CompareMode = 'off' | 'side' | 'overlay' | 'diff';

export interface EngineStatus {
  frame: number;
  playing: boolean;
  fps: number;
  renderMs: number;
  steps: number;
  videoFrame: number | null;
  videoExact: boolean;
}

export class PlayerEngine {
  runtime!: CharaForgeRuntime;
  scene!: Scene;
  clock!: PlaybackClock;
  frame = 0;
  compare: CompareMode = 'off';
  overlayMix = 0.5;
  diffGain = 4;
  video: HTMLVideoElement | null = null;
  private videoTex: Texture | null = null;
  private videoSource: VideoSource | null = null;
  private videoExact = false;
  private videoBusy = false;
  private rendered = -1;
  private dirty = true;
  private raf = 0;
  private fpsCount = 0;
  private fpsT0 = performance.now();
  private status: EngineStatus = { frame: 0, playing: false, fps: 0, renderMs: 0, steps: 0, videoFrame: null, videoExact: false };
  onStatus: (s: EngineStatus) => void = () => {};

  /** 求值 Worker 数（0 = 同步路径） */
  workers = 0;
  private busy = false;
  /** 当前 rAF 回调的时间戳（与 vsync 对齐），用于帧间隔统计 */
  private tickTs = 0;
  /** 帧时间统计（bench 脚本读取）：相邻两次上屏的间隔、显示帧号跳变 */
  readonly stats = { intervals: [] as number[], jumps: 0, dropped: 0, presented: 0, lastPresent: 0, lastFrame: -1, log: [] as unknown[] };

  async init(canvas: HTMLCanvasElement, workers: number): Promise<void> {
    this.workers = workers;
    this.runtime = await CharaForgeRuntime.create({ assetBase: '/assets/', canvas, workers });
  }

  resetStats(): void {
    this.stats.intervals = [];
    this.stats.jumps = 0;
    this.stats.dropped = 0;
    this.stats.presented = 0;
    this.stats.lastPresent = 0;
    this.stats.lastFrame = -1;
    this.stats.log = [];
  }

  async load(scene: string | object): Promise<Scene> {
    this.clock?.dispose();
    this.scene = await this.runtime.loadScene(scene);
    this.clock = new PlaybackClock(this.scene, (u) => this.runtime.resolveAsset(u));
    await this.clock.load().catch((e) => console.warn('[player] 音频加载失败', e));
    this.frame = 0;
    this.dirty = true;
    return this.scene;
  }

  /** 加载参考视频（仓库相对路径，缺省 assets/video/DeepSeek-chan.mp4）。 */
  async loadVideo(uri: string): Promise<void> {
    this.video = await loadVideo(`/${uri.replace(/^\/+/, '')}`);
    this.videoSource = new VideoSource({ resource: this.video, autoPlay: false });
    this.videoSource.autoUpdate = false;
    this.videoTex = new Texture({ source: this.videoSource });
    this.dirty = true;
  }

  start(): void {
    const loop = (ts: number) => {
      this.raf = requestAnimationFrame(loop);
      this.tickTs = ts;
      this.tick();
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    cancelAnimationFrame(this.raf);
  }

  get duration(): number {
    return this.scene.duration;
  }

  async togglePlay(): Promise<void> {
    if (this.clock.playing) this.pause();
    else await this.play();
  }

  async play(): Promise<void> {
    if (this.frame >= this.duration - 1) this.frame = 0;
    await this.clock.play(this.frame);
    if (this.video && this.compare !== 'off') {
      this.video.currentTime = (this.frame + 0.5) / this.clock.fps;
      void this.video.play().catch(() => {});
    }
  }

  pause(): void {
    this.frame = Math.min(this.duration - 1, this.clock.pause());
    this.video?.pause();
    this.dirty = true;
    void this.syncVideoExact();
  }

  seek(frame: number): void {
    const f = Math.max(0, Math.min(this.duration - 1, Math.floor(frame)));
    const wasPlaying = this.clock.playing;
    if (wasPlaying) this.clock.pause();
    this.frame = f;
    this.clock.seek(f);
    this.dirty = true;
    if (wasPlaying) void this.play();
    else void this.syncVideoExact();
  }

  step(delta: number): void {
    this.seek(this.frame + delta);
  }

  setCompare(m: CompareMode): void {
    this.compare = m;
    this.dirty = true;
    if (m !== 'off') void this.syncVideoExact();
    else this.video?.pause();
  }

  setOverlay(mix: number, gain: number): void {
    this.overlayMix = mix;
    this.diffGain = gain;
    this.dirty = true;
  }

  currentMarker(): Marker | undefined {
    const ms = (this.scene.markers ?? []).filter((m) => this.frame >= m.start && this.frame < (m.end ?? m.start + 1));
    return ms[ms.length - 1];
  }

  /** 暂停时把参考视频精确定位到当前帧 */
  private async syncVideoExact(): Promise<void> {
    if (!this.video || this.compare === 'off' || this.videoBusy) return;
    this.videoBusy = true;
    try {
      let target = -1;
      while (target !== this.frame && !this.clock.playing) {
        target = this.frame;
        this.videoExact = false;
        await seekVideoFrame(this.video, target, this.clock.fps);
      }
      this.videoExact = !this.clock.playing && target === this.frame;
      this.videoSource?.update();
      this.dirty = true;
    } finally {
      this.videoBusy = false;
    }
  }

  private tick(): void {
    if (!this.scene || this.busy) return;
    if (this.clock.playing) {
      const f = this.clock.frame();
      if (f >= this.duration) {
        this.frame = this.duration - 1;
        this.pause();
      } else this.frame = f;
      if (this.video && this.compare !== 'off') {
        // 播放中视频按自己的时钟走；漂移超过 2 帧时重新对齐（精确对齐只在暂停/逐帧时保证）
        const vf = timeToFrame(this.video.currentTime, this.clock.fps);
        if (Math.abs(vf - this.frame) > 2) this.video.currentTime = (this.frame + 0.5) / this.clock.fps;
        this.videoSource?.update();
        this.videoExact = false;
      }
    }
    const comparing = this.compare === 'overlay' || this.compare === 'diff';
    if (this.frame === this.rendered && !this.dirty && !(comparing && this.clock.playing)) {
      this.emit(0, 0);
      return;
    }
    const f = this.frame;
    const tickTs = this.tickTs;
    const needRender = f !== this.rendered || this.dirty;
    this.dirty = false;
    this.busy = true;
    // Worker 路径：拿到 f 的数据后立即预取 f+1（播放时下一帧大概率就是它），与本帧绘制重叠
    // 播放时预取后面几帧（过期的预取在发给 Worker 之前就会被丢弃，不浪费算力）
    const job = needRender
      ? this.runtime.renderFrameAsync(f, { prefetch: this.clock.playing ? [f + 1, f + 2, f + 3, f + 4] : [] })
      : Promise.resolve(null);
    job
      .then((st) => {
        if (comparing && this.videoTex) {
          this.runtime.compositor.presentDebug(this.videoTex, this.compare === 'diff' ? 'diff' : 'overlay', this.overlayMix, this.diffGain);
        } else {
          this.runtime.present();
        }
        this.rendered = f;
        this.fpsCount++;
        const now = performance.now();
        const s = this.stats;
        // 用发起本次渲染的 rAF 时间戳（vsync 对齐）计算上屏间隔：16.7 ms 的整数倍，>16.7 即掉帧
        // 结果晚于下一个 vsync 才回来时，实际上屏发生在之后的 rAF 周期，按完成时刻归到对应周期
        const late = Math.max(0, Math.floor((now - tickTs) / (1000 / 60)));
        const ts = tickTs + late * (1000 / 60);
        if (s.lastPresent) s.intervals.push(ts - s.lastPresent);
        if (s.lastFrame >= 0 && f - s.lastFrame > 1) {
          s.jumps++;
          s.dropped += f - s.lastFrame - 1;
        }
        s.lastPresent = ts;
        s.lastFrame = f;
        s.presented++;
        if (st && s.log.length < 2000) s.log.push({ f, t: Math.round(now), ms: +st.ms.toFixed(1), wait: +(st.waitMs ?? 0).toFixed(1), wk: +(st.workerMs ?? 0).toFixed(1), steps: st.steps });
        this.emit(st?.ms ?? 0, st?.steps ?? 0);
      })
      .catch((e) => console.error('[player] 渲染失败', e))
      .finally(() => {
        this.busy = false;
      });
  }

  private emit(ms: number, steps: number): void {
    const now = performance.now();
    if (now - this.fpsT0 >= 500) {
      this.status.fps = (this.fpsCount * 1000) / (now - this.fpsT0);
      this.fpsCount = 0;
      this.fpsT0 = now;
    }
    if (ms) this.status.renderMs = this.status.renderMs * 0.9 + ms * 0.1;
    if (steps) this.status.steps = steps;
    this.status.frame = this.frame;
    this.status.playing = this.clock.playing;
    this.status.videoFrame = this.video ? timeToFrame(this.video.currentTime, this.clock.fps) : null;
    this.status.videoExact = this.videoExact;
    this.onStatus(this.status);
  }
}
