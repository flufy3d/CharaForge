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
  /** 当前渲染请求已等待的毫秒数（> 0：远距离 seek 后 Worker 还在补物理） */
  waitMs: number;
  muted: boolean;
  /** 至少有一帧上屏了 */
  ready: boolean;
  /** 后台预热进度（各图层的 in 帧） */
  prewarm: { done: number; total: number };
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
  private status: EngineStatus = { frame: 0, playing: false, fps: 0, renderMs: 0, steps: 0, videoFrame: null, videoExact: false, waitMs: 0, muted: false, ready: false, prewarm: { done: 0, total: 0 } };
  onStatus: (s: EngineStatus) => void = () => {};

  /** 求值 Worker 数（0 = 同步路径） */
  workers = 0;
  private busy = false;
  private busySince = 0;
  /** 当前 rAF 回调的时间戳（与 vsync 对齐），用于帧间隔统计 */
  private tickTs = 0;
  /** 帧时间统计（bench 脚本读取）：相邻两次上屏的间隔、显示帧号跳变 */
  readonly stats = { intervals: [] as number[], jumps: 0, dropped: 0, presented: 0, lastPresent: 0, lastFrame: -1, log: [] as unknown[], avLag: [] as number[] };

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
    this.stats.avLag = [];
  }

  /** 音频加载失败的原因（null = 正常或场景没有音频）；失败时退化为系统时钟、无声播放。 */
  audioError: string | null = null;

  async load(scene: string | object, onStage: (stage: string) => void = () => {}): Promise<Scene> {
    this.clock?.dispose();
    onStage('加载场景、模型与贴图，创建图层实例');
    this.scene = await this.runtime.loadScene(scene);
    this.clock = new PlaybackClock(this.scene, (u) => this.runtime.resolveAsset(u));
    onStage('解码音频');
    this.audioError = null;
    await this.clock.load().catch((e) => {
      console.warn('[player] 音频加载失败', e);
      this.audioError = String(e?.message ?? e);
    });
    this.frame = 0;
    this.dirty = true;
    return this.scene;
  }

  /*
   * 后台预热：全片场景有几十个图层，播放第一次走到某图层的 in 时，Worker 要做 90 帧物理预热（约 0.1～0.3 秒），
   * 主线程第一次画这个实例要建 GL 资源（约 0.15～0.4 秒），实测会卡 0.1～0.6 秒、一次丢几十帧。
   * 预热 = 把各图层的 in 帧离屏渲染一遍、不上屏；之后求值器在 in 处留有状态（游标 / 最近状态），播放到那里直接续上。
   * 全部预热约 13 秒，所以不阻塞加载：第一帧出来就可以播放，预热只在暂停且无事可做时一帧一帧进行；
   * 第一次预热时把所有 in 帧作为预取一起发给 Worker（物理预热在多个 Worker 上并行）。播放中不做主线程部分，
   * 只把前方 2 秒内将要出现的图层作为预取提前发给 Worker。
   * 画面只是帧号的函数（规范保证 seek 与顺序播放逐位相同），预热不改变任何一帧的结果，只是挪动开销。
   */
  private warmQueue: number[] = [];
  private warmTotal = 0;
  private warmKicked = false;
  private warming = false;
  /** 合成缓冲被预热帧覆盖了（播放中的对照模式会直接复用合成缓冲，需要重画） */
  private compStale = false;

  /** 开始后台预热（场景加载之后调用；再次调用会重新开始）。 */
  startPrewarm(): void {
    this.warmQueue = [...new Set(this.scene.layers.map((l) => l.in))].filter((f) => f > 0 && f < this.duration).sort((a, b) => a - b);
    this.warmTotal = this.warmQueue.length;
    this.warmKicked = false;
    this.status.prewarm = { done: 0, total: this.warmTotal };
  }

  private warmStep(): void {
    const f = this.warmQueue.shift()!;
    const prefetch = this.warmKicked ? [] : [...this.warmQueue];
    this.warmKicked = true;
    this.busy = true;
    this.warming = true;
    this.busySince = performance.now();
    this.runtime
      .renderFrameAsync(f, { prefetch })
      .catch((e) => console.warn('[player] 预热失败', f, e))
      .finally(() => {
        this.busy = false;
        this.warming = false;
        this.compStale = true;
        this.status.prewarm = { done: this.warmTotal - this.warmQueue.length, total: this.warmTotal };
        this.onStatus(this.status);
      });
  }

  /** 播放中：已经走过的 in 不用再预热；前方 2 秒内的 in 作为预取发给 Worker（只做 Worker 部分）。 */
  private warmAhead(f: number): number[] {
    if (!this.warmQueue.length) return [];
    const ahead = this.warmQueue.filter((x) => x > f + 4 && x <= f + 120);
    const before = this.warmQueue.length;
    this.warmQueue = this.warmQueue.filter((x) => x > f + 120);
    if (this.warmQueue.length !== before) this.status.prewarm = { done: this.warmTotal - this.warmQueue.length, total: this.warmTotal };
    return ahead;
  }

  /** 静音只改增益，时钟照常以音频时间为准。 */
  setMuted(m: boolean): void {
    this.clock.setMuted(m);
    this.status.muted = m;
    this.onStatus(this.status);
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
    if (this.compare !== 'off') this.playVideo();
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
    if (m === 'off') this.video?.pause();
    else if (this.clock.playing) this.playVideo(); // 播放中打开对照：视频也要跟着播，不然 tick 会每帧重新 seek 一个暂停的视频
    else void this.syncVideoExact();
  }

  /** 播放中让参考视频从当前帧起按自己的时钟播放（tick 里漂移 > 2 帧时重对齐）。 */
  private playVideo(): void {
    if (!this.video) return;
    this.video.currentTime = (this.frame + 0.5) / this.clock.fps;
    void this.video.play().catch(() => {});
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
    if (!this.scene) return;
    if (this.busy) {
      // 后台预热中、用户没有新的请求：不算等待
      if (this.warming && this.frame === this.rendered && !this.dirty && !this.clock.playing) return;
      // 远距离 seek 时 Worker 要从最近的物理快照逐帧补到目标帧：等待超过 250 ms 就报给 UI（每 250 ms 一次）
      const w = performance.now() - this.busySince;
      if (w > 250 && Math.floor(w / 250) !== Math.floor(this.status.waitMs / 250)) {
        this.status.waitMs = w;
        this.onStatus(this.status);
      }
      return;
    }
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
      if (!this.clock.playing && this.warmQueue.length) {
        this.warmStep();
        return;
      }
      this.emit(0, 0);
      return;
    }
    const f = this.frame;
    const tickTs = this.tickTs;
    const needRender = f !== this.rendered || this.dirty || this.compStale;
    this.dirty = false;
    this.compStale = false;
    this.busy = true;
    this.busySince = performance.now();
    // Worker 路径：拿到 f 的数据后立即预取 f+1（播放时下一帧大概率就是它），与本帧绘制重叠
    // 播放时预取后面几帧（过期的预取在发给 Worker 之前就会被丢弃，不浪费算力），以及前方 2 秒内新图层的 in 帧
    const job = needRender
      ? this.runtime.renderFrameAsync(f, { prefetch: this.clock.playing ? [f + 1, f + 2, f + 3, f + 4, ...this.warmAhead(f)] : [] })
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
        // 音画同步：上屏时刻音频时钟所在的帧 − 上屏的帧（> 0 表示画面落后于声音，单位帧）
        if (this.clock.playing && s.avLag.length < 20000) s.avLag.push(this.clock.now() * this.clock.fps - f);
        if (st && s.log.length < 2000) s.log.push({ f, t: Math.round(now), ms: +st.ms.toFixed(1), wait: +(st.waitMs ?? 0).toFixed(1), wk: +(st.workerMs ?? 0).toFixed(1), steps: st.steps });
        this.emit(st?.ms ?? 0, st?.steps ?? 0);
      })
      .catch((e) => console.error('[player] 渲染失败', e))
      .finally(() => {
        this.busy = false;
        if (this.status.waitMs) {
          this.status.waitMs = 0;
          this.onStatus(this.status);
        }
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
    this.status.ready = this.rendered >= 0;
    this.onStatus(this.status);
  }
}
