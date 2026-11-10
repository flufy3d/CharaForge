/*
 * 播放器引擎（与 UI 无关）：运行时 + 时钟 + 参考视频 + 渲染循环。
 *
 * 画面只是帧号的函数：每个 rAF 取 clock.frame()，若与上一帧不同就 renderFrame(frame)。
 * 掉帧（渲染慢于 60fps）时会跳过若干帧的绘制，但 LayerEvaluator 会把中间帧的物理逐帧补上
 * （只做 a～f 步，不做 Core update 和绘制），所以画面与「逐帧渲染」时的同一帧逐位相同。
 */
import { CharaForgeRuntime } from '@charaforge/runtime';
import type { EditEffect, Marker, Scene } from '@charaforge/scene';
import { Texture, VideoSource } from 'pixi.js';
import { PlaybackClock } from './clock.ts';
import { CORE_PATH, pageUrl } from './util.ts';
import { loadVideo, seekVideoFrame, timeToFrame } from './videoSync.ts';

export type CompareMode = 'off' | 'side' | 'overlay' | 'diff';

/** PlaybackClock 依赖的场景音频/时基快照：结构编辑后据此判断是否需要重建时钟。 */
function audioSignature(scene: Scene): string {
  return JSON.stringify({
    fps: scene.timebase.fps,
    assets: (scene.assets.audio ?? []).map((a) => [a.id, a.uri]),
    clips: scene.audio ?? [],
  });
}

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
  /** 新图层预热进度（各图层 in 帧的 Worker 预取） */
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
    // 资源与 Core 的 URL 都相对页面地址解析（页面可能在反向代理的路径前缀下），绝对 URL 再交给运行时和 Worker
    this.runtime = await CharaForgeRuntime.create({ assetBase: pageUrl('assets/'), coreUrl: pageUrl(CORE_PATH), canvas, workers });
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
  /** 当前 PlaybackClock 对应的音频/时基快照；结构编辑后据此判断是否需要重建时钟。 */
  private audioState = '';
  /** 当前求值器与 PlaybackClock 使用的 fps（fps 改变时必须重建求值器）。 */
  private appliedFps = 0;

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
    this.audioState = audioSignature(this.scene);
    this.appliedFps = this.scene.timebase.fps;
    this.frame = 0;
    this.dirty = true;
    return this.scene;
  }

  /**
   * 重建播放时钟（音频资产或时基改变后）。保留当前帧、播放状态与静音状态。
   * 旧时钟的 AudioContext 被关闭，新时钟重新解码当前场景用到的音频。
   */
  async reloadClock(): Promise<void> {
    if (!this.scene) return;
    const frame = this.frame;
    const muted = this.clock?.muted ?? this.status.muted;
    const wasPlaying = this.clock?.playing ?? false;
    this.clock?.dispose();
    const clock = new PlaybackClock(this.scene, (u) => this.runtime.resolveAsset(u));
    clock.setMuted(muted);
    this.clock = clock;
    this.audioError = null;
    await clock.load().catch((e) => {
      console.warn('[player] 音频加载失败', e);
      this.audioError = String((e as Error)?.message ?? e);
    });
    clock.seek(frame);
    this.status.muted = muted;
    this.audioState = audioSignature(this.scene);
    this.appliedFps = this.scene.timebase.fps;
    if (wasPlaying) await clock.play(frame);
    this.dirty = true;
  }

  /**
   * 编辑器把一轮场景修改应用到正在运行的引擎（不刷新页面）。
   * effects 与 scene 包的 EditEffect 对应：
   *   - structure：runtime.syncScene 整体同步（内部已重建求值器并刷新 Worker）；
   *   - evaluator：重建主线程求值器并刷新 Worker 场景副本（参数 / motion / expression 修改）；
   *   - workers：只刷新 Worker 场景副本；
   *   - audio / clock：重建播放时钟；结构编辑若改了时基或音频也自动重建。
   * 当前帧保留（越界时夹到 [0, duration−1]），下一帧重新渲染。
   */
  async applySceneEdits(scene: Scene, effects: readonly string[]): Promise<void> {
    const has = (e: EditEffect) => effects.includes(e);
    const structural = has('structure');
    const prevAudio = this.audioState;
    if (structural) {
      // syncScene 返回解析后的场景（与传入的是同一对象，URL 加载时除外）
      this.scene = await this.runtime.syncScene(scene);
    } else {
      // 非结构编辑假定传入的就是 runtime 当前持有的同一个对象。
      if (scene !== this.scene) this.scene = scene;
      if (has('evaluator')) await this.runtime.rebuildEvaluators();
      else if (has('workers')) await this.runtime.reloadWorkers();
    }

    // fps 决定 LayerEvaluator 的 dt 与 warmup 步长：只改时钟不够，求值器也要重建。
    if (!structural && !has('evaluator') && this.scene.timebase.fps !== this.appliedFps) {
      await this.runtime.rebuildEvaluators();
    }

    this.frame = Math.max(0, Math.min(this.duration - 1, this.frame));
    const clockChanged = has('audio') || has('clock') || audioSignature(this.scene) !== prevAudio;
    if (clockChanged) await this.reloadClock();
    this.dirty = true;
  }

  /*
   * 新图层的预热（只在 Worker 上做，不阻塞主线程、不改变像素）。
   * 播放第一次走到某图层的 in 时，Worker 要从 in − warmup 起把物理逐帧推进（90 帧，约 0.05～0.3 秒）；
   * 不提前做，这一帧等不到结果、会丢几帧。所以：
   *   1. 场景加载后把所有图层的 in 帧一次性作为预取交给 Worker（runtime.prefetch：按帧号排队，不抢播放所需的帧）；
   *   2. 播放中，前方 3 秒内将出现的图层的 in 帧每帧都放进预取列表（同一请求只发一次；往回 seek 之后同样生效）。
   * 主线程第一次画某个实例以前也要卡 0.1～0.7 秒（每个实例约 1000 个 GL 缓冲的创建，WSL 的 Mesa d3d12 下很慢），
   * 已在 glProxy.ts 改为同一模型的实例共用静态缓冲，不再需要在主线程上预热。
   * 画面只是帧号的函数（规范保证 seek 与顺序播放逐位相同），预取只是挪动开销。
   */
  private layerIns: number[] = [];
  private lookahead = false;

  /** 开始预热（场景加载之后调用）。 */
  startPrewarm(): void {
    this.layerIns = [...new Set(this.scene.layers.map((l) => l.in))].filter((f) => f > 0 && f < this.duration).sort((a, b) => a - b);
    this.lookahead = true;
    const total = this.layerIns.length;
    let done = 0;
    this.status.prewarm = { done, total };
    for (const f of this.layerIns) {
      void this.runtime.prefetch([f]).then(() => {
        this.status.prewarm = { done: ++done, total };
        // 预热进度回调可能在 seek 之后才返回；同步一下当前帧，避免编辑器/播放器状态栏被旧帧号覆盖。
        this.status.frame = this.frame;
        this.status.playing = this.clock.playing;
        if (!this.clock.playing || done === total) this.onStatus(this.status);
      });
    }
  }

  /** 播放中：前方 3 秒内将出现的图层的 in 帧。 */
  private warmAhead(f: number): number[] {
    if (!this.lookahead) return [];
    return this.layerIns.filter((x) => x > f + 4 && x <= f + 180);
  }

  /** 静音只改增益，时钟照常以音频时间为准。 */
  setMuted(m: boolean): void {
    this.clock.setMuted(m);
    this.status.muted = m;
    this.onStatus(this.status);
  }

  /** 加载参考视频（仓库相对路径，缺省 assets/video/DeepSeek-chan.mp4）。 */
  async loadVideo(uri: string): Promise<void> {
    this.video = await loadVideo(pageUrl(uri));
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

  /** 编辑器修改了场景对象后调用：下一帧重新渲染，不改变当前帧号。 */
  redraw(): void {
    this.dirty = true;
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
      this.emit(0, 0);
      return;
    }
    const f = this.frame;
    const tickTs = this.tickTs;
    const needRender = f !== this.rendered || this.dirty;
    this.dirty = false;
    this.busy = true;
    this.busySince = performance.now();
    // Worker 路径：拿到 f 的数据后立即预取 f+1（播放时下一帧大概率就是它），与本帧绘制重叠
    // 播放时预取后面几帧（过期的预取在发给 Worker 之前就会被丢弃，不浪费算力），以及前方 3 秒内新图层的 in 帧
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
