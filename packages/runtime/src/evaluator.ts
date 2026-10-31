/*
 * 图层求值器：严格按 scene.ts 规范第 3、4 条推进一个模型图层。
 *
 * 单帧 step(g, inputFrame)：
 *   a. resetPose
 *   b. motion 片段（数组顺序）
 *   c. expression 片段（数组顺序）
 *   d. stage=prePhysics 参数轨（数组顺序）   → 外部覆盖 pre（拟合器钩子）
 *   e. 物理前进 1/fps
 *   f. stage=postPhysics 参数轨（数组顺序）  → 外部覆盖 post
 *   g. 部件不透明度轨 → Core update（只在要绘制的那一帧做）
 *
 * 状态定义：S(g) = 从 start = in − warmup 开始逐帧 step 到 g 的物理状态；预热帧 (g < in) 使用帧 in 的输入。
 * S(start − 1) = 物理初始状态。motion/expression 无状态（见 motion.ts），所以 S 只含物理。
 *
 * 访问任意帧 f 时：从「≤ f−1 的最近已知规范状态」出发（当前游标 / 上一步之前的快照 / 检查点 / 初始状态），
 * 恢复后逐帧重放到 f−1，再 step(f)。顺序播放时就是增量推进一帧；seek 时是检查点 + 重放。
 * 两条路径执行的是完全相同的浮点运算序列，因此结果逐位相同（测试 determinism.test.ts 验证）。
 *
 * 一次性覆盖（overrides）只作用于被请求的那一帧，不进入规范状态链：带覆盖 step 之前会先保存 S(f−1)，
 * 之后的访问从 S(f−1) 出发，不受覆盖影响。
 */
import { sample, sampleCurve, type Animatable, type ModelLayer, type ParamTrack } from '@charaforge/scene';
import type { CubismExpressionMotion } from '@cubism/framework/motion/cubismexpressionmotion';
import type { CubismMotion } from '@cubism/framework/motion/cubismmotion';
import type { InstanceState, ModelInstance } from './instance.ts';
import { applyExpressionClip, applyMotionClip, fileFadeIn, fileFadeOut, motionDuration } from './motion.ts';

/** 外部参数覆盖：pre 在规范 d 步之后、e 步之前写入；post 在 f 步之后写入。值按 set 语义。 */
export interface ParamOverrides {
  pre?: Record<string, number>;
  post?: Record<string, number>;
  /** 部件不透明度覆盖（g 步之后） */
  parts?: Record<string, number>;
}

export const CHECKPOINT_INTERVAL = 30;
/** 最近若干帧的规范状态（环形缓存），让小幅回退（逐帧后退、Worker 乱序请求）不必从检查点重放。 */
export const RECENT_STATES = 64;

interface ResolvedMotionClip {
  motion: CubismMotion;
  start: number;
  /** 结束（片段时间，秒）；undefined = 无结束 */
  endTime?: number;
  offset: number;
  speed: number;
  loop: boolean;
  fadeIn: number;
  fadeOut: number;
  weight: Animatable<number> | undefined;
}

interface ResolvedExprClip {
  expr: CubismExpressionMotion;
  start: number;
  endTime?: number;
  fadeIn: number;
  fadeOut: number;
  weight: ResolvedMotionClip['weight'];
}

interface ResolvedTrack {
  index: number;
  track: ParamTrack;
  start: number;
  end: number;
  mode: 'set' | 'add' | 'multiply';
}

export interface LayerStateToken {
  cursor: number | null;
  clean: boolean;
  state: InstanceState;
  prev: { frame: number; state: InstanceState } | null;
}

export class LayerEvaluator {
  readonly layer: ModelLayer;
  readonly instance: ModelInstance;
  readonly fps: number;
  readonly dt: number;
  readonly start: number;
  private readonly motions: ResolvedMotionClip[];
  private readonly exprs: ResolvedExprClip[];
  private readonly pre: ResolvedTrack[];
  private readonly post: ResolvedTrack[];
  private readonly parts: { index: number; track: NonNullable<ModelLayer['partOpacity']>[number] }[];
  private readonly physicsOn: boolean;
  private readonly initialState: InstanceState;
  private readonly checkpoints = new Map<number, InstanceState>();
  private readonly recent = new Map<number, InstanceState>();

  /** 实例当前持有的是哪一帧 step 之后的状态；null = 初始状态（S(start−1)）。 */
  private cursor: number | null = null;
  /** 当前游标状态是否为规范状态（没有被一次性覆盖污染）。 */
  private clean = true;
  /** 最近一次 step 之前的规范状态 S(cursor−1)。 */
  private prev: { frame: number; state: InstanceState } | null = null;
  /**
   * 游标帧完成到哪一级：0 = 只有物理状态（a～f）；1 = 参数与部件不透明度就绪（含 g 步的部件轨与覆盖）；
   * 2 = Core 也已 update，可以直接绘制。restoreToken 之后为 0。
   */
  private stage: 0 | 1 | 2 = 0;
  /** 统计：最近一次 prepare 实际 step 的帧数。 */
  lastSteps = 0;
  /** 统计：最近一次 prepare 是否做了 Core update。 */
  lastUpdated = false;

  constructor(
    layer: ModelLayer,
    instance: ModelInstance,
    fps: number,
    resolveMotion: (id: string) => CubismMotion,
    resolveExpr: (id: string) => CubismExpressionMotion,
  ) {
    this.layer = layer;
    this.instance = instance;
    this.fps = fps;
    this.dt = 1 / fps;
    this.physicsOn = (layer.physics?.enabled ?? true) && instance.physics !== null;
    this.start = layer.in - (this.physicsOn ? layer.physics?.warmup ?? 0 : 0);

    this.motions = (layer.motions ?? []).map((c) => {
      const motion = resolveMotion(c.motion);
      const speed = c.speed ?? 1;
      const offset = c.offset ?? 0;
      const loop = c.loop ?? false;
      let endTime: number | undefined;
      if (c.end !== undefined) endTime = (c.end - c.start) / fps;
      else if (!loop) endTime = Math.max(0, (motionDuration(motion) - offset) / speed);
      return {
        motion, start: c.start, endTime, offset, speed, loop,
        fadeIn: c.fadeIn ?? fileFadeIn(motion),
        fadeOut: c.fadeOut ?? fileFadeOut(motion),
        weight: c.weight,
      };
    });
    this.exprs = (layer.expressions ?? []).map((c) => {
      const expr = resolveExpr(c.expression);
      return {
        expr, start: c.start,
        endTime: c.end !== undefined ? (c.end - c.start) / fps : undefined,
        fadeIn: c.fadeIn ?? fileFadeIn(expr),
        fadeOut: c.fadeOut ?? fileFadeOut(expr),
        weight: c.weight,
      };
    });
    const tracks = (layer.params ?? []).map((track) => ({
      index: instance.paramIndexOf(track.id),
      track,
      start: track.range?.[0] ?? layer.in,
      end: track.range?.[1] ?? layer.out,
      mode: track.mode ?? 'set',
    })).filter((t) => t.index >= 0);
    this.pre = tracks.filter((t) => (t.track.stage ?? 'prePhysics') === 'prePhysics');
    this.post = tracks.filter((t) => t.track.stage === 'postPhysics');
    this.parts = (layer.partOpacity ?? []).map((track) => ({ index: instance.partIndexOf(track.id), track })).filter((p) => p.index >= 0);

    instance.resetPhysics();
    if (this.physicsOn && layer.physics?.stabilize) {
      // physics.stabilize：预热之前用帧 in 的输入（a～d）把物理置于静止平衡态（Cubism stabilization）
      this.evalInputs(layer.in);
      instance.stabilizePhysics();
    }
    this.initialState = instance.saveState();
  }

  isActive(frame: number): boolean {
    return frame >= this.layer.in && frame < this.layer.out;
  }

  /**
   * 让实例处于帧 f 的求值结果：
   *   'draw'    （缺省）完成 g 步（部件不透明度）并 Core update，可以直接绘制；
   *   'params'  完成到参数与部件不透明度（不做 Core update）——Worker 路径把它们交给 Worker 去 update；
   *   'advance' 只把物理状态推进到 S(f)（不可见图层：保持状态链连续，不做顶点更新）。
   * 兼容旧签名：第三个参数为 boolean 时 true = 'draw'，false = 'advance'。
   */
  prepare(frame: number, overrides?: ParamOverrides, mode: 'draw' | 'params' | 'advance' | boolean = 'draw'): void {
    const m = mode === true ? 'draw' : mode === false ? 'advance' : mode;
    const need = m === 'draw' ? 2 : m === 'params' ? 1 : 0;
    const f = Math.floor(frame);
    const hasOv = !!overrides && !!(overrides.pre || overrides.post || overrides.parts);
    this.lastSteps = 0;
    this.lastUpdated = false;
    if (this.cursor === f && this.clean && !hasOv) {
      if (this.stage >= need) return;
      if (this.stage === 1 && need === 2) {
        this.instance.update();
        this.stage = 2;
        this.lastUpdated = true;
        return;
      }
    }

    // 找 ≤ f−1 的最近规范状态
    const target = f - 1;
    let baseFrame = this.start - 1;
    let baseState = this.initialState as InstanceState | null;
    const consider = (fr: number, st: InstanceState | null) => {
      if (fr <= target && fr > baseFrame) { baseFrame = fr; baseState = st; }
    };
    if (this.cursor !== null && this.clean) consider(this.cursor, null); // null = 实例当前状态，无需恢复
    if (this.prev) consider(this.prev.frame, this.prev.state);
    const r = this.recent.get(target);
    if (r) consider(target, r);
    // 检查点：从 target 向下取最近的
    const cpFrame = Math.floor((target - this.start) / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL + this.start + CHECKPOINT_INTERVAL - 1;
    for (let c = cpFrame; c > baseFrame; c -= CHECKPOINT_INTERVAL) {
      const s = this.checkpoints.get(c);
      if (s) { consider(c, s); break; }
    }
    if (baseState) this.instance.restoreState(baseState);

    for (let g = baseFrame + 1; g <= target; g++) {
      this.step(g, undefined, 0);
      const st = this.instance.saveState();
      if ((g - this.start + 1) % CHECKPOINT_INTERVAL === 0 && !this.checkpoints.has(g)) this.checkpoints.set(g, st);
      this.remember(g, st);
    }
    const prevState = baseFrame === target && baseState ? baseState : this.instance.saveState();
    this.prev = { frame: target, state: prevState };
    this.remember(target, prevState);
    this.step(f, hasOv ? overrides : undefined, need);
    if (!hasOv) {
      // 规范状态 S(f) 也记下来（检查点 + 最近帧），之后回退访问不必从头重放
      const st = this.instance.saveState();
      if ((f - this.start + 1) % CHECKPOINT_INTERVAL === 0 && !this.checkpoints.has(f)) this.checkpoints.set(f, st);
      this.remember(f, st);
    }
    this.cursor = f;
    this.clean = !hasOv;
    this.stage = need as 0 | 1 | 2;
    this.lastUpdated = need === 2;
  }

  private remember(frame: number, st: InstanceState): void {
    if (this.recent.has(frame)) return;
    this.recent.set(frame, st);
    if (this.recent.size > RECENT_STATES) {
      // 删除离当前最远的
      let far = frame, d = -1;
      for (const k of this.recent.keys()) if (Math.abs(k - frame) > d) { d = Math.abs(k - frame); far = k; }
      this.recent.delete(far);
    }
  }

  /**
   * Worker 路径：主线程实例刚导入了帧 g 的 Core 动态数据（参数、部件、顶点全部被覆盖）。
   * 若游标正是 g，实例现在就是 g 的完整结果（stage 2）；否则参数数组已不对应游标帧（stage 0，物理状态不受影响）。
   */
  notifyImported(g: number): void {
    this.stage = this.cursor === g ? (this.clean ? 2 : this.stage) : 0;
  }

  /** 规范 a～d（不含覆盖），用于 stabilize 的输入。 */
  private evalInputs(f: number): void {
    const inst = this.instance;
    inst.resetPose();
    this.applyClips(f);
    this.applyTracks(this.pre, f);
  }

  /** 单帧求值（规范 a～g）。g < in 时为预热帧，输入取帧 in。 */
  private step(g: number, ov: ParamOverrides | undefined, level: number): void {
    this.lastSteps++;
    const inst = this.instance;
    const f = Math.max(g, this.layer.in); // 输入帧
    // a
    inst.resetPose();
    this.applyClips(f);
    // d
    this.applyTracks(this.pre, f);
    if (ov?.pre) inst.setParams(ov.pre);
    // e
    if (this.physicsOn) inst.stepPhysics(this.dt);
    // f
    this.applyTracks(this.post, f);
    if (ov?.post) inst.setParams(ov.post);
    if (level < 1) return;
    // g
    for (const p of this.parts) inst.setPartOpacity(p.track.id, sampleCurve(p.track.curve, f));
    if (ov?.parts) for (const id in ov.parts) inst.setPartOpacity(id, ov.parts[id]!);
    if (level >= 2) inst.update();
  }

  /** 规范 b、c：motion 与 expression 片段。 */
  private applyClips(f: number): void {
    const model = this.instance.model;
    // b
    for (const c of this.motions) {
      if (f < c.start) continue;
      const elapsed = (f - c.start) / this.fps;
      if (c.endTime !== undefined && elapsed >= c.endTime) continue;
      const weight = sample(c.weight, f, 1);
      applyMotionClip(model, c.motion, { elapsed, endTime: c.endTime, fadeIn: c.fadeIn, fadeOut: c.fadeOut, weight }, c.offset + elapsed * c.speed, c.loop);
    }
    // c
    for (const c of this.exprs) {
      if (f < c.start) continue;
      const elapsed = (f - c.start) / this.fps;
      if (c.endTime !== undefined && elapsed >= c.endTime) continue;
      const weight = sample(c.weight, f, 1);
      applyExpressionClip(model, c.expr, { elapsed, endTime: c.endTime, fadeIn: c.fadeIn, fadeOut: c.fadeOut, weight });
    }
  }

  private applyTracks(tracks: ResolvedTrack[], f: number): void {
    for (const t of tracks) {
      if (f < t.start || f >= t.end) continue;
      const w = sample(t.track.weight, f, 1);
      this.instance.setParamByIndex(t.index, sampleCurve(t.track.curve, f), t.mode, w);
    }
  }

  /** 丢弃检查点与游标（场景被修改后调用）。 */
  invalidate(): void {
    this.checkpoints.clear();
    this.recent.clear();
    this.cursor = null;
    this.clean = true;
    this.prev = null;
    this.stage = 0;
    this.instance.restoreState(this.initialState);
  }

  saveToken(): LayerStateToken {
    return { cursor: this.cursor, clean: this.clean, state: this.instance.saveState(), prev: this.prev };
  }

  restoreToken(t: LayerStateToken): void {
    this.cursor = t.cursor;
    this.clean = t.clean;
    this.prev = t.prev;
    this.instance.restoreState(t.state);
    // 恢复的是物理状态；模型参数数组不一定对应游标帧，下次 prepare 同一帧时要重新求值。
    this.stage = 0;
  }

  get checkpointCount(): number {
    return this.checkpoints.size;
  }
}
