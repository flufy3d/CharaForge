/*
 * 每个镜头的图层规划：哪些实例成为图层、图层区间、开关常量/切换、哪些段逐帧拟合模糊/不透明度/裁切。
 * 由 shots.json 推导；shots 里不准的地方（实例出现时间、开窗裁切等）按画面观察修正，并在汇报里列出。
 */
import type { Shot } from '../../../packages/scene/src/index.ts';
import type { FitOptions, LayerPlan } from './fitshot.ts';

const ALL_CONSTS = ['c:LeanInput', 'c:Clothes2', 'c:breathPhase', 'c:breathPeriod', 'c:blur'];

export function planFor(shot: Shot, opts: { hairBake?: boolean } = {}): { plans: LayerPlan[]; options: FitOptions } {
  const base = (id: string, over: Partial<LayerPlan> = {}): LayerPlan => {
    const inst = shot.instances.find((i) => i.id === id)!;
    const constants = { ...inst.variant };
    // Clothes2（体型）作为镜头常量拟合；shots 的 variant 只当初值
    const constInit: Record<string, number> = {};
    if (constants.Clothes2 !== undefined) constInit['c:Clothes2'] = constants.Clothes2;
    return {
      id: inst.id,
      inst,
      in: shot.start,
      out: shot.end,
      constants,
      switches: [],
      blurRanges: [],
      opacityRanges: [],
      consts: ALL_CONSTS,
      constInit,
      colorFilter: false,
      hairBake: !!opts.hairBake,
      z: inst.z,
      ...over,
    };
  };
  switch (shot.id) {
    case 's43':
    case 's24':
      // 角色有整体调色（偏亮、低对比、偏蓝）：拟合镜头内常量的颜色滤镜
      return { plans: [base(shot.instances[0]!.id, { colorFilter: true })], options: { previewEvery: 6 } };
    case 's15': {
      // [1300, 1509)：1300 起左、右两个实例完整可见（尚未虚化）；约 1340~1360 两者虚化到 σ≈8；
      // 中间实例约 1352 起在圆形开窗里淡入（shots 没有这个 clip），约 1400 开窗消失。
      // 1187~1299（泼溅擦除、细线框全身、放大 wink、翻牌出现 Q 版）见汇报。
      const S = 1300;
      const center = base('s15.center', {
        in: 1352,
        clipRange: [1352, 1400],
        opacityRanges: [[1352, 1368]],
        // key4 爱心眼：画面上 1404~1429 是闭眼笑，1430 睁眼时已经是爱心眼（高饱和品红像素从 1430 开始出现），
        // 所以切换发生在闭眼期间、精确帧不可观测；按 VTS 表情热键的 0.5 秒淡入取 [1400, 1430)。
        // （shots 的 1450 偏晚；用拟合虹膜位置做的自动检测会把头发青色高光当成虹膜，见汇报。）
        switches: [{ param: 'key4', from: 0, to: 1, window: [1400, 1431], est: 1400, fade: 30 }],
      });
      const per = ['c:LeanInput', 'c:Clothes2', 'c:breathPhase', 'c:breathPeriod'];
      const left = base('s15.left', { in: S, blurRanges: [[S, shot.end]], consts: per });
      const chibi = base('s15.chibi', { in: S, blurRanges: [[S, shot.end]], consts: per, constInit: { 'c:Clothes2': 1 } });
      return { plans: [center, left, chibi], options: { previewEvery: 8, order: ['s15.center', 's15.left', 's15.chibi'] } };
    }
    default:
      throw new Error(`镜头 ${shot.id} 还没有拟合规划`);
  }
}

/** shots 效果强度曲线的线性插值（区间外保持端点） */
const strengthAt = (ks: { frame: number; value: number }[], f: number) => {
  if (f <= ks[0]!.frame) return ks[0]!.value;
  if (f >= ks[ks.length - 1]!.frame) return ks[ks.length - 1]!.value;
  let i = 0;
  while (ks[i + 1]!.frame <= f) i++;
  const t = (f - ks[i]!.frame) / (ks[i + 1]!.frame - ks[i]!.frame);
  return ks[i]!.value + (ks[i + 1]!.value - ks[i]!.value) * t;
};

/**
 * 第三轮续（fit4）的 s15 规划：全部从 shots 读取（recon 重标后的结构），不再硬编码帧号。
 *   图层起点 = 实例轨迹里第一个有 bbox 的帧（翻牌实例取 cardFlip 效果起点）；
 *   开关 = 实例 switchEvents（frame、fade、value）；景深 = depthOfField 效果（instances、strength 曲线）；
 *   翻牌 = cardFlip 效果（strength 即 scaleX）；圆形开窗 = 实例 clip 关键帧 + opacity 轨（之后不裁切）。
 *   变换由 shots 眼睛锚点先验约束（traj2），不做 reg 粗配准。
 */
export function planS15v3(shot: Shot): { plans: LayerPlan[]; order: string[] } {
  const inst = (id: string) => shot.instances.find((i) => i.id === id)!;
  const fx = (kind: string, id: string) => shot.effects.find((e: any) => e.kind === kind && (!e.instances || e.instances.includes(id))) as any;
  const firstSeen = (id: string) => (inst(id) as any).track.find((t: any) => t.bbox)?.frame ?? shot.start;
  const plan = (id: string): LayerPlan => {
    const I = inst(id) as any;
    const flip = fx('cardFlip', id);
    const dof = fx('depthOfField', id);
    const start = flip ? flip.start : firstSeen(id);
    const switches = (I.switchEvents ?? []).map((e: any) => ({ param: e.param, from: 0, to: e.value, window: [e.frame - 8, e.frame + 8] as [number, number], est: e.frame, fade: e.fade ?? 0 }));
    let clipKeys: LayerPlan['clipKeys'];
    let clipRange: [number, number] | undefined;
    if (I.clip?.length) {
      const op = I.track.filter((t: any) => t.opacity !== undefined && t.opacity !== null).map((t: any) => ({ frame: t.frame, value: t.opacity }));
      clipKeys = I.clip.map((c: any) => ({ f: c.frame, cx: c.rect[0] + c.rect[2] / 2, cy: c.rect[1] + c.rect[3] / 2, r: c.radius, a: op.length ? strengthAt(op, c.frame) : 1 }));
      clipRange = [I.clip[0].frame, I.clip[I.clip.length - 1].frame + 1];
    }
    return {
      id, inst: I, in: start, out: shot.end, constants: { ...I.variant }, switches, blurRanges: dof ? [[dof.start, shot.end]] : [], opacityRanges: [],
      consts: [], colorFilter: false, hairBake: false, z: I.z,
      ...(dof ? { dofStrength: (f: number) => (dof.strength?.length ? strengthAt(dof.strength, f) : 8) } : {}),
      ...(flip ? { flipCurve: (f: number) => strengthAt(flip.strength, f) } : {}),
      ...(clipKeys ? { clipKeys, clipRange } : {}),
      reg: false, // reg 在 s15（c3）上把左实例推偏约 150px、s24 上也有害：关闭，改用 shots 锚点先验（traj2）
    } as LayerPlan;
  };
  const left = plan('s15.left'), chibi = plan('s15.chibi'), center = plan('s15.center');
  return { plans: [left, chibi, center], order: ['s15.left', 's15.chibi', 's15.center'] };
}
