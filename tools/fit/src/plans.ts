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

/**
 * 第三轮续（fit4）的 s15 规划：三个图层都覆盖到各自出现为止的全段。
 *   left：1187 起（泼溅擦除后可见），1340~1360 景深模糊从 0 升到 σ≈8（shots 的 depthOfField 效果强度曲线）；
 *   chibi：约 1229 以「翻牌」出现（scaleX 在 1229~1237 从 0 展开到 1，按画面估计），1264 起 Clothes2 0→1（fade 8，shots）；
 *   center：约 1352 在圆形开窗里淡入（clip 圆心/半径/不透明度逐关键帧拟合），1400 前后爱心眼（key4，闭眼期间切换，见第二轮）。
 */
export function planS15v3(shot: Shot): { plans: LayerPlan[]; order: string[] } {
  const inst = (id: string) => shot.instances.find((i) => i.id === id)!;
  const dof = shot.effects.find((e) => e.kind === 'depthOfField');
  const dofStrength = (f: number) => {
    if (!dof?.strength?.length) return 8;
    const ks = dof.strength;
    if (f <= ks[0]!.frame) return ks[0]!.value;
    if (f >= ks[ks.length - 1]!.frame) return ks[ks.length - 1]!.value;
    let i = 0;
    while (ks[i + 1]!.frame <= f) i++;
    const t = (f - ks[i]!.frame) / (ks[i + 1]!.frame - ks[i]!.frame);
    return ks[i]!.value + (ks[i + 1]!.value - ks[i]!.value) * t;
  };
  const base = (id: string, over: Partial<LayerPlan>): LayerPlan => ({
    id, inst: inst(id), in: shot.start, out: shot.end, constants: { ...inst(id).variant }, switches: [], blurRanges: [], opacityRanges: [],
    consts: [], colorFilter: false, hairBake: false, z: inst(id).z, ...over,
  });
  const left = base('s15.left', { in: 1187, blurRanges: [[1340, shot.end]], dofStrength });
  const chibi = base('s15.chibi', {
    in: 1229, blurRanges: [[1340, shot.end]], dofStrength, flipRamp: [1229, 1237],
    switches: [{ param: 'Clothes2', from: 0, to: 1, window: [1256, 1280], est: 1264, fade: 8 }],
  });
  const center = base('s15.center', {
    in: 1352, clipRange: [1352, 1400],
    switches: [{ param: 'key4', from: 0, to: 1, window: [1400, 1431], est: 1400, fade: 30 }],
  });
  // s15 的 shots 眼睛轨迹是人工锚点间的线性插值（尺度可差约 2 倍）：变换初值先做粗配准（traj2 的 reg）
  for (const q of [left, chibi, center]) (q as LayerPlan & { reg?: boolean }).reg = true;
  return { plans: [left, chibi, center], order: ['s15.left', 's15.chibi', 's15.center'] };
}
