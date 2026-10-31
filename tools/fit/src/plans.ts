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
