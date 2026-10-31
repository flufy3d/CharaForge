/*
 * 每个镜头的图层规划：哪些实例成为图层、图层区间、开关常量/切换、哪些段拟合模糊。
 * 都由 shots.json 推导；shots 里不准的地方（实例出现时间等）在这里按拟合时的观察修正，并在汇报里说明。
 */
import type { Shot } from '../../../packages/scene/src/index.ts';
import type { FitOptions, LayerPlan } from './fitshot.ts';

export function planFor(shot: Shot): { plans: LayerPlan[]; options: FitOptions } {
  const base = (i: number, over: Partial<LayerPlan> = {}): LayerPlan => {
    const inst = shot.instances[i]!;
    return {
      id: inst.id,
      inst,
      in: shot.start,
      out: shot.end,
      constants: { ...inst.variant },
      switches: inst.switchEvents.map((e) => ({
        param: e.param,
        from: inst.variant[e.param] ?? 0,
        to: e.value,
        window: [e.frame - 8, e.frame + (e.fade ?? 0) + 8] as [number, number],
        est: e.frame,
        fade: e.fade ?? 0,
      })),
      blurRanges: [],
      fitOpacity: false,
      colorFilter: false,
      shotConsts: [{ key: 'LeanInput', lo: -30, hi: 0, sd: 3, tau: 10 }],
      z: inst.z,
      ...over,
    };
  };
  switch (shot.id) {
    case 's43':
    case 's24':
      // 两个镜头的角色都有整体调色（偏亮、低对比、偏蓝），拟合镜头内常量的颜色滤镜
      return { plans: [base(0, { colorFilter: true })], options: { previewEvery: 4 } };
    case 's15': {
      // 只拟合景深三实例段 [1404, 1509)：此前的「细线框全身 → 放大 wink → 翻牌出 Q 版 → 圆形开窗出中间实例」
      // 构图变化与转场太多（详见汇报）；三个实例在本段都完整可见。
      const A = 1404;
      const byId = (id: string) => shot.instances.findIndex((i) => i.id === id);
      const center = base(byId('s15.center'), { in: A, switches: [], colorFilter: true });
      // key4（爱心眼）在 shots 里是 1450 ±、fade 30：作为逐帧权重维度拟合，再拟合成斜坡
      center.switches = [{ param: 'key4', from: 0, to: 1, window: [1430, 1495], est: 1450, fade: 30 }];
      // 背景两实例被中间实例遮挡：拟合它们时把中间实例的 bbox 当排除区，并先拟合背景、再拟合中间实例
      const left = base(byId('s15.left'), { in: A, blurRanges: [[A, shot.end]], fitExcludeInstances: ['s15.center'] });
      const chibi = base(byId('s15.chibi'), { in: A, blurRanges: [[A, shot.end]], switches: [], fitExcludeInstances: ['s15.center'] });
      // Q 版实例：Clothes2 在 1264 起切换，本段已稳定；具体是 0.4（体型1）还是 1（体型2）不确定，作为镜头常量拟合
      chibi.shotConsts = [...chibi.shotConsts, { key: 'Clothes2', lo: 0.3, hi: 1, sd: 0.1, tau: 1 }];
      return { plans: [center, left, chibi], options: { previewEvery: 6, order: ['s15.left', 's15.chibi', 's15.center'] } };
    }
    default:
      throw new Error(`镜头 ${shot.id} 还没有拟合规划`);
  }
}
