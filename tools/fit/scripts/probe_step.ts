// 探针：物理输入阶跃响应（输入在第 5 帧从 0 跳到 v，打印输出参数随帧变化）
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const pairs: [string, number, string[]][] = [
  ['AngleX', 30, ['ParamAngleX', 'FrontHairX1', 'SideHairX1']],
  ['AngleZ', 30, ['ParamAngleZ', 'LateHairZ']],
  ['PositionX', 30, ['ParamBodyAngleX', 'HipAngleX', 'ParamBodyAngleRX']],
  ['PositionY', 30, ['ParamBodyAngleY', 'ShoulderY', 'HipAngleY']],
  ['PositionZ', 30, ['ParamBodyAngleZ', 'LEGAngleZ2']],
  ['LeanInput', -30, ['Param50', 'Param53', 'ParamBodyAngleY']],
  ['ParamBreath', 1, ['ParamBodyAngleY', 'Shine']],
];
const h = await openHarness();
try {
  const r = await h.page.evaluate(async (pairs) => {
    const H = window.__charaforge;
    const id = await H.createInstance('models/deepseek-chan/DeepSeek/大肥鱼.model3.json');
    const out: Record<string, number[][]> = {};
    for (const [inp, v, outs] of pairs) {
      H.resetInstancePhysics(id);
      for (let i = 0; i < 60; i++) H.stepInstance(id, { physics: true });
      const rows: number[][] = [];
      for (let i = 0; i < 70; i++) {
        H.stepInstance(id, { pre: { [inp]: i >= 5 ? v : 0 }, physics: true });
        const p = H.instanceParams(id);
        rows.push(outs.map((o) => +p[o]!.toFixed(2)));
      }
      out[`${inp}=${v} → ${outs.join(',')}`] = rows;
    }
    return out;
  }, pairs);
  for (const [k, rows] of Object.entries(r)) console.log(k, '\n  ', [5, 6, 7, 8, 10, 15, 20, 30, 45, 69].map((i) => `${i}:${rows[i]!.join('/')}`).join('  '));
} finally { await h.close(); }
