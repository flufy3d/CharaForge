// 临时探针：检查 harness、画布信息、控制器参数经物理到输出参数的响应
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const MODEL = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
const h = await openHarness();
try {
  console.log(h.info);
  const r = await h.page.evaluate(async (m) => {
    const H = window.__charaforge;
    const id = await H.createInstance(m);
    const info = H.instanceInfo(id);
    const out: any = { canvas: info.canvas, n: info.parameters.length };
    const seq: any[] = [];
    for (let i = 0; i < 40; i++) {
      H.stepInstance(id, { pre: { AngleX: i >= 5 ? 30 : 0, EyeSmile: 0, AngleY: i >= 5 ? 20 : 0 }, physics: true });
      const p = H.instanceParams(id);
      seq.push([i, +p.ParamAngleX.toFixed(2), +p.ParamAngleY.toFixed(2), +p.ParamBodyAngleX.toFixed(2)]);
    }
    out.seq = seq;
    const sel = ['AngleX','ParamAngleX','ParamEyeLOpen','ParamEyeROpen','EyeSmile','ParamMouthOpenY','ParamMouthForm','Clothes1','Clothes5','ParamBreath','PositionX','PositionY','PositionZ','LeanInput','ParamEyeBallX','ParamBrowLY','JawOpen','key11'];
    out.params = info.parameters.filter((p: any) => sel.includes(p.id));
    return out;
  }, MODEL);
  console.log(JSON.stringify(r, null, 0));
} finally { await h.close(); }
