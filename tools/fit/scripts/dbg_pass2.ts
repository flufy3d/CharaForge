// 调试：画模型之后再做 pass
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const h = await openHarness();
try {
  const r = await h.page.evaluate(async () => {
    const H = window.__charaforge;
    const rt = (window as any).__cfRuntime;
    const c = rt.compositor;
    const out: any = {};
    const id = await H.createInstance('models/deepseek-chan/DeepSeek/大肥鱼.model3.json');
    H.stepInstance(id, {});
    const inst = rt.instance(id);
    const M = [0.05, 0, 0, 0.05, 600, 100];
    const L = c.layerSource(inst, M, [], true);
    out.layerCorner = Array.from(c.readRT(L, 0, 0, 1, 1).data);
    out.layerSize = [L.width, L.height, c.tmpA?.width, c.tmpA?.height];
    const X = c.pooledRT(L.width, L.height, ':dbgX');
    c.copy(L, X);
    out.copyCorner = Array.from(c.readRT(X, 0, 0, 1, 1).data);
    out.copyCenter = Array.from(c.readRT(X, 690, 300, 1, 1).data);
    const L2 = c.layerSource(inst, M, [{ type: 'color', brightness: 0, contrast: 1, saturation: 1, multiply: [1, 1, 1], offset: [0, 0, 0] }], true);
    out.colorCorner = Array.from(c.readRT(L2, 0, 0, 1, 1).data);
    // 不画模型，直接对已清空的 tmp 做 color pass
    c.clearRT(c.layerRT, [0, 0, 0, 0]);
    out.afterClear = Array.from(c.readRT(c.layerRT, 0, 0, 1, 1).data);
    c.copy(c.layerRT, X);
    out.copyOfCleared = Array.from(c.readRT(X, 0, 0, 1, 1).data);
    return out;
  });
  console.log(JSON.stringify(r));
} finally { await h.close(); }
