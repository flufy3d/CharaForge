// 调试：一次性覆盖是否生效
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const h = await openHarness();
try {
  const r = await h.page.evaluate(async () => {
    const H = window.__charaforge;
    const scene = { format: 'charaforge.scene', version: 1, meta: { title: 't' }, timebase: { fps: 60 }, duration: 20, composition: { width: 64, height: 32, background: null },
      assets: { models: [{ id: 'm', uri: 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json' }] },
      layers: [{ id: 'l', kind: 'model', model: 'm', in: 0, out: 20, transform: { x: 32, y: 16, scale: 0.002 },
        params: [{ id: 'AngleX', curve: { baked: { start: 0, values: Array(20).fill(0) } } }] }] };
    await H.loadScene(scene);
    const out: any[] = [];
    await H.renderFrame(5, {});
    out.push(['plain', H.layerParams('l').AngleX, H.layerParams('l').ParamAngleX]);
    await H.renderFrame(5, { overrides: { l: { pre: { AngleX: 25 } } } });
    out.push(['ov', H.layerParams('l').AngleX, H.layerParams('l').ParamAngleX]);
    const rt = (window as any).__cfRuntime;
    rt.renderFrame(5, { overrides: { l: { pre: { AngleX: -25 } } } });
    out.push(['ov-direct', H.layerParams('l').AngleX, H.layerParams('l').ParamAngleX]);
    return out;
  });
  console.log(JSON.stringify(r));
} finally { await h.close(); }
