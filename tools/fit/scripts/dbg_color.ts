// 调试：颜色滤镜对透明像素的输出
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const h = await openHarness();
try {
  const r = await h.page.evaluate(async () => {
    const H = window.__charaforge;
    const rt = (window as any).__cfRuntime;
    const scene = { format: 'charaforge.scene', version: 1, meta: { title: 't' }, timebase: { fps: 60 }, duration: 2, composition: { width: 64, height: 32, background: null },
      assets: { models: [{ id: 'm', uri: 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json' }] },
      layers: [{ id: 'l', kind: 'model', model: 'm', in: 0, out: 2, transform: { x: 32, y: 16, scale: 0.002 }, filters: [] as any[] }] };
    const out: any = {};
    for (const filt of [[], [{ type: 'color' }], [{ type: 'color', brightness: 0.1 }], [{ type: 'color', contrast: 0.6 }], [{ type: 'blur', sigma: 1 }]]) {
      scene.layers[0]!.filters = filt;
      await H.loadScene(scene);
      const px = await H.renderFrame(0, { read: { region: { x: 0, y: 0, width: 1, height: 1 } } });
      const c = rt.compositor;
      out[JSON.stringify(filt)] = Array.from(atob(px.data!)).map((ch) => ch.charCodeAt(0));
    }
    return out;
  });
  console.log(JSON.stringify(r, null, 1));
} finally { await h.close(); }
