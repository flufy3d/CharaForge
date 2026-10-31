// 探针：打印指定参数的范围与默认值
import { openHarness } from '../../../apps/player/scripts/harness-node.ts';
const re = new RegExp(process.argv[2] ?? '.');
const h = await openHarness();
try {
  const r = await h.page.evaluate(async () => {
    const H = window.__charaforge;
    const id = await H.createInstance('models/deepseek-chan/DeepSeek/大肥鱼.model3.json');
    return H.instanceInfo(id).parameters;
  });
  for (const p of r) if (re.test(p.id)) console.log(p.id, p.min, p.max, p.default);
} finally { await h.close(); }
