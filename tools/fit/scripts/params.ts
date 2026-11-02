// 诊断：按场景规范求值图层（runtime 的 LayerEvaluator），打印指定帧的模型参数（物理之后）
//   node --import tools/fit/src/geom/register.mjs --experimental-strip-types tools/fit/scripts/params.ts <scene.json> <layerId> <frames> [正则]
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileFetcher, loadRuntime } from '../src/geom/core.ts';
import { MODEL_URI, REPO } from '../src/paths.ts';
const [scenePath, layerId, framesArg, reArg] = process.argv.slice(2);
const doc = JSON.parse(readFileSync(scenePath!, 'utf8'));
const layer = (doc.layers ?? [doc]).find((l: any) => l.id === layerId);
const rt = await loadRuntime();
const res = await rt.ModelResource.load(null, pathToFileURL(join(REPO, 'assets', MODEL_URI)).href, fileFetcher);
const inst = res.createInstance();
const noClip = (id: string): never => { throw new Error(id); };
const ev = new rt.LayerEvaluator(layer, inst, 60, noClip, noClip);
const re = new RegExp(reArg ?? '^(ParamAngle[XYZ]|ParamBodyAngle[XYZ]|Shoulder[YZ]|Param5[034]|Clothes2|LeanInput)$');
for (const f of framesArg!.split(',').map(Number)) {
  ev.prepare(f, undefined, 'draw');
  const v = inst.getParams();
  console.log(f, Object.entries(v).filter(([k]) => re.test(k)).map(([k, x]) => `${k}=${(x as number).toFixed(2)}`).join(' '));
}
