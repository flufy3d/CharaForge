/*
 * 求值 / Core update Worker（每个 Worker 一份 Cubism Core + Framework）。
 *
 * - 「主场」图层：Worker 持有它的 LayerEvaluator（物理状态链），按规范 a～g 求值到「参数就绪」；
 * - Core update：Core update 是（参数值, 部件不透明度）的纯函数，任意 Worker 都可以做。
 *   主场 Worker 把 Core 输入经 MessagePort 交给主线程指定的（最空闲的）Worker，那个 Worker 做 update
 *   并把动态数据直接回给主线程。
 * 这样物理链不重复计算，最贵的 Core update 在所有 Worker 间并行；结果与主线程同步路径逐位相同。
 * 由 worker.ts 在 Core 加载之后动态 import（Framework 有模块在加载时就引用全局 Live2DCubismCore）。
 */
import { parseScene, type ModelLayer } from '@charaforge/scene';
import { ensureCubism } from './cubism.ts';
import { LayerEvaluator, type ParamOverrides } from './evaluator.ts';
import type { ModelInstance } from './instance.ts';
import { ModelResource } from './resources.ts';
import { loadSceneAssets, resolver } from './sceneAssets.ts';
import type { PeerUpdate, WorkerRequest, WorkerResponse } from './workerProtocol.ts';

const scope = self as unknown as {
  postMessage(msg: WorkerResponse, transfer?: Transferable[]): void;
};

let assetBase = '';
let selfIndex = 0;
let ports: (MessagePort | null)[] = [];
const resources = new Map<string, Promise<ModelResource>>();
const updaters = new Map<string, Promise<ModelInstance>>();
let layers = new Map<string, LayerEvaluator>();
const freeBuffers: ArrayBuffer[] = [];

function loadModel(url: string): Promise<ModelResource> {
  let p = resources.get(url);
  if (!p) {
    p = ModelResource.load(null, url);
    resources.set(url, p);
  }
  return p;
}

/** 每个模型一个专门做 Core update 的实例（与主场图层的实例分开，互不干扰）。 */
function updaterFor(url: string): Promise<ModelInstance> {
  let p = updaters.get(url);
  if (!p) {
    p = loadModel(url).then((r) => r.createInstance());
    updaters.set(url, p);
  }
  return p;
}

export async function handlePeer(msg: PeerUpdate): Promise<void> {
  const inst = await updaterFor(msg.modelUrl);
  const t0 = performance.now();
  inst.updateFromInputs(msg.inputs);
  const packet = inst.exportDynamics(freeBuffers.pop());
  scope.postMessage({ type: 'update', id: msg.id, ms: performance.now() - t0, evalMs: msg.evalMs, steps: msg.steps, packet, from: selfIndex }, [packet.buf]);
}

export async function handle(msg: WorkerRequest, enqueuePeer: (m: PeerUpdate) => void): Promise<void> {
  switch (msg.type) {
    case 'recycle':
      if (freeBuffers.length < 8) freeBuffers.push(msg.buf);
      return;
    case 'init':
      assetBase = msg.assetBase;
      selfIndex = msg.index;
      ports = msg.ports;
      ensureCubism(msg.memoryBytes);
      scope.postMessage({ type: 'ok', id: msg.id });
      return;
    case 'scene': {
      for (const ev of layers.values()) ev.instance.release();
      layers = new Map();
      const scene = parseScene(msg.scene);
      const mine = scene.layers.filter((l): l is ModelLayer => l.kind === 'model' && msg.homeLayers.includes(l.id));
      const abs = (u: string) => new URL(u, assetBase).href;
      const assets = await loadSceneAssets(scene, (u) => loadModel(abs(u)), abs, new Set(mine.map((l) => l.model)));
      for (const layer of mine) {
        const inst = assets.models.get(layer.model)!.createInstance();
        layers.set(layer.id, new LayerEvaluator(layer, inst, scene.timebase.fps, resolver(assets.motions, 'motion'), resolver(assets.expressions, 'expression')));
      }
      // 预先建好本场景所有模型的 update 实例
      await Promise.all(scene.assets.models.map((m) => updaterFor(abs(m.uri))));
      scope.postMessage({ type: 'ok', id: msg.id });
      return;
    }
    case 'eval': {
      const ev = layers.get(msg.layerId);
      if (!ev) throw new Error(`Worker ${selfIndex} 不负责图层 ${msg.layerId}`);
      const t0 = performance.now();
      ev.prepare(msg.frame, msg.overrides as ParamOverrides | undefined, msg.mode);
      if (msg.mode === 'advance') {
        scope.postMessage({ type: 'done', id: msg.id, steps: ev.lastSteps });
        return;
      }
      const inputs = ev.instance.exportInputs();
      const peer: PeerUpdate = { type: 'update', id: msg.id, modelUrl: ev.instance.resource.url, inputs, steps: ev.lastSteps, evalMs: performance.now() - t0 };
      if (msg.target === selfIndex) enqueuePeer(peer);
      else ports[msg.target]!.postMessage(peer, [inputs.params.buffer as ArrayBuffer, inputs.parts.buffer as ArrayBuffer]);
      return;
    }
  }
}
