/*
 * 场景资源解析：模型 / motion / expression，主线程与 Worker 共用。
 */
import type { Scene } from '@charaforge/scene';
import type { CubismExpressionMotion } from '@cubism/framework/motion/cubismexpressionmotion';
import type { CubismMotion } from '@cubism/framework/motion/cubismmotion';
import type { ModelResource } from './resources.ts';

export interface SceneAssets {
  models: Map<string, ModelResource>;
  motions: Map<string, CubismMotion>;
  expressions: Map<string, CubismExpressionMotion>;
}

export async function loadSceneAssets(
  scene: Scene,
  loadModel: (uri: string) => Promise<ModelResource>,
  resolveAsset: (uri: string) => string,
  onlyModels?: Set<string>,
): Promise<SceneAssets> {
  const models = new Map<string, ModelResource>();
  const motions = new Map<string, CubismMotion>();
  const expressions = new Map<string, CubismExpressionMotion>();
  const want = (id: string) => !onlyModels || onlyModels.has(id);
  await Promise.all(scene.assets.models.filter((m) => want(m.id)).map(async (m) => models.set(m.id, await loadModel(m.uri))));
  await Promise.all([
    ...(scene.assets.motions ?? []).filter((m) => want(m.model)).map(async (m) => {
      const res = models.get(m.model);
      if (!res) throw new Error(`motion ${m.id} 引用了不存在的模型 ${m.model}`);
      motions.set(m.id, await res.loadMotion(resolveAsset(m.uri)));
    }),
    ...(scene.assets.expressions ?? []).filter((e) => want(e.model)).map(async (e) => {
      const res = models.get(e.model);
      if (!res) throw new Error(`expression ${e.id} 引用了不存在的模型 ${e.model}`);
      expressions.set(e.id, await res.loadExpression(resolveAsset(e.uri)));
    }),
  ]);
  return { models, motions, expressions };
}

export function resolver<T>(map: Map<string, T>, what: string): (id: string) => T {
  return (id) => {
    const v = map.get(id);
    if (!v) throw new Error(`未知 ${what} ${id}`);
    return v;
  };
}
