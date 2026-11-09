import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  EASE_PRESETS,
  applyEditChanges,
  applyEditCommands,
  easeKeyframes,
  easePreset,
  parseScene,
  previewAt,
  sample,
  undoEditChanges,
  type AnimationPreset,
  type EditCommand,
  type Scene,
} from '../src/index.ts';

function makeScene(): Scene {
  return parseScene({
    format: 'charaforge.scene',
    version: 1,
    meta: { title: 'edit test', description: 'base' },
    timebase: { fps: 60 },
    duration: 120,
    composition: { width: 640, height: 360, background: '#000000' },
    assets: {
      models: [{ id: 'm', uri: 'models/m.model3.json' }],
      motions: [{ id: 'idle', model: 'm', uri: 'motions/idle.motion3.json' }],
      expressions: [{ id: 'smile', model: 'm', uri: 'expressions/smile.exp3.json' }],
      audio: [{ id: 'bgm', uri: 'audio/bgm.wav' }],
      images: [{ id: 'img', uri: 'data:image/png;base64,AAAA', width: 8, height: 8 }],
    },
    camera: { x: 320, y: 180, zoom: 1 },
    layers: [
      {
        id: 'model',
        name: 'main',
        kind: 'model',
        model: 'm',
        in: 0,
        out: 120,
        transform: { x: 320, y: 180, scale: 1 },
        opacity: 1,
        clip: { x: 0, y: 0, width: 640, height: 360 },
        filters: [
          { type: 'blur', sigma: 0 },
          { type: 'color', contrast: 1, saturation: 1 },
        ],
        params: [{ id: 'ParamAngleX', curve: { keys: [{ t: 0, v: 0 }, { t: 60, v: 30 }] } }],
      },
      {
        id: 'pic',
        kind: 'image',
        image: 'img',
        in: 0,
        out: 120,
        transform: { x: 100, y: 100, scale: 1 },
      },
    ],
    audio: [{ audio: 'bgm', start: 0, duration: 1 }],
    markers: [{ id: 'intro', start: 0, end: 60, label: 'intro' }],
  });
}

function opacityAt(scene: Scene, frame: number, layerIndex = 0): number {
  return sample(scene.layers[layerIndex]!.opacity, frame, 1);
}

test('setScene 更新元数据、时间、画布、背景并校验结果', () => {
  const scene = makeScene();
  const result = applyEditCommands(scene, {
    op: 'setScene',
    patch: { title: '新标题', description: '新描述', fps: 30, duration: 240, width: 1280, height: 720, background: null },
  });
  assert.equal(scene.meta.title, '新标题');
  assert.equal(scene.meta.description, '新描述');
  assert.equal(scene.timebase.fps, 30);
  assert.equal(scene.duration, 240);
  assert.deepEqual(scene.composition, { width: 1280, height: 720, background: null });
  assert.ok(result.effects.includes('render'));
  assert.ok(result.effects.includes('structure'));
  assert.ok(result.effects.includes('audio'));
  assert.ok(result.effects.includes('clock'));
  assert.equal(result.changes.some((change) => change.path === '/meta/title'), true);
  parseScene(scene);

  assert.throws(() => applyEditCommands(makeScene(), { op: 'setScene', patch: { fps: 0 } }), /fps/);
  assert.throws(() => applyEditCommands(makeScene(), { op: 'setScene', patch: { duration: 1.5 } }), /整数/);
  assert.throws(() => applyEditCommands(makeScene(), { op: 'setScene', patch: { unknown: 1 } as never }), /不允许/);
});

test('图片/模型图层：新增、复制、移动、删除及图片资源回收', () => {
  const scene = makeScene();
  const addImage = applyEditCommands(scene, {
    op: 'addImageLayer',
    asset: { id: 'star', uri: 'data:image/png;base64,AAAA', width: 8, height: 8 },
    layer: { id: 'star-layer', kind: 'image', image: 'star', in: 0, out: 120, transform: { x: 50, y: 50, scale: 1 } },
    index: 0,
  });
  assert.equal(scene.layers[0]!.id, 'star-layer');
  assert.ok(addImage.effects.includes('structure'));
  assert.ok(scene.assets.images!.some((asset) => asset.id === 'star'));

  applyEditCommands(scene, { op: 'duplicateLayer', id: 'star-layer', newId: 'star-2', name: '副本' });
  assert.equal(scene.layers[1]!.id, 'star-2');
  assert.equal((scene.layers[1] as { image: string }).image, 'star');
  assert.equal(scene.layers[1]!.name, '副本');

  applyEditCommands(scene, { op: 'moveLayer', id: 'star-2', index: scene.layers.length - 1 });
  assert.equal(scene.layers[scene.layers.length - 1]!.id, 'star-2');

  applyEditCommands(scene, { op: 'removeLayer', id: 'star-layer' });
  assert.ok(scene.assets.images!.some((asset) => asset.id === 'star'), '另一层仍引用图片资源');
  applyEditCommands(scene, { op: 'removeLayer', id: 'star-2' });
  assert.equal(scene.assets.images!.some((asset) => asset.id === 'star'), false);
  parseScene(scene);

  const addModel = applyEditCommands(scene, {
    op: 'addModelLayer',
    layer: { id: 'extra', kind: 'model', model: 'm', in: 10, out: 100, transform: { x: 0, y: 0, scale: 0.5 } },
  });
  assert.ok(addModel.effects.includes('structure'));
  assert.equal(scene.layers[scene.layers.length - 1]!.id, 'extra');
  applyEditCommands(scene, { op: 'removeLayer', id: 'extra' });
  parseScene(scene);

  assert.throws(
    () =>
      applyEditCommands(scene, {
        op: 'addModelLayer',
        layer: { id: 'bad', kind: 'model', model: 'missing', in: 0, out: 1, transform: { x: 0, y: 0, scale: 1 } },
      }),
    /模型资源不存在/,
  );
});

test('updateLayer：transform 浅合并、null 删除、未知字段失败', () => {
  const scene = makeScene();
  applyEditCommands(scene, {
    op: 'updateLayer',
    id: 'model',
    patch: { transform: { x: 100, rotation: 45 }, opacity: 0.5, blendMode: 'add' },
  });
  assert.equal(sample(scene.layers[0]!.transform.x, 0, 0), 100);
  assert.equal(sample(scene.layers[0]!.transform.y, 0, 0), 180);
  assert.equal(sample(scene.layers[0]!.transform.rotation, 0, 0), 45);
  assert.equal(sample(scene.layers[0]!.opacity, 0, 1), 0.5);
  assert.equal(scene.layers[0]!.blendMode, 'add');
  parseScene(scene);

  applyEditCommands(scene, { op: 'updateLayer', id: 'model', patch: { clip: null, physics: null } });
  assert.equal('clip' in scene.layers[0]!, false);
  assert.equal('physics' in scene.layers[0]!, false);
  parseScene(scene);

  const before = structuredClone(scene);
  assert.throws(() => applyEditCommands(scene, { op: 'updateLayer', id: 'model', patch: { nope: 1 } as never }), /不允许修改字段/);
  assert.deepEqual(scene, before);

  assert.throws(() => applyEditCommands(scene, { op: 'updateLayer', id: 'model', patch: { id: 'x' } as never }), /不可修改/);
  assert.throws(() => applyEditCommands(scene, { op: 'updateLayer', id: 'pic', patch: { model: 'm' } as never }), /不允许修改字段/);
  applyEditCommands(scene, { op: 'updateLayer', id: 'pic', patch: { attach: null } });
});

test('关键帧：常量转换、插入、更新、移动、缓动、删除与 flatten', () => {
  const scene = makeScene();
  const target = { kind: 'layer' as const, id: 'model', path: 'opacity' };

  const added = applyEditCommands(scene, { op: 'addKeyframe', target, t: 0, v: 0 });
  assert.equal(added.changes.some((change) => change.path === '/layers/0/opacity'), true);
  assert.equal(opacityAt(scene, 50), 0);

  applyEditCommands(scene, { op: 'addKeyframe', target, t: 20, v: 10 });
  applyEditCommands(scene, { op: 'addKeyframe', target, t: 40, v: 0 });
  assert.equal(opacityAt(scene, 10), 5);
  assert.equal(opacityAt(scene, 30), 5);

  applyEditCommands(scene, { op: 'updateKeyframe', target, t: 20, v: 20, newT: 30 });
  assert.equal(opacityAt(scene, 15), 10);
  assert.equal(sample(scene.layers[0]!.opacity, 30, 1), 20);

  applyEditCommands(scene, { op: 'setEase', target, t: 0, ease: 'step' });
  assert.equal(opacityAt(scene, 15), 0);
  applyEditCommands(scene, { op: 'setEase', target, ease: 'linear' });
  const curve = scene.layers[0]!.opacity as { keys: { ease?: unknown }[] };
  assert.equal(curve.keys.every((key) => key.ease === 'linear'), true);

  applyEditCommands(scene, { op: 'removeKeyframe', target, t: 30, flatten: false });
  assert.equal((scene.layers[0]!.opacity as { keys: unknown[] }).keys.length, 2);
  applyEditCommands(scene, { op: 'removeKeyframe', target, t: 40, flatten: false });
  assert.equal((scene.layers[0]!.opacity as { keys: unknown[] }).keys.length, 1);
  applyEditCommands(scene, { op: 'removeKeyframe', target, t: 0 });
  assert.equal(typeof scene.layers[0]!.opacity, 'number', '删完关键帧后压成常量');
  parseScene(scene);

  applyEditCommands(scene, {
    op: 'setKeyframes',
    target,
    keys: [
      { t: 10, v: 5 },
      { t: 10, v: 9 },
      { t: 0, v: 0 },
    ],
  });
  const keys = (scene.layers[0]!.opacity as { keys: { t: number; v: number }[] }).keys;
  assert.deepEqual(keys.map((key) => [key.t, key.v]), [[0, 0], [10, 9]]);

  applyEditCommands(scene, { op: 'setKeyframes', target, keys: [] });
  assert.equal(scene.layers[0]!.opacity, undefined, '空 keys 删除可选动画属性');
  assert.equal(opacityAt(scene, 99), 1);
  parseScene(scene);

  assert.throws(() => applyEditCommands(scene, { op: 'updateKeyframe', target, t: 999, v: 1 }), /没有关键帧/);
});

test('keyframe 目标：clip、filters 缺省值、param 轨道创建与效果', () => {
  const scene = makeScene();

  applyEditCommands(scene, {
    op: 'addKeyframe',
    target: { kind: 'layer', id: 'model', path: 'clip.radius' },
    t: 5,
    v: 12,
  });
  assert.equal(sample(scene.layers[0]!.clip!.radius, 5, 0), 12);

  applyEditCommands(scene, {
    op: 'addKeyframe',
    target: { kind: 'layer', id: 'model', path: 'filters.1.brightness' },
    t: 5,
    v: 0.25,
  });
  assert.equal(sample((scene.layers[0]!.filters![1] as { brightness?: unknown }).brightness as never, 5, 0), 0.25);

  assert.throws(
    () =>
      applyEditCommands(scene, {
        op: 'addKeyframe',
        target: { kind: 'layer', id: 'model', path: 'filters.0.brightness' },
        t: 0,
        v: 1,
      }),
    /不支持动画属性/,
  );

  const modelLayer = scene.layers[0]!;
  if (modelLayer.kind !== 'model') throw new Error('test scene');
  const track = modelLayer.params![0]!;
  const existing = applyEditCommands(scene, {
    op: 'addKeyframe',
    target: { kind: 'param', id: 'model', param: 'ParamAngleX' },
    t: 30,
    v: 15,
  });
  assert.equal(modelLayer.params![0], track, '已有 ParamTrack 对象原地保留');
  assert.equal(sample(track.curve, 30, 0), 15);
  assert.ok(existing.effects.includes('workers'));
  assert.equal(existing.effects.includes('evaluator'), false);

  const created = applyEditCommands(scene, {
    op: 'addKeyframe',
    target: { kind: 'param', id: 'model', param: 'ParamAngleY' },
    t: 10,
    v: 7,
  });
  const createdTrack = modelLayer.params!.find((item) => item.id === 'ParamAngleY')!;
  assert.ok(createdTrack);
  assert.equal(sample(createdTrack.curve, 10, 0), 7);
  assert.ok(created.effects.includes('evaluator'));
  assert.ok(created.effects.includes('workers'));

  const removed = applyEditCommands(scene, {
    op: 'setKeyframes',
    target: { kind: 'param', id: 'model', param: 'ParamAngleY' },
    keys: [],
  });
  assert.equal(modelLayer.params!.some((item) => item.id === 'ParamAngleY'), false);
  assert.ok(removed.effects.includes('evaluator'));

  const noClip = makeScene();
  applyEditCommands(noClip, { op: 'updateLayer', id: 'model', patch: { clip: null } });
  assert.throws(
    () =>
      applyEditCommands(noClip, {
        op: 'addKeyframe',
        target: { kind: 'layer', id: 'model', path: 'clip.x' },
        t: 0,
        v: 1,
      }),
    /没有 clip/,
  );
});

test('camera：更新、关键帧、删除与 preview', () => {
  const scene = makeScene();
  applyEditCommands(scene, {
    op: 'updateCamera',
    patch: { x: { keys: [{ t: 0, v: 0 }, { t: 60, v: 100 }] }, zoom: 2, rotation: null },
  });
  assert.equal(sample(scene.camera!.x, 30, 0), 50);
  assert.equal(sample(scene.camera!.zoom, 0, 1), 2);
  assert.equal(scene.camera!.rotation, undefined);
  parseScene(scene);

  applyEditCommands(scene, {
    op: 'setKeyframes',
    target: { kind: 'camera', path: 'zoom' },
    keys: [{ t: 0, v: 1 }, { t: 60, v: 3 }],
    ease: 'linear',
  });
  assert.equal(sample(scene.camera!.zoom, 30, 1), 2);

  const preview = applyEditCommands(scene, [
    { op: 'preview', frame: 30 },
    { op: 'noop' },
  ]).preview!;
  assert.equal(preview.frame, 30);
  assert.equal(preview.camera!.x, 50);
  assert.equal(preview.activeLayers.length, 2);
  assert.equal(preview.activeLayers[0]!.kind, 'model');
  assert.equal(preview.audio[0]!.active, true);
  assert.deepEqual(preview.markers.map((marker) => marker.id), ['intro']);

  assert.equal(previewAt(scene, 60).markers.length, 0, 'end 不含在区间内');
  assert.equal(previewAt(scene, 60).audio[0]!.active, false);

  applyEditCommands(scene, { op: 'setCamera', camera: null });
  assert.equal(scene.camera, undefined);
  assert.equal(previewAt(scene, 0).camera, null);
  assert.throws(() => applyEditCommands(scene, { op: 'updateCamera', patch: { zoom: 1 } }), /没有 camera/);
});

test('motion / expression：复用或创建资源、更新、删除', () => {
  const scene = makeScene();
  const first = applyEditCommands(scene, {
    op: 'addMotion',
    layer: 'model',
    uri: 'motions/idle.motion3.json',
    start: 10,
    end: 50,
    loop: true,
  });
  assert.equal(scene.assets.motions!.length, 1, '按 uri 复用已有资源');
  assert.equal((scene.layers[0] as { motions?: unknown[] }).motions!.length, 1);
  assert.ok(first.effects.includes('evaluator'));
  parseScene(scene);

  const second = applyEditCommands(scene, {
    op: 'addMotion',
    layer: 'model',
    uri: 'motions/wave.motion3.json',
    start: 0,
  });
  assert.equal(scene.assets.motions!.length, 2);
  assert.ok(second.effects.includes('structure'));
  const wave = scene.assets.motions!.find((asset) => asset.uri === 'motions/wave.motion3.json')!;
  assert.equal(wave.model, 'm');
  assert.equal(wave.id, 'wave');

  applyEditCommands(scene, {
    op: 'updateMotion',
    layer: 'model',
    index: 0,
    patch: { end: 60, offset: 0.5, fadeIn: 0, loop: null },
  });
  const motionLayer = scene.layers[0] as { motions: { motion?: string; end?: number; offset?: number; loop?: boolean }[] };
  assert.equal(motionLayer.motions[0]!.end, 60);
  assert.equal(motionLayer.motions[0]!.offset, 0.5);
  assert.equal(motionLayer.motions[0]!.loop, undefined);
  applyEditCommands(scene, { op: 'updateMotion', layer: 'model', index: 1, patch: { motion: 'idle' } });
  assert.equal(motionLayer.motions[1]!.motion, 'idle');
  applyEditCommands(scene, { op: 'updateMotion', layer: 'model', index: 1, patch: { motion: wave.id } });
  assert.throws(
    () => applyEditCommands(scene, { op: 'updateMotion', layer: 'model', index: 0, patch: { speed: 0 } }),
    /大于 0/,
  );
  applyEditCommands(scene, { op: 'removeMotion', layer: 'model', index: 0 });
  assert.equal(motionLayer.motions.length, 1);

  applyEditCommands(scene, {
    op: 'addExpression',
    layer: 'model',
    uri: 'expressions/wink.exp3.json',
    start: 20,
  });
  const exprs = scene.assets.expressions!;
  assert.equal(exprs.length, 2);
  const exprLayer = scene.layers[0] as { expressions: { expression: string; end?: number }[] };
  assert.equal(exprLayer.expressions.length, 1);
  applyEditCommands(scene, { op: 'updateExpression', layer: 'model', index: 0, patch: { end: 80 } });
  assert.equal(exprLayer.expressions[0]!.end, 80);
  applyEditCommands(scene, { op: 'removeExpression', layer: 'model', index: 0 });
  assert.equal(exprLayer.expressions.length, 0);

  assert.throws(
    () =>
      applyEditCommands(scene, {
        op: 'addMotion',
        layer: 'pic',
        uri: 'motions/idle.motion3.json',
        start: 0,
      }),
    /只能加到模型图层/,
  );
});

test('audio：新增/更新/删除并返回 audio + clock', () => {
  const scene = makeScene();
  const result = applyEditCommands(scene, {
    op: 'addAudio',
    uri: 'audio/bgm.wav',
    start: 10,
    offset: 0.25,
    gain: 0.8,
  });
  assert.equal(scene.audio!.length, 2);
  assert.equal(scene.audio![1]!.audio, 'bgm');
  assert.ok(result.effects.includes('audio'));
  assert.ok(result.effects.includes('clock'));

  applyEditCommands(scene, { op: 'updateAudio', index: 1, patch: { duration: 1.5, gain: null, start: 20 } });
  assert.equal(scene.audio![1]!.duration, 1.5);
  assert.equal(scene.audio![1]!.gain, undefined);
  assert.equal(scene.audio![1]!.start, 20);
  parseScene(scene);

  applyEditCommands(scene, { op: 'addAudio', uri: 'audio/sfx.wav' });
  assert.equal(scene.assets.audio!.length, 2);
  assert.equal(scene.assets.audio!.find((asset) => asset.uri === 'audio/sfx.wav')!.id, 'sfx');

  applyEditCommands(scene, { op: 'removeAudio', index: 2 });
  assert.equal(scene.audio!.length, 2);
  assert.throws(() => applyEditCommands(scene, { op: 'removeAudio', index: 99 }), /越界/);
});

test('marker：新增/更新/删除', () => {
  const scene = makeScene();
  applyEditCommands(scene, { op: 'addMarker', marker: { id: 'beat', start: 30, kind: 'beat' } });
  assert.equal(scene.markers!.length, 2);
  applyEditCommands(scene, { op: 'updateMarker', id: 'beat', patch: { end: 90, label: 'drop' } });
  assert.equal(scene.markers![1]!.end, 90);
  assert.equal(scene.markers![1]!.label, 'drop');
  applyEditCommands(scene, { op: 'removeMarker', id: 'beat' });
  assert.equal(scene.markers!.length, 1);
  assert.throws(() => applyEditCommands(scene, { op: 'addMarker', marker: { id: 'intro', start: 0 } }), /已存在/);
});

test('undo/redo 与失败回滚', () => {
  const scene = makeScene();
  const original = structuredClone(scene);
  const commands: EditCommand[] = [
    { op: 'addKeyframe', target: { kind: 'layer', id: 'model', path: 'opacity' }, t: 10, v: 0.2 },
    { op: 'updateCamera', patch: { zoom: { keys: [{ t: 0, v: 1 }, { t: 60, v: 2 }] } } },
    { op: 'addAudio', uri: 'audio/extra.wav', start: 5 },
    { op: 'duplicateLayer', id: 'model', newId: 'model-2' },
  ];
  const result = applyEditCommands(scene, commands);
  const edited = structuredClone(scene);
  assert.notDeepEqual(scene, original);

  undoEditChanges(scene, result.changes);
  assert.deepEqual(scene, original);

  applyEditChanges(scene, result.changes);
  assert.deepEqual(scene, edited);
  parseScene(scene);

  const rollback = makeScene();
  const rollbackOriginal = structuredClone(rollback);
  assert.throws(
    () =>
      applyEditCommands(rollback, [
        { op: 'updateLayer', id: 'model', patch: { opacity: 0.2 } },
        { op: 'updateLayer', id: 'model', patch: { bogus: 1 } as never },
      ]),
    /不允许修改字段/,
  );
  assert.deepEqual(rollback, rollbackOriginal);

  const validationRollback = makeScene();
  const validationOriginal = structuredClone(validationRollback);
  assert.throws(
    () => applyEditCommands(validationRollback, { op: 'updateLayer', id: 'model', patch: { in: 50, out: 10 } }),
    /out 必须大于 in/,
  );
  assert.deepEqual(validationRollback, validationOriginal);

  const unchecked = makeScene();
  applyEditCommands(unchecked, { op: 'updateLayer', id: 'model', patch: { in: 50, out: 10 } }, { validate: 'none' });
  assert.throws(() => parseScene(unchecked));

  const finalCheck = makeScene();
  assert.throws(
    () =>
      applyEditCommands(
        finalCheck,
        [
          { op: 'updateLayer', id: 'model', patch: { opacity: 0.5 } },
          { op: 'updateLayer', id: 'model', patch: { in: 50, out: 10 } },
        ],
        { validate: 'final' },
      ),
    /out 必须大于 in/,
  );
  assert.deepEqual(finalCheck, makeScene());
});

test('ease 预设与 easeKeyframes', () => {
  assert.equal(easePreset('linear'), 'linear');
  assert.equal(easePreset('hold'), 'step');
  assert.deepEqual(easePreset('easeIn'), [0.42, 0, 1, 1]);
  assert.deepEqual(easePreset([0.1, 0.2, 0.3, 0.4]), [0.1, 0.2, 0.3, 0.4]);
  assert.equal(EASE_PRESETS.easeInOut instanceof Array, true);
  assert.throws(() => easePreset('nope' as never), /未知缓动/);

  const keys = [{ t: 0, v: 0 }, { t: 10, v: 1, ease: 'step' as const }];
  easeKeyframes(keys, 'smooth');
  assert.equal(keys[0]!.ease, 'smooth');
  assert.equal(keys[1]!.ease, 'smooth');
});

test('animate：所有预设生成关键帧、保留区间外关键帧、解析通过', () => {
  const presets: AnimationPreset[] = [
    'fadeIn',
    'fadeOut',
    'float',
    'pulse',
    'spin',
    'bounce',
    'blink',
    'slideIn',
    'shake',
    'zoomIn',
  ];
  for (const preset of presets) {
    const scene = makeScene();
    const layer = scene.layers[0]!;
    layer.transform.x = { keys: [{ t: 0, v: 320 }, { t: 100, v: 400 }] };
    const result = applyEditCommands(scene, { op: 'animate', id: 'model', preset, start: 10, end: 70 });
    assert.ok(result.effects.includes('render'));
    parseScene(scene);
    const x = layer.transform.x as { keys: { t: number }[] };
    assert.equal(x.keys.some((key) => key.t === 0), true, `${preset} 保留区间外首关键帧`);
    assert.equal(x.keys.some((key) => key.t === 100), true, `${preset} 保留区间外尾关键帧`);
    const hasAnimation =
      typeof layer.opacity === 'object' ||
      typeof layer.transform.y === 'object' ||
      typeof layer.transform.scale === 'object' ||
      typeof layer.transform.rotation === 'object' ||
      typeof layer.transform.x === 'object';
    assert.equal(hasAnimation, true);
  }

  const floatScene = makeScene();
  applyEditCommands(floatScene, { op: 'animate', id: 'model', preset: 'float', start: 10, end: 70, intensity: 30 });
  assert.equal(sample(floatScene.layers[0]!.transform.y, 40, 0), 150);
  assert.equal(sample(floatScene.layers[0]!.transform.y, 10, 0), 180);
  assert.equal(sample(floatScene.layers[0]!.transform.y, 70, 0), 180);

  const defaultEnd = makeScene();
  applyEditCommands(defaultEnd, { op: 'animate', id: 'model', preset: 'fadeIn', start: 30 });
  const opacity = defaultEnd.layers[0]!.opacity as { keys: { t: number }[] };
  assert.equal(opacity.keys[opacity.keys.length - 1]!.t, 90);
});

test('previewAt 返回活跃图层、相机、音频与标记', () => {
  const scene = makeScene();
  applyEditCommands(scene, {
    op: 'setKeyframes',
    target: { kind: 'layer', id: 'model', path: 'transform.x' },
    keys: [{ t: 0, v: 0 }, { t: 120, v: 120 }],
  });
  applyEditCommands(scene, { op: 'setKeyframes', target: { kind: 'camera', path: 'zoom' }, keys: [{ t: 0, v: 1 }, { t: 120, v: 2 }] });
  const preview = previewAt(scene, 30);
  assert.equal(preview.frame, 30);
  assert.equal(preview.duration, 120);
  assert.equal(preview.camera!.zoom, 1.25);
  assert.equal(preview.activeLayers[0]!.x, 30);
  assert.equal(preview.activeLayers[0]!.model, 'm');
  assert.equal(preview.activeLayers[1]!.image, 'img');
  assert.equal(preview.audio.length, 1);
  assert.equal(preview.audio[0]!.active, true);
  assert.equal(preview.markers.length, 1);

  const noCamera = makeScene();
  applyEditCommands(noCamera, { op: 'setCamera', camera: null });
  assert.equal(previewAt(noCamera, 0).camera, null);
  assert.throws(() => previewAt(scene, -1), /不能小于 0/);
});

test('命令/字段别名兼容', () => {
  const scene = makeScene();
  applyEditCommands(scene, { op: 'updateScene', patch: { title: 'alias' } });
  assert.equal(scene.meta.title, 'alias');

  // target 省略 kind，用 layer 字段；addKeyframe 用 time/value 别名
  applyEditCommands(scene, {
    op: 'addKey',
    target: { layer: 'model', path: 'opacity' },
    time: 5,
    value: 0.25,
  } as never);
  assert.equal(opacityAt(scene, 5), 0.25);

  applyEditCommands(scene, {
    op: 'ease',
    target: { kind: 'layer', id: 'model', path: 'opacity' },
    t: 5,
    ease: 'step',
  });
  const curve = scene.layers[0]!.opacity as { keys: { ease?: unknown }[] };
  assert.equal(curve.keys[0]!.ease, 'step');

  applyEditCommands(scene, { op: 'nop' });
  parseScene(scene);
});

test('options.frame 与 preview 命令之外不生成 preview', () => {
  const scene = makeScene();
  assert.equal(applyEditCommands(scene, { op: 'noop' }).preview, undefined);
  const withFrame = applyEditCommands(scene, { op: 'noop' }, { frame: 25 });
  assert.equal(withFrame.preview!.frame, 25);
  assert.equal(withFrame.changes.length, 0);

  const withPreview = applyEditCommands(scene, { op: 'preview', frame: 10 });
  assert.equal(withPreview.preview!.frame, 10);
});
