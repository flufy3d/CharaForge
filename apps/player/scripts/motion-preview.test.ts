/*
 * 动作 / 表情查看器的纯逻辑测试：
 *   - motion3 Meta / exp3 Parameters 解析
 *   - 目录扫描结果 → 场景资源 uri
 *   - 临时预览 scene 能通过 parseScene
 *   - 模型画布 fit
 *
 *   pnpm --filter @charaforge/player test:viewer
 */
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { parseScene } from '@charaforge/scene';
import {
  ANIMATION_DIR,
  EXPRESSION_PREVIEW_FRAMES,
  PREVIEW_FPS,
  PREVIEW_MODEL_URI,
  THUMBNAIL_COLUMNS,
  THUMBNAIL_TILE,
  assetKey,
  assetUriFromDir,
  buildPreviewScene,
  buildThumbnailScene,
  catalogAssets,
  declaredAssetUri,
  declaredExpressionFiles,
  declaredMotionFiles,
  displayName,
  fitTransform,
  isExpressionFile,
  isMotionFile,
  parseExpressionInfo,
  parseMotionMeta,
  previewDuration,
  type Catalog,
  type ExpressionAssetInfo,
  type MotionAssetInfo,
} from '../src/motionCatalog.ts';

const REPO = resolve(import.meta.dirname, '..', '..', '..');
const ANIMATION = join(REPO, ANIMATION_DIR);
const EXP3 = join(REPO, ANIMATION_DIR.replace(/Animation$/, 'Exp3'));

const idleMeta = {
  Duration: 5.983,
  Fps: 60,
  Loop: true,
  CurveCount: 21,
  TotalSegmentCount: 518,
  TotalPointCount: 1473,
  UserDataCount: 0,
};

test('解析 motion3 Meta', () => {
  const meta = parseMotionMeta({ Version: 3, Meta: idleMeta, Curves: [] });
  assert.deepEqual(meta, {
    duration: 5.983,
    fps: 60,
    loop: true,
    curveCount: 21,
    totalSegmentCount: 518,
    totalPointCount: 1473,
    userDataCount: 0,
  });
  assert.equal(parseMotionMeta({ Meta: { Duration: 1 } }).curveCount, 0);
  assert.throws(() => parseMotionMeta({}), /Meta/);
  assert.throws(() => parseMotionMeta({ Meta: { Duration: 'x' } }), /Duration/);
});

test('解析 exp3 Parameters', () => {
  const info = parseExpressionInfo({
    Type: 'Live2D Expression',
    Parameters: [
      { Id: 'key3', Value: 1, Blend: 'Add' },
      { Id: 'key4', Value: -0.5, Blend: 'Multiply' },
    ],
  });
  assert.deepEqual(info.parameters, [
    { id: 'key3', value: 1, blend: 'Add' },
    { id: 'key4', value: -0.5, blend: 'Multiply' },
  ]);
  assert.deepEqual(parseExpressionInfo({ Type: 'Live2D Expression' }), { parameters: [] });
  assert.throws(() => parseExpressionInfo({ Parameters: {} }), /不是数组/);
});

test('model3 声明的 Motions / Expressions 解析（优先于目录）', () => {
  const model3 = {
    FileReferences: {
      Motions: { Idle: [{ File: 'Animation/IDLE.motion3.json' }], Tap: [{ File: 'Animation/tap.motion3.json' }] },
      Expressions: [{ Name: '脸红', File: 'Exp3/脸红.exp3.json' }],
    },
  };
  assert.deepEqual(declaredMotionFiles(model3), ['Animation/IDLE.motion3.json', 'Animation/tap.motion3.json']);
  assert.deepEqual(declaredExpressionFiles(model3), [{ file: 'Exp3/脸红.exp3.json', name: '脸红' }]);
  assert.deepEqual(declaredMotionFiles({}), []);
  assert.deepEqual(declaredExpressionFiles({ FileReferences: { Expressions: 'bad' } }), []);
});

test('文件名与资源 uri', () => {
  assert.equal(displayName('IDLE.motion3.json'), 'IDLE');
  assert.equal(displayName('脸红.exp3.json'), '脸红');
  assert.equal(isMotionFile('IDLE.motion3.json'), true);
  assert.equal(isMotionFile('IDLE.exp3.json'), false);
  assert.equal(isExpressionFile('脸红.exp3.json'), true);
  assert.equal(assetUriFromDir(ANIMATION_DIR, 'IDLE.motion3.json'), 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json');
  assert.equal(declaredAssetUri(PREVIEW_MODEL_URI, 'Animation/IDLE.motion3.json'), 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json');
  assert.equal(declaredAssetUri(PREVIEW_MODEL_URI, '../Other/a.motion3.json'), 'models/deepseek-chan/Other/a.motion3.json');
});

test('动作临时预览 scene 通过 parseScene', () => {
  const meta = parseMotionMeta({ Meta: idleMeta });
  const duration = previewDuration('motion', meta);
  assert.equal(duration, 359);
  const scene = buildPreviewScene({
    kind: 'motion',
    file: 'IDLE.motion3.json',
    uri: 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json',
    motionMeta: meta,
    loop: true,
  });
  const parsed = parseScene(scene);
  assert.equal(parsed.timebase.fps, PREVIEW_FPS);
  assert.equal(parsed.duration, 359);
  assert.equal(parsed.composition.width, 720);
  assert.equal(parsed.composition.height, 720);
  assert.equal(parsed.composition.background, '#202830');
  assert.equal(parsed.assets.models[0]!.uri, PREVIEW_MODEL_URI);
  assert.equal(parsed.assets.motions?.[0]!.uri, 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json');
  const layer = parsed.layers[0]!;
  assert.equal(layer.kind, 'model');
  if (layer.kind !== 'model') return;
  assert.equal(layer.motions?.[0]!.loop, true);
  assert.equal(layer.motions?.[0]!.fadeIn, 0);
  assert.equal(layer.physics?.warmup, 45);
  assert.equal(parsed.audio, undefined);
  assert.equal(parsed.assets.audio, undefined);
});

test('循环关闭与表达式临时预览 scene', () => {
  const meta = parseMotionMeta({ Meta: idleMeta });
  const off = buildPreviewScene({
    kind: 'motion',
    file: 'IDLE.motion3.json',
    uri: 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json',
    motionMeta: meta,
    loop: false,
  });
  const offScene = parseScene(off);
  const offLayer = offScene.layers[0]!;
  assert.equal(offLayer.kind, 'model');
  if (offLayer.kind === 'model') assert.equal(offLayer.motions?.[0]!.loop, false);

  const expr = buildPreviewScene({
    kind: 'expression',
    file: '脸红.exp3.json',
    uri: 'models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json',
  });
  const exprScene = parseScene(expr);
  assert.equal(exprScene.duration, EXPRESSION_PREVIEW_FRAMES);
  assert.equal(exprScene.assets.expressions?.[0]!.uri, 'models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json');
  const exprLayer = exprScene.layers[0]!;
  assert.equal(exprLayer.kind, 'model');
  if (exprLayer.kind === 'model') {
    assert.equal(exprLayer.expressions?.[0]!.expression, 'preview');
    assert.equal(exprLayer.expressions?.[0]!.fadeIn, 0);
    assert.equal(exprLayer.motions, undefined);
  }
});

test('表情预览可叠加 IDLE 动作', () => {
  const expr = buildPreviewScene({
    kind: 'expression',
    file: '脸红.exp3.json',
    uri: 'models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json',
    idleMotionUri: 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json',
  });
  const scene = parseScene(expr);
  assert.equal(scene.assets.motions?.[0]!.uri, 'models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json');
  assert.equal(scene.assets.expressions?.[0]!.uri, 'models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json');
  const layer = scene.layers[0]!;
  assert.equal(layer.kind, 'model');
  if (layer.kind === 'model') {
    assert.equal(layer.motions?.[0]!.motion, 'idle');
    assert.equal(layer.motions?.[0]!.loop, true);
    assert.equal(layer.expressions?.[0]!.expression, 'preview');
  }
});

test('缩略图 contact sheet scene 通过 parseScene', () => {
  const meta = { duration: 2, fps: 60, loop: true, curveCount: 1, totalSegmentCount: 1, totalPointCount: 1, userDataCount: 0 };
  const motion = (file: string): MotionAssetInfo => ({
    kind: 'motion',
    name: displayName(file),
    file,
    uri: `models/deepseek-chan/DeepSeek/Animation/${file}`,
    dir: ANIMATION_DIR,
    size: 100,
    mtime: '',
    source: 'directory',
    meta,
  });
  const expression = (file: string): ExpressionAssetInfo => ({
    kind: 'expression',
    name: displayName(file),
    file,
    uri: `models/deepseek-chan/DeepSeek/Exp3/${file}`,
    dir: ANIMATION_DIR.replace(/Animation$/, 'Exp3'),
    size: 100,
    mtime: '',
    source: 'directory',
    info: { parameters: [{ id: 'key1', value: 1, blend: 'Add' }] },
  });
  const cat: Catalog = {
    modelUri: PREVIEW_MODEL_URI,
    motions: [motion('a.motion3.json'), motion('b.motion3.json')],
    expressions: [expression('脸红.exp3.json'), expression('爱心眼.exp3.json')],
    sources: { motions: 'directory', expressions: 'directory' },
    warnings: [],
  };
  assert.deepEqual(catalogAssets(cat).map(assetKey), [
    'motion:a.motion3.json',
    'motion:b.motion3.json',
    'expression:脸红.exp3.json',
    'expression:爱心眼.exp3.json',
  ]);

  const scene = parseScene(buildThumbnailScene(cat));
  assert.equal(scene.composition.width, THUMBNAIL_COLUMNS * THUMBNAIL_TILE);
  assert.equal(scene.composition.height, THUMBNAIL_TILE);
  assert.equal(scene.duration, 1);
  assert.equal(scene.assets.motions?.length, 2);
  assert.equal(scene.assets.expressions?.length, 2);
  assert.equal(scene.layers.length, 4);
  assert.deepEqual(scene.layers.map((l) => l.id), ['thumb.m0', 'thumb.m1', 'thumb.e2', 'thumb.e3']);
  for (const [i, layer] of scene.layers.entries()) {
    assert.equal(layer.kind, 'model');
    if (layer.kind !== 'model') continue;
    assert.equal(layer.physics?.enabled, false);
    assert.equal(layer.in, 0);
    assert.equal(layer.out, 1);
    assert.deepEqual(layer.clip, {
      x: (i % THUMBNAIL_COLUMNS) * THUMBNAIL_TILE,
      y: Math.floor(i / THUMBNAIL_COLUMNS) * THUMBNAIL_TILE,
      width: THUMBNAIL_TILE,
      height: THUMBNAIL_TILE,
    });
    if (i < 2) {
      assert.equal(layer.motions?.[0]!.loop, true);
      assert.ok((layer.motions?.[0]!.offset ?? 0) > 0);
    } else {
      assert.equal(layer.expressions?.[0]!.start, 0);
    }
  }
});

test('模型画布等比 fit 到预览画面', () => {
  const t = fitTransform({ width: 4096, height: 2048 }, 720, 720, 0.96);
  assert.equal(t.x, 360);
  assert.equal(t.y, 360);
  assert.equal(t.scale, (720 / 4096) * 0.96);
  assert.throws(() => fitTransform({ width: 0, height: 1 }, 720, 720), /画布尺寸/);
});

test('真实素材存在时：8 个 motion3 与 23 个 exp3 都能解析', (t) => {
  if (!existsSync(ANIMATION) || !existsSync(EXP3)) {
    t.skip('assets/ 未安装（先运行 pnpm setup:assets）');
    return;
  }
  const motionFiles = readdirSync(ANIMATION).filter(isMotionFile);
  const expressionFiles = readdirSync(EXP3).filter(isExpressionFile);
  assert.equal(motionFiles.length, 8);
  assert.equal(expressionFiles.length, 23);
  const metas = motionFiles.map((f) => parseMotionMeta(JSON.parse(readFileSync(join(ANIMATION, f), 'utf8')), f));
  assert.ok(metas.some((m) => m.duration === 5.983 && m.loop && m.curveCount === 21));
  assert.ok(metas.some((m) => m.duration === 13.15 && m.curveCount === 24));
  for (const f of expressionFiles) parseExpressionInfo(JSON.parse(readFileSync(join(EXP3, f), 'utf8')), f);
});
