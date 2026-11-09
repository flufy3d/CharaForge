/*
 * 饰品 AI 命令层（CLI / JSON）。
 *
 * 输入：
 *   pnpm decoration:command --scene <scene.json> --input <commands.json> [--out <scene.json>] [--dry-run]
 *   或从 stdin 读 JSON。
 *
 * 命令（commands 数组，或单个命令对象）：
 *   { "op": "addDecoration", "asset": ImageAsset, "layer": ImageLayer }
 *   { "op": "updateDecoration", "id": "<image-layer-id>", "patch": { transform?, attach?, opacity?, ... } }
 *   { "op": "removeDecoration", "id": "<image-layer-id>" }
 *   { "op": "setDecorationVisible", "id": "<image-layer-id>", "visible": true|false }
 *   { "op": "preview", "frame": 0 }
 *
 * 行为：
 *   - 每个命令后都跑 parseScene；非法 id、尺寸、URI、attach 目标等直接失败；
 *   - 不写回 --scene；--out 才导出新 scene；--dry-run 连 --out 也不写；
 *   - stdout 返回 { ok, diff, preview, ... }，diff 为简化的 JSON Patch 风格；
 *   - 修改的 ImageLayer 带 provenance.source = "ai"。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { parseScene } from '@charaforge/scene';

const OPS = new Set(['addDecoration', 'updateDecoration', 'removeDecoration', 'setDecorationVisible', 'preview']);
const UPDATE_KEYS = new Set(['name', 'in', 'out', 'space', 'transform', 'opacity', 'clip', 'filters', 'blendMode', 'attach', 'semantics']);
const AI_TOOL = 'decoration-command';
const ID_RE = /^[A-Za-z0-9_.:-]+$/;

function usage() {
  console.error(`用法：pnpm decoration:command --scene <scene.json> [--input <commands.json>] [--out <scene.json>] [--dry-run]

不传 --input 时从 stdin 读 JSON。命令格式见 apps/player/scripts/decoration-command.mjs 文件头。
`);
}

function fail(msg) {
  throw new Error(msg);
}

function clone(v) {
  return structuredClone(v);
}

function readInput(path) {
  const text = !path || path === '-' ? readFileSync(0, 'utf8') : readFileSync(resolve(process.env.INIT_CWD ?? process.cwd(), path), 'utf8');
  try {
    return JSON.parse(text);
  } catch (e) {
    fail(`命令 JSON 解析失败：${String(e?.message ?? e)}`);
  }
}

function commandsOf(v) {
  if (Array.isArray(v)) return v;
  if (v && Array.isArray(v.commands)) return v.commands;
  if (v && typeof v.op === 'string') return [v];
  fail('输入必须是命令数组、{ commands: [...] } 或单个命令对象');
}

function imageLayerIndex(scene, id) {
  return scene.layers.findIndex((l) => l.kind === 'image' && l.id === id);
}

function checkId(id, what) {
  if (typeof id !== 'string' || !ID_RE.test(id)) fail(`${what} 非法：${id}`);
}

function provenance(op) {
  return { source: 'ai', tool: AI_TOOL, detail: { command: op } };
}

function previewOf(scene, frame) {
  return {
    frame,
    duration: scene.duration,
    composition: scene.composition,
    layers: scene.layers.length,
    imageAssets: (scene.assets.images ?? []).map((a) => ({ id: a.id, width: a.width, height: a.height, uri: a.uri })),
    imageLayers: scene.layers
      .filter((l) => l.kind === 'image')
      .map((l) => ({ id: l.id, image: l.image, attach: l.attach?.layer ?? null, opacity: typeof l.opacity === 'number' ? l.opacity : 1 })),
  };
}

function applyCommand(scene, cmd, changes, previews, index) {
  if (!cmd || typeof cmd !== 'object') fail(`命令 #${index} 不是对象`);
  if (!OPS.has(cmd.op)) fail(`命令 #${index} 未知 op：${cmd.op}`);

  if (cmd.op === 'addDecoration') {
    if (!cmd.asset || typeof cmd.asset !== 'object') fail(`addDecoration 缺少 asset`);
    if (!cmd.layer || typeof cmd.layer !== 'object') fail(`addDecoration 缺少 layer`);
    if (cmd.layer.kind !== 'image') fail(`addDecoration.layer.kind 必须是 "image"`);
    checkId(cmd.asset.id, 'asset.id');
    checkId(cmd.layer.id, 'layer.id');
    if (scene.layers.some((l) => l.id === cmd.layer.id)) fail(`图层 ID 已存在：${cmd.layer.id}`);
    if ((scene.assets.images ?? []).some((a) => a.id === cmd.asset.id)) fail(`图片资源 ID 已存在：${cmd.asset.id}`);
    const asset = clone(cmd.asset);
    const layer = clone(cmd.layer);
    layer.provenance = provenance('addDecoration');
    if (!scene.assets.images) scene.assets.images = [];
    scene.assets.images.push(asset);
    scene.layers.push(layer);
    parseScene(scene);
    changes.push({ op: 'add', path: `/assets/images/${scene.assets.images.length - 1}`, value: clone(asset) });
    changes.push({ op: 'add', path: `/layers/${scene.layers.length - 1}`, value: clone(layer) });
    return;
  }

  if (cmd.op === 'updateDecoration') {
    checkId(cmd.id, 'id');
    const idx = imageLayerIndex(scene, cmd.id);
    if (idx < 0) fail(`找不到图片图层：${cmd.id}`);
    const patch = cmd.patch;
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) fail(`updateDecoration.patch 必须是对象`);
    for (const k of Object.keys(patch)) if (!UPDATE_KEYS.has(k)) fail(`updateDecoration 不允许修改字段：${k}`);
    const layer = scene.layers[idx];
    const before = clone(layer);
    if ('transform' in patch) {
      if (!patch.transform || typeof patch.transform !== 'object') fail('transform 必须是对象');
      layer.transform = { ...layer.transform, ...clone(patch.transform) };
    }
    if ('attach' in patch) {
      if (patch.attach === null) delete layer.attach;
      else if (!patch.attach || typeof patch.attach !== 'object') fail('attach 必须是对象或 null');
      else layer.attach = { ...(layer.attach ?? {}), ...clone(patch.attach) };
    }
    for (const k of ['name', 'in', 'out', 'space', 'opacity', 'clip', 'filters', 'blendMode', 'semantics']) {
      if (k in patch) layer[k] = clone(patch[k]);
    }
    layer.provenance = provenance('updateDecoration');
    parseScene(scene);
    changes.push({ op: 'replace', path: `/layers/${idx}`, before, after: clone(layer) });
    return;
  }

  if (cmd.op === 'removeDecoration') {
    checkId(cmd.id, 'id');
    const idx = imageLayerIndex(scene, cmd.id);
    if (idx < 0) fail(`找不到图片图层：${cmd.id}`);
    const before = clone(scene.layers[idx]);
    scene.layers.splice(idx, 1);
    changes.push({ op: 'remove', path: `/layers/${idx}`, before });
    const imageId = before.image;
    if (imageId && !scene.layers.some((l) => l.kind === 'image' && l.image === imageId)) {
      const ai = (scene.assets.images ?? []).findIndex((a) => a.id === imageId);
      if (ai >= 0) {
        const asset = clone(scene.assets.images[ai]);
        scene.assets.images.splice(ai, 1);
        changes.push({ op: 'remove', path: `/assets/images/${ai}`, before: asset });
      }
    }
    parseScene(scene);
    return;
  }

  if (cmd.op === 'setDecorationVisible') {
    if (typeof cmd.visible !== 'boolean') fail('setDecorationVisible.visible 必须是布尔值');
    checkId(cmd.id, 'id');
    const idx = imageLayerIndex(scene, cmd.id);
    if (idx < 0) fail(`找不到图片图层：${cmd.id}`);
    const layer = scene.layers[idx];
    const before = clone(layer);
    layer.opacity = cmd.visible ? (typeof layer.opacity === 'number' && layer.opacity > 0 ? layer.opacity : 1) : 0;
    layer.provenance = provenance('setDecorationVisible');
    parseScene(scene);
    changes.push({ op: 'replace', path: `/layers/${idx}`, before, after: clone(layer) });
    return;
  }

  // preview
  const frame = cmd.frame ?? 0;
  if (!Number.isInteger(frame) || frame < 0) fail('preview.frame 必须是非负整数');
  previews.push(previewOf(scene, frame));
}

async function main() {
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const { values } = parseArgs({
    args: argv,
    options: {
      scene: { type: 'string' },
      input: { type: 'string' },
      out: { type: 'string' },
      'dry-run': { type: 'boolean' },
      help: { type: 'boolean' },
    },
  });
  if (values.help) {
    usage();
    return;
  }
  if (!values.scene) {
    usage();
    process.exitCode = 2;
    return;
  }

  const scenePath = resolve(process.env.INIT_CWD ?? process.cwd(), values.scene);
  const scene = parseScene(JSON.parse(readFileSync(scenePath, 'utf8')));
  const input = readInput(values.input);
  const commands = commandsOf(input);
  const changes = [];
  const previews = [];

  commands.forEach((cmd, i) => applyCommand(scene, cmd, changes, previews, i));

  const outPath = values.out ? resolve(process.env.INIT_CWD ?? process.cwd(), values.out) : null;
  if (outPath && !values['dry-run']) writeFileSync(outPath, JSON.stringify(scene, null, 2) + '\n');

  const result = {
    ok: true,
    scene: scenePath,
    out: outPath,
    dryRun: !!values['dry-run'],
    commandCount: commands.length,
    changed: changes.length > 0,
    diff: changes,
    preview: previews.length ? previews[previews.length - 1] : null,
  };
  console.log(JSON.stringify(result, null, 2));
}

main().catch((e) => {
  console.error(JSON.stringify({ ok: false, error: String(e?.message ?? e) }, null, 2));
  process.exit(1);
});
