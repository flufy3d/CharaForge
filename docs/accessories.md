# 外部饰品图片（ImageAsset / ImageLayer）

本期目标：把 AI 生成或手工准备的透明 PNG/WebP 作为场景资产，挂到模型图层上或独立放在画面里。图片是 **scene 层引用**，不修改模型文件。

> 当前进度：阶段 1 完成 schema、`parseScene` 语义校验和示例；阶段 2 完成 runtime 图片解码/纹理/绘制与 attach；阶段 3 完成导入 CLI 和 `assets/accessories/index.json`；阶段 4 完成播放器内的最小饰品编辑器（`?edit=1`）；阶段 5 完成 AI 命令层（CLI/JSON）。

## 数据

### ImageAsset

```json
{ "id": "hat", "uri": "accessories/hat.png", "width": 512, "height": 512, "sha256": "…" }
```

- 放在 `assets.images`（可选数组）。
- `uri` 相对于资源根目录 `assets/`；也允许 `data:image/png;base64,…` / `data:image/webp;base64,…`，示例和测试用。
- 单边 ≤ 4096，总像素 ≤ 16 MP（16,777,216）；运行时解码后的尺寸必须与 `width/height` 一致。
- `sha256` 可选，供导入目录/去重使用；runtime 不重新校验文件摘要。
- 只接受 PNG / WebP。图片是不可信数据：只解析、上传纹理，不执行任何内容。

### ImageLayer

```json
{
  "id": "hat",
  "kind": "image",
  "image": "hat",
  "in": 0,
  "out": 240,
  "transform": { "x": 0, "y": 0, "scale": 8, "anchorX": 256, "anchorY": 256 },
  "opacity": 1,
  "attach": { "layer": "main", "anchor": [0.5, 0.12], "offset": [0, 0] }
}
```

- `transform` 复用 `Transform2D`；对图片图层，`anchorX/anchorY` 是图片像素，缺省为图片中心。
- `opacity`、`clip`、`filters`、`blendMode` 与模型图层同语义。
- 图片图层没有 `physics`、`motions`、`expressions`、`params`、`partOpacity`、`frameHold`。
- 图层顺序就是 z 顺序；图片可以和模型交错，`attach` 不改变层级。
- 有 `attach` 时 `space` 被忽略，由目标模型图层的 `space` 决定是否受相机影响。

## attach 语义（MVP）

只跟随目标模型图层的整体变换，**不解析骨骼、参数或部件**。目标必须是模型图层；目标在 `f` 不活跃时图片不绘制。

```text
p = 图片局部坐标 (0..W, 0..H)
ax = anchorX ?? W/2, ay = anchorY ?? H/2
sx = scale·scaleX·(flipX ? -1 : 1), sy = scale·scaleY

Own(p) = (x,y) + R(rotation)·(sx·(px−ax), sy·(py−ay))

无 attach:
  Screen(p) = space === "screen" ? Own(p) : Camera(f)·Own(p)

有 attach:
  g  = 目标有 frameHold 时 holdFrame(target, f)，否则 f
  Lt = T(target.x,target.y)
     · R(rotateWith ? target.rotation : 0)
     · diag(scaleWith ? target.sx : 1, scaleWith ? target.sy : 1)
     · T(−target.anchorX,−target.anchorY)
  Mt = target.space === "screen" ? Lt : Camera(g)·Lt
  A  = (anchor[0]·targetCanvas.width, anchor[1]·targetCanvas.height)
  O  = offset ?? (0,0)
  Screen(p) = Mt·(A + O + Own(p))
```

- `anchor` 是目标模型画布的归一化坐标（0..1）。
- `offset` 与 attach 时的 `transform.x/y` 都按目标模型画布像素解释。
- `rotateWith`、`scaleWith` 缺省 `true`；`scaleWith=false` 时不继承目标缩放和 `flipX`。
- 已知限制：头部转动、表情、物理等骨骼级变化不会带动饰品；需要更细的绑定时再引入骨骼/部件锚点。

## runtime 实现

- `ImageResource`（`packages/runtime/src/resources.ts`）：检查 PNG/WebP 文件头 → `createImageBitmap(premultiplyAlpha: "premultiply")` → Pixi `ImageSource`（`alphaMode: "premultiplied-alpha"`、线性过滤、自动 mipmap）→ 按绝对 URL 缓存。
- 解码后的实际尺寸必须与 `ImageAsset.width/height` 一致，否则加载失败；运行时也检查 4096/16MP 上限。
- `Compositor.drawImage` 用 Pixi `Sprite` + `setFromMatrix` 把图片纹理画进图层缓冲，再复用现有 `filters → clip → opacity → blendMode` 输出链。
- 图层计划按 `scene.layers` 数组顺序生成；图片和模型可以交错，attach 不改变 z 顺序。
- 图片图层不进入 `LayerEvaluator`，不推进物理/不参与 Core update；Worker 只接收模型图层。
- 目标模型图层在 `f` 不活跃时，attach 图片不绘制；目标 `opacity=0` 时仍按其变换绘制图片。
- `rotateWith/scaleWith` 缺省 true；关闭时分别剥离目标旋转/缩放（`scaleWith=false` 同时不继承 `flipX`），平移和相机始终保留。
- 回归：旧全片场景 `scenes/deepseek-chan/scene.json` 的 1000–1099 共 100 帧，在阶段 2 改动前后、阶段 6 最终复跑时逐帧 PNG SHA-256 完全相同（等价 SSIM 1.0）。

## 导入 CLI

```bash
pnpm accessory:import --input .data/hat.png --id hat --name "帽子" \
  --scene packages/scene/examples/accessory.scene.json \
  --attach main --anchor 0.5,0.12 --offset 0,0
```

- 只接受 PNG / WebP（按文件头判断，不看扩展名）；用浏览器 canvas 解码检查尺寸、透明像素和生成缩略图，不执行图片内容。
- 要求至少有一个非不透明像素（`alpha < 255`）和至少一个可见像素；全不透明图会被拒绝。
- 单边 ≤ 4096、总像素 ≤ 16 MP、文件 ≤ 16 MB。
- 输出：
  - `assets/accessories/<id>.png|webp`
  - `assets/accessories/<id>.thumb.png`（96×96）
  - `assets/accessories/index.json`（`id/name/file/size/bytes/sha256/thumbnail`）
- stdout 打印 `{ "asset": ImageAsset, "layer": ImageLayer }`；asset 粘到 `assets.images`，layer 粘到 `layers`。默认不改场景。
- `--scene` 只用于读取 `duration/composition` 和校验 attach 目标，不会写回场景。
- `--force` 覆盖已有 id/文件；`--dry-run` 只校验并打印 JSON。
- 不传 `--attach` 时生成独立 `space:"screen"` 图层，位置为画面中心。
- 解码/缩略图使用仓库已有的 Playwright Chromium（与 `pnpm render` 相同），不新增依赖；首次使用需 `pnpm exec playwright install chromium-headless-shell`。

## 播放器编辑器（`?edit=1`）

```bash
pnpm dev
# Chrome / Edge 打开：
# http://localhost:5173/?scene=packages/scene/examples/accessory.scene.json&edit=1
```

- 左侧列出所有图片图层；「导入图片」接受 PNG/WebP，导入后先挂到「导入默认绑定」选择的模型图层（缺省第一个模型图层），锚点缺省 `[0.5,0.15]`。
- 选中图层后可调 X/Y、缩放、旋转、不透明度、attach 目标、anchor、offset、跟随旋转/缩放和可见开关。所有修改直接改当前 scene 对象并 `engine.redraw()`，不重载模型。
- 导入的图片先用 blob URL 预览；保存时替换成 `assets/accessories/<id>.png|webp` 并写 `sha256`。
- 保存优先用 File System Access API：第一次点「保存」选择**仓库根目录**，之后写 scene json 和 `assets/accessories/`；不支持时降级为下载 scene.json + 图片文件。
- 保存路径可在面板里改；默认取 `?scene=` 的路径，没有 `?scene=` 时用示例场景，避免误覆盖全片场景。
- 验收路径：打开示例场景 → 导入透明帽子 → 调 attach/缩放/位置 → 保存 → 刷新同一 URL，图层仍在。
- 编辑器导入的图片不更新 `assets/accessories/index.json`（目录由 `pnpm accessory:import` 维护）；场景本身仍是唯一编辑源。

## AI 命令层（CLI/JSON）

```bash
pnpm --silent decoration:command \
  --scene packages/scene/examples/accessory.scene.json \
  --input .data/command.json \
  --out .data/command.scene.json
```

输入是命令数组、`{ "commands": [...] }` 或单个命令：

```json
{ "op": "addDecoration", "asset": { "id": "hat", "uri": "accessories/hat.png", "width": 512, "height": 512 }, "layer": { "id": "hat", "kind": "image", "image": "hat", "in": 0, "out": 240, "transform": { "x": 0, "y": 0, "scale": 2 }, "attach": { "layer": "main", "anchor": [0.5, 0.15] } } }
{ "op": "updateDecoration", "id": "hat", "patch": { "transform": { "scale": 3 }, "attach": { "anchor": [0.5, 0.12] } } }
{ "op": "removeDecoration", "id": "hat" }
{ "op": "setDecorationVisible", "id": "hat", "visible": false }
{ "op": "preview", "frame": 0 }
```

- 每个命令后都执行 `parseScene`；非法 id、尺寸、URI、attach 目标会直接失败。
- `addDecoration` 的 `asset.uri` 指向的图片需已由 `pnpm accessory:import` 放到 `assets/accessories/`（命令层不复制文件）。
- `addDecoration/updateDecoration/setDecorationVisible` 写入的图层带 `provenance: { source: "ai", tool: "decoration-command" }`。
- `updateDecoration.patch` 可改 `name/in/out/space/transform/opacity/clip/filters/blendMode/attach/semantics`；`transform`/`attach` 是浅合并，`attach: null` 解除绑定。
- `removeDecoration` 删除图层；图片资源不再被任何图片图层引用时一并删除。
- `preview` 不改场景，返回该时刻的图层/资源摘要；实际预览用 `pnpm render --scene <out>`。
- 默认不写回 `--scene`；`--out` 才导出新场景。`--dry-run` 只校验和返回 diff，不写 `--out`。
- stdout 是 JSON；用 `pnpm --silent` 避免 pnpm 的脚本标题混进 stdout。diff 是简化的 JSON Patch（`add/replace/remove` + 路径 + before/after）。
- 目前是 CLI/JSON；HTTP/MCP 层后续再接。

## 示例与验证

- 示例：`packages/scene/examples/accessory.scene.json`（64×64 棋盘 data URI，含一个 attach 图层和一个独立屏幕图层）。
- schema：`packages/scene/schema/scene.v1.json`，由 `pnpm schema` 生成。
- 校验：

```bash
pnpm --filter @charaforge/scene test
pnpm test:runtime
pnpm -r run typecheck
pnpm accessory:import --input .data/import/hat.png --id hat --scene packages/scene/examples/accessory.scene.json --attach main --dry-run
pnpm --silent decoration:command --scene packages/scene/examples/accessory.scene.json --input .data/command.json --out .data/command.scene.json --dry-run
pnpm dev   # ?scene=packages/scene/examples/accessory.scene.json&edit=1
pnpm render --scene packages/scene/examples/accessory.scene.json --frames 0-30 --out .data/render/acc
```

## 已知限制

- attach 只跟随目标模型图层的整体变换，不跟骨骼/参数/部件；头部转动、表情、物理不会带动饰品。
- 目标模型图层在 `f` 不活跃时 attach 图片不绘制；目标 `opacity=0` 时仍按变换绘制图片。
- 图片纹理按 URL 跨场景缓存，只在 runtime `dispose()` 时统一释放。
- 编辑器导入的图片不更新 `assets/accessories/index.json`；目录由导入 CLI 维护。
- AI 命令层目前是 CLI/JSON，没有 HTTP 服务或 MCP；`preview` 返回摘要，实际预览用 `pnpm render`。
- 图片导入/缩略图依赖 Playwright Chromium；没有 ffmpeg 回退。
- `assets/` 不进 git；场景引用本地图片时需随场景一起拷贝 `assets/accessories/`。

## 授权

AI 生成的饰品图可能带有来源/版权限制。不要把大图或版权不明的图提交进 git；导入目录只记录本地文件的 `sha256`/来源说明，随场景分发前请自行确认授权。

AI 生图工具调研见 [imagine-agy.md](imagine-agy.md)：`imagine-agy` 可以生成 JPEG，但没有 alpha，不能直接当透明饰品；扁平饰品推荐让 agy 生成 SVG 后再栅格化。
