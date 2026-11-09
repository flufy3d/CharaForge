# 外部饰品图片（ImageAsset / ImageLayer）

本期目标：把 AI 生成或手工准备的透明 PNG/WebP 作为场景资产，挂到模型图层上或独立放在画面里。图片是 **scene 层引用**，不修改模型文件。

> 当前进度：阶段 1 完成 schema、`parseScene` 语义校验和示例；阶段 2 完成 runtime 图片解码/纹理/绘制与 attach。导入 CLI、编辑器和 AI 命令层在后续阶段。

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
- 回归：旧全片场景 `scenes/deepseek-chan/scene.json` 的 1000–1099 共 100 帧，在阶段 2 改动前后逐帧 PNG SHA-256 完全相同。

## 示例与验证

- 示例：`packages/scene/examples/accessory.scene.json`（64×64 棋盘 data URI，含一个 attach 图层和一个独立屏幕图层）。
- schema：`packages/scene/schema/scene.v1.json`，由 `pnpm schema` 生成。
- 校验：

```bash
pnpm --filter @charaforge/scene test
pnpm test:runtime
pnpm -r run typecheck
pnpm render --scene packages/scene/examples/accessory.scene.json --frames 0-30 --out .data/render/acc
```

## 授权

AI 生成的饰品图可能带有来源/版权限制。不要把大图或版权不明的图提交进 git；导入目录只记录本地文件的 `sha256`/来源说明，随场景分发前请自行确认授权。
