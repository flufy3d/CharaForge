# 场景 / 动作编辑器

本期把 CharaForge 从「只能播放复刻好的全片」推进到「能编辑并导出 MV」：

- **场景编辑**：图层树、变换、透明度、裁切、滤镜、混合模式、相机、背景色、外部图片/生成元素。
- **动作编辑**：motion3 / exp3 片段、参数轨、部件不透明度、物理开关；动作库直接列出模型 `Animation/` 与 `Exp3/`。
- **多轨时间线**：镜头标记、相机、47 个图层、动作/表情片段、任意多条音频轨（音乐轨）。
- **关键帧与插值**：线性 / 平滑 / easeIn / easeOut / easeInOut / step，以及自定义三次贝塞尔；一键动画预设。
- **MV 导出**：逐帧渲染 + 离线混音，编码为 MP4（AVC/AAC）或 WebM（VP9/Opus），带音乐。
- **人和 AI 都能改**：浏览器 `window.__editor`、本地 HTTP 命令接口、CLI 命令层共用同一套 schema。

入口：

```bash
pnpm dev
# 根路径默认就是编辑器：
# http://localhost:5173/
# 直接打开全片 MV：
# http://localhost:5173/?scene=scenes/deepseek-chan/scene.mv.json&frame=2400
# 轻量示例（背景光晕 + 星星元素 + 音乐）：
# http://localhost:5173/?scene=packages/scene/examples/editor-demo.scene.json
```

`pnpm preview` 使用同一个中间件，也可以打开 `http://localhost:4173/`。
`index.html`（默认编辑器）、`editor.html`（直接入口）、`player.html`（独立播放器）都已加入 `vite.config.ts` 的构建输入；
`editor.html` 与 `index.html` 内容相同，便于旧链接继续使用。独立播放器只作为预览/参考视频对照页，主入口是编辑器。

轻量示例场景：`packages/scene/examples/editor-demo.scene.json`（6 秒，背景光晕 + 两颗星星 + 粉色气泡 + IDLE 动作 + 音乐轨），
适合快速试时间线、关键帧和导出。

正式 MV 有两个版本：

- `scenes/deepseek-chan/scene.json`：原始拟合场景（47 个角色图层，透明背景，用于回归/对比）。
- `scenes/deepseek-chan/scene.mv.json`：在原场景上叠加程序生成的近似背景与小元素（点阵边框、天空、深色光晕、照片卡、气泡、闪光、聊天窗口、转场泼溅等，共 128 个图层）。编辑器默认打开这一版。

背景版的主要图层：

| 图层 | 时段 | 内容 |
|---|---|---|
| `bg.dots` | 全片 | 浅蓝点阵 + 蓝色白点边框 |
| `bg.sky.1/2` | 254–676、1157–1198 | 天空/云朵（Photo Booth、泼溅后） |
| `bg.dark.1..4` | 1527–1707、2639–2690、2969–3097、3478–3662 | 深蓝径向光晕 |
| `bg.orange` / `bg.watercolor` | 3280–3296 / 3382–3408 | 橙色半调 / 水彩云雾转场 |
| `chat` / `booth` | 0–254 / 254–394 | 聊天窗口 / Photo Booth 白框 |
| `halo.*` | 多个片段 | 角色背后的圆环光晕 |
| `card.*` / `cardWide.*` | 多处 | 白色照片卡（简笔角色剪影） |
| `bubble.*` / `sparkle.*` | s19 之后多处 | 蓝色气泡、四角闪光 |
| `splash.*` | 转场帧 | 蓝/红蓝/白/深蓝泼溅、白色剪影 |
| `sticker.question` / `sticker.check` | s22–24 / s26 | 红色问号、对勾 |

觉得某层太抢眼时，在时间线或图层列表选中它，改 `opacity` 或直接删除即可；背景图都是内嵌 WebP data URI，不依赖 `assets/` 里的外部文件。

## 界面

```text
┌ 顶部：场景路径 / 保存路径 / 保存 / 撤销 / 重做 / 导出视频 / AI 徽标 ┐
├ 左栏（图层 / 素材 / 动作 / 元素）┬ 预览舞台 ┬ 右栏（检查器） ┤
│ 图层树、导入、动作库、生成元素   │ 画布     │ 场景/图层/音频/标记 │
├────────────────────────────── 多轨时间线 ──────────────────────────┤
│ 镜头标记 / 相机 / 音频轨 / 图层行 / 展开的属性轨 / 固定参数轨        │
└────────────────────────────────────────────────────────────────────┘
```

快捷键：`空格` 播放/暂停，`←`/`→` ±1 帧（`Shift` ±10），`Home`/`End`，
`Ctrl/Cmd+Z` 撤销，`Ctrl/Cmd+Shift+Z` / `Ctrl/Cmd+Y` 重做，`Ctrl/Cmd+S` 保存，
`Delete` 删除选中的时间线关键帧，`E` 打开导出。

### 场景编辑

- 左栏「图层」显示全部图层（顶部 = 最上层）。可选中、显示/隐藏、上移/下移、复制、删除。
- 右栏检查器按类型显示字段：
  - 模型图层：入点/出点、space、blendMode、透明度、变换、裁切、滤镜、物理、motion/expression、参数轨。
  - 图片图层：图片资源、变换、裁切、透明度、attach 目标/anchor/offset、跟随旋转/缩放。
- 舞台可直接拖动选中图层（平移）。有曲线时在播放头写关键帧，不会把拟合曲线压平。
- 「场景」检查器可改标题、描述、帧率、总帧数、画面尺寸、背景色/透明、相机。
- 属性字段旁的 **◆** 按钮：当前帧有关键帧则删除，没有则在当前帧添加。
- 关键帧行支持「上一个 / 下一个」跳转；缓动下拉框作用于当前关键帧（或时间线选中的关键帧）。

### 素材 / 元素 / 背景

「素材」面板列出场景里的图片与音频资源，可以导入：

- 图片：PNG / WebP（透明背景），上传到 `assets/editor/`，也可在接口不可用时内嵌 data URI。
- 音频：WAV / MP3 / OGG / M4A，上传到 `assets/editor/`；添加后成为独立的音频轨。

「元素」面板用于快速做背景和小元素：

- 生成圆形、星星、爱心、闪光、气泡、文字 PNG（画布 2D 生成、透明背景），直接变成图片图层。
- 生成线性渐变背景，自动插到图层最底部并铺满画面。
- 「跟随选中角色」使用 `ImageLayer.attach`，元素跟随角色的整体变换。
- 一键动画：淡入、淡出、漂浮、脉冲、旋转、弹跳、闪烁、滑入、抖动、放大。
  这些会生成关键帧，之后仍可在时间线里微调。

### 动作 / 表情

「动作」面板通过 `/__charaforge/list-assets` 列出模型 `Animation/` 与 `Exp3/` 下的素材
（当前 DeepSeek-chan：8 个 motion3、23 个 exp3），也可以用 `model3.json` 的声明优先。

- 选中模型图层后点击「添加动作 / 添加表情」，片段起点 = 当前播放头。
- 时间线中动作/表情条可拖动、左右裁剪；右栏可改 start/end/offset/speed/loop/fadeIn/fadeOut/weight。
- 参数轨：右栏参数列表支持搜索；◆ 写关键帧，「固定」把该参数轨显示到时间线底部。

### 多轨音频

- 每次「添加音频」生成一条新的音频轨（可重复添加同一素材，形成多轨）。
- 时间线音频条：拖动移动起点，左端裁剪改变 `offset`，右端裁剪改变 `duration`；条内填充高度表示 `gain`。
- 右栏「音频」检查器可改 start / offset / duration / gain、删除片段、跳到片段起点。
- 播放和导出都按同一套音频片段语义混音（`scene.audio`），不是简单播放一个整段音频。

### 时间线

- 横向缩放：`Ctrl/Cmd + 滚轮`，或工具栏 `− / + / 适应`。
- `Shift + 滚轮` 横向滚动；普通滚轮纵向滚动。
- 关键帧：拖动改时间（默认吸附到整数帧，`Alt` 临时关闭吸附），单击选中，`Shift` 多选，双击跳到该帧。
- 工具栏「删除关键帧」批量删除选中项；缓动下拉可批量应用插值。
- 图层行边缘可拖入点/出点；动作/表情/音频条可移动和裁剪。
- 底部「参数」组显示固定的参数轨。

## 保存与打开

- 顶部「场景」是打开路径，「保存到」是写盘路径，两者可以不同。
- 保存通过本地接口 `POST /__charaforge/editor/save-scene` 写到 `scenes/` 或 `packages/scene/examples/`；
  覆盖已有文件时自动写 `<path>.bak.json`。
- 接口不可用时降级为下载 `scene.json`。
- 大场景（> 20 图层或 > 2000 条参数轨）保存为紧凑 JSON，避免 4 MB 场景展开成十几 MB；
  小示例仍然缩进 2 空格，方便阅读和 diff。
- 保存前执行 `parseScene`，非法场景不会写盘。

## 导出 MV 视频（带音乐）

### 编辑器按钮

顶部「导出视频」：

- 范围：全部 / 当前镜头 / 自定义起始帧到结束帧（结束帧不含）。
- 格式：自动（优先 MP4/AVC+AAC，失败回退 WebM/VP9+Opus）、MP4、WebM。
- 质量：high / medium / low。
- 背景：默认黑；透明场景导出视频时没有 alpha，背景必须显式合成。
- 导出时暂停预览并把每帧 `renderFrameAsync → present → 2D 画布 → 编码`；
  音频用 `OfflineAudioContext` 按 `scene.audio` 重新混音，再与视频一起封装。
- 完成后直接下载 `<标题>.mp4|webm`；导出期间右上角显示音频/视频/封装进度。

### CLI

```bash
# 全片（60 秒，3868 帧，MP4/AVC+AAC）
pnpm export:video --scene scenes/deepseek-chan/scene.mv.json --out .data/export/deepseek-full.mp4 --workers 2

# 指定帧范围、WebM、低质量
pnpm export:video --scene scenes/deepseek-chan/scene.mv.json --from 2368 --to 2442 \
  --format webm --quality low --out .data/export/s24.webm

# 覆盖尺寸/背景
pnpm export:video --scene packages/scene/examples/minimal.scene.json --from 0 --to 60 \
  --width 640 --height 360 --background '#204060' --out .data/export/minimal.mp4
```

参数全部可选：`--from` / `--to`（`[from, to)`）、`--format auto|mp4|webm`、`--quality high|medium|low`、
`--width` / `--height`、`--background`、`--workers`、`--out`。默认输出到 `.data/export/`。

实现使用 [Mediabunny](https://mediabunny.dev/)（MPL-2.0）的 WebCodecs 封装；不依赖 ffmpeg。
编码有损，因此导出的像素不会与 runtime PNG 逐位相同，但每帧来源与播放器完全一致（同一份 `CharaForgeRuntime`）。

实测（RTX 5080 Laptop，`--workers 2 --quality high`）：

- `scenes/deepseek-chan/scene.json`：3868 帧 → 64.467 s，37,967,113 字节，59.2 s。
- `scenes/deepseek-chan/scene.mv.json`（128 图层，含背景/小元素）：3868 帧 → 64.467 s，47,324,791 字节，63.7 s。

## AI / 脚本调用

### 浏览器 `window.__editor`

编辑器就绪后暴露稳定接口：

```js
await window.__editor.ready;

window.__editor.summary();
// { scenePath, title, fps, duration, composition, frame, layers: 47, markers: 43, audio: 1, ... }

window.__editor.scene();          // 当前场景对象的深拷贝
window.__editor.state();          // 选中、脏状态、catalog、导出进度等
window.__editor.seek(2400);
await window.__editor.play();
window.__editor.select({ kind: 'layer', id: 's24.main' });

// 编辑：命令格式与 CLI / HTTP 完全一致
const r = await window.__editor.apply({
  op: 'addKeyframe',
  target: { kind: 'layer', id: 's24.main', path: 'transform.x' },
  t: 2400, v: 700, ease: 'smooth',
});
// r = { ok, changes, effects, preview? }

await window.__editor.undo();
await window.__editor.redo();
await window.__editor.save('scenes/deepseek-chan/edited.scene.json');
await window.__editor.addMotion('models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json');
await window.__editor.addExpression('models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json');
await window.__editor.importImage('data:image/png;base64,...', { name: 'hat', attachToSelected: true });
await window.__editor.exportVideo({ from: 2368, to: 2442, format: 'mp4' });
```

### 本地 HTTP 接口

`pnpm dev` / `pnpm preview` 的中间件提供（仅本机 loopback）：

| 方法 | 路由 | 说明 |
|---|---|---|
| GET | `/__charaforge/editor/ping` | 健康检查 |
| GET | `/__charaforge/editor/scenes` | `scenes/` + 示例场景列表 |
| GET | `/__charaforge/editor/scene?path=...` | 读取并校验一个场景（只读） |
| GET | `/__charaforge/editor/assets?kind=image\|audio` | `assets/editor/` 素材列表 |
| POST | `/__charaforge/editor/upload?kind=image\|audio&name=...` | 原始字节上传 |
| POST | `/__charaforge/editor/save-scene` | `{ path, scene, backup? }` |
| POST | `/__charaforge/editor/command` | `{ scene, commands, out?, dryRun?, frame? }` |

示例：

```bash
curl -s http://localhost:5173/__charaforge/editor/ping

curl -s -X POST http://localhost:5173/__charaforge/editor/command \
  -H 'Content-Type: application/json' \
  -d '{
    "scene": "scenes/deepseek-chan/scene.mv.json",
    "commands": [
      { "op": "addKeyframe",
        "target": { "kind": "layer", "id": "s24.main", "path": "transform.x" },
        "t": 2400, "v": 700, "ease": "smooth" },
      { "op": "preview", "frame": 2400 }
    ],
    "out": "scenes/deepseek-chan/edited.scene.json"
  }'
```

响应包含 `diff`、`effects`、`preview` 和（`out` 时写出的）新场景。请求体上限 64 MiB，
上传上限 32 MiB；路径只能用仓库相对路径并受白名单/防穿越/symlink 检查。

### CLI 命令层

```bash
pnpm editor:command --scene scenes/deepseek-chan/scene.mv.json \
  --input .data/edit-commands.json --out .data/edited.scene.json

# stdin / dry-run / 预览
cat .data/edit-commands.json | pnpm --silent editor:command \
  --scene scenes/deepseek-chan/scene.mv.json --dry-run --frame 2400
```

输入可以是命令数组、`{ "commands": [...] }` 或单个命令对象；stdout 是 JSON。
和浏览器/HTTP 共用 `packages/scene/src/edit.ts`，因此行为、校验和 diff 完全一致。

## 命令格式

完整类型见 `packages/scene/src/edit.ts` 的 `EditCommand`。常用：

```jsonc
// 场景
{ "op": "setScene", "patch": { "title": "...", "background": "#101820", "duration": 3868 } }
{ "op": "setCamera", "camera": { "x": 640, "y": 360, "zoom": 1 } }

// 图层
{ "op": "updateLayer", "id": "s24.main",
  "patch": { "in": 2368, "out": 2442, "transform": { "x": 640, "y": 300 }, "opacity": 1 } }
{ "op": "moveLayer", "id": "bg", "index": 0 }
{ "op": "duplicateLayer", "id": "s24.main", "newId": "s24.copy" }
{ "op": "removeLayer", "id": "s24.copy" }
{ "op": "addImageLayer", "asset": { "id": "bg", "uri": "editor/bg.png", "width": 1280, "height": 720 },
  "layer": { "id": "bg", "kind": "image", "image": "bg", "in": 0, "out": 3868,
             "space": "screen", "transform": { "x": 640, "y": 360, "scale": 1 } }, "index": 0 }

// 关键帧（target.path 支持 transform.x/y/scale/scaleX/scaleY/rotation、opacity、
// clip.*、filters.<i>.<prop>；也支持 camera 与 param）
{ "op": "addKeyframe", "target": { "kind": "layer", "id": "bg", "path": "opacity" },
  "t": 0, "v": 0, "ease": "smooth" }
{ "op": "setKeyframes", "target": { "kind": "layer", "id": "bg", "path": "transform.scale" },
  "keys": [ { "t": 0, "v": 0.8 }, { "t": 60, "v": 1.1, "ease": "smooth" }, { "t": 120, "v": 1 } ] }
{ "op": "setEase", "target": { "kind": "layer", "id": "bg", "path": "transform.scale" },
  "t": 60, "ease": "easeOut" }
{ "op": "removeKeyframe", "target": { "kind": "layer", "id": "bg", "path": "opacity" }, "t": 0 }

// 动作 / 表情 / 音频
{ "op": "addMotion", "layer": "s24.main",
  "uri": "models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json",
  "start": 2368, "fadeIn": 0 }
{ "op": "addExpression", "layer": "s24.main",
  "uri": "models/deepseek-chan/DeepSeek/Exp3/脸红.exp3.json", "start": 2400 }
{ "op": "addAudio", "uri": "video/DeepSeek-chan.wav", "start": 0, "gain": 0.8 }
{ "op": "updateAudio", "index": 0, "patch": { "gain": 0.6, "start": 120 } }

// 一键动画
{ "op": "animate", "id": "bg", "preset": "float", "start": 0, "end": 180, "intensity": 24 }
```

缓动预设：`linear`、`smooth`、`easeIn`、`easeOut`、`easeInOut`、`step`/`hold`，
或直接给 `[x1,y1,x2,y2]`（CSS cubic-bezier）。一键动画 preset：
`fadeIn`、`fadeOut`、`float`、`pulse`、`spin`、`bounce`、`blink`、`slideIn`、`shake`、`zoomIn`。

`applyEditCommands` 原地修改场景并记录 JSON Pointer 形式的 `changes`；任一命令失败会整批回滚。
`effects` 告诉调用方如何刷新预览：

| effect | 含义 |
|---|---|
| `render` | 下一帧重画即可（变换/透明度/裁切/滤镜/相机/背景/标记） |
| `workers` | 参数关键帧值变了，Worker 里的场景副本要刷新 |
| `evaluator` | 动作/表情/参数轨结构变了，主线程求值器要重建（并刷新 Worker） |
| `structure` | 图层增删/排序/画面尺寸/图片资源变了，运行时整体同步 |
| `audio` / `clock` | 音频资产/片段/时基变了，重建播放时钟 |

## 架构

| 文件 | 职责 |
|---|---|
| `packages/scene/src/edit.ts` | 纯 TS 编辑命令、关键帧/缓动/一键动画、diff、撤销、预览摘要 |
| `apps/player/scripts/editor-command.ts` | AI/脚本 CLI |
| `apps/player/server/editorApi.ts` | 本地 HTTP 编辑/上传/保存接口 |
| `apps/player/src/editor/store.ts` | 编辑器状态、历史、素材、自动刷新、导出、`window.__editor` |
| `apps/player/src/editor/shell.tsx` | 布局、舞台、传输控制、导出对话框、快捷键 |
| `apps/player/src/editor/panels.tsx` | 图层/素材/动作/元素面板 |
| `apps/player/src/editor/inspector.tsx` | 场景/图层/音频/标记检查器、AnimField、参数轨 |
| `apps/player/src/editor/timeline.tsx` | 多轨时间线、关键帧/片段拖拽、标尺、缩放 |
| `apps/player/src/editor/export.ts` | 逐帧 + 离线混音 + Mediabunny 封装 |
| `apps/player/scripts/export-video.ts` | 导出 CLI（Playwright + export.html） |
| `packages/runtime/src/runtime.ts` | `syncScene` / `rebuildEvaluators` / `reloadWorkers` 热更新 |

## 测试

```bash
pnpm --filter @charaforge/scene test        # 编辑命令核心（关键帧、插值、音频、撤销…）
pnpm --filter @charaforge/runtime test      # 运行时确定性 + syncScene 热更新
pnpm test:editor-api                        # HTTP 接口安全/上传/保存/命令
pnpm test:editor                            # Playwright：编辑器加载、时间线、AI 命令、多轨、保存、导出
pnpm --filter @charaforge/player typecheck
pnpm --filter @charaforge/player build
```

`test:editor` 会验证：47 图层/43 标记/1 音频、编辑器与 harness 同一帧像素 SHA-256 相同、
关键帧添加/撤销、生成元素+一键动画、动作/表情添加与撤销、追加音乐轨、保存到临时场景、导出 4 帧 WebM（含音频）。

## 已知限制

- 视频有损编码，像素不与 runtime PNG 逐位相同；导出帧来源与播放器同一份运行时。
- 透明场景导出视频会合成到指定背景色（MP4/WebM 默认不含 alpha）；需要 alpha 请用 WebM/VP9 + `alpha: 'keep'` 的扩展。
- 参数轨数量很大（全片 15,800 条），编辑器默认只展开选中的参数轨，其他用搜索 + 固定到时间线。
- `attach` 只跟随图片/模型图层的整体变换，不跟骨骼、表情和物理。
- 动作库列表依赖 dev/preview 中间件；纯静态托管 `dist/` 时不可用（模型读取本来也需要该中间件）。
- 没有骨骼编辑、蒙皮权重编辑、口型自动同步；这些属于后续阶段。
- Mediabunny 以 MPL-2.0 依赖使用，未修改其源码。
