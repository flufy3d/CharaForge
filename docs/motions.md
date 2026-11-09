# 动作 / 表情素材查看器

本期目标：只读地查看当前模型（DeepSeek-chan / 大肥鱼）的全部 motion3 动作与 exp3 表情，点击后在角色上预览，
支持播放/暂停/seek/循环、缩略图/图集，以及表情预览叠加 IDLE。**不做**动作编辑、骨骼编辑、参数轨编辑，
也不修改 `scenes/deepseek-chan/scene.json` 或任何模型文件。

入口：

```bash
pnpm dev
# Chrome / Edge 打开 http://localhost:5173/motions.html
```

`pnpm preview`（`vite preview`）使用同一个中间件，也可以直接打开 `http://localhost:4173/motions.html`。
页面已加入 `apps/player/vite.config.ts` 的 `build.rollupOptions.input`，`pnpm --filter @charaforge/player build`
会产出 `dist/motions.html`。

## 数据来源

当前 `大肥鱼.model3.json` 的 `FileReferences` 只有 `Moc / Textures / Physics / DisplayInfo`，没有 `Motions`，
也没有 `Expressions`；全片 `scene.json` 里 motion 片段数量为 0（动作已经 baked 成参数轨）。因此不能只依赖
`ModelResource.listDeclared()`。

列表来源优先级：

1. **model3.json 声明优先**：若 `FileReferences.Motions` 非空，动作列表只使用其中的文件；表情同理，
   `FileReferences.Expressions` 非空时只使用声明项。
2. **目录回退**：没有声明时，调用只读接口扫描：
   - 动作：`assets/models/deepseek-chan/DeepSeek/Animation/`
   - 表情：`assets/models/deepseek-chan/DeepSeek/Exp3/`

> 选择“接口扫描”（方案 A）而不是生成 `motions.index.json`：`assets/` 被 gitignore，生成文件容易过期，
> 还要保证 `pnpm dev` / `pnpm preview` 前一定跑过生成脚本；接口方式在两种启动方式下都直接可用。
>
> 当前模型走目录回退：8 个 motion3、23 个 exp3（2026-11 核对）。

列表与元数据都只读取文件内容做 JSON 解析，不执行任何内容。中文文件名请求仍走
`packages/runtime/src/asciiUrl.ts` 的 `_f.<base64url>` 路径段，与播放器、runtime 一致。

## 只读列表接口

```http
GET /__charaforge/list-assets?dir=assets/models/deepseek-chan/DeepSeek/Animation
```

```json
{
  "dir": "assets/models/deepseek-chan/DeepSeek/Animation",
  "files": [
    { "name": "IDLE.motion3.json", "size": 35673, "mtime": "2026-11-09T03:36:00.000Z" }
  ]
}
```

- 实现：`apps/player/server/assetList.ts`，由 `vite.config.ts` 的 `repoFiles()` 在进入静态白名单前挂载；
  `pnpm dev` 和 `pnpm preview` 共用。
- 只接受 `GET`（其他方法 405），只列文件名/大小/修改时间，**不读取文件内容**。
- 只允许 `assets/models/` 下、末级目录为 `Animation` 或 `Exp3` 的目录。
- 拒绝：绝对路径、盘符、反斜杠、NUL、空段、`.`、`..`（URL 解码后的也算）、非允许末级目录。
- `realpath` 后再次确认目标仍在 `assets/models/` 的真实路径下；目录内的 symlink 文件直接跳过。
- 错误返回 JSON：400（非法 dir）、403（越界/非允许目录）、404（目录或 assets 不存在）、405（非 GET）。
- 文件按名称排序；`mtime` 为 ISO 8601 字符串。
- 这是只读目录 API，不是通用文件服务器；没有写、删、执行能力。

## 元数据

动作（motion3）解析 `Meta`：

| 字段 | 用途 |
|---|---|
| `Duration` | 秒；列表显示，预览时长 = `max(120, ceil(Duration × 60))` 帧 |
| `Fps` | motion3 自身采样帧率，列表元数据 |
| `Loop` | 初始循环开关 |
| `CurveCount` | 曲线数 |
| `TotalSegmentCount` / `TotalPointCount` | 段数/点数 |
| `UserDataCount` | 用户数据数 |

表情（exp3）解析 `Parameters`：每项 `Id`、`Value`、`Blend`（Add / Multiply / Overwrite）。表情没有时长/循环，
预览固定 120 帧（2 秒）。

素材列表项显示缩略图、显示名和动作/表情摘要；右侧元数据面板显示文件名、大小、修改时间、来源（model3 声明 / 目录扫描）、
资源 uri 等。解析失败的单个文件会跳过，并在页面上以 notice 列出 warning，不会让整个查看器崩溃。

## 临时预览 scene

页面复用 `PlayerEngine` / `CharaForgeRuntime` / `Compositor`，不另写渲染器。每次点击素材时构造一个临时 scene，
通过 `PlayerEngine.load()` 加载；不写盘，也不修改 `scenes/deepseek-chan/scene.json`。

动作示例（IDLE）：

```json
{
  "format": "charaforge.scene",
  "version": 1,
  "meta": { "title": "动作预览：IDLE" },
  "timebase": { "fps": 60 },
  "duration": 359,
  "composition": { "width": 720, "height": 720, "background": "#202830" },
  "assets": {
    "models": [{ "id": "m", "uri": "models/deepseek-chan/DeepSeek/大肥鱼.model3.json" }],
    "motions": [{
      "id": "preview",
      "model": "m",
      "uri": "models/deepseek-chan/DeepSeek/Animation/IDLE.motion3.json"
    }]
  },
  "layers": [{
    "id": "model",
    "kind": "model",
    "model": "m",
    "in": 0,
    "out": 359,
    "transform": { "x": 360, "y": 360, "scale": 0.16 },
    "physics": { "warmup": 45 },
    "motions": [{ "motion": "preview", "start": 0, "loop": true, "fadeIn": 0 }]
  }]
}
```

- 固定 `timebase.fps = 60`、`composition = 720×720`、背景 `#202830`；动作时长
  `max(120, ceil(Meta.Duration × 60))`，表情固定 120 帧。
- `fadeIn: 0`：点击后立即看到动作/表情，不必等文件自带的 1 秒淡入。
- `loop` 初始取 `Meta.Loop`，UI 可切换。循环开启时，页面在播放自然到达末帧后自动从头重播；关闭时播放到
  自然结束/场景末帧后停止。表情没有 loop，开关禁用。
- `physics.warmup = 45`，减少第一帧的物理突变。
- `transform` 是占位值。`loadScene` 后从 `runtime.layerInstance('model')?.canvas` 读取模型画布尺寸，按
  `scale = min(720 / canvas.width, 720 / canvas.height) × 0.96`、位置 `(360, 360)` 居中 fit；不写死模型缩放。
- 预览 scene 不写 `assets.audio` / `audio`，页面也不调用 `loadVideo()`；`PlaybackClock` 没有音频时退化为
  `performance.now()`，所以不需要用户手势即可播放。
- 表情预览缺省叠加 IDLE：`assets.motions` 增加 IDLE，`layer.motions` 与 `layer.expressions` 同时存在
  （按规范先 motion 后 expression）。取消勾选「叠加 IDLE」或使用 `?idle=0` 时回到“默认姿态 + 单个 exp3”。
  切换类表情（如竖食指开/关）在两种模式下的观感会不同，以实际预览为准。

## 缩略图与图集

- 页面启动时（缓存未命中）用一次 **contact sheet** 渲染生成全部素材缩略图：
  - 一个临时 scene，每个素材一个模型图层，默认 6 列 × 160px 方格；
  - 物理关闭，只渲染帧 0；motion 通过 `offset` 取动作中段（约 `min(1.5s, Duration × 0.35)`），
    表情从帧 0 立即生效；
  - 每个图层带 tile 大小的 `clip`，不会互相覆盖；
  - `runtime.readPixels()` 读出整张图后按 tile 裁成 160×160 PNG data URL。
- 缩略图缓存在 `localStorage`（key `charaforge.motion-thumbs.v1`），缓存键包含 model uri 与每个素材的
  `kind/file/size/mtime`；素材变化后自动失效。缓存不可用或超配额时只在本次会话内使用。
- 左侧列表直接显示缩略图；顶部「图集」打开全部素材的图集，点击任一缩略图切换到该素材。
- 生成失败不会阻塞预览：列表回退为「动 / 表」占位符，页面显示 warning；刷新可重试。
- 缩略图只是列表/图集展示，不写盘、不进 git，也不改变 `assets/` 内容。

## 页面与调试接口

URL 参数：

| 参数 | 含义 |
|---|---|
| `kind=motion\|expression` | 初始素材类型，缺省 motion |
| `name=<文件名>` | 初始素材文件名（含扩展名），缺省 IDLE（`kind=expression` 时缺省第一个表情） |
| `frame=<n>` | 初始帧，缺省 0 |
| `loop=0` | 动作循环初始关闭 |
| `idle=0` | 表情预览不叠加 IDLE（缺省叠加） |

快捷键：`空格` 播放/暂停，`←`/`→` ±1 帧（`Shift` ±10），`Home`/`End`，`L` 切换循环。

页面暴露只读测试钩子 `window.__motionViewer`（稳定 JSON，供后续代理/Playwright 使用）：

```ts
window.__motionViewer.readyState            // 'booting' | 'ready' | 'error'
await window.__motionViewer.ready           // 启动完成（成功或失败）
window.__motionViewer.catalog()             // { modelUri, motions, expressions, sources, warnings }
window.__motionViewer.preview()             // { kind, file, name, uri, duration, fps, loop, withIdle, meta/info, ... }
window.__motionViewer.selected()            // { kind, file } | null
await window.__motionViewer.select(kind, file)
await window.__motionViewer.seekAndRender(frame)
await window.__motionViewer.hashFrame()     // 当前合成像素 SHA-256（0.5 缩放）
window.__motionViewer.thumbnails()          // { "motion:IDLE.motion3.json": "data:image/png;base64,...", ... } | null
await window.__motionViewer.setLoop(true|false)
await window.__motionViewer.setWithIdle(true|false)
window.__motionViewer.status()              // EngineStatus
```

## 测试

```bash
pnpm --filter @charaforge/player test:viewer
# 等价：pnpm test:viewer
```

包含：

- `apps/player/scripts/asset-list.test.ts`：列表接口返回形状、排序、中文名、路径穿越/反斜杠/绝对路径/盘符/
  非允许目录拒绝、目录内 symlink 跳过、目录本身 symlink 越界拒绝；真实 assets 存在时校验 8 / 23。
- `apps/player/scripts/motion-preview.test.ts`：motion3 Meta、exp3 Parameters、资源 uri、临时动作/表情 scene
  （含叠加 IDLE）、缩略图 contact sheet scene 通过 `parseScene`、fit 公式、真实 8 / 23 文件解析。
- `apps/player/scripts/motion-viewer.test.ts`：Playwright smoke，打开页面、确认 8 / 23、确认 31 张缩略图与
  图集、点击 IDLE、seek 到 0/90 后像素摘要变化、点击“脸红”表情并验证叠加 IDLE 后画面随帧变化、
  页面无未捕获错误。

回归命令：

```bash
pnpm -r run typecheck
pnpm test:runtime
pnpm --filter @charaforge/player build
```

## 已知限制

- 列表接口只存在于 `pnpm dev` / `pnpm preview` 的仓库中间件；用普通静态服务器托管 `dist/` 时不可用
  （现有播放器的模型/场景读取本来也需要这个中间件）。
- `assets/` 不进 git；新克隆仓库未运行 `pnpm setup:assets` 时，页面显示安装提示，没有素材可列。
- 首次打开（缩略图缓存未命中）会多一次 contact sheet 渲染，素材多时需要数秒；之后走 localStorage 缓存。
- 缩略图不是视频/动图，只截取动作中段的一帧；具体动作仍以右侧预览为准。
- 切换素材、循环开关或「叠加 IDLE」开关会重新加载临时 scene（模型/motion/exp3 资源按 URL 缓存，不会重复下载），
  可能有短暂重载感。
- 同一时间只预览一个动作或一个表情；不做多动作混合、参数轨编辑、骨骼编辑。
- 模型按原作者条款禁止再分发，查看器只供本地使用，不要公开部署或把 `assets/` 提交进 git。
