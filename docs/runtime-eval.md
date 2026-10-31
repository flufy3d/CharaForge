# 运行时评估与实现说明（子代理 A）

## 1. easy-live2d 评估

评估对象：<https://github.com/Panzer-Jack/easy-live2d>，提交 `458bf76`（2026-09-24），MIT 许可。
读的源码：`packages/core/src/`（`Live2DSprite.ts`、`model/Live2DModel.ts`、`rendering/ModelRenderer.ts`、
`core/TimeManager.ts`、`core/Live2DContext.ts`、`rendering/WebGLBackend.ts`、`model/EffectController.ts`、`loader/*`）。

| # | 要求 | 结论 | 依据 |
|---|---|---|---|
| 1 | 手动按固定 1/60 步长推进，物理确定 | **不满足** | `ModelRenderer.render()` 每次绘制都调用 `timeManager.update()`，dt 取 `performance.now()` 差值并夹到 0.1s（`TimeManager.ts`）；推进挂在 Pixi 的 `onRender` 上（`Live2DSprite.ts` 构造函数），没有公开的 `step(dt)`。另外 `Live2DModel.update()` 在 motion 结束时调用 `startRandomMotion`（随机挑 idle），`EffectController` 默认启用 `CubismEyeBlink`（内部随机间隔）和 `CubismBreath`、拖拽跟随——都是规范禁止的隐式行为。 |
| 2 | 在 motion/expression → 外部参数 → 物理 之间插入覆盖；以及物理之后的覆盖 | **不满足**（只有后者） | `Live2DModel.update()` 的顺序是 loadParameters → motion → saveParameters → expression → effects（眨眼/呼吸/拖拽）→ pose → physics → `_applyParameterOverrides()` → `model.update()`。只有「物理之后」一个覆盖点；d 步（物理之前）无钩子，中间还夹着隐式 effects。 |
| 3 | 离屏渲染并读回像素 | **不满足** | `Live2DContext.initialize()` 在初始化时用 `gl.getParameter(FRAMEBUFFER_BINDING)` 记下帧缓冲（即画布默认帧缓冲），之后 `ModelRenderer` 始终画到它上面，不跟随 Pixi 当前的 render target，`renderer.render({ target: RenderTexture })` 拿不到模型像素；没有读回接口。 |
| 4 | 每实例完整状态（物理粒子、motion/expression 管理器）保存/恢复 | **不满足** | 无任何快照接口；状态分散在 Framework 的 `CubismPhysics`、`CubismMotionManager`、`CubismExpressionMotionManager` 与 `TimeManager` 中。 |
| 5 | 多实例，各自渲染到独立 render target | **部分** | 可以创建多个 `Live2DSprite`，但每个实例独立加载 moc/贴图（`ModelLoader` 无共享缓存，本模型 16.5MB moc + 5×4096² 贴图会重复占用），且都画在初始化时的同一帧缓冲上。 |

**结论**：三条硬要求都不满足，不直接依赖。基于官方 Cubism Framework R5 自写薄封装 `@charaforge/runtime`。
从 easy-live2d 借鉴的只有一个思路（MIT，已在 `packages/runtime/src/compositor.ts` 文件头注明）：
Cubism 绘制时绑定独立的 VAO，并在绘制后让 Pixi 重置 GL 状态缓存。没有复制其代码。

## 2. 架构

```
packages/runtime/
  scripts/build-framework.mjs  vendor/cubism/framework/src → .cache/cubism-framework/（ESM + d.ts）
                               另从缓存的 SDK zip 只读取出 Framework/Shaders/WebGL/*，生成同步可用的 shaders.js
  src/cubism.ts       Framework 启动（Core 堆 512MB）、同步注入 Cubism 着色器
  src/resources.ts    ModelResource：moc / 贴图（预乘、mipmap）/ physics3 / cdi3 / motion / exp3，多实例共享
  src/instance.ts     ModelInstance：CubismModel + CubismRenderer_WebGL + CubismPhysics；参数/部件读写、绘制、快照
  src/physicsState.ts CubismPhysics 内部状态的逐位快照
  src/motion.ts       motion / expression 片段的无状态求值（每次新建 QueueEntry，调用 Framework 的 doUpdateParameters）
  src/evaluator.ts    LayerEvaluator：规范 a～g、预热、检查点、seek 重放、一次性覆盖
  src/compositor.ts   Pixi v8 合成：图层 RT → blur/motionBlur/color → clip×opacity → blendMode；读回
  src/runtime.ts      CharaForgeRuntime：场景加载、renderFrame / renderFrameAsync、frameHold、可见性裁剪、状态、单实例接口
  src/worker*.ts      Worker：主场图层求值（物理链）+ Core update；workerPool.ts 调度
  src/glProxy.ts      给 Cubism 渲染器的 GL 包装：静态 UV/索引缓冲缓存、翻转时的绕序修正
  src/sceneAssets.ts  场景资源解析（主线程与 Worker 共用）
apps/player/
  vite.config.ts      dev 中间件按白名单暴露 assets/、vendor/cubism/core/、scenes/、packages/scene/examples/（支持 Range）
  index.html + src/main.tsx, engine.ts, clock.ts, videoSync.ts, demo.ts   播放器
  harness.html + src/harness.ts, harness-api.ts                            Playwright 用接口
  scripts/harness-node.ts   启动 Vite + GPU 无头 Chromium（render / gallery / 测试 / 拟合器共用）
  scripts/render.ts, gallery.ts, verify-video.ts, png.ts
```

### 关键实现决定

- **motion/expression 无状态**：Cubism 的 MotionManager 是按墙钟推进的队列，无法随机访问。改为每帧用片段定义 + 帧号
  构造一个 `CubismMotionQueueEntry`（开始/淡入/结束时刻），再调 Framework 自己的 `doUpdateParameters`，
  曲线插值、V2 循环修正、Add/Multiply/Overwrite 都是官方实现。于是实例的跨帧状态只有物理，
  `saveState()` = 物理快照 + 求值游标，「motion/expression 管理器状态」自然被覆盖（不存在）。
  时间约定见 `motion.ts` 文件头：淡出在结束时刻前 fadeOut 秒内完成（与 Cubism 设 endTime 后一致）；
  非循环 motion 无 `end` 时在自然结束处淡出；循环不逐圈重新淡入。
- **seek 与顺序播放逐位相同**：`LayerEvaluator.prepare(f)` 从「≤ f−1 的最近规范状态」（当前游标 / 上一步之前的快照 /
  每 30 帧一个检查点 / 初始状态）恢复并逐帧重放，两条路径执行同一浮点运算序列。重放的中间帧只做 a～f，不做 Core update 与绘制。
- **物理初始状态**：加载后的初始状态，不调用 `stabilization()`（规范第 4 条只规定从 in−warmup 推进）。
- **覆盖钩子**：`renderFrame(f, { overrides: { [layerId]: { pre, post, parts } } })`，pre 在 d 之后 e 之前，post 在 f 之后；
  只作用于该帧（带覆盖的 step 前先保存 S(f−1)，不污染后续帧）。
- **frameHold（规范 4b）**：显示帧 f 在段内映射到 g；图层状态、transform、opacity、filters、clip 以及**相机**都在 g 求值；
  物理仍逐帧推进（S(g) 与无保持时同一帧完全相同，测试已验证）。图层是否存在按 f 判断。
- **坐标**：模型画布像素 → 合成像素按规范公式组成一个仿射矩阵，再乘上 Cubism 单位 → 画布像素
  （`px = originX + u·PPU`，`py = originY − v·PPU`）。Pixi RenderTexture 的 GL 第 0 行是图像顶部，
  所以读回不用翻转。绕序翻转时（det<0）对带背面剔除的模型用 GL 代理交换 frontFace（本模型无剔除 drawable）。
- **Cubism 着色器**：R5 渲染器默认在首次绘制时异步 fetch 着色器文件（首帧画不出东西且时机不确定），改为同步注入源码。
- **Core 堆**：Core 的 WASM 堆不会增长；默认 16MB 装不下本模型的 16.5MB moc，设为 512MB。
- **贴图**：`createImageBitmap(premultiplyAlpha:'premultiply')` + mipmap（LINEAR_MIPMAP_LINEAR），与 Cubism 预乘管线一致。
- **抗锯齿**：`composition.antialias`（缺省 true）时 Cubism 先画进 4×MSAA 渲染缓冲，再 blit resolve 到图层 RenderTexture；false 时直接画。
- **physics.stabilize**：构造求值器时先按帧 in 的输入做 a～d，再调用 Cubism `stabilization()`，结果作为 S(in − warmup − 1)。
- **不可见图层**（opacity ≤ 0、clip 面积 0、模型画布矩形按模糊半径外扩后完全出画/出 clip）：只推进物理（a～f），不做 Core update 与绘制。
- **frameHold 复用**：保持段内 g 已画过时，直接用缓存的（滤镜后）图层缓冲做输出 pass，不求值、不绘制。

## 3. 滤镜与合成语义（实现细节，规范未写的部分）

- blur：可分离高斯，半径 ceil(3σ)，`texelFetch` 整数采样，缓冲外视为透明，权重按全部采样点归一化（边缘会变透明）。
- motionBlur：沿 (cos angle, sin angle)（y 向下，顺时针为正）在 [−L/2, L/2] 上取 max(2, ceil(L)+1) 个双线性样本的平均，缓冲外透明。
- color：按 scene.ts 描述，作用于反预乘颜色，结果夹到 [0,1] 再预乘。
- clip：圆角矩形（屏幕/合成像素），覆盖率 = clamp(0.5 − d/max(feather,1), 0, 1)，d 为到边界的有符号距离，
  即羽化带以边界为中心、宽 feather；feather = 0 为 1 像素抗锯齿。
- opacity 与 clip 在同一个输出 pass 里相乘，再用 Pixi 的预乘 blendMode（normal/add/multiply/screen）混入合成缓冲。
- 读回缩放：输出像素在源区域上取 k×k（k = ceil(1/scale)，≤8）个双线性样本平均。

## 4. GPU 与确定性

### 无头 Chromium 的 GPU 启动参数（Playwright 1.64.1 / Chromium 156，RTX 2060，驱动 580）

- **必须去掉 `DISPLAY`**：本机 `DISPLAY=:1` 的 X 服务器对新连接无响应，Chromium 的 GPU 进程连 X 时永久阻塞
  （strace 可见 `connect(@"/tmp/.X11-unix/X1")` 不返回），表现为 `getContext('webgl2')` 卡死、有头模式启动超时。无头模式不需要 X。
- 缺省（`CHARAFORGE_GPU=vulkan`）：
  `--use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan --use-vulkan=native --ignore-gpu-blocklist`
  + `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`
  → `ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA NVIDIA GeForce RTX 2060 (0x00001F11)), NVIDIA)`
- 备选（`CHARAFORGE_GPU=egl`）：`--use-angle=gl-egl --ignore-gpu-blocklist`
  + `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`（否则 EGL 选 Mesa/Intel 核显）
  → `ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 2060/PCIe/SSE2, OpenGL ES 3.2)`
- 不加参数时是 SwiftShader（且在本机同样因 X 问题挂起）。
- `harness-node.ts` 启动后检查 renderer 字符串，不含 `NVIDIA`（`CHARAFORGE_EXPECT_GPU` 可改）就报错，不会悄悄落到软件渲染。

### 确定性发现

- **ANGLE/GL-EGL 后端的 MSAA 不确定**：同一输入连续两次绘制到 4×MSAA 缓冲再 resolve，约 50 个边缘像素有 ±1 差异。
  关掉 MSAA 后完全确定；**ANGLE/Vulkan 后端开 MSAA 也逐位确定**。因此缺省用 Vulkan + MSAA 4；
  用 EGL 时请设 `CHARAFORGE_MSAA=0`（harness URL 参数 `?msaa=0`）。
- 其余部分（物理、motion、合成 pass、读回）在两个后端上都逐位确定。
- 跨机器 / 跨驱动不保证逐位一致（GPU 光栅化与驱动有关）；确定性承诺是「同一机器、同一后端」。

### 测试结果

`pnpm test:runtime`（= `pnpm --filter @charaforge/runtime test`），场景含 3 个图层、motion、expression、
pre/postPhysics 与 add 模式轨道、全部三种滤镜、clip、screen 混合、相机旋转缩放、flipX、预热：

```
# [gpu] ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA NVIDIA GeForce RTX 2060 (0x00001F11)), NVIDIA)  (msaa 4)
ok 1 - 使用 GPU 渲染（不是 SwiftShader）
ok 2 - 顺序播放两遍逐位相同
ok 3 - 乱序 seek 与顺序播放逐位相同
ok 4 - saveState / restoreState 与一次性覆盖
ok 5 - 单实例：显式参数 + 物理状态保存/恢复
ok 6 - frameHold：段内 f 与 g 像素逐位相同，段外与无保持一致
ok 7 - Worker 路径与同步路径逐位相同（顺序 + seek + 覆盖）
ok 8 - composition.antialias：false 时不做 MSAA，等价于 MSAA=0 的渲染
ok 9 - physics.stabilize：预热前置于平衡态，结果不同于未稳定、且确定
ok 10 - 不可见图层（opacity=0 / 出画 / clip 面积 0）跳过 Core update，但物理链连续
```

（`CHARAFORGE_GPU=egl CHARAFORGE_MSAA=0` 下前 6 项同样通过。）
所有性能优化完成后，`pnpm render` 的输出与优化前的输出 `diff -r` 无差异。
`pnpm render --scene packages/scene/examples/minimal.scene.json --frames 0-60 --out .data/render/a` 与 `.../b` 两次输出 `diff -r` 无差异
（PNG 由 `png.ts` 用固定参数编码，文件字节只取决于像素）。

## 5. 参考视频逐帧对齐

方法：`pnpm verify:video`（`apps/player/scripts/verify-video.ts`）。对每个检查帧 f，在无头 Chromium 里用与播放器相同的
`seekVideoFrame()`（seek 到 (f+0.5)/fps，等 `seeked` + `requestVideoFrameCallback`）取出画面，与 ffmpeg 解码的
f−1、f、f+1 计算 MAE。检查帧：片头片尾、若干中间帧，以及 ffmpeg 场景检测（`select='gt(scene,0.35)'`）得到的 13 处快切的前一帧/切点/后一帧。

结果：**50/50 帧精确**。与正确帧的 MAE ≈ 0.3～1.3（解码器/YUV→RGB 差异），快切处与相邻帧的 MAE 为 40～97，区分明确。
例：`1707: MAE(f−1)=94.765, MAE(f)=1.024, MAE(f+1)=1.080`。
原 mp4（H.264 High、2 个 B 帧、不规则关键帧间隔）在 Chromium 中已能逐帧精确定位，所以**没有**另外生成短 GOP 参考视频，
`scripts/setup-assets.mjs` 未改动。代价是暂停时远距离跳转需要从前一个关键帧解码（最长几秒），逐帧步进不受影响。
注意：Playwright 的 Chromium 支持 H.264；Firefox 等其他浏览器未验证。

## 6. 性能（i7-8750H 6 核 12 线程笔记本 + RTX 2060，1280×720，无头 Chromium + Vulkan）

### 剖析结论

- 每个实例每个渲染帧恰好 1 次 `csmUpdateModel`（Core update，≈ 10.3 ms，单线程无竞争时；多核满载时 ≈ 14 ms）；
  seek 重放、预热、不可见图层、frameHold 复用帧都不做 Core update（测试 10 断言了计数）。
- 主线程绘制（Cubism 渲染器）≈ 3.0 ms/实例，其中 `bufferData` 占 24%（每个 drawable 重传 UV 与索引）。
- 参数求值 + 物理 ≈ 1.5 ms/实例/帧；导入 Core 动态数据（约 470 KB）≈ 0.13 ms/实例。

### 按收益排序的改动与数据（`demo:3`，vsync 对齐的上屏间隔）

| 步骤 | fps | 说明 |
|---|---|---|
| 0. 同步路径（全部在主线程） | 14–18 | 3 × (10.3 Core + 1.5 求值 + 3 绘制) ≈ 45 ms |
| 1. Worker 做「求值 + Core update」，每图层一个 Worker | 32 | Worker 端 ≈ 16 ms/帧，贴着 60fps 上限，一旦晚一帧就连锁掉帧 |
| 2. 拉模式调度：主线程按帧号优先级派发，只在 Worker 空闲时发，过期的预取在发出前丢弃；最近 64 帧物理快照环形缓存 | 35 | 消除了「过期任务堵在 Worker 队列」和「回退请求从检查点重放 30 帧」 |
| 3. 求值留在图层的主场 Worker（物理链不重复），Core update 经 Worker 间 MessagePort 交给最空闲的 Worker；主线程只导入与绘制 | 37 | 主线程每帧 ≈ 8 ms，但上屏间隔仍是 16.7/33 ms 交替 |
| 4. **音频时钟平滑**：`AudioContext.currentTime` 按 10.7/21.3 ms 的音频块跳变（实测），导致帧号不均匀前进、部分 rAF 没有新帧；改用 `getOutputTimestamp()` 按 performance.now 外推（已含输出延迟，单调） | 59 | 这一项对桌面浏览器同样适用 |
| 5. 动态数据缓冲回收（主线程 → Worker 转移复用）、状态栏 4 Hz 刷新 | 59 | 降低 GC 与 React 开销 |
| 6. Cubism 静态 UV/索引缓冲缓存（GL 包装对象；先试 Proxy，trap 开销反而更大，改为预绑定方法的普通对象） | 59.5 | 像素与优化前逐位相同（render 输出 diff 无差异） |

### 最终结果（`pnpm bench -- demo:N 秒数 workers`）

| 场景 | workers | fps | 上屏间隔 avg / p50 / p95 / p99（ms） | 丢帧率 |
|---|---|---|---|---|
| demo:1 | 5 | 59.7 | 16.78 / 16.7 / 16.7 / 16.8 | 0.2% |
| demo:2 | 5 | 59.5 | 16.83 / 16.7 / 16.8 / 16.8 | 0.2% |
| **demo:3** | **5** | **59.3** | **16.88 / 16.7 / 16.8 / 33.3** | **1.0%** |
| **demo:3** | **6** | **59.5** | **16.82 / 16.7 / 16.8 / 33.3** | **0.6%** |
| demo:4 | 6 | 29.9 | 33.5 / 33.3 / 50.1 / 66.7 | 50% |
| demo:3（同步路径） | 0 | 18.1 | 55.2 / 50.0 / 66.7 / 83.3 | 70% |

测量方法：`pnpm bench -- demo:3 20 5`（`apps/player/scripts/bench.ts`）。无头 Chromium（与测试同一套 GPU 参数）打开播放器
`/?scene=demo:3&workers=5`，开始播放，预热 3 秒后统计 N 秒：每次上屏记录发起它的 rAF 时间戳（vsync 对齐；结果晚于下一个 vsync
返回时按完成时刻归入对应周期），给出上屏次数/秒、间隔分位数、显示帧号的跳变（丢帧）。播放器状态栏也显示实测 fps。
测量时机器上同时运行着用户的桌面 Chromium（持续约 70% CPU），丢帧主要来自这类背景负载。
播放器缺省 Worker 数 = clamp(逻辑核数/2 − 1, 2, 6)，本机为 5；URL 参数 `workers=N` 可改，`workers=0` 为同步路径。

4 实例：4 × 14 ms 的 Core update + 主线程 ≈ 12 ms，需要约 4.5 个核满载，加上 GPU 进程已超过本机 6 个物理核，只有 30 fps。
要上 60 需要降低 Core update 本身的开销（它是闭源 WASM，无法优化），或在 4 实例镜头里对远景实例降低更新频率（会改变画面，需规范支持）。

## 7. Harness / 运行时接口摘要

完整类型与说明：`apps/player/src/harness-api.ts`。Node 端用 `apps/player/scripts/harness-node.ts` 的 `openHarness()`。

- harness URL 参数：`?workers=N`（Worker 路径，逐位相同但不支持 saveState）、`?msaa=N`；缺省同步路径。
- 场景：`loadScene(path|object)`、`renderFrame(f, { overrides, read })`、`readPixels({ region, scale, format })`、
  `saveState()/restoreState(h)/dropState(h)`、`layerParams(layerId)`
- 单实例：`createInstance(modelUri)`、`instanceInfo(id)`（画布信息、509 个参数含中文名与范围、部件）、
  `stepInstance(id, { pre, post, parts, physics, dt })`、`resetInstancePhysics`、`instanceParams`、
  `renderInstance(id, { transform, width, height, region, scale, format })`、`saveInstanceState/restoreInstanceState`
- 读回格式：`rgba`（base64 RGBA8，预乘 alpha，行从上到下）或 `sha256`（只返回摘要）
- 运行时 TS 接口（页面内直接用）：`CharaForgeRuntime.create/loadScene/renderFrame/present/readPixels/saveState/restoreState/
  createInstance/stepInstance/renderInstance`，`ModelInstance.setParam(id, v, mode, weight)/getParams/setPartOpacity/stepPhysics/saveState/restoreState`。

## 8. 已知问题

1. 3 实例 60fps 依赖多核：Worker 路径需要约 3～4 个空闲核；4 实例在本机只有 30 fps（见第 6 节）。
2. Worker 路径不支持 saveState/restoreState（物理链在 Worker 里）；拟合请用同步路径（harness 缺省）。
3. MSAA 的逐位确定性依赖后端（GL-EGL 不确定，Vulkan 确定），跨驱动更无保证。
4. 播放中的参考视频按自己的时钟走（漂移超过 2 帧才重对齐），**精确对齐只在暂停/逐帧时保证**；叠加/差异图请暂停后看。
5. render CLI 每帧通过 CDP 传 base64 像素再在 Node 端编码 PNG，约 0.25 s/帧（720p）；拟合时请用 region/scale 或 `sha256`。
6. 不支持 pose3（本模型没有）；model3.json 的 EyeBlink/LipSync 组不生效（按规范无隐式行为）。
7. 带背面剔除的模型在翻转时的绕序修正已实现但未用真实模型验证（本模型没有剔除 drawable）。
8. 工作区里新出现的 `tools/fit`（子代理 C）缺 tsconfig / typescript 依赖，`pnpm -r typecheck` 会因它失败；
   scene / runtime / player 三个包均通过（`pnpm -r --no-bail typecheck`）。

## 9. 对 scene schema 的建议

第一轮的 7 条已被采纳（规范 1、3、3b、4b、5，`composition.antialias`、`physics.stabilize`），均已实现并有测试。新的建议：

1. `physics.stabilize` 与 `warmup` 同时给出时的顺序已按规范实现（先 stabilize，再预热）；建议在规范里注明 stabilize 使用的输入不含一次性覆盖。
2. 多实例镜头若需要更高帧率，可考虑给图层加「求值帧率」类字段（如 `evalEvery`），但会改变画面，需总负责人决定。
