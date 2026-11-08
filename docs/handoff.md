# CharaForge 第一期交接文档

写于 2026-11-06，供在另一台机器上继续开发。先读本文，再读 `docs/pipeline.md`（目录、契约和验收口径）、`docs/gpu-acceleration.md`（提速方案）、`tools/fit/README.md`（拟合引擎）。

## 1. 第一期目标与已拍板的决策

**目标**：用 DeepSeek-chan 模型和我们自己的运行时、场景格式，复刻参考 PV（`assets/video/DeepSeek-chan.mp4`，1280×720、60fps、3868 帧、64.47 秒）里的角色动画。最终交付一个网页，播放同样的效果，同时同步播放原音频。

**范围**
- 要做：每个镜头的缩放、裁切、平移，同屏多个角色实例，实例透明度，景深模糊，所有模型参数（表情、换装、配饰开关）。
- 不做：背景、设定稿底图、手指、对话气泡、✓ 和 ?、鼠标光标、白色剪影转场、粒子和光效。
- 第二期再做：核心功能 6～9（AI 类）。第 10 项（行为语义）只在 schema 里预留了字段。动捕数据导入也放到以后（架构上已经支持）。

**验收**（详见 `docs/pipeline.md` 最后一节）
- 指标是 SSIM@360：渲染图和参考帧都降采样到 640×360 后计算，只算角色遮罩区域，扣掉排除区。
- **普通帧**：逐帧 ≥ 0.6。门槛经过两次下调：0.85 → 0.7 → 0.6，都由用户决定。
- **目测帧**：同一帧有 2 个及以上实例，或最大实例的头高不到 200px。SSIM 照常统计，但不卡门槛，以肉眼对比时动作、表情、开关一致为准。
- **特效帧**：带半调、径向模糊、转场、粒子等效果，或者重度模糊的帧。同样以肉眼对比为准。
- 判定优先级：特效帧 > 目测帧 > 普通帧。
- 播放器带调试模式（原视频对照、差异图、逐帧步进）；每一帧都确定、可复现；音画同步。

**授权**
- 代码采用 MIT 许可。
- Cubism Core 和 Framework 都不进仓库，由 `pnpm setup:cubism` 从官方地址下载，并校验 sha256。
- 模型不进仓库，由 `pnpm setup:assets` 下载。模型条款禁止再分发模型文件，所以静态页**不公开部署**，只在本地查看。README 里写明了原作者发布页。
- **路线 A 注意**：Cubism Core 适用 Live2D Proprietary Software License。自己解析 moc3、实现变形器之前，请先确认这样做不违反该授权（尤其是其中的逆向工程条款）。只要 GPU 模型仅用于搜索参数，最终渲染仍然走官方 Core，风险最小。

**技术栈**：pnpm monorepo、TypeScript、Vite、Pixi.js v8、Cubism Framework R5；视频分析用 Python + OpenCV；拟合用 Playwright 无头 Chrome（GPU）。**拟合用的渲染和播放器必须是同一份 runtime 代码**；新的 GPU 前向模型只能用来搜索参数，最终产出仍然写成 scene.json，由 runtime 渲染并打分。

## 2. 代码结构与现状

| 位置 | 内容 | 状态 |
|---|---|---|
| `packages/scene` | 场景格式 `charaforge.scene` v1 和镜头表格式 `charaforge.shots` v1（TypeBox），JSON Schema，曲线求值，校验 | 稳定 |
| `packages/runtime` | Cubism 封装：确定性物理、参数覆盖钩子、离屏渲染、Worker 并行、合成器（滤镜、clip、frameHold、resample、scaleX/Y） | 稳定，13 项确定性测试 |
| `apps/player` | 播放器：音频时钟、调试模式；harness 页面；CLI：`pnpm render`、`render:gallery`、`bench`、`verify:video` | 稳定；3 个实例约 59.5fps（原机器）；P5 打磨过，WSL 机器实测见第 5 节 |
| `tools/recon` | 镜头分析，`manual.py` 是唯一的人工数据源，产出 `scenes/deepseek-chan/shots.json` | 46 个镜头，经过多轮修正 |
| `tools/fit` | 拟合引擎 fit4：面捕输入空间轨迹，加光流、雅可比、LM，加逐帧精修，加批量驱动 `scripts/batch.sh`（支持断点续跑） | 能用，但慢、有闪烁，见第 3 节 |
| `tools/qa` | `metric.py`（指标，唯一实现）和 `report.py`（全片报告、HTML、差异视频） | 能用；有未完成的改动，见第 4 节 |
| `scenes/deepseek-chan/scene.json` | 全片复刻场景：47 个图层，22 个拟合单元 | 全片可以播放 |

**常用命令**：`pnpm install && pnpm run setup`，然后 `pnpm dev`（播放器）、`pnpm test:runtime`、`pnpm render --scene ... --frames a-b --out dir`。

## 3. 当前质量与已知问题

**全片质检**（门槛 0.6，基于提交 73497b2 时的 scene.json，报告在原机器的 `.data/qa/report_full/`，没有进 git）

| 帧类别 | 有效帧 | 均值 | ≥ 0.6 |
|---|---|---|---|
| 普通帧 | 814 | 0.606 | 442（54%） |
| 目测帧 | 1644 | 0.561 | 707 |
| 特效帧 | 894 | 0.562 | 465 |

- 普通帧不达标的帧数：s13 80、s06 76、s25 54、s12 50、s19 38、s17 30、s23 26、s08 14。s04、s24、s26 基本全部达标。
- 最差的目测帧和特效帧镜头：s38 0.34、s35 0.33、s28 0.32、s21 0.48、s46 0.45。

**用户看过全片差异视频后反馈的问题**
1. **闪烁（是真问题）**：共 139 个尖峰帧，判定标准是我们的帧间变化 > 3 倍原视频，且 > 4。集中在 s21（32 帧）、s19（19）、s41（18）、s15（15）、s32（11）、s40（7）。原因有三：
   - 逐帧精修（postPhysics 全参数偏移，加上变换偏移）是逐帧独立求解的，没有时间正则；
   - 眼睛开合网格搜索也是逐帧独立的；
   - 变换偏移没有上限。

   s41 的 3543–3550 帧：原视频是平稳的半身，我们却渲染成了跳动的脸部特写。根因是 shots.json 里这一段的眼睛锚点标大了约 2 倍，置信度却是 0.9，拟合器的跑偏检测信任了这个锚点，把变换拉偏了。
2. **重影**：
   - 大部分来自差异视频的合成方式。`report.py` 的 sbs 布局把渲染合成到原视频上，没盖住的地方会透出原视频的角色；
   - 另外在错位的镜头里，确实也有两个角色错开的情况；
   - s30 的 2698–2740 段，shots.json 缺少 `s30.main` 的 bbox、head、eyes 字段，质检因此误判。

**这次 WIP 提交里没有验证过的改动**（都已提交，但没有跑完验证）
- `tools/fit`（C3）：
  - 逐帧精修加了时间平滑：先 3 点中值，再做二阶差分惩罚的 Whittaker 平滑，λ=4，参数是 `FIT_MODEL.pfSmooth`；
  - 变换偏移加了上限：位移 ≤ 0.3 倍眼距，尺度 ±5%，旋转 ≤ 3°；
  - 重锚门控：只有锚点偏差大**并且**光流残差也大时，才重锚；
  - 精修后对照锚点做发散回退；
  - 眼睛开合结果做 5 点中值；
  - 新增 `scripts/flicker.py`。

  原计划的验证是：先在 s41、s21 上看闪烁是否消失，再用 s24 回归（普通帧 50/50 ≥ 0.6，均值 ≥ 0.76）。**验证被中途停止，没有结果。**
- `tools/qa`（D）：给门槛相关的字段做通用化（新增 `fracGEthr` / `framesBelowThr`，旧的 0.7 字段按真正的 0.7 计算）、加闪烁指标、sbs 布局改成只显示纯渲染。**做到一半**，需要检查后补完。
- `tools/recon`（B4）：改了 s06、s19、s21、s25、s30、s41 的标注（s06 是窗帘边框随时间起伏的排除区和窗口裁切等）。**做到一半**。validate 和 parseShots 都能通过，但内容没有经过复核。以下几项还要核实：
  - s41.main 3538–3558 的眼睛锚点应该是约 (120,133)、(227,133)，眼距约 107px；
  - s30.main 2698–2740 要补上 bbox、head、eyes；
  - s30.full 2690–2770 的 blurSigma 应该接近 0；
  - 其余待审查的镜头：s13、s12、s17、s23、s08、s38、s35、s28。

## 4. 路线 A：可微分的 GPU Live2D（下一步主线）

背景和其他方案见 `docs/gpu-acceleration.md`。现有引擎的耗时构成是：逐帧精修 31%、雅可比 19%、求解 15%、渲染 15%、光流 14%。瓶颈在 Cubism Core 只能在 CPU 上跑（asm.js），每次更新 10～14ms，而有限差分需要大量更新。

**目标**：把「面捕输入 → VTS 映射与平滑 → 模型参数 → 变形器 → 物理 → 光栅化 → 损失」整条链路做成可微分的 GPU 实现（PyTorch + nvdiffrast），对整段轨迹一起做梯度优化。
- 拿 s24 做验收：几分钟内拟合到当前水平，普通帧 0.765；
- 全片控制在 1 小时以内。

**建议步骤**
1. **moc3 解析器**（先确认授权）：读出 parts、deformers（warp 和 rotation）、art meshes、keyform bindings（参数关键值和混合规则）、draw order、clipping masks、混合模式和颜色。用 Core 做对拍：对随机参数，比较各 drawable 的顶点位置、不透明度、渲染顺序，**顶点误差 < 0.1px** 才算合格。
2. **变形器的 PyTorch 实现**：
   - warp deformer 是带二次插值和外插的网格变形；
   - rotation deformer 是旋转、缩放和原点；
   - keyform 在多个参数上做多线性插值；
   - 可以批量、可以求导。
3. **物理**：把 Cubism Framework 的 `CubismPhysics`（开源，在 `vendor/cubism/framework/src/physics/`）移植成 PyTorch，按固定 dt=1/60 逐帧推进，可微分。用 runtime 的输出逐帧对拍。
4. **光栅化**：用 nvdiffrast 按 drawable 顺序光栅化贴图三角形。要处理：clipping mask、预乘 alpha、加算和乘算混合、多重颜色或屏幕色（如果模型用到）、resample 和 blur 滤镜，以及图层变换和 clip。和 runtime 渲染对拍，SSIM > 0.99。
5. **前向和损失**：
   - 前向链路复用 `tools/fit/src/vts.ts` 的映射语义（ParameterSettings 加平滑），在 Python 里重新实现一份；
   - 损失是 SSIM@360，加上边缘或光流项，再加锚点先验（小角色时才作为数据项）；
   - **时间正则**（二阶差分）**直接作为优化目标的一部分**，从根上消除闪烁；
   - 变换只用稀疏关键帧表示，限制变化速度。
6. **导出**：写成 scene.json 的参数曲线和变换曲线（规范见 `packages/scene/src/scene.ts`），用 `@charaforge/scene` 的 `parseScene` 校验，然后用 `pnpm render` 和 `tools/qa/metric.py` 做最终打分。**最终指标只认 runtime 渲染出来的结果。**
7. **顺带要修的错误**：
   - 闪烁：靠时间正则解决，验收用 `tools/fit/scripts/flicker.py` 或 `tools/qa/flicker.py`，尖峰帧数接近 0；
   - 重影：sbs 改成纯渲染，并修正错位；
   - 补完第 3 节的 shots 审查，修 s41、s30 的锚点；
   - 修完再出一次全片报告和差异视频给用户看。

## 5. 剩余阶段

- **P4（完成，提交 f39cce0）**：用新引擎重拟合全片，替换 `scenes/deepseek-chan/scene.json`。
  - 28 个拟合单元、47 个图层，全部来自 `tools/gpu` 的整段轨迹联合优化（冷启动，`a5_batch.py --max-frames-per-batch 450`，拟合约 3 小时 18 分）。
  - 全片质检（runtime 渲染 + `metric.py`，门槛 0.6，报告在拟合机器的 `.data/qa/p4b/`）：普通帧 **834/916 ≥ 0.6**（旧引擎 543/901），均值 **0.750**；
    目测帧均值 0.733（1478/1696 ≥ 0.6）；特效帧均值 0.690（596/760）；闪烁尖峰 **21**（旧引擎 143）。
  - 遗留问题：s13 窗口展开、s26 笔刷和 ✓ 的排除区、s12，以及剩余的闪烁帧（s13 5、s32 5、s15 4、s31 2 等）。
- **P5（完成，未提交）**：播放器打磨、`pnpm build` 静态页、README。
  - **播放器修复**：
    - 启动只建一个引擎（以前 React StrictMode 下建两个运行时共用一个 canvas，`window.__player` 偶尔指向被丢弃的那个）；
    - 加载状态（分阶段 + 已用时间）、出错提示（缺 Cubism Core / 模型 / 场景 / 校验失败，带处理办法；音频、参考视频加载失败给非致命提示）；
      缺 Core 时以前是白屏（Framework 在模块加载时就抛错），现在 `index.html` 直接给出提示；
    - 快捷键在按钮、滑块、单选框获得焦点时也能用（以前点过对照模式的单选框后快捷键全部失效），空格不再同时触发有焦点的按钮；
      新增 `1`～`4` 对照模式、`M` 静音；控制栏显示快捷键；
    - 播放中打开对照模式时参考视频跟着播放（以前视频停着，每帧被重新 seek）；
    - 差异图 / 叠加改为把渲染合成到原视频上再比较（与指标的 C_f 同口径；以前合成到黑底，背景整片都算差异）——`packages/runtime/src/shaders.ts` 的 `DEBUG_FRAG`，只用于 `presentDebug`；
    - 署名去重（scene 的 `meta.credits` 和模型 `credits` 相同，以前显示两遍并报 React key 警告），链接可点，加模型发布页与「禁止再分发、不公开部署」说明；
    - 远距离 seek 时显示「定位中」；「复制本帧链接」；画面高度限制在视口内，控制条不被挤出首屏；
    - **后台预热**（`PlayerEngine.startPrewarm`，`?prewarm=0` 关闭）：暂停时把各图层的 in 帧离屏渲染一遍。不预热时全片播放到新图层要卡 0.1～0.7 秒
      （Worker 90 帧物理预热 + 主线程首次画实例），60 秒丢帧 22～24%；
    - 合成器启动时预先链接全部 pass 着色器（`Compositor.warmShaders`）：WSL 的 ANGLE/D3D12 下 color、resample 首次链接各约 0.3 秒，以前卡在 s04、s06 的第一帧；
    - `pnpm build` 后 Worker 里启动 Framework 时 Cubism Core 6 的 WASM 还没实例化完（`Cannot read properties of undefined (reading 'apply')`，
      开发模式下因为模块逐个加载而碰巧没事），新增 `packages/runtime/src/coreReady.ts` 等它就绪（Worker 与主线程）。
    - 运行时改动不影响输出：`pnpm test:runtime` 12 通过、1 跳过；全片抽 9 帧 `pnpm render` 与 P4 的渲染逐字节相同。
  - **实测**（WSL2、RTX 5080 Laptop、8 逻辑核 → 缺省 3 个 Worker；测量时另一个拟合任务占用 GPU 64～85%，数据偏保守）：
    | 场景 | fps | 上屏间隔 p99 / 最大 | 丢帧率 | 画面落后声音（帧）均值 / 最大 |
    |---|---|---|---|---|
    | 全片 60 秒，后台预热完（12.5 秒）再播 | **58.97** | 33.4 / 50.1 ms | 1.75% | 0.76 / 3.1 |
    | 全片 60 秒，就绪后立即播（未预热） | 45.48 | 217 / 717 ms | 24% | 0.76 / 9.6 |
    | `demo:3`（3 个大实例），3 Worker（缺省），3 次 | 56.75 / 53.55 / 53.15 | 33～50 / 50～83 ms | 5.5～11.3% | 1.1～1.2 / ≤ 5.0 |
    | `demo:3`，4 / 5 / 6 Worker | 56.55、51.10 / 55.25 / 49.90 | 33～50 / 33～117 ms | 5.8～16.5% | 1.2～1.6 / ≤ 4.4 |
    | `demo:1`、`demo:2`（3 Worker） | 57.73、60.07 / 58.13 | 16.8～33.4 / ≤ 50 ms | 0～3.6% | 0.4～0.9 / ≤ 2.7 |
    | `demo:4`（6 Worker） | 32.13、22.67 | 83～133 / 100～150 ms | 46～62% | 2.2～3.0 / ≤ 9.3 |

    原机器（12 逻辑核、5～6 个 Worker、无其他负载）`demo:3` 为 59.3～59.5 fps；这台 8 核机器 3 个 Worker 不够、6 个 Worker 又和主线程 / GPU 进程抢核。
    音画同步：时钟就是音频输出时间（`getOutputTimestamp` 外推），60 秒内帧号前进 59.97～60.00 帧/秒，前 10% 与后 10% 样本的滞后均值 0.71 / 0.735 帧，没有漂移；
    把音频请求屏蔽、改用系统时钟时卡顿情况相同，说明卡顿与音频时钟无关。`pnpm verify:video`：50/50 帧逐帧精确。
    方法：`pnpm bench <场景> <秒数> [workers] [起始帧]`（改过：真实点击播放、报告音画滞后与预热时间；`BENCH_QUERY`、`BENCH_PLAY_NOW`）。
  - **静态页**：`pnpm build` → `apps/player/dist/`（约 1.5MB、15 个文件，不含模型、视频、音频、场景、Cubism Core；含编译后的 Framework）；
    `pnpm preview`（新增，`vite preview` 复用白名单中间件 `repoFiles()`）在 `http://localhost:4173/` 完整播放全片、调试模式可用。
    打包目录改为 `_app/`（缺省 `assets/` 会和白名单 `/assets/` 冲突）。
  - **README**：项目说明与管线图、三层授权与模型署名、路线 A 说明、环境要求（WSL d3d12）、安装（含 gpu 依赖组两步安装）、播放器与调试模式、静态页、
    复现整条管线（setup → recon → fit → render → qa → build，命令都实际核对过）、验收口径与 P4 结果、目录结构。
  - **没做完 / 建议**：
    - 未预热就播放仍会在新图层处卡顿。主线程部分（每个新实例 0.1～0.6 秒，越往后越大）不是着色器编译也不是 JS，推测是每个图层一个实例、
      各自的 GL 资源（遮罩缓冲 2048² 等，47 个实例）。根治要改运行时：Worker 路径下主线程实例只是绘制目标（动态数据全部导入覆盖），
      可以在时间上不重叠的图层之间复用，或者释放远离播放头的实例的 GL 资源。需要总负责人决定。
    - 预热进行中按播放，最多要等当前预热帧做完（约 0.1～0.6 秒）才开始走。
    - 4 个实例同屏在这台机器上只有 23～32 fps（原机器 30 fps），见 `docs/runtime-eval.md` 第 6 节。
    - 其他浏览器（Firefox、Safari）未验证。

## 6. 环境与操作注意

- **新机器**：
  1. 运行 `pnpm install && pnpm run setup`，会下载 SDK、模型和视频，校验 sha256，并建立 Python 3.12 的 venv（`uv sync --project tools`）。
  2. 运行 `pnpm test:runtime` 确认 GPU 可用。启动器会检查 WebGL renderer，必须是 NVIDIA；原机器需要去掉 `DISPLAY` 才不会卡住，说明见 `docs/runtime-eval.md`。
- **WSL2 机器**（RTX 5080 Laptop，2026-11）：
  - GPU 走 Mesa d3d12（WSL 里没有 NVIDIA 的 Vulkan / EGL ICD）。`harness-node.ts` 自动检测并缺省为 egl + `GALLIUM_DRIVER=d3d12`
    + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` + **MSAA 0**（MSAA 4 在此后端不确定），不用设环境变量；显式设置的优先。
    renderer 应为 `ANGLE (Microsoft Corporation, D3D12 (NVIDIA GeForce RTX 5080 Laptop GPU), OpenGL ES 3.1)`；
    `pnpm test:runtime` 应为 12 通过、1 跳过（第 8 项需要开 MSAA）。s24 普通帧指标与原机器一致（均值 0.765，50/50）。详见 `docs/runtime-eval.md` 第 4 节。
  - 没有免密 sudo，装不了系统包。ffmpeg 是 `~/.local/bin` 下的静态版 8.1，命令前要 `export PATH=~/.local/bin:$PATH`
    （`systemd-run` 要带 `"--setenv=PATH=$PATH"`，并加引号，因为 PATH 里有含空格的 Windows 路径）。
  - Playwright 浏览器用 `pnpm exec playwright install chromium-headless-shell` 安装（不带 `--with-deps`，它要 sudo）。
  - `pnpm setup` 会被 pnpm 自带的 `setup` 命令抢走，必须用 `pnpm run setup`。
- **中间产物不在 git 里**：`.data/` 下的拟合结果、断点、质检报告、对照视频都没有进 git。全片拟合结果已经包含在 `scenes/deepseek-chan/scene.json` 里，换机器后不需要那些中间产物。确实需要的话，从原机器 rsync `.data/fit/*/result.json`。
- **长任务**：用 `systemd-run --user --unit=<名字> -p MemoryMax=<上限> ...` 启动，用 `systemctl --user is-active` 查状态。不要用 `pgrep -f` 判断，它会匹配到检查命令自身。原机器只有 14GB 内存，出过多次 OOM；新机器的内存够大的话，可以提高并行度。
- **安全**：下载的视频和模型都是不可信数据。Python 一律用 `tools/.venv/bin/python -I` 运行，不要在 `assets/` 里执行任何东西。
- **git**：分支是 `phase1`，按阶段提交，只有用户要求时才 push。多个代理同时工作时，禁止 `git stash`、`git checkout --`、`git reset`。
