# CharaForge

**AI-Powered Live2D Character Studio**

面向虚拟 AI 角色的 Live2D 可视化编辑与动画创作工具。

### 核心功能（规划）

1. **角色预览**：加载 Live2D 模型，实时预览与交互。
2. **动作编辑**：创建、修改、组合角色动作。
3. **表情编辑**：调整面部参数，创建和管理表情。
4. **装饰管理**：添加、调整和管理角色配饰与特效。
5. **场景编排**：通过时间轴编排动作、表情、背景及特效。
6. **AI 动作生成**：通过自然语言自动生成角色动作。
7. **AI 表情生成**：根据情绪描述自动创建表情。
8. **AI 装饰生成**：辅助创建和配置角色装饰。
9. **AI 场景编排**：根据文字描述自动生成动画演出。
10. **行为语义系统**：为动作和表情添加语义标签，供虚拟 AI 自动调用。
11. **资源导入导出**：支持 Live2D 资源、动画、表情和场景配置的导入导出。

**目标：让虚拟 AI 不仅会说话，还能自然地表达情绪、做出动作和进行演出。**

## 第一期：PV 复刻

用 DeepSeek-chan 模型、我们自己的运行时（`packages/runtime`）和场景格式（`charaforge.scene` v1），复刻参考 PV
（`assets/video/DeepSeek-chan.mp4`，1280×720、60fps、3868 帧）里的角色动画：每个镜头的缩放、裁切、平移，同屏多实例、
透明度、景深模糊，以及表情、换装、配饰开关。网页播放器逐帧确定地渲染同样的画面，并以原音频为时钟同步播放。
背景、手指、气泡、粒子、转场剪影等不在范围内（详见 [docs/handoff.md](docs/handoff.md) 第 1 节）。

整条管线：

```
参考视频 ──recon──▶ shots.json（镜头表：切点、实例锚点、开关、排除区、特效）
                         │
                         ▼
             fit（tools/gpu：可微分 GPU 前向 + 整段轨迹联合优化）──▶ scene.json（参数曲线、图层变换）
                         │
                         ▼
     runtime 渲染（pnpm render，与播放器同一份代码）──qa──▶ SSIM@360 / 闪烁 / 报告与差异视频
                         │
                         ▼
               播放器（pnpm dev / pnpm build + pnpm preview）
```

当前全片结果见下文「验收口径与当前结果」。

## 授权

本项目分三层授权：

| 部分 | 许可 | 说明 |
|---|---|---|
| CharaForge 代码 | [MIT](LICENSE) | 本仓库中的全部代码 |
| Live2D Cubism Core | [Live2D Proprietary Software License](https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html) | **不包含在仓库中**，`pnpm setup:cubism` 从 Live2D 官方地址下载 Cubism SDK for Web 5-r.5 并校验 sha256，装到 `vendor/cubism/core/` |
| Live2D Cubism Framework | [Live2D Open Software License](https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html) | 同上，源码装到 `vendor/cubism/framework/`，构建时编译到 `.cache/cubism-framework/`（都不进仓库） |
| DeepSeek-chan 模型 | 原作者的使用条款（随模型附带，`assets/models/deepseek-chan/terms/`） | **不包含在仓库中**，`pnpm setup:assets` 下载；条款**禁止再分发**模型文件 |

**模型署名**：**DeepSeek-chan** — Illustration & PSD Layer Separation: Kamiya Ryoko ／
Live2D Rigging: 霓虹猫_NeonNeko ／ Free model distributed by Kamiya Ryoko
（[VGen](https://vgen.co/KAMIYA)、[X @kamiya_ryoko](https://x.com/kamiya_ryoko)）。
同样的署名写在 `scenes/deepseek-chan/scene.json` 的 `meta.credits` 和模型资源的 `credits` 里，播放器页脚会显示。

模型请从原作者的发布页获取并支持原作者：
<https://vgen.co/KAMIYA/product/deepseek-chan-deepseek-free-live2d-model-layered-psd-psd/1c55a71a-2747-4616-819d-2dc0a934089d>

**不公开部署**：播放器（包括 `pnpm build` 的静态页）只在本机查看。页面运行时要加载模型、参考视频和音频，
公开部署就等于把模型文件再分发出去，违反模型条款。`pnpm build` 的产物里也不含任何模型、视频、音频或 Cubism Core（见「静态页」一节）；
产物里有编译后的 Cubism Framework 代码（Live2D Open Software License），同样只供本地使用。

**路线 A（moc3 解析器）**：`tools/gpu/l2d/` 里的 moc3 解析器、PyTorch 变形器、物理与光栅化只是**内部拟合工具**，
用来在 GPU 上搜索参数；它不进播放器、不对外分发。最终画面一律由官方 Cubism Core + Framework 渲染（`packages/runtime`），
打分也只认 runtime 的渲染结果。Cubism Core 授权里有限制逆向工程的条款，这条路线的风险由项目负责人知悉并决定保留
（[docs/handoff.md](docs/handoff.md) 第 1 节「路线 A 注意」）。

## 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Linux，或 Windows 上的 WSL2（Ubuntu 24.04 实测） |
| GPU | NVIDIA（渲染、测试、拟合都会检查 WebGL renderer 是否为 NVIDIA，防止悄悄落到软件渲染） |
| Node.js / pnpm | Node ≥ 22（用 `--experimental-strip-types` 直接跑 TS 脚本），pnpm 10 |
| Python | 3.12，由 [uv](https://docs.astral.sh/uv/) 管理（`tools/.venv`） |
| ffmpeg | 在 `PATH` 里（setup 解码音频、recon、qa 用）；WSL 机器上是 `~/.local/bin` 下的静态版 |
| CUDA | 只有 GPU 拟合（`tools/gpu`）需要：CUDA 13.0 工具链（`CUDA_HOME=/usr/local/cuda-13.0`），torch 2.14.1+cu130，nvdiffrast 0.4.0 |
| 浏览器 | 播放器用 Chrome / Edge（需要 WebGL2、Web Audio、H.264；Firefox 未验证） |

**WSL2 说明**：WSL 里没有 NVIDIA 的 Vulkan / EGL 驱动，无头 Chromium 只能经 Mesa 的 d3d12 驱动走 Windows 的 D3D12。
`apps/player/scripts/harness-node.ts` 检测到 WSL（有 `/usr/lib/wsl/lib/libd3d12.so`、没有 NVIDIA Vulkan ICD）时自动使用
`--use-angle=gl-egl` + `GALLIUM_DRIVER=d3d12` + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`，并把 MSAA 设为 0（此后端开 MSAA 不确定）。
renderer 应为 `ANGLE (Microsoft Corporation, D3D12 (NVIDIA GeForce …), OpenGL ES 3.1)`。显式设置的 `CHARAFORGE_GPU` /
`CHARAFORGE_MSAA` 等环境变量优先。细节见 [docs/runtime-eval.md](docs/runtime-eval.md) 第 4 节。
在 WSL 里运行 `pnpm dev` 后，用 Windows 上的 Chrome / Edge 打开 `http://localhost:5173` 即可（WSL2 自动转发 localhost，页面用 Windows 原生 GPU）。

## 安装

```bash
pnpm install
pnpm run setup        # = setup:cubism + setup:assets + setup:py
```

- `pnpm run setup` 不能写成 `pnpm setup`（后者是 pnpm 自带命令）。
- `setup:cubism`：下载 Cubism SDK for Web 5-r.5（校验 sha256），装 Core 与 Framework 到 `vendor/cubism/`，运行前会打印两份 Live2D 许可的链接。
- `setup:assets`：下载模型（去掉 131MB 的分层 PSD，保留原文件名和使用条款）和参考视频，校验 sha256，再用 ffmpeg 把音频解码成 `assets/video/DeepSeek-chan.wav`（PCM，避免 AAC 编码延迟造成音画偏移）。
- `setup:py`：`uv sync --project tools`，建立 `tools/.venv`（numpy、opencv、scikit-image、jsonschema、pillow）。
- Playwright 浏览器：`pnpm exec playwright install chromium-headless-shell`（没有 sudo 时不要加 `--with-deps`）。

**GPU 拟合依赖**（`tools/pyproject.toml` 的 `gpu` 依赖组）。nvdiffrast 安装时要编译，并且关闭了构建隔离（`no-build-isolation-package`），
编译时要用到已经装好的 torch，所以分两步：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
uv sync --project tools --group gpu --no-install-package nvdiffrast   # 1. torch 2.14.1+cu130、triton、ninja 等
uv sync --project tools --group gpu                                   # 2. 编译并安装 nvdiffrast 0.4.0（约 1 分钟）
```

注意：`uv sync` 缺省会删掉没被选中的依赖组。装了 gpu 组之后，再跑 `pnpm run setup` / `pnpm setup:py`（不带 `--group gpu`）
会把 torch 卸掉；之后同步请用 `uv sync --project tools --group gpu`。

**验证**：

```bash
pnpm test:runtime            # 运行时确定性测试；WSL：12 通过、1 跳过（第 8 项需要 MSAA）；原生 Linux + Vulkan：13 通过
pnpm -r run typecheck
tools/.venv/bin/python -I tools/gpu/tests/test_a1_moc3.py    # moc3 解析器与 Core 对拍；tools/gpu/tests/*.py 都可以这样直接运行
```

## 播放器

```bash
pnpm dev               # http://localhost:5173/
```

缺省打开全片场景 `scenes/deepseek-chan/scene.json`（约 25MB，47 个图层），点「播放」或按空格开始，
原音频为时钟：画面帧号 = floor(音频时间 × 60)。

- 加载约 6 秒出第一帧（页面显示加载阶段），之后**可以立即播放**。加载后 Worker 会提前为各图层做物理预热
  （约 4 秒，不阻塞播放、不改变画面，控制栏显示「预热图层 k/37」），播放中也会提前预热前方 3 秒内出现的图层。
- 实测（WSL2、RTX 5080 Laptop、8 逻辑核、3 个 Worker）：加载完立即播放全片 60 秒 58.5～59.0 fps、丢帧 1.7～2.5%
  （另一拟合任务把 GPU 占到 84～92% 时 3.6～3.8%），画面相对声音滞后均值约 0.8 帧，60 秒内无漂移。
  数据与方法见 [docs/handoff.md](docs/handoff.md) 第 5 节。

URL 参数：

| 参数 | 含义 |
|---|---|
| `scene=<仓库相对路径>` | 场景文件，缺省 `scenes/deepseek-chan/scene.json`；`demo:N` 为内置的 N 实例演示场景 |
| `video=<仓库相对路径>` | 参考视频，缺省 `assets/video/DeepSeek-chan.mp4` |
| `debug=1` | 打开调试面板 |
| `compare=side\|overlay\|diff` | 初始对照模式（配合 `debug=1`） |
| `frame=<n>` | 初始帧 |
| `workers=<n>` | 求值 Worker 数，缺省 clamp(逻辑核数/2 − 1, 2, 6)；0 = 主线程同步求值 |
| `prewarm=0` | 关闭新图层的 Worker 预热（只用于诊断；关掉后全片丢帧约 10%） |

快捷键：`空格` 播放/暂停，`←`/`→` ±1 帧（`Shift` ±10），`Home`/`End`，`[`/`]` 上/下一个镜头，`D` 调试面板，
调试面板打开时 `1`～`4` 切换对照模式（关 / 并排 / 叠加 / 差异图），`M` 静音。

**调试模式**（`D` 或 `?debug=1`）：

- **原视频对照**：并排（右侧原视频）、叠加（渲染合成到原视频上，再与原视频按不透明度混合，错位处出现重影）、
  差异图（|渲染合成到原视频 − 原视频| × 放大倍数，与验收指标的 C_f 同一口径，没画到角色的地方为 0）。
- **逐帧步进**：暂停时参考视频逐帧精确对齐（seek 到帧中点并等新帧上屏，`pnpm verify:video` 验证 50/50 帧精确），
  面板显示「已逐帧对齐」；播放中视频按自己的时钟走、漂移超过 2 帧才重新对齐，叠加 / 差异图请暂停后看。
- **复制本帧链接**：把当前帧、调试与对照模式写进地址栏并复制，打开即复现同一帧。
- 每帧确定：画面只是帧号的函数，任意 seek、掉帧补算物理后都与顺序播放逐位相同（`pnpm test:runtime` 第 2、3、7 项）。
  远距离 seek 时 Worker 要从最近的物理快照补算，画面左上角显示「定位中」，实测 0.2～1 秒。

其他命令：

```bash
pnpm render --scene scenes/deepseek-chan/scene.json --frames 2368-2441 --out .data/render/s24   # 离线渲染 PNG（与播放器同一份渲染代码）
pnpm render:gallery                          # 开关 / 表情图集 → .data/gallery/
pnpm bench demo:3 20                         # 帧率基准（参数：场景 秒数 [workers] [起始帧]），同时报告音画滞后
pnpm bench scenes/deepseek-chan/scene.json 60    # 全片（带原音频）
pnpm verify:video                            # 参考视频逐帧定位精度（与 ffmpeg 解码比对）
```

## 静态页（`pnpm build`）

```bash
pnpm build             # → apps/player/dist/（约 1.5MB，只有页面和打包后的代码）
pnpm preview           # → http://localhost:4173/
```

- 产物**不含**模型、参考视频、音频、场景文件和 Cubism Core，也不要把它们复制进去。页面按 URL 引用仓库里的原文件：
  `/assets/…`（模型、视频、音频）、`/vendor/cubism/core/…`（Core）、`/scenes/…`、`/packages/scene/examples/…`、`/packages/scene/schema/…`。
- 提供这些 URL 的是 `apps/player/vite.config.ts` 里的白名单中间件 `repoFiles()`：只读，只开放上面几个目录，
  拒绝 `..` 穿越，支持 Range 请求（视频精确 seek 需要）。`pnpm dev` 和 `pnpm preview`（`vite preview`）用的是同一个中间件，
  所以 `pnpm preview` 就能完整播放全片，调试模式、原视频对照也都可用。
- 打包代码放在 `dist/_app/`（不用 Vite 缺省的 `assets/`，否则会和白名单的 `/assets/` 抢同一个 URL 前缀）。
- 不能用 `file://` 直接打开 `dist/index.html`（Worker、fetch、Range 请求都需要 HTTP），也不要换成别的静态服务器：
  那样要么拿不到 `/assets/`，要么就得把模型复制进可发布的目录。
- `vite preview` 缺省只监听 localhost。**不要**加 `--host`，也不要部署到公网（模型禁止再分发）。

## 复现整条管线

以下命令都在仓库根目录、WSL / Linux 下运行。视频和模型是不可信数据：Python 一律 `tools/.venv/bin/python -I`，不要在 `assets/` 里执行任何东西。
中间产物都在 `.data/`（不进 git）。长任务建议用 `systemd-run --user … -p MemoryMax=…` 启动（见 [docs/handoff.md](docs/handoff.md) 第 6 节）；
WSL 下 ffmpeg 在 `~/.local/bin`，先 `export PATH=~/.local/bin:$PATH`。

### 0. setup

见「安装」：`pnpm install && pnpm run setup`，GPU 依赖组两步安装，`pnpm test:runtime` 确认 GPU 可用。

### 1. recon：镜头表 `scenes/deepseek-chan/shots.json`

`tools/recon/manual.py` 是人工标注的唯一来源（切点、实例锚点、开关、排除区、特效），其余是自动步骤。详见 [tools/recon/README.md](tools/recon/README.md)。

```bash
tools/.venv/bin/python -I tools/recon/extract.py            # 解码：逐帧 JPEG、320×180 缓存、视频信息 → .data/recon/
tools/.venv/bin/python -I tools/recon/detect_cuts.py        # 切点候选（再加 --flow 补运动掩盖的硬切）
tools/.venv/bin/python -I tools/recon/cadence.py            # 30fps 素材的帧重复段
tools/.venv/bin/python -I tools/recon/transitions.py        # 转场里角色可见区域（依赖上一次 build 的 shots.json）
tools/.venv/bin/python -I tools/recon/manual.py             # 人工标注 → annotations.json
tools/.venv/bin/python -I tools/recon/build.py              # 跟踪加密 + 虹膜细化 → shots.json、缩略图、.data/recon/index.html
# 校验：JSON Schema + 语义（Python），parseShots（TS）
tools/.venv/bin/python -I tools/recon/validate.py --sha --check-cadence
node --experimental-strip-types -e "import('./packages/scene/src/index.ts').then(m=>m.parseShots(JSON.parse(require('fs').readFileSync('scenes/deepseek-chan/shots.json','utf8'))))"
```

首次（或改了锚点后）按 build → transitions → manual → build 跑两遍。校对用 `.data/recon/index.html`。

### 2. fit：GPU 引擎拟合 → scene.json

拟合按**单元**进行（`continues` 串起来的镜头合成一个单元，一个或多个图层），每个单元的整段轨迹在 GPU 上联合优化，
导出成 scene.json 图层。原理、参数与各单元结果见 [tools/gpu/docs/fit.md](tools/gpu/docs/fit.md)，提速见 [tools/gpu/docs/perf.md](tools/gpu/docs/perf.md)。

```bash
export PATH=~/.local/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 单个单元（镜头可以是单元内任一镜头），约 3～20 分钟；--quick 迭代数 ÷4 用于调试
tools/.venv/bin/python -I tools/gpu/scripts/a5_fit.py --shot s24 --out .data/gpu/fit/runs/s24
# 合并进临时全片场景（--shot 给单元内全部镜头；parseScene 校验通过才写出）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_merge.ts \
  --base scenes/deepseek-chan/scene.json --shot s24 --layer .data/gpu/fit/runs/s24/layers.json --out .data/gpu/fit/runs/s24/scene.json

# 全片批量（28 个单元，按时间顺序每 4 个一批、每批不超过 450 帧；RTX 5080 Laptop 约 3 小时 18 分）
systemd-run --user --unit=a5-full --collect --working-directory=$HOME/CharaForge \
  "--setenv=PATH=$PATH" --setenv=CUDA_HOME=$CUDA_HOME --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -p MemoryMax=8G \
  tools/.venv/bin/python -I tools/gpu/scripts/a5_batch.py --out .data/gpu/fit/full --max-frames-per-batch 450
systemctl --user is-active a5-full          # 进度看 .data/gpu/fit/full/batch.log
#   断点续跑：原样重跑同一条命令。已完成的单元（units/<单元>/done.json 指纹一致）跳过，中断的那一批整批重跑。
#   指纹 = shots.json + 拟合代码 + 参数的哈希，改了其中任何一项，相关单元都会重拟。
#   只跑部分单元：--units s04,s08；只看分批：--dry；只重新合成：--assemble-only
# 合成全片场景（a5_batch 结束时会自动做；不会覆盖 scenes/deepseek-chan/scene.json）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a5_assemble.ts \
  --units .data/gpu/fit/full/units --out .data/gpu/fit/full/scene.json
# 验收通过后，人工把结果提升为正式场景
cp .data/gpu/fit/full/scene.json scenes/deepseek-chan/scene.json
```

旧的 CPU 拟合引擎 fit4（`tools/fit`，CMA-ES + 无头 Chromium 有限差分）已被取代但保留，GPU 引擎的单元规划（`plans.ts`）
和导出格式仍复用它的代码；用法见 [tools/fit/README.md](tools/fit/README.md)。

### 3. render：runtime 渲染

```bash
pnpm render --scene .data/gpu/fit/full/scene.json --frames 0-3867 --out .data/qa/full/renders
```

最终指标**只认 runtime 渲染**（与播放器同一份代码，`pnpm render` 两次运行的 PNG 逐位相同）。下一步的 `report.py --render` 也会自己调用它。

### 4. qa：指标、闪烁、报告与差异视频

指标定义见 [docs/pipeline.md](docs/pipeline.md) 最后一节，唯一实现 `tools/qa/metric.py`；用法见 [tools/qa/README.md](tools/qa/README.md)。

```bash
# 单镜头指标 JSON（SSIM@360 主指标、SSIM@720、覆盖率、帧类别、闪烁）
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/gpu/fit/runs/s24/render --shots s24 --out .data/gpu/fit/runs/s24/metric.json
# 闪烁（帧间变化尖峰）
tools/.venv/bin/python -I tools/qa/flicker.py .data/qa/full/renders 0-3867
# 全片报告 + 并排差异视频（先 pnpm render 再打分；index.html、frames.csv、shots.csv、diff_sbs.mp4；
# RTX 5080 Laptop 上渲染约 9 分钟、打分与视频约 20 分钟）
systemd-run --user --unit=cf-qa-report --collect --working-directory=$HOME/CharaForge "--setenv=PATH=$PATH" "--setenv=HOME=$HOME" -p MemoryMax=3G \
  tools/.venv/bin/python -I tools/qa/report.py --scene .data/gpu/fit/full/scene.json --render \
    --renders .data/qa/full/renders --out .data/qa/full
```

报告打开 `.data/qa/full/index.html`；差异视频 `.data/qa/full/diff_sbs.mp4`（原视频 | 纯渲染 | 差异图，带原音频与逐帧标注）。

### 5. build：播放

```bash
pnpm build && pnpm preview      # 或开发模式 pnpm dev
```

## 验收口径与当前结果

口径见 [docs/pipeline.md](docs/pipeline.md)「验收指标的定义」：SSIM@360（渲染按 alpha 合成到参考帧上，与参考帧一起降到 640×360，
只在角色遮罩内、扣掉排除区计算）。帧分三类，判定优先级 特效帧 > 目测帧 > 普通帧：

- **普通帧**：逐帧 SSIM@360 ≥ 0.6；
- **目测帧**（同屏 ≥ 2 个实例，或最大实例头高 < 200px）、**特效帧**（半调、径向模糊、转场、粒子、重度模糊等）：照常统计，以并排、叠加对比时肉眼一致为准。

**P4 全片结果**（GPU 引擎冷启动重拟合，28 个单元、47 个图层；runtime 渲染 + `metric.py`，门槛 0.6；报告在拟合机器的 `.data/qa/p4b/`）：

| 帧类别 | 有效帧 | SSIM@360 均值 | ≥ 0.6 |
|---|---|---|---|
| 普通帧 | 916 | **0.750** | **834 / 916**（旧引擎 543 / 901） |
| 目测帧 | 1696 | 0.733 | 1478 |
| 特效帧 | 760 | 0.690 | 596 |

闪烁尖峰 **21** 帧（旧引擎 143）。遗留问题：s13 窗口展开、s26 笔刷和 ✓ 的排除区、s12，以及剩余的闪烁帧。

## 目录结构

| 路径 | 内容 |
|---|---|
| `apps/player/` | 播放器（React + Pixi，`src/main.tsx` UI、`engine.ts` 渲染循环、`clock.ts` 音频时钟、`videoSync.ts` 参考视频逐帧对齐）；`harness.html` 给测试和拟合用；`scripts/` 下是 render、bench、verify-video 等 CLI |
| `packages/runtime/` | 运行时：Cubism 封装、确定性物理、Worker 并行求值、合成器（滤镜、clip、frameHold）；`test/` 确定性测试 |
| `packages/scene/` | 场景格式 `charaforge.scene` v1、镜头表 `charaforge.shots` v1（TypeBox 定义、JSON Schema、`parseScene` / `parseShots`） |
| `scenes/deepseek-chan/` | `shots.json`（recon 产出、人工校对）、`scene.json`（全片复刻场景，fit 产出） |
| `tools/recon/` | 参考视频镜头分析（Python） |
| `tools/gpu/` | GPU 拟合引擎：`l2d/`（moc3 解析、变形器、物理、nvdiffrast 光栅化，内部工具）、`fit/`（整段轨迹优化、导出）、`scripts/`（`a5_fit.py`、`a5_batch.py`、`a5_assemble.ts` 等）、`docs/`、`tests/` |
| `tools/fit/` | 旧 CPU 拟合引擎 fit4（保留；单元规划与导出代码仍被 GPU 引擎复用） |
| `tools/qa/` | 验收指标 `metric.py`、闪烁 `flicker.py`、全片报告 `report.py` |
| `scripts/` | `setup-cubism.mjs`、`setup-assets.mjs`（下载并校验外部资源） |
| `docs/` | `pipeline.md`（目录、契约、验收口径）、`handoff.md`（交接与阶段状态）、`runtime-eval.md`（运行时实现、GPU 与确定性、性能）、`gpu-acceleration.md` |
| `vendor/`、`assets/`、`.cache/`、`.data/` | 外部 SDK、模型与视频、Framework 编译缓存、中间产物（都不进 git） |
