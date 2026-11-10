# CharaForge

**AI-Powered Live2D Character Studio**

面向虚拟 AI 角色的 Live2D 可视化编辑与动画创作工具。

### 核心功能（部分已实现）

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

## 第一期：PV 复刻（已完成）

第一期已完成；全片结果、已知问题和 Windows 原生播放方式见下文。

用 DeepSeek-chan 模型、我们自己的运行时（`packages/runtime`）和场景格式（`charaforge.scene` v1），复刻参考 PV
（`assets/video/DeepSeek-chan.mp4`，1280×720、60fps、3868 帧）里的角色动画：每个镜头的缩放、裁切、平移，同屏多实例、
透明度、景深模糊，以及表情、换装、配饰开关。网页播放器逐帧确定地渲染同样的画面，并以原音频为时钟同步播放。
背景、手指、气泡、粒子、转场剪影等不在范围内；验收口径见 [docs/pipeline.md](docs/pipeline.md) 最后一节。

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
| [Mediabunny](https://mediabunny.dev/)（视频导出） | [MPL-2.0](https://www.mozilla.org/en-US/MPL/2.0/) | npm 依赖，未修改源码；只用于浏览器内 WebCodecs 封装 |

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
（授权风险详见 [docs/gpu-acceleration.md](docs/gpu-acceleration.md) 的「风险」一节）。

## 环境要求

| 项目 | 要求 |
|---|---|
| 系统 | 播放器：Windows 10/11 原生（Chrome / Edge）或 Linux；拟合、无头渲染、QA：Linux 或 Windows 上的 WSL2（Ubuntu 24.04 实测） |
| GPU | 播放器本身不检查；测试、无头渲染与拟合会检查 WebGL renderer 是否为 NVIDIA，防止悄悄落到软件渲染 |
| Node.js / pnpm | Node ≥ 22（用 `--experimental-strip-types` 直接跑 TS 脚本），pnpm 10 |
| Python | 3.12，由 [uv](https://docs.astral.sh/uv/) 管理（`tools/.venv`） |
| ffmpeg | 在 `PATH` 里（setup 解码音频、recon、qa 用）；WSL 机器上是 `~/.local/bin` 下的静态版 |
| CUDA | 只有 GPU 拟合（`tools/gpu`）需要：CUDA 13.0 工具链（`CUDA_HOME=/usr/local/cuda-13.0`），torch 2.14.1+cu130，nvdiffrast 0.4.0 |
| 浏览器 | 播放器用 Chrome / Edge（需要 WebGL2、Web Audio、H.264；Firefox 未验证） |

**Windows / WSL2 说明**：播放器是浏览器里的 WebGL2 页面，Windows 原生运行 `pnpm dev`（或分两行 `pnpm build`、`pnpm preview`）后用 Chrome / Edge 打开即可，不需要 WSL；GPU 走 Windows 的 ANGLE / D3D，不经过 WSL。2026-11-09 已实测 `pnpm build` + `pnpm preview` 能播放全片。

WSL 里的特殊配置只影响无头 Chromium（`pnpm render`、确定性测试、拟合）：WSL 里没有 NVIDIA 的 Vulkan / EGL 驱动，无头 Chromium 只能经 Mesa 的 d3d12 驱动走 Windows 的 D3D12。
`apps/player/scripts/harness-node.ts` 检测到 WSL（有 `/usr/lib/wsl/lib/libd3d12.so`、没有 NVIDIA Vulkan ICD）时自动使用
`--use-angle=gl-egl` + `GALLIUM_DRIVER=d3d12` + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`，并把 MSAA 设为 0（此后端开 MSAA 不确定）。
renderer 应为 `ANGLE (Microsoft Corporation, D3D12 (NVIDIA GeForce …), OpenGL ES 3.1)`。显式设置的 `CHARAFORGE_GPU` /
`CHARAFORGE_MSAA` 等环境变量优先。细节见 [docs/runtime-eval.md](docs/runtime-eval.md) 第 4 节。
在 WSL 里运行 `pnpm dev` 后，用 Windows 上的 Chrome / Edge 打开 `http://localhost:5173` 也可以（WSL2 自动转发 localhost，页面仍用 Windows 原生 GPU）。

## 安装

Windows 原生播放不依赖 WSL；先装 Node ≥ 22 和 pnpm。没有 pnpm 时用 Corepack 启用（版本由 `packageManager` 固定为 10.18.0）：

```powershell
corepack enable
corepack install --global pnpm@10.18.0
```

或 `npm install -g pnpm@10.18.0`。装完重开终端，在仓库根目录运行：

```bash
pnpm install
pnpm run setup        # = setup:cubism + setup:assets + setup:py
```

- `pnpm run setup` 不能写成 `pnpm setup`（后者是 pnpm 自带命令）。
- `setup:cubism`：下载 Cubism SDK for Web 5-r.5（校验 sha256），装 Core 与 Framework 到 `vendor/cubism/`，运行前会打印两份 Live2D 许可的链接。
- `setup:assets`：下载模型（去掉 131MB 的分层 PSD，保留原文件名和使用条款）和参考视频，校验 sha256，再用 ffmpeg 把音频解码成 `assets/video/DeepSeek-chan.wav`（PCM，避免 AAC 编码延迟造成音画偏移）。
- `setup:py`：`uv sync --project tools`，建立 `tools/.venv`（numpy、opencv、scikit-image、jsonschema、pillow）。
- Windows 原生安装同样需要 `ffmpeg.exe` 在 `PATH` 里（`setup:assets` 用它把参考音频解码成 PCM）。
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

> Windows 原生播放已在 2026-11-09 验证；Windows 下的 `pnpm test:runtime`、`pnpm bench` 尚未跑，数据待补。

## 独立播放器（`/player.html`）

`pnpm dev` 的根路径默认打开编辑器；独立播放器保留在 `/player.html`，用于只读预览、参考视频对照和性能测试。

```bash
pnpm dev               # 编辑器 http://localhost:5173/
                       # 独立播放器 http://localhost:5173/player.html
```

Windows 原生同样适用；PowerShell 5.1 不支持 `&&`，`pnpm build` 和 `pnpm preview` 分两行执行即可。

播放器缺省打开全片场景 `scenes/deepseek-chan/scene.json`（约 4.3MB，47 个图层），点「播放」或按空格开始，
原音频为时钟：画面帧号 = floor(音频时间 × 60)。

- 加载约 6 秒出第一帧（页面显示加载阶段），之后**可以立即播放**。加载后 Worker 会提前为各图层做物理预热
  （约 4 秒，不阻塞播放、不改变画面，控制栏显示「预热图层 k/37」），播放中也会提前预热前方 3 秒内出现的图层。
- 实测（WSL2、RTX 5080 Laptop、8 逻辑核、3 个 Worker）：加载完立即播放全片 60 秒 58.5～59.6 fps、丢帧 0.7～2.5%
  （另一拟合任务把 GPU 占到 84～92% 时 3.6～3.8%），画面相对声音滞后均值约 0.8 帧，60 秒内无漂移。
  数据与方法见 [docs/runtime-eval.md](docs/runtime-eval.md) 第 6 节；Windows 原生播放尚未做正式 bench。

独立播放器 URL 参数（`/player.html`；场景/动作编辑器也支持 `scene=`、`frame=`、`workers=`）：

| 参数 | 含义 |
|---|---|
| `scene=<仓库相对路径>` | 场景文件，缺省 `scenes/deepseek-chan/scene.json`；`demo:N` 为内置的 N 实例演示场景 |
| `video=<仓库相对路径>` | 参考视频，缺省 `assets/video/DeepSeek-chan.mp4` |
| `debug=1` | 打开调试面板 |
| `compare=side\|overlay\|diff` | 初始对照模式（配合 `debug=1`） |
| `frame=<n>` | 初始帧 |
| `workers=<n>` | 求值 Worker 数，缺省 clamp(逻辑核数/2 − 1, 2, 6)；0 = 主线程同步求值 |
| `prewarm=0` | 关闭新图层的 Worker 预热（只用于诊断；关掉后全片丢帧约 10%） |
| `edit=1` | 打开饰品编辑器：导入/调整/绑定/保存 ImageLayer，见 [docs/accessories.md](docs/accessories.md) |

快捷键：`空格` 播放/暂停，`←`/`→` ±1 帧（`Shift` ±10），`Home`/`End`，`[`/`]` 上/下一个镜头，`D` 调试面板，
调试面板打开时 `1`～`4` 切换对照模式（关 / 并排 / 叠加 / 差异图），`M` 静音。

### 动作 / 表情 / 体型素材查看器

```bash
pnpm dev
# Chrome / Edge 打开：
# http://localhost:5173/motions.html
```

- 独立页面，列出当前模型 `Animation/` 下的 8 个 motion3 动作和 `Exp3/` 下的 23 个 exp3；
  其中 3 个自动识别为「体型」，20 个作为普通表情。点击即可在角色上预览。
- 默认看示例模型；用 `?model=models/<路径>/<名称>.model3.json` 可以查看 `assets/models/` 下的其它模型（例如溟月），
  `?modelName=` 可覆盖顶部显示名。
- **体型是独立槽位**：体型 + 表情 + 动作可以同时生效，切换体型不会顶掉表情/动作；
  适用于“同一套 rig 的三种体型（exp3 参数预设）”。自动按名称/参数识别体型，也可以用模型目录下的
  `charaforge.bodies.json` 显式声明和指定默认体型。
- 预览按角色实际内容的包围盒自动取景，小体型切换后也会占满画面；支持滚轮缩放、拖动平移、双击/`F`/`0` 适应。
- 动作显示名称、时长、loop、曲线数等元数据；支持搜索、播放/暂停、逐帧、seek、循环开关（快捷键与播放器一致，另加 `L` 循环）。
- 列表带 160×160 缩略图；顶部「图集」打开全部素材的 contact sheet。缩略图由一次网格 scene 渲染生成，
  再按每个素材的内容包围盒收紧，缓存在 localStorage。
- 预览使用临时 scene（720×720、60fps），不写盘、不加载原参考视频/音频，也不影响 `?scene=...` 全片播放器和 `?edit=1` 饰品编辑器。
- 列表来源：`model3.json` 的 `FileReferences.Motions / Expressions` 优先；没有声明时由只读接口
  `/__charaforge/list-assets` 扫描 `assets/models/**/Animation` 与 `Exp3`。接口契约、安全限制和临时 scene 细节见
  [docs/motions.md](docs/motions.md)。

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
pnpm accessory:import --input .data/hat.png --id hat --scene packages/scene/examples/accessory.scene.json --attach main
                                             # 导入 PNG/WebP 饰品 → assets/accessories/，stdout 输出 ImageAsset + ImageLayer
pnpm --silent decoration:command --scene packages/scene/examples/accessory.scene.json --input .data/command.json --out .data/command.scene.json
                                             # AI 命令层：add/update/remove/setVisible/preview，返回 diff
pnpm bench demo:3 20                         # 帧率基准（参数：场景 秒数 [workers] [起始帧]），同时报告音画滞后
pnpm bench scenes/deepseek-chan/scene.json 60    # 全片（带原音频）
pnpm verify:video                            # 参考视频逐帧定位精度（与 ffmpeg 解码比对）
pnpm test:viewer                             # 动作 / 表情 / 体型查看器：列表接口、临时 scene、缩放/适应、Playwright smoke
pnpm test:editor                             # 编辑器 Playwright smoke（加载/时间线/AI 命令/多轨/保存/导出）
pnpm test:editor-api                         # 本地编辑 HTTP 接口（上传/保存/命令/安全限制）
pnpm editor:command --scene <scene.json> --input <commands.json> [--out <scene.json>]  # AI 编辑命令
pnpm export:video --scene <scene.json> [--from 0 --to 3868] [--format mp4|webm|auto] --out <file>
                                             # 逐帧导出 MV（带音频；不依赖 ffmpeg）
```

## 场景 / 动作编辑器与 MV 导出

```bash
pnpm dev
# 根路径默认就是编辑器：
# http://localhost:5173/
# 直接编辑全片 MV：
# http://localhost:5173/?scene=scenes/deepseek-chan/scene.mv.json&frame=2400
# 轻量示例（背景光晕 + 星星元素 + 音乐）：
# http://localhost:5173/?scene=packages/scene/examples/editor-demo.scene.json
```

`/editor.html` 仍保留为直接入口；独立播放器在 `/player.html`（只用于预览/参考视频对照）。
编辑器顶部右侧有「独立播放器 ↗」和「素材 / 动作查看器 ↗」两个入口；播放器会带当前场景和帧号，未保存的修改需要先保存。

> `scenes/deepseek-chan/scene.json` 是原始拟合场景（只有角色动画，透明背景）；
> `scenes/deepseek-chan/scene.mv.json` 在它基础上叠加了近似背景、点阵边框、光环、照片卡、气泡、闪光和转场泼溅，
> 编辑器默认打开后者。原场景保留，可随时对比或重新生成。

- **场景编辑**：图层树、变换、透明度、裁切、滤镜、混合模式、相机、背景色、导入图片/音频、生成形状/渐变背景。
- **动作编辑**：motion3 / exp3 片段、参数轨、部件不透明度、物理开关；动作库列出模型的 8 个动作、20 个表情和 3 个体型；
  「设为体型」会替换当前图层上的旧体型（从入点持续到出点），不影响表情/动作片段。
- **多轨时间线**：镜头标记、相机、全部图层、动作/表情片段、任意多条音频轨（音乐轨）；片段可拖动/裁剪，关键帧可拖动/多选/批量缓动。
  时间线支持 Ctrl+滚轮围绕鼠标缩放、缩放滑块 / ± / 适应，中键拖动或“平移”模式左键拖动平移。
- **关键帧与插值**：linear / smooth / easeIn / easeOut / easeInOut / step / 自定义三次贝塞尔；一键动画（淡入、漂浮、脉冲、旋转、弹跳、闪烁、滑入、抖动、放大）。
- **导出 MV（带音乐）**：编辑器按钮或 CLI，逐帧渲染 + 离线混音，输出 MP4（AVC/AAC）或 WebM（VP9/Opus）。
- **AI 可调用**：浏览器 `window.__editor`、本地 HTTP `/__charaforge/editor/command`、CLI `pnpm editor:command` 共用 `packages/scene/src/edit.ts` 命令层。

```bash
# AI / 脚本编辑：命令 → 新场景（stdout 返回 diff / effects / preview）
pnpm editor:command --scene scenes/deepseek-chan/scene.mv.json \
  --input .data/edit-commands.json --out .data/edited.scene.json

# 全片导出（60 秒，1280×720，带原音频）
pnpm export:video --scene scenes/deepseek-chan/scene.mv.json --out .data/export/deepseek-full.mp4 --workers 2

# 只导出 s24 镜头、WebM
pnpm export:video --scene scenes/deepseek-chan/scene.mv.json --from 2368 --to 2442 \
  --format webm --out .data/export/s24.webm
```

- 编辑器保存到 `scenes/` 下的场景 JSON；覆盖已有文件时自动写 `.bak.json`。大场景保存为紧凑 JSON。
- 全片场景、镜头、音频、时间线加载后可直接预览；编辑器的预览与播放器/harness 同一帧逐位一致。
- 视频导出是有损编码，像素不与 runtime PNG 逐位相同；帧来源与播放器是同一份 `CharaForgeRuntime`。
- 2026-11-10 实测：`scene.mv.json` 全片 3868 帧（128 图层，含背景/小元素）导出为 1280×720 MP4（AVC+AAC），
  时长 64.467 s，47,324,791 字节，耗时 63.7 s（RTX 5080 Laptop，workers=2）；原始 `scene.json` 为 37.9 MB / 59.2 s。
- 完整界面说明、命令格式、HTTP 接口和已知限制见 [docs/editor.md](docs/editor.md)。

## 外部饰品（ImageAsset / ImageLayer）

外部 AI 生成或手工准备的透明 PNG/WebP 可以作为 scene 资产挂到角色上，不修改模型文件。完整字段、attach 公式和限制见 [docs/accessories.md](docs/accessories.md)。

```bash
pnpm accessory:import --input .data/hat.png --id hat --scene packages/scene/examples/accessory.scene.json --attach main
pnpm dev    # ?scene=packages/scene/examples/accessory.scene.json&edit=1
pnpm --silent decoration:command --scene packages/scene/examples/accessory.scene.json --input .data/command.json --out .data/command.scene.json
```

- 图片：PNG/WebP，单边 ≤ 4096、总像素 ≤ 16 MP；导入 CLI 要求至少一个透明/半透明像素。
- schema：`scene.assets.images` + `kind:"image"` 图层，`version` 仍为 1；旧场景不写 `assets.images` 时行为不变。
- attach 只跟随目标模型图层的整体变换，不跟骨骼/参数；已知限制和授权注意见文档。
- 图片文件放 `assets/accessories/`（gitignore，不进仓库）；AI 生成图请自行确认授权，不要把大图/版权不明素材提交进 git。

## 静态页（`pnpm build`）

```bash
pnpm build             # → apps/player/dist/（约 2.2MB，只有页面和打包后的代码）
pnpm preview           # → http://localhost:4173/（默认编辑器）；播放器 /player.html
```

- 页面：`index.html`（默认场景 / 动作编辑器）、`editor.html`（编辑器直接入口，内容同 index）、
  `player.html`（独立播放器 / 参考视频对照）、`motions.html`（动作 / 表情 / 体型查看器）、
  `export.html`（逐帧视频导出页，`scripts/export-video.ts` 使用）、`harness.html`（Playwright 测试接口）。
- 产物**不含**模型、参考视频、音频、场景文件和 Cubism Core，也不要把它们复制进去。页面按 URL 引用仓库里的原文件：
  `assets/…`（模型、视频、音频）、`vendor/cubism/core/…`（Core）、`scenes/…`、`packages/scene/examples/…`、`packages/scene/schema/…`。
- 所有这些请求都是**相对页面地址**的（`apps/player/src/util.ts` 的 `pageUrl`，基于 `document.baseURI`；`vite.config.ts` 的 `base: './'`
  让打包产物、Worker 脚本也用相对路径），所以页面可以挂在反向代理的任意路径前缀下，例如
  `https://<机器>.ts.net/api/local-proxy/p/5173/`（Tailscale 上的本地代理）。前缀地址要以 `/` 结尾。
  `pnpm build` 的产物在只剥前缀、不改写内容的代理后面也能用；`pnpm dev` 依赖代理把 Vite 的模块路径（`/@fs/…`、`/node_modules/…`）加上前缀
  （用户用的本地代理会做，模块能正常加载），热更新的 websocket 仍连根路径，走代理时可能不可用，刷新页面即可。
- 请求路径只含 ASCII：模型文件名是中文（`大肥鱼.*`），有的代理会把路径解码成 UTF-8 再转发而出错（500）。
  播放器与运行时（主线程和 Worker）发请求前，把含非 ASCII 字符的路径段换成 `_f.<base64url(UTF-8)>`（`packages/runtime/src/asciiUrl.ts`），
  白名单中间件把它解码回原名后照常做白名单与防穿越检查（支持 Range）。纯 ASCII 的路径保持原样。
- 提供这些 URL 的是 `apps/player/vite.config.ts` 里的白名单中间件 `repoFiles()`：只读，只开放上面几个目录，
  拒绝 `..` 穿越，支持 Range 请求（视频精确 seek 需要）。另外提供只读素材目录接口
  `/__charaforge/list-assets?dir=assets/models/.../Animation|Exp3`（见 [docs/motions.md](docs/motions.md)）。
  `pnpm dev` 和 `pnpm preview`（`vite preview`）用的是同一个中间件，所以 `pnpm preview` 就能完整播放全片，
  调试模式、原视频对照、动作 / 表情 / 体型查看器也都可用。
- 打包代码放在 `dist/_app/`（不用 Vite 缺省的 `assets/`，否则会和白名单的 `/assets/` 抢同一个 URL 前缀）。
- 不能用 `file://` 直接打开 `dist/index.html`（Worker、fetch、Range 请求都需要 HTTP），也不要换成别的静态服务器：
  那样要么拿不到 `/assets/`，要么就得把模型复制进可发布的目录。
- `vite preview` 缺省只监听 localhost。**不要**加 `--host`，也不要部署到公网（模型禁止再分发）。

## 复现整条管线

以下整条管线（recon / fit / render / qa）的命令都在仓库根目录、WSL / Linux 下运行；播放器本身不受此限制，Windows 原生也可以跑（见上文）。
视频和模型是不可信数据：Python 一律 `tools/.venv/bin/python -I`，不要在 `assets/` 里执行任何东西。
中间产物都在 `.data/`（不进 git）。长任务建议用 `systemd-run --user … -p MemoryMax=…` 启动；
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
pnpm build
pnpm preview      # 编辑器 http://localhost:4173/；播放器 http://localhost:4173/player.html
# 或开发模式 pnpm dev（同样：/ 编辑器，/player.html 播放器）
```

## 验收口径与当前结果

口径见 [docs/pipeline.md](docs/pipeline.md)「验收指标的定义」：SSIM@360（渲染按 alpha 合成到参考帧上，与参考帧一起降到 640×360，
只在角色遮罩内、扣掉排除区计算）。帧分三类，判定优先级 特效帧 > 目测帧 > 普通帧：

- **普通帧**：逐帧 SSIM@360 ≥ 0.6；
- **目测帧**（同屏 ≥ 2 个实例，或最大实例头高 < 200px）、**特效帧**（半调、径向模糊、转场、粒子、重度模糊等）：照常统计，以并排、叠加对比时肉眼一致为准。

**P4 全片结果**（GPU 引擎冷启动重拟合，28 个单元、47 个图层；SSIM 精修每单元 72 帧；runtime 渲染 + `metric.py`，门槛 0.6；报告在拟合机器的 `.data/qa/p4c/`）：

| 帧类别 | 有效帧 | SSIM@360 均值 | ≥ 0.6 |
|---|---|---|---|
| 普通帧 | 916 | **0.760** | **863 / 916**（旧引擎 543 / 901） |
| 目测帧 | 1696 | 0.749 | 1489 |
| 特效帧 | 750 | 0.703 | 590 |

闪烁尖峰 **23** 帧（旧引擎 143），其中多数在转场或粒子特效里，属于误报；真正的跳变只剩 s13（1117、1119、1131）、s09 783、s21 1912。普通帧不达标的 53 帧主要在 s13（37 帧）。

**已知问题**：

- s13 普通帧仍有 37 帧低于 0.6；s09 783、s21 1912 各 1 个闪烁跳变。其余闪烁尖峰多在转场或粒子特效里，属于误报。
- s13 的窗口展开、s26 的笔刷和 ✓ 排除区、s12 仍有肉眼可见差异，尚未逐帧修完。
- P4 渲染曾出现 15 帧「定格」（964～1079）和若干帧与干净渲染不同；同一命令重跑 3 次未复现，怀疑与 GPU 被拟合任务占满有关。质检前应让 GPU 空闲，或渲染后用「相邻两帧逐字节相同、但有图层的保持帧号不同」筛一遍。
- 4 个实例同屏在 8 逻辑核机器上约 23～32 fps；3 个实例约 58～59 fps（WSL 实测，见 [docs/runtime-eval.md](docs/runtime-eval.md) 第 6 节）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `apps/player/` | 编辑器 + 独立播放器（React + Pixi）：`index.html` 默认编辑器、`player.html` 独立播放器、`src/editor/` 场景/动作编辑器（store、时间线、检查器、导出）、`src/main.tsx` 播放器 UI、`src/editor.tsx` 旧饰品编辑器、`engine.ts` 渲染循环、`clock.ts` 音频时钟、`videoSync.ts` 参考视频逐帧对齐；`harness.html` 测试接口；`server/editorApi.ts` 本地编辑 HTTP API；`scripts/` 下是 render、export-video、editor-command、bench、verify-video、import-accessory、decoration-command 等 CLI |
| `packages/runtime/` | 运行时：Cubism 封装、确定性物理、Worker 并行求值、合成器（滤镜、clip、frameHold）、场景热更新（`syncScene` / `rebuildEvaluators` / `reloadWorkers`）；`test/` 确定性与编辑热更新测试 |
| `packages/scene/` | 场景格式 `charaforge.scene` v1、镜头表 `charaforge.shots` v1（TypeBox 定义、JSON Schema、`parseScene` / `parseShots`）、编辑命令层 `src/edit.ts`（关键帧、插值、一键动画、diff/撤销） |
| `scenes/deepseek-chan/` | `shots.json`（recon 产出、人工校对）、`scene.json`（全片复刻场景，fit 产出） |
| `tools/recon/` | 参考视频镜头分析（Python） |
| `tools/gpu/` | GPU 拟合引擎：`l2d/`（moc3 解析、变形器、物理、nvdiffrast 光栅化，内部工具）、`fit/`（整段轨迹优化、导出）、`scripts/`（`a5_fit.py`、`a5_batch.py`、`a5_assemble.ts` 等）、`docs/`、`tests/` |
| `tools/fit/` | 旧 CPU 拟合引擎 fit4（保留；单元规划与导出代码仍被 GPU 引擎复用） |
| `tools/qa/` | 验收指标 `metric.py`、闪烁 `flicker.py`、全片报告 `report.py` |
| `scripts/` | `setup-cubism.mjs`、`setup-assets.mjs`（下载并校验外部资源） |
| `docs/` | `editor.md`（场景/动作编辑器、时间线、MV 导出、AI 命令）、`pipeline.md`（目录、契约、验收口径）、`accessories.md`（外部饰品图片/attach）、`imagine-agy.md`（AI 生图工具调研与调用约定）、`runtime-eval.md`（运行时实现、GPU 与确定性、性能）、`gpu-acceleration.md`（历史提速方案与风险记录） |
| `vendor/`、`assets/`、`.cache/`、`.data/` | 外部 SDK、模型与视频、Framework 编译缓存、中间产物（都不进 git） |
