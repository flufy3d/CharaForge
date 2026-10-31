# tools/fit — 参数拟合引擎（@charaforge/fit）

从参考视频逐帧反推 DeepSeek-chan 的模型参数、图层变换与开关，产出 `scenes/deepseek-chan/scene.json`。
渲染全部经由 harness 页面里的 `@charaforge/runtime`（与播放器、`pnpm render` 同一份代码），优化器（CMA-ES）在 Node 端，
多个无头 Chromium 页面并行评估候选。

## 重跑（第二轮打样）

```bash
# 0. 依赖：pnpm install；pnpm setup（Cubism、模型、视频、Python 环境）。需要 NVIDIA GPU（harness 会检查）。

# 1. 拟合（每个镜头写 .data/fit/<shot>/result.json、fit.log、preview/*.png）。本轮用的命令：
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit --shot s24,s43,s15 --hair
#    --workers N   并行页面数（缺省 4）        --hair      头发按发束分组的 postPhysics 叠加（每组 1 个自由度）
#    --effort K    逐帧优化代数倍率（缺省 1）   --only ID   只重拟合该图层，其余沿用上次的 result.json
#    --until F / --start F / --preview N        调试用；FIT_QUICK=1 跳过锚点的多起点网格

# 2. 合成场景（parseScene 校验通过才写出）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble

# 3. 渲染（播放器同一份渲染代码）
pnpm render --scene scenes/deepseek-chan/scene.json --frames 3668-3721 --out .data/qa/render/s43
pnpm render --scene scenes/deepseek-chan/scene.json --frames 2368-2441 --out .data/qa/render/s24
pnpm render --scene scenes/deepseek-chan/scene.json --frames 1187-1508 --out .data/qa/render/s15

# 4. 指标（docs/pipeline.md 的定义，唯一实现）：SSIM@360（主）与 SSIM@720（辅）、覆盖率
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s43 --shots s43 --out .data/qa/metric_s43.json
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s24 --shots s24 --out .data/qa/metric_s24.json
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s15 --shots s15 --out .data/qa/metric_s15.json

# 5. 对照视频（四宫格：原视频 | 合成 | 差异 | SSIM@360 热力图；烧录帧号与 SSIM；60fps 带原音频）与最差/中位/最好截图
for s in s43 s24 s15; do tools/.venv/bin/python -I tools/qa/metric.py review --renders .data/qa/render/$s --shots $s --out .data/fit/review; done

# 拟合引擎与场景求值路径逐像素一致性抽查（应为 0 字节差异）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/scripts/check_consistency.ts s24 2368,2416,2440
```

## 第三轮续：面捕输入空间的轨迹拟合（`fit4`，当前主线；合成基准接近但未全部达标）

前向模型：面捕输入（每个新画面一个节点）→ `src/vts.ts`（读 `大肥鱼.vtube.json` 的 ParameterSettings：区间线性映射、
指数平滑 α = 1/(1 + c·smoothing)，c 为全局常量）→ 模型参数 → runtime 物理 → 顶点/渲染。图层变换为稀疏关键帧
（每 24 帧，zoom 镜头每 12 帧）。雅可比链：顶点对可见参数（姿态输出、直接参数、159 个次级物理输出）的有限差分
× 映射斜率 × 平滑 × 物理脉冲响应（runtime 物理在镜头静止态附近线性化）。镜头常量（体型、前倾）多假设各自收敛按 SSIM@360 挑，
外观常量（blur、resample、color、呼吸相位/周期）用像素损失。

```bash
# 新合成基准（真值在面捕输入空间；由第二轮结果逆映射投影 + 平滑，变换取稀疏关键帧，c = 0.15）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts synth2 --shot s24,s43
for s in "s24 2368-2441" "s43 3668-3721"; do set -- $s
  pnpm render --scene .data/fit/synth/truth2_$1.json --frames $2 --out .data/fit/synth/truth2_$1
  tools/.venv/bin/python -I tools/qa/synth.py encode --renders .data/fit/synth/truth2_$1 --frames $2 --out .data/fit/synth/$1_synth2.mp4
done
# 拟合（结果 .data/fit/s24_v2/result.json，含输入空间状态 state，可与 truth2_s24_state.json 比较）
FIT_VIDEO=$PWD/.data/fit/synth/s24_synth2.mp4 FIT_VIDEO_OFFSET=2368 FIT_TAG=_v2 \
  node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit4 --shot s24 --itersB 22
#    --smoothC c（缺省 0.15）、--fixed '{"lean":0,"clothes2":0}'（跳过常量搜索）、--refine N（逐节点像素精修，目前有害，缺省关）
FIT_TAG=_v2 node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble --out .data/fit/synth/fit_v2.json
pnpm render --scene .data/fit/synth/fit_v2.json --frames 2368-2441 --out .data/fit/synth/r_v2
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/fit/synth/r_v2 --shots s24 --video .data/fit/synth/s24_synth2.mp4 --video-offset 2368
```

## 第三轮：光流对应 + 整段轨迹 Gauss-Newton（`fit3`，已被 fit4 取代）

```bash
# 合成基准（真值 = 第二轮 s24 / s43 的拟合结果；退化方式同上限实验）
for s in "s24 2368-2441 2368" "s43 3668-3721 3668"; do set -- $s
  cp .data/fit/$1/result.json .data/fit/synth/truth_r2_$1.json
  tools/.venv/bin/python -I tools/qa/synth.py truth-scene --result .data/fit/synth/truth_r2_$1.json --out .data/fit/synth/truth_$1.json
  pnpm render --scene .data/fit/synth/truth_$1.json --frames $2 --out .data/fit/synth/truth_$1
  tools/.venv/bin/python -I tools/qa/synth.py encode --renders .data/fit/synth/truth_$1 --frames $2 --out .data/fit/synth/$1_synth.mp4
done
# 拟合合成参考（结果在 .data/fit/s24_t3/）
FIT_VIDEO=$PWD/.data/fit/synth/s24_synth.mp4 FIT_VIDEO_OFFSET=2368 FIT_TAG=_t3 \
  node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit3 --shot s24 --hair --itersA 4 --itersB 10
FIT_TAG=_t3 node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble --out .data/fit/synth/fit_t3.json
pnpm render --scene .data/fit/synth/fit_t3.json --frames 2368-2441 --out .data/fit/synth/r_t3
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/fit/synth/r_t3 --shots s24 --video .data/fit/synth/s24_synth.mp4 --video-offset 2368
#    选项：--geom N（几何 Worker 数，缺省 CPU 核数−2）、--workers N（渲染页面）、--noconst（跳过镜头常量搜索）
#    诊断：FIT_INIT_JSON='{"c:LeanInput":-27,"by":-25}' 覆盖所有节点的初值
```

组成：`src/traj.ts`（轨迹 GN）、`src/geom/`（worker_threads 里运行 runtime 的 LayerEvaluator + Cubism Core，
求采样顶点与有限差分雅可比；`register.mjs` 只做 @cubism/framework 的模块解析）、`flow.py`（OpenCV DIS 光流，
前后向一致性，多进程）。渲染、损失、复核仍全部经由 harness 里的 runtime。

## 上限实验（已知真值的合成参考）

```bash
mkdir -p .data/fit/synth && cp .data/fit/s24/result.json .data/fit/synth/truth_result_s24.json
tools/.venv/bin/python -I tools/qa/synth.py truth-scene --result .data/fit/synth/truth_result_s24.json --out .data/fit/synth/truth_scene.json
pnpm render --scene .data/fit/synth/truth_scene.json --frames 2368-2441 --out .data/fit/synth/truth
tools/.venv/bin/python -I tools/qa/synth.py encode --renders .data/fit/synth/truth --frames 2368-2441 --out .data/fit/synth/s24_synth.mp4
# 真值渲染 vs 退化后的自己（管线理论上限）
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/fit/synth/truth --shots s24 \
    --video .data/fit/synth/s24_synth.mp4 --video-offset 2368
# 用同一套拟合去拟合合成参考（结果写到 .data/fit/s24_synth/，不覆盖正式结果）
FIT_VIDEO=$PWD/.data/fit/synth/s24_synth.mp4 FIT_VIDEO_OFFSET=2368 FIT_TAG=_synth \
    node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit --shot s24 --hair
FIT_TAG=_synth node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble --out .data/fit/synth/fit_scene.json
pnpm render --scene .data/fit/synth/fit_scene.json --frames 2368-2441 --out .data/fit/synth/fit
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/fit/synth/fit --shots s24 \
    --video .data/fit/synth/s24_synth.mp4 --video-offset 2368
# 诊断：给定图层在某帧的拟合损失（比较真值与拟合结果）
FIT_VIDEO=... FIT_VIDEO_OFFSET=2368 node --experimental-strip-types tools/fit/scripts/eval_layers.ts <result.json> s24 2416
```

`.data/fit/calib.json` 是模型参考姿态下的虹膜中心与「网格质心 → 青色质心」偏移（首次运行自动标定，`calib.png` 为校验图）。

## 结构

| 文件 | 作用 |
|---|---|
| `src/cli.ts` | 命令行；模型眼睛标定 |
| `src/plans.ts` | 每个镜头的图层规划（实例 → 图层、区间、开关、景深/淡入/开窗区间、镜头常量） |
| `src/fitshot.ts` | 拟合流程：帧重复检测 → 开关画面检测 → 逐图层（锚点金字塔多起点 + 分块坐标下降 → 倒序 → 正序，发散重锚）→ 导出 |
| `src/page.ts` | 注入 harness 页面的拟合引擎：按规范 a～g 推进 runtime 的 ModelInstance，用 runtime 的 Compositor 合成，读回，算损失 |
| `src/pool.ts` | 多页面并行（N 个 Chromium 进程，状态广播、候选切分） |
| `src/model.ts` | 拟合参数集合与映射（依据 `大肥鱼.vtube.json`；头/身解耦；姿态输出 postPhysics；呼吸；发束分组） |
| `src/cmaes.ts` | CMA-ES（无依赖，确定性随机数） |
| `src/refvideo.ts` | ffmpeg 解码参考帧（颜色转换与 `tools/qa/metric.py` 一致）；`FIT_VIDEO` 可换合成参考 |
| `src/shotsutil.ts` | shots 插值、排除区、帧重复检测、frameHold 段、参考帧虹膜检测、爱心眼检测 |
| `src/assemble.ts` | 合成 scene.json（全片时长、原音频、署名、镜头标记） |
| `scripts/` | 诊断探针：参数视觉影响（probe_params）、物理阶跃响应（probe_step）、参数范围、损失曲面、损失评估（eval_layers）、一致性检查 |

## 要点

- **物理**：拟合物理输入（AngleX/Y/Z、PositionX/Y/Z、ParamBreath、LeanInput、Clothes2），物理逐帧正向推进；
  Cubism 物理输出滞后，且头部滚转、身体、前倾有 5～15 帧低通（`scripts/probe_step.ts`），所以头、身的主角度输出
  （ParamAngleX/Y/Z、ParamBodyAngleX/Y/Z）用 postPhysics set 直接写；`--hair` 时头发按发束分组叠加（postPhysics add）。
- **损失**（角色遮罩内、扣除排除区）：灰度 SSIM + 颜色 Charbonnier + 虹膜位置（同一检测器作用于两边）+ 皮肤/深色头发
  软类别重叠 + 梯度幅值 NCC（我们的图层单独 vs 参考帧，对背景鲁棒，金字塔粗层权重更大）+ 时间正则。
  汇报用的指标只来自 `tools/qa/metric.py`。
- **帧重复**：在实例头部区域内检测，只拟合新画面，输出 `frameHold`。
