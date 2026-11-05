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

## P3：全片批量（通用规划 + 批量驱动）

所有非打样镜头的规划都从 shots.json 自动推出（`src/plans.ts` 的 `fitUnits` + `planAuto`）；s15、s24、s43 是打样镜头
（`LEGACY_SHOTS`），沿用原规划和 `.data/fit/<shot>/result.json`，批量里不重跑。

- **拟合单元**：`continues` 把同一实例跨镜头连起来的镜头并成一个单元（虚拟镜头：区间、排除区、效果取并），
  同一条 continues 链 = 一个图层（图层 ID = 链根实例 ID），物理不重置。结果写 `.data/fit/<单元首镜头>/result.json`
  （`shots` 字段列出覆盖的镜头，assemble 按它打镜头标记）。continues 指向打样镜头的短尾段（s16.center、s25.prev）不单独起图层。
- **图层**：in = 第一个有 bbox 的采样（翻牌取 cardFlip 起点），out = 最后一个 bbox 之后第一个 bbox=null 的采样；
  variant 为常量，switchEvents（含 fade，from 取切换前累计值）为斜坡；链上后续镜头 variant 与累计状态不符时补阶跃并记问题。
- **clip**：circleWindow 作用的实例 = 圆形开窗（同 s15）；其余 clip = 画中画圆角矩形（整层裁切，关键帧外保持端点）；
  轨迹 opacity 作为不透明度轨。**模糊**：depthOfField 效果，或轨迹 blurSigma ≥ 1.5 的区间（blur = dofK × 强度）。
  **cadence**：shots.json 的帧保持段优先于画面检测。pan / scale 与 zoom 一样加密变换关键帧（每 12 帧）。
- **常量**：Clothes2 取 variant（缺省 0）；前倾只对单元里头最大且头高 ≥ 300px 的实例搜索，其余固定 0；
  颜色滤镜对有 colorGrade（实例或效果）、单实例或 main 实例拟合。
- 没有 eyes 的实例用 head 框估初值（不进锚点先验）；eyes、head 都没有的实例跳过并记入问题清单。

```bash
# 列出全片单元与规划（不拟合）；JSON 里有 skipped 与 shots 问题清单（issues）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts units --out .data/fit/units.json
# 单个单元（参数可以是单元内任一镜头）
FIT_MODEL='{"bodyJac":true}' node --expose-gc --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit4 --shot s17 --workers 3 --geom 5 --itersB 40
# 只合成某些单元的临时场景
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble --only s16 --out /tmp/x.json

# 批量驱动：tools/fit/scripts/batch.sh [all | 镜头ID,...]（逐单元：拟合 → 合成临时场景 → render → metric score → review → 摘要一行）
#   已有 result.json 的单元跳过拟合（未打分的补打分）；失败记 FAIL 与原因后继续；最后 assemble 全片 scenes/deepseek-chan/scene.json
#   环境变量：FIT_MODEL（缺省 {"bodyJac":true}）、FIT_ARGS（追加 fit4 参数）、ITERS_B（缺省按图层帧数和 ≤200→40、≤400→20、否则 10）、
#            REVIEW=0（不出对照视频）、KEEP_RENDERS=1（保留渲染 PNG）
#   产出：.data/fit/batch/summary.tsv、<单元>.fit.log、metric_<单元>.json；对照视频 .data/fit/review/<镜头>.mp4
systemd-run --user --unit=cf-batch --collect --working-directory=$PWD --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  -p MemoryMax=11G -p StandardOutput=append:$PWD/.data/fit/batch/run.log -p StandardError=append:$PWD/.data/fit/batch/run.log \
  bash tools/fit/scripts/batch.sh all
systemctl --user is-active cf-batch     # 中断后原样重跑同一命令即可续跑（单元级跳过 + 图层/阶段断点）
```

### 逐帧精修的时间正则与发散回退（2026-11-06，修闪烁）

- 逐帧精修每轮解完后，所有偏移量（变换 dx/dy/dls/drot 与可见参数）沿节点序列先 3 点中值、再 Whittaker 平滑
  （二阶差分惩罚 λ，`FIT_MODEL='{"pfSmooth":4}'`，0 关闭）。
- 变换偏移上限：位移 0.3 眼距、尺度 ±5%、旋转 3°；超限或锚点偏差比轨迹阶段大 0.5 眼距以上的节点帧回退到轨迹结果并冻结（日志「发散回退」）。
- 眼睛开合网格搜索的结果两眼各做 5 点中值（不允许单帧、两帧来回切换）后再写回。
- 轨迹阶段的重锚加门控：锚点偏差 > 1.5 眼距**且**该帧光流残差 > max(6px, 2×中位) 才重锚；画面已对上的判为锚点可疑（日志「锚点可疑」）。
- `--fixed` 可按图层给：`--fixed '{"s41.main":{"lean":-3.75}}'`。
- 闪烁统计：`tools/.venv/bin/python -I tools/fit/scripts/flicker.py <渲染目录> <起-止>`。

## 真实镜头（第三轮续，当前产出）

```bash
# s43、s24：单图层；镜头常量（体型、前倾）多假设按 SSIM@360 挑
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit4 --shot s43,s24 --itersB 22
# s15：三图层（left → chibi → center 顺序拟合，已拟合的图层参与渲染），翻牌 scaleX、开窗 clip、景深、开关斜坡（plans.ts 的 planS15v3）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit4 --shot s15 --fixed '{"lean":0}' --itersB 10
#    平滑系数：--smoothC c（缺省 0.15；标定见汇报）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble
# 然后按上面「重跑」第 3～5 步 pnpm render、metric.py score / review
```

外观常量：清晰度（blur、resample）由 `tools/fit/sharp.py` 用「角色内部多尺度梯度能量」匹配估计，只用光流残差最低、
观测充分的帧，与几何误差无关；颜色、呼吸相位/周期再在这些帧上用像素损失拟合。图层起点不做 stabilize、预热 90 帧；
镜头首末两个节点和排除区刚结束的帧，数据项权重 ×3。

## 第四轮：逐帧精修 + 诊断工具（C2）

真实镜头与合成基准的差距**来自几何精度，不是外观**（`appear.py`：用光流把我们的渲染扭到参考上，s24 稳定段分区
SSIM@360 0.56 → 0.90，脸 0.76 → 0.94，扫描 blur/resample/颜色都不再提高；参考帧自身平移 2px → 0.83、3px → 0.69）。
轨迹 GN（面捕输入空间 + 物理）把整体姿态收敛到几个像素，但鲁棒核把耳朵、后发、肩这些大误差区当离群点，
剩余位移中位 3～14px。所以 fit4 在最后加了**逐帧精修**（缺省开启，`FIT_MODEL='{"perframe":0}'` 关闭）：
每个新画面帧独立放开全部可见参数（姿态输出、次级物理输出逐个，postPhysics add 偏移）与变换偏移，
光流 GN 12 次（Cauchy 尺度 20px 退火到 6px，弱先验拉回 0），节点之间线性插值后导出。

```bash
# 前向模型 / 求解器变体（诊断用；见 src/modelx.ts）：decouple、bodyJac、keyEvery、consts、clamp、anneal、reg、perframe、upper
FIT_MODEL='{"bodyJac":true}' tools/fit/scripts/exp.sh <tag> s24 --fixed '{"lean":-26.25,"clothes2":0}' --itersB 40
#   = 拟合（FIT_TAG=_<tag>）→ 合成临时场景 → pnpm render → metric score → 分区误差，摘要在 .data/fit/diag/exp_<tag>_<shot>.txt
# 分区误差（按 drawable 所属部件：脸、前发、侧发、后发、耳、头饰、领子、身体）
node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types tools/fit/scripts/meshes.ts <scene.json> s24.main 2396,2399 mesh.json
tools/.venv/bin/python -I tools/fit/regions.py mesh.json --renders <png 目录> --shot s24
# 外观上限 / 几何上限（光流扭正 + blur/resample/颜色扫描；--control N 为对照）
tools/.venv/bin/python -I tools/fit/appear.py mesh.json --renders <png 目录> --shot s24
# 单帧参数变体对比、单帧自由 CMA、求值后的参数
node --experimental-strip-types tools/fit/scripts/variants.ts <scene.json> s24.main s24 2420 '[{"Clothes2":0.2,"align":1}]' out.jpg
node --experimental-strip-types tools/fit/scripts/freefit.ts <scene.json> s24.main s24 2400,2420 tf,pose,body,hair,ears
node --import ./tools/fit/src/geom/register.mjs --experimental-strip-types tools/fit/scripts/params.ts <scene.json> s24.main 2420
```

### 锚点先验与内存（第五轮）

- **锚点先验**（所有镜头）：shots 里 confidence ≥ 0.4 的 eyes 锚点（落在节点帧 ±1）进 GN 数据项——两只眼的虹膜主网格
  （ArtMesh236/134，与页面 irisScreen 同一组）顶点质心对锚点，Cauchy 尺度 max(10px, 0.15 × 眼距)，权重约 80 个光流点。
  每次 GN 前做发散检查：偏差 > 1.5 眼距的锚点帧，把最近的变换关键帧重锚到 shots 眼睛（日志「发散检查」）。
- **reg 已对 s15 关闭**（c3 上把左实例推偏约 150px）；`FIT_MODEL='{"reg":true}'` 仍可强制。
- **内存**：每个几何 Worker 约 600MB（一份 Core + 模型），雅可比缓存 = 节点数 × 2 × 采样点 × 可见参数 × 4 字节
  （s15 左实例 280 节点 × 4000 点时约 1.6GB，逐帧精修再算一份）。现在：采样点按缓存 ≤ 1GB 自动下调、逐帧精修前释放轨迹阶段缓存；
  fit4 缺省 `--workers 3 --geom 5`（推荐；目标峰值 ≤ 7GB）。用 systemd-run 跑长任务并设 `-p MemoryMax=11G`。

- **断点续跑**（缺省开启，`FIT_RESUME=0` 关闭）：图层内各阶段完成后写 `.data/fit/<shot><tag>/work/ckpt_<图层>.json`
  （traj = 轨迹 + 外观 + 开窗 → perframe → eyes），整个图层完成后写 `.data/fit/<shot><tag>/done_<图层>.json`；
  指纹（FIT_MODEL、拟合参数、shots.json 内容）一致才复用。中断后原样重跑同一命令即可接着跑。
- **页面内存**：眼睛开合网格搜索每 10 帧对各页面做一次 CDP 强制 GC（读回缓冲的外部内存回收滞后，曾把 chrome-headless 涨到 OOM），
  每 5 帧查一次实时 cgroup 内存（memory.current − inactive_file）。

### 粗配准（reg）与速度

- `reg`：变换初值的多尺度模板匹配（`tools/fit/reg.py`；只渲染本图层，以渲染出的两眼中点为缩放/旋转中心，
  梯度幅值的掩码 NCC）。图层规划 `reg: true` 时开启，目前只有 s15（shots 眼睛轨迹为锚点插值，尺度差约 2 倍）。
  s24 上有害（0.687 → 0.524：快速运动 + 白色划过特效的前三个关键帧给出假峰，尺度 ×0.5），s43 小幅有益（27 → 28/30）。
- 逐帧精修 16 次（缺省 12）：耳朵分区 0.54 → 0.56～0.57，逐帧精修耗时 +50%，不改缺省。
- 两个镜头并行（各 `--geom 5 --workers 2`）总吞吐只提高约 14%（CPU 瓶颈：有限差分雅可比 + 光流）。
- 生成 shots 修正建议：`node --experimental-strip-types tools/fit/scripts/eyes.ts scenes/deepseek-chan/scene.json s15 20`

## 第三轮续：面捕输入空间的轨迹拟合（`fit4`）

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
