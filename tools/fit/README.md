# tools/fit — 参数拟合引擎（@charaforge/fit）

从参考视频逐帧反推 DeepSeek-chan 的模型参数、图层变换与开关，产出 `scenes/deepseek-chan/scene.json`。
渲染全部经由 harness 页面里的 `@charaforge/runtime`（与播放器、`pnpm render` 同一份代码），优化器（CMA-ES）在 Node 端。

## 重跑

```bash
# 0. 依赖：pnpm install；pnpm setup（Cubism、模型、视频、Python 环境）。需要 NVIDIA GPU（harness 会检查）。

# 1. 拟合（每个镜头写 .data/fit/<shot>/result.json、fit.log、preview/*.png）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts fit --shot s43,s24,s15
#    调试选项：--until F（只拟合到 F 之前）、--start F（首个拟合帧不早于 F）、--preview N（每 N 个拟合帧存一张对照图）
#    FIT_QUICK=1 跳过首帧的多起点网格；--only s15.center 只重拟合该图层，其余图层沿用上次的 result.json
#    （本轮 s15 就是先整段拟合，再用 --only s15.center 重拟合了中间实例）

# 2. 合成场景（parseScene 校验通过才写出）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts assemble

# 3. 用播放器同一份渲染代码渲染各镜头的帧
pnpm render --scene scenes/deepseek-chan/scene.json --frames 3668-3721 --out .data/qa/render/s43
pnpm render --scene scenes/deepseek-chan/scene.json --frames 2368-2441 --out .data/qa/render/s24
pnpm render --scene scenes/deepseek-chan/scene.json --frames 1187-1508 --out .data/qa/render/s15
#    注意：当前 runtime 有清屏色 bug（见 src/render.ts 文件头），frameHold 的保持帧、带 color/blur 滤镜的图层
#    在透明处会被画成不透明白色，pnpm render 的结果因此无效。runtime 修复前用下面的等价入口
#    （同一 harness、同一 renderFrame，只在每次 pass 前把 GL 清屏色设回 0；非受影响帧与 pnpm render 逐字节相同）：
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts render --frames 3668-3721 --out .data/qa/render_patched/s43
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts render --frames 2368-2441 --out .data/qa/render_patched/s24
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/src/cli.ts render --frames 1187-1508 --out .data/qa/render_patched/s15

# 拟合引擎与场景求值路径逐像素一致性检查（本轮 s24 抽查 3 帧，0 字节差异）
node --experimental-strip-types --disable-warning=ExperimentalWarning tools/fit/scripts/check_consistency.ts s24 2368,2416,2440

# 4. 验收指标（docs/pipeline.md 的定义，唯一实现；--renders 指向上一步的目录）
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s43 --shots s43 --out .data/qa/metric_s43.json
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s24 --shots s24 --out .data/qa/metric_s24.json
tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/qa/render/s15 --shots s15 --out .data/qa/metric_s15.json

# 5. 对照视频（四宫格 + 帧号/SSIM + 原音频，60fps）与关键帧截图（最差/中位/最好）
tools/.venv/bin/python -I tools/qa/metric.py review --renders .data/qa/render/s43 --shots s43 --out .data/fit/review
```

`.data/fit/calib.json` 是模型参考姿态下的虹膜中心（首次运行时自动标定，`calib.png` 为校验图），删掉会重新标定。

## 结构

| 文件 | 作用 |
|---|---|
| `src/cli.ts` | 命令行；模型眼睛标定 |
| `src/plans.ts` | 每个镜头的图层规划（由 shots.json 推导：实例 → 图层、区间、开关常量/切换、模糊段） |
| `src/fitshot.ts` | 逐镜头拟合流程：帧重复检测 → 首帧全局搜索 → 逐帧粗/细两级 CMA-ES → 开关斜坡 → 导出图层 |
| `src/page.ts` | 注入 harness 页面的拟合引擎：按规范 a～g 逐帧推进 runtime 的 ModelInstance、用 runtime 的 Compositor 合成、读回、损失 |
| `src/model.ts` | 拟合参数集合（依据 `大肥鱼.vtube.json` 的面捕映射）与状态 → 模型参数的映射 |
| `src/cmaes.ts` | CMA-ES（无依赖，确定性随机数） |
| `src/refvideo.ts` | ffmpeg 解码参考帧（颜色转换参数与 `tools/qa/metric.py` 一致），缓存在 `.data/fit/ref/` |
| `src/shotsutil.ts` | shots 插值、排除区、帧重复检测、frameHold 段生成 |
| `src/assemble.ts` | 合成 scene.json（全片时长、原音频、署名、镜头标记） |
| `scripts/` | 调试探针（损失曲面扫描、参数组合拼图等），不参与产出 |

## 要点

- **物理**：拟合的是物理输入（控制器参数 AngleX/Y/Z、PositionX/Y/Z），物理逐帧正向推进。Cubism 物理输出比输入滞后一帧，
  所以帧 f 的物理输入写在 f−1 和 f，直接参数（眼、嘴、眉）写在 f。每个候选都从缓存的规范状态出发重新推进，
  首帧阶段从 stabilize + 预热开始，和 `pnpm render` 的求值路径一致。
- **损失**（只在角色遮罩内，扣除排除区）：灰度 SSIM（盒式窗口）+ 颜色 Charbonnier + 虹膜位置项（同一检测器作用于我们的
  合成图与参考帧）+ 皮肤/深色头发的软类别重叠项（对背景不敏感，并惩罚漏画）+ 时间正则。汇报用的指标只来自 `tools/qa/metric.py`。
- **帧重复**：在实例头部区域内检测（角色 30fps、背景 60fps 的段落也能识别），只拟合新画面，输出 `frameHold`。
