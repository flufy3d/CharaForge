# tools/qa：验收指标与质检报告

指标定义与验收口径见 `docs/pipeline.md` 最后一节。`metric.py` 是指标唯一实现（fit 也调用它），
`report.py` 只做汇总和呈现，不改任何计算定义。

视频、模型都是不可信数据：Python 一律 `tools/.venv/bin/python -I` 运行；缺依赖改 `tools/pyproject.toml` 后 `uv sync --project tools`。

## 帧分类（每帧 `class` + `why`）

优先级：**特效帧 > 目测帧 > 普通帧**。

| class | 判定 | 门槛 |
|---|---|---|
| `effect` 特效帧 | 落在 halftone / radialBlur / zoomBlur / swirl / silhouette / fadeFromWhite / fadeFromBlack / particles 效果区间；或镜头 transitionIn 区间；或任一实例 blurSigma > 3 | 不卡门槛，肉眼为准 |
| `visual` 目测帧 | 同一帧 ≥ 2 个可见实例；或最大可见实例的 head 高度 < 200px | 不卡门槛，并排/叠加对比为准 |
| `normal` 普通帧 | 其余 | 逐帧 SSIM@360 ≥ 门槛（`metric.THRESHOLD_360`，现为 0.6） |

- 可见实例：shots.json 实例 track 的 bbox 非 null（插值规则与覆盖率的 bbox 相同：相邻采样任一端为 null，区间内视为不可见）。
- head 高度：在非 null 的相邻 head 采样之间线性插值，取 head[3]。
- `why` 只列出所属类别的依据（普通帧为空）。另有 `whyVisual`：目测依据，与类别无关（特效帧上也会给出），
  以及 `visibleInstances`、`maxHeadH`。

## metric.py

```bash
# 指标 JSON（renders 为 pnpm render 输出目录，frame_NNNNNN.png）
tools/.venv/bin/python -I tools/qa/metric.py score --renders <dir> --shots s24,s43 [--frames a-b] --out <json>
# 单镜头四宫格对照视频 + 最差/中位/最好帧截图
tools/.venv/bin/python -I tools/qa/metric.py review --renders <dir> --shots s43 --out <dir>
```

输出 JSON 只新增、不删改字段：
- 每镜头 `summary` 新增 `visual`（与 `normal`、`effect` 同结构）和 `threshold360`（现行门槛的值，现为 0.6）；
- `ssim360` 统计里，按现行门槛计算的是通用字段 `fracGEthr`、`framesBelowThr`（门槛再变时字段名不变）；
- 旧字段按字面数值计算并保留，只为向后兼容：`fracGE0.85` / `framesBelow0.85` 按 0.85，`fracGE0.7` / `framesBelow0.7` 按 0.7；
- `normalPass` 按现行门槛判定：普通帧有效帧数 > 0 且最小值 ≥ `threshold360`；
- 每帧新增 `visibleInstances`、`maxHeadH`、`whyVisual`。

## report.py：全片质检报告

```bash
# 全片：先用当前 scene.json 渲染已拟合镜头，再出报告和并排差异视频（重任务，见下方 systemd-run）
tools/.venv/bin/python -I tools/qa/report.py --render

# 只看部分镜头（边拟合边看）；用现成渲染目录，不出视频
tools/.venv/bin/python -I tools/qa/report.py --shots s15,s24 --renders .data/qa/report/renders --no-video

# 换场景文件、换输出目录、换视频布局
tools/.venv/bin/python -I tools/qa/report.py --scene .data/fit/diag/scene_x.json --render \
    --renders .data/qa/rx/renders --out .data/qa/rx --layout heat
```

参数：
- `--shots s15,s24`：只跑这些镜头（缺省全部 46 个）。视频只包含这些镜头，音频取原 wav 的对应区段并拼接，保持音画同步。
- `--scene`：判断哪些镜头已拟合：镜头 [start, end) 与任一模型图层的 [in, out) 有交集即算已拟合（不看图层 id；continues 串起来的跨镜头实例会合并成一个图层，如 s16.next 覆盖 s17）；`--render` 时也用它渲染。缺省 `scenes/deepseek-chan/scene.json`。
- `--renders`：渲染帧目录，缺省 `<out>/renders`。不加 `--render` 时直接读取，缺帧会报错。
- `--render`：先跑 `pnpm render`，只渲染选中镜头与图层区间的交集（会覆盖同名帧）。已拟合镜头里没有图层覆盖的帧按空渲染（全透明）计算，遮罩 <1%，不计有效帧。
- `--layout`：差异视频布局：
  - `sbs`：并排，2560×720，缺省；
  - `overlay`：渲染以 50% 不透明度叠在原视频上，错位处会出现重影；
  - `heat`：原视频灰度作底，叠 SSIM@360 热力图，0.5 以下红色，1 为绿色，排除区显示为蓝灰；
  - `quad`：四宫格，依次是原视频、合成、差异图、热力图。
- `--no-video`、`--worst N`：每镜头每类保留 N 张最差帧缩略图，缺省 2。`--crf`、`--preset` 是 x264 参数。

输出目录缺省为 `.data/qa/report/`：

| 文件 | 内容 |
|---|---|
| `frames.csv` | 逐帧：frame, shot, fitted, class, why, ssim360, ssim720, coverage, maskArea, visibleInstances, maxHeadH, whyVisual（未拟合镜头的帧也列出类别，SSIM 为空） |
| `shots.csv` | 逐镜头：状态，以及每类的 frames / valid / mean / min / p5 / fracGE<门槛> / below<门槛>（列名带现行门槛值，如 `normal_fracGE0.6`）、coverageMean |
| `report.json` | 每镜头 `summary`（与 `metric.py score` 同结构）、`perFrame`、`classFrames`、`status`、`thumbs`；全片 `overall`；`ranking` |
| `index.html` | 静态报告，可直接本地打开，无外部依赖：全片汇总、最差镜头排行、逐镜头表、每镜头 SSIM 曲线（内联 SVG，悬停看数值）、最差帧三联缩略图 |
| `thumbs/` | 最差帧三联图，依次为原视频、渲染（合成到原视频背景）、差异图 |
| `diff_<layout>.mp4` | 60fps 差异视频，带原音频。左上角烧录帧号、镜头、类别、SSIM；未拟合镜头显示原视频并标注「未拟合」 |

镜头状态：
- `达标`：普通帧全部 ≥ 门槛；
- `不达标`：存在低于门槛的普通帧；
- `无有效普通帧`：没有有效的普通帧（只有目测帧或特效帧，或普通帧都落在无图层/无角色区段），以肉眼为准；
- `未拟合`：镜头区间与 scene.json 里任何模型图层都没有交集。

最差镜头排行：
- 普通帧按低于门槛的帧数降序排，再按均值升序；
- 目测帧、特效帧单独列表，各自按均值升序。

统计只包含已拟合镜头；无角色帧（遮罩 < 1% 画面）不计入有效帧。

### 内存（机器 14GB，拟合会占 7～8GB）

全片渲染和视频编码要用 systemd-run 限制内存：

```bash
LOG=$PWD/.data/qa/report/run.log
systemd-run --user --unit=cf-qa-report --collect --working-directory=$PWD --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  -p MemoryMax=3G -p StandardOutput=append:$LOG -p StandardError=append:$LOG \
  bash tools/qa/run_report.sh [--shots s15,s24 ...]   # = report.py --render --layout sbs "$@"
systemctl --user is-active cf-qa-report
```

report.py 逐帧流式解码参考视频，不会整段载入内存。连续的镜头共用一个解码管道。

速度参考：
- 已拟合帧约 2 帧/秒，主要耗时在全分辨率 SSIM@720；
- 未拟合帧只做合成和编码，速度快得多。
