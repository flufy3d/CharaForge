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
| `normal` 普通帧 | 其余 | 逐帧 SSIM@360 ≥ 门槛（`--threshold`，缺省 `metric.THRESHOLD_360` = 0.6） |

- 可见实例：shots.json 实例 track 的 bbox 非 null（插值规则与覆盖率的 bbox 相同：相邻采样任一端为 null，区间内视为不可见）。
- head 高度：在非 null 的相邻 head 采样之间线性插值，取 head[3]。
- `why` 只列出所属类别的依据（普通帧为空）。另有 `whyVisual`：目测依据，与类别无关（特效帧上也会给出），
  以及 `visibleInstances`、`maxHeadH`。

## metric.py

```bash
# 指标 JSON（renders 为 pnpm render 输出目录，frame_NNNNNN.png）
tools/.venv/bin/python -I tools/qa/metric.py score --renders <dir> --shots s24,s43 [--frames a-b] [--threshold 0.6] --out <json>
# 单镜头四宫格对照视频 + 最差/中位/最好帧截图
tools/.venv/bin/python -I tools/qa/metric.py review --renders <dir> --shots s43 --out <dir>
```

门槛是参数 `--threshold`（`score`、`review`、`report.py` 都有），缺省 `metric.THRESHOLD_360` = 0.6（现行普通帧门槛）。
门槛只改变下列「按门槛」的字段，不改 SSIM 本身。

输出 JSON 只新增、不删改字段：
- 每帧：`class`（normal / visual / effect）、`why`（所属类别的依据）、`whyVisual`、`visibleInstances`、`maxHeadH`；
  `passThr`（只有有效普通帧为 true/false，目测帧、特效帧、无角色帧为 null）、`verdict`（判定文字，如「普通帧 SSIM@360 0.642 ≥ 0.6」「目测帧，不卡门槛」）；
  闪烁字段 `flicker`、`flickerRef`、`flickerSpike`、`flickerRefDup`（见下节，未判定为 null）。
- 每镜头 `summary`：`threshold360`（本次门槛）；`normal` / `visual` / `effect` 三类各自的 `frames`、`ssim360`、`ssim720` 统计；
  顶层 `ssim360` / `ssim720` 是三类合计，只作参考。
- `ssim360` 统计里按门槛计算的通用字段：`thr`、`framesGEthr`（≥ 门槛帧数）、`fracGEthr`（= framesGEthr / valid）、
  `framesBelowThr`（< 门槛帧号列表；framesGEthr + len(framesBelowThr) = valid）。门槛再变时字段名不变。
  visual、effect 里的这些字段只是同一门槛下的分布参考，不是判定。
- 旧字段按字面数值计算并保留，只为向后兼容：`fracGE0.85` / `framesBelow0.85` 按 0.85，`fracGE0.7` / `framesBelow0.7` 按 0.7，`fracGE0.9` 按 0.9。
- `normalPass`：普通帧有效帧数 > 0 且最小值 ≥ `threshold360`。
- 每镜头 `flicker`（闪烁汇总）；全片 `overall` 保留旧的 `valid` / `ssim360` / `ssim720`（所有有效帧均值），
  新增 `overall.summary`（与镜头 summary 同结构，分三类）和 `flickerOverall`。

## flicker.py：闪烁（时间稳定性）

唯一实现。`metric.py score` 和 `report.py` 都会逐帧调用它；也可单独跑：

```bash
tools/.venv/bin/python -I tools/qa/flicker.py <渲染目录> <起帧-止帧(含)> [--json out.json]
# 旧用法仍可用（转调上面这个）：
tools/.venv/bin/python -I tools/fit/scripts/flicker.py <渲染目录> <起帧-止帧(含)>
```

口径（本节，唯一实现见 `flicker.py`）：
- 渲染（非预乘 RGB，不合成到原视频）和参考帧（与 metric 同一套 ffmpeg 颜色转换）都用 INTER_AREA 降到 320×180 灰度；
- 区域 = 相邻两帧渲染 alpha 都 > 0.5 的像素，不超过 200 像素不判定；
- Δ我们、Δ原视频 = 该区域内相邻两帧灰度差的平均；
- **尖峰**（`flickerSpike`）：Δ我们 > 3 × max(Δ原视频, 1) 且 Δ我们 > 4；
- **镜头切点排除**：每个镜头的第一帧不和上一镜头比较（单独跑时按 shots.json 的镜头起点判断）；
- **原视频重复帧单独统计**（`flickerRefDup`）：Δ原视频 < 0.3（原视频这一区域没变，多为重复帧/定格）且 Δ我们 > 2。
  同时满足尖峰条件的帧也计入尖峰，汇总里 `refDupSpikes` 给出重叠数。

汇总字段（每镜头 `flicker`、全片 `flickerOverall`）：`samples`（判定帧数）、`spikes`、`spikeFrames`、`refDupMoving`、
`refDupMovingFrames`、`refDupSpikes`、`medianOurs`、`medianRef`。

与旧 `tools/fit/scripts/flicker.py`（提交 e6cad80 及以前）的差别：旧脚本的参考帧用 ffmpeg 缺省缩放直接取 320×180 的亮度平面，
渲染在 uint8 上缩放；现在参考帧统一为 metric 的全范围 RGB，两边都用 INTER_AREA 在浮点上缩放再转灰度。
同一份 s41 渲染上两者的尖峰帧完全相同（18 帧），Δ 中位数略有出入（原视频 14.21 → 13.77，我们 16.37 → 16.36）。

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
- `--threshold`：普通帧门槛，缺省 0.6（`metric.THRESHOLD_360`）。HTML、CSV、视频标注、状态判定都用这一个值，写进 `report.json` 的 `threshold360`；`--rehtml` 时从 report.json 读回。
- `--layout`：差异视频布局：
  - `sbs`：三栏 3840×720，缺省。依次是原视频 | **我们的纯渲染**（按 alpha 合成到中性灰底，`--bg checker` 换棋盘底；不合成到原视频，所以没盖住的地方不会透出原角色造成假重影）| 差异图 |C_f − R_f|×3（`--third heat` 换成 SSIM@360 热力图）。只是可视化变了，指标仍按定义把渲染合成到参考帧上算；
  - `sbs-comp`：旧的两栏并排 2560×720，右栏是合成到原视频上的 C_f（指标输入本身，没盖住处会透出原角色），只用于核对；
  - `overlay`：渲染以 50% 不透明度叠在原视频上，错位处会出现重影；
  - `heat`：原视频灰度作底，叠 SSIM@360 热力图，0.5 以下红色，1 为绿色，排除区显示为蓝灰；
  - `quad`：四宫格，依次是原视频、合成、差异图、热力图。
- `--no-video`、`--worst N`：每镜头每类保留 N 张最差帧缩略图，缺省 2。`--crf`、`--preset` 是 x264 参数。

输出目录缺省为 `.data/qa/report/`：

| 文件 | 内容 |
|---|---|
| `frames.csv` | 逐帧：frame, shot, fitted, class, why, ssim360, ssim720, coverage, maskArea, visibleInstances, maxHeadH, whyVisual, passThr, verdict, flicker, flickerRef, flickerSpike, flickerRefDup（未拟合镜头的帧也列出类别，SSIM 为空） |
| `shots.csv` | 逐镜头：状态，以及每类的 frames / valid / mean / min / p5 / GE<门槛> / fracGE<门槛> / below<门槛>（列名带本次门槛值，如 `normal_GE0.6`），coverageMean，flickerSamples / flickerSpikes / flickerRefDupMoving / flickerSpikeFrames |
| `report.json` | `threshold360`；每镜头 `summary`（与 `metric.py score` 同结构）、`flicker`、`perFrame`、`classFrames`、`status`、`thumbs`；全片 `overall`（同 summary 结构）、`flickerOverall`；`ranking` |
| `index.html` | 静态报告，可直接本地打开，无外部依赖：全片汇总（三类分开，含 ≥门槛帧数）、最差镜头排行、闪烁按镜头排行、逐镜头表、每镜头 SSIM 曲线（内联 SVG，悬停看数值，顶部红线为闪烁尖峰）、最差帧三联缩略图 |
| `thumbs/` | 最差帧三联图，依次为原视频、纯渲染（中性底）、差异图 |
| `diff_<layout>.mp4` | 60fps 差异视频，带原音频。左上角烧录帧号、镜头、类别、SSIM@360/@720、普通帧是否 ≥ 门槛、闪烁尖峰；未拟合镜头显示原视频并标注「未拟合」 |

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
