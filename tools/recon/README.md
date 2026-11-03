# tools/recon — 参考视频镜头分析

产出 `scenes/deepseek-chan/shots.json`（格式见 `packages/scene/src/shots.ts`）。
所有脚本都用 `tools/.venv/bin/python -I` 运行（视频/模型是不可信数据，只解析不执行）。CPU 即可，不用 GPU。

## 重跑

```bash
# 0. 依赖（numpy / opencv-python-headless / jsonschema / pillow，已在 tools/pyproject.toml）
uv sync --project tools

# 1. 解码：每帧全分辨率 JPEG → .data/recon/frames/NNNNNN.jpg（帧号 0 起），
#    320x180 缓存 → .data/recon/small.npy，ffprobe 信息 + sha256 → .data/recon/video.json
tools/.venv/bin/python -I tools/recon/extract.py

# 2. 切点候选（自动，只是候选）：.data/recon/metrics.npz、cut_candidates.json、flow_residual.npy
tools/.venv/bin/python -I tools/recon/detect_cuts.py
tools/.venv/bin/python -I tools/recon/detect_cuts.py --flow     # 运动补偿残差，补漏被运动掩盖的硬切

# 2b. 帧重复段（30fps 素材每帧出现两次）→ .data/recon/cadence.json（build 写入顶层 cadence）
tools/.venv/bin/python -I tools/recon/cadence.py

# 3. 人工标注 → annotations.json（manual.py 是唯一事实来源，改标注就改它）
#    转场里角色可见部分依赖 transitions.json：它用上一次 build 的 shots.json 取实例 bbox，
#    所以首次（或改了锚点后）按 build → transitions → manual → build 跑两遍
tools/.venv/bin/python -I tools/recon/transitions.py
tools/.venv/bin/python -I tools/recon/manual.py

# 4. 跟踪加密 + 眼睛细化 + 输出 shots.json、缩略图、总览图、index.html
tools/.venv/bin/python -I tools/recon/build.py           # 选项：--only s05,s06  --no-thumbs  --no-track

# 5. 校验（Python：JSON Schema draft-07 + 语义；TS：parseShots）
tools/.venv/bin/python -I tools/recon/validate.py --sha --check-cadence
node --experimental-strip-types -e "import('./packages/scene/src/index.ts').then(m=>m.parseShots(JSON.parse(require('fs').readFileSync('scenes/deepseek-chan/shots.json','utf8'))))"
```

光流跟踪结果缓存在 `.data/recon/track_cache.json`（按锚点内容签名失效）；删掉它会全部重跟（约 1 分钟）。

## 校对

打开 `.data/recon/index.html`：每个镜头一行，左边帧区间/类别/过渡，中间缩略图拼图
（`.data/recon/thumbs/<shotId>.jpg`，均匀 12 帧含首帧和末帧，左上角烧录帧号；粗框=head，圆=eyes，
细框=bbox，白框=clip，红色半透明=排除区），右边实例/开关/排除区/效果/备注。总览图 `.data/recon/overview.jpg`。

人工复核辅助工具（产物都在 `.data/recon/review/`）：

| 工具 | 用途 |
|---|---|
| `strips.py b1,b2,... out.jpg` | 每个切点一行连续 8 帧（b-4..b+3），逐帧确认切点 |
| `sheet.py range A B STEP out.jpg` / `frames` / `crop` | 任意帧拼图 |
| `measure.py f1,f2 x,y,w,h out.jpg --scale 2 --grid 10` | 局部放大 + 视频像素坐标网格，人工量坐标 |
| `fixsheet.py [s04,...]` | 每镜头锚点帧整帧 0.5 倍 + 坐标网格 + 当前 head/eyes（`fix_<shot>.jpg`） |
| `anchorsheet.py` | 每个锚点按 head 框放大 |
| `eyesheet.py out.jpg --shots s21 --every 1` | 眼睛采样放大图（红圈 = ±3px） |

已复核的切点条带：`review/s1.jpg`…`s16.jpg`。

## 数值来源与可信度

| 量 | 来源 | 可信度 |
|---|---|---|
| 切点（start/end） | 自动候选（mad / HSV 直方图 / 边缘变化率 / 光流残差）→ 人工逐个看前后 4 帧确认 | 硬切逐帧准确；转场类镜头的边界取“遮挡物开始覆盖/完全退出”的帧，±1~2 帧 |
| 锚点 head / bbox | 人工在坐标网格图上读数（两轮，第二轮在 0.5 倍整帧网格图上重读） | head ±6~10px；bbox 粗估（±20px），只作初值 |
| 锚点之间的 head / bbox / eyes | LK 光流（头部角点）相似变换，前向+后向跟踪按时间加权融合，两端严格等于锚点；跟踪失败退化为线性插值；`trackMode: linear` 的实例只做线性插值 | 头部移动平滑时 ±10px；快速推拉/遮挡/模糊时更差 |
| eyes | 人工读数/跟踪传播后，`eyes.py` 用虹膜青色连通域细化（两眼都成功且通过几何合理性检查才采用） | 细化成功的采样约 ±2px（虹膜中心，不是眼眶中心）；未细化的（闭眼、><、圈圈眼、爱心眼、钢盆阴影、模糊）误差 10~30px。每个实例 notes 里列出了细化成功的帧 |
| rotation | 细化成功的采样：两眼连线角度；其余：人工粗估 + 光流相对旋转 | ±5°，脸部偏航时眼线角度 ≠ 平面内旋转 |
| opacity / blurSigma | 人工看图粗估 | 只表示“有/无/很重”，数值不可当真 |
| variant / switchEvents | 人工看图对照部件（见下） | 帧号多为 ±5 帧，个别标了 confidence |
| motionHints | 人工推测（多个开关同时出现与某 motion 的曲线一致） | 低（0.3~0.5），起点多在镜头外 |
| exclusions / effects | 人工 | 矩形粗略、偏大；排除区宁大勿小 |

## 第二版约定（schema 76ddb6b）

- 实例 ID 全局唯一：`<镜头ID>.<名字>`；`continues` 指向更早镜头的实例 ID（转场里旧画面的延续、
  窗口展开时窗口外残留的上一镜头、s25→s26 同一窗口、s16→s17、s27→s28）。
- 转场镜头（s05 s07 s09 s11 s14 s16 s20 s27 s31 s37 s39 s42）都是 hasCharacter=true：
  延续实例的 bbox 是「与转场前一帧比对仍未被遮挡部分」的外接框（`transitions.py`，source=`diff-vs-ref`），
  head/eyes 沿用转场前一帧的值，直到头部/眼睛被盖住（与转场末帧颜色一致）为止；完全被盖住后 bbox=null。
  遮挡物的外接框按帧写成排除区（reason = splash / silhouette / halftone / cloud / pageTurn），
  只保留面积 ≥ 区域 0.5% 的连通域。新镜头开头的擦除残留同样写成排除区。
- `clip` 是关键帧数组（`manual.py` 的 `CLIPS`）；s15 的细线框不裁切角色（角色会超出框），不给 clip。
  s15 中间实例的圆形气泡窗口（1354~1397）用 radius = 边长/2 的正方形圆角矩形表示圆。
- 实例字段 `eyePairSearch: false`（annotations.json，只影响 build）：虹膜细化不做 head 框内整体搜索，只在粗估点附近吸附；
  多实例重叠/景深（s15）时整体搜索会锁到别的实例的眼睛。
- `cadence`：`cadence.py` 判定。重复帧 = 与前一帧差值 > 12 的像素占比 < 0.5%；每对帧的第一帧必须有变化证据，
  极大连续段裁剪到首尾有证据的对，至少 4 对才算。静止画面只有在两侧都有交替证据时才被包含在段内
  （静止帧本来就满足“每两帧重复”）。只有背景 60fps、角色 30fps 的“准重复”段不写入
  （`.data/recon/cadence_partial.json` 供参考）。
- TrackSample：`source` ∈ manual / flow / linear / hold / diff-vs-ref，带 `+iris-refine` 表示 eyes 经虹膜细化；
  `confidence` 按来源给（manual 0.7、flow 0.5、linear 0.35、hold 0.2~0.3、diff-vs-ref 0.4，细化 +0.2）；
  `eyesState` 由 build.py 判定（见 `annotate_eyes_state` 注释）。
- SwitchEvent `fade`：表情热键（key*/Clothes*）默认 30 帧（VTS 0.5s），Movement*（motion 触发）为 0；
  实测明显不同的（钢盆 15、脱鞋 8、Q 版 8~9）按实测填，note 里注明。

## 文件

- `common.py` 路径/解码/读帧；`extract.py` 解码；`detect_cuts.py` 切点候选；
- `manual.py` 人工标注源（镜头边界、实例、锚点、开关、排除区、效果，以及第二轮锚点修正 `FIX`）；
- `cadence.py` 帧重复段；`transitions.py` 转场期间可见区域/遮挡物估计（产出 transitions.json，受版本控制）；
- `track.py` 锚点加密；`eyes.py` 虹膜细化；`build.py` 汇总输出；`validate.py` 校验；
- `annotations.json` 由 manual.py 生成，build.py 的输入（提交它便于 diff）。
