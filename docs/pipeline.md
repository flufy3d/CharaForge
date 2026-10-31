# 管线约定（子代理之间的契约）

## 目录

| 路径 | 内容 | 是否进 git |
|---|---|---|
| `vendor/cubism/` | Cubism Core（`core/`）与 Framework 源码（`framework/src/`），`pnpm setup:cubism` 安装 | 否 |
| `assets/models/deepseek-chan/DeepSeek/` | 模型（`大肥鱼.model3.json` 等），`pnpm setup:assets` 安装 | 否 |
| `assets/video/DeepSeek-chan.mp4` / `.wav` | 参考视频 / 解码后的 PCM 音频 | 否 |
| `packages/scene/schema/*.json` | 由 `pnpm schema` 从 TypeBox 定义生成的 JSON Schema（draft-07） | 是 |
| `scenes/deepseek-chan/shots.json` | 镜头表（recon 产出，人工校对后提交） | 是 |
| `scenes/deepseek-chan/scene.json` | 复刻场景（fit 产出） | 是 |
| `.data/recon/` | 抽帧、缩略图拼图、recon 中间产物 | 否 |
| `.data/fit/` | 拟合中间产物、逐镜头结果 | 否 |
| `.data/qa/` | 质检报告、差异视频 | 否 |

资源根目录（`assetRoot`）= 仓库下的 `assets/`。场景里的 `uri` 都相对于它。

## 契约

- **场景格式** `charaforge.scene` v1：定义在 `packages/scene/src/scene.ts`，文件头注释里写了坐标系、
  单帧求值顺序、物理确定性、合成顺序等规范性约定，播放器、拟合器、编辑器都必须遵守。
- **镜头表** `charaforge.shots` v1：定义在 `packages/scene/src/shots.ts`。
- TS 端用 `parseScene` / `parseShots`（结构校验 + 语义校验）；Python 端用
  `jsonschema.Draft7Validator` 加载 `packages/scene/schema/*.json` 校验。
- 修改 schema 只能由总负责人进行；子代理发现 schema 不够用，要在汇报里提出，不能自行修改。

## 安全

下载的压缩包、视频、模型都视为不可信数据：只解析，不执行；Python 一律用 `-I` 运行；
不要把脚本放进资源目录，也不要在资源目录里执行任何东西。

## 验收指标的定义（fit 和 qa 必须用同一份实现：`tools/qa/metric.py`）

2026-10-31 与用户确认：主指标在 640×360 尺度上计算，用来衡量正常观看尺度下动作和表情的差异，
抵消视频压缩、画面放大变软、渲染器滤波这类和动作无关的差异。具体门槛等打样测出可达上限后再和用户商定。

对每个有角色的帧 f：

1. **角色遮罩** M_f = dilate(α_f > 0.1, 半径 3px) − 排除区（全分辨率）。
   - α_f 是我们渲染结果的 alpha，所有图层合成后的结果；
   - 排除区 = shots.json 里覆盖帧 f 的 exclusions 矩形之并。
2. **合成**：我们的渲染结果按 alpha 合成到参考帧上（等价于用原视频作背景），得到 C_f；参考帧记为 R_f。
3. **主指标 SSIM@360**：
   - C_f、R_f 用 `cv2.resize(..., (640, 360), interpolation=cv2.INTER_AREA)` 降采样；
   - 遮罩用同样的方法降采样，取 > 0.5 的部分；
   - 在 RGB 三个通道上分别计算 SSIM 图：高斯窗 σ = 1.5，K1 = 0.01，K2 = 0.03，数据范围 0–255（与 `skimage.metrics.structural_similarity(gaussian_weights=True, use_sample_covariance=False)` 一致）；
   - 三个通道取平均，再在降采样后的遮罩上取平均。
4. **辅助指标 SSIM@720**：同上，但不降采样，用于观察细节。
5. M_f 面积小于 1% 画面时，这一帧记为「无角色」，不参与统计。
6. **覆盖率**（防止"漏画部分角色、遮罩就跟着变小"的漏洞）：
   - 参考侧角色区域取 shots.json 里该帧的 bbox，也就是 recon 估计的可见外接框；
   - 记录 |M_f ∩ bbox| / |bbox ∖ 排除区|；
   - 覆盖率明显偏低的帧需要人工看。
7. **镜头指标**：两种 SSIM 各自的均值、最小值、5% 分位数，以及 ≥ 0.9 的帧所占比例。全片指标对所有有效帧求平均。

**验收门槛**（2026-10-31 与用户确认，依据：合成上限实验 SSIM@360 均值 0.934、最低 0.910）：
- **普通帧**：逐帧 SSIM@360 ≥ 0.85。
- **特效帧**：单独列清单、单独统计，不套用上面的门槛，以并排、叠加时肉眼对比为准。特效帧包括：
  - 落在 shots.json 中 kind 为 halftone、radialBlur、zoomBlur、swirl、silhouette、fadeFromWhite、fadeFromBlack、particles 的 effects 区间内的帧；
  - 落在镜头 transitionIn 过渡区间内的帧；
  - 角色估计模糊 blurSigma > 3 的帧。

  metric.py 要输出每帧属于普通帧还是特效帧，以及判定依据。
- 无论哪类帧，都要满足：并排、叠加对比时肉眼看不出角色动作和表情的差异。
