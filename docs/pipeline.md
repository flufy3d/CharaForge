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
