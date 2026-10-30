# CharaForge

**AI-Powered Live2D Character Studio**

面向虚拟 AI 角色的 Live2D 可视化编辑与动画创作工具。

### 核心功能

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

## 快速开始

```bash
pnpm install
pnpm setup          # 下载 Cubism SDK、模型与参考视频（校验 sha256），建立 Python 环境
```

管线约定见 [docs/pipeline.md](docs/pipeline.md)。

## 授权

本项目分三层授权：

| 部分 | 许可 | 说明 |
|---|---|---|
| CharaForge 代码 | [MIT](LICENSE) | 本仓库中的全部代码 |
| Live2D Cubism Core | [Live2D Proprietary Software License](https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html) | **不包含在仓库中**，`pnpm setup:cubism` 从官方地址下载并校验 |
| Live2D Cubism Framework | [Live2D Open Software License](https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html) | 同上，随 SDK 一起安装到 `vendor/cubism/` |
| DeepSeek-chan 模型 | 原作者的使用条款 | **不包含在仓库中**，`pnpm setup:assets` 下载；不得再分发模型文件 |

模型署名：**DeepSeek-chan** — Illustration & PSD Layer Separation: Kamiya Ryoko ／
Live2D Rigging: 霓虹猫_NeonNeko ／ Free model distributed by Kamiya Ryoko
（[VGen](https://vgen.co/KAMIYA)、[X @kamiya_ryoko](https://x.com/kamiya_ryoko)）。

模型请从原作者的发布页获取并支持原作者：
<https://vgen.co/KAMIYA/product/deepseek-chan-deepseek-free-live2d-model-layered-psd-psd/1c55a71a-2747-4616-819d-2dc0a934089d>

`pnpm build` 产出的静态页面仅供本地查看，不公开部署（公开部署并附带模型文件会构成再分发）。
