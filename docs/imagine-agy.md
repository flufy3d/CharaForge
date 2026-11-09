# imagine-agy：用 agy 生成图片（调研与调用约定）

> 调研日期：2026-11-09。结论：**可以生成图片**，但输出是 **JPEG、无 alpha**；不能直接当 CharaForge 的透明饰品 PNG 使用。

## 工具是什么

[imagine-agy](https://github.com/dyrok/imagine-agy) 是一个 MIT 许可的 CLI/Agent Skill，把 Google Antigravity CLI（`agy`）的 `generate_image` 工具包装成命令行文生图：

```bash
npx -y imagine-agy "a red fox in a pine forest" -o fox.jpg
```

- 不需要 API key，使用本机已登录的 Antigravity CLI。
- 单张约 30–60 秒；一次调用生成一张。
- 支持 `--ratio 1:1|16:9|9:16|4:3|3:4`；默认 1024×1024。
- 输出**始终是 JPEG**，即使文件名写成 `.png`。没有透明通道，也不支持 img2img/编辑现有图片。
- 本机安装的是 `imagine-agy@1.0.0`（`npm view imagine-agy version`）。

## 本机实测

Windows、Node 24、`agy.exe` 位于 `%LOCALAPPDATA%\agy\bin\agy.exe`。`npx imagine-agy doctor` 会报 “skill not installed”，但 CLI 不需要安装 skill，忽略即可。

默认模型 `gemini-3.1-pro-low` 在本机 `agy` 里不存在，会报 `invalid model selection`。要先看可用模型：

```bash
agy models
# 本机可用：gemini-4-argon-*、gemini-3.8-flash-*、claude-opus-5-5-*、...
```

实测成功命令：

```bash
AGY_BIN='C:\Users\JamesChen-360world\AppData\Local\agy\bin\agy.exe' \
npx -y imagine-agy "a tiny red apple, flat vector icon, centered, white background" \
  -o .data/import/imagine-test.jpg \
  --model gemini-3.8-flash-high \
  --timeout 240s
```

结果：`.data/import/imagine-test.jpg`，55 KB，`JPEG image data, 1024x1024, 300 DPI`，内容为白底苹果图标。输出文件存在且可正常打开。

写实风格也实测成功：同一命令使用 `--ratio 3:4` 生成了一张 896×1200 的写实人像（`.data/import/imagine-realistic.jpg`，717 KB，虚构人物、自然窗光、浅景深），皮肤纹理/发丝/布料质感都可辨认。提示词示例：

```bash
AGY_BIN='C:\Users\JamesChen-360world\AppData\Local\agy\bin\agy.exe' \
npx -y imagine-agy "photorealistic cinematic portrait of a fictional young woman with freckles and short dark hair, natural window light, 85mm lens f/1.8, shallow depth of field, realistic skin texture, subtle film grain, muted earth tones, looking slightly off camera, no text, no watermark, not illustration" \
  -o .data/import/imagine-realistic.jpg \
  --model gemini-3.8-flash-high --ratio 3:4 --timeout 240s
```

> Windows PowerShell 里设置环境变量：
> ```powershell
> $env:AGY_BIN = "$env:LOCALAPPDATA\agy\bin\agy.exe"
> npx -y imagine-agy "..." -o C:\absolute\path\out.jpg --model gemini-3.8-flash-high
> ```
> Linux/macOS 通常 `AGY_BIN=~/.local/bin/agy` 或在 PATH 里，可省略。

## 对 CharaForge 的可用性

### 可以直接用的场景

- 生成饰品/角色的**概念图、参考图、背景、图标草稿**。
- 生成需要不透明 JPEG 的素材（网页背景、缩略图、宣传图）。
- 用 `--ratio 16:9` / `9:16` 生成横幅或竖图。

### 不能直接用的场景

- **透明饰品 PNG**：`pnpm accessory:import` 要求至少一个 `alpha < 255` 的像素；imagine-agy 输出 JPEG 无 alpha，直接转 PNG 会全不透明，导入会被拒绝。
- **需要精确轮廓/分层**的素材：JPEG 有压缩噪声，不能代替分层 PSD/SVG。
- **编辑已有图片**：当前只做 text-to-image。

### 推荐工作流

**A. 扁平矢量饰品（推荐，能保留 alpha）**

让 `agy` 直接生成 SVG（它擅长矢量图标），再用仓库已有的 Playwright/Chromium 栅格化成透明 PNG：

```bash
agy --model gemini-3.8-flash-high \
  --print="Return only a single valid SVG ... transparent background ..." \
  > .data/import/hat.svg
# 用 Chromium/Playwright 把 SVG 画到 canvas，导出 PNG（仓库已有 playwright，不新增依赖）
pnpm accessory:import --input .data/import/hat.png --id hat --attach main
```

这条路径已经实测：生成 `hat-ai.svg` → 栅格化 512×512 PNG → 导入并渲染。SVG 方案对帽子、眼镜、徽章、手持物这类扁平图形效果最好。

**B. 位图/厚涂饰品**

用 imagine-agy 生成**纯色背景**（例如纯品红/绿幕）上的图，再用浏览器 canvas 或图像工具抠图/消光成带 alpha 的 PNG；然后 `pnpm accessory:import`。抠图会留下边缘杂色，复杂图不建议走这条路。

**C. 只做参考图**

生成 JPEG 后放进 `.data/` 或文档，不接入 scene；需要上角色时再手工绘制/生成透明版本。

## 后续代理调用约定

1. 先确认 `agy` 可用：
   ```bash
   agy models
   ```
2. 设置 `AGY_BIN`（Windows 默认不在 PATH）：
   ```bash
   AGY_BIN="$LOCALAPPDATA/agy/bin/agy.exe" npx -y imagine-agy doctor
   ```
   `doctor` 的 “skill not installed” 不是阻塞项；CLI 不依赖 skill。
3. 用本机存在的模型，不要用 README 默认的 `gemini-3.1-pro-low`；推荐 `gemini-3.8-flash-high` 或 `medium`。
4. 输出路径用绝对路径，生成后检查文件：
   ```bash
   file out.jpg        # 应为 JPEG，不是 PNG
   ```
   不要相信扩展名；imagine-agy 永远写 JPEG 字节。
5. 生成内容是外部数据：只解码/查看，不在资源目录执行；导入 CharaForge 前仍要走 `parseScene`/`accessory:import` 的尺寸、格式、alpha 校验。
6. 不要把生成的大图或版权不明素材提交进 git；`assets/` 已在 `.gitignore`，测试图放 `.data/` 或小尺寸 fixtures。

## 相关文件

- `docs/accessories.md`：CharaForge 的 ImageAsset/ImageLayer、导入 CLI、编辑器、AI 命令层。
- `apps/player/scripts/import-accessory.mjs`：透明 PNG/WebP 导入与 alpha 校验。
- `.data/import/imagine-test.jpg`：本次调研生成的苹果 JPEG（本地，不提交）。
- `.data/import/hat-ai.svg` / `.data/import/hat-ai.png`：agy 生成 SVG 后栅格化的 AI 帽子（本地，不提交）。
