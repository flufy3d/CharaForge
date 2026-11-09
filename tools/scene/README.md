# scene 后处理

## `compact.py`

把 `scene.json` 里逐帧烘焙（`baked`）的动画曲线压成稀疏关键帧（`keys`），同时压掉拟合器留下的逐帧抖动。

```bash
# 输出到新文件（推荐先这样看报告）
python tools/scene/compact.py scenes/deepseek-chan/scene.json --out /tmp/scene.compact.json

# 确认没问题后原地替换
python tools/scene/compact.py scenes/deepseek-chan/scene.json --in-place
```

> 这个脚本只处理 `baked` 曲线，所以要在**原始烘焙场景**上跑。已经压缩成 keys 后想重新调 σ/容差，
> 先 `git checkout -- scenes/deepseek-chan/scene.json` 恢复，再跑一次；直接对压缩版重跑会二次平滑（工具会拒绝，除非加 `--force`）。

默认行为：

- `postPhysics` 参数轨：5 点中值去单帧尖峰 → σ=1.5 帧高斯低通（两端 3 帧渐隐，入/出帧不变）。
- `pre` 参数轨：只替换「涨上去又立刻回来」的孤立单帧点，不动眨眼/口型。
- 变换、clip、opacity、模糊：不平滑，只抽帧，避免动到镜头运动。
- 每条曲线用 RDP 垂距抽帧；常量参数轨只留 1 个 key。
- 如果某条曲线 keys 反而比 baked 大（短曲线/高频曲线），保留 baked；可用 `--no-hybrid` 强制全部输出 keys。
- 数值按类别保留 2～5 位小数。
- 参数轨上重复的 `provenance` 每图层每类只保留 1 份（内容相同，渲染无关）；可用 `--keep-param-provenance` 全保留。

常用参数：

| 参数 | 说明 |
|---|---|
| `--sigma 1.5` | postPhysics 高斯平滑 σ（帧）；`0` 关闭 |
| `--no-median` | 关闭 postPhysics 5 点中值去尖峰 |
| `--no-despike-pre` | 关闭 pre 孤立点替换 |
| `--tolerance-scale 1.0` | 抽帧容差倍率；越大 key 越少 |
| `--no-hybrid` | 所有曲线都输出 keys |
| `--keep-param-provenance` | 保留每条参数轨的 provenance（默认去重） |
| `--force` | 输入已经是 keys 压缩版时仍继续（只处理剩余 baked 曲线；会二次平滑） |
| `--quiet` | 不打印体积/误差报告 |

脚本只依赖 Python 标准库，不需要 `uv sync`。
