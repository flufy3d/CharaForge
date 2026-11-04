#!/usr/bin/env bash
# 全片质检：渲染已拟合镜头（当前 scene.json）→ 全片报告 + 并排差异视频
set -euo pipefail
tools/.venv/bin/python -I tools/qa/report.py --render --layout sbs "$@"
