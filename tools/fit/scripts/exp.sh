#!/usr/bin/env bash
# 诊断实验：拟合（FIT_MODEL 变体）→ 合成临时场景 → pnpm render → metric score → 分区误差（稳定段）
#   FIT_MODEL='{"decouple":true}' tools/fit/scripts/exp.sh <tag> <shot> [fit4 的其他参数...]
# 结果：.data/fit/<shot>_<tag>/（拟合）、.data/fit/diag/exp_<tag>_<shot>.txt（指标摘要）
set -euo pipefail
cd "$(dirname "$0")/../../.."
TAG=$1; SHOT=$2; shift 2
NODE="node --experimental-strip-types --disable-warning=ExperimentalWarning"
case $SHOT in
  s24) RANGE=2368-2441; STABLE=$(seq -s, 2396 3 2441);;
  s43) RANGE=3668-3721; STABLE=$(seq -s, 3695 2 3721);;
  s15) RANGE=1187-1508; STABLE=$(seq -s, 1360 6 1508);;
esac
OUT=.data/fit/diag/exp_${TAG}_${SHOT}.txt
FIT_TAG=_$TAG $NODE tools/fit/src/cli.ts fit4 --shot $SHOT "$@" > .data/fit/diag/exp_${TAG}_${SHOT}.fit.log 2>&1
FIT_TAG=_$TAG $NODE tools/fit/src/cli.ts assemble --out .data/fit/diag/scene_${TAG}.json > /dev/null
rm -rf .data/fit/diag/r_${TAG}_${SHOT}
pnpm -s render --scene .data/fit/diag/scene_${TAG}.json --frames $RANGE --out .data/fit/diag/r_${TAG}_${SHOT} > /dev/null 2>&1
{
  echo "== $TAG $SHOT FIT_MODEL=${FIT_MODEL:-} args=$*"
  grep -h '结束\|完成' .data/fit/${SHOT}_${TAG}/fit.log | cut -c1-240
  tools/.venv/bin/python -I tools/qa/metric.py score --renders .data/fit/diag/r_${TAG}_${SHOT} --shots $SHOT --out .data/fit/diag/metric_${TAG}_${SHOT}.json 2>&1 | tail -8
  for L in $(python3 -c "import json;print(' '.join(l['id'] for l in json.load(open('.data/fit/diag/scene_${TAG}.json'))['layers'] if l['id'].startswith('${SHOT}.')))"); do
    $NODE --import ./tools/fit/src/geom/register.mjs tools/fit/scripts/meshes.ts .data/fit/diag/scene_${TAG}.json $L $STABLE .data/fit/diag/mesh_${TAG}_${SHOT}.json > /dev/null 2>&1
    echo "-- 分区（$L，稳定段）"
    tools/.venv/bin/python -I tools/fit/regions.py .data/fit/diag/mesh_${TAG}_${SHOT}.json --renders .data/fit/diag/r_${TAG}_${SHOT} --shot $SHOT
    [ "$SHOT" = s15 ] || break
  done
} > $OUT 2>&1
cat $OUT
