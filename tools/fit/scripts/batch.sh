#!/usr/bin/env bash
# P3 批量驱动：按镜头顺序逐个拟合单元（continues 连起来的镜头合成一个单元）→ 立即打分 → 记一行摘要 → 最后合成全片 scene.json。
#
#   tools/fit/scripts/batch.sh [all | s08,s17,...]      # 参数为镜头 ID（自动换成所在单元），缺省 all
#
# 环境变量：
#   FIT_MODEL   缺省 '{"bodyJac":true}'（与打样正式结果一致）
#   FIT_ARGS    追加给 fit4 的参数（如 --fixed '{"lean":-10}'），对所有单元生效
#   ITERS_B     固定 --itersB；缺省按单元的图层帧数之和：≤200 → 40，≤400 → 20，其余 10
#   REVIEW=0    不生成对照视频（缺省生成到 .data/fit/review/<镜头>.mp4）
#   KEEP_RENDERS=1  保留渲染 PNG（缺省打分、review 后删除）
#
# 已有 .data/fit/<单元>/result.json 的单元跳过拟合（打样镜头 s15/s24/s43 一律跳过、不重跑），但若还没打过分会补打分；
# 拟合中断后原样重跑即可（fit4 的图层断点 done_<图层>.json / 阶段断点 work/ckpt_*）。
# 某个单元失败：在摘要里记 FAIL 与日志末行，继续下一个。
# 产出：.data/fit/batch/summary.tsv（每单元一行）、<单元>.fit.log、metric_<单元>.json；最后 scenes/deepseek-chan/scene.json。
# 长任务请用 systemd-run 启动（见 tools/fit/README.md「P3 批量」）。
set -uo pipefail
cd "$(dirname "$0")/../../.."
NODE="node --expose-gc --experimental-strip-types --disable-warning=ExperimentalWarning"
PY="tools/.venv/bin/python -I"
SEL=${1:-all}
B=.data/fit/batch
mkdir -p "$B" .data/fit/review
SUM=$B/summary.tsv
[ -f "$SUM" ] || printf 'time\tunit\tshots\tstatus\tfitMin\tmemPeakG\tnormal(n/mean/min/>=0.7)\tvisual(n/mean/min)\teffect(n/mean/min)\tcoverage\tnote\n' > "$SUM"
export FIT_MODEL=${FIT_MODEL:-'{"bodyJac":true}'}

$NODE tools/fit/src/cli.ts units --out "$B/units.json" > "$B/units.txt" || { echo "units 失败"; exit 1; }
# 选中的单元：单元 ID、区间（含两端）、itersB、镜头列表、是否打样
mapfile -t ROWS < <(python3 - "$B/units.json" "$SEL" "${ITERS_B:-}" <<'PY'
import json, sys
units = json.load(open(sys.argv[1])); sel = sys.argv[2]; fixed = sys.argv[3]
want = None if sel == 'all' else set(sel.split(','))
for u in units:
    if want is not None and not (want & set(u['shots'])): continue
    tot = sum(l['out'] - l['in'] for l in u['layers'])
    ib = fixed or (40 if tot <= 200 else 20 if tot <= 400 else 10)
    print(u['unit'], f"{u['range'][0]}-{u['range'][1] - 1}", ib, ','.join(u['shots']), 1 if u.get('legacy') else 0, len(u['layers']))
PY
)

for row in "${ROWS[@]}"; do
  read -r U RANGE IB SHOTS LEGACY NL <<< "$row"
  T0=$(date +%s); STATUS=ok; NOTE=""
  RES=.data/fit/$U/result.json
  if [ -f "$RES" ]; then
    STATUS=skip-fit
    [ -f "$B/metric_$U.json" ] && { echo "[batch] $U 已有结果与打分，跳过"; continue; }
  elif [ "$LEGACY" = 1 ]; then
    printf '%s\t%s\t%s\tFAIL\t\t\t\t\t\t\t打样镜头缺 result.json（不在批量里重跑）\n' "$(date +%F_%T)" "$U" "$SHOTS" >> "$SUM"; continue
  elif [ "$NL" = 0 ]; then
    printf '%s\t%s\t%s\tSKIP\t\t\t\t\t\t\t没有可拟合的实例（见 units.json skipped/issues）\n' "$(date +%F_%T)" "$U" "$SHOTS" >> "$SUM"; continue
  else
    echo "[batch] $(date +%T) 拟合 $U（$SHOTS，帧 $RANGE，itersB $IB）"
    # shellcheck disable=SC2086
    if ! eval "$NODE tools/fit/src/cli.ts fit4 --shot $U --workers 3 --geom 5 --itersB $IB ${FIT_ARGS:-}" >> "$B/$U.fit.log" 2>&1; then
      NOTE=$(grep -a 'Error\|错误\|超过上限' "$B/$U.fit.log" | tail -1 | cut -c1-200 | tr '\t' ' ')
      [ -n "$NOTE" ] || NOTE=$(tail -1 "$B/$U.fit.log" | cut -c1-200 | tr '\t' ' ')
      printf '%s\t%s\t%s\tFAIL\t%s\t\t\t\t\t\t%s\n' "$(date +%F_%T)" "$U" "$SHOTS" "$(( ($(date +%s) - T0) / 60 ))" "$NOTE" >> "$SUM"
      continue
    fi
  fi
  FITMIN=$(( ($(date +%s) - T0) / 60 ))
  MEM=$(grep -ah '内存' ".data/fit/$U/fit.log" 2>/dev/null | sed -n 's/.*cgroup \([0-9.]*\)G.*/\1/p' | sort -n | tail -1)
  # 打分：只含本单元图层的临时场景 → 渲染 → metric score（按单元内各镜头）→ review
  R=$B/r_$U
  if ! { $NODE tools/fit/src/cli.ts assemble --only "${SHOTS%%,*}" --out "$B/scene_$U.json" > /dev/null \
      && rm -rf "$R" && pnpm -s render --scene "$B/scene_$U.json" --frames "$RANGE" --out "$R" > "$B/$U.render.log" 2>&1 \
      && $PY tools/qa/metric.py score --renders "$R" --shots "$SHOTS" --out "$B/metric_$U.json" > "$B/$U.score.log" 2>&1; }; then
    printf '%s\t%s\t%s\tFAIL-score\t%s\t%s\t\t\t\t\t%s\n' "$(date +%F_%T)" "$U" "$SHOTS" "$FITMIN" "$MEM" "$(tail -1 "$B/$U.score.log" 2>/dev/null | cut -c1-160)" >> "$SUM"
    continue
  fi
  LINE=$(python3 - "$B/metric_$U.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))['shots']
rows = [r for s in m.values() for r in s['perFrame'] if r.get('ssim360') is not None]
def st(c, gate=False):
    v = [r['ssim360'] for r in rows if r.get('class') == c]
    if not v: return '0'
    s = f"{len(v)}/{sum(v)/len(v):.3f}/{min(v):.3f}"
    return s + (f"/{sum(x >= 0.7 for x in v)}" if gate else '')
cov = [r['coverage'] for r in rows if r.get('coverage') is not None]
print('\t'.join([st('normal', True), st('visual'), st('effect'), f"{sum(cov)/len(cov):.3f}" if cov else '']))
PY
)
  if [ "${REVIEW:-1}" != 0 ]; then
    for S in ${SHOTS//,/ }; do
      $PY tools/qa/metric.py review --renders "$R" --shots "$S" --out .data/fit/review >> "$B/$U.review.log" 2>&1 || NOTE="$NOTE review $S 失败"
    done
  fi
  [ "${KEEP_RENDERS:-0}" = 1 ] || rm -rf "$R"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%F_%T)" "$U" "$SHOTS" "$STATUS" "$FITMIN" "$MEM" "$LINE" "$NOTE" >> "$SUM"
  echo "[batch] $(date +%T) $U 完成：$LINE"
done

# 全片合成（parseScene 校验通过才写出）
$NODE tools/fit/src/cli.ts assemble 2>&1 | tail -1 | tee -a "$B/assemble.log"
echo "[batch] ALLDONE $(date)"
