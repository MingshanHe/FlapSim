#!/usr/bin/env bash
# H5 x seeds (discussion): residual networks trained with +-60 deg unknown actuator lag vs the robustly tuned
# hand law. About 2 h per seed + 1 h lag scan.
source "$(dirname "$0")/common.sh"
RES="$RAW/h5"
mkdir -p "$RES" "$OUT/e7_policies"
ARMS="snn_neat_delay ann_neat snn_fixed"

for s in $SEEDS; do
  step e7_arms_s$s "$RES/DONE_seed$s" \
    bash experiments/algorithms/wingbeat_snn/run_arms.sh "$SCRATCH" "$RES" $s --residual \
    --residual-base "$P1_RETUNE" --readout-std 0.3 --spike-scale 5 --levels 0.75 1.0 1.25 \
    --validation-levels 1.0 1.25 --lag-range 60 --workers $ARM_WORKERS \
    --population $RES_POP --generations $RES_GENS --cases $RES_CASES --seconds $RES_SECONDS
done

POL=()
for s in $SEEDS; do
  for m in $ARMS; do
    POL+=(--policy "h5_${m}_s$s=$RES/${m}_seed$s/best_policy.json")
    mkdir -p "$OUT/e7_policies/${m}_seed$s"
    cp "$RES/${m}_seed$s/"{best_policy.json,history.jsonl,results.json} "$OUT/e7_policies/${m}_seed$s/" 2>/dev/null
  done
done
step e7_lag_scan "$OUT/e7_h5_seeds.json" \
  $PYTHON -W ignore -m $EVAL lag_scan --scratch "$SCRATCH" --workers "$WORKERS" \
  --retuned "$P1_RETUNE" --retuned "$H5_ROBUST" --levels 1.0 1.25 --lags $H5_LAGS --count $N_H5 --offset 640000 \
  "${POL[@]}" --output "$OUT/e7_h5_seeds.json"
echo "h5 campaign finished $(date -Is)"
