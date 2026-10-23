#!/usr/bin/env bash
# Paper claim C3 (negative result, made robust): NEAT/fixed-topology SNN and ANN residuals on B2+ x seeds,
# against the retuned hand law on held-out gusts. About 2 h per seed (3 arms in parallel) + 1 h comparison.
source "$(dirname "$0")/common.sh"
RES="$RAW/residual"
mkdir -p "$RES" "$OUT/e6_policies"
ARMS="snn_neat_delay ann_neat snn_fixed"

for s in $SEEDS; do
  step e6_arms_s$s "$RES/DONE_seed$s" \
    bash experiments/algorithms/wingbeat_snn/run_arms.sh "$SCRATCH" "$RES" $s --residual --readout-std 0.3 \
    --spike-scale 5 --levels 0.75 1.0 1.25 1.5 --validation-levels 1.25 1.5 --workers $ARM_WORKERS \
    --population $RES_POP --generations $RES_GENS --cases $RES_CASES --seconds $RES_SECONDS
done

POL=()
for s in $SEEDS; do
  for m in $ARMS; do
    POL+=(--policy "${m}_s$s=$RES/${m}_seed$s/best_policy.json")
    mkdir -p "$OUT/e6_policies/${m}_seed$s"
    cp "$RES/${m}_seed$s/"{best_policy.json,history.jsonl,results.json} "$OUT/e6_policies/${m}_seed$s/" 2>/dev/null
  done
done
step e6_compare "$OUT/e6_residual_seeds.json" \
  $PYTHON -W ignore -m $EVAL compare --scratch "$SCRATCH" --workers "$WORKERS" --retuned "$P1_RETUNE" \
  --levels $SEEDCMP_LEVELS --count $N_SEEDCMP --offset 630000 "${POL[@]}" --output "$OUT/e6_residual_seeds.json"
echo "residual campaign finished $(date -Is)"
