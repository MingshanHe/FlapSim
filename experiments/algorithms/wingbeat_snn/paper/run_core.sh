#!/usr/bin/env bash
# Paper core (claim C1): open-loop authority, gust-tolerance curves, timing/lag curve, model-mismatch
# robustness and controller ablation. Only hand-designed controllers; no training. About 3 h on 18 workers.
source "$(dirname "$0")/common.sh"

step e1_authority "$OUT/e1/p0_e1.json" \
  $PYTHON -W ignore -m experiments.algorithms.wingbeat_feasibility.preliminary e1 \
  --output "$OUT/e1" --scratch "$SCRATCH" --workers "$WORKERS" --e1-seeds $E1_SEEDS

step e2_gust_curves "$OUT/e2_gust_curves.json" \
  $PYTHON -W ignore -m $EVAL compare --scratch "$SCRATCH" --workers "$WORKERS" --retuned "$P1_RETUNE" \
  --levels $CURVE_LEVELS --count $N_CURVE --offset 600000 --output "$OUT/e2_gust_curves.json"

step e3_lag_curve "$OUT/e3_lag_curve.json" \
  $PYTHON -W ignore -m $EVAL lag_scan --scratch "$SCRATCH" --workers "$WORKERS" \
  --retuned "$P1_RETUNE" --retuned "$H5_ROBUST" --levels $LAG_LEVELS --lags $LAGS --count $N_LAG --offset 610000 \
  --output "$OUT/e3_lag_curve.json"

step e4_robustness "$OUT/e4_robustness.json" \
  $PYTHON -W ignore -m $EVAL robustness --scratch "$SCRATCH" --workers "$WORKERS" --retuned "$P1_RETUNE" \
  --levels $ROB_LEVELS --count $N_ROB --output "$OUT/e4_robustness.json"

step e5_ablation "$OUT/e5_ablation.json" \
  $PYTHON -W ignore -m $EVAL ablation --scratch "$SCRATCH" --workers "$WORKERS" --retuned "$P1_RETUNE" \
  --pll-tuned "$PLL_TUNED_PILOT" --levels $ABL_LEVELS --count $N_ABL --output "$OUT/e5_ablation.json"
