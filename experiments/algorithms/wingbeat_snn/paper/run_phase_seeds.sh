#!/usr/bin/env bash
# Paper claim C2: raw-IMU phase estimators (GRU, surrogate-gradient SNN, NEAT SNN) x seeds vs default and tuned PLL,
# offline and in closed loop. About 3.5 h for 5 seeds on 20 logical CPUs.
source "$(dirname "$0")/common.sh"
DATA="$RAW/phase_data"
MODELS="$RAW/phase_models"
mkdir -p "$MODELS" "$OUT/e8_models"
GRU_THREADS=$(( WORKERS / 4 > 0 ? WORKERS / 4 : 1 ))
NEAT_WORKERS=$(( WORKERS / 2 > 0 ? WORKERS / 2 : 1 ))

step e8_record "$DATA/summary.json" \
  $PYTHON -W ignore -m experiments.algorithms.wingbeat_snn.phase_data record --scratch "$SCRATCH" \
  --output "$DATA" --workers "$WORKERS" --scale $RECORD_SCALE

for s in $SEEDS; do
  step e8_gru_s$s "$MODELS/gru_s${s}_history.json" \
    $PYTHON -W ignore -m $TRAIN gru --data "$DATA" --output "$MODELS" --steps $GRAD_STEPS --threads $GRU_THREADS \
    --seed $s --tag _s$s &
  step e8_snn_grad_s$s "$MODELS/snn_grad_s${s}_history.json" \
    $PYTHON -W ignore -m $TRAIN snn_grad --data "$DATA" --output "$MODELS" --steps $GRAD_STEPS --threads $GRU_THREADS \
    --hidden 48 --tau-m 0.005 --tau-s 0.0025 --seed $s --tag _s$s &
  step e8_snn_neat_s$s "$MODELS/snn_neat_s${s}_history.json" \
    $PYTHON -W ignore -m $TRAIN snn_neat --data "$DATA" --output "$MODELS" --generations $NEAT_GENS \
    --population $NEAT_POP --workers $NEAT_WORKERS --seed $s --tag _s$s &
  wait
done

step e8_pll_tune "$MODELS/pll_tuned.json" \
  $PYTHON -W ignore -m $TRAIN pll_tune --data "$DATA" --output "$MODELS" --workers "$WORKERS"
step e8_offline "$MODELS/offline_test.json" \
  $PYTHON -W ignore -m $TRAIN offline_test --data "$DATA" --output "$MODELS"

EST=(--estimator "truth=truth" --estimator "pll_tuned=$MODELS/pll_tuned.json")
for s in $SEEDS; do
  EST+=(--estimator "gru_s$s=$MODELS/gru_s$s.pt" --estimator "snn_grad_s$s=$MODELS/snn_grad_s$s.json"
        --estimator "snn_neat_s$s=$MODELS/snn_neat_s$s.json")
done
step e8_closed_loop "$OUT/e8_phase_closed_loop.json" \
  $PYTHON -W ignore -m $EVAL phase_closed_loop --scratch "$SCRATCH" --workers "$WORKERS" --retuned "$P1_RETUNE" \
  --levels $PHASE_LEVELS --count $N_PHASECMP --offset 620000 "${EST[@]}" --output "$OUT/e8_phase_closed_loop.json"

# Small files only (models are a few hundred kB each); the 19 MB recordings stay in runs/.
cp "$DATA/summary.json" "$OUT/e8_models/data_summary.json"
cp "$MODELS"/offline_test.json "$MODELS"/pll_tuned.json "$MODELS"/*_history.json "$MODELS"/gru_s*.pt \
   "$MODELS"/snn_grad_s*.json "$MODELS"/snn_neat_s*.json "$OUT/e8_models/" 2>/dev/null
echo "phase campaign finished $(date -Is)"
