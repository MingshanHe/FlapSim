#!/usr/bin/env bash
# Whole campaign in the recommended order; every step is skipped if its output already exists, so the
# script can simply be restarted after an interruption. Run it detached:
#   tmux new-session -d -s wingbeat_paper "PYTHON=$(which python) bash experiments/algorithms/wingbeat_snn/paper/run_all.sh 2>&1 | tee -a runs/wingbeat_paper.log"
DIR="$(dirname "$0")"
bash "$DIR/check_env.sh" || exit 1
bash "$DIR/run_core.sh"
bash "$DIR/run_phase_seeds.sh"
bash "$DIR/run_residual_seeds.sh"
bash "$DIR/run_h5_seeds.sh"
source "$DIR/common.sh"
$PYTHON -W ignore -m experiments.algorithms.wingbeat_snn.paper_stats --input "$OUT" --output "$OUT/report"
echo "campaign finished $(date -Is)"
