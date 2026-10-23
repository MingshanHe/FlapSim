# plan v7 full, extra SNN+ES seeds 6-10 on GoogleCloud (approximate simulator; exploratory extension of the
# preregistered seeds 1-5). Same settings as v7 full: imitation (descend9 train, 40 epochs), then careful ES
# (lr 0.01, sigma 0.02, pop 16, 4 cases/task/gen, 20 gens, every centre saved), selection 32/task, blind 32/task on
# base 2_098_300_000. Re-running this script resumes ES from checkpoint.npz.
cd ~/flapsim && . .venv/bin/activate && export OMP_NUM_THREADS=1
OUT=runs/spike_steer/v7ext
SEEDS="6 7 8 9 10"
mkdir -p $OUT
cp -n artifacts/spike_steer/plan_v7/{baseline.json,collect_train.npz,collect_val.npz} $OUT/
for s in $SEEDS; do
  [ -f $OUT/imitate_spec_s$s.json ] || python -W ignore -m experiments.algorithms.spike_steer.descend9 train --output $OUT --seed $s --threads 6 > $OUT/train_s$s.log 2>&1 &
done; wait
echo "== imitation done $(date -u)"
for s in $SEEDS; do
  python -W ignore -m experiments.algorithms.spike_steer.descend9b finetune --output $OUT --seed $s --tag _full --lr 0.01 --sigma 0.02 --pop 16 --cases-per-task 4 --save-every 1 --generations 20 --workers 6 > $OUT/finetune_s${s}_full.log 2>&1 &
done; wait
echo "== ES done $(date -u)"
for s in $SEEDS; do
  python -W ignore -m experiments.algorithms.spike_steer.descend9b reselect --output $OUT --seed $s --tag _full --n-select 32 --workers 30
  python -W ignore -m experiments.algorithms.spike_steer.descend9b evaluate --output $OUT --seed $s --tag _full --n-test 32 --test-base 2098300000 --workers 30 --machine GoogleCloud
done
echo "== v7ext done $(date -u)"
