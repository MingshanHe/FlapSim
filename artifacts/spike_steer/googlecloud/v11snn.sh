# plan v11 exploratory SNN seeds 6-10 on GoogleCloud (approximate simulator). descend12 defaults (v11 prereg).
cd ~/flapsim && . .venv/bin/activate && export OMP_NUM_THREADS=1
M="python -W ignore -m experiments.algorithms.spike_steer.descend12"
for s in 6 7 8 9 10; do $M finetune --seed $s --workers 6 > runs/spike_steer/v11/finetune_s${s}_v11.log 2>&1 & done; wait
echo "== ES done $(date -u)"
for s in 6 7 8 9 10; do
  $M reselect --seed $s --n-select 64 --workers 30
  $M evaluate --seed $s --n-test 64 --test-base 2099100000 --workers 30 --machine GoogleCloud
done
echo "== v11snn done $(date -u)"
