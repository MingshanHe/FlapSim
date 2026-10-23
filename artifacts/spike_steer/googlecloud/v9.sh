# plan v9 exploratory seeds 6-10 on GoogleCloud (approximate; not counted in M4). Imitation starts reused from v7ext.
cd ~/flapsim && . .venv/bin/activate && export OMP_NUM_THREADS=1
for s in 6 7 8 9 10; do
  python -W ignore -m experiments.algorithms.spike_steer.descend10 finetune --seed $s --tag _careful --lr 0.01 --sigma 0.02 --save-every 1 --generations 20 --workers 6 > runs/spike_steer/v9/finetune_s${s}_careful.log 2>&1 &
done; wait
echo "== ES done $(date -u)"
for s in 6 7 8 9 10; do
  python -W ignore -m experiments.algorithms.spike_steer.descend10 reselect --seed $s --tag _careful --n-select 32 --workers 30
  python -W ignore -m experiments.algorithms.spike_steer.descend10 evaluate --seed $s --tag _careful --n-test 64 --test-base 2098700000 --workers 30 --machine GoogleCloud
done
echo "== v9 done $(date -u)"
