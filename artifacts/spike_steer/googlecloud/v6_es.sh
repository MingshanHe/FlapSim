cd ~/flapsim && . .venv/bin/activate && export OMP_NUM_THREADS=1
for s in 1 2 3 5; do python -W ignore -m experiments.algorithms.spike_steer.descend8 finetune --seed $s --workers 8 --generations 60 >> runs/spike_steer/v6/finetune_s$s.log 2>&1 & done
wait; echo "== ES done $(date)"
