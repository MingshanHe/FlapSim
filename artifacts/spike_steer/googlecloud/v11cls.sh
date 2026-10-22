# plan v11 amendment 1: classical upper-bound arm, seeds 3-4 on GoogleCloud (approximate)
cd ~/flapsim && . .venv/bin/activate && export OMP_NUM_THREADS=1
M="python -W ignore -m experiments.algorithms.spike_steer.descend_cls"
for s in 3 4; do $M finetune --seed $s --workers 15 > runs/spike_steer/v11/cls_es_s$s.log 2>&1 & done; wait
echo "== cls ES done $(date -u)"
for s in 3 4; do $M reselect --seed $s --workers 30; $M evaluate --seed $s --workers 30 --machine GoogleCloud; done
echo "== v11cls done $(date -u)"
