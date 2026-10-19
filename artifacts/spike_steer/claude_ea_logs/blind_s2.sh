cd /root/flapsim && . .venv/bin/activate
SPEC=runs/spike_steer/bench_r3_local/train/finetune_s2/spec_val_g9.json
python -W ignore -m experiments.algorithms.spike_steer.descend_r3 evaluate --output runs/spike_steer/bench_r3_local --seed 2 --spec $SPEC --tag r3_s2 --variants intact --workers 2 > logs/blind_r3_s2.log 2>&1
python -W ignore -m experiments.algorithms.spike_steer.descend_r3 evaluate --output runs/spike_steer/bench_r3_local --seed 2 --spec $SPEC --tag r3_s2_abl --variants clamp_fast,nodelay_all,noref,nopulse --workers 2 > logs/blind_r3_s2_abl.log 2>&1
