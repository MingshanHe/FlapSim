#!/usr/bin/env bash
cd /root/flapsim && . .venv/bin/activate
for S in 1 3 4; do
  W=30 GENS=60 HOURS=12 PY=python bash artifacts/spike_steer/v4/handoff_s3/run_r3_seed.sh $S > logs/r3_s$S.log 2>&1
  echo "seed $S exit $?" >> logs/r3_queue.done
done
