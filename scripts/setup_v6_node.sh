#!/usr/bin/env bash
# One-step setup of a plan-v6 worker node (Ubuntu 22.04/24.04). Run from anywhere:
#   bash setup_v6_node.sh            # clone/update ~/flapsim, venv, pinned deps, self-tests, speed test
# Prints a final REPORT block; nothing is trained.
set -euo pipefail
REPO=https://github.com/MingshanHe/FlapSim.git
BRANCH=research/original-env-snn-neat
DIR=${FLAPSIM_DIR:-$HOME/flapsim}

echo "== machine"; nproc; lscpu | grep -E "Model name|Thread\(s\) per core|^Core|Socket" || true; free -g | head -2

need=()
command -v python3.11 >/dev/null || need+=(python3.11 python3.11-venv python3.11-dev)
command -v g++ >/dev/null || need+=(g++)
command -v git >/dev/null || need+=(git)
command -v tmux >/dev/null || need+=(tmux)
dpkg -s libosmesa6 >/dev/null 2>&1 || need+=(libosmesa6)
if [ ${#need[@]} -gt 0 ]; then
  echo "== apt install ${need[*]}"
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  if ! apt-cache policy python3.11 | grep -q Candidate || apt-cache policy python3.11 | grep -q "Candidate: (none)"; then
    $SUDO apt-get update -y && $SUDO apt-get install -y software-properties-common && $SUDO add-apt-repository -y ppa:deadsnakes/ppa
  fi
  $SUDO apt-get update -y && $SUDO apt-get install -y "${need[@]}"
fi

echo "== code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" fetch -q origin && git -C "$DIR" checkout -q $BRANCH && git -C "$DIR" pull -q --ff-only
else git clone -q -b $BRANCH $REPO "$DIR"; fi
cd "$DIR"; git log --oneline -1

echo "== venv"
[ -d .venv ] || python3.11 -m venv .venv
. .venv/bin/activate
pip install -q -U pip
pip install -q -e ".[rl,video]" neat-python==2.0.0
pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q mujoco==3.8.0 numpy==2.2.4 gymnasium==1.4.0 imageio==2.38.1 pillow==12.3.0 matplotlib==3.11.2
python -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ['mujoco','numpy','torch','gymnasium','neat-python']]"

echo "== plan v6 working dir"
mkdir -p runs/spike_steer/v6
cp -n artifacts/spike_steer/plan_v6/baseline.json artifacts/spike_steer/plan_v6/imitate_spec_s*.json runs/spike_steer/v6/

export OMP_NUM_THREADS=1
echo "== selftests"
python -W ignore -m experiments.algorithms.spike_steer.descend8 selftest
python -W ignore -m experiments.algorithms.spike_steer.cache8 | tail -n 3

echo "== speed (CloudPC-2: 27 s)"
python -W ignore -c "
import time
from experiments.algorithms.spike_steer import bench8 as E, bench as B
E.load_signs(); B.DASH_M['value'] = .15; B.worker_init(str(E.SOURCE), .15)
c = E.cases8('nod', E.BASES['tune'], 1)[0]
q = dict(go_ref=-.6, go_beats=9, brake_beats=6, brake_ref=.2, gated=True)
t = time.time(); r = E.run_episode8(B._ENV, 'nod', c, E.Classical2(B.law_tuned(), q))
print('nod episode wall %.1f s, score %.12f pulses %d (CloudPC-2: 1.288582208870, 42 pulses; must match)' % (time.time()-t, r['score'], r['pulses']))
"
echo "== push check"; git push --dry-run -q origin HEAD 2>&1 && echo "push: OK" || echo "push: NO CREDENTIALS"
echo "== REPORT: setup done on $(hostname), $(nproc) vCPU"
