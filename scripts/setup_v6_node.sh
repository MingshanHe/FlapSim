#!/usr/bin/env bash
# One-step setup of a plan-v6 worker node (Ubuntu or Debian; Python 3.11.15 via uv, as on CloudPC-2). Run from anywhere:
#   bash setup_v6_node.sh            # clone/update ~/flapsim, venv, pinned deps, self-tests, speed test
# Prints a final REPORT block; nothing is trained.
set -euo pipefail
REPO=https://github.com/MingshanHe/FlapSim.git
BRANCH=research/original-env-snn-neat
DIR=${FLAPSIM_DIR:-$HOME/flapsim}

echo "== machine"; nproc; lscpu | grep -E "Model name|Thread\(s\) per core|^Core|Socket" || true; free -g | head -2

SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
need=()
for pkg in g++ git tmux curl; do command -v $pkg >/dev/null || need+=($pkg); done
if [ ${#need[@]} -gt 0 ]; then echo "== apt install ${need[*]}"; $SUDO apt-get update -y && $SUDO apt-get install -y "${need[@]}"; fi
$SUDO apt-get install -y libosmesa6 >/dev/null 2>&1 || echo "(libosmesa6 not available: only needed for video rendering)"

# Python 3.11.15 via uv (same build as CloudPC-2; works on Ubuntu and Debian, no PPA)
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11.15

echo "== code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" fetch -q origin && git -C "$DIR" checkout -q $BRANCH && git -C "$DIR" pull -q --ff-only
else git clone -q -b $BRANCH $REPO "$DIR"; fi
cd "$DIR"; git log --oneline -1

echo "== venv"
[ -d .venv ] || uv venv -q --python 3.11.15 .venv
. .venv/bin/activate
uv pip install -q -e ".[rl,video]" neat-python==2.0.0
uv pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -q mujoco==3.8.0 numpy==2.2.4 gymnasium==1.4.0 imageio==2.38.1 pillow==12.3.0 matplotlib==3.11.2
python -c "import sys, importlib.metadata as m; print(sys.version.split()[0]); [print(p, m.version(p)) for p in ['mujoco','numpy','torch','gymnasium','neat-python']]"

echo "== plan v6 working dir"
mkdir -p runs/spike_steer/v6
cp -n artifacts/spike_steer/plan_v6/baseline.json artifacts/spike_steer/plan_v6/imitate_spec_s*.json runs/spike_steer/v6/

export OMP_NUM_THREADS=1
echo "== selftests"
python -W ignore -m experiments.algorithms.spike_steer.descend8 selftest
python -W ignore -m experiments.algorithms.spike_steer.cache8 | tail -n 3

echo "== speed + exactness (CloudPC-2: 27 s, score 1.288582208870, 42 pulses)"
python -W ignore -c "
import sys, time
from experiments.algorithms.spike_steer import bench8 as E, bench as B
E.load_signs(); B.DASH_M['value'] = .15; B.worker_init(str(E.SOURCE), .15)
c = E.cases8('nod', E.BASES['tune'], 1)[0]
q = dict(go_ref=-.6, go_beats=9, brake_beats=6, brake_ref=.2, gated=True)
t = time.time(); r = E.run_episode8(B._ENV, 'nod', c, E.Classical2(B.law_tuned(), q))
ok = ('%.12f' % r['score']) == '1.288582208870' and r['pulses'] == 42
print('nod episode wall %.1f s, score %.12f, pulses %d -> %s' % (time.time()-t, r['score'], r['pulses'], 'EXACT MATCH' if ok else 'MISMATCH'))
sys.exit(0 if ok else 3)
"
echo "== push check"; git push --dry-run -q origin HEAD 2>&1 && echo "push: OK" || echo "push: NO CREDENTIALS"

# optional: START_SEEDS="1 2 3 5" bash setup_v6_node.sh  -> ES for those seeds in tmux session v6_es, workers split evenly
if [ -n "${START_SEEDS:-}" ]; then
  n=$(echo $START_SEEDS | wc -w); W=$(( $(nproc) / n )); [ $W -lt 4 ] && W=4
  cat > "$HOME/v6_es.sh" <<EOS
cd $DIR && . .venv/bin/activate && export OMP_NUM_THREADS=1
for s in $START_SEEDS; do python -W ignore -m experiments.algorithms.spike_steer.descend8 finetune --seed \$s --workers $W --generations 60 >> runs/spike_steer/v6/finetune_s\$s.log 2>&1 & done
wait; echo "== ES done \$(date)"
EOS
  tmux new-session -d -s v6_es "bash $HOME/v6_es.sh 2>&1 | tee -a $HOME/v6_es.log"
  echo "== ES started: seeds $START_SEEDS, $W workers each (tmux session v6_es)"
fi
echo "== REPORT: setup done on $(hostname), $(nproc) vCPU"
