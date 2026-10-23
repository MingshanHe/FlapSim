#!/usr/bin/env bash
# Regenerate everything from the markdown draft + archived data, package, commit and push.
#   bash docs/paper/finalise.sh [commit message]
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export PYTHONPATH=$PWD
TEC=${TECTONIC:-/tmp/claude-0/-/3299ee74-1658-4032-bab6-e72cf1fac322/scratchpad/tectonic/tectonic}
$PY -W ignore docs/paper/make_method_figs.py >/dev/null
$PY -W ignore docs/paper/make_results_figs.py 2>&1 | tail -1
$PY -W ignore docs/paper/export_csv.py 2>&1 | grep -v onnx | tail -2
$PY -W ignore docs/paper/make_pptx.py 2>&1 | grep -v onnx | tail -1
$PY docs/paper/md2tex.py
( cd docs/paper/latex && timeout 600 "$TEC" main.tex 2>&1 | grep -E "^error|Missing character" | head -5 || true )
cp docs/paper/latex/main.pdf docs/paper/draft_method_results.pdf
( cd docs/paper/latex && rm -f ../latex_source.zip && zip -qr ../latex_source.zip main.tex README.md figs )
( cd docs && rm -f /tmp/flapsim_paper_package.zip && zip -qr /tmp/flapsim_paper_package.zip paper -x "paper/handoff/*" "paper/latex_source.zip" "paper/_*" )
ls -la /tmp/flapsim_paper_package.zip | awk '{print "package", $5, "bytes"}'
mkdir -p artifacts/spike_steer/plan_v14/rows && cp runs/spike_steer/v14/*mlp_ppo*.json artifacts/spike_steer/plan_v14/rows/ 2>/dev/null || true
git add docs/paper artifacts/spike_steer/plan_v14
git commit -q -m "${1:-paper: regenerated package}

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XJCwSxuAndV8x1hbQ9Lmq2" || true
git push -q origin research/original-env-snn-neat && echo pushed
