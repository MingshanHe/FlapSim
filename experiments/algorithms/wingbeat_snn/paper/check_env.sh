#!/usr/bin/env bash
# E0: verify a new machine before the long runs (about 2 min).
source "$(dirname "$0")/common.sh"
fail=0
command -v g++ > /dev/null || { echo "missing g++ (native physics is compiled at first use)"; fail=1; }
command -v tmux > /dev/null || echo "note: tmux not found; long runs should be detached some other way"
$PYTHON - <<'PY' || fail=1
import sys, importlib
need = {'mujoco': '3.8.0', 'numpy': None, 'torch': None, 'neat': None, 'gymnasium': None}
for name, version in need.items():
    m = importlib.import_module(name)
    v = getattr(m, '__version__', '?')
    flag = '' if version is None or v == version else f'   <-- expected {version}'
    print(f'{name:10s} {v}{flag}')
import flapsim
print('flapsim   ', flapsim.__file__)
print('python    ', sys.version.split()[0])
PY
$PYTHON -m pytest -q -p no:cacheprovider tests/test_wingbeat_snn.py || fail=1
[ $fail = 0 ] && echo "environment OK" || { echo "environment check FAILED"; exit 1; }
