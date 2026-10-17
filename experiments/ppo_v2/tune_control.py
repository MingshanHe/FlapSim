"""Fit stabilizer gains against full-physics diagnostic episodes."""
import contextlib
import io
import json
from pathlib import Path
from scipy.optimize import differential_evolution
from .probe_control import trial


def objective(gains):
    with contextlib.redirect_stdout(io.StringIO()):
        r=trial(gains,seconds=2.)
    return 20*(2-r['seconds'])+r['rmse']


def main():
    result=differential_evolution(objective,[(-5,5),(-.8,.8),(-5,5),(-.8,.8)],
                    maxiter=12,popsize=5,workers=10,updating='deferred',
                    polish=False,seed=7,disp=True)
    data={'gains':result.x.tolist(),'cost':float(result.fun)}
    Path('runs/tuned_gains.json').write_text(json.dumps(data,indent=2))
    print(data,flush=True)
    trial(result.x.tolist(),seconds=8.,output='runs/tuned_baseline.csv')


if __name__=='__main__':
    main()
