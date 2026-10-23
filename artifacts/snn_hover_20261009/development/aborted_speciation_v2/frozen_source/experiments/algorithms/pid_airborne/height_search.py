"""Tune height PID after the 100 ms startup resolves the difficult release."""
import argparse,itertools,json,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .initialization import init_worker,evaluate
from ..neat_imu.train import write_json
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    base=json.loads((a.root/'generation_003.json').read_text())[0]['parameters']
    specs=[dict(base,release='rest',motor_start_s=.1,motor=3190.,height_gains=[kp,kd,ki])
           for kp,kd,ki in itertools.product([250.,500.,900.],[150.,300.,500.],[80.,200.])]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=8,initializer=init_worker,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,[(p,[310100000,310100001],12.,2.) for p in specs]))
    # Evaluate the full recorded trajectory statistics, not just time survived.
    for r in rows:
        r['height_selection_score']=float(np.mean([e['elapsed']-.5*e['tilt_rms_deg']-10*e['height_rmse_m'] for e in r['episodes']]))
    rows.sort(key=lambda r:r['height_selection_score'],reverse=True)
    write_json(a.root/'height_search.json',dict(wall_seconds=time.monotonic()-started,rows=rows))
    print(json.dumps(rows[:3]),flush=True)
if __name__=='__main__':main()
