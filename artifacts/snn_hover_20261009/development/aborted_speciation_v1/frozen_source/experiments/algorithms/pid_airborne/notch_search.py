"""Independent development screen of notch filtering and velocity feedback."""
import itertools,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .advanced import init_worker,evaluate
from ..neat_imu.train import write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();started=time.monotonic();specs=[]
    for kp,kd,motor,kv in itertools.product([-.5,.1,.5],[.01,.1,.5],[3200,3461],[0,.1]):
        specs.append(dict(kp=[.3,kp],kd=[.02,kd],motor=motor,velocity_gain=[kv,kv],sensor_tau=.001,action_tau=.004))
    with ProcessPoolExecutor(max_workers=6,initializer=init_worker,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,[(p,[310100000,310100001,310100002,310100003],3.,2.) for p in specs]))
    rows.sort(key=lambda r:r['score'],reverse=True)
    write_json(a.root/'notch_screen.json',rows)
    print(json.dumps(dict(wall_seconds=time.monotonic()-started,best=rows[:3])),flush=True)
if __name__=='__main__':main()
