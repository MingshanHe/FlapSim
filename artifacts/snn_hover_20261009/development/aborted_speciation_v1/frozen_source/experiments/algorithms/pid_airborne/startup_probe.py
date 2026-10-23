"""Development test of a gradual motor start from a motionless mechanism."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .initialization import init_worker,evaluate
from ..neat_imu.train import write_json
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    p=json.loads((a.root/'generation_003.json').read_text())[0]['parameters']
    specs=[dict(p,release='rest',motor_start_s=t) for t in [.05,.1,.2,.3]]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,[(p,[310100000,310100001],8.,1.) for p in specs]))
    write_json(a.root/'startup_probe.json',rows);print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
