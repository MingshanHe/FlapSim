"""Development-only long-horizon diagnostic; truth height is explicitly labeled."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .core import init_worker,evaluate
from ..neat_imu.train import write_json
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    p=json.loads((a.root/'generation_003.json').read_text())[0]['parameters']
    specs=[dict(p,height_source=source) for source in ['imu','truth']]
    with ProcessPoolExecutor(max_workers=2,initializer=init_worker,initargs=(str(a.root),)) as pool:
        rows=list(pool.map(evaluate,[(p,[310100000,310100002],30.,1.) for p in specs]))
    write_json(a.root/'long_development_probe.json',rows);print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
