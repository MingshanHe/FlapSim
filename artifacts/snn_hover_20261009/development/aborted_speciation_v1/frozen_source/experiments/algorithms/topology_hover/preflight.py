import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from .environment import HoverEnv,Zero,case,rollout

def run(item):
    sensor,c=item;env=HoverEnv(mode='pid',sensor_mode=sensor)
    return dict(sensor=sensor,**rollout(env,Zero(),c))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);tasks=[]
    for sensor,conditions,n,seconds in [('height',['nominal','stress'],8,20.),('truth',['nominal','stress'],2,20.),('imu',['nominal','stress'],2,20.)]:
        for condition in conditions:
            for i in range(n):tasks.append((sensor,case(320000000+i,seconds,condition,i)))
    (a.root/'preflight_protocol.json').write_text(json.dumps(tasks,indent=2))
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(run,x) for x in tasks]
        for f in as_completed(futures):
            rows.append(f.result());(a.root/'preflight.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-started,episodes=rows),indent=2))
            print(json.dumps(dict(completed=len(rows),total=len(tasks),last=rows[-1])),flush=True)

if __name__=='__main__':main()
