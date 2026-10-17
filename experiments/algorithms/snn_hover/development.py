"""Predeclared development-only physical envelope checks; no test-set access."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from .common import evaluate,write
from .environment import case

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'development';out.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for angle in [5,10,15,20,30,45]:
        for direction in [0,90,180,270]:
            c=case(409900100+len(tasks),8.,'nominal',len(tasks)%8);c.update(tilt_deg=angle,direction_deg=direction)
            tasks.append(c)
    for strength in [.000016,.000032,.000064]:
        for direction in [0,90,180,270]:
            c=case(409900100+len(tasks),10.,'nominal',len(tasks)%8)
            c.update(impulse_torque=strength,impulse_start=5.,impulse_duration=.1,direction_deg=direction);tasks.append(c)
    write(out/'protocol.json',dict(status='frozen_before_runs',controller='unchanged_PID',cases=tasks))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(evaluate,(None,c,'pid','height',False)) for c in tasks]
        for f in as_completed(futures):
            r=f.result();rows.append(r);write(out/'pid_envelope.json',rows)
            print(json.dumps(dict(completed=len(rows),total=len(tasks),tilt=r['maximum_tilt_deg'],elapsed=r['elapsed'],success=r['success'])),flush=True)
    write(out/'results.json',dict(status='completed',episodes=len(rows),actual_actor_steps=sum(r['steps'] for r in rows),wall_seconds=time.monotonic()-start))
if __name__=='__main__':main()
