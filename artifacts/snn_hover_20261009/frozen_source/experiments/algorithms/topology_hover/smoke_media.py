"""Two-second NON-held-out check of trajectory plots and the complete video path."""
import argparse
import json
import subprocess
from pathlib import Path

from .common import write
from .environment import case
from .evaluate import trace_task,save_trace
from .render import plot_trace,render


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    representatives={}
    for method in ['pid','neat','rnn_ppo']:
        entry=dict(mode='pid' if method=='pid' else 'residual',sensor='height')
        policy=args.campaign/'pilot'/f'{method}_seed1201'/'best_policy.json'
        spec=None if method=='pid' else json.loads(policy.read_text())
        if spec is not None:representatives[method]=dict(path=str(policy.resolve()))
        result=trace_task((entry,spec,case(339900223,2.,'nominal',0)))
        save_trace(root,f'{method}_nominal',result)
    write(root/'representatives.json',representatives)
    for method in ['pid','neat','rnn_ppo']:plot_trace(root,f'{method}_nominal')
    render(root,'nominal')
    media={}
    for name,seconds,fps in [('comparison_nominal.mp4',3.,60.),('neat_wings_nominal_10x.mp4',10.,25.)]:
        path=root/'videos'/name
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration',
            '-of','json',str(path)],check=True,capture_output=True,text=True)
        data=json.loads(probe.stdout);stream=data['streams'][0]
        numerator,denominator=map(float,stream['avg_frame_rate'].split('/'))
        assert numerator/denominator==fps
        assert abs(float(data['format']['duration'])-seconds)<.05
        media[name]=data
    recorded=json.loads((root/'videos/verification_nominal.json').read_text())
    assert all(value['peak_to_peak']>10 for value in recorded['wing_joint_ranges_deg'].values())
    write(root/'verification.json',dict(status='passed',
        scope='Media pipeline check on non-held-out seed 339900223; not evidence for algorithm performance.',
        simulation_seconds=2.,media=media,rendering=recorded,
        wing_joint_ranges_deg=recorded['wing_joint_ranges_deg']))
    print(json.dumps(dict(status='passed',output=str(root),wing_joint_ranges_deg=recorded['wing_joint_ranges_deg'])))


if __name__=='__main__':
    main()
