"""Demonstrate video sampling aliasing without changing the frozen flight."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .audit import AuditedAirborne
from .results import portable
from ..neat_imu.train import write_json


def main():
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    from scipy.signal import find_peaks
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((a.root/'test_results.json').read_text())
    expected=next(r for r in frozen['episodes'] if r['policy']=='imu_nominal' and r['seed']==310900000)
    joints=['carrier_2_joint','left_flapping_joint','right_flapping_joint',
            'left_feathering_joint','right_feathering_joint','motor_joint']
    font=ImageFont.truetype('DejaVuSans.ttf',21)
    small=ImageFont.truetype('DejaVuSans.ttf',18)
    path=a.output/'wings_sampling_comparison.mp4'
    samples=[];panels=[];low_image=None;low_time=None;last_frame=None;frame_count=0
    with portable(a.root) as stage:
        env=AuditedAirborne(stage)
        renderer=mujoco.Renderer(env.sim.model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.distance=.31;camera.azimuth=0;camera.elevation=-22
        writer=imageio.get_writer(str(path),fps=25,codec='libx264',quality=8,macro_block_size=None)
        original=env.advance
        def advance(command,parameters):
            nonlocal low_image,low_time,last_frame,frame_count
            original(command,parameters)
            t=env.steps*.002;d=env.sim.data
            samples.append([t,*[float(d.joint(n).qpos[0]) for n in joints],float(d.joint('motor_joint').qvel[0])])
            if 5.-1e-9<=t<=6.+1e-9 and env.steps%2==0:
                camera.lookat[:]=d.xpos[env.env.body]+[0,0,.01]
                renderer.update_scene(d,camera=camera)
                high=Image.fromarray(renderer.render())
                if env.steps%20==0 or low_image is None:low_image=high.copy();low_time=t
                if t<=5.:return
                pic=Image.new('RGB',(1280,720),(16,27,42));pic.paste(high,(0,100));pic.paste(low_image,(640,100))
                draw=ImageDraw.Draw(pic)
                draw.text((20,12),'SAME PID FLIGHT / WING CLOSE-UP / 10x SLOW MOTION',font=font,fill='white')
                draw.text((20,50),'250 samples/s: wing movement is resolved',font=small,fill=(75,220,225))
                draw.text((660,50),'25 samples/s: original sampling, held frames',font=small,fill=(245,170,75))
                draw.text((20,78),f'Simulation time: {t:.3f} s',font=small,fill='white')
                draw.text((660,78),f'Last sampled at: {low_time:.3f} s',font=small,fill='white')
                flap=np.rad2deg(samples[-1][2:4])
                draw.text((20,605),f'Actual flap joint angles: left {flap[0]:+.1f} deg / right {flap[1]:+.1f} deg',font=font,fill='white')
                draw.text((20,645),'Identical dynamics and controller; only video sampling differs. Camera follows the body.',font=small,fill=(180,205,220))
                draw.text((20,678),'Source: first held-out seed 310900000, simulation 5-6 s. No interpolated wing motion.',font=small,fill=(180,205,220))
                last_frame=np.asarray(pic);writer.append_data(last_frame);frame_count+=1
                if frame_count in [3,8,13]:panels.append(last_frame.copy())
        env.advance=advance
        try:
            result=env.rollout(expected['parameters'],310900000,6.,2.,True)
            np.testing.assert_allclose(result['trace'],np.array(expected['trace'])[:3000],rtol=0,atol=1e-12)
            np.testing.assert_allclose(result['audit_trace'],np.array(expected['audit_trace'])[:3000],rtol=0,atol=1e-12)
            for _ in range(25):writer.append_data(last_frame)
            Image.fromarray(panels[0]).save(a.output/'preview.png')
            for i,panel in enumerate(panels):Image.fromarray(panel).save(a.output/f'check_frame_{i}.png')
        finally:writer.close();renderer.close();env.env.close()
    data=np.array(samples);segment=data[data[:,0]>=5.-1e-9]
    frequency=float((segment[-1,1]-segment[0,1])/(2*np.pi*(segment[-1,0]-segment[0,0])))
    wing=[]
    for column,name in [(2,'left'),(3,'right')]:
        peaks,_=find_peaks(segment[:,column],prominence=.1,distance=10)
        peak_frequency=float((len(peaks)-1)/(segment[peaks[-1],0]-segment[peaks[0],0]))
        wing.append(dict(wing=name,flap_peak_frequency_hz=peak_frequency,peak_count=len(peaks),
            min_deg=float(np.rad2deg(segment[:,column].min())),max_deg=float(np.rad2deg(segment[:,column].max())),
            peak_to_peak_deg=float(np.rad2deg(np.ptp(segment[:,column])))))
    np.savez_compressed(a.output/'joint_trace.npz',values=data)
    summary=dict(seed=310900000,source_root=str(a.root),segment_seconds=[5,6],
        columns=['time',*joints,'motor_velocity_rad_s'],carrier_cycles_hz=frequency,wings=wing,
        original_video_fps=25,old_sample_phase_advance_deg=(frequency/25-1)*360,
        apparent_alias_hz=abs(frequency-25),high_sample_hz=250,playback_fps=25,slowdown=10,
        flight_frames=frame_count,hold_frames=25,trace_verified_atol=1e-12,
        policy_sha256=hashlib.sha256((a.root/'selected_pid.json').read_bytes()).hexdigest(),
        scope='Same frozen trajectory; original 25 Hz snapshots held between updates for a paired slow-motion sampling comparison.')
    write_json(a.output/'verification.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
