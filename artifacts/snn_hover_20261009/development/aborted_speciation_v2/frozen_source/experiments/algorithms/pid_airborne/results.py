"""Plot recorded traces and replay the predeclared episode without selecting a win."""
import argparse,hashlib,json,shutil,tempfile
from contextlib import contextmanager
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from flapsim import load_config
from flapsim.rl.env import PRESET
from .audit import AuditedAirborne
from ..neat_imu.train import write_json


@contextmanager
def portable(root):
    """Relocate only mesh paths; never change the published model file."""
    root=Path(root)
    with tempfile.TemporaryDirectory(prefix='flapsim_pid_replay_') as temp:
        stage=Path(temp);(stage/'calibration').mkdir()
        for relative in ['phase_bank.npz','calibration/trim.json']:
            shutil.copyfile(root/relative,stage/relative)
        tree=ET.parse(root/'robot_upper7_lower8.xml')
        assets=(load_config(PRESET).path.parent/'../meshes').resolve()
        tree.getroot().find('compiler').set('meshdir',str(assets))
        tree.write(stage/'robot_upper7_lower8.xml',encoding='utf-8',xml_declaration=True)
        yield stage


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((root/'test_results.json').read_text())
    row=next(r for r in data['episodes'] if r['policy']=='imu_nominal' and r['seed']==310900000)
    a=np.array(row['trace']);audit=np.array(row['audit_trace']);t=a[:,0]
    start_z=a[:,3]-audit[:,3]
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,constrained_layout=True)
    ax=axes[0,0];ax.plot(t,a[:,3],label='True height');ax.plot(t,start_z+a[:,15],label='IMU estimate',alpha=.8)
    ax.axhline(start_z[0],color='gray',ls='--',label='Release height');ax.set(ylabel='Height (m)');ax.legend()
    ax=axes[0,1];ax.plot(t,np.rad2deg(a[:,7]),label='Roll');ax.plot(t,np.rad2deg(a[:,8]),label='Pitch');ax.set(ylabel='Angle (deg)');ax.legend()
    ax=axes[1,0];ax.plot(t,audit[:,1]);ax.axhline(0,color='red');ax.set(ylabel='Conservative clearance (m)')
    ax=axes[1,1];ax.plot(t,a[:,12]);ax.set(ylabel='Motor target (rad/s)')
    ax=axes[2,0];ax.plot(t,np.rad2deg(a[:,10]),label='Alpha');ax.plot(t,np.rad2deg(a[:,11]),label='Beta');ax.set(ylabel='Gimbal target (deg)',xlabel='Time (s)');ax.legend()
    ax=axes[2,1];ax.plot(t,a[:,1]-a[0,1],label='X');ax.plot(t,a[:,2]-a[0,2],label='Y');ax.set(ylabel='Horizontal displacement (m)',xlabel='Time (s)');ax.legend()
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('PID, IMU only — first held-out nominal seed 310900000\n7 g + 8 g; yaw and horizontal position are uncontrolled')
    fig.savefig(root/'pid_hover_trace.png',dpi=160);plt.close(fig)
    np.savetxt(root/'nominal_first_trace.csv',a,delimiter=',',comments='',header=
        't,x,y,z,vx,vy,vz,roll,pitch,yaw,alpha_target,beta_target,motor_target,imu_x,imu_y,imu_z,filtered_roll,filtered_pitch,filtered_yaw,alpha_actual,beta_actual')


def video(root,label,destination):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image,ImageDraw,ImageFont
    record=json.loads((root/'test_results.json').read_text())
    expected=next(r for r in record['episodes'] if r['policy']==label and r['seed']==310900000)
    p=expected['parameters'];font=ImageFont.truetype('DejaVuSans.ttf',18)
    small=ImageFont.truetype('DejaVuSans.ttf',15);large=ImageFont.truetype('DejaVuSans.ttf',24)
    destination.parent.mkdir(parents=True,exist_ok=True)
    original_hash=hashlib.sha256((root/'robot_upper7_lower8.xml').read_bytes()).hexdigest()
    with portable(root) as stage:
        env=AuditedAirborne(stage);renderer=mujoco.Renderer(env.sim.model,height=720,width=940)
        camera=mujoco.MjvCamera();camera.distance=1.3;camera.azimuth=135;camera.elevation=-15
        writer=imageio.get_writer(str(destination),fps=25,codec='libx264',quality=8,macro_block_size=None)
        height_path=[];estimate_path=[];tilt_path=[];last_frame=None;frames=0
        def frame():
            nonlocal last_frame,frames
            position=env.sim.data.xpos[env.env.body]
            camera.lookat[:]=[position[0],position[1],.48]
            renderer.update_scene(env.sim.data,camera=camera)
            picture=Image.new('RGB',(1280,720),(16,27,42));picture.paste(Image.fromarray(renderer.render()),(0,0))
            d=ImageDraw.Draw(picture);t=env.steps*.002
            tilt=np.rad2deg(np.arccos(np.clip(env.sim.data.xmat[env.env.body,8],-1,1)))
            d.rectangle((0,0,940,75),fill=(16,27,42));d.text((20,12),'PID HOVER / 7 g + 8 g / YAW FREE',font=large,fill='white')
            d.text((20,45),f'{label} | seed 310900000 | t = {t:.2f} s | playback 1x',font=font,fill=(180,205,220))
            d.text((960,24),f'Height  {position[2]:.3f} m',font=font,fill='white')
            d.text((960,55),f'Tilt     {tilt:.1f} deg',font=font,fill='white')
            d.text((960,86),f'Motor   {env.action[2]:.0f} rad/s',font=font,fill='white')
            d.text((960,117),f'Clearance > {env.minimum_clearance:.3f} m',font=small,fill='white')
            def point(time,value,y0,scale):return (970+min(time/20,1)*285,y0-value*scale)
            height_path.append(point(t,position[2],370,140))
            estimate_path.append(point(t,env.start[2]+env.frontend.imu.position[2],370,140))
            tilt_path.append(point(t,min(tilt,80),590,2.6))
            d.text((965,175),'Height vs time (0-20 s)',font=small,fill='white')
            for z in [0,.4,.8,1.2]:
                y=370-z*140;d.line((970,y,1255,y),fill=(58,70,80));d.text((965,y-17),f'{z:.1f} m',font=small,fill=(150,170,185))
            if len(height_path)>1:
                d.line(height_path,fill=(65,215,225),width=2);d.line(estimate_path,fill=(245,165,75),width=2)
                d.line(tilt_path,fill=(150,210,115),width=2)
            d.text((965,385),'Cyan: true / orange: IMU',font=small,fill=(180,205,220))
            d.text((965,425),'Tilt vs time (0-20 s)',font=small,fill='white')
            for deg in [0,20,40]:
                y=590-deg*2.6;d.line((970,y,1255,y),fill=(58,70,80));d.text((965,y-17),f'{deg} deg',font=small,fill=(150,170,185))
            offset=position-env.start
            d.text((965,618),f'X/Y drift {offset[0]:+.2f}/{offset[1]:+.2f} m',font=small,fill='white')
            d.text((965,648),'Camera follows X/Y only.',font=small,fill=(180,205,220))
            d.text((965,674),'Ground height stays fixed.',font=small,fill=(180,205,220))
            last_frame=np.asarray(picture);writer.append_data(last_frame);frames+=1
        original_advance=env.advance
        def capture(command,parameters):
            original_advance(command,parameters)
            if env.steps%20==0:frame()
        env.advance=capture
        try:
            result=env.rollout(p,310900000,20.,2.,True)
            np.testing.assert_allclose(result['trace'],expected['trace'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(result['audit_trace'],expected['audit_trace'],rtol=0,atol=1e-12)
            if env.steps%20:frame()
            for _ in range(25):writer.append_data(last_frame)
            Image.fromarray(last_frame).save(destination.with_suffix('.png'))
            write_json(destination.with_suffix('.json'),dict(label=label,seed=310900000,elapsed=result['elapsed'],
                trace_verified_atol=1e-12,portable_mesh_relocation=True,model_sha256=original_hash,
                fps=25,slowdown=1,hold_seconds=1,flight_frames=frames,
                flapping_motion_note='25 fps undersamples the wing beat; flight state and timing are accurate.',
                summary={k:v for k,v in result.items() if k not in ['trace','audit_trace']}))
        finally:writer.close();renderer.close();env.env.close()
    assert hashlib.sha256((root/'robot_upper7_lower8.xml').read_bytes()).hexdigest()==original_hash
    return destination


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--video',type=Path);p.add_argument('--label',default='imu_nominal',choices=['imu_nominal','truth_height_diagnostic']);a=p.parse_args()
    if a.video:print(video(a.root,a.label,a.video))
    else:plot(a.root)
if __name__=='__main__':main()
