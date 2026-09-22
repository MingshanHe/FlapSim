"""Render the calibrated rectangular-wing run, checking against saved motion/lift."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import findfont
from rect_material_experiment import config_for, OUTPUT
from rect_simulation import ProfileBladeElement
from flapsim import Simulator


def render(hz=15):
    result = json.loads((OUTPUT/f'simulation/rect/{hz}hz/result.json').read_text())
    calibration = json.loads((OUTPUT/f'simulation/rect/{hz}hz/calibration.json').read_text())
    command = calibration[-1]['command_rad_s']
    saved = np.load(OUTPUT/result['source'])
    config = config_for()
    assert hashlib.sha256(config.path.read_bytes()).hexdigest() == result['model_sha256']
    assert hashlib.sha256((OUTPUT/'rect_profile.npz').read_bytes()).hexdigest() == result['profile_sha256']
    sim = Simulator(config, fluid=ProfileBladeElement(config.fluid, OUTPUT/'rect_profile.npz'))
    folder = OUTPUT/'video'
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype(findfont('DejaVu Sans'), 20)
    small = ImageFont.truetype(findfont('DejaVu Sans'), 17)
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    cameras = []
    for lookat, distance, azimuth, elevation in (
        ((0, 0, .545), .26, 135, -20),
        ((0, -.005, .555), .19, 70, -50),
    ):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        cameras.append(camera)
    paths = {mode: folder/f'rectangular_{hz}hz_{mode}.mp4' for mode in ('realtime','slow8x')}
    writers = {mode: imageio.get_writer(str(path), fps=50 if mode=='realtime' else 25,
                                        codec='libx264', quality=8, pixelformat='yuv420p',
                                        ffmpeg_params=['-movflags','+faststart'])
               for mode, path in paths.items()}
    max_angle_error = max_lift_error = 0.
    frame_count = dict(realtime=0, slow8x=0)
    try:
        for i in range(4001):
            if i:
                phase = min((i-1)/2000/.15, 1)
                sim.set_speed(command*(.5-.5*np.cos(np.pi*phase)))
                sim.step(100)
            # Exact saved state matching; rendering never prescribes joint motion.
            angles = [sim.data.joint(j).qpos[0] for j in ('motor_joint','left_feathering_joint','left_flapping_joint')]
            expected = [saved[k][i] for k in ('motor_joint_angle_rad','pitch_rad','flap_rad')]
            max_angle_error = max(max_angle_error, float(np.max(np.abs(np.array(angles)-expected))))
            if i % 10:
                continue
            aero = sim.fluid.forces(sim.data)[:,2].sum()
            max_lift_error = max(max_lift_error, abs(float(aero-saved['aero_Fz_N'][i])))
            panels=[]
            for camera in cameras:
                renderer.update_scene(sim.data,camera=camera)
                panels.append(renderer.render().copy())
            base = Image.new('RGB',(1280,592),(17,24,39))
            base.paste(Image.fromarray(np.concatenate(panels,axis=1)),(0,112))
            for mode,writer in writers.items():
                if mode=='realtime' and i%40:
                    continue
                frame=base.copy();draw=ImageDraw.Draw(frame)
                rate='REAL TIME (1x)' if mode=='realtime' else 'SLOW MOTION (8x)'
                draw.text((18,9),f'Rectangular wing | {hz} Hz test | {rate} | simulation t = {sim.data.time:.3f} s',font=font,fill='white')
                draw.text((18,39),f'Flapping {np.degrees(angles[2]):+6.1f} deg | Passive pitch {np.degrees(angles[1]):+6.1f} deg | Lift {aero*1000:+6.2f} mN',font=small,fill=(160,215,255))
                draw.text((18,67),f'Single left wing | PET frame + 10 um Kapton | measured {result["measured_wing_frequency_hz"]:.3f} Hz | mean lift {result["mean_lift_mN"]:.2f} mN',font=small,fill=(190,199,214))
                draw.text((18,121),'Assembly',font=small,fill='white')
                draw.text((658,121),'Wing mechanism / passive feathering',font=small,fill='white')
                writer.append_data(np.asarray(frame));frame_count[mode]+=1
                if mode=='slow8x' and i==2000:
                    frame.save(folder/f'rectangular_{hz}hz_preview.png')
            if i%1000==0:
                print(f'Rendered t={sim.data.time:.1f}s / 2.0s',flush=True)
    finally:
        for writer in writers.values(): writer.close()
        renderer.close()
    assert max_angle_error < 1e-8, max_angle_error
    assert max_lift_error < 1e-10, max_lift_error
    assert not np.any(sim.data.warning.number)
    metadata=dict(hz=hz, simulation_duration_s=float(sim.data.time), render_sample_hz=200,
                  playback_fps=dict(realtime=50,slow8x=25),frame_count=frame_count,
                  max_saved_angle_error_rad=max_angle_error,max_saved_lift_error_N=max_lift_error,
                  source=result['source'],model_sha256=result['model_sha256'],profile_sha256=result['profile_sha256'],
                  files={k:str(p.relative_to(OUTPUT)) for k,p in paths.items()})
    (folder/f'rectangular_{hz}hz_video.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hz',type=int,choices=range(10,16),default=15)
    args=parser.parse_args()
    render(args.hz)
