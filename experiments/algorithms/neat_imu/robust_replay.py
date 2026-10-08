"""Replay a predeclared nominal or disturbed held-out episode."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .robust import RobustEnv
from .residual import ResidualPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--group', choices=('nominal', 'stress'), default='nominal')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--video', type=Path)
    p.add_argument('--slowdown', type=int, choices=(1, 2, 4), default=4)
    args = p.parse_args()
    metadata = json.loads((args.run/'run.json').read_text())
    if metadata.get('physics_model_sha256'):
        model_path=Path(metadata['physics_model_path'])
        if hashlib.sha256(model_path.read_bytes()).hexdigest()!=metadata['physics_model_sha256']:
            raise ValueError('physics model changed since training; restore the recorded model before replay')
    case = metadata['test_'+args.group][args.index]
    signed = metadata['arguments'].get('variant') == 'gyro6'
    if signed:
        from .activation import SignedHoverEnv, MappedPolicy
        if 'initialization' in metadata['environment']:
            from .hover_initial import HoverInitialEnv as SignedHoverEnv
        if 'model_path' in metadata['environment']:
            from .mass_model import MassVariantEnv as SignedHoverEnv
    env_type = SignedHoverEnv if signed else RobustEnv
    env = env_type(**metadata['environment'], render_mode='rgb_array' if args.video else None)
    env.configure_case(case)
    policy = MappedPolicy(args.run/'best_policy.json', metadata['action_map']) if signed else ResidualPolicy(args.run/'best_policy.json')
    writer = None
    try:
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec='libx264')

        def frame(info):
            picture = Image.fromarray(env.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, picture.width, 65), fill=(20, 28, 40))
            draw.text((10, 8), f'NEAT | {args.group} test | seed {case["seed"]} | {args.slowdown}x slow motion', fill='white')
            draw.text((10, 27), f'Simulation {info.get("elapsed",0):.2f} s | position RMSE {100*info.get("tracking_rmse",0):.1f} cm', fill='white')
            draw.text((10, 46), f'Status: {info.get("termination_reason") or "flying"} | target: {case["seconds"]:g} s and RMSE < 10 cm', fill='white')
            return np.asarray(picture)

        obs, info = env.reset(seed=case['seed'])
        policy.reset(obs)
        tilt_squared = 0.
        if writer:
            writer.append_data(frame(info))
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            tilt = math.acos(float(np.clip(env.sim.data.xmat[env.body, 8], -1, 1)))
            tilt_squared += tilt**2
            if writer and (env.steps % (4//args.slowdown) == 0 or terminated or truncated):
                writer.append_data(frame(info))
            if terminated or truncated:
                break
        result = dict(seed=case['seed'], case=case, elapsed=info['elapsed'], success=info['is_success'],
                      termination=info['termination_reason'], tracking_rmse_m=info['tracking_rmse'],
                      tilt_rmse_deg=math.degrees(math.sqrt(tilt_squared/env.steps)), slowdown=args.slowdown)
        if args.video:
            args.video.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == '__main__':
    main()
