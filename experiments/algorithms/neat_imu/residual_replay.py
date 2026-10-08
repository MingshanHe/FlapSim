"""Replay the calibrated recurrent controller with the exact training mapping."""
import argparse
import json
from pathlib import Path

import numpy as np
from .residual import ResidualHoverEnv, ResidualPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--seed', type=int, default=871007000)
    p.add_argument('--video', type=Path)
    p.add_argument('--slowdown', type=int, choices=(1, 2, 4), default=1)
    args = p.parse_args()
    metadata = json.loads((args.run/'run.json').read_text())
    settings = dict(metadata['environment'], seconds=metadata['evaluation_seconds'], initial_degrees=5.)
    env = ResidualHoverEnv(**settings, render_mode='rgb_array' if args.video else None)
    policy = ResidualPolicy(args.run/'best_policy.json')
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
            draw.rectangle((0, 0, picture.width, 64), fill=(20, 28, 40))
            t = info.get('elapsed', 0.)
            error = info.get('tracking_rmse', 0.)
            speed = info.get('motor_command', env.center[2])
            draw.text((12, 8), f'3-angle recurrent NEAT | seed {args.seed} | {args.slowdown}x slow motion', fill='white')
            draw.text((12, 27), f'Simulation {t:.2f} s | position RMSE {error*100:.1f} cm | motor {speed:.0f} rad/s', fill='white')
            draw.text((12, 46), f'Status: {info.get("termination_reason") or "flying"} | target: 8 s, RMSE < 10 cm', fill='white')
            return np.asarray(picture)

        obs, info = env.reset(seed=args.seed)
        policy.reset(obs)
        if writer:
            writer.append_data(frame(info))
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            if writer and (env.steps % (4//args.slowdown) == 0 or terminated or truncated):
                writer.append_data(frame(info))
            if terminated or truncated:
                break
        result = dict(seed=args.seed, elapsed=info['elapsed'], success=info['is_success'],
                      termination=info['termination_reason'], position_rmse_m=info['tracking_rmse'],
                      attitude_rmse_deg=info['attitude_rmse_deg'].tolist(), playback_slowdown=args.slowdown)
        if args.video:
            args.video.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == '__main__':
    main()
