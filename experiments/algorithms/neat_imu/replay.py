"""Replay a saved three-angle NEAT policy in full flight, optionally to video."""
import argparse
import json
from pathlib import Path

from .environment import ImuAttitudeEnv
from .policy import ImuPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=840008000)
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run/'run.json').read_text())
    settings = metadata['arguments']
    policy = ImuPolicy(args.run/'best_policy.json')
    env = ImuAttitudeEnv(seconds=args.seconds, motor_speed=settings['motor_speed'],
                         initial_degrees=settings['initial_degrees'],
                         inputs=settings.get('inputs', 3),
                         render_mode='rgb_array' if args.video else None)
    writer = None
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec='libx264')
        obs, _ = env.reset(seed=args.seed)
        if writer:
            writer.append_data(env.render())
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs))
            if writer and (env.steps % 4 == 0 or terminated or truncated):
                writer.append_data(env.render())
            if terminated or truncated:
                break
        print(json.dumps(dict(seed=args.seed, elapsed=info['elapsed'],
                              success=info['is_success'], termination=info['termination_reason'],
                              position_rmse_m=info['tracking_rmse'],
                              attitude_rmse_deg=info['attitude_rmse_deg'].tolist()), indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == '__main__':
    main()
