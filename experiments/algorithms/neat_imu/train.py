"""Train low-dimensional NEAT hovering, validate, then evaluate held-out seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import random
import time

import neat
import numpy as np

from .environment import ImuAttitudeEnv, INPUT_NAMES
from .policy import encode_network, decode_network

ENV = None


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def initialize_worker(settings):
    global ENV
    ENV = ImuAttitudeEnv(**settings)


def rollout(task):
    payload, seeds, seconds, trace = task
    network = decode_network(payload) if payload is not None else None
    ENV.set_seconds(seconds)
    results = []
    for seed in seeds:
        observation, reset_info = ENV.reset(seed=seed)
        reward_sum, records = 0., []
        while True:
            action = np.zeros(3) if network is None else network.activate(observation)
            observation, reward, terminated, truncated, info = ENV.step(action)
            reward_sum += reward
            if trace:
                records.append([info['elapsed'], *np.rad2deg(info['rpy']),
                                info['alpha'], info['beta'], *info['position'],
                                info['motor_command']])
            if terminated or truncated:
                break
        result = dict(seed=seed, fitness=reward_sum/ENV.horizon,
                      elapsed=info['elapsed'], steps=ENV.steps,
                      success=info['is_success'], termination=info['termination_reason'],
                      initial_rpy_deg=np.rad2deg(reset_info['rpy']).tolist(),
                      attitude_rmse_deg=info['attitude_rmse_deg'].tolist(),
                      max_tilt_deg=info['max_tilt_deg'],
                      tracking_rmse_m=info['tracking_rmse'])
        if trace:
            result['trace'] = records
        results.append(result)
    return results


def summarize(episodes):
    return dict(episodes=len(episodes), successes=sum(r['success'] for r in episodes),
                mean_fitness=float(np.mean([r['fitness'] for r in episodes])),
                mean_elapsed=float(np.mean([r['elapsed'] for r in episodes])),
                mean_tracking_rmse_m=float(np.mean([r['tracking_rmse_m'] for r in episodes])),
                mean_attitude_rmse_deg=np.mean([r['attitude_rmse_deg'] for r in episodes], axis=0).tolist(),
                control_steps=sum(r['steps'] for r in episodes),
                termination_counts={key: sum(r['termination'] == key for r in episodes)
                                    for key in sorted({r['termination'] for r in episodes})})


class Training(neat.reporting.BaseReporter):
    def __reduce__(self):
        # SpeciesSet includes its reporter registry in upstream checkpoints.
        # A live executor/Population is runtime state, not evolutionary state.
        # Restore this callback as an inert, importable upstream reporter; a
        # continuation driver must attach its own live evaluation callbacks.
        return neat.reporting.BaseReporter, ()

    def __init__(self, args, pool, config):
        self.args, self.pool, self.config = args, pool, config
        self.generation = 0
        self.history = []
        self.validation = []
        self.best_score = -np.inf
        self.best = None
        self.started = time.monotonic()
        self.train_steps = self.validation_steps = 0

    def start_generation(self, generation):
        self.generation = generation

    def evaluate(self, genomes, config):
        rng = np.random.default_rng(np.random.SeedSequence([self.args.seed, self.generation, 601]))
        seeds = rng.integers(100000, 800000000, self.args.scenarios).tolist()
        payloads = [encode_network(genome, config) for _, genome in genomes]
        results = list(self.pool.map(rollout, [(p, seeds, self.args.seconds, False) for p in payloads]))
        for (_, genome), episodes in zip(genomes, results):
            genome.fitness = float(np.mean([r['fitness'] for r in episodes]))
            self.train_steps += sum(r['steps'] for r in episodes)
        winner_index = int(np.argmax([g.fitness for _, g in genomes]))
        winner = genomes[winner_index][1]
        write_json(self.args.output/'generation_best.json', payloads[winner_index])
        row = dict(generation=self.generation, best_fitness=winner.fitness,
                   mean_fitness=float(np.mean([g.fitness for _, g in genomes])),
                   species=len(self.population.species.species),
                   hidden_nodes=len(winner.nodes)-3,
                   enabled_connections=sum(c.enabled for c in winner.connections.values()),
                   training_steps=self.train_steps, seeds=seeds)
        if self.generation % self.args.validate_every == 0 or self.generation == self.args.generations-1:
            episodes = list(self.pool.map(rollout, [(payloads[winner_index], [s],
                            self.args.validation_seconds, False) for s in range(830007000, 830007006)]))
            episodes = [r for group in episodes for r in group]
            summary = summarize(episodes)
            self.validation_steps += summary['control_steps']
            self.validation.append(dict(generation=self.generation, **summary, results=episodes))
            row['validation_fitness'] = summary['mean_fitness']
            if summary['mean_fitness'] > self.best_score:
                self.best_score = summary['mean_fitness']
                self.best = copy.deepcopy(winner)
                self.best_generation = self.generation
                write_json(self.args.output/'best_policy.json', payloads[winner_index])
                with (self.args.output/'best_genome.pkl').open('wb') as f:
                    pickle.dump(self.best, f)
            write_json(self.args.output/'validation.json', self.validation)
        row['wall_seconds'] = time.monotonic()-self.started
        self.history.append(row)
        write_json(self.args.output/'history.json', self.history)
        write_json(self.args.output/'status.json', dict(status='training', **row,
                   planned_generations=self.args.generations, best_validation=self.best_score))
        print(json.dumps(row), flush=True)


def plot_results(output, history, evaluations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    generations = [r['generation']+1 for r in history]
    axes[0, 0].plot(generations, [r['best_fitness'] for r in history], label='Generation best')
    axes[0, 0].plot(generations, [r['mean_fitness'] for r in history], label='Population mean')
    validation = [r for r in history if 'validation_fitness' in r]
    axes[0, 0].plot([r['generation']+1 for r in validation],
                    [r['validation_fitness'] for r in validation], 'o-', label='Validation')
    axes[0, 0].set(xlabel='Generation', ylabel='Normalized return', title='NEAT hovering: alpha, beta, motor speed')
    axes[0, 0].legend(fontsize=8)
    for i, (name, episodes) in enumerate(evaluations.items()):
        trace = np.array(episodes[0]['trace'])
        for k, label in enumerate(('x', 'y', 'z')):
            axes[1, i].plot(trace[:, 0], trace[:, k+6]-(.8 if k == 2 else 0), label=label)
        axes[1, i].axhline(.1, color='grey', linestyle=':', linewidth=.6)
        axes[1, i].axhline(-.1, color='grey', linestyle=':', linewidth=.6)
        axes[1, i].set(xlabel='Time (s)', ylabel='Position error (m)', title=f'{name}: first held-out seed')
        axes[1, i].legend(fontsize=8)
    trace = np.array(evaluations['neat'][0]['trace'])
    axes[0, 1].plot(trace[:, 0], np.rad2deg(trace[:, 4]), label='alpha / pitch')
    axes[0, 1].plot(trace[:, 0], np.rad2deg(trace[:, 5]), label='beta / roll')
    axes[0, 1].set(xlabel='Time (s)', ylabel='Command (deg)', title='Applied gimbal targets')
    axes[0, 1].legend(fontsize=8)
    motor_axis = axes[0, 1].twinx()
    motor_axis.plot(trace[:, 0], trace[:, 9], color='grey', alpha=.6)
    motor_axis.set_ylabel('Motor command (rad/s)', color='grey')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'training_results.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--scenarios', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--validation-seconds', type=float, default=4.)
    parser.add_argument('--test-seconds', type=float, default=8.)
    parser.add_argument('--test-episodes', type=int, default=16)
    parser.add_argument('--validate-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=707)
    parser.add_argument('--motor-speed', type=float, default=3400.)
    parser.add_argument('--initial-degrees', type=float, default=5.)
    parser.add_argument('--inputs', type=int, choices=(3, 6, 12), default=3)
    parser.add_argument('--seed-policy', type=Path,
                        help='Seed one individual from a saved direct-input policy; starts a NEW population')
    args = parser.parse_args()
    for key in ('generations', 'population', 'workers', 'scenarios', 'test_episodes', 'validate_every'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    for key in ('seconds', 'validation_seconds', 'test_seconds'):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         str(Path(__file__).with_name('config.ini')))
    config.pop_size = args.population
    config.genome_config.num_inputs = args.inputs
    config.genome_config.input_keys = [-i-1 for i in range(args.inputs)]
    config.motor_speed = args.motor_speed
    config.max_motor_speed = 5000.
    config.save(str(args.output/'neat_config.ini'))
    random.seed(args.seed)
    np.random.seed(args.seed)
    settings = dict(seconds=args.seconds, motor_speed=args.motor_speed,
                    initial_degrees=args.initial_degrees, inputs=args.inputs)
    # Construct once before workers so native compilation is serial and physics is checked.
    probe = ImuAttitudeEnv(**settings)
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    versions={n: importlib.metadata.version(n) for n in ('neat-python', 'mujoco', 'numpy', 'gymnasium')},
                    observation=INPUT_NAMES[args.inputs],
                    normalization=dict(attitude_rad=.35, gyro_rad_s=10., position_m=.2, velocity_m_s=.5),
                    action=['alpha=pitch target', 'beta=roll target', 'motor speed'],
                    motor_range_rad_s=[0., probe.source_config.actuator.max_velocity],
                    limits_rad=probe.sim.model.actuator_ctrlrange.tolist(),
                    mass_kg=float(probe.sim.model.body_mass.sum()),
                    gravity=probe.sim.model.opt.gravity.tolist(),
                    control_dt=probe.config.control_dt, physics_dt=probe.sim.model.opt.timestep,
                    initial_hidden_nodes=0, teacher_used=False,
                    task='Free-flight hovering at (0, 0, 0.8 m)',
                    success='Complete requested horizon and 3D position RMSE below 0.1 m',
                    validation_seeds=list(range(830007000, 830007006)),
                    test_seeds=list(range(840008000, 840008000+args.test_episodes)))
    if args.seed_policy:
        manifest['seed_policy_sha256'] = hashlib.sha256(args.seed_policy.read_bytes()).hexdigest()
        manifest['initialization'] = 'New population seeded by an earlier evolved policy; not exact checkpoint resume'
    source_files = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).with_name('config.ini')]
    source_files += list(Path('flapsim').rglob('*.py')) + list(Path('flapsim/rl').glob('*.cpp'))
    source_files += [Path('flapsim/config/training/ist_v2_flight.json'), probe.source_config.path]
    manifest['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    write_json(args.output/'run.json', manifest)
    probe.close()
    population = neat.Population(config)
    # One neutral trim individual; all other individuals start randomly from scratch.
    zero = next(iter(population.population.values()))
    for node in zero.nodes.values():
        node.bias = 0.
    for connection in zero.connections.values():
        connection.weight = 0.
    if args.seed_policy:
        if args.population < 2:
            parser.error('seeding requires population >= 2')
        payload = json.loads(args.seed_policy.read_text())
        seeded = list(population.population.values())[1]
        if (payload['inputs'] != config.genome_config.input_keys or
                payload['outputs'] != config.genome_config.output_keys or
                any(n['key'] not in seeded.nodes for n in payload['nodes'])):
            parser.error('seed policy must match input/output contract and have no hidden nodes')
        for connection in seeded.connections.values():
            connection.enabled = False
        for node in payload['nodes']:
            target = seeded.nodes[node['key']]
            target.bias, target.response = node['bias'], node['response']
            target.activation, target.aggregation = node['activation'], node['aggregation']
            for source, weight in node['links']:
                connection = seeded.connections[(source, node['key'])]
                connection.enabled, connection.weight = True, weight
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(settings,)) as pool:
        training = Training(args, pool, config)
        training.population = population
        population.add_reporter(training)
        population.add_reporter(neat.Checkpointer(10, filename_prefix=str(args.output/'neat-checkpoint-')))
        population.run(training.evaluate, args.generations)
        policy = encode_network(training.best, config)
        write_json(args.output/'selected.json', dict(generation=training.best_generation,
                   validation_fitness=training.best_score,
                   policy_sha256=hashlib.sha256((args.output/'best_policy.json').read_bytes()).hexdigest(),
                   hidden_nodes=len(training.best.nodes)-3,
                   enabled_connections=sum(c.enabled for c in training.best.connections.values())))
        write_json(args.output/'status.json', dict(status='testing', completed_generations=len(training.history)))
        evaluations = {}
        for name, payload in (('neat', policy), ('zero', None)):
            groups = list(pool.map(rollout, [(payload, [s], args.test_seconds, True) for s in manifest['test_seeds']]))
            evaluations[name] = [r for group in groups for r in group]
    summaries = {name: summarize(episodes) for name, episodes in evaluations.items()}
    for name, episodes in evaluations.items():
        with (args.output/f'{name}_trajectory.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'time_s', 'roll_deg', 'pitch_deg', 'yaw_deg',
                             'alpha_rad', 'beta_rad', 'x_m', 'y_m', 'z_m', 'motor_rad_s'])
            for ep in episodes:
                writer.writerows([[ep['seed'], *r] for r in ep['trace']])
    plot_results(args.output, training.history, evaluations)
    write_json(args.output/'evaluation.json', {name: [{k: v for k, v in ep.items() if k != 'trace'}
                                                  for ep in episodes] for name, episodes in evaluations.items()})
    result = dict(status='completed', generations=len(training.history), population=args.population,
                  best_generation=training.best_generation,
                  training_control_steps=training.train_steps,
                  validation_control_steps=training.validation_steps,
                  testing_control_steps=sum(s['control_steps'] for s in summaries.values()),
                  wall_seconds=time.monotonic()-training.started, test=summaries)
    write_json(args.output/'results.json', result)
    write_json(args.output/'status.json', result)
    (args.output/'REPORT.md').write_text(
        '# 低维 NEAT 悬停训练\n\n'
        f'完成 {args.generations} 代，种群 {args.population}，随机种子 {args.seed}。'
        f'耗时 {result["wall_seconds"]/60:.1f} 分钟。\n\n'
        f'输入 {args.inputs} 维：{", ".join(INPUT_NAMES[args.inputs])}。'
        '使用 neat-python 2.0.0 前馈拓扑演化，从零隐藏节点开始，无教师。'
        +(f'本次新种群的一只个体继承先前演化策略 {args.seed_policy}；并非精确检查点续训。'
          if args.seed_policy else '全部从零开始（含一只零动作配平个体）。')+
        '输出 alpha→pitch、beta→roll、电机转速。角度限幅 ±0.3 rad，电机 0–5000 rad/s，'
        f'零动作电机配平 {args.motor_speed:g} rad/s，20 ms 执行器命令平滑。\n\n'
        f'完整自由飞行，15 g，重力 9.81 m/s²，物理步长 5 µs，控制周期 10 ms。'
        f'每回合初始三个姿态角各在 ±{args.initial_degrees:g}° 独立均匀采样；初始角速度为零。'
        '初始位置、线速度各有标准差 0.01 m、0.01 m/s 的随机扰动。'
        '沿用原悬停奖励；地面、高倾角、1 m 位置边界或数值失败会提前终止。'
        '传感输入为仿真估计代理，未建模真实噪声或延迟。\n\n'
        f'每个体训练 {args.scenarios} 个 {args.seconds:g} 秒场景；'
        f'用 6 个固定、独立的 {args.validation_seconds:g} 秒验证场景选择策略，'
        f'选中第 {training.best_generation+1} 代。最终测试 {args.test_episodes} 个未参与选模的'
        f' {args.test_seconds:g} 秒场景，与零云台角度基线配对。\n\n'
        '| 策略 | 成功/总数 | 平均飞行秒数 | 位置 RMSE (cm) | roll / pitch / yaw RMSE (°) |\n'
        '|---|---:|---:|---:|---|\n'+''.join(
            f'| {name} | {s["successes"]}/{s["episodes"]} | {s["mean_elapsed"]:.3f} | '
            +f'{100*s["mean_tracking_rmse_m"]:.2f} | '
            +' / '.join(f'{v:.2f}' for v in s['mean_attitude_rmse_deg'])+' |\n'
            for name, s in summaries.items())+
        '\n成功须飞满测试时限且三维位置 RMSE <0.1 m。提前终止的 RMSE 只覆盖存活时段，'
        '不可据此认定稳定。三姿态输入版本无法直接感知位置、高度和速度漂移。'
        '单次训练不能证明多随机种子的可靠性；训练完成不保证稳定性达标。\n\n'
        '策略：`best_policy.json`；原始基因：`best_genome.pkl`；配置：`neat_config.ini`；'
        '曲线：`training_results.png`；测试明细：`evaluation.json`；轨迹：`*_trajectory.csv`。'
        '检查点为 neat-python 种群状态，仅加载可信的本地 pickle 文件。\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
