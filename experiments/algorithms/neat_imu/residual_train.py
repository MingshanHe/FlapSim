"""Evolve a small recurrent NEAT controller around a calibrated hover trim."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import random
import shutil
import time

import neat
import numpy as np
from .residual import ResidualHoverEnv, action_map, gain_network, encode, seed_genome
from .residual_probe import init_worker, run, summary
from .train import write_json, plot_results
from .plot_policy import draw


def selection_key(s):
    return (s['successes'], s['mean_fitness'], s['mean_elapsed'], -s['mean_tracking_rmse_m'])


def make_config(population):
    c = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                    neat.DefaultStagnation, str(Path(__file__).with_name('config.ini')))
    c.pop_size = population
    g = c.genome_config
    g.feed_forward = False
    g.num_hidden = 0
    g.weight_mutate_rate, g.weight_mutate_power, g.weight_replace_rate = .15, .03, .005
    g.bias_mutate_rate, g.bias_mutate_power, g.bias_replace_rate = .15, .005, 0.
    g.weight_min_value, g.weight_max_value = -12., 12.
    g.node_add_prob, g.node_delete_prob = .05, .005
    g.conn_add_prob, g.conn_delete_prob = .1, .005
    g.enabled_mutate_rate = .002
    c.species_set_config.compatibility_threshold = 1.2
    c.stagnation_config.max_stagnation = 12
    c.reproduction_config.elitism = 2
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trim', type=Path, required=True)
    p.add_argument('--initialization', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=24)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--seed', type=int, default=709)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if min(args.population, args.generations, args.workers) < 1 or args.population < 4:
        p.error('population >=4 and positive generations/workers required')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    trim = json.loads(args.trim.read_text())
    candidates = json.loads((args.initialization/'candidates.json').read_text())
    mapping = action_map(trim)
    if candidates[0]['payload']['action_map'] != mapping:
        p.error('initialization and calibrated action map differ')
    seed_payload = candidates[0]['payload']
    zero = gain_network(0., 0., .05, mapping)
    settings = dict(trim=trim)
    env = ResidualHoverEnv(**settings)
    model_path = env.source_config.path
    env.close()
    config = make_config(args.population)
    random.seed(args.seed)
    np.random.seed(args.seed)
    population = neat.Population(config)
    for i, genome in enumerate(population.population.values()):
        payload = seed_payload if i == 0 else zero if i == 1 else candidates[(i-2) % min(8, len(candidates))]['payload']
        seed_genome(genome, config, payload)
        if i > 1:
            for gene in genome.connections.values():
                if random.random() < .2:
                    gene.weight += random.gauss(0., .01)
            for node in genome.nodes.values():
                if node.key in config.genome_config.output_keys:
                    node.bias += random.gauss(0., .002)
    population.species = neat.DefaultSpeciesSet(config.species_set_config, population.reporters)
    population.species.speciate(config, population.population, 0)
    while len(population.species.species) > max(2, args.population//4):
        config.species_set_config.compatibility_threshold *= 1.5
        population.species = neat.DefaultSpeciesSet(config.species_set_config, population.reporters)
        population.species.speciate(config, population.population, 0)
    config.save(str(args.output/'neat_config.ini'))
    # Only upstream reporters are stored in the species registry; process pools
    # live in the evaluation closure and are never serialized in checkpoints.
    population.add_reporter(neat.Checkpointer(1 if args.smoke else 5,
                            filename_prefix=str(args.output/'neat-checkpoint-')))
    validation_seeds = list(range(891007000 if args.smoke else 861007000,
                                  (891007000 if args.smoke else 861007000)+(2 if args.smoke else 6)))
    test_seeds = list(range(892007000 if args.smoke else 871007000,
                            (892007000 if args.smoke else 871007000)+(2 if args.smoke else 16)))
    horizon = .05 if args.smoke else 8.
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('config.ini'), model_path]
    files += list(Path('flapsim').rglob('*.py'))+list(Path('flapsim/rl').glob('*.cpp'))
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    created_utc=datetime.now(timezone.utc).isoformat(), inputs=['roll', 'pitch', 'yaw'],
                    outputs=['alpha', 'beta', 'motor_speed'], recurrent=True,
                    action_map=mapping, environment=settings, validation_seeds=validation_seeds,
                    test_seeds=test_seeds, evaluation_seconds=horizon,
                    initialization='Small engineered angle-memory graphs selected by documented gain probes; no teacher',
                    trim_sha256=hashlib.sha256(args.trim.read_bytes()).hexdigest(),
                    initialization_sha256=hashlib.sha256((args.initialization/'candidates.json').read_bytes()).hexdigest(),
                    source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                    versions={n: importlib.metadata.version(n) for n in ('neat-python', 'mujoco', 'numpy')})
    write_json(args.output/'run.json', manifest)
    shutil.copy2(args.trim, args.output/'trim.json')
    write_json(args.output/'initial_policy.json', seed_payload)
    best = dict(payload=seed_payload, generation=-1, score=None, genome=None)
    history, validations = [], []
    budgets = dict(training_steps=0, validation_steps=0, final_test_steps=0)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(settings,)) as pool:
        def validate(payload, generation):
            rows = [r for batch in pool.map(run, [(payload, [s], horizon, 5., False) for s in validation_seeds]) for r in batch]
            s = summary(rows)
            budgets['validation_steps'] += s['control_steps']
            validations.append(dict(generation=generation, **s, results=rows))
            write_json(args.output/'validation.json', validations)
            return s

        best['score'] = validate(seed_payload, -1)
        write_json(args.output/'best_policy.json', seed_payload)
        print(json.dumps(dict(stage='initial_validation', **best['score'])), flush=True)

        def evaluate(genomes, cfg):
            generation = population.generation
            if args.smoke:
                seconds, degrees, scenarios = .05, 1., 1
            else:
                # Progression changes training only; all selection uses the
                # same 8-second, +/-5-degree validation distribution.
                fraction = generation/args.generations
                seconds, degrees = (2., 1.) if fraction < .25 else (4., 2.) if fraction < .5 else (8., 5.)
                scenarios = 3
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, generation, 710]))
            seeds = rng.integers(100000, 800000000, scenarios).tolist()
            payloads = [encode(g, cfg, mapping) for _, g in genomes]
            groups = list(pool.map(run, [(payload, seeds, seconds, degrees, False) for payload in payloads]))
            for (_, genome), rows in zip(genomes, groups):
                # Mean plus a lower-tail term discourages specializing in a
                # single favorable initial attitude in this small population.
                fitness = [r['fitness'] for r in rows]
                genome.fitness = .75*float(np.mean(fitness))+.25*min(fitness)
                budgets['training_steps'] += sum(r['steps'] for r in rows)
            order = np.argsort([g.fitness for _, g in genomes])[::-1]
            winner = genomes[int(order[0])][1]
            row = dict(generation=generation, seconds=seconds, initial_degrees=degrees,
                       best_fitness=winner.fitness, mean_fitness=float(np.mean([g.fitness for _, g in genomes])),
                       species=len(population.species.species), hidden_nodes=len(winner.nodes)-3,
                       enabled_connections=sum(c.enabled for c in winner.connections.values()),
                       training_steps=budgets['training_steps'])
            write_json(args.output/'generation_best.json', payloads[int(order[0])])
            if generation % 4 == 0 or generation == args.generations-1:
                # Two candidates reduce dependence on one noisy training rank.
                for index in order[:min(2, len(order))]:
                    s = validate(payloads[int(index)], generation)
                    row['validation_fitness'] = max(row.get('validation_fitness', -np.inf), s['mean_fitness'])
                    if selection_key(s) > selection_key(best['score']):
                        best.update(payload=payloads[int(index)], generation=generation,
                                    score=s, genome=copy.deepcopy(genomes[int(index)][1]))
                        write_json(args.output/'best_policy.json', best['payload'])
                        with (args.output/'best_genome.pkl').open('wb') as f:
                            pickle.dump(best['genome'], f)
            row['wall_seconds'] = time.monotonic()-start
            history.append(row)
            write_json(args.output/'history.json', history)
            write_json(args.output/'status.json', dict(status='training', planned_generations=args.generations,
                       best_validation=best['score'], **row))
            print(json.dumps(row), flush=True)

        population.run(evaluate, args.generations)
        write_json(args.output/'selected.json', dict(generation=best['generation'], validation=best['score'],
                   policy_sha256=hashlib.sha256((args.output/'best_policy.json').read_bytes()).hexdigest()))
        write_json(args.output/'status.json', dict(status='testing'))
        evaluations = {}
        for name, payload in (('neat', best['payload']), ('trim_only', zero), ('initial_memory', seed_payload)):
            evaluations[name] = [r for batch in pool.map(run, [(payload, [s], horizon, 5., True) for s in test_seeds]) for r in batch]
    summaries = {name: summary(rows) for name, rows in evaluations.items()}
    for name, rows in evaluations.items():
        budgets['final_test_steps'] += sum(r['steps'] for r in rows)
        with (args.output/f'{name}_trajectory.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'time_s', 'roll_deg', 'pitch_deg', 'yaw_deg', 'alpha_rad', 'beta_rad', 'x_m', 'y_m', 'z_m', 'motor_rad_s'])
            for row in rows:
                w.writerows([[row['seed'], *record] for record in row['trace']])
    plot_results(args.output, history, {'neat': evaluations['neat'], 'zero': evaluations['trim_only']})
    draw(args.output/'best_policy.json', args.output/'network.png')
    write_json(args.output/'evaluation.json', {k: [{a: b for a, b in r.items() if a != 'trace'} for r in rows]
                                              for k, rows in evaluations.items()})
    result = dict(status='completed', generations=args.generations, population=args.population,
                  selected_generation=best['generation'], test=summaries, budgets=budgets,
                  wall_seconds=time.monotonic()-start)
    write_json(args.output/'results.json', result)
    write_json(args.output/'status.json', result)
    report = ('# 配平点附近的循环 NEAT 悬停训练\n\n'
              f'neat-python 2.0.0，{args.population}个体、{args.generations}代，种子{args.seed}。'
              f'训练与评估耗时{result["wall_seconds"]/60:.1f}分钟。\n\n'
              '网络每步仅接收roll/pitch/yaw三个姿态角，减去已标定的固定目标并除以0.35 rad。'
              '使用有状态循环NEAT；初始网络含姿态低通记忆与类似PD的连接，先通过独立开发场景筛选增益。'
              '这些节点和连接随后参与NEAT演化。没有PPO教师、外接PID、额外传感输入或在线高度控制器。'
              '每回合网络状态重置，只重复初始当前观测做初始化，不读未来数据。\n\n'
              f'电机指令为 {mapping["center"][2]:.3f} + {mapping["span"][2]:.3f} × u₃ rad/s；'
              f'alpha = {mapping["center"][0]:.6f} + 0.2 × u₁ rad；'
              f'beta = {mapping["center"][1]:.6f} + 0.2 × u₂ rad。'
              'u各在[-1,1]，物理目标仅平滑一次（20 ms），不逐步累加。'
              f'固定基座双翼升力{trim["lift_gf"]:.4f} gf，扑翼{trim["flapping_hz"]:.4f} Hz，'
              f'实际电机转速{trim["actual_motor_rad_s"]:.2f} rad/s。'
              f'严格力矩容差是否通过：{trim["converged"]}；升力配平不是自由飞行稳定性的证明。\n\n'
              '保持15 g整机、9.81 m/s²重力、5 µs物理步长、100 Hz控制、完整自由基座。'
              '使用原0.1秒预热；在配平云台目标下重建预热状态。初始姿态围绕配平姿态扰动，'
              '位置和线速度仍各有标准差0.01 m、0.01 m/s扰动。\n\n'
              '课程从2秒/±1°、4秒/±2°到8秒/±5°。奖励联合位置、速度、姿态、角速度和平滑性；'
              '奖励可读仿真状态，这些状态不会传入策略。固定6个8秒/±5°场景选模，'
              '成功率优先，其次验证回报。最终16个新种子测试，成功需飞满8秒且三维位置RMSE<0.1 m。\n\n'
              '|策略|成功/总数|平均飞行(s)|平均位置RMSE(cm)|\n|---|---:|---:|---:|\n')
    for name, s in summaries.items():
        report += f'|{name}|{s["successes"]}/{s["episodes"]}|{s["mean_elapsed"]:.3f}|{s["mean_tracking_rmse_m"]*100:.2f}|\n'
    report += ('\ntrim_only为相同初态下固定配平指令；initial_memory为NEAT开始前的记忆网络。'
               '若选回initial，不得归因于NEAT改善。提前失败的误差仅覆盖存活时段；'
               '不以短时误差或静态15 gf宣称已实现悬停。只有三个姿态角仍不能观测位置和高度漂移。\n\n'
               f'本轮训练{budgets["training_steps"]:,}步，验证{budgets["validation_steps"]:,}步，'
               f'最终测试{budgets["final_test_steps"]:,}步。标定与初始化探针另记于上级目录，必须另计成本。'
               '\n\n来源思想：在名义控制周围学习修正量可参见 [Residual RL](https://arxiv.org/abs/1812.03201)；'
               '本实现用NEAT演化近似配平点附近的修正，不声称复现该论文算法。\n')
    (args.output/'REPORT.md').write_text(report, encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
