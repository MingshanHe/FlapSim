"""Compare NEAT search settings and train with lower-tail disturbance fitness."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import random
import time
import zipfile

import neat
import numpy as np

from .residual import action_map
from .robust import StableMemoryGenome, BoundedSpeciesSet, initialize, export, init_worker, rollout, summarize, robust_score, cases
from .train import write_json

PROFILES = {
    'focused': dict(weight_mutate_rate=.5, weight_mutate_power=.07, weight_replace_rate=.01,
                    bias_mutate_rate=.35, bias_mutate_power=.02, bias_replace_rate=.005,
                    conn_add_prob=.16, node_add_prob=.04),
    'exploratory': dict(weight_mutate_rate=.8, weight_mutate_power=.18, weight_replace_rate=.025,
                       bias_mutate_rate=.5, bias_mutate_power=.05, bias_replace_rate=.01,
                       conn_add_prob=.25, node_add_prob=.1),
}


def make_config(path, population, profile, variant='memory3'):
    genome_type = StableMemoryGenome
    if variant == 'gyro6':
        from .activation import ActivationMemoryGenome
        genome_type = ActivationMemoryGenome
    path.write_text(Path(__file__).with_name('config.ini').read_text().replace('[DefaultGenome]', f'[{genome_type.__name__}]').replace('[DefaultSpeciesSet]', '[BoundedSpeciesSet]'))
    cfg = neat.Config(genome_type, neat.DefaultReproduction, BoundedSpeciesSet,
                      neat.DefaultStagnation, str(path))
    cfg.pop_size, cfg.reset_on_extinction = population, True
    g = cfg.genome_config
    g.feed_forward, g.num_hidden = False, 0
    for key, value in PROFILES[profile].items():
        setattr(g, key, value)
    g.weight_min_value, g.weight_max_value = -12., 12.
    g.bias_min_value, g.bias_max_value = -1., 1.
    g.conn_delete_prob, g.node_delete_prob, g.enabled_mutate_rate = .015, .01, .005
    g.compatibility_weight_coefficient = 1.
    cfg.species_set_config.compatibility_threshold = .3
    cfg.stagnation_config.max_stagnation = 10
    cfg.stagnation_config.species_elitism = 2
    cfg.reproduction_config.elitism = 1
    cfg.reproduction_config.survival_threshold = .35
    cfg.reproduction_config.min_species_size = 2
    if variant == 'gyro6':
        g.num_inputs, g.input_keys = 6, [-1,-2,-3,-4,-5,-6]
        g.activation_options = ['tanh','clamped','sin','gauss']
        g.activation_mutate_rate = .12
        g.response_min_value, g.response_max_value = .1, 3.
        g.response_mutate_rate, g.response_mutate_power = .25, .08
        g.time_constant_min_value, g.time_constant_max_value = .01, .15
        g.time_constant_init_mean, g.time_constant_init_stdev = .04, .02
        g.time_constant_mutate_rate, g.time_constant_mutate_power = .25, .01
        g.time_constant_replace_rate = .01
    return cfg


def selection_key(summary):
    return summary['successes'], summary['robust_score'], summary['p10_elapsed']


def main():
    global initialize, export, init_worker, rollout, cases
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trim', type=Path, required=True)
    p.add_argument('--parent', type=Path, nargs='+', required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=PROFILES, default='exploratory')
    p.add_argument('--population', type=int, default=48)
    p.add_argument('--generations', type=int, default=32)
    p.add_argument('--scenarios', type=int, default=3)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=810)
    p.add_argument('--variant', choices=('memory3','gyro6'), default='memory3')
    p.add_argument('--hidden',type=int,default=0,help='additional homologous 6 -> H -> 3 seed layer')
    p.add_argument('--initialization',choices=('release','trim','momentum'),default='release')
    p.add_argument('--warmup',type=float,default=.1)
    p.add_argument('--model-path',type=Path,help='independent experimental MJCF (gyro6 only)')
    p.add_argument('--curriculum-start-degrees',type=float,default=5.)
    p.add_argument('--validation-base',type=int)
    p.add_argument('--test-base',type=int)
    p.add_argument('--resume', type=Path, help='trusted local NEAT checkpoint; runs until --generations')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.population < 8 or min(args.generations, args.workers, args.scenarios) < 1:
        p.error('population >=8 and positive counts required')
    if args.hidden<0 or args.warmup<0 or not 0<=args.curriculum_start_degrees<=10:
        p.error('invalid hidden count, warmup or initial disturbance')
    if args.variant!='gyro6' and (args.hidden or args.initialization!='release' or args.model_path is not None):
        p.error('hidden-layer, powered trim and custom model options require --variant gyro6')
    if args.model_path is not None:
        args.model_path=args.model_path.resolve()
        if not args.model_path.is_file():
            p.error('model-path must be an existing MJCF file')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trim = json.loads(args.trim.read_text())
    mapping, settings = action_map(trim), dict(trim=trim)
    if args.variant == 'gyro6':
        from .activation import initialize, export, init_worker, rollout, cases, signed_map
        mapping, settings = signed_map(trim), dict(trim=trim,sensor_inputs=6)
        if args.hidden or args.initialization!='release' or args.model_path is not None:
            from .architecture import initialize,init_worker
            settings.update(initialization=args.initialization,warmup_seconds=args.warmup)
        if args.model_path is not None:
            settings['model_path']=str(args.model_path)
            if trim.get('model_sha256') and trim['model_sha256']!=hashlib.sha256(args.model_path.read_bytes()).hexdigest():
                p.error('trim calibration belongs to a different model')
    parents = [json.loads(path.read_text()) for path in args.parent]
    baseline = json.loads(args.baseline.read_text())
    if any(payload['action_map'] != mapping for payload in parents+( [baseline] if args.variant=='memory3' else [])):
        p.error('all action maps must match calibrated residual limits')
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.resume:
        population = neat.Checkpointer.restore_checkpoint(str(args.resume))
        cfg = population.config
        if cfg.pop_size != args.population or population.generation >= args.generations:
            p.error('resume population must match and target generation must be later')
        cfg.species_set_type = BoundedSpeciesSet
        population.species.__class__ = BoundedSpeciesSet
        reporters = neat.reporting.ReporterSet()
        population.reporters = population.species.reporters = population.reproduction.reporters = reporters
        population.reproduction.stagnation.reporters = reporters
        population.species.speciate(cfg, population.population, population.generation)
        # Fitness normalization is now fixed across curriculum stages.
        for species in population.species.species.values():
            species.fitness_history = []
            species.last_improved = population.generation
    else:
        cfg = make_config(args.output/'neat_config.ini', args.population, args.profile, args.variant)
        cfg.genome_config.hidden_seed_count=args.hidden
        population = neat.Population(cfg)
        # Parent graphs may contain higher node IDs than a fresh population.
        max_parent_node = max(n['key']+(0 if p.get('memory_design') else 6)
                              for p in parents for n in p['nodes'])
        cfg.genome_config.node_indexer = itertools.count(max(11 if args.variant=='gyro6' else 8,max_parent_node+(100 if args.variant=='gyro6' else 0))+1)
        for i, genome in enumerate(population.population.values()):
            initialize(genome, cfg, mapping, rng,
                       parents[i % len(parents)] if i < max(len(parents), args.population//3) else None)
            if len(parents) <= i < args.population//3:
                genome.mutate(cfg.genome_config)
        max_node = max(k for genome in population.population.values() for k in genome.nodes)
        cfg.genome_config.node_indexer = itertools.count(max_node+1)
    target_species = max(2, min(8, args.population//6))
    if not args.resume:
        for _ in range(30):
            population.species = BoundedSpeciesSet(cfg.species_set_config, population.reporters)
            population.species.speciate(cfg, population.population, 0)
            n = len(population.species.species)
            if max(2, target_species-1) <= n <= target_species:
                break
            cfg.species_set_config.compatibility_threshold *= .9
    start_generation = population.generation
    cfg.save(str(args.output/'neat_config.ini'))
    population.add_reporter(neat.Checkpointer(generation_interval=1 if args.smoke else 4,
                            filename_prefix=str(args.output/'neat-checkpoint-')))
    seconds = .05 if args.smoke else 8.
    validation_base = 889810000 if args.smoke else 893810000 if args.variant=='gyro6' else 881810000
    if args.validation_base is not None and not args.smoke: validation_base=args.validation_base
    validation = cases(range(validation_base, validation_base+(2 if args.smoke else 6)), seconds)
    validation += cases(range(validation_base+10000, validation_base+10000+(2 if args.smoke else 6)), seconds, True)
    # These seeds are never evaluated by pilot searches or used for model selection.
    test_base = 989810000 if args.smoke else 993810000 if args.variant=='gyro6' else 991810000
    if args.test_base is not None and not args.smoke: test_base=args.test_base
    test_nominal = cases(range(test_base, test_base+(2 if args.smoke else 32)), seconds)
    test_stress = cases(range(test_base+10000, test_base+10000+(2 if args.smoke else 32)), seconds, True)
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('config.ini')]
    files += list(Path('flapsim').rglob('*.py'))+list(Path('flapsim').rglob('*.cpp'))
    files += [Path('flapsim/config/training/ist_v2_flight.json')]
    from flapsim import load_config
    from flapsim.rl.env import PRESET
    physics_model=args.model_path or load_config(PRESET).path
    physics_model_hash=hashlib.sha256(physics_model.read_bytes()).hexdigest()
    (args.output/'physics_model.xml').write_bytes(physics_model.read_bytes())
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    arguments={k: [str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
                               for k, v in vars(args).items()},
                    inputs=['roll', 'pitch', 'yaw']+(['gyro_x','gyro_y','gyro_z'] if args.variant=='gyro6' else []), outputs=['alpha', 'beta', 'motor_speed'],
                    memory='Six angle and three gyro stable identity EMA nodes; time constants 10..150 ms, activations, response, topology evolve' if args.variant=='gyro6' else 'Six fixed identity EMA nodes (20/80 ms); other nodes, control connections and topology evolve',
                    initial_hidden_layer=args.hidden,
                    initialization='Prior NEAT graphs plus asymmetric gain and random recurrent graphs; no teacher',
                    environment=settings, action_map=mapping,
                    physics_model_path=str(physics_model),physics_model_sha256=physics_model_hash,
                    validation_cases=validation, test_nominal=test_nominal, test_stress=test_stress,
                    profile=PROFILES[args.profile],
                    start_generation=start_generation,
                    resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
                    parent_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.parent},
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    write_json(args.output/'run.json', manifest)
    with zipfile.ZipFile(args.output/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path))
        archive.write(physics_model,'physics_model.xml')
    write_json(args.output/'baseline_policy.json', baseline)
    history, validations = [], []
    budgets = dict(training_steps=0, validation_steps=0, test_steps=0)
    best = dict(payload=baseline, score=None, generation=-1, origin='previous_baseline')
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(settings,)) as pool:
        def evaluate_cases(payload, batch, trace=False):
            return [row for rows in pool.map(rollout, [(payload, [case], trace) for case in batch]) for row in rows]

        def validate(payload, generation, origin):
            rows = evaluate_cases(payload, validation)
            summary = summarize(rows)
            budgets['validation_steps'] += summary['control_steps']
            validations.append(dict(generation=generation, origin=origin, summary=summary, episodes=rows))
            write_json(args.output/'validation.json', validations)
            return summary

        for i, payload in enumerate([baseline]+parents):
            score = validate(payload, -1, 'baseline' if i == 0 else f'parent_{i-1}')
            if best['score'] is None or selection_key(score) > selection_key(best['score']):
                best.update(payload=payload, score=score, origin='previous_baseline' if i == 0 else f'parent_{i-1}')
        write_json(args.output/'best_policy.json', best['payload'])
        print(json.dumps(dict(stage='initial_validation', **best['score'])), flush=True)

        def evaluate(genomes, config):
            generation = population.generation
            fraction = generation/args.generations
            # Equal individual-evaluation pilot budgets must use the same
            # curriculum and mutation scale despite different generation counts.
            if args.pilot: fraction=0.
            horizon = .05 if args.smoke else 2. if fraction < .3 else 4. if fraction < .6 else 8.
            seeds = np.random.default_rng(np.random.SeedSequence([args.seed, generation, 724])).integers(10000, 800000000, args.scenarios)
            batch = []
            for i, seed in enumerate(seeds):
                case = cases([seed], horizon, stress=i % 3 != 0)[0]
                if fraction < .3:
                    case['degrees']=args.curriculum_start_degrees
                if i % 3 != 0 and fraction < .3:
                    case.update(angular_rate=.1, noise_degrees=.1, motor_uncertainty=.01)
                batch.append(case)
            # Anneal numeric changes, retaining structural exploration and diversity.
            anneal = 1. if fraction < .7 else .5
            for key in ('weight_mutate_power', 'bias_mutate_power'):
                setattr(config.genome_config, key, PROFILES[args.profile][key]*anneal)
            payloads = [export(genome, config, mapping) for _, genome in genomes]
            groups = list(pool.map(rollout, [(payload, batch, False) for payload in payloads]))
            for (_, genome), rows in zip(genomes, groups):
                # Tiny complexity cost breaks ties, not a surrogate for control quality.
                genome.fitness = robust_score(rows)-1e-5*sum(c.enabled for c in genome.connections.values())
                budgets['training_steps'] += sum(row['steps'] for row in rows)
            order = np.argsort([genome.fitness for _, genome in genomes])[::-1]
            winner = genomes[int(order[0])][1]
            row = dict(generation=generation, seconds=horizon, cases=batch,
                       best_fitness=winner.fitness, mean_fitness=float(np.mean([g.fitness for _, g in genomes])),
                       species=len(population.species.species), species_merges=population.species.last_merges,
                       compatibility_threshold=config.species_set_config.compatibility_threshold,
                       mutation_scale=anneal, nodes=len(winner.nodes), connections=sum(c.enabled for c in winner.connections.values()),
                       training_steps=budgets['training_steps'])
            if generation % 3 == 0 or generation == args.generations-1:
                for index in order[:2]:
                    score = validate(payloads[int(index)], generation, 'candidate')
                    if selection_key(score) > selection_key(best['score']):
                        best.update(payload=payloads[int(index)], score=score, generation=generation, origin='evolved')
                        write_json(args.output/'best_policy.json', best['payload'])
                        with (args.output/'best_genome.pkl').open('wb') as file:
                            pickle.dump(genomes[int(index)][1], file)
                row['selected_validation'] = best['score']
            # Adapt future speciation rather than leaving the whole population in one group.
            threshold = config.species_set_config.compatibility_threshold
            if row['species'] < target_species:
                threshold *= .9
            elif row['species'] > target_species+2:
                threshold *= 1.1
            config.species_set_config.compatibility_threshold = float(np.clip(threshold, .025, 3.))
            row['wall_seconds'] = time.monotonic()-started
            history.append(row)
            write_json(args.output/'history.json', history)
            write_json(args.output/'status.json', dict(status='training', planned_generations=args.generations,
                       selected_generation=best['generation'], selected_origin=best['origin'], **row))
            print(json.dumps({k: v for k, v in row.items() if k != 'cases'}), flush=True)

        population.run(evaluate, args.generations-population.generation)
        write_json(args.output/'selected.json', {k: v for k, v in best.items() if k != 'payload'})
        evaluation = {}
        if not args.pilot:
            for group, batch in (('nominal', test_nominal), ('stress', test_stress)):
                evaluation[group] = {}
                for label, payload in (('new', best['payload']), ('previous', baseline)):
                    rows = evaluate_cases(payload, batch, True)
                    budgets['test_steps'] += sum(row['steps'] for row in rows)
                    evaluation[group][label] = rows
            write_json(args.output/'evaluation.json', evaluation)
    result = dict(status='completed', profile=args.profile, population=args.population, generations=args.generations,
                  start_generation=start_generation, evaluated_generations=len(history),
                  selected_generation=best['generation'], selected_origin=best['origin'],
                  validation=best['score'], test={group: {name: summarize(rows) for name, rows in pairs.items()}
                                                  for group, pairs in evaluation.items()},
                  budgets=budgets, wall_seconds=time.monotonic()-started)
    write_json(args.output/'results.json', result)
    write_json(args.output/'status.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
