"""Complete generations, four common scenarios, durable generation boundaries."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from pathlib import Path
import pickle
import random
import time
import numpy as np
from .common import read, write, append
from .environment import case, scenarios
from .evaluate import worker_init, evaluate_task, evaluate, selection
from .evolution import create_population, export


def horizon(generation, generations):
    return (2., 4., 8.)[min(2, generation*3//generations)]


def allocated_budget(population, generations):
    return sum(population*4*round(horizon(g, generations)*500) for g in range(generations))


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    initial = read(args.initial)
    cfg, pop = create_population(out, initial, args.seed, args.population)
    state = dict(status='training', training_steps=0, allocated_steps=0, validation_steps=0,
                 generation=0, kind=initial['kind'], mode=args.mode, seed=args.seed,
                 budget=allocated_budget(args.population, args.generations))
    best = None
    checkpoint = out/'checkpoint.pkl.gz'
    if checkpoint.exists():
        with gzip.open(checkpoint, 'rb') as handle:
            pop, rng_state, state, best = pickle.load(handle)
        random.setstate(rng_state)
    start = time.monotonic()
    cases = (scenarios('validation', 2, .02) if args.smoke else
             scenarios('validation', 16, 20.) + scenarios('stress_validation', 16, 20., 'stress'))
    if args.smoke:
        state['budget'] = args.population*args.generations*4*10
    with ProcessPoolExecutor(args.workers, initializer=worker_init, initargs=(args.source,)) as pool:
        def validation(spec, generation):
            nonlocal best
            result = evaluate(pool, spec, args.mode, cases)
            state['validation_steps'] += result['summary']['steps']
            write(out/f'validation_{generation:03d}.json', result)
            if best is None or selection(result['summary']) > selection(best):
                best = result['summary']
                write(out/'best_policy.json', spec)
                write(out/'selection.json', dict(generation=generation, summary=best))
            return result['summary']
        if state['generation'] == 0 and best is None:
            validation(initial, 0)
        def fitness(genomes, config):
            generation = state['generation']
            seconds = .02 if args.smoke else horizon(generation, args.generations)
            batch = [case(1_200_000_000+args.seed*10000+generation*4+i, seconds,
                          'nominal' if (generation*4+i) % 3 == 0 else 'train', (generation*4+i) % 8)
                     for i in range(4)]
            specifications = [export(genome, initial['kind']) for _, genome in genomes]
            rows = list(pool.map(evaluate_task, [(spec, args.mode, c) for spec in specifications for c in batch]))
            archived = []
            for index, ((gid, genome), spec) in enumerate(zip(genomes, specifications)):
                episodes = rows[index*4:index*4+4]
                genome.fitness = float(np.mean([r['return_sum']/seconds for r in episodes]))
                archived.append(dict(id=gid, fitness=genome.fitness, policy=spec, episodes=episodes,
                                     parents=getattr(genome, 'parents', []),
                                     mutation=getattr(genome, 'mutation_event', {}),
                                     species=pop.species.get_species_id(gid)))
            champion = max(archived, key=lambda row: row['fitness'])
            with gzip.open(out/f'generation_{generation:03d}.json.gz', 'wt') as handle:
                import json
                json.dump(dict(individuals=archived, cases=batch,
                               merges=pop.species.merge_events), handle)
            state['training_steps'] += sum(r['steps'] for r in rows)
            state['allocated_steps'] += len(genomes)*4*round(seconds*500)
            state['generation'] += 1
            if state['generation'] % args.validate_every == 0 or state['generation'] == args.generations:
                validation(champion['policy'], state['generation'])
            state.update(best_validation=best, wall_seconds=time.monotonic()-start)
            write(out/'status.json', state)
            append(out/'history.jsonl', state)
            print(state, flush=True)
        while state['generation'] < args.generations:
            pop.run(fitness, 1)
            temporary = out/'checkpoint.tmp.pkl.gz'
            with gzip.open(temporary, 'wb') as handle:
                pickle.dump((pop, random.getstate(), state, best), handle)
            temporary.replace(checkpoint)
    state['status'] = 'completed'
    write(out/'results.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['direct', 'residual'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--generations', type=int, default=24)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--validate-every', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
