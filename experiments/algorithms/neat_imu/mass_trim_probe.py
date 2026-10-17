"""Check whether approximate mean-force trim survives powered free release."""
from concurrent.futures import ProcessPoolExecutor
import json
from .activation import cases
from .architecture import trim_network
from .hover_initial_probe import evaluate
from .mass_audit_run import ROOT
from .robust import summarize
from .train import write_json


def main():
    output = ROOT / 'trim_release'
    output.mkdir(exist_ok=False)
    trim = json.loads((ROOT / 'nominal_trim/trim.json').read_text())
    payload = trim_network(trim)
    write_json(ROOT / 'nominal_seed_policy.json', payload)
    batch = cases(range(898810000, 898810008), 8.)
    specifications = [('release', .1), ('trim', .1), ('trim', .8), ('momentum', .1)]
    tasks = [(dict(trim=trim, sensor_inputs=6, model_path=trim['model_path'],
                   initialization=mode, warmup_seconds=warmup), payload, [case])
             for mode, warmup in specifications for case in batch]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(evaluate, tasks))
    results = []
    for index, (mode, warmup) in enumerate(specifications):
        episodes = [row for group in rows[index * 8:(index + 1) * 8] for row in group]
        results.append(dict(initialization=mode, warmup_seconds=warmup,
                            summary=summarize(episodes), episodes=episodes))
    write_json(output / 'results.json', dict(status='completed',
               scope='Approximate fixed-base mean-force seed; strict moment trim has not converged',
               results=results, control_steps=sum(row['summary']['control_steps'] for row in results)))
    print(json.dumps([{k: v for k, v in row.items() if k != 'episodes'} for row in results], indent=2))


if __name__ == '__main__':
    main()
