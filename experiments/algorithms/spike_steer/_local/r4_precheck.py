"""Round-4 design pre-check (claude-ea, training-split cases only, never a blind set): the 'at most one learned pulse
onset per 37 ms wingbeat' constraint (HomePC height_escape diagnostic, variant b) applied without re-tuning to every
selected round-2 / round-3 spec, intact vs b, on 64 nod + 64 dash cases at a fresh training offset.

    python -m experiments.algorithms.spike_steer._local.r4_precheck --workers 30
"""
import argparse
import importlib.util
import json
import multiprocessing as mp
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .. import bench as B

spec_ = importlib.util.spec_from_file_location('hediag', 'artifacts/spike_steer/plan_v4/r3/diag/height_escape_diag.py')
D = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(D)

OFFSET = 90_000     # training split: ES 100_000+gen*100, validation 50_000/60_000, envelope diag 70_000, HomePC diag 80_000
P = 'artifacts/spike_steer/plan_v4'
SPECS = {'r2_s1_g44': f'{P}/r2/finetune_s1/spec_val_g44.json', 'r2_s2_g29': f'{P}/r2/finetune_s2/spec_val_g29.json',
         'r2_s3c_g59': f'{P}/r2/finetune_s3_cloudpc/spec_val_g59.json', 'r2_s5_g14': f'{P}/r2/finetune_s5/spec_val_g14.json',
         'r3_s1_g59': f'{P}/r3/finetune_s1/spec_val_g59.json', 'r3_s2_g9': f'{P}/r3/finetune_s2/spec_val_g9.json',
         'r3_s5_g9': f'{P}/r3/finetune_s5/spec_val_g9.json'}


def job(args):
    label, variant, task, c, spec = args
    res = D.cf_task((variant, task, c, spec))
    res['label'] = label
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=30)
    p.add_argument('--n', type=int, default=64)
    a = p.parse_args()
    D.R.install(fresh_test=False)
    base_tr = json.loads(D.BASE_R3_TRAIN.read_text())
    norm = base_tr['normalisers']
    cases = {t: B.cases(t, 'train', None, n=a.n, offset=OFFSET) for t in ('nod', 'dash')}
    jobs = [(lab, v, t, c, json.loads(Path(f).read_text())) for lab, f in SPECS.items() for v in ('intact', 'b') for t in cases for c in cases[t]]
    if not (D.SRC/'robot_upper7_lower8.xml').exists():
        B.prepare_source(D.SRC)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('fork'), initializer=D._init, initargs=(str(D.SRC), base_tr['prereg']['dash_m'] or .25)) as ex:
        rows = list(ex.map(job, jobs, chunksize=1))
    summ = {}
    for lab in SPECS:
        for v in ('intact', 'b'):
            for t in cases:
                rs = [r for r in rows if r['label'] == lab and r['variant'] == v and r['task'] == t]
                crashes = [r['termination'] for r in rs if not r['flown']]
                s = dict(ratio=st.mean(r['score'] for r in rs)/norm[t], crashes=len(crashes),
                         tilt=sum(c == 'tilt' for c in crashes), height=sum(c == 'height_escape' for c in crashes),
                         tilt_peak_max=max(r['tilt_peak_deg'] for r in rs), n_over60=sum(r['tilt_peak_deg'] > 60 for r in rs),
                         z_max_mean=st.mean(r['z_max'] for r in rs))
                summ[f'{lab}|{v}|{t}'] = s
                print(f"{lab:11s} {v:6s} {t:4s} ratio {s['ratio']:.3f} crashes {s['crashes']:2d} (tilt {s['tilt']}, height {s['height']}) "
                      f"tilt>60 {s['n_over60']:2d} max {s['tilt_peak_max']:.0f} zmax {s['z_max_mean']:.2f}", flush=True)
    out = Path('runs/spike_steer/r4_precheck'); out.mkdir(parents=True, exist_ok=True)
    (out/'precheck.json').write_text(json.dumps(dict(offset=OFFSET, n=a.n, normalisers=norm, specs=SPECS, summary=summ, rows=rows)))


if __name__ == '__main__':
    main()
