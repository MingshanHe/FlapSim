"""Read all repeats; preserve sensor axes and raw data, subtract a stated tare."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from analyze_comparison import load_force, write_csv

FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER/'rect_validation'


def analyze_real():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(exist_ok=True)
    rows, sources = [], []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
    for ki, kind in enumerate(('curve', 'rect')):
        paths = sorted((FOLDER/f'0831_{kind}_capton').rglob('*.csv.csv'))
        assert len(paths) == 18
        for path in paths:
            a = load_force(path)
            hz = int(re.search(r'_(\d+)Hz_', path.name)[1])
            epath = path.with_name(path.name.replace('.csv.csv', '_encoder.csv'))
            e = np.genfromtxt(epath, delimiter=',', names=True)
            keep = (e['t_s'] >= .5) & (e['t_s'] < e['t_s'][-1]-.3)
            frequency = abs(np.polyfit(e['t_s'][keep], e['shaft_deg'][keep]/360, 1)[0])
            on = (a[:, 0] >= .5) & (a[:, 0] < 4)
            off = (a[:, 0] >= 5.5) & (a[:, 0] < a[-1, 0]-.1)
            assert on.sum() > 1000 and off.sum() > 100
            baseline = a[off, 1:7].mean(0)
            mean = a[on, 1:7].mean(0)-baseline
            row = dict(kind=kind, hz=hz, repeat=path.parent.name, encoder_hz=frequency,
                       source=str(path.relative_to(FOLDER)), off_end_s=float(a[-1, 0]-.1))
            for i, axis in enumerate(('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz')):
                row[f'{axis}_mean'] = float(mean[i])
                row[f'{axis}_baseline'] = float(baseline[i])
                row[f'{axis}_on_std'] = float(a[on, i+1].std())
                row[f'{axis}_off_std'] = float(a[off, i+1].std())
                variants = [a[(a[:, 0] >= l) & (a[:, 0] < r), i+1].mean()
                            - a[(a[:, 0] >= lo) & (a[:, 0] < a[-1, 0]-.1), i+1].mean()
                            for l, r in ((.5, 3.5), (1, 4), (.5, 4)) for lo in (5.5, 6)]
                row[f'{axis}_window_min'] = float(min(variants))
                row[f'{axis}_window_max'] = float(max(variants))
            rows.append(row)
            for p in (path, epath, path.with_name(path.name.replace('.csv.csv', '.csvfc10Hz.csv')),
                      path.with_name(path.name.replace('.csv.csv', '_power.csv'))):
                sources.append(dict(path=str(p.relative_to(FOLDER)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            if hz == 15 and path.parent.name == '01':
                for i in range(3):
                    ax = axes[ki, i]
                    ax.plot(a[:, 0], a[:, i+1]*1000, lw=.3)
                    ax.axhline(baseline[i]*1000, color='black', ls='--', lw=1)
                    ax.axvspan(.5, 4, color='green', alpha=.1)
                    ax.axvspan(5.5, a[-1, 0]-.1, color='orange', alpha=.2)
                    ax.set(title=f'{kind}, 15 Hz, F{"xyz"[i]}', xlabel='Time [s]', ylabel='Raw sensor force [mN]')
    fig.savefig(OUTPUT/'raw_windows.png', dpi=160)
    plt.close(fig)
    write_csv(OUTPUT/'real_runs.csv', rows)
    (OUTPUT/'source_manifest.json').write_text(json.dumps(sources, indent=2))
    (OUTPUT/'real_runs.json').write_text(json.dumps(rows, indent=2))
    groups = []
    for kind in ('curve', 'rect'):
        for hz in range(10, 16):
            subset = [r for r in rows if r['kind'] == kind and r['hz'] == hz]
            row = dict(kind=kind, hz=hz, n=len(subset), encoder_hz=np.mean([r['encoder_hz'] for r in subset]))
            for axis in ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'):
                row[f'{axis}_mean'] = np.mean([r[f'{axis}_mean'] for r in subset])
                row[f'{axis}_sd'] = np.std([r[f'{axis}_mean'] for r in subset], ddof=1)
            groups.append(row)
            print(kind, hz, 'encoder', round(row['encoder_hz'], 4),
                  'Fx,Fy,Fz mN:', np.round([row[f'{a}_mean']*1000 for a in ('Fx', 'Fy', 'Fz')], 3))
    write_csv(OUTPUT/'real_grouped.csv', groups)
    (OUTPUT/'real_grouped.json').write_text(json.dumps(groups, indent=2))
    return rows, groups


if __name__ == '__main__':
    analyze_real()
