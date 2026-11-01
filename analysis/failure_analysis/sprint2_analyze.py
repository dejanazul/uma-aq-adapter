import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from ase.io import iread
from scipy.stats import spearmanr

sys.path.insert(0, 'analysis/zeroshot')
from uma_zeroshot_analyze import COMPOSITION, MU_OC20

D = 'results/failure_analysis/sprint2/'
TASKS = ('oc20', 'omat')
NEW = {'Ba', 'Ce', 'F', 'La', 'Li', 'Mg'}
ALPHA, DRAWS, TOL = 0.05 / 5, 2000, 0.1
MANIFEST = {'test': 'results/zeroshot/uma_zeroshot_manifest.csv', 'val': D + 'sprint2_val_manifest.csv'}
md = lambda x: print(x.round(4).to_markdown(), end='\n\n')


def composition(name):
    count, s = Counter(), name.replace('*', '')
    while '(' in s:
        s = re.sub(r'\(([^()]*)\)(\d*)', lambda m: m.group(1) * int(m.group(2) or 1), s)
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', s):
        count[el] += int(n or 1)
    return dict(count)


assert all(composition(k) == v for k, v in COMPOSITION.items())
assert composition('*NH2N(CH3)2') == {'N': 2, 'H': 8, 'C': 2} and composition('*COCH2OH') == {'C': 2, 'O': 2, 'H': 3}


def gas_mu():
    gas = pd.DataFrame(json.load(open(D + 'raw_val/gas.json')))
    assert gas.converged.all(), 'gate 3: gas relaxation not converged'
    mu = {}
    for task, g in gas.groupby('task'):
        e = g.set_index('molecule').energy_eV
        h = e.H2 / 2
        o = e.H2O - 2 * h
        mu[task] = dict(H=h, O=o, C=e.CO - o, N=e.N2 / 2)
    return mu, gas.drop(columns='positions')


def build(split, mu):
    m = pd.read_csv(MANIFEST[split])
    raw = pd.read_json(D + f'raw_{split}/rows.jsonl', lines=True).drop_duplicates('key', keep='last')
    status = m.key.map(raw.set_index('key').status).fillna('missing')
    m = m.merge(raw[raw.status == 'ok'].drop(columns=['n_atoms', 'status']), on='key', how='left')
    slab = m[m.kind == 'slab'].set_index('slab_id')
    a = m[m.kind == 'adslab'].copy()
    a['slab_mag'] = a.slab_id.map(slab.total_magnetization)
    a['dmag'] = (a.total_magnetization - a.slab_mag).abs()
    for t in TASKS:
        ref = MU_OC20 if t == 'oc20' else mu['omat']
        gas = a.adsorbate.map(lambda x: sum(n * ref[e] for e, n in composition(x).items()))
        a[f'eads_{t}'] = a[f'pred_total_eV_{t}'] - a.slab_id.map(slab[f'pred_total_eV_{t}']) - gas
        a[f'err_{t}'] = a[f'eads_{t}'] - a.adsorption_energy
        a[f'abs_err_{t}'] = a[f'err_{t}'].abs()
    f = pd.read_csv(f'results/zeroshot/aqcat25_{split}_ood_both_final_frames.csv').set_index('relaxation')
    a['material_class'] = a.relaxation.map(f.material_class)
    a['organic'] = a.relaxation.map(f.organic).astype(bool)
    a['new_element'] = a.relaxation.map(f.slab_elements.str.split().map(lambda e: bool(set(e) & NEW)))
    a['bulk'] = a.slab_id.str.split('_').str[0]
    a['spin_on'] = ~a.is_spin_off
    a['paper_filter_ok'] = (a.fmax <= 50) & (a.adsorption_energy.abs() <= 10)
    a['split'] = split
    return a, status


def stats(d, w_spin, w_cell):
    out = {f'mae_{t}': d[f'abs_err_{t}'].mean() for t in TASKS}
    out['T1_omat_minus_oc20'] = out['mae_omat'] - out['mae_oc20']
    s = d[d.spin_on]
    out['T2_omat_minus_oc20_spin_on'] = s.abs_err_omat.mean() - s.abs_err_oc20.mean()
    for t in TASKS:
        out[f'mae_b_{t}'] = (d[f'err_{t}'] - d.groupby('adsorbate')[f'err_{t}'].transform('mean')).abs().mean()
    out['diff_b_omat_minus_oc20'] = out['mae_b_omat'] - out['mae_b_oc20']
    diffs, ws = [], []
    for k, g in d.groupby('spin_on'):
        new, old = g[g.new_element].abs_err_oc20, g[~g.new_element].abs_err_oc20
        if len(new) and len(old):
            diffs.append(new.mean() - old.mean())
            ws.append(w_spin[k])
    out['T3_new_element_diff'] = np.average(diffs, weights=ws) if diffs else np.nan
    diffs, ws = [], []
    for k, g in d.groupby(['material_class', 'new_element']):
        on, off = g[g.spin_on].abs_err_oc20, g[~g.spin_on].abs_err_oc20
        if len(on) and len(off):
            diffs.append(on.mean() - off.mean())
            ws.append(w_cell[k])
    out['T4_spin_diff'] = np.average(diffs, weights=ws) if diffs else np.nan
    out['T5_spearman_dmag'] = spearmanr(s.abs_err_oc20, s.dmag).statistic
    return out


def bootstrap(d, w_spin, w_cell):
    rng = np.random.default_rng(0)
    groups = {b: g for b, g in d.groupby('bulk')}
    bulks = np.array(list(groups))
    draws = pd.DataFrame(
        [
            stats(pd.concat([groups[b] for b in rng.choice(bulks, len(bulks))], ignore_index=True), w_spin, w_cell)
            for _ in range(DRAWS)
        ]
    )
    return {
        k: [float(np.nanquantile(draws[k], ALPHA / 2)), float(np.nanquantile(draws[k], 1 - ALPHA / 2))] for k in draws
    }


def run_tests(final):
    w_spin = final.spin_on.value_counts(normalize=True).to_dict()
    w_cell = final.groupby(['material_class', 'new_element']).size().div(len(final)).to_dict()
    point, ci = stats(final, w_spin, w_cell), bootstrap(final, w_spin, w_cell)
    cells = final.groupby(['material_class', 'new_element']).spin_on.agg(['sum', 'size'])
    dropped = cells[(cells['sum'] == 0) | (cells['sum'] == cells['size'])]
    return point, ci, dropped


def verdict(ci):
    lo, hi = ci
    return 'pass (+)' if lo > 0 else 'pass (-)' if hi < 0 else 'fail'


def frame_metrics(d, t):
    return pd.Series(
        dict(
            frames=len(d),
            relaxations=d.relaxation.nunique(),
            energy_mae=d[f'abs_err_{t}'].mean(),
            energy_median=d[f'abs_err_{t}'].median(),
            pct_le_0p2=100 * (d[f'abs_err_{t}'] <= 0.2).mean(),
            pct_le_0p1=100 * (d[f'abs_err_{t}'] <= TOL).mean(),
            mean_signed=d[f'err_{t}'].mean(),
            force_mae_free=d[f'force_mae_free_{t}'].mean(),
            force_vec_err_mean_free=d[f'force_vec_err_mean_free_{t}'].mean(),
        )
    )


def unconverged_slabs():
    bad = set()
    for atoms in iread('data/zeroshot/uma_zeroshot_frames_val.extxyz', index=':'):
        if atoms.info['kind'] == 'slab':
            free = atoms.get_tags() != 0
            if np.linalg.norm(atoms.get_forces()[free], axis=1).max() > 0.05:
                bad.add(atoms.info['key'].split(':', 1)[1])
    m = pd.read_csv(MANIFEST['val'])
    return set(m.loc[m.key.str.removeprefix('val:').isin(bad), 'slab_id'])


def main():
    mu, gas = gas_mu()
    print('## Gates\n')
    frames, summary = [], {'mu': mu}
    for split in ('test', 'val'):
        a, status = build(split, mu)
        missing = a[[f'eads_{t}' for t in TASKS]].isna().any(axis=1)
        print(f'- {split}: status {status.value_counts().to_dict()}; adslab frames without E_ads: {int(missing.sum())}')
        frames.append(a)
    a = pd.concat(frames, ignore_index=True)
    a.to_csv(D + 'sprint2_frames.csv', index=False)
    z = pd.read_csv('results/zeroshot/uma_zeroshot_frames.csv').set_index('key')
    t = a[a.split == 'test']
    rep = (t.err_oc20 - t.key.map(z.err)).abs().max()
    print(f'- test oc20 vs stage Z err, max |diff| = {rep:.2e} eV\n')
    summary['test_oc20_vs_stageZ_max_diff'] = float(rep)

    print('## Gas references\n')
    md(gas)
    md(
        pd.DataFrame(
            {**mu, 'OC20 official': MU_OC20, 'AQCat25': {'H': -3.4944, 'O': -7.1590, 'C': -7.2654, 'N': -8.1351}}
        )
    )

    use = a[a.paper_filter_ok & a.err_oc20.notna() & a.err_omat.notna()]
    final = use[use.is_final & ~use.organic]
    val = final[final.split == 'val']
    print(
        f'## Tests (val, non-organic final frames passing the filter: {len(val)} relaxations, {val.bulk.nunique()} bulks)\n'
    )
    bad = unconverged_slabs()
    sens = val[~val.slab_id.isin(bad)]
    print(f'Sensitivity: {len(bad)} unconverged slabs, {len(val) - len(sens)} relaxations removed\n')
    rows = []
    for name, d in (('main', val), ('sensitivity', sens)):
        point, ci, dropped = run_tests(d)
        summary[name] = dict(
            point=point,
            ci=ci,
            n=len(d),
            n_bulks=d.bulk.nunique(),
            T4_dropped_cells=[list(map(str, k)) for k in dropped.index],
        )
        print(
            f'{name}: T4 cells without both spin groups: {len(dropped)} '
            f'({", ".join(f"{c}/{n}" for c, n in dropped.index) or "-"})\n'
        )
        for k in (
            'T1_omat_minus_oc20',
            'T2_omat_minus_oc20_spin_on',
            'T3_new_element_diff',
            'T4_spin_diff',
            'T5_spearman_dmag',
        ):
            rows.append(
                dict(run=name, test=k, value=point[k], ci_low=ci[k][0], ci_high=ci[k][1], verdict=verdict(ci[k]))
            )
        rows.append(
            dict(
                run=name,
                test='diff_b_omat_minus_oc20 (descriptive)',
                value=point['diff_b_omat_minus_oc20'],
                ci_low=ci['diff_b_omat_minus_oc20'][0],
                ci_high=ci['diff_b_omat_minus_oc20'][1],
                verdict='-',
            )
        )
        for k in ('mae_oc20', 'mae_omat', 'mae_b_oc20', 'mae_b_omat'):
            rows.append(dict(run=name, test=k, value=point[k], ci_low=ci[k][0], ci_high=ci[k][1], verdict='-'))
    md(pd.DataFrame(rows).set_index(['run', 'test']))
    p = summary['main']['point']
    print(
        f'T1 relative to mae_oc20: {100 * p["T1_omat_minus_oc20"] / p["mae_oc20"]:.1f} %; '
        f'T2 relative to spin-on mae_oc20: {100 * p["T2_omat_minus_oc20_spin_on"] / val[val.spin_on].abs_err_oc20.mean():.1f} %; '
        f'share of T1 removed by (b): {100 * (1 - p["diff_b_omat_minus_oc20"] / p["T1_omat_minus_oc20"]):.1f} %\n'
    )

    print('## Descriptive\n')
    for split in ('test', 'val'):
        u = use[(use.split == split) & ~use.organic]
        print(f'### {split}: per-frame metrics (non-organic, passing the filter)\n')
        md(
            pd.DataFrame(
                {f'{t}, all frames': frame_metrics(u, t) for t in TASKS}
                | {f'{t}, final': frame_metrics(u[u.is_final], t) for t in TASKS}
            ).T
        )
        f = final[final.split == split]
        print(f'### {split}: final frames, spin x new element x task (MAE eV, n)\n')
        md(
            f.groupby(['spin_on', 'new_element']).agg(
                n=('relaxation', 'size'),
                bulks=('bulk', 'nunique'),
                mae_oc20=('abs_err_oc20', 'mean'),
                mae_omat=('abs_err_omat', 'mean'),
            )
        )
        print(f'### {split}: final frames per material class\n')
        md(
            f.groupby('material_class').agg(
                n=('relaxation', 'size'), mae_oc20=('abs_err_oc20', 'mean'), mae_omat=('abs_err_omat', 'mean')
            )
        )
        print(f'### {split}: final frames per adsorbate (signed error = bias)\n')
        md(
            f.groupby('adsorbate').agg(
                n=('relaxation', 'size'),
                mean_err_oc20=('err_oc20', 'mean'),
                mean_err_omat=('err_omat', 'mean'),
                mae_oc20=('abs_err_oc20', 'mean'),
                mae_omat=('abs_err_omat', 'mean'),
            )
        )
        o = use[(use.split == split) & use.is_final & use.organic]
        print(f'### {split}: organic (final frames, {len(o)} relaxations)\n')
        md(pd.DataFrame({t: frame_metrics(o, t) for t in TASKS}).T)
    cols = [
        'relaxation',
        'slab_id',
        'adsorbate',
        'material_class',
        'spin_on',
        'new_element',
        'adsorption_energy',
        'eads_oc20',
        'eads_omat',
        'err_oc20',
        'err_omat',
        'dmag',
    ]
    for t in TASKS:
        print(f'### val: 10 worst cases, {t}\n')
        md(val.nlargest(10, f'abs_err_{t}')[cols].set_index('relaxation'))
    json.dump(summary, open(D + 'sprint2_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
