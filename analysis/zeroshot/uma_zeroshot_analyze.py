import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MU_OC20 = {'H': -3.477, 'O': -7.204, 'C': -7.282, 'N': -8.083}
COMPOSITION = {
    '*CH2*CH2': {'C': 2, 'H': 4},
    '*CO': {'C': 1, 'O': 1},
    '*COHCH2': {'C': 2, 'H': 3, 'O': 1},
    '*NH2NH2': {'N': 2, 'H': 4},
    '*NNCH3': {'N': 2, 'C': 1, 'H': 3},
    '*OCHCH2': {'C': 2, 'H': 3, 'O': 1},
    '*ONNO2': {'N': 2, 'O': 3},
}
ALPHA, DRAWS, TOL = 0.05 / 3, 2000, 0.1


def build():
    m = pd.read_csv('results/zeroshot/uma_zeroshot_manifest.csv')
    raw = pd.DataFrame([json.loads(l) for l in open('results/zeroshot/uma_zeroshot_raw.jsonl')])
    raw = raw[raw.status == 'ok'].drop_duplicates('key', keep='last')
    m = m.merge(raw[['key', 'pred_total_eV', 'force_mae_free', 'force_mae_all', 'runtime_s']], on='key', how='left')
    slab = m[m.kind == 'slab'].set_index('slab_id')
    a = m[m.kind == 'adslab'].copy()
    a['slab_pred'] = a.slab_id.map(slab.pred_total_eV)
    a['slab_mag'] = a.slab_id.map(slab.total_magnetization)
    a['gas_oc20'] = a.adsorbate.map(lambda x: sum(n * MU_OC20[e] for e, n in COMPOSITION[x].items()))
    a['eads_uma'] = a.pred_total_eV - a.slab_pred - a.gas_oc20
    a['err'] = a.eads_uma - a.adsorption_energy
    a['abs_err'] = a.err.abs()
    a['dmag'] = (a.total_magnetization - a.slab_mag).abs()
    f = pd.read_csv('results/zeroshot/aqcat25_test_ood_both_final_frames.csv')[
        ['relaxation', 'material_class', 'organic']
    ]
    a = a.merge(f, on='relaxation', how='left')
    a['bulk'] = a.slab_id.str.split('_').str[0]
    a['spin_on'] = ~a.is_spin_off
    a['paper_filter_ok'] = (a.fmax <= 50) & (a.adsorption_energy.abs() <= 10)
    return a


def stats(d, weights):
    out = dict(mae=d.abs_err.mean(), fail_rate=(d.abs_err > TOL).mean())
    bias = d.groupby('adsorbate').err.transform('mean')
    out['mae_debiased'] = (d.err - bias).abs().mean()
    out['bias_share'] = 1 - out['mae_debiased'] / out['mae']
    diffs, ws = [], []
    for c, g in d.groupby('material_class'):
        on, off = g[g.spin_on].abs_err, g[~g.spin_on].abs_err
        if len(on) and len(off):
            diffs.append(on.mean() - off.mean())
            ws.append(weights[c])
    out['T1_spin_diff_eV'] = np.average(diffs, weights=ws)
    nm = d.material_class == 'non-metal'
    out['T2_nonmetal_ratio'] = d[nm].abs_err.mean() / d[~nm].abs_err.mean()
    s = d[d.spin_on]
    out['T3_spearman_dmag'] = spearmanr(s.abs_err, s.dmag).statistic
    return out


def bootstrap(d, weights):
    rng = np.random.default_rng(0)
    groups = {b: g for b, g in d.groupby('bulk')}
    bulks = np.array(list(groups))
    draws = []
    for _ in range(DRAWS):
        sample = pd.concat([groups[b] for b in rng.choice(bulks, len(bulks))], ignore_index=True)
        draws.append(stats(sample, weights))
    draws = pd.DataFrame(draws)
    return {
        k: [float(np.nanquantile(draws[k], ALPHA / 2)), float(np.nanquantile(draws[k], 1 - ALPHA / 2))] for k in draws
    }


def frame_metrics(d):
    return pd.Series(
        dict(
            frames=len(d),
            relaxations=d.relaxation.nunique(),
            energy_mae=d.abs_err.mean(),
            energy_median=d.abs_err.median(),
            pct_le_0p2=100 * (d.abs_err <= 0.2).mean(),
            pct_le_0p1=100 * (d.abs_err <= TOL).mean(),
            mean_signed=d.err.mean(),
            force_mae_all=d.force_mae_all.mean(),
            force_mae_free=d.force_mae_free.mean(),
        )
    )


def main():
    a = build()
    missing = a.pred_total_eV.isna() | a.slab_pred.isna()
    a.to_csv('results/zeroshot/uma_zeroshot_frames.csv', index=False)
    use = a[~missing & a.paper_filter_ok]
    final = use[use.is_final & ~use.organic.astype(bool)]
    weights = final.material_class.value_counts(normalize=True).to_dict()
    point = stats(final, weights)
    ci = bootstrap(final, weights)
    summary = dict(
        n_adslab_frames=len(a),
        n_missing_prediction=int(missing.sum()),
        n_filtered_out=int((~a.paper_filter_ok).sum()),
        n_final_test=len(final),
        n_bulks_test=final.bulk.nunique(),
        class_weights=weights,
        ci_level=1 - ALPHA,
        point={k: float(v) for k, v in point.items()},
        ci=ci,
    )
    print(json.dumps(summary, indent=1))

    tables = {}
    allfr = use[~use.organic.astype(bool)]
    tables['per_frame_paper_style'] = pd.DataFrame(
        {
            'non-organic, all frames': frame_metrics(allfr),
            'non-organic, final': frame_metrics(final),
            'organic, all frames': frame_metrics(use[use.organic.astype(bool)]),
            'all, all frames (incl. organic)': frame_metrics(use),
            'unfiltered, all frames': frame_metrics(a[~missing]),
        }
    ).T
    tables['final_by_class'] = final.groupby('material_class').apply(frame_metrics)
    tables['final_by_class_spin'] = final.groupby(['material_class', 'spin_on']).apply(frame_metrics)
    tables['final_by_adsorbate'] = final.groupby('adsorbate').apply(frame_metrics)
    tables['worst10_final'] = final.nlargest(10, 'abs_err')[
        [
            'relaxation',
            'slab_id',
            'adsorbate',
            'material_class',
            'spin_on',
            'adsorption_energy',
            'eads_uma',
            'err',
            'dmag',
        ]
    ].set_index('relaxation')
    for name, t in tables.items():
        print(f'\n### {name}\n')
        print(t.round(4).to_markdown())
        summary.setdefault('tables', {})[name] = json.loads(t.reset_index().to_json(orient='records'))
    json.dump(summary, open('results/zeroshot/uma_zeroshot_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
