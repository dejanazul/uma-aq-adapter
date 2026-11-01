import hashlib
import json
import os
import pickle
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = 'results/failure_analysis/sprint4/'
SPLITS = ('test', 'val')
DRAWS, MIN_BULKS, EFFECT = 2000, 5, 0.1
A_MAIN, A_DESC = 0.05 / 4, 0.05
ELEMENTS = ('F', 'Mn', 'Fe')
FLAGS = ('desorbed', 'dissociated', 'intercalated', 'surface_changed')
STAGE_U = 'results/oc20_stage_u/nonmetal_debug/stageU_{}_table.csv'
md = lambda x: print(x.round(4).to_markdown(), end='\n\n')


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {
            ('numpy.core.multiarray', 'scalar'),
            ('numpy._core.multiarray', 'scalar'),
            ('numpy', 'dtype'),
        }:
            return super().find_class(module.replace('numpy.core', 'numpy._core'), name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def miller(m):
    return ''.join(str(int(i)) if i >= 0 else f'-{-int(i)}' for i in m)


def draw_counts(d):
    bulks = np.array(sorted(d.bulk.unique()))
    idx = np.random.default_rng(0).integers(0, len(bulks), (DRAWS, len(bulks)))
    return bulks, np.stack([np.bincount(i, minlength=len(bulks)) for i in idx])


def ci(x, alpha):
    return [float(np.nanquantile(x, alpha / 2)), float(np.nanquantile(x, 1 - alpha / 2))]


def per_bulk(d, mask, col, bulks):
    g = pd.DataFrame(dict(s=np.where(mask, d[col], 0.0), n=mask.astype(int))).groupby(d.bulk.values).sum()
    return g.reindex(bulks, fill_value=0)


def strat_effect(d, x, col, C, bulks, alpha):
    rows, parts = [], []
    for s in sorted(d.stratum.unique()):
        ins, out = (d.stratum == s) & x, (d.stratum == s) & ~x
        nb = d.bulk[ins].nunique()
        ok = nb >= MIN_BULKS and out.any()
        rows.append(
            dict(
                stratum=s,
                n_x=int(ins.sum()),
                bulks_x=nb,
                n_rest=int(out.sum()),
                mae_x=d[col][ins].mean() if ins.any() else np.nan,
                mae_rest=d[col][out].mean(),
                included=ok,
            )
        )
        if ok:
            parts.append((ins.values, out.values))
    if not parts:
        return dict(point=np.nan, ci=[np.nan, np.nan], strata=rows)
    num_p, w_p, num_b, w_b = 0.0, 0.0, 0.0, 0.0
    with np.errstate(invalid='ignore', divide='ignore'):
        for ins, out in parts:
            a, b = per_bulk(d, ins, col, bulks), per_bulk(d, out, col, bulks)
            diff_p = d[col][ins].mean() - d[col][out].mean()
            num_p, w_p = num_p + ins.sum() * diff_p, w_p + ins.sum()
            na = C @ a.n.values
            diff_b = C @ a.s.values / na - C @ b.s.values / (C @ b.n.values)
            num_b, w_b = num_b + na * np.nan_to_num(diff_b), w_b + np.where(np.isnan(diff_b), 0, na)
        draws = num_b / w_b
    return dict(point=num_p / w_p, ci=ci(draws, alpha), strata=rows)


def h4_effect(d, col, C, bulks, alpha):
    on, off = (d.stratum == 'S1').values, (d.stratum == 'S0').values
    a, b = per_bulk(d, on, col, bulks), per_bulk(d, off, col, bulks)
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = C @ a.s.values / (C @ a.n.values) - C @ b.s.values / (C @ b.n.values)
    return dict(
        point=d[col][on].mean() - d[col][off].mean(),
        ci=ci(draws, alpha),
        strata=[
            dict(stratum='S1', n=int(on.sum()), bulks=d.bulk[on].nunique()),
            dict(stratum='S0', n=int(off.sum()), bulks=d.bulk[off].nunique()),
        ],
    )


def verdict_element(t, v):
    if any(np.isnan(r['point']) for r in (t, v)):
        return 'not assessable'
    holds = [r['point'] >= EFFECT and r['ci'][0] > 0 for r in (t, v)]
    explained = [(r['ci'][0] <= 0 <= r['ci'][1]) or r['point'] < EFFECT for r in (t, v)]
    return 'holds' if all(holds) else 'explained' if all(explained) else 'partial'


def verdict_h4(t, v):
    if all(r['ci'][1] < EFFECT for r in (t, v)):
        return 'pass (spin flag has no effect)'
    if all(r['point'] >= EFFECT and r['ci'][0] > 0 for r in (t, v)):
        return 'fail (spin flag still has an effect)'
    return 'inconclusive'


def hypotheses(e, col='abs_err_oc20', alpha=A_MAIN):
    out = {}
    for split in SPLITS:
        d = e[e.split == split].reset_index(drop=True)
        bulks, C = draw_counts(d)
        out[split] = {
            f'H{i + 1}_{el}': strat_effect(d, d[el].values, col, C, bulks, alpha) for i, el in enumerate(ELEMENTS)
        }
        out[split]['H4_spin_flag'] = h4_effect(d, col, C, bulks, alpha)
    return out


def table(res, s3=None):
    rows = []
    for h in res['test']:
        t, v = res['test'][h], res['val'][h]
        r = dict(
            hypothesis=h,
            test=t['point'],
            test_ci_low=t['ci'][0],
            test_ci_high=t['ci'][1],
            val=v['point'],
            val_ci_low=v['ci'][0],
            val_ci_high=v['ci'][1],
            verdict=verdict_h4(t, v) if h.startswith('H4') else verdict_element(t, v),
        )
        if s3 is not None and not h.startswith('H4'):
            el = h.split('_')[1]
            r['share_left_test'], r['share_left_val'] = t['point'] / s3[('test', el)], v['point'] / s3[('val', el)]
        rows.append(r)
    return pd.DataFrame(rows).set_index('hypothesis')


def regression(e, cols):
    rows = []
    for split in SPLITS:
        d = e[e.split == split].reset_index(drop=True)
        X = np.column_stack([np.ones(len(d))] + [d[c].astype(float).values for c in cols])
        y = d.abs_err_oc20.values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        bulks, C = draw_counts(d)
        rows_of = d.groupby('bulk').indices
        draws = []
        for counts in C:
            i = np.concatenate([np.repeat(rows_of[b], c) for b, c in zip(bulks, counts) if c])
            draws.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
        draws = np.array(draws)
        for j, c in enumerate(['intercept'] + cols):
            lo, hi = ci(draws[:, j], A_DESC)
            rows.append(dict(split=split, term=c, coef=beta[j], ci_low=lo, ci_high=hi))
    t = pd.DataFrame(rows).pivot(index='term', columns='split', values=['coef', 'ci_low', 'ci_high'])
    t.columns = [f'{a}_{b}' for a, b in t.columns]
    return t.loc[['intercept'] + cols]


def ood_proxies(e, path):
    with open(path, 'rb') as f:
        mapping = NumpyOnlyUnpickler(f).load()
    train = {
        frozenset(re.findall(r'[A-Z][a-z]?', v['bulk_symbols'])) for v in mapping.values() if v['split'] == 'train'
    }
    train_el = set().union(*train)
    sets = e.slab_elements.str.split().map(frozenset)
    e['chemsys_in_oc20'] = sets.map(lambda s: s in train)
    e['max_jaccard_oc20'] = sets.map(lambda s: max(len(s & b) / len(s | b) for b in train))
    e['n_new_elements'] = sets.map(lambda s: len(s - train_el))
    return mapping, len(train), len(train_el)


def ood_tests(e):
    rows = []
    for split in SPLITS:
        d = e[e.split == split].reset_index(drop=True)
        bulks, C = draw_counts(d)
        rows_of = d.groupby('bulk').indices
        idx = [np.concatenate([np.repeat(rows_of[b], c) for b, c in zip(bulks, counts) if c]) for counts in C]
        for f in ('max_jaccard_oc20', 'n_new_elements'):
            x, y = d[f].astype(float).values, d.abs_err_oc20.values
            draws = [spearmanr(x[i], y[i]).statistic for i in idx]
            lo, hi = ci(draws, A_DESC)
            rows.append(
                dict(
                    split=split,
                    proxy=f,
                    kind='spearman',
                    value=spearmanr(x, y).statistic,
                    ci_low=lo,
                    ci_high=hi,
                    bulks_in=np.nan,
                )
            )
        m = d.chemsys_in_oc20.values
        a, b = per_bulk(d, m, 'abs_err_oc20', bulks), per_bulk(d, ~m, 'abs_err_oc20', bulks)
        with np.errstate(invalid='ignore', divide='ignore'):
            draws = C @ a.s.values / (C @ a.n.values) - C @ b.s.values / (C @ b.n.values)
        lo, hi = ci(draws, A_DESC)
        rows.append(
            dict(
                split=split,
                proxy='chemsys_in_oc20',
                kind='MAE(in) - MAE(rest)',
                value=d.abs_err_oc20[m].mean() - d.abs_err_oc20[~m].mean() if m.any() else np.nan,
                ci_low=lo,
                ci_high=hi,
                bulks_in=d.bulk[m].nunique(),
            )
        )
    t = pd.DataFrame(rows)
    verdicts = {}
    for p, g in t.groupby('proxy'):
        ok = []
        for r in g.itertuples():
            excl = r.ci_low > 0 or r.ci_high < 0
            ok.append(
                excl
                and (
                    abs(r.value) >= 0.10
                    if r.kind == 'spearman'
                    else (r.bulks_in >= MIN_BULKS and abs(r.value) >= EFFECT)
                )
            )
        signs = np.sign(g.value.values)
        verdicts[p] = (
            'candidate'
            if all(ok) and signs[0] == signs[1]
            else 'mixed (opposite signs)'
            if all(ok)
            else 'not replicated'
            if any(ok)
            else '-'
        )
    t['verdict'] = t.proxy.map(verdicts)
    return t


def case_study(e, mapping):
    u = pd.concat([pd.read_csv(STAGE_U.format(s)).assign(set=s) for s in ('discovery', 'confirm')])
    u['mpid'] = u.system_id.map(lambda i: mapping[i]['bulk_mpid'])
    u['miller'] = u.system_id.map(lambda i: miller(mapping[i]['miller_index']))
    u['top'] = u.system_id.map(lambda i: bool(mapping[i]['top']))
    u['ads'] = u.system_id.map(lambda i: mapping[i]['ads_symbols'])
    v = e[e.split == 'val'].copy()
    p = v.slab_id.str.split('_')
    v['mpid'], v['miller'], v['top'] = p.str[0], p.str[1], p.str[3] == 'True'
    m = v.merge(u[['mpid', 'miller', 'top', 'system_id', 'ads', 'dE_U']], on=['mpid', 'miller', 'top'])
    return m.groupby(['slab_id', 'chemsys', 'material_class']).agg(
        aq_n=('relaxation', 'nunique'),
        aq_mae=('abs_err_oc20', 'mean'),
        aq_mean_dmag=('dmag', 'mean'),
        oc20_ids=('system_id', lambda s: ' '.join(sorted(set(s)))),
        oc20_ads=('ads', lambda s: ' '.join(sorted(set(s)))),
        oc20_abs_err=('dE_U', lambda s: s.drop_duplicates().abs().mean()),
    )


def forest(main, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    hs = list(main.index)
    for k, (split, c) in enumerate((('test', 'tab:blue'), ('val', 'tab:orange'))):
        y = np.arange(len(hs)) + (k - 0.5) * 0.25
        x = main[split].values
        ax.errorbar(
            x,
            y,
            xerr=[x - main[f'{split}_ci_low'].values, main[f'{split}_ci_high'].values - x],
            fmt='o',
            c=c,
            label=split,
            capsize=3,
        )
    ax.axvline(0, c='k', lw=0.8)
    ax.axvline(EFFECT, c='gray', ls=':', label='0.1 eV')
    ax.set_yticks(range(len(hs)), hs)
    ax.invert_yaxis()
    ax.set(xlabel='controlled MAE difference (eV), CI 98.75 %')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def strata_plot(res, path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for a, el in zip(ax, ELEMENTS):
        for k, (split, c) in enumerate((('test', 'tab:blue'), ('val', 'tab:orange'))):
            st = pd.DataFrame(res[split][f'H{ELEMENTS.index(el) + 1}_{el}']['strata'])
            x = np.arange(len(st)) + (k - 0.5) * 0.35
            a.bar(x, st.mae_x, 0.35, color=c, label=f'{split} {el}')
            a.scatter(x, st.mae_rest, c='k', marker='_', s=200, zorder=3)
            for xi, r in zip(x, st.itertuples()):
                a.text(
                    xi, 0.01, f'{r.bulks_x}b', ha='center', fontsize=6, rotation=90, color='w' if r.included else 'r'
                )
        a.set_xticks(range(len(st)), st.stratum)
        a.set_title(f'{el} (bar) vs the rest (black line)')
        a.legend(fontsize=7)
    ax[0].set_ylabel('MAE |ΔE_ads| (eV)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    path = sys.argv[1]
    md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
    assert md5.startswith('6b5d4850'), md5
    e = pd.read_csv('results/failure_analysis/sprint3/sprint3_relaxations.csv')
    assert e.groupby('split').size().to_dict() == {'test': 932, 'val': 963}
    els = e.slab_elements.str.split()
    for el in ELEMENTS:
        e[el] = els.map(lambda x, el=el: el in x)
    cut = e[e.spin_on].dmag.quantile([1 / 3, 2 / 3]).values
    e['stratum'] = np.where(~e.spin_on, 'S0', np.where(e.dmag <= cut[0], 'S1', np.where(e.dmag <= cut[1], 'S2', 'S3')))
    mapping, n_sys, n_el = ood_proxies(e, path)

    print('## Gates and definitions\n')
    print(f'- mapping MD5 {md5}; OC20 train bulk chemical systems: {n_sys}, elements: {n_el}')
    print(f'- population {e.groupby("split").size().to_dict()}, bulks {e.groupby("split").bulk.nunique().to_dict()}')
    print(f'- |dmag| tertile cut points (spin-on, test+val pooled): {cut[0]:.4f}, {cut[1]:.4f} μB\n')
    md(
        e.groupby(['split', 'stratum'])
        .agg(
            n=('relaxation', 'size'),
            bulks=('bulk', 'nunique'),
            mae=('abs_err_oc20', 'mean'),
            dmag_median=('dmag', 'median'),
        )
        .unstack(0)
    )

    print('## Main hypotheses H1–H4 (|ΔE_ads|, CI 98.75 %)\n')
    s3 = {}
    for split in SPLITS:
        d = e[e.split == split]
        for el in ELEMENTS:
            s3[(split, el)] = d.abs_err_oc20[d[el]].mean() - d.abs_err_oc20[~d[el]].mean()
    res = hypotheses(e)
    main_t = table(res, s3)
    md(main_t)
    for h in res['test']:
        print(f'### Strata {h}\n')
        md(pd.concat({s: pd.DataFrame(res[s][h]['strata']).set_index('stratum') for s in SPLITS}, axis=1))

    print('## F pooled test + val (label: not replicated, pooled; CI 98.75 %)\n')
    d = e.reset_index(drop=True)
    bulks, C = draw_counts(d)
    fj = strat_effect(d, d['F'].values, 'abs_err_oc20', C, bulks, A_MAIN)
    print(f'F pooled: {fj["point"]:.4f} [{fj["ci"][0]:.4f}; {fj["ci"][1]:.4f}]\n')
    md(pd.DataFrame(fj['strata']).set_index('stratum'))

    print('## Second response: signed error (descriptive, CI 95 %)\n')
    signed = table(hypotheses(e, 'err_oc20', A_DESC))
    md(signed.drop(columns='verdict'))

    print('## Sensitivity (CI 98.75 %)\n')
    sens = {}
    for name, keep in (
        ('without AdsorbML flags', ~e[list(FLAGS)].any(axis=1)),
        ('without unconverged slabs (fmax > 0.05)', e.slab_fmax_free <= 0.05),
    ):
        sub = e[keep]
        t = table(hypotheses(sub))
        sens[name] = t
        print(f'### {name}: {sub.groupby("split").size().to_dict()} relaxations\n')
        md(t)
        flips = [h for h in t.index if t.verdict[h] != main_t.verdict[h]]
        print(f'Verdicts changed: {flips or "-"}\n')

    print('## Supporting regression (OLS |ΔE_ads|, eV, CI 95 % bootstrap over bulks)\n')
    for c in ('non-metal', 'metalloid', 'metalloid+non-metal'):
        e[f'class_{c}'] = e.material_class == c
    e['n_atoms_per100'] = e.n_atoms / 100
    cols = list(ELEMENTS) + [
        'dmag',
        'spin_on',
        'n_atoms_per100',
        'class_non-metal',
        'class_metalloid',
        'class_metalloid+non-metal',
        'chemsys_in_oc20',
        'max_jaccard_oc20',
        'n_new_elements',
    ]
    reg = regression(e, cols)
    md(reg)

    print('## Task 20: chemical OOD proxies vs |ΔE_ads| (Sprint 3 rule, CI 95 %)\n')
    md(e.groupby('split')[['chemsys_in_oc20', 'n_new_elements']].agg(['sum', 'mean']))
    print(f'max_jaccard_oc20 per split: {e.groupby("split").max_jaccard_oc20.describe().round(3).to_dict("index")}\n')
    ood = ood_tests(e)
    md(ood.set_index(['proxy', 'split']))

    print('## Case study: surfaces shared with OC20 stage U (descriptive)\n')
    cs = case_study(e, mapping)
    md(cs)

    print('## Descriptive: Cr and Co\n')
    for el in ('Cr', 'Co'):
        m = els.map(lambda x, el=el: el in x)
        md(
            e[m]
            .groupby(['split', 'stratum'])
            .agg(n=('relaxation', 'size'), bulks=('bulk', 'nunique'), mae=('abs_err_oc20', 'mean'))
        )

    os.makedirs(D + 'plots', exist_ok=True)
    forest(main_t, D + 'plots/01_forest_hypotheses.png')
    strata_plot(res, D + 'plots/02_strata_elements.png')
    main_t.to_csv(D + 'sprint4_hypotheses.csv')
    e.to_csv(D + 'sprint4_relaxations.csv', index=False)
    json.dump(
        dict(
            mapping_md5=md5,
            dmag_cuts=cut.tolist(),
            main=main_t.reset_index().to_dict('records'),
            f_pooled=dict(point=fj['point'], ci=fj['ci']),
            signed=signed.reset_index().to_dict('records'),
            sensitivity={k: v.reset_index().to_dict('records') for k, v in sens.items()},
            regression=reg.reset_index().to_dict('records'),
            ood=ood.to_dict('records'),
            case_study=cs.reset_index().to_dict('records'),
        ),
        open(D + 'sprint4_summary.json', 'w'),
        indent=1,
        default=float,
    )


if __name__ == '__main__':
    main()
