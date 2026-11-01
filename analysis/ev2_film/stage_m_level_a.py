import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

D = 'results/ev2_film/'
SRC = 'results/failure_analysis/sprint4/sprint4_relaxations.csv'
SPLITS = ('test', 'val')
DRAWS, EFFECT = 2000, 0.1
A_MAIN, A_DESC = 0.05 / 4, 0.05
CUTS = (0.1408, 1.0890)
FLAGS = ('desorbed', 'dissociated', 'intercalated', 'surface_changed')
md = lambda x: print(x.round(4).to_markdown(), end='\n\n')


def draw_counts(d):
    bulks = np.array(sorted(d.bulk.unique()))
    idx = np.random.default_rng(0).integers(0, len(bulks), (DRAWS, len(bulks)))
    return bulks, np.stack([np.bincount(i, minlength=len(bulks)) for i in idx])


def ci(x, alpha):
    return [float(np.nanquantile(x, alpha / 2)), float(np.nanquantile(x, 1 - alpha / 2))]


def per_bulk(d, mask, col, bulks):
    g = pd.DataFrame(dict(s=np.where(mask, d[col], 0.0), n=mask.astype(int))).groupby(d.bulk.values).sum()
    return g.reindex(bulks, fill_value=0)


def spearman_ci(d, x, y, alpha):
    d = d.reset_index(drop=True)
    bulks, C = draw_counts(d)
    rows_of = d.groupby('bulk').indices
    xv, yv = d[x].astype(float).values, d[y].astype(float).values
    draws = [
        spearmanr(xv[i], yv[i]).statistic
        for i in (np.concatenate([np.repeat(rows_of[b], c) for b, c in zip(bulks, k) if c]) for k in C)
    ]
    return dict(point=float(spearmanr(xv, yv).statistic), ci=ci(draws, alpha), n=len(d), bulks=int(d.bulk.nunique()))


def slab_rows(d):
    on = d[d.spin_on]
    g = on.groupby('slab_id').adm
    return on[g.transform('size').ge(2) & g.transform('nunique').gt(1)].reset_index(drop=True)


def within_slab(d, x, y, alpha):
    g = d.groupby('slab_id')
    rx = g[x].transform(lambda s: rankdata(s) - (len(s) + 1) / 2)
    ry = g[y].transform(lambda s: rankdata(s) - (len(s) + 1) / 2)
    bulks, C = draw_counts(d)
    t = pd.DataFrame(dict(xy=rx * ry, xx=rx * rx, yy=ry * ry)).groupby(d.bulk.values).sum().reindex(bulks)
    s, p = C @ t.values, t.sum().values
    return dict(
        point=float(p[0] / np.sqrt(p[1] * p[2])),
        ci=ci(s[:, 0] / np.sqrt(s[:, 1] * s[:, 2]), alpha),
        n=len(d),
        slabs=int(d.slab_id.nunique()),
        bulks=int(d.bulk.nunique()),
    )


def excl0(r):
    return r['ci'][0] > 0 or r['ci'][1] < 0


def verdict_m1(rs):
    if not all(excl0(r) for r in rs):
        return 'fragile'
    return 'robust' if all(r['point'] >= 0.30 and r['ci'][0] > 0 for r in rs) else 'weakened'


def verdict_m2(t, v):
    rs = (t, v)
    if all(r['point'] >= 0.30 and r['ci'][0] > 0 for r in rs):
        return 'holds (medium)'
    if all(r['point'] >= 0.10 and r['ci'][0] > 0 for r in rs):
        return 'holds (small)'
    if all(not excl0(r) or r['point'] < 0.10 for r in rs):
        return 'vanishes'
    return 'mixed'


def verdict_m3(t, v):
    rs = (t, v)
    if all(r['point'] >= 0.30 and r['ci'][0] > 0 for r in rs):
        return 'direction matches (medium)'
    if all(r['point'] >= 0.10 and r['ci'][0] > 0 for r in rs):
        return 'direction matches (small)'
    if all(r['point'] <= -0.10 and r['ci'][1] < 0 for r in rs):
        return 'opposite direction'
    if all(not excl0(r) or abs(r['point']) < 0.10 for r in rs):
        return 'no direction'
    return 'mixed'


def verdict_m4(t, v):
    rs = (t, v)
    if all(r['point'] >= EFFECT and r['ci'][0] > 0 for r in rs):
        return 'omat more robust (≥ 0.1 eV)'
    if all(r['ci'][0] > 0 for r in rs):
        return 'omat more robust (small)'
    if all(not excl0(r) for r in rs):
        return 'same'
    if all(r['ci'][1] < 0 for r in rs):
        return 'omat worse'
    return 'mixed'


def group_mean(d, mask, col, bulks, C, alpha):
    a = per_bulk(d, mask, col, bulks)
    with np.errstate(invalid='ignore', divide='ignore'):
        draws = C @ a.s.values / (C @ a.n.values)
    return dict(
        point=float(d[col][mask].mean()), ci=ci(draws, alpha), n=int(mask.sum()), bulks=int(d.bulk[mask].nunique())
    ), draws


def did(d, alpha):
    d = d.reset_index(drop=True)
    bulks, C = draw_counts(d)
    pts, drs = {}, {}
    for h in ('oc20', 'omat'):
        for s in ('S1', 'S3'):
            r, dr = group_mean(d, (d.stratum == s).values, f'db_{h}', bulks, C, alpha)
            pts[h, s], drs[h, s] = r['point'], dr
    f = lambda x: (x['oc20', 'S3'] - x['oc20', 'S1']) - (x['omat', 'S3'] - x['omat', 'S1'])
    return dict(
        point=float(f(pts)),
        ci=ci(f(drs), alpha),
        rise_oc20=pts['oc20', 'S3'] - pts['oc20', 'S1'],
        rise_omat=pts['omat', 'S3'] - pts['omat', 'S1'],
    )


def fmt(r):
    return f'{r["point"]:.3f} [{r["ci"][0]:.3f}; {r["ci"][1]:.3f}]'


def scatter_plot(e, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for a, split in zip(ax, SPLITS):
        d = e[(e.split == split) & e.spin_on]
        a.scatter(d.dabs, d.err_oc20, s=8, alpha=0.5, c=np.where(d.dabs < 0, 'tab:blue', 'tab:red'))
        a.set_xscale('symlog', linthresh=0.1)
        a.axhline(0, c='k', lw=0.8)
        a.axvline(0, c='k', lw=0.8)
        for c in CUTS[:1]:
            a.axvspan(-c, c, color='gray', alpha=0.15)
        a.set(
            title=f'{split}: spin-on (blue = quenched, red = increased)',
            xlabel='Δ|m| = |m_ads| − |m_slab| (μB, symlog)',
        )
    ax[0].set_ylabel('signed E_ads oc20 error (eV)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def strata_plot(st, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, split in zip(ax, SPLITS):
        t = st.xs(split, level='split')
        x = np.arange(len(t))
        a.bar(x - 0.2, t['db_oc20'], 0.4, label='oc20 (debiased)')
        a.bar(x + 0.2, t['db_omat'], 0.4, label='omat (debiased)')
        a.set_xticks(x, t.index)
        a.set_title(split)
        a.legend(fontsize=8)
    ax[0].set_ylabel('MAE |ΔE_ads| debiased per adsorbate (eV)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def self_check():
    x = np.tile([0.0, 1, 2, 3], 6)
    slab = np.repeat(np.arange(6), 4)
    d = pd.DataFrame(dict(slab_id=slab, bulk=slab // 2, x=x, y=x - 10 * slab))
    assert abs(within_slab(d, 'x', 'y', A_DESC)['point'] - 1) < 1e-12
    assert spearmanr(d.x, d.y).statistic < 0.5


def main():
    self_check()
    e = pd.read_csv(SRC)
    assert e.groupby('split').size().to_dict() == {'test': 932, 'val': 963}
    e['dabs'] = e.total_magnetization.abs() - e.slab_mag.abs()
    e['adm'] = e.dabs.abs()
    assert (e.adm <= e.dmag + 1e-9).all()
    flip = (e.total_magnetization * e.slab_mag < 0) & (np.minimum(e.total_magnetization.abs(), e.slab_mag.abs()) > 0.1)
    assert flip.groupby(e.split).sum().to_dict() == {'test': 12, 'val': 14}
    e['stratum'] = np.where(~e.spin_on, 'S0', np.where(e.adm <= CUTS[0], 'S1', np.where(e.adm <= CUTS[1], 'S2', 'S3')))
    e['flagged'] = e[list(FLAGS)].any(axis=1)
    for h in ('oc20', 'omat'):
        e[f'db_{h}'] = (e[f'err_{h}'] - e.groupby(['split', 'adsorbate'])[f'err_{h}'].transform('mean')).abs()
    e['abs_err_adsdb'] = e.abs_err_oc20 - e.groupby(['split', 'adsorbate']).abs_err_oc20.transform('mean')
    e['err_adsdb'] = e.err_oc20 - e.groupby(['split', 'adsorbate']).err_oc20.transform('mean')
    out = {}

    print('## Gates and definitions\n')
    print(f'- population {e.groupby("split").size().to_dict()}, bulks {e.groupby("split").bulk.nunique().to_dict()}')
    print(
        f'- moment sign flipped (both |m| > 0.1 μB): {flip.groupby(e.split).sum().to_dict()}; adm ≤ dmag in every row'
    )
    print(
        f'- slab_fmax_free missing: {int(e.slab_fmax_free.isna().sum())}; AdsorbML-flagged: {e.groupby("split").flagged.sum().to_dict()}\n'
    )
    md(
        e.groupby(['split', 'stratum'])
        .agg(
            n=('relaxation', 'size'),
            bulks=('bulk', 'nunique'),
            mae_oc20=('abs_err_oc20', 'mean'),
            adm_median=('adm', 'median'),
        )
        .unstack(0)
    )

    print('## M1 — Spearman(|ΔE_ads|, adm), CI 98.75 %\n')
    subsets = {
        'full': np.ones(len(e), bool),
        '(a) without AdsorbML flags': ~e.flagged,
        '(b) converged reference slab': e.slab_fmax_free <= 0.05,
        '(c) both': ~e.flagged & (e.slab_fmax_free <= 0.05),
    }
    m1 = {
        name: {s: spearman_ci(e[keep & (e.split == s)], 'adm', 'abs_err_oc20', A_MAIN) for s in SPLITS}
        for name, keep in subsets.items()
    }
    md(
        pd.DataFrame(
            {n: {s: f'{fmt(r[s])} (n {r[s]["n"]}, {r[s]["bulks"]} bulk)' for s in SPLITS} for n, r in m1.items()}
        ).T
    )
    out['M1'] = dict(results=m1, verdict=verdict_m1([r[s] for r in m1.values() for s in SPLITS]))
    print(f'**Verdict M1: {out["M1"]["verdict"]}**\n')
    print('Descriptive (CI 95 %):\n')
    desc = {'old dmag, full': ('dmag', np.ones(len(e), bool)), 'adm, spin-on only': ('adm', e.spin_on.values)}
    out['M1_desc'] = {
        n: {s: spearman_ci(e[m & (e.split == s)], c, 'abs_err_oc20', A_DESC) for s in SPLITS}
        for n, (c, m) in desc.items()
    }
    md(pd.DataFrame({n: {s: fmt(r[s]) for s in SPLITS} for n, r in out['M1_desc'].items()}).T)

    print('## M2 — within-slab rank correlation (|ΔE_ads|, adm), spin-on, CI 98.75 %\n')
    rows = {s: slab_rows(e[e.split == s]) for s in SPLITS}
    m2 = {s: within_slab(rows[s], 'adm', 'abs_err_oc20', A_MAIN) for s in SPLITS}
    pooled = {s: spearman_ci(rows[s], 'adm', 'abs_err_oc20', A_DESC) for s in SPLITS}
    sens = {s: within_slab(rows[s], 'adm', 'abs_err_adsdb', A_DESC) for s in SPLITS}
    md(
        pd.DataFrame(
            {
                s: {
                    'within-slab ρ_w (98.75 %)': fmt(m2[s]),
                    'plain Spearman, same rows (95 %)': fmt(pooled[s]),
                    'ρ_w, adsorbate-debiased |error| (95 %)': fmt(sens[s]),
                    'n / slab / bulk': f'{m2[s]["n"]} / {m2[s]["slabs"]} / {m2[s]["bulks"]}',
                }
                for s in SPLITS
            }
        )
    )
    out['M2'] = dict(
        results=m2,
        pooled=pooled,
        adsorbate_debiased=sens,
        verdict=verdict_m2(m2['test'], m2['val']),
        verdict_adsorbate_debiased=verdict_m2(sens['test'], sens['val']),
    )
    print(
        f'**Verdict M2: {out["M2"]["verdict"]}** (adsorbate-debiased sensitivity: {out["M2"]["verdict_adsorbate_debiased"]})\n'
    )

    print('## M3 — direction: within-slab rank correlation (signed error, Δ|m|), spin-on, CI 98.75 %\n')
    m3 = {s: within_slab(rows[s], 'dabs', 'err_oc20', A_MAIN) for s in SPLITS}
    sens3 = {s: within_slab(rows[s], 'dabs', 'err_adsdb', A_DESC) for s in SPLITS}
    glob = {s: spearman_ci(e[(e.split == s) & e.spin_on], 'dabs', 'err_oc20', A_DESC) for s in SPLITS}
    md(
        pd.DataFrame(
            {
                s: {
                    'within-slab ρ_w (98.75 %)': fmt(m3[s]),
                    'ρ_w, adsorbate-debiased error (95 %)': fmt(sens3[s]),
                    'Spearman, all spin-on (95 %)': fmt(glob[s]),
                }
                for s in SPLITS
            }
        )
    )
    out['M3'] = dict(
        results=m3,
        adsorbate_debiased=sens3,
        spearman_spin_on=glob,
        verdict=verdict_m3(m3['test'], m3['val']),
        verdict_adsorbate_debiased=verdict_m3(sens3['test'], sens3['val']),
    )
    print(
        f'**Verdict M3: {out["M3"]["verdict"]}** (adsorbate-debiased sensitivity: {out["M3"]["verdict_adsorbate_debiased"]})\n'
    )
    print('Signed error per group (spin-on, CI 95 %; prediction: quenched < 0, increased > 0):\n')
    grp = {}
    for s in SPLITS:
        d = e[(e.split == s) & e.spin_on].reset_index(drop=True)
        bulks, C = draw_counts(d)
        q, dq = group_mean(d, (d.dabs < -CUTS[0]).values, 'err_oc20', bulks, C, A_DESC)
        z, _ = group_mean(d, (d.dabs.abs() <= CUTS[0]).values, 'err_oc20', bulks, C, A_DESC)
        n, dn = group_mean(d, (d.dabs > CUTS[0]).values, 'err_oc20', bulks, C, A_DESC)
        grp[s] = dict(
            quenched=q,
            near_zero=z,
            increased=n,
            increased_minus_quenched=dict(point=n['point'] - q['point'], ci=ci(dn - dq, A_DESC)),
        )
    md(
        pd.DataFrame(
            {
                s: {k: fmt(v) + (f' (n {v["n"]}, {v["bulks"]} bulk)' if 'n' in v else '') for k, v in g.items()}
                for s, g in grp.items()
            }
        )
    )
    out['M3_groups'] = grp

    print('## M4 — omat head as control: DiD of the MAE rise S1 → S3, debiased per adsorbate, CI 98.75 %\n')
    m4 = {s: did(e[e.split == s], A_MAIN) for s in SPLITS}
    md(
        pd.DataFrame(
            {
                s: {
                    'DiD (oc20 − omat)': fmt(m4[s]),
                    'rise oc20': round(m4[s]['rise_oc20'], 4),
                    'rise omat': round(m4[s]['rise_omat'], 4),
                }
                for s in SPLITS
            }
        )
    )
    out['M4'] = dict(results=m4, verdict=verdict_m4(m4['test'], m4['val']))
    print(f'**Verdict M4: {out["M4"]["verdict"]}**\n')
    print('MAE per stratum (eV), raw and debiased per adsorbate:\n')
    st = e.groupby(['split', 'stratum'])[['abs_err_oc20', 'abs_err_omat', 'db_oc20', 'db_omat']].mean()
    md(st)
    sp4 = {
        f'{h}, debiased |error|': {s: spearman_ci(e[e.split == s], 'adm', f'db_{h}', A_DESC) for s in SPLITS}
        for h in ('oc20', 'omat')
    }
    print('Spearman(debiased |error|, adm), whole population (CI 95 %):\n')
    md(pd.DataFrame({n: {s: fmt(r[s]) for s in SPLITS} for n, r in sp4.items()}).T)
    out['M4_desc'] = dict(strata=st.reset_index().to_dict('records'), spearman=sp4)

    print('## Verdict summary\n')
    md(pd.DataFrame({k: {'verdict': out[k]['verdict']} for k in ('M1', 'M2', 'M3', 'M4')}).T)

    os.makedirs(D + 'plots', exist_ok=True)
    scatter_plot(e, D + 'plots/01_signed_err_vs_dabs.png')
    strata_plot(st, D + 'plots/02_strata_oc20_vs_omat.png')
    json.dump(out, open(D + 'stage_m_level_a_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
