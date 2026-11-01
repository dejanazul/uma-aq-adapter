import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = 'results/ev2_film/'
sys.path.insert(0, 'analysis/ev2_film')
from stage_m_level_a import CUTS, EFFECT, FLAGS, SPLITS, ci, draw_counts, excl0, fmt, per_bulk, spearman_ci, within_slab

A_MAIN, A_DESC = 0.05 / 3, 0.05
MODELS = ('midfilm', 'infilm')
md = lambda x: print(x.round(4).to_markdown(), end='\n\n')


def load():
    rows = pd.read_json(D + 'film/rows.jsonl', lines=True)
    print(f'- rows.jsonl: {len(rows)} rows, status {rows.groupby(["model", "status"]).size().to_dict()}')
    rows = rows.drop_duplicates(['key', 'model'], keep='last')
    ok = rows[rows.status == 'ok']
    wide = ok.pivot(index='key', columns='model', values=['e_on', 'e_off', 'e_lo_off'])
    wide.columns = [f'{v}_{m}' for v, m in wide.columns]
    e = pd.read_csv('results/failure_analysis/sprint4/sprint4_relaxations.csv')
    assert e.groupby('split').size().to_dict() == {'test': 932, 'val': 963}
    e = e.merge(wide, left_on='key', right_index=True, how='left')
    missing = e[[f'e_on_{m}' for m in MODELS]].isna().any(axis=1)
    print(f'- relaxations without predictions from both models: {int(missing.sum())}\n')
    e['dabs'] = e.total_magnetization.abs() - e.slab_mag.abs()
    e['adm'] = e.dabs.abs()
    e['stratum'] = np.where(~e.spin_on, 'S0', np.where(e.adm <= CUTS[0], 'S1', np.where(e.adm <= CUTS[1], 'S2', 'S3')))
    e['flagged'] = e[list(FLAGS)].any(axis=1)
    for m in MODELS:
        e[f'g_{m}'] = e[f'e_on_{m}'] - e[f'e_off_{m}']
        e[f'gset_{m}'] = e[f'e_on_{m}'] - e[f'e_lo_off_{m}']
        e[f'gfid_{m}'] = e[f'e_off_{m}'] - e[f'e_lo_off_{m}']
        e[f'film_err_on_{m}'] = e[f'e_on_{m}'] - e.adsorption_energy
        e[f'film_err_off_{m}'] = e[f'e_off_{m}'] - e.adsorption_energy
    return e[~missing].reset_index(drop=True), int(missing.sum())


def mae_diff(d, a, b, alpha):
    d = d.reset_index(drop=True)
    bulks, C = draw_counts(d)
    d = d.assign(_a=d[a].abs(), _b=d[b].abs())
    m = np.ones(len(d), bool)
    pa, pb = per_bulk(d, m, '_a', bulks), per_bulk(d, m, '_b', bulks)
    draws = C @ pa.s.values / (C @ pa.n.values) - C @ pb.s.values / (C @ pb.n.values)
    return dict(
        point=float(d._a.mean() - d._b.mean()),
        ci=ci(draws, alpha),
        mae_a=float(d._a.mean()),
        mae_b=float(d._b.mean()),
        n=len(d),
        bulks=int(d.bulk.nunique()),
    )


def slab_rows(d, col):
    on = d[d.spin_on]
    g = on.groupby('slab_id')[col]
    return on[g.transform('size').ge(2) & g.transform('nunique').gt(1)].reset_index(drop=True)


def ols(d, x, y, alpha):
    d = d.reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), d[x].values])
    Y = d[y].values
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    r2 = 1 - ((Y - X @ beta) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    bulks, C = draw_counts(d)
    rows_of = d.groupby('bulk').indices
    draws = []
    for k in C:
        i = np.concatenate([np.repeat(rows_of[b], c) for b, c in zip(bulks, k) if c])
        draws.append(np.linalg.lstsq(X[i], Y[i], rcond=None)[0][1])
    return dict(point=float(beta[1]), ci=ci(draws, alpha), intercept=float(beta[0]), r2=float(r2))


def verdict_corr(t, v):
    rs = (t, v)
    if all(r['point'] >= 0.30 and r['ci'][0] > 0 for r in rs):
        return 'sedang'
    if all(r['point'] >= 0.10 and r['ci'][0] > 0 for r in rs):
        return 'small'
    if all(not excl0(r) or r['point'] < 0.10 for r in rs):
        return 'zero'
    return 'mixed'


def verdict_k3(t, v):
    rs = (t, v)
    if all(r['point'] >= EFFECT and r['ci'][0] > 0 for r in rs):
        return 'large'
    if all(r['ci'][0] > 0 for r in rs):
        return 'partial'
    if all(not excl0(r) or r['point'] < 0 for r in rs):
        return 'none'
    return 'mixed'


def ktests(e, gcol, alpha, rows_filter=None):
    out = {}
    for s in SPLITS:
        d = e[(e.split == s) & e.spin_on]
        if rows_filter is not None:
            d = d[rows_filter(d)]
        d = d.assign(neg_g=-d[gcol], corrected=d.err_oc20 + d[gcol]).reset_index(drop=True)
        sr = slab_rows(d, gcol).assign(neg_g=lambda x: -x[gcol])
        out[s] = dict(
            K1=spearman_ci(d, 'neg_g', 'err_oc20', alpha),
            K2=within_slab(sr, 'neg_g', 'err_oc20', alpha),
            K3=mae_diff(d, 'err_oc20', 'corrected', alpha),
        )
    out['verdict'] = dict(
        K1=verdict_corr(out['test']['K1'], out['val']['K1']),
        K2=verdict_corr(out['test']['K2'], out['val']['K2']),
        K3=verdict_k3(out['test']['K3'], out['val']['K3']),
    )
    return out


def ktable(res):
    return pd.DataFrame(
        {
            s: {
                k: fmt(res[s][k]) + (f' (UMA {res[s][k]["mae_a"]:.3f} → {res[s][k]["mae_b"]:.3f})' if k == 'K3' else '')
                for k in ('K1', 'K2', 'K3')
            }
            for s in SPLITS
        }
    ).assign(verdict=pd.Series(res['verdict']))


def branch(g0, res, robust_same):
    if not g0:
        return 'D — counterfactual invalid'
    v = res['verdict']
    if not robust_same:
        return 'E — not robust (the two models differ in verdict category)'
    if v['K1'] == 'sedang' and v['K2'] == 'sedang' and v['K3'] == 'large':
        return 'A — spin energy explains the UMA error (strong)'
    if {v['K1'], v['K2']} & {'sedang', 'small'} and v['K3'] in ('large', 'partial'):
        return 'B — explains part'
    if v['K1'] == 'zero' and v['K3'] == 'none':
        return 'C — does not explain'
    return 'mixed'


def self_check():
    v = lambda k1, k2, k3: dict(verdict=dict(K1=k1, K2=k2, K3=k3))
    assert branch(False, v('sedang', 'sedang', 'large'), True).startswith('D')
    assert branch(True, v('sedang', 'sedang', 'large'), False).startswith('E')
    assert branch(True, v('sedang', 'sedang', 'large'), True).startswith('A')
    assert branch(True, v('sedang', 'zero', 'partial'), True).startswith('B')
    assert branch(True, v('zero', 'zero', 'none'), True).startswith('C')
    assert branch(True, v('small', 'small', 'none'), True) == 'mixed'
    r = lambda p, lo, hi: dict(point=p, ci=[lo, hi])
    assert verdict_k3(r(0.12, 0.05, 0.2), r(0.11, 0.02, 0.2)) == 'large'
    assert verdict_k3(r(0.05, 0.01, 0.1), r(0.11, 0.02, 0.2)) == 'partial'


def scatter(e, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, s in zip(ax, SPLITS):
        d = e[(e.split == s) & e.spin_on]
        sc = a.scatter(-d.g_midfilm, d.err_oc20, s=8, alpha=0.6, c=np.log10(d.adm + 1e-3), cmap='viridis')
        lim = np.nanquantile(np.abs(np.r_[d.g_midfilm, d.err_oc20]), 0.995)
        a.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='y = x (mechanism prediction)')
        a.axhline(0, c='gray', lw=0.5)
        a.axvline(0, c='gray', lw=0.5)
        a.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            title=f'{s}: spin-on',
            xlabel='−g = −[E_ads(on) − E_ads(off)] EV2-in+midFiLM (eV)',
        )
        a.legend(fontsize=8)
    ax[0].set_ylabel('signed UMA oc20 error (eV)')
    fig.colorbar(sc, ax=ax, label='log10 adm (μB)')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def strata_bars(tab, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, s in zip(ax, SPLITS):
        t = tab[tab.split == s]
        x = np.arange(len(t))
        a.bar(x - 0.2, t.mae_uma, 0.4, label='UMA oc20')
        a.bar(x + 0.2, t.mae_corrected, 0.4, label='UMA + g (midFiLM)')
        a.set_xticks(x, t.stratum)
        a.set_title(s)
        a.legend(fontsize=8)
    ax[0].set_ylabel('MAE E_ads (eV)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    self_check()
    print('## Data gates\n')
    e, n_missing = load()
    out = dict(n_missing=n_missing)
    info = json.load(open(D + 'film/info.json'))
    print(
        f'- info.json: torch {info["torch"]}, GPU {info["gpu"]}, fairchem {info["fairchem_core"]}, frames SHA-256 {info["frames_sha256"]}'
    )
    for m, v in info['models'].items():
        print(f'- {m}: {v["file"]}, SHA-256 {v["sha256"]}, runtime {v.get("runtime_s")} s')
    print()

    print('## G0 — validity: MAE E_ads EV2-FiLM vs UMA oc20 on the same rows (CI 95 %)\n')
    g0 = {}
    for m in MODELS:
        g0[m] = {}
        for s in SPLITS:
            on = e[(e.split == s) & e.spin_on]
            off = e[(e.split == s) & ~e.spin_on]
            g0[m][s] = dict(
                spin_on=mae_diff(on, f'film_err_on_{m}', 'err_oc20', A_DESC),
                spin_off=mae_diff(off, f'film_err_off_{m}', 'err_oc20', A_DESC),
            )
        g0[m]['pass'] = all(g0[m][s]['spin_on']['mae_a'] < g0[m][s]['spin_on']['mae_b'] for s in SPLITS)
    md(
        pd.DataFrame(
            {
                (m, s): {
                    'spin-on: MAE FiLM (mode on)': g0[m][s]['spin_on']['mae_a'],
                    'spin-on: MAE UMA': g0[m][s]['spin_on']['mae_b'],
                    'spin-on: FiLM − UMA [CI]': fmt(g0[m][s]['spin_on']),
                    'spin-off: MAE FiLM (mode off)': g0[m][s]['spin_off']['mae_a'],
                    'spin-off: MAE UMA': g0[m][s]['spin_off']['mae_b'],
                }
                for m in MODELS
                for s in SPLITS
            }
        )
    )
    print(
        f'**G0: midfilm {"pass" if g0["midfilm"]["pass"] else "fail"}, infilm {"pass" if g0["infilm"]["pass"] else "fail"}**\n'
    )
    out['G0'] = g0

    print('## G1 — model physics (descriptive)\n')
    g1 = {}
    for m in MODELS:
        t = e.groupby(['split', 'stratum'])[f'g_{m}'].agg(median_abs=lambda x: x.abs().median(), mean='mean', n='size')
        print(f'### {m}: g = E_ads(on) − E_ads(off) per strata (eV)\n')
        md(t.unstack(0))
        sp = {s: spearman_ci(e[(e.split == s) & e.spin_on], f'g_{m}', 'dabs', A_DESC) for s in SPLITS}
        print(f'Spearman(g, Δ|m|) spin-on (predicted negative): test {fmt(sp["test"])}, val {fmt(sp["val"])}\n')
        g1[m] = dict(strata=t.reset_index().to_dict('records'), spearman_g_dabs=sp)
    out['G1'] = g1

    print('## Main tests K1–K3 (CI 98.33 %)\n')
    res = {m: ktests(e, f'g_{m}', A_MAIN) for m in MODELS}
    for m in MODELS:
        print(f'### {m}\n')
        md(ktable(res[m]))
    robust_same = res['midfilm']['verdict'] == res['infilm']['verdict']
    out['K'] = res
    out['robust_same'] = robust_same
    out['branch'] = branch(g0['midfilm']['pass'] and g0['infilm']['pass'], res['midfilm'], robust_same)
    print(f'Same verdict category in both models: {robust_same}\n')
    print(f'**Branch: {out["branch"]}**\n')

    print('## Descriptive (main model midfilm, CI 95 %)\n')
    desc = {}
    for s in SPLITS:
        d = e[(e.split == s) & e.spin_on].assign(neg_g=lambda x: -x.g_midfilm)
        desc[s] = dict(slope=ols(d, 'neg_g', 'err_oc20', A_DESC))
        off = e[(e.split == s) & ~e.spin_on]
        excess = d.err_oc20.abs().mean() - off.err_oc20.abs().mean()
        desc[s]['excess_spin_on'] = float(excess)
        desc[s]['share_excess_removed'] = float(res['midfilm'][s]['K3']['point'] / excess)
    md(
        pd.DataFrame(
            {
                s: {
                    'slope err ~ −g [CI]': fmt(desc[s]['slope']),
                    'intercept': round(desc[s]['slope']['intercept'], 4),
                    'R²': round(desc[s]['slope']['r2'], 4),
                    'excess MAE spin-on vs spin-off': round(desc[s]['excess_spin_on'], 4),
                    'share of excess removed (K3 / excess)': round(desc[s]['share_excess_removed'], 3),
                }
                for s in SPLITS
            }
        )
    )
    rows = []
    for (s, st), d in e[e.spin_on].groupby(['split', 'stratum']):
        rows.append(
            dict(
                split=s,
                stratum=st,
                n=len(d),
                mae_uma=d.err_oc20.abs().mean(),
                mae_corrected=(d.err_oc20 + d.g_midfilm).abs().mean(),
                mean_g=d.g_midfilm.mean(),
                mean_err=d.err_oc20.mean(),
            )
        )
    st_tab = pd.DataFrame(rows)
    print('ΔMAE per strata (spin-on):\n')
    md(st_tab.assign(delta=st_tab.mae_uma - st_tab.mae_corrected).set_index(['split', 'stratum']))
    desc['strata'] = st_tab.to_dict('records')
    print('K1–K3 without AdsorbML-flagged relaxations:\n')
    noflag = ktests(e, 'g_midfilm', A_DESC, rows_filter=lambda d: ~d.flagged)
    md(ktable(noflag))
    desc['no_flag'] = noflag
    out['desc'] = desc

    print('## Addendum: confounder diagnostics "off mode ≈ OC20 style" (D1–D3, CI 95 %)\n')
    dg = {}
    for m in MODELS:
        dg[m] = {}
        for s in SPLITS:
            d = e[e.split == s]
            on, off = d[d.spin_on], d[~d.spin_on]
            s1, s3 = d[d.stratum == 'S1'], d[d.stratum == 'S3']
            bulks, C = draw_counts(d.reset_index(drop=True))
            dd = d.reset_index(drop=True).assign(_ag=lambda x: x[f'g_{m}'].abs())
            a, b = (
                per_bulk(dd, (dd.stratum == 'S3').values, '_ag', bulks),
                per_bulk(dd, (dd.stratum == 'S1').values, '_ag', bulks),
            )
            draws = C @ a.s.values / (C @ a.n.values) - C @ b.s.values / (C @ b.n.values)
            dg[m][s] = dict(
                D1_median_abs_gfid_magnetic=float(on[f'gfid_{m}'].abs().median()),
                D1_median_abs_gfid_nonmagnetic=float(off[f'gfid_{m}'].abs().median()),
                D2_spearman_g_gset=spearman_ci(on, f'g_{m}', f'gset_{m}', A_DESC),
                D2_ratio_gfid_gset=float(on[f'gfid_{m}'].abs().median() / on[f'gset_{m}'].abs().median()),
                D3_median_abs_g_S1=float(s1[f'g_{m}'].abs().median()),
                D3_median_abs_g_S3=float(s3[f'g_{m}'].abs().median()),
                D3_mean_abs_g_S3_minus_S1=dict(
                    point=float(s3[f'g_{m}'].abs().mean() - s1[f'g_{m}'].abs().mean()), ci=ci(draws, A_DESC)
                ),
                D3_spearman_g_dabs=g1[m]['spearman_g_dabs'][s],
            )
        dg[m]['D3_supports_spin'] = all(
            dg[m][s]['D3_mean_abs_g_S3_minus_S1']['ci'][0] > 0
            and dg[m][s]['D3_spearman_g_dabs']['point'] <= -0.10
            and dg[m][s]['D3_spearman_g_dabs']['ci'][1] < 0
            for s in SPLITS
        )
        print(f'### {m}\n')
        md(
            pd.DataFrame(
                {s: {k: (fmt(v) if isinstance(v, dict) else round(v, 4)) for k, v in dg[m][s].items()} for s in SPLITS}
            )
        )
    dg['spin_claim_allowed'] = all(dg[m]['D3_supports_spin'] for m in MODELS)
    print(f"**'Spin energy' claim allowed (D3 in both splits and both models): {dg['spin_claim_allowed']}**\n")
    out['addendum'] = dg

    print('## Secondary test: full OC20 setting, g_set = E_ads(on) − E_ads(lo-off) (descriptive, CI 95 %)\n')
    sec = {m: ktests(e, f'gset_{m}', A_DESC) for m in MODELS}
    for m in MODELS:
        print(f'### {m}\n')
        md(ktable(sec[m]))
    dec = e[e.spin_on].groupby('split')[['g_midfilm', 'gfid_midfilm', 'gset_midfilm', 'err_oc20']].mean()
    print('Decomposition of spin-on means (midfilm, eV): g_set = g + g_fid\n')
    md(dec)
    off_rows = {}
    for s in SPLITS:
        d = e[(e.split == s) & ~e.spin_on].assign(corr_set=lambda x: x.err_oc20 + x.gset_midfilm)
        off_rows[s] = mae_diff(d, 'err_oc20', 'corr_set', A_DESC)
    print(
        f'Spin-off, UMA corrected with g_set (= fidelity effect only): test {fmt(off_rows["test"])}, val {fmt(off_rows["val"])} eV\n'
    )
    out['secondary'] = dict(K=sec, decomposition=dec.reset_index().to_dict('records'), spin_off_gset=off_rows)

    os.makedirs(D + 'plots', exist_ok=True)
    scatter(e, D + 'plots/03_film_err_vs_gap.png')
    strata_bars(st_tab, D + 'plots/04_film_strata_correction.png')
    keep = [
        'key',
        'split',
        'relaxation',
        'bulk',
        'slab_id',
        'adsorbate',
        'spin_on',
        'stratum',
        'adm',
        'dabs',
        'flagged',
        'adsorption_energy',
        'err_oc20',
    ] + [c for c in e.columns if c.endswith(MODELS)]
    e[keep].to_csv(D + 'film_relaxations.csv', index=False)
    json.dump(out, open(D + 'stage_m_film_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
