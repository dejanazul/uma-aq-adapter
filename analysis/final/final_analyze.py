import json

import numpy as np
import pandas as pd

D = 'results/adapter/final/'
OC20 = 'results/oc20_stage_u/'
DRAWS, TOL = 2000, 0.1
md = lambda t: print(t.round(4).to_markdown(), end='\n\n')


def rows(step):
    r = pd.read_json(D + f'{step}.jsonl', lines=True)
    bad = r[r.status != 'ok']
    r = r[r.status == 'ok'].drop_duplicates('key', keep='last').copy()
    r[['model', 'k']] = r.key.str.split('|', n=1, expand=True)
    return r, len(bad)


def boot(d, stat, by, alpha, seed=0):
    rng = np.random.default_rng(seed)
    if by is None:
        draws = [stat(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(DRAWS)]
    else:
        groups = {b: g for b, g in d.groupby(by)}
        keys = np.array(list(groups))
        draws = [stat(pd.concat([groups[b] for b in rng.choice(keys, len(keys))])) for _ in range(DRAWS)]
    return float(stat(d)), [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))]


def verdict(res):
    neg = all(hi < 0 for _, (lo, hi) in res.values())
    if any(lo > 0 for _, (lo, hi) in res.values()):
        return 'worse'
    if neg and all(p <= -TOL for p, _ in res.values()):
        return 'large'
    if neg:
        return 'present'
    if all(lo <= 0 <= hi for _, (lo, hi) in res.values()):
        return 'none'
    return 'mixed'


def aq(summary):
    r, nbad = rows('aq')
    rel = pd.read_csv('results/failure_analysis/sprint4/sprint4_relaxations.csv')
    slab_err = r[r.kind == 'slab'].set_index(['split', 'slab_id']).total_err_eV
    a = r[r.kind == 'adslab'].set_index('k')
    rel['err_ft'] = rel.key.map(a.total_err_eV) - pd.Series(list(zip(rel.split, rel.slab_id))).map(slab_err).to_numpy()
    rel['fmae_ft'] = rel.key.map(a.force_mae_free)
    assert rel.err_ft.notna().all(), 'missing predictions'
    rel['abs_ft'], rel['abs_zs'] = rel.err_ft.abs(), rel.err_oc20.abs()
    dmae = lambda d: d.abs_ft.mean() - d.abs_zs.mean()
    out = dict(n_failed_rows=nbad)
    for name, sub in (('P1', rel), ('P2', rel[rel.stratum == 'S3'])):
        res = {s: boot(g, dmae, 'bulk', 0.025) for s, g in sub.groupby('split')}
        out[name] = dict(
            per_split={
                s: dict(
                    dMAE=p,
                    ci97_5=ci,
                    n=int((sub.split == s).sum()),
                    mae_ft=float(sub[sub.split == s].abs_ft.mean()),
                    mae_zs=float(sub[sub.split == s].abs_zs.mean()),
                )
                for s, (p, ci) in res.items()
            },
            verdict=verdict(res),
        )
    print('## AQCat25 OOD-Both, final frames (P1, P2)\n')
    for name in ('P1', 'P2'):
        t = pd.DataFrame(out[name]['per_split']).T
        t['ci97_5'] = t.ci97_5.map(lambda c: f'[{c[0]:.4f}, {c[1]:.4f}]')
        print(f'**{name}** ({"all" if name == "P1" else "S3"}): verdict **{out[name]["verdict"]}**\n')
        md(t)
    g = rel.groupby(['split', 'stratum'])
    desc = pd.DataFrame(
        dict(
            n=g.size(),
            mae_ft=g.abs_ft.mean(),
            mae_zs=g.abs_zs.mean(),
            signed_ft=g.err_ft.mean(),
            signed_zs=g.err_oc20.mean(),
            pct_le_0p1_ft=100 * g.abs_ft.apply(lambda x: (x <= TOL).mean()),
            pct_le_0p1_zs=100 * g.abs_zs.apply(lambda x: (x <= TOL).mean()),
        )
    )
    print('Descriptive per stratum:\n')
    md(desc)
    gs = rel.groupby('split')
    tot = pd.DataFrame(
        dict(
            mae_ft=gs.abs_ft.mean(),
            mae_zs=gs.abs_zs.mean(),
            median_ft=gs.abs_ft.median(),
            median_zs=gs.abs_zs.median(),
            pct_le_0p1_ft=100 * gs.abs_ft.apply(lambda x: (x <= TOL).mean()),
            pct_le_0p1_zs=100 * gs.abs_zs.apply(lambda x: (x <= TOL).mean()),
            force_mae_free_ft=gs.fmae_ft.mean(),
            force_mae_free_zs=gs.force_mae_free_oc20.mean(),
        )
    )
    print('Descriptive, overall (final frames):\n')
    md(tot)
    out['descriptive_by_stratum'] = json.loads(desc.reset_index().to_json(orient='records'))
    out['descriptive_total'] = json.loads(tot.reset_index().to_json(orient='records'))
    rel.to_csv(D + 'aq_final_relaxations.csv', index=False)
    summary['aqcat25'] = out


def aqz(summary):
    r, nbad = rows('aqz')
    zs = pd.read_csv('results/zeroshot/uma_zeroshot_frames.csv')
    pred = r.set_index('k')
    zs['err_frame_ft'] = zs.key.map(pred.total_err_eV)
    slab = r[r.kind == 'slab'].merge(
        pd.read_csv('results/zeroshot/uma_zeroshot_manifest.csv')[['key', 'slab_id']], left_on='k', right_on='key'
    )
    zs['err_ft'] = zs.err_frame_ft - zs.slab_id.map(slab.set_index('slab_id').total_err_eV)
    zs['fmae_ft'] = zs.key.map(pred.force_mae_free)
    use = zs[zs.paper_filter_ok & ~zs.organic.astype(bool) & zs.err_ft.notna()].copy()
    use['abs_ft'] = use.err_ft.abs()
    use['pos'] = use.groupby('relaxation').fid.rank(method='first').astype(int)
    dmae = lambda d: d.abs_ft.mean() - d.abs_err.mean()
    p, ci = boot(use, dmae, 'bulk', 0.05)
    out = dict(
        n_frames=len(use),
        n_relaxations=int(use.relaxation.nunique()),
        n_failed_rows=nbad,
        mae_ft=float(use.abs_ft.mean()),
        mae_zs=float(use.abs_err.mean()),
        dMAE=p,
        ci95=ci,
        force_mae_free_ft=float(use.fmae_ft.mean()),
        force_mae_free_zs=float(use.force_mae_free.mean()),
    )
    print('## AQCat25 test, all trajectory frames (9b, descriptive)\n')
    print(f'- {out["n_frames"]} frames from {out["n_relaxations"]} relaxations (paper filter, non-organic)')
    print(
        f'- MAE E_ads per frame: L2-hybrid {out["mae_ft"]:.4f} eV, zero-shot {out["mae_zs"]:.4f} eV, '
        f'dMAE {p:.4f} eV, CI 95 % [{ci[0]:.4f}, {ci[1]:.4f}]'
    )
    print(f'- free-atom force MAE per frame: {out["force_mae_free_ft"]:.4f} vs {out["force_mae_free_zs"]:.4f} eV/A\n')
    g = use.groupby('pos')
    t = pd.DataFrame(
        dict(
            n=g.size(),
            mae_ft=g.abs_ft.mean(),
            mae_zs=g.abs_err.mean(),
            fmae_ft=g.fmae_ft.mean(),
            fmae_zs=g.force_mae_free.mean(),
        )
    )
    print('Per frame position along the trajectory (1 = first, 10 = last):\n')
    md(t)
    out['by_position'] = json.loads(t.reset_index().to_json(orient='records'))
    summary['aqcat25_all_frames'] = out


def oc20(summary):
    r, nbad = rows('oc20')
    u = pd.read_json(OC20 + 'nonmetal_debug/stageU_raw.jsonl', lines=True)
    u = u[u.status == 'ok']
    slab_of = json.load(open(OC20 + 'nonmetal_debug/stageU_slab_map.json'))
    tabs = []
    for s in ('discovery', 'confirm'):
        t = pd.read_csv(OC20 + f'ood_both_error_map/{s}_table.csv')[['system_id', 'bulk_mpid', 'catalyst_class']]
        t['set'], t['slab_id'] = s, t.system_id.map(slab_of[s])
        tabs.append(t)
    t = pd.concat(tabs, ignore_index=True)
    src = {
        'stage_u': (
            u[(u.kind == 'adslab')].set_index(['set', 'id']).total_err_eV,
            u[u.kind == 'slab'].set_index('id').total_err_eV,
        )
    }
    for m in ('final', 'uma'):
        x = r[r.model == m]
        src[m] = (
            x[x.kind == 'adslab'].set_index(['set', 'id']).total_err_eV,
            x[x.kind == 'slab'].set_index('id').total_err_eV,
        )
    for m, (ads, slab) in src.items():
        t[f'dE_{m}'] = pd.Series(list(zip(t.set, t.system_id.astype(str)))).map(ads).to_numpy() - t.slab_id.astype(
            str
        ).map(slab)
    assert t.filter(like='dE_').notna().all().all(), 'missing OC20 predictions'
    mae = {m: float(t[f'dE_{m}'].abs().mean()) for m in src}
    p, ci = boot(t, lambda d: d.dE_final.abs().mean() - d.dE_stage_u.abs().mean(), 'bulk_mpid', 0.05)
    p2, ci2 = boot(t, lambda d: d.dE_final.abs().mean() - d.dE_uma.abs().mean(), 'bulk_mpid', 0.05)
    out = dict(
        n_systems=len(t),
        n_failed_rows=nbad,
        mae=mae,
        G=dict(dMAE=p, ci95=ci, passed=bool(ci[0] <= 0)),
        vs_uma_rerun=dict(dMAE=p2, ci95=ci2),
        max_abs_diff_final_vs_uma_eV=float((t.dE_final - t.dE_uma).abs().max()),
        max_abs_diff_final_vs_stage_u_eV=float((t.dE_final - t.dE_stage_u).abs().max()),
    )
    print('## OC20 OOD-Both, set stage U (G)\n')
    print(
        f'- {len(t)} systems. MAE E_ads: L2-hybrid (task oc20) {mae["final"]:.4f} eV, released UMA rerun '
        f'{mae["uma"]:.4f} eV, stage U {mae["stage_u"]:.4f} eV'
    )
    print(
        f'- **G**: dMAE vs stage U {p:.2e} eV, CI 95 % [{ci[0]:.2e}, {ci[1]:.2e}] -> **{"pass" if ci[0] <= 0 else "fail"}**'
    )
    print(
        f'- max |dE_final - dE_UMA rerun| {out["max_abs_diff_final_vs_uma_eV"]:.2e} eV; '
        f'vs stage U {out["max_abs_diff_final_vs_stage_u_eV"]:.2e} eV\n'
    )
    summary['oc20'] = out


def forget(summary):
    r, nbad = rows('forget')
    r['e_pa'] = r.total_err_eV.abs() / r.n_atoms
    out = dict(n_failed_rows=nbad, failed_detail=None)
    print('## All UMA tasks (official val, 300 structures per dataset; descriptive)\n')
    lines = []
    for ds, g in r.groupby('dataset'):
        w = g.pivot(index='k', columns='model', values=['e_pa', 'force_mae_all', 'pred_total_eV'])
        line = dict(dataset=ds, task=g.task.iloc[0], n=len(w))
        for m in ('uma', 'l2', 'l2h'):
            line[f'E_mae_pa_{m}'] = float(w['e_pa'][m].mean())
            line[f'F_mae_{m}'] = float(w['force_mae_all'][m].mean())
        for m in ('l2', 'l2h'):
            dd = pd.DataFrame(dict(a=w['e_pa'][m], b=w['e_pa']['uma']))
            p, ci = boot(dd, lambda d: d.a.mean() - d.b.mean(), None, 0.05)
            line[f'dE_mae_pa_{m}'], line[f'dE_ci95_{m}'] = p, ci
            line[f'max_abs_dE_{m}_eV'] = float((w['pred_total_eV'][m] - w['pred_total_eV']['uma']).abs().max())
            line[f'no_forgetting_{m}'] = bool(ci[0] <= 0)
        lines.append(line)
    t = pd.DataFrame(lines).set_index('dataset')
    cols = [
        'task',
        'n',
        'E_mae_pa_uma',
        'E_mae_pa_l2h',
        'E_mae_pa_l2',
        'F_mae_uma',
        'F_mae_l2h',
        'F_mae_l2',
        'max_abs_dE_l2h_eV',
        'max_abs_dE_l2_eV',
        'no_forgetting_l2h',
        'no_forgetting_l2',
    ]
    md(t[cols])
    bad = r if nbad == 0 else pd.read_json(D + 'forget.jsonl', lines=True).query('status != "ok"')
    if nbad:
        out['failed_detail'] = bad.groupby(['dataset', 'model']).status.first().to_dict()
        print('Failed rows:', out['failed_detail'], '\n')
    out['per_dataset'] = json.loads(t.reset_index().to_json(orient='records'))
    summary['all_tasks'] = out


def main():
    summary = {}
    for f in (aq, aqz, oc20, forget):
        try:
            f(summary)
        except FileNotFoundError as e:
            print(f'(skipped {f.__name__}: {e})\n')
    json.dump(summary, open(D + 'final_summary.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    assert verdict({'a': (-0.2, [-0.3, -0.1]), 'b': (-0.15, [-0.2, -0.05])}) == 'large'
    assert verdict({'a': (-0.05, [-0.08, -0.01]), 'b': (-0.15, [-0.2, -0.05])}) == 'present'
    assert verdict({'a': (0.0, [-0.1, 0.1]), 'b': (0.0, [-0.1, 0.1])}) == 'none'
    assert verdict({'a': (0.1, [0.05, 0.2]), 'b': (-0.15, [-0.2, -0.05])}) == 'worse'
    assert verdict({'a': (-0.05, [-0.08, -0.01]), 'b': (0.0, [-0.1, 0.1])}) == 'mixed'
    main()
