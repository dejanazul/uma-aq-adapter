import json
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import chemical_symbols, covalent_radii
from ase.geometry import find_mic, get_distances
from ase.io import iread
from ase.neighborlist import natural_cutoffs
from ase.visualize.plot import plot_atoms
from fairchem.data.oc.utils.flag_anomaly import DetectTrajAnomaly
from scipy.stats import spearmanr

D, S2 = 'results/failure_analysis/sprint3/', 'results/failure_analysis/sprint2/'
P = D + 'plots/'
SPLITS = ('test', 'val')
DRAWS, ALPHA, MIN_BULKS, RHO_SMALL, RHO_STRONG, GROUP_DIFF = 2000, 0.05, 5, 0.10, 0.30, 0.1
EXTXYZ = {'test': 'data/zeroshot/uma_zeroshot_frames.extxyz', 'val': 'data/zeroshot/uma_zeroshot_frames_val.extxyz'}
MANIFEST = {'test': 'results/zeroshot/uma_zeroshot_manifest.csv', 'val': S2 + 'sprint2_val_manifest.csv'}
FLAGS = ('desorbed', 'dissociated', 'intercalated', 'surface_changed')
FEATURES = [
    'n_atoms',
    'n_free',
    'n_ads',
    'min_dist_all',
    'min_dist_ads_surf',
    'ads_coordination',
    'max_disp_ads',
    'max_disp_surf',
    'n_frames_traj',
    'dmag',
    'slab_fmax_free',
    'fvec_mean_final',
    'fvec_max_final',
]
SKIN = 0.3
md = lambda x: print(x.round(4).to_markdown(), end='\n\n')


def bonded(atoms, a_idx, b_idx):
    r = np.array(natural_cutoffs(atoms)) + SKIN
    _, d = get_distances(atoms.positions[a_idx], atoms.positions[b_idx], atoms.cell, atoms.pbc)
    return d < r[a_idx, None] + r[None, b_idx], d


def load_pred(split):
    pred = {}
    for f in sorted(os.listdir(S2 + f'raw_{split}')):
        if f.startswith('forces_'):
            z = np.load(S2 + f'raw_{split}/' + f)
            ends = np.cumsum(z['n_atoms'])
            for k, e, n in zip(z['keys'], ends, z['n_atoms']):
                pred[str(k)] = z['f_oc20'][e - n : e]
    return pred


def extract(split, frames):
    m = pd.read_csv(MANIFEST[split]).set_index('key')
    pred = load_pred(split)
    fr = frames.set_index('key')
    first_fid = m[m.kind == 'adslab'].groupby('relaxation').fid.min()
    initial, final, slabs, structs, atoms_rows = {}, {}, {}, [], []
    for atoms in iread(EXTXYZ[split]):
        key = atoms.info['key']
        r = m.loc[key]
        f_dft, f_pred = atoms.get_forces(), pred[key].astype(float)
        assert len(f_pred) == len(atoms), key
        tags = atoms.get_tags()
        free = np.where(tags != 0)[0]
        err = np.linalg.norm(f_pred[free] - f_dft[free], axis=1)
        if r.kind == 'slab':
            slabs[r.slab_id] = atoms
            structs.append(
                dict(
                    key=key,
                    kind='slab',
                    slab_id=r.slab_id,
                    n_free=len(free),
                    fvec_mean=err.mean(),
                    fvec_max=err.max(),
                    slab_fmax_free=np.linalg.norm(f_dft[free], axis=1).max(),
                )
            )
            continue
        if r.fid == first_fid[r.relaxation]:
            initial[r.relaxation] = atoms
        if key not in fr.index:
            continue
        if fr.at[key, 'is_final']:
            final[r.relaxation] = atoms
        ads, slab = np.where(tags == 2)[0], np.where(tags != 2)[0]
        bond, d = bonded(atoms, ads, slab)
        d_ads = np.zeros(len(atoms))
        d_ads[slab] = d.min(axis=0)
        cat = np.full(len(atoms), 2)
        cat[ads] = 0
        cat[slab[bond.any(axis=0)]] = 1
        n_top = max(1, math.ceil(0.1 * len(free)))
        a, b = f_pred[free], f_dft[free]
        cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
        worst = err.argmax()
        s = dict(
            key=key,
            kind='adslab',
            relaxation=r.relaxation,
            n_free=len(free),
            fvec_mean=err.mean(),
            fvec_max=err.max(),
            top10_share=np.sort(err)[-n_top:].sum() / err.sum(),
            worst_angle_deg=np.degrees(np.arccos(np.clip(cos[worst], -1, 1))),
            worst_fdft=np.linalg.norm(b[worst]),
            worst_fpred=np.linalg.norm(a[worst]),
        )
        for c, name in enumerate(('ads', 'bind', 'other')):
            sel = cat[free] == c
            s[f'share_err_{name}'], s[f'share_n_{name}'] = err[sel].sum() / err.sum(), sel.mean()
        structs.append(s)
        atoms_rows.append(
            pd.DataFrame(
                dict(
                    key=key,
                    Z=atoms.numbers[free],
                    cat=cat[free],
                    d_ads=d_ads[free],
                    err=err,
                    abs_dx=np.abs(a - b).ravel()[0::3],
                    abs_dy=np.abs(a - b).ravel()[1::3],
                    abs_dz=np.abs(a - b).ravel()[2::3],
                )
            )
        )
    return initial, final, slabs, pd.DataFrame(structs), pd.concat(atoms_rows, ignore_index=True)


def geometry(rel, init, fin, slab):
    tags = fin.get_tags()
    ads, sl, surf = np.where(tags == 2)[0], np.where(tags != 2)[0], np.where(tags == 1)[0]
    bond, d = bonded(fin, ads, sl)
    conn = DetectTrajAnomaly(init, fin, tags)._get_connectivity(fin)
    assert (conn[np.ix_(ads, sl)] == bond).all(), rel
    dall = fin.get_all_distances(mic=True)
    np.fill_diagonal(dall, np.inf)
    disp = find_mic(fin.positions - init.positions, fin.cell, fin.pbc)[1]
    det = DetectTrajAnomaly(init, fin, tags, final_slab_atoms=slab)
    return dict(
        n_ads=len(ads),
        min_dist_all=dall.min(),
        min_dist_ads_surf=d.min(),
        binding_element=chemical_symbols[fin.numbers[ads[d.min(axis=1).argmin()]]],
        ads_coordination=int(bond.any(axis=0).sum()),
        max_disp_ads=disp[ads].max(),
        max_disp_surf=disp[surf].max() if len(surf) else 0.0,
        desorbed=det.is_adsorbate_desorbed(),
        dissociated=det.is_adsorbate_dissociated(),
        intercalated=det.is_adsorbate_intercalated(),
        surface_changed=det.has_surface_changed(),
    )


def build():
    fr = pd.read_csv(S2 + 'sprint2_frames.csv')
    use = fr[fr.paper_filter_ok & fr.err_oc20.notna() & fr.err_omat.notna()]
    rels, structs, atoms_all = [], [], []
    for split in SPLITS:
        u = use[use.split == split]
        initial, final, slabs, st, at = extract(split, u)
        e = u[u.is_final].copy()
        traj = pd.read_parquet(f'data/parquet/{split}_ood_both.parquet').reset_index()
        traj['relaxation'] = traj.sid.str.removesuffix('_rerun')
        e['n_frames_traj'] = e.relaxation.map(traj.groupby('relaxation').size())
        e['rerun_only'] = e.relaxation.map(
            traj.groupby('relaxation').sid.apply(lambda s: s.str.endswith('_rerun').all())
        )
        ff = pd.read_csv(f'results/zeroshot/aqcat25_{split}_ood_both_final_frames.csv').set_index('relaxation')
        e['slab_elements'] = e.relaxation.map(ff.slab_elements)
        e['chemsys'] = e.slab_elements.str.split().map(lambda x: '-'.join(sorted(x)))
        g = pd.DataFrame(
            [
                dict(relaxation=r, **geometry(r, initial[r], final[r], slabs[sid]))
                for r, sid in zip(e.relaxation, e.slab_id)
            ]
        )
        e = e.merge(g, on='relaxation')
        ss = st[st.kind == 'slab'].set_index('slab_id')
        e['slab_fmax_free'] = e.slab_id.map(ss.slab_fmax_free)
        sa = st[st.kind == 'adslab'].set_index('key')
        e['fvec_mean_final'], e['fvec_max_final'] = e.key.map(sa.fvec_mean), e.key.map(sa.fvec_max)
        st['split'], at['split'] = split, split
        rels.append(e)
        structs.append(st)
        atoms_all.append(at)
    return (
        pd.concat(rels, ignore_index=True),
        pd.concat(structs, ignore_index=True),
        pd.concat(atoms_all, ignore_index=True),
        use,
    )


def draw_counts(d):
    bulks = np.array(sorted(d.bulk.unique()))
    idx = np.random.default_rng(0).integers(0, len(bulks), (DRAWS, len(bulks)))
    return bulks, np.stack([np.bincount(i, minlength=len(bulks)) for i in idx])


def ci(x):
    return [float(np.nanquantile(x, ALPHA / 2)), float(np.nanquantile(x, 1 - ALPHA / 2))]


def group_table(e, split):
    d = e[e.split == split]
    bulks, C = draw_counts(d)
    groups = [
        ('element', el, d.slab_elements.str.split().map(lambda x, el=el: el in x))
        for el in sorted({x for s in e.slab_elements for x in s.split()})
    ]
    groups += [('chemsys', c, d.chemsys == c) for c in sorted(e.chemsys.unique())]
    groups += [('binding_element', b, d.binding_element == b) for b in sorted(e.binding_element.unique())]
    groups += [('anomaly', f, d[f]) for f in FLAGS]
    groups += [('material_class', c, d.material_class == c) for c in sorted(e.material_class.unique())]
    groups += [('spin_on', 'True', d.spin_on)]
    rows = []
    for kind, name, mask in groups:
        mask = mask.values.astype(bool)
        n_b = d.bulk[mask].nunique()
        row = dict(split=split, kind=kind, group=name, n=int(mask.sum()), bulks=n_b, bulks_rest=d.bulk[~mask].nunique())
        if mask.sum() and (~mask).sum():
            row['mae_in'], row['mae_rest'] = d.abs_err_oc20[mask].mean(), d.abs_err_oc20[~mask].mean()
            row['diff'] = row['mae_in'] - row['mae_rest']
            if n_b >= MIN_BULKS:
                x = d.abs_err_oc20.values
                ins = (
                    pd.DataFrame(dict(s=np.where(mask, x, 0), n=mask, s2=np.where(~mask, x, 0), n2=~mask))
                    .groupby(d.bulk.values)
                    .sum()
                    .reindex(bulks, fill_value=0)
                )
                with np.errstate(invalid='ignore', divide='ignore'):
                    diff = C @ ins.s.values / (C @ ins.n.values) - C @ ins.s2.values / (C @ ins.n2.values)
                row['ci_low'], row['ci_high'] = ci(diff)
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_table(e, split):
    d = e[e.split == split].reset_index(drop=True)
    bulks, C = draw_counts(d)
    rows_of = d.groupby('bulk').indices
    cols = ['abs_err_oc20'] + FEATURES
    X = d[cols].astype(float).values
    point = spearmanr(X).statistic[0, 1:]
    rng_rows = [np.concatenate([np.repeat(rows_of[b], c) for b, c in zip(bulks, counts) if c]) for counts in C]
    draws = np.array([spearmanr(X[i]).statistic[0, 1:] for i in rng_rows])
    out = pd.DataFrame(dict(split=split, feature=FEATURES, rho=point))
    out[['ci_low', 'ci_high']] = [ci(draws[:, j]) for j in range(len(FEATURES))]
    return out


def verdict_rho(t, v):
    ok = [(abs(r.rho) >= RHO_SMALL) and (r.ci_low > 0 or r.ci_high < 0) for r in (t, v)]
    if all(ok) and np.sign(t.rho) == np.sign(v.rho):
        return 'strong candidate' if min(abs(t.rho), abs(v.rho)) >= RHO_STRONG else 'candidate'
    if all(ok):
        return 'mixed (opposite signs)'
    return 'not replicated' if any(ok) else '-'


def verdict_group(t, v):
    def ok(r):
        return (
            r.bulks >= MIN_BULKS
            and abs(r.get('diff', 0)) >= GROUP_DIFF
            and not pd.isna(r.get('ci_low'))
            and (r.ci_low > 0 or r.ci_high < 0)
        )

    a, b = ok(t), ok(v)
    if a and b:
        return 'candidate' if np.sign(t['diff']) == np.sign(v['diff']) else 'mixed (opposite signs)'
    return 'not replicated' if a or b else '-'


def metrics(e, st, at, use):
    rows = {}
    for split in SPLITS:
        x = e[e.split == split].abs_err_oc20
        rows[f'{split}: E_ads final frame (eV)'] = dict(
            n=len(x),
            MAE=x.mean(),
            RMSE=np.sqrt((x**2).mean()),
            median=x.median(),
            max=x.max(),
            p90=x.quantile(0.9),
            p95=x.quantile(0.95),
            p99=x.quantile(0.99),
        )
        f = e[e.split == split]
        y = (f.pred_total_eV_oc20 - f.dft_total_eV).abs() / f.n_atoms
        rows[f'{split}: |ΔE_total|/atom final adslab frame (eV/atom)'] = dict(
            n=len(y),
            MAE=y.mean(),
            RMSE=np.sqrt((y**2).mean()),
            median=y.median(),
            max=y.max(),
            p90=y.quantile(0.9),
            p95=y.quantile(0.95),
            p99=y.quantile(0.99),
        )
    print('### Energy (task 14)\n')
    md(pd.DataFrame(rows).T)
    rows = {}
    for split in SPLITS:
        for label, keys in (
            ('all adslab frames', st[(st.split == split) & (st.kind == 'adslab')].key),
            ('final adslab frames', e[e.split == split].key),
            ('reference slab', st[(st.split == split) & (st.kind == 'slab')].key),
        ):
            a = at[at.key.isin(set(keys))] if label != 'reference slab' else None
            s = st[st.key.isin(set(keys))]
            if a is None:
                rows[f'{split}: {label}'] = dict(
                    structures=len(s), mean_struct_mean=s.fvec_mean.mean(), mean_struct_max=s.fvec_max.mean()
                )
                continue
            comp = np.concatenate([a.abs_dx, a.abs_dy, a.abs_dz])
            rows[f'{split}: {label}'] = dict(
                structures=len(s),
                atoms=len(a),
                comp_MAE=comp.mean(),
                comp_RMSE=np.sqrt((comp**2).mean()),
                vec_mean=a.err.mean(),
                vec_median=a.err.median(),
                vec_p90=a.err.quantile(0.9),
                vec_p95=a.err.quantile(0.95),
                vec_p99=a.err.quantile(0.99),
                vec_max=a.err.max(),
                mean_struct_mean=s.fvec_mean.mean(),
                mean_struct_max=s.fvec_max.mean(),
            )
    print(
        '### Forces (eV/Å, free atoms; comp = pooled per component, vec = |ΔF_i| per atom, '
        'struct_mean/max = mean/max |ΔF_i| per structure, then averaged)\n'
    )
    md(pd.DataFrame(rows).T)


def force_analysis(st, at, e):
    out = {}
    bulk_of = e.set_index('relaxation').bulk
    for split in SPLITS:
        s = st[(st.split == split) & (st.kind == 'adslab')].copy()
        s['bulk'] = s.relaxation.map(bulk_of)
        s = s[s.bulk.notna()]
        hi, lo = s.fvec_mean.quantile(0.9), s.fvec_mean.median()
        s['group'] = np.where(s.fvec_mean >= hi, 'high', np.where(s.fvec_mean <= lo, 'low', 'middle'))
        s = s[s.group != 'middle']
        s['enrich_adsbind'] = (s.share_err_ads + s.share_err_bind) - (s.share_n_ads + s.share_n_bind)
        cols = [
            'n_free',
            'fvec_mean',
            'top10_share',
            'share_err_ads',
            'share_n_ads',
            'share_err_bind',
            'share_n_bind',
            'share_err_other',
            'share_n_other',
            'enrich_adsbind',
            'worst_angle_deg',
            'worst_fdft',
            'worst_fpred',
        ]
        print(f'### {split}: high- vs low-error structures (mean per structure)\n')
        t = s.groupby('group')[cols].mean().T
        t.loc['structures'] = s.group.value_counts()
        t.loc['bulks'] = s.groupby('group').bulk.nunique()
        md(t)
        bulks, C = draw_counts(s)
        res = {}
        for c in ('enrich_adsbind', 'top10_share', 'share_err_ads', 'worst_angle_deg'):
            g = (
                pd.DataFrame(
                    dict(
                        sh=np.where(s.group == 'high', s[c], 0),
                        nh=s.group == 'high',
                        sl=np.where(s.group == 'low', s[c], 0),
                        nl=s.group == 'low',
                    )
                )
                .groupby(s.bulk.values)
                .sum()
                .reindex(bulks, fill_value=0)
            )
            with np.errstate(invalid='ignore', divide='ignore'):
                h = C @ g.sh.values / (C @ g.nh.values) - C @ g.sl.values / (C @ g.nl.values)
            point = s[s.group == 'high'][c].mean() - s[s.group == 'low'][c].mean()
            res[c] = dict(diff_high_minus_low=point, ci_low=ci(h)[0], ci_high=ci(h)[1])
        print('Difference high − low, CI 95 % bootstrap over bulks:\n')
        md(pd.DataFrame(res).T)
        a = at[at.key.isin(set(s.key))].merge(s[['key', 'group']], on='key')
        a['element'] = [chemical_symbols[z] for z in a.Z]
        print(f'### {split}: |ΔF_i| per element (free atoms), top 15 in the high group\n')
        t = a.pivot_table(index='element', columns='group', values='err', aggfunc=['mean', 'size'])
        t.columns = [f'{x}_{g}' for x, g in t.columns]
        md(t.sort_values('mean_high', ascending=False).head(15))
        a['bin'] = np.where(
            a.cat == 0, 'adsorbate', pd.cut(a.d_ads, [0, 1, 2, 3, 4, 5, 6, np.inf], right=False).astype(str)
        )
        print(
            f'### {split}: |ΔF_i| by distance to the adsorbate (free slab atoms, 1 Å bins; adsorbate row = adsorbate atoms)\n'
        )
        t = a.pivot_table(index='bin', columns='group', values='err', aggfunc=['mean', 'size'])
        t.columns = [f'{x}_{g}' for x, g in t.columns]
        md(t)
        out[split] = dict(res=res, s=s, a=a)
    return out


def top20(e, st):
    rows, sets = [], {}
    cols = [
        'relaxation',
        'split',
        'adsorbate',
        'slab_id',
        'chemsys',
        'material_class',
        'spin_on',
        'n_atoms',
        'adsorption_energy',
        'eads_oc20',
        'eads_omat',
        'err_oc20',
        'err_omat',
        'fvec_mean_final',
        'fvec_max_final',
        'dmag',
        'min_dist_ads_surf',
        'binding_element',
        'ads_coordination',
        'max_disp_ads',
        *FLAGS,
    ]
    for split in SPLITS:
        d = e[e.split == split].set_index('relaxation')
        s = st[(st.split == split) & (st.kind == 'adslab') & st.relaxation.isin(d.index)]
        worst = s.groupby('relaxation').agg(fvec_mean_worst=('fvec_mean', 'max'), fvec_max_worst=('fvec_max', 'max'))
        d = d.join(worst)
        for metric in ('abs_err_oc20', 'fvec_mean_worst', 'fvec_max_worst'):
            top = d.nlargest(20, metric).reset_index()
            sets[(split, metric)] = set(top.relaxation)
            top['list'], top['rank'] = metric, np.arange(1, 21)
            rows.append(top[['list', 'rank', metric] + cols].rename(columns={metric: 'value'}))
            print(f'### {split}: top-20 {metric}\n')
            md(top[['rank', metric] + cols].set_index('rank'))
    print('### Top-20 overlap (relaxation level)\n')
    ov = {
        split: {
            'E ∩ F_mean': len(sets[(split, 'abs_err_oc20')] & sets[(split, 'fvec_mean_worst')]),
            'E ∩ F_max': len(sets[(split, 'abs_err_oc20')] & sets[(split, 'fvec_max_worst')]),
            'F_mean ∩ F_max': len(sets[(split, 'fvec_mean_worst')] & sets[(split, 'fvec_max_worst')]),
            'all three': len(
                sets[(split, 'abs_err_oc20')] & sets[(split, 'fvec_mean_worst')] & sets[(split, 'fvec_max_worst')]
            ),
        }
        for split in SPLITS
    }
    md(pd.DataFrame(ov))
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(D + 'sprint3_top20.csv', index=False)
    return ov, t


def plots(e, st, at, fa):
    color = {'test': 'tab:blue', 'val': 'tab:orange'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, split in zip(ax, SPLITS):
        d = e[e.split == split]
        a.scatter(d.adsorption_energy, d.eads_oc20, s=6, c=np.where(d.spin_on, 'tab:red', 'tab:gray'), alpha=0.6)
        lim = [min(d.adsorption_energy.min(), d.eads_oc20.min()), max(d.adsorption_energy.max(), d.eads_oc20.max())]
        a.plot(lim, lim, 'k--', lw=0.8)
        a.set(title=f'{split} (red = spin-on), n={len(d)}', xlabel='E_ads DFT (eV)', ylabel='E_ads UMA oc20 (eV)')
    fig.tight_layout()
    fig.savefig(P + '01_parity_eads.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(-4, 1.2, 50)
    for split in SPLITS:
        ax.hist(e[e.split == split].abs_err_oc20, bins=bins, histtype='step', label=split, color=color[split])
    ax.axvline(0.1, ls=':', c='k', label='0.1 eV')
    ax.set(xscale='log', xlabel='|ΔE_ads| (eV), final frame', ylabel='relaxations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(P + '02_hist_abs_err_eads.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(-4, 1.5, 60)
    for split in SPLITS:
        ax.hist(at[at.split == split].err, bins=bins, histtype='step', label=split, color=color[split], density=True)
    ax.set(xscale='log', xlabel='|ΔF_i| (eV/Å), free atoms, all adslab frames', ylabel='density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(P + '03_hist_force_err_atom.png', dpi=130)
    plt.close(fig)

    for fname, y, ylab in (
        ('04_abs_err_eads_vs_natoms.png', 'abs_err_oc20', '|ΔE_ads| (eV)'),
        ('05_force_err_vs_natoms.png', 'fvec_mean_final', 'mean |ΔF_i| final frame (eV/Å)'),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for split in SPLITS:
            d = e[e.split == split]
            ax.scatter(d.n_atoms, d[y], s=6, alpha=0.5, label=split, c=color[split])
        ax.set(yscale='log', xlabel='number of atoms', ylabel=ylab)
        ax.legend()
        fig.tight_layout()
        fig.savefig(P + fname, dpi=130)
        plt.close(fig)

    ex = e.assign(element=e.slab_elements.str.split()).explode('element')
    g = ex.groupby(['element', 'split']).abs_err_oc20.agg(['mean', 'size']).unstack()
    g = g.sort_values(('mean', 'val'), ascending=False)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    x = np.arange(len(g))
    for k, split in enumerate(SPLITS):
        ax.bar(x + (k - 0.5) * 0.4, g[('mean', split)], 0.4, label=split, color=color[split])
    ax.set_xticks(
        x,
        [
            f'{el}\n{int(g.loc[el, ("size", "test")] if not pd.isna(g.loc[el, ("size", "test")]) else 0)}/'
            f'{int(g.loc[el, ("size", "val")] if not pd.isna(g.loc[el, ("size", "val")]) else 0)}'
            for el in g.index
        ],
        fontsize=6,
    )
    ax.set(ylabel='MAE |ΔE_ads| (eV)', title='MAE per slab element (label: n test/val)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(P + '06_mae_per_element.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw=dict(width_ratios=[1, 0.6, 3]))
    for a, col in zip(ax, ('material_class', 'spin_on', 'adsorbate')):
        t = e.groupby([col, 'split']).abs_err_oc20.mean().unstack()
        t.plot.bar(ax=a, color=[color[s] for s in t.columns], width=0.8)
        a.set(ylabel='MAE (eV)', xlabel=col)
        a.tick_params(axis='x', labelsize=7)
    fig.tight_layout()
    fig.savefig(P + '07_mae_class_spin_adsorbate.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for split in SPLITS:
        s = st[(st.split == split) & (st.kind == 'adslab')]
        ax.scatter(s.fvec_mean, s.fvec_max, s=3, alpha=0.3, label=split, c=color[split])
    ax.set(
        xscale='log', yscale='log', xlabel='mean |ΔF_i| per structure (eV/Å)', ylabel='max |ΔF_i| per structure (eV/Å)'
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(P + '08_force_max_vs_mean.png', dpi=130)
    plt.close(fig)

    feats = FEATURES
    fig, ax = plt.subplots(2, 7, figsize=(20, 6))
    for a, f in zip(ax.flat, feats):
        data, labels = [], []
        for split in SPLITS:
            d = e[e.split == split]
            data += [
                d[d.abs_err_oc20 <= d.abs_err_oc20.median()][f],
                d[d.abs_err_oc20 >= d.abs_err_oc20.quantile(0.9)][f],
            ]
            labels += [f'{split}\ngood', f'{split}\nbad']
        a.boxplot(data, showfliers=False)
        a.set_xticks(range(1, 5), labels, fontsize=6)
        a.set_title(f, fontsize=8)
    ax.flat[-1].axis('off')
    fig.suptitle('Features: best half vs worst decile of |ΔE_ads|')
    fig.tight_layout()
    fig.savefig(P + '09_features_best_vs_worst.png', dpi=130)
    plt.close(fig)


def render_worst(e, at, n=3):
    keys = {}
    for split in SPLITS:
        for r in e[e.split == split].nlargest(n, 'abs_err_oc20').itertuples():
            keys[r.key] = r
    found = {}
    for split in SPLITS:
        for atoms in iread(EXTXYZ[split]):
            if atoms.info['key'] in keys:
                found[atoms.info['key']] = atoms
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(len(keys), 2, figsize=(9, 3.6 * len(keys)))
    for row, (k, r) in enumerate(keys.items()):
        atoms = found[k]
        err = np.zeros(len(atoms))
        free = np.where(atoms.get_tags() != 0)[0]
        err[free] = at[at.key == k].err.values
        vmax = err.max()
        colors = [cmap(v / vmax) if i in set(free) else (0.8, 0.8, 0.8, 1) for i, v in enumerate(err)]
        for a, rot in zip(ax[row], ('0x,0y,0z', '-90x,0y,0z')):
            plot_atoms(atoms, a, rotation=rot, colors=colors, radii=0.8 * covalent_radii[atoms.numbers] / 1.0)
            a.axis('off')
        ax[row, 0].set_title(
            f'{r.split} {r.relaxation} {r.adsorbate} {r.chemsys}\n|ΔE|={r.abs_err_oc20:.2f} eV, '
            f'max |ΔF_i|={vmax:.2f} eV/Å (yellow)',
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(P + '10_render_worst.png', dpi=130)
    plt.close(fig)


def main():
    os.makedirs(P, exist_ok=True)
    e, st, at, use = build()
    print('## Gates\n')
    rows = pd.concat([pd.read_json(S2 + f'raw_{s}/rows.jsonl', lines=True) for s in SPLITS]).drop_duplicates(
        'key', keep='last'
    )
    chk = st.merge(rows[['key', 'force_vec_err_mean_free_oc20', 'force_vec_err_max_free_oc20']], on='key')
    g1 = max(
        (chk.fvec_mean - chk.force_vec_err_mean_free_oc20).abs().max(),
        (chk.fvec_max - chk.force_vec_err_max_free_oc20).abs().max(),
    )
    n = e.groupby('split').size().to_dict()
    nonorg = e[~e.organic]
    print(
        f'- per-structure force error recomputed from npz + extxyz vs sprint 2 rows.jsonl: {len(chk)} structures, '
        f'max |diff| = {g1:.2e} eV/Å'
    )
    print(
        f'- final frames (all, incl. organic): {n}; non-organic: {nonorg.groupby("split").size().to_dict()} '
        f'(sprint 2: test 932, val 963)'
    )
    print('- bond matrix of bonded() equal to AdsorbML NeighborList connectivity on every final frame (assert)')
    print(
        f'- relaxations whose first stored frame is from the _rerun continuation (true initial frame missing): '
        f'{e.groupby("split").rerun_only.sum().to_dict()}\n'
    )
    assert g1 < 1e-3 and nonorg.groupby('split').size().to_dict() == {'test': 932, 'val': 963}
    org = e[e.organic]
    e = nonorg.reset_index(drop=True)
    e.to_csv(D + 'sprint3_relaxations.csv', index=False)
    st.to_csv(D + 'sprint3_structures.csv', index=False)
    at = at[at.key.isin(set(use[~use.organic].key))]

    print('## Metrics (task 14, oc20)\n')
    metrics(e, st[~st.key.isin(set(use[use.organic].key))], at, use)
    print(
        f'Organic (final frames, reported separately): {org.groupby("split").abs_err_oc20.agg(["size", "mean", "median", "max"]).round(4).to_dict("index")}\n'
    )

    print('## AdsorbML anomalies (final frames)\n')
    t = e.groupby('split')[list(FLAGS)].sum()
    t['any'] = e.assign(any=e[list(FLAGS)].any(axis=1)).groupby('split')['any'].sum()
    md(t)
    for f in FLAGS:
        x = e[e[f]]
        if len(x):
            print(
                f'{f}: MAE flagged {x.abs_err_oc20.mean():.3f} eV (n={len(x)}) vs rest '
                f'{e[~e[f]].abs_err_oc20.mean():.3f}; per split {x.groupby("split").size().to_dict()}'
            )
    print()

    print('## Worst cases (task 15)\n')
    ov, top = top20(e, st)

    print('## Chemical groups and anomalies (task 16, MAE of group − MAE of the rest, eV)\n')
    groups = pd.concat([group_table(e, s) for s in SPLITS], ignore_index=True)
    gi = groups.set_index(['kind', 'group', 'split'])
    gv = []
    for (kind, name), _ in groups.groupby(['kind', 'group']):
        t_, v_ = gi.loc[(kind, name, 'test')], gi.loc[(kind, name, 'val')]
        gv.append(dict(kind=kind, group=name, verdict=verdict_group(t_, v_)))
    gv = pd.DataFrame(gv)
    wide = groups.pivot_table(
        index=['kind', 'group'],
        columns='split',
        values=['n', 'bulks', 'mae_in', 'diff', 'ci_low', 'ci_high'],
        aggfunc='first',
    )
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    wide = wide.join(gv.set_index(['kind', 'group']))
    for kind in ('material_class', 'spin_on', 'anomaly', 'binding_element'):
        print(f'### {kind}\n')
        md(wide.loc[kind])
    el = wide.loc['element']
    print('### element (elements with ≥ 5 bulks in both splits; the rest only in the CSV)\n')
    md(el[(el.bulks_test >= MIN_BULKS) & (el.bulks_val >= MIN_BULKS)].sort_values('diff_val', ascending=False))
    cs = wide.loc['chemsys']
    print(
        f'### chemsys: {len(cs)} systems; with ≥ 5 bulks in both splits: '
        f'{int(((cs.bulks_test >= MIN_BULKS) & (cs.bulks_val >= MIN_BULKS)).sum())}. '
        f'10 highest val MAE (descriptive, small n)\n'
    )
    md(cs.sort_values('mae_in_val', ascending=False).head(10))
    print('### adsorbate (descriptive, cannot be replicated)\n')
    md(
        e.groupby(['split', 'adsorbate']).agg(
            n=('relaxation', 'size'),
            bulks=('bulk', 'nunique'),
            mae=('abs_err_oc20', 'mean'),
            mean_err=('err_oc20', 'mean'),
        )
    )

    print('## Continuous features vs |ΔE_ads| (task 17, Spearman, CI 95 % bootstrap over bulks)\n')
    sp = pd.concat([spearman_table(e, s) for s in SPLITS], ignore_index=True)
    si = sp.set_index(['feature', 'split'])
    spw = sp.pivot(index='feature', columns='split', values=['rho', 'ci_low', 'ci_high'])
    spw.columns = [f'{a}_{b}' for a, b in spw.columns]
    spw['verdict'] = [verdict_rho(si.loc[(f, 'test')], si.loc[(f, 'val')]) for f in spw.index]
    md(spw.loc[FEATURES])

    print('## Per-atom forces (task 18)\n')
    fa = force_analysis(st, at, e)

    cand = pd.concat(
        [spw.reset_index().assign(kind='feature').rename(columns={'feature': 'group'}), wide.reset_index()],
        ignore_index=True,
    )
    cand.to_csv(D + 'sprint3_candidates.csv', index=False)
    print('## Candidate rule summary\n')
    md(
        cand[cand.verdict != '-'][
            [
                'kind',
                'group',
                'verdict',
                'rho_test',
                'rho_val',
                'diff_test',
                'diff_val',
                'ci_low_test',
                'ci_high_test',
                'ci_low_val',
                'ci_high_val',
            ]
        ].set_index(['kind', 'group'])
    )

    plots(e, st[st.key.isin(set(use[~use.organic].key)) | (st.kind == 'slab')], at, fa)
    render_worst(e, at)
    summary = dict(
        n=e.groupby('split').size().to_dict(),
        force_gate_max_diff=float(g1),
        anomalies=t.to_dict('index'),
        top20_overlap=ov,
        force_share={s: fa[s]['res'] for s in SPLITS},
        candidates=cand[cand.verdict != '-'][['kind', 'group', 'verdict']].to_dict('records'),
    )
    json.dump(summary, open(D + 'sprint3_summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
