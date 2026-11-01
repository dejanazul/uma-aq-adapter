import json
import sys

import numpy as np
import pandas as pd

N_PER, N_SLAB, VAL_FRAC, CUTS = 5, 3, 0.05, (0.1408, 1.0890)
KINDS = ('idadslab02c', 'idadslab23c')


def even(g, n):
    return g.iloc[np.unique(np.round(np.linspace(0, len(g) - 1, n)).astype(int))]


def main(train_pq, slabs_pq, out):
    t = pd.read_parquet(train_pq).reset_index()
    t['relaxation'] = t.sid.astype(str).str.removesuffix('_rerun')
    t = t[t.relaxation.str.extract(r'^([a-z]+\d*[a-z]*)_')[0].isin(KINDS)].sort_values(['relaxation', 'fid'])
    final = t.groupby('relaxation').tail(1).set_index('relaxation')
    s = pd.read_parquet(slabs_pq).reset_index().sort_values(['sid', 'fid'])
    slab_final = s.groupby('sid').tail(1)
    dup = slab_final.slab_id.duplicated(keep=False)
    slab_mag = slab_final[~dup].set_index('slab_id').total_magnetization

    rel = final[['slab_id', 'is_spin_off', 'total_magnetization']].copy()
    rel['slab_mag'] = rel.slab_id.map(slab_mag)
    rel = rel[rel.slab_mag.notna()]
    rel['adm'] = (rel.total_magnetization.abs() - rel.slab_mag.abs()).abs()
    rel['stratum'] = np.where(
        rel.is_spin_off, 'S0', np.where(rel.adm <= CUTS[0], 'S1', np.where(rel.adm <= CUTS[1], 'S2', 'S3'))
    )
    rel['bulk'] = rel.slab_id.str.split('_').str[0]
    bulks = np.array(sorted(rel.bulk.unique()))
    val_bulks = set(np.random.default_rng(0).permutation(bulks)[: int(round(VAL_FRAC * len(bulks)))])
    rel['is_val'] = rel.bulk.isin(val_bulks)

    ok = t[(t.fmax <= 50) & (t.adsorption_energy.abs() <= 10) & t.relaxation.isin(rel.index)]
    rows = []
    for r, g in ok.groupby('relaxation', sort=False):
        info = rel.loc[r]
        if info.is_val:
            sel, split = g.iloc[[len(g) // 2, len(g) - 1]].drop_duplicates('index'), 'val'
        else:
            sel, split = even(g, N_PER), 'train_low' if info.stratum in ('S0', 'S1') else 'train_high'
        rows.append(sel.assign(split=split, stratum=info.stratum, bulk=info.bulk, archive='train'))
    ads = pd.concat(rows)

    used = rel.groupby('slab_id').is_val.first()
    srows = []
    for _, g in s[s.slab_id.isin(used.index) & (s.fmax <= 50)].groupby('sid', sort=False):
        is_val = used[g.slab_id.iloc[0]]
        sel = g.tail(1) if is_val else even(g, N_SLAB)
        srows.append(
            sel.assign(
                split='val' if is_val else 'train_slab',
                stratum='slab',
                bulk=g.slab_id.iloc[0].split('_')[0],
                archive='slabs',
            )
        )
    slabs = pd.concat(srows)

    allf = pd.concat([ads, slabs])
    allf['file'] = allf.frame_id.str.split('::').str[0]
    allf['row'] = allf.frame_id.str.split('::').str[1].astype(int)
    allf = allf[
        [
            'archive',
            'file',
            'row',
            'split',
            'sid',
            'fid',
            'stratum',
            'bulk',
            'slab_id',
            'adsorbate',
            'total_magnetization',
        ]
    ]
    allf.to_csv(f'{out}/selection.csv', index=False)
    summary = dict(
        relaxations=int(len(rel)),
        relaxations_no_slab=int(final.slab_id.map(slab_mag).isna().sum()),
        slab_ids_duplicated=int(slab_final[dup].slab_id.nunique()),
        bulks=int(len(bulks)),
        val_bulks=len(val_bulks),
        strata_relaxations=rel.groupby(['is_val', 'stratum']).size().unstack(0).fillna(0).astype(int).to_dict(),
        frames_per_split=allf.groupby(['split', 'stratum']).size().to_dict(),
        files_needed=allf.groupby('archive').file.nunique().to_dict(),
    )
    summary['frames_per_split'] = {f'{a}|{b}': int(v) for (a, b), v in summary['frames_per_split'].items()}
    json.dump(summary, open(f'{out}/selection_summary.json', 'w'), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == '__main__':
    main(*sys.argv[1:])
