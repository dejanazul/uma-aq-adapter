import glob
import gzip
import hashlib
import os
import shutil
import sys

import numpy as np
import pandas as pd
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write

OUTPUTS = {
    'test': ('', 'results/zeroshot/uma_zeroshot_manifest.csv'),
    'val': ('_val', 'results/failure_analysis/sprint2/sprint2_val_manifest.csv'),
}


def db_paths(split):
    return {os.path.basename(p): p for p in glob.glob(f'data/raw/{split}/**/*.aselmdb', recursive=True)}


def read_rows(rows, dbs):
    for name, part in rows.groupby(rows.frame_id.str.split('::').str[0]):
        with connect(dbs[name], type='aselmdb') as db:
            for r in part.itertuples():
                row = db.get(id=int(r.frame_id.split('::')[1]) + 1)
                atoms = row.toatoms()
                energy, forces = atoms.get_potential_energy(), atoms.get_forces()
                atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
                yield r, atoms


def main(n_per, split):
    suffix, manifest = OUTPUTS[split]
    d = pd.read_parquet(f'data/parquet/{split}_ood_both.parquet').reset_index()
    d['relaxation'] = d.sid.str.removesuffix('_rerun')
    d = d.sort_values(['relaxation', 'fid'])
    picked = []
    for _, g in d.groupby('relaxation', sort=True):
        idx = np.unique(np.round(np.linspace(0, len(g) - 1, n_per)).astype(int))
        sel = g.iloc[idx].copy()
        sel['is_final'] = False
        sel.iloc[-1, sel.columns.get_loc('is_final')] = True
        picked.append(sel)
    ads = pd.concat(picked)
    ads['kind'] = 'adslab'

    s = pd.read_parquet(f'data/parquet/{split}_ood_slabs.parquet').reset_index()
    s = s[s.slab_id.isin(ads.slab_id.unique())]
    s = s.loc[s.groupby('sid').fid.idxmax()].copy()
    assert s.slab_id.is_unique, 'one slab relaxation per slab_id expected'
    s['relaxation'], s['is_final'], s['kind'] = s.sid, True, 'slab'

    cols = [
        'kind',
        'frame_id',
        'relaxation',
        'sid',
        'fid',
        'is_final',
        'slab_id',
        'adsorbate',
        'is_spin_off',
        'adsorption_energy',
        'total_energy',
        'fmax',
        'total_magnetization',
    ]
    out = pd.concat([ads[cols], s[cols]], ignore_index=True)
    out['key'] = ('' if split == 'test' else f'{split}:') + out.kind + ':' + out.frame_id
    assert out.key.is_unique

    os.makedirs('data/zeroshot', exist_ok=True)
    path = f'data/zeroshot/uma_zeroshot_frames{suffix}.extxyz'
    frames = []
    for kind, source in (('adslab', f'{split}_ood_both'), ('slab', f'{split}_ood_slabs')):
        part = out[out.kind == kind]
        for r, atoms in read_rows(part, db_paths(source)):
            assert abs(atoms.get_potential_energy() - r.total_energy) < 1e-5, r.key
            atoms.info.update(key=r.key, kind=kind)
            frames.append(atoms)
    order = {k: i for i, k in enumerate(out.key)}
    frames.sort(key=lambda a: order[a.info['key']])
    write(path, frames, format='extxyz')
    with open(path, 'rb') as src, gzip.open(path + '.gz', 'wb') as dst:
        shutil.copyfileobj(src, dst)
    out['n_atoms'] = [len(a) for a in frames]
    out.to_csv(manifest, index=False)
    for p in (path + '.gz', manifest):
        print(p, hashlib.sha256(open(p, 'rb').read()).hexdigest())
    print(out.groupby(['kind', 'is_final']).size().to_string(), '\ntotal', len(out), 'atoms', out.n_atoms.sum())


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10, sys.argv[2] if len(sys.argv) > 2 else 'test')
