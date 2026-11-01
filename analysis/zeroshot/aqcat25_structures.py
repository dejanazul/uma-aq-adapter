import glob
import os
import sys
import tarfile
import numpy as np
import pandas as pd
from ase.db import connect

ORGANIC = {'H', 'O', 'N', 'C', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se'}
METALLOID = {'B', 'Si', 'Ge', 'As', 'Sb', 'Te'}
SPIN_ELEMENTS = {'Ce', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Os', 'Ru', 'V', 'W'}


def material_class(els):
    m, n = bool(els & METALLOID), bool(els & ORGANIC)
    return {
        (False, False): 'metal',
        (True, False): 'metalloid',
        (False, True): 'non-metal',
        (True, True): 'metalloid+non-metal',
    }[(m, n)]


def main(split):
    raw = f'data/raw/{split}'
    if not glob.glob(f'{raw}/**/*.aselmdb', recursive=True):
        os.makedirs(raw, exist_ok=True)
        with tarfile.open(f'data/tar/{split}.tar.gz') as t:
            t.extractall(raw, filter='data')
    dbs = {os.path.basename(p): p for p in glob.glob(f'{raw}/**/*.aselmdb', recursive=True)}
    d = pd.read_parquet(f'data/parquet/{split}.parquet').reset_index()
    base = d.sid.str.removesuffix('_rerun')
    last = d.loc[d.groupby(base).fid.idxmax()]
    out = []
    for name, rows in last.groupby(last.frame_id.str.split('::').str[0]):
        with connect(dbs[name], type='aselmdb') as db:
            for r in rows.itertuples():
                atoms = db.get(id=int(r.frame_id.split('::')[1]) + 1).toatoms()
                f = np.linalg.norm(atoms.get_forces(), axis=1)
                fixed = atoms.get_tags() == 0
                els = {s for s, t in zip(atoms.get_chemical_symbols(), atoms.get_tags()) if t != 2}
                out.append(
                    dict(
                        relaxation=base[r.Index],
                        sid=r.sid,
                        frame_id=r.frame_id,
                        fid=r.fid,
                        adsorbate=r.adsorbate,
                        slab_id=r.slab_id,
                        n_atoms=len(atoms),
                        n_fixed=int(fixed.sum()),
                        fmax_parquet=r.fmax,
                        fmax_all=f.max(),
                        fmax_free=f[~fixed].max() if (~fixed).any() else np.nan,
                        adsorption_energy=r.adsorption_energy,
                        total_energy=r.total_energy,
                        energy_db=atoms.get_potential_energy(),
                        is_spin_off=r.is_spin_off,
                        total_magnetization=r.total_magnetization,
                        slab_elements=' '.join(sorted(els)),
                        material_class=material_class(els),
                        organic=els <= ORGANIC,
                        spin_rule_on=bool(els & SPIN_ELEMENTS),
                    )
                )
    pd.DataFrame(out).to_csv(f'results/zeroshot/aqcat25_{split}_final_frames.csv', index=False)
    print('wrote', len(out), 'relaxations')


if __name__ == '__main__':
    assert material_class({'Pt', 'Ni'}) == 'metal' and material_class({'Si', 'O'}) == 'metalloid+non-metal'
    assert material_class({'Ga', 'As'}) == 'metalloid' and material_class({'Cu', 'S'}) == 'non-metal'
    main(sys.argv[1])
