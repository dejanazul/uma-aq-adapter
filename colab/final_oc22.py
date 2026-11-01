import glob
import os
import pickle

import lmdb
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

import final_eval as fe

ROOT = f'{fe.FORGET}/oc22_val/s2ef_total_train_val_test_lmdbs/data/oc22/s2ef-total'


def open_env(p):
    return lmdb.open(p, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1)


def to_atoms(d):
    if '_store' not in d.__dict__:
        from torch_geometric.data import Data

        d = Data(**{k: v for k, v in d.__dict__.items() if v is not None})
    a = Atoms(
        numbers=d.atomic_numbers.long().numpy(),
        positions=d.pos.numpy(),
        cell=d.cell.view(3, 3).numpy(),
        pbc=True,
        tags=d.tags.long().numpy() if 'tags' in d else None,
    )
    a.set_constraint(FixAtoms(mask=d.fixed.bool().numpy()))
    a.calc = SinglePointCalculator(a, energy=float(d.y), forces=d.force.numpy())
    return a


def jobs():
    files = sorted(glob.glob(f'{ROOT}/val_id/*.lmdb') + glob.glob(f'{ROOT}/val_ood/*.lmdb'))
    envs = [open_env(p) for p in files]
    keys = []
    for i, e in enumerate(envs):
        with e.begin() as t:
            keys += [(i, k) for k, _ in t.cursor() if k != b'length']
    print('oc22_val oc22', len(files), 'files', len(keys), 'rows', flush=True)
    rng = np.random.default_rng(0)
    for g in np.sort(rng.choice(len(keys), size=fe.N_FORGET, replace=False)):
        i, k = keys[g]
        with envs[i].begin() as t:
            atoms = to_atoms(pickle.loads(t.get(k)))
        name = f'{os.path.basename(os.path.dirname(files[i]))}/{os.path.basename(files[i])}'
        yield f'oc22_val:{name}:{k.decode()}', atoms, 'oc22', dict(dataset='oc22_val')


def main():
    units = {'uma': fe.uma_unit(), 'l2': fe.l2_unit()[0], 'l2h': fe.l2_unit(hybrid=True)[0]}
    assert 'oc22' in units['uma'].dataset_to_tasks
    fe.run(units, jobs(), fe.Out('/content/final/forget.jsonl'))
    print('step done oc22', flush=True)


if __name__ == '__main__':
    main()
