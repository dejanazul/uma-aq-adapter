import hashlib
import io
import json
import lzma
import os
import sys
import tarfile
from collections import defaultdict

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.db import connect
from ase.io import read

NONMETAL = {'H', 'O', 'N', 'C', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Se'}
SHARD = 20000


def prepare(atoms, energy, forces):
    a = atoms.copy()
    fixed = a.get_tags() == 0
    for c in a.constraints:
        if isinstance(c, FixAtoms):
            assert set(c.index.tolist()) == set(np.where(fixed)[0].tolist()), 'FixAtoms disagrees with tag 0'
    a.set_constraint(FixAtoms(mask=fixed))
    a.calc = SinglePointCalculator(a, energy=float(energy), forces=np.asarray(forces, float))
    return a


class ShardWriter:
    def __init__(self, root, name):
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.n, self.db, self.natoms = 0, None, []

    def write(self, atoms, **kv):
        if self.n % SHARD == 0:
            self.db = connect(os.path.join(self.dir, f'data.{self.n // SHARD:04d}.aselmdb'), type='aselmdb')
        self.db.write(atoms, **kv)
        self.natoms.append(len(atoms))
        self.n += 1
        if self.n % 1000 == 0:
            self.close()

    def close(self):
        np.savez_compressed(os.path.join(self.dir, 'metadata.npz'), natoms=np.array(self.natoms, dtype=np.int64))


def is_organic(atoms):
    slab = {s for s, t in zip(atoms.get_chemical_symbols(), atoms.get_tags()) if t != 2}
    return slab <= NONMETAL


def fit_elem_refs(comps, energies):
    X = np.array([np.bincount(z, minlength=100) for z in comps], float)
    coef = np.linalg.lstsq(X, np.array(energies), rcond=None)[0]
    return coef.tolist()


class HashingReader:
    def __init__(self, f):
        self.f, self.md5, self.n = f, hashlib.md5(), 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.md5.update(b)
        self.n += len(b)
        return b


def stream(url, token=None):
    import requests

    r = requests.get(url, headers={'Authorization': f'Bearer {token}'} if token else {}, stream=True, timeout=60)
    r.raise_for_status()
    r.raw.decode_content = False
    return HashingReader(r.raw)


def smoke(out, aq_db, oc20_xz, n):
    n = int(n)
    db = connect(aq_db, type='aselmdb', readonly=True)
    ws = {k: ShardWriter(out, k) for k in ('aq_train_low', 'aq_train_high', 'aq_train_slab', 'aq_val')}
    comps, es = [], []
    for i in range(1, 4 * n + 1):
        r = db.get(i)
        a = r.toatoms()
        p = prepare(a, a.get_potential_energy(), a.get_forces())
        ws[list(ws)[(i - 1) // n]].write(p)
        comps.append(a.numbers)
        es.append(a.get_potential_energy())
    frames = read(lzma.open(oc20_xz, 'rt'), index=f':{2 * n}', format='extxyz')
    ow = {k: ShardWriter(out, k) for k in ('oc20_train', 'oc20_val')}
    for j, a in enumerate(frames):
        ow['oc20_train' if j < n else 'oc20_val'].write(prepare(a, a.get_potential_energy(), a.get_forces()))
    for w in (*ws.values(), *ow.values()):
        w.close()
    json.dump(fit_elem_refs(comps, es), open(os.path.join(out, 'aq_elem_refs.json'), 'w'))
    print('smoke dbs', {k: w.n for k, w in {**ws, **ow}.items()})


def aq(out, selection, url_train, url_slabs):
    import pandas as pd

    sel = pd.read_csv(selection)
    token = os.environ['HF_TOKEN']
    writers = {s: ShardWriter(out, f'aq_{s}') for s in sel.split.unique()}
    comps, es, stats = [], [], defaultdict(int)
    for archive, url in (('train', url_train), ('slabs', url_slabs)):
        want = {f: g for f, g in sel[sel.archive == archive].groupby('file')}
        src = stream(url, token)
        with tarfile.open(fileobj=src, mode='r|gz') as tar:
            for m in tar:
                name = os.path.basename(m.name)
                if not m.isfile() or name not in want:
                    continue
                path = os.path.join('/content/tmp_aq', name)
                os.makedirs('/content/tmp_aq', exist_ok=True)
                with open(path, 'wb') as fh:
                    fh.write(tar.extractfile(m).read())
                db = connect(path, type='aselmdb', readonly=True)
                for r in want[name].itertuples():
                    row = db.get(int(r.row) + 1)
                    a = row.toatoms()
                    if is_organic(a):
                        stats['organic_skipped'] += 1
                        continue
                    writers[r.split].write(
                        prepare(a, a.get_potential_energy(), a.get_forces()),
                        sid=str(r.sid),
                        fid=int(r.fid),
                        frame_id=f'{name}::{int(r.row)}',
                    )
                    stats[r.split] += 1
                    if r.split.startswith('train'):
                        comps.append(a.numbers)
                        es.append(a.get_potential_energy())
                del db
                os.remove(path)
                print(archive, name, dict(stats), flush=True)
        stats[f'{archive}_bytes'], stats[f'{archive}_md5'] = src.n, src.md5.hexdigest()
    for w in writers.values():
        w.close()
    json.dump(fit_elem_refs(comps, es), open(os.path.join(out, 'aq_elem_refs.json'), 'w'))
    json.dump(dict(stats), open(os.path.join(out, 'aq_stats.json'), 'w'), indent=1)


def oc20(out, url, n_val, md5):
    n_val = int(n_val)
    frames = []
    src = stream(url)
    with tarfile.open(fileobj=src, mode='r|') as tar:
        tw = ShardWriter(out, 'oc20_train')
        for m in tar:
            if not m.name.endswith('.extxyz.xz'):
                continue
            text = lzma.decompress(tar.extractfile(m).read()).decode()
            for a in read(io.StringIO(text), index=':', format='extxyz'):
                frames.append(prepare(a, a.get_potential_energy(), a.get_forces()))
                if len(frames) > n_val:
                    tw.write(frames.pop(0))
            print('oc20', m.name, tw.n, flush=True)
    vw = ShardWriter(out, 'oc20_val')
    for a in frames:
        vw.write(a)
    tw.close()
    vw.close()
    while src.read(1 << 20):
        pass
    stats = dict(
        train=tw.n, val=vw.n, bytes=src.n, md5=src.md5.hexdigest(), md5_expected=md5, md5_ok=src.md5.hexdigest() == md5
    )
    json.dump(stats, open(os.path.join(out, 'oc20_stats.json'), 'w'), indent=1)
    assert stats['md5_ok'], stats


if __name__ == '__main__':
    {'smoke': smoke, 'aq': aq, 'oc20': oc20}[sys.argv[1]](*sys.argv[2:])
