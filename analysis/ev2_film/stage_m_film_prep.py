import hashlib

import pandas as pd
from ase.io import iread, write

SRC = {'test': 'data/zeroshot/uma_zeroshot_frames.extxyz', 'val': 'data/zeroshot/uma_zeroshot_frames_val.extxyz'}
OUT = 'data/zeroshot/film_frames.extxyz'

e = pd.read_csv('results/failure_analysis/sprint4/sprint4_relaxations.csv')
assert e.groupby('split').size().to_dict() == {'test': 932, 'val': 963}
assert e.key.is_unique
want = e.set_index('key')
frames = []
for split, path in SRC.items():
    for a in iread(path):
        k = a.info['key']
        if k in want.index:
            r = want.loc[k]
            assert r.split == split and a.info['kind'] == 'adslab' and len(a) == r.n_atoms
            assert abs(a.get_potential_energy() - r.total_energy) < 1e-6
            a.calc = None
            a.info = dict(key=k, split=split, relaxation=r.relaxation, is_spin_off=bool(r.is_spin_off))
            frames.append(a)
assert len(frames) == len(e), (len(frames), len(e))
write(OUT, frames, format='extxyz')
print(len(frames), 'frames', OUT, hashlib.sha256(open(OUT, 'rb').read()).hexdigest())
