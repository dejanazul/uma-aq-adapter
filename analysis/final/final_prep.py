import hashlib
import os

import pandas as pd
from ase.io import iread, write

OUT = 'data/final_eval/aq_final.extxyz'
SOURCES = {
    'test': ('data/zeroshot/uma_zeroshot_frames.extxyz', 'results/zeroshot/uma_zeroshot_manifest.csv'),
    'val': (
        'data/zeroshot/uma_zeroshot_frames_val.extxyz',
        'results/failure_analysis/sprint2/sprint2_val_manifest.csv',
    ),
}


def main():
    rel = pd.read_csv('results/failure_analysis/sprint4/sprint4_relaxations.csv')
    frames = []
    for split, (xyz, manifest) in SOURCES.items():
        r = rel[rel.split == split]
        m = pd.read_csv(manifest)
        slabs = m[(m.kind == 'slab') & m.slab_id.isin(r.slab_id)]
        want = {
            **{k: ('adslab', s) for k, s in zip(r.key, r.slab_id)},
            **{k: ('slab', s) for k, s in zip(slabs.key, slabs.slab_id)},
        }
        got = 0
        for a in iread(xyz):
            if a.info['key'] in want:
                a.info['kind'], a.info['slab_id'] = want[a.info['key']]
                a.info['split'] = split
                frames.append(a)
                got += 1
        assert got == len(want), (split, got, len(want))
        assert slabs.slab_id.nunique() == r.slab_id.nunique(), 'every relaxation needs its slab frame'
        print(split, len(r), 'adslab', len(slabs), 'slab')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    write(OUT, frames, format='extxyz')
    print(OUT, len(frames), hashlib.sha256(open(OUT, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
