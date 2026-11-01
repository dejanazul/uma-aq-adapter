import hashlib
import json
import re
import sys

AQCAT25_COMMIT = '2ec3a36c138ef0150ac9b1d783a7acf0aadfeda9'
EXTXYZ = ('data/zeroshot/uma_zeroshot_frames.extxyz', 'data/zeroshot/uma_zeroshot_frames_val.extxyz')
ENERGIES_SHA256 = 'e2661b68188c58338b989451738ce47f14f168d6169ac20f7a153e44e8bbc1e5'
FILES = (
    'results/zeroshot/uma_zeroshot_raw.jsonl',
    'results/failure_analysis/sprint2/raw_test/rows.jsonl',
    'results/failure_analysis/sprint2/raw_val/rows.jsonl',
    'results/adapter/final/aq.jsonl',
    'results/adapter/final/aqz.jsonl',
    'results/adapter/final/gate.jsonl',
    'results/adapter/final/gateh.jsonl',
)
AQ_VAL_FRAMES = 'results/adapter/aq_val_frames.csv'
LABELS = ('dft_total_eV', 'total_err_eV')
FRAME = re.compile(r'^((test|val):)?(adslab|slab):data\.\d{4}\.aselmdb::\d+$')


def frame_key(key):
    return key.split('|', 1)[-1]


def is_aqcat25(key):
    k = frame_key(key)
    return bool(FRAME.match(k)) or k.startswith('aq_val:')


def energies_sha256(energies):
    return hashlib.sha256(json.dumps(sorted(energies.items())).encode()).hexdigest()


def rewrite(path, fn):
    out = []
    for line in open(path):
        d = json.loads(line)
        assert json.dumps(d) == line.rstrip('\n'), f'{path}: line is not in json.dumps form'
        out.append(json.dumps(fn(d)) + '\n')
    open(path, 'w').write(''.join(out))


def strip(d):
    if is_aqcat25(d['key']):
        for f in LABELS:
            if f in d:
                d[f] = None
    return d


def dft_energies():
    from ase.io import iread

    energies = {a.info['key']: a.get_potential_energy() for p in EXTXYZ for a in iread(p)}
    got = energies_sha256(energies)
    assert got == ENERGIES_SHA256, (
        f'DFT energies in {EXTXYZ} (SHA-256 {got}) differ from the release; rebuild them from AQCat25 commit '
        f'{AQCAT25_COMMIT} with analysis/zeroshot/uma_zeroshot_prep.py'
    )
    try:
        import pandas as pd

        m = pd.read_csv(AQ_VAL_FRAMES)
        for name, g in m.groupby('parquet'):
            p = pd.read_parquet(f'data/parquet/{name}.parquet', columns=['frame_id', 'total_energy'])
            e = p.set_index('frame_id').total_energy
            energies.update(zip(g.key, g.frame_id.map(e)))
    except FileNotFoundError as err:
        print(f'aq_val gate rows not restored ({err.filename} missing)', file=sys.stderr)
    return energies


def restore(energies):
    def fn(d):
        if is_aqcat25(d['key']) and d.get('dft_total_eV', 0) is None:
            k = frame_key(d['key'])
            if k in energies:
                d['dft_total_eV'] = energies[k]
                if 'total_err_eV' in d:
                    d['total_err_eV'] = d['pred_total_eV'] - energies[k]
        return d

    return fn


if __name__ == '__main__':
    cmd = sys.argv[1]
    assert cmd in ('strip', 'restore'), 'usage: python -I analysis/record2/labels.py strip|restore'
    fn = strip if cmd == 'strip' else restore(dft_energies())
    for path in FILES:
        rewrite(path, fn)
        rows = [json.loads(l) for l in open(path)]
        left = sum(1 for d in rows if is_aqcat25(d['key']) and d.get('dft_total_eV', 0) is None)
        print(path, len(rows), 'rows', left, 'AQCat25 rows without labels')
