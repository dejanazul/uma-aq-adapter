import hashlib
import json
import sys
import tarfile
import urllib.request

OMAT = 'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/{}.tar.gz'
OMAT_SETS = [
    'rattled-1000',
    'rattled-1000-subsampled',
    'rattled-500',
    'rattled-500-subsampled',
    'rattled-300',
    'rattled-300-subsampled',
    'aimd-from-PBE-1000-npt',
    'aimd-from-PBE-1000-nvt',
    'aimd-from-PBE-3000-npt',
    'aimd-from-PBE-3000-nvt',
    'rattled-relax',
]
OC22 = 'https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/s2ef_total_train_val_test_lmdbs.tar.gz'
OC22_MD5 = 'ebea523c6f8d61248a37b4dd660b11e6'


class HashingReader:
    def __init__(self, f):
        self.f, self.md5, self.n = f, hashlib.md5(), 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.md5.update(b)
        self.n += len(b)
        return b


def is_val(name):
    parts = name.split('/')
    return any(p.startswith('val') for p in parts[:-1])


def fetch(url, out_dir, keep=lambda name: True):
    src = HashingReader(urllib.request.urlopen(url))
    kept = 0
    with tarfile.open(fileobj=src, mode='r|gz') as tar:
        for m in tar:
            if m.isfile() and keep(m.name):
                tar.extract(m, out_dir, filter='data')
                kept += 1
                print('  kept', m.name, m.size, flush=True)
    src.read()
    return dict(bytes=src.n, md5=src.md5.hexdigest(), files_kept=kept)


def main(drive):
    res = {}
    for s in OMAT_SETS:
        print('omat24', s, flush=True)
        res[f'omat24/{s}'] = fetch(OMAT.format(s), f'{drive}/omat24_val/{s}')
    print('oc22', flush=True)
    r = fetch(OC22, f'{drive}/oc22_val', keep=is_val)
    r['md5_expected'], r['md5_ok'] = OC22_MD5, r['md5'] == OC22_MD5
    res['oc22'] = r
    json.dump(res, open(f'{drive}/forget_data.json', 'w'), indent=1)
    print(json.dumps(res['oc22']))
    assert r['md5_ok'], 'OC22 MD5 mismatch'


if __name__ == '__main__':
    assert is_val('s2ef_total/val_id/data.0000.lmdb') and is_val('x/val_ood/a.lmdb') and not is_val('x/train/val.lmdb')
    main(sys.argv[1])
