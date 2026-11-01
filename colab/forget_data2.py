import json
import sys

from forget_data import fetch

BASE = 'https://dl.fbaipublicfiles.com/opencatalystproject/data'
SOURCES = {
    'omol25_val_neutral': (f'{BASE}/omol/250514/neutral_val.tar.gz', 124_585_282),
    'oc25_val': (f'{BASE}/oc25/250825/oc25_val.tar.gz', 1_644_441_991),
    'omol25_val': (f'{BASE}/omol/250514/val.tar.gz', 21_293_980_744),
}


def main(drive, sources=SOURCES, record='forget_data2.json'):
    res = {}
    for name, (url, size) in sources.items():
        print(name, url, flush=True)
        r = fetch(url, f'{drive}/{name}')
        r.update(url=url, bytes_expected=size, bytes_ok=r['bytes'] == size)
        res[name] = r
        print(json.dumps(r), flush=True)
        json.dump(res, open(f'{drive}/{record}', 'w'), indent=1)
    bad = [n for n, r in res.items() if not r['bytes_ok']]
    assert not bad, f'byte count mismatch: {bad}'


if __name__ == '__main__':
    main(sys.argv[1])
