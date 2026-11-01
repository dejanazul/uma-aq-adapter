import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import iread

ROOT = Path('/content/aqcat25-ev2/checkpoints_aqcat_ev2')
MODELS = {
    'midfilm': (
        'ev2-in+midFiLM-AQCat25+OC20-20M_20251008_223220.pt',
        'e1f14d50590102dbdf64491a6ae328df6ba0ca2ebb947fbe72213820ae67eb50',
    ),
    'infilm': (
        'ev2-inFiLM-OC20-ft-AQCat25+OC20-20M_20251008_223143.pt',
        '4af91c787c7f8234be911dafbc97fa1ff9f3914beacca5d117b3a899c657300e',
    ),
}
MODES = {'on': (False, False), 'off': (True, False), 'lo_off': (True, True)}


def load(name):
    from fairchem.core.common.relaxation.ase_utils import patched_calc

    path = ROOT / MODELS[name][0]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == MODELS[name][1], (name, sha)
    return patched_calc(checkpoint_path=str(path), cpu=False), sha


def predict(calc, atoms, mode):
    a = atoms.copy()
    a.info['is_spin_off'], a.info['is_low_fi'] = MODES[mode]
    calc.reset()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = a.get_forces()
    free = a.get_tags() != 0 if len(set(a.get_tags())) > 1 else np.ones(len(a), bool)
    return e, float(np.linalg.norm(f[free], axis=1).max())


def smoke(out):
    from ase.build import add_adsorbate, hcp0001, molecule

    slab = hcp0001('Co', size=(3, 4, 4), orthogonal=True)
    add_adsorbate(slab, molecule('CO'), 3.0, 'ontop')
    slab.center(vacuum=10.0, axis=2)
    slab.set_pbc(True)
    res = {}
    for name in MODELS:
        calc, sha = load(name)
        e = {m: predict(calc, slab, m)[0] for m in MODES}
        again = predict(calc, slab, 'on')[0]
        a = slab.copy()
        a.info['is_spin_off'] = False
        a.calc = calc
        first = a.get_potential_energy()
        a.info['is_spin_off'] = True
        stale = a.get_potential_energy()
        res[name] = dict(sha256=sha, energies=e, repeat_on=again, no_reset_returns_cached=bool(stale == first))
        assert abs(e['on'] - e['off']) > 1e-4 and abs(e['off'] - e['lo_off']) > 1e-4, e
        assert abs(again - e['on']) < 1e-5, (again, e['on'])
    json.dump(res, open(out / 'smoke.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('frames', nargs='?')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        return smoke(out)
    import torch
    import fairchem.core

    frames = list(iread(args.frames))
    rows_path = out / 'rows.jsonl'
    done = (
        {(r['key'], r['model']) for r in map(json.loads, rows_path.read_text().splitlines())}
        if rows_path.exists()
        else set()
    )
    info = dict(
        n_frames=len(frames),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        fairchem_core=fairchem.core.__version__,
        frames_sha256=hashlib.sha256(Path(args.frames).read_bytes()).hexdigest(),
        models={},
    )
    for name in MODELS:
        calc, sha = load(name)
        info['models'][name] = dict(file=MODELS[name][0], sha256=sha)
        t_model = time.time()
        with open(rows_path, 'a') as fh:
            for i, a in enumerate(frames):
                if (a.info['key'], name) in done:
                    continue
                t = time.time()
                row = dict(
                    key=a.info['key'], split=a.info['split'], relaxation=a.info['relaxation'], model=name, status='ok'
                )
                try:
                    for m in MODES:
                        row[f'e_{m}'], row[f'fmax_{m}'] = predict(calc, a, m)
                except Exception as exc:
                    row.update(status='failed', error=repr(exc)[:300])
                row['runtime_s'] = round(time.time() - t, 4)
                fh.write(json.dumps(row) + '\n')
                fh.flush()
                if i % 200 == 0:
                    print(f'{name} {i}/{len(frames)} {row["runtime_s"]}s', flush=True)
        info['models'][name]['runtime_s'] = round(time.time() - t_model, 1)
        del calc
        torch.cuda.empty_cache()
    json.dump(info, open(out / 'info.json', 'w'), indent=1)
    print('eval finished')


if __name__ == '__main__':
    main()
