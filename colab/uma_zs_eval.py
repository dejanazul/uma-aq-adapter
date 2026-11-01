import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import iread

FRAMES = '/content/uma_zeroshot_frames.extxyz'
OUT = Path('/content/uma_zs_raw.jsonl')
MODEL_FILE = 'checkpoints/uma-s-1p2.pt'


def load_uma():
    from huggingface_hub import hf_hub_download
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    import fairchem.core

    path = Path(hf_hub_download('facebook/UMA', MODEL_FILE, cache_dir='/content/hf_cache')).resolve()
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    unit = load_predict_unit(str(path), inference_settings='traineval', device='cuda')
    info = dict(
        fairchem_core=fairchem.core.__version__,
        checkpoint=MODEL_FILE,
        checkpoint_sha256=digest.hexdigest(),
        inference_settings='traineval',
    )
    return FAIRChemCalculator(unit, task_name='oc20'), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10**6)
    args = parser.parse_args()
    done = set()
    if OUT.exists():
        done = {r['key'] for r in map(json.loads, open(OUT)) if r['status'] == 'ok'}
    calc, info = load_uma()
    print(json.dumps({'status': 'model_loaded', 'already_done': len(done), **info}), flush=True)
    with open(OUT, 'a') as sink:
        for k, atoms in enumerate(iread(FRAMES, index=':')):
            if k >= args.limit:
                break
            key = atoms.info['key']
            if key in done:
                continue
            row = dict(key=key, n_atoms=len(atoms))
            try:
                dft_e, dft_f = atoms.get_potential_energy(), atoms.get_forces()
                free = atoms.get_tags() != 0
                atoms = atoms.copy()
                atoms.calc = calc
                start = time.perf_counter()
                energy = float(atoms.get_potential_energy())
                forces = atoms.get_forces()
                row['runtime_s'] = round(time.perf_counter() - start, 4)
                assert np.isfinite(energy) and np.isfinite(forces).all(), 'Nonfinite prediction'
                delta = np.abs(forces - dft_f)
                row.update(
                    status='ok',
                    dft_total_eV=float(dft_e),
                    pred_total_eV=energy,
                    n_free=int(free.sum()),
                    force_mae_free=float(delta[free].mean()) if free.any() else None,
                    force_mae_all=float(delta.mean()),
                    force_max_err=float(delta.max()),
                )
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
            sink.write(json.dumps(row, allow_nan=False) + '\n')
            sink.flush()
            if k % 200 == 0 or row['status'] != 'ok':
                print(json.dumps({x: row.get(x) for x in ('key', 'status', 'runtime_s', 'error')}), flush=True)
    print(json.dumps({'status': 'eval_done', **info}), flush=True)


if __name__ == '__main__':
    main()
