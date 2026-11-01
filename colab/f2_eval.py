import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from ase.io import iread

MODEL_FILE = 'checkpoints/uma-s-1p2.pt'
TASKS = ('oc20', 'omat')
CHUNK = 500


def load_uma():
    from huggingface_hub import hf_hub_download
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    import fairchem.core

    path = Path(hf_hub_download('facebook/UMA', MODEL_FILE, cache_dir='/content/hf_cache')).resolve()
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            md5.update(block)
            sha.update(block)
    unit = load_predict_unit(str(path), inference_settings='traineval', device='cuda')
    info = dict(
        fairchem_core=fairchem.core.__version__,
        checkpoint=MODEL_FILE,
        checkpoint_md5=md5.hexdigest(),
        checkpoint_sha256=sha.hexdigest(),
        inference_settings='traineval',
    )
    return {t: FAIRChemCalculator(unit, task_name=t) for t in TASKS}, info


def relax_gas(calcs, out):
    from ase.build import molecule
    from ase.optimize import BFGS

    rows = []
    for task, calc in calcs.items():
        for name in ('CO', 'H2', 'H2O', 'N2'):
            atoms = molecule(name, cell=[20, 20, 20], pbc=True)
            atoms.center()
            atoms.calc = calc
            opt = BFGS(atoms, logfile=None)
            converged = bool(opt.run(fmax=0.01, steps=500))
            d = atoms.get_all_distances(mic=True)
            rows.append(
                dict(
                    task=task,
                    molecule=name,
                    energy_eV=float(atoms.get_potential_energy()),
                    fmax=float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                    steps=opt.nsteps,
                    converged=converged,
                    min_bond_A=float(d[d > 0].min()),
                    positions=atoms.positions.tolist(),
                )
            )
    (out / 'gas.json').write_text(json.dumps(rows, indent=1))
    print(json.dumps({'status': 'gas_done', 'converged': all(r['converged'] for r in rows)}), flush=True)


def flush(buf, out, sink):
    if not buf:
        return
    n = len(list(out.glob('forces_*.npz')))
    np.savez(
        out / f'forces_{n:05d}.npz',
        keys=np.array([b[0]['key'] for b in buf]),
        n_atoms=np.array([b[0]['n_atoms'] for b in buf]),
        **{f'f_{t}': np.concatenate([b[1][t] for b in buf]).astype(np.float32) for t in TASKS},
    )
    for row, _ in buf:
        sink.write(json.dumps(row, allow_nan=False) + '\n')
    sink.flush()
    buf.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--gas', action='store_true')
    parser.add_argument('frames', nargs='+')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows_path = out / 'rows.jsonl'
    done = {json.loads(l)['key'] for l in open(rows_path)} if rows_path.exists() else set()
    calcs, info = load_uma()
    print(json.dumps({'status': 'model_loaded', 'already_done': len(done), **info}), flush=True)
    (out / 'info.json').write_text(json.dumps(info))
    if args.gas:
        relax_gas(calcs, out)
    buf, k = [], 0
    with open(rows_path, 'a') as sink:
        for frames in args.frames:
            for atoms in iread(frames, index=':'):
                key = atoms.info['key']
                if key in done:
                    continue
                k += 1
                row, forces = dict(key=key, n_atoms=len(atoms)), {}
                try:
                    dft_f = atoms.get_forces()
                    free = atoms.get_tags() != 0
                    row.update(dft_total_eV=float(atoms.get_potential_energy()), n_free=int(free.sum()))
                    for task, calc in calcs.items():
                        a = atoms.copy()
                        a.calc = calc
                        start = time.perf_counter()
                        energy, f = float(a.get_potential_energy()), a.get_forces()
                        assert np.isfinite(energy) and np.isfinite(f).all(), f'Nonfinite prediction ({task})'
                        assert f.shape == (len(atoms), 3), f'Force shape {f.shape} ({task})'
                        comp, vec = np.abs(f - dft_f), np.linalg.norm(f - dft_f, axis=1)
                        row.update(
                            {
                                f'runtime_s_{task}': round(time.perf_counter() - start, 4),
                                f'pred_total_eV_{task}': energy,
                                f'force_mae_free_{task}': float(comp[free].mean()) if free.any() else None,
                                f'force_vec_err_mean_free_{task}': float(vec[free].mean()) if free.any() else None,
                                f'force_vec_err_max_free_{task}': float(vec[free].max()) if free.any() else None,
                            }
                        )
                        forces[task] = f
                    row['status'] = 'ok'
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:1000])
                    forces = {t: np.full((len(atoms), 3), np.nan) for t in TASKS}
                buf.append((row, forces))
                if len(buf) >= CHUNK:
                    flush(buf, out, sink)
                if k % 1000 == 1 or row['status'] != 'ok':
                    print(json.dumps({x: row.get(x) for x in ('key', 'status', 'runtime_s_oc20', 'error')}), flush=True)
        flush(buf, out, sink)
    print(json.dumps({'status': 'eval_done', **info}), flush=True)


if __name__ == '__main__':
    main()
