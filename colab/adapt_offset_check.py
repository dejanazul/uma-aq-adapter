import glob
import json
import sys

import numpy as np
from ase.db import connect


def frames(d):
    for path in sorted(glob.glob(f'{d}/*.aselmdb')):
        for r in connect(path, type='aselmdb', readonly=True).select():
            yield r.toatoms()


def errors(calc, d):
    comp, err, nat = [], [], []
    for a in frames(d):
        dft = a.get_potential_energy()
        a.calc = calc
        err.append(a.get_potential_energy() - dft)
        comp.append(np.bincount(a.numbers, minlength=100))
        nat.append(len(a))
    return np.array(comp, float), np.array(err), np.array(nat)


def offset_test(comp, err, nat):
    c, *_ = np.linalg.lstsq(comp[::2], err[::2], rcond=None)
    res = err[1::2] - comp[1::2] @ c
    return dict(
        n=len(err),
        mae_eV=float(np.abs(err).mean()),
        mean_err_per_atom_eV=float((err / nat).mean()),
        std_err_per_atom_eV=float((err / nat).std()),
        heldout_raw_mae_eV=float(np.abs(err[1::2]).mean()),
        heldout_residual_mae_eV=float(np.abs(res).mean()),
    )


def main(ckpt, data_dir, out, pairs='aqcat25:aq_val,oc20:oc20_val'):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

    pu = load_predict_unit(ckpt, inference_settings='traineval', device='cuda')
    res = {}
    for task, split in (p.split(':') for p in pairs.split(',')):
        comp, err, nat = errors(FAIRChemCalculator(pu, task_name=task), f'{data_dir}/{split}')
        res[split] = offset_test(comp, err, nat)
        print(split, json.dumps(res[split]), flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    comp = rng.integers(0, 5, (40, 100)).astype(float)
    t = offset_test(comp, comp @ rng.normal(size=100) * 0, comp.sum(1))
    assert t['heldout_residual_mae_eV'] < 1e-9
    comp = rng.integers(0, 5, (400, 100)).astype(float)
    t = offset_test(comp, comp @ rng.normal(size=100), comp.sum(1))
    assert t['heldout_residual_mae_eV'] < 1e-6 < t['heldout_raw_mae_eV']
    main(*sys.argv[1:])
