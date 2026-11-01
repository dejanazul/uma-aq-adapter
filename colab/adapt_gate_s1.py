import glob
import json
import sys

import numpy as np
from ase.db import connect


def main(ckpt, val_dir, out, n=1000):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

    calc = FAIRChemCalculator(load_predict_unit(ckpt, inference_settings='traineval', device='cpu'), task_name='oc20')
    rows = []
    for path in sorted(glob.glob(f'{val_dir}/*.aselmdb')):
        for r in connect(path, type='aselmdb', readonly=True).select():
            a = r.toatoms()
            dft = a.get_potential_energy()
            a.calc = calc
            rows.append((len(a), a.get_potential_energy() - dft))
            if len(rows) % 100 == 0:
                print(len(rows), flush=True)
            if len(rows) >= int(n):
                break
        if len(rows) >= int(n):
            break
    nat, err = np.array(rows).T
    res = dict(
        n=len(rows),
        mae_eV=float(np.abs(err).mean()),
        mae_per_atom_meV=float(1000 * np.abs(err / nat).mean()),
        mean_err_eV=float(err.mean()),
        max_abs_err_eV=float(np.abs(err).max()),
    )
    res['passed'] = res['mae_per_atom_meV'] < 10
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps(res))
    assert res['passed'], 'gate S1 failed'


if __name__ == '__main__':
    main(*sys.argv[1:])
