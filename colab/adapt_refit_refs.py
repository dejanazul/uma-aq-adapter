import glob
import json
import sys

import numpy as np

SPLITS = ('aq_train_low', 'aq_train_high', 'aq_train_slab')


def oc20_refs(ck):
    t = next(t for t in ck.tasks_config if t['name'] == 'oc20_energy')
    return np.array(t['element_references']['element_references']['_args_'][0], float)


def predict(ckpt, data_dir, stride, shard, nshard, out):
    from ase.db import connect
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

    calc = FAIRChemCalculator(load_predict_unit(ckpt, inference_settings='traineval', device='cuda'), task_name='oc20')
    stride, shard, nshard = int(stride), int(shard), int(nshard)
    comp, dft, pred, i = [], [], [], 0
    for split in SPLITS:
        for path in sorted(glob.glob(f'{data_dir}/{split}/*.aselmdb')):
            db = connect(path, type='aselmdb', readonly=True)
            for rid in range(1, db.count() + 1):
                take = i % stride == 0 and (i // stride) % nshard == shard
                i += 1
                if not take:
                    continue
                a = db.get(rid).toatoms()
                dft.append(a.get_potential_energy())
                a.calc = calc
                pred.append(a.get_potential_energy())
                comp.append(np.bincount(a.numbers, minlength=100))
                if len(dft) % 500 == 0:
                    print(shard, len(dft), flush=True)
    np.savez(out, comp=np.array(comp, float), dft=np.array(dft), pred=np.array(pred))
    print('done', shard, len(dft), flush=True)


def refit(comp, dft, pred, refs_oc20):
    c, *_ = np.linalg.lstsq(comp, dft - pred, rcond=None)
    return refs_oc20 + c


def fit(ckpt, out_refs, out_ckpt, *shards):
    import torch

    z = [np.load(s) for s in shards]
    comp, dft, pred = (np.concatenate([x[k] for x in z]) for k in ('comp', 'dft', 'pred'))
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    ref0 = oc20_refs(ck)
    refs = refit(comp, dft, pred, ref0)
    err = pred + comp @ (refs - ref0) - dft
    print(
        json.dumps(
            dict(
                n=len(dft),
                raw_mae_oc20_eV=float(np.abs(pred - dft).mean()),
                fit_mae_eV=float(np.abs(err).mean()),
                elements_seen=int((comp.sum(0) > 0).sum()),
            )
        )
    )
    t = next(t for t in ck.tasks_config if t['name'] == 'aqcat25_energy')
    t['element_references']['element_references']['_args_'] = [refs.tolist()]
    json.dump(refs.tolist(), open(out_refs, 'w'))
    torch.save(ck, out_ckpt)
    print('wrote', out_refs, out_ckpt)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    c = rng.integers(0, 5, (300, 100)).astype(float)
    r0, true = rng.normal(size=100), rng.normal(size=100)
    p = rng.normal(size=300)
    assert np.allclose(p + c @ (refit(c, p + c @ true, p, r0) - r0), p + c @ true)
    {'predict': predict, 'fit': fit}[sys.argv[1]](*sys.argv[2:])
