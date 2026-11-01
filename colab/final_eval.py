import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from ase.calculators.calculator import all_changes
from ase.io import iread

R2 = '/content/ckpt/uma-s-1p2-aqcat25-r2.pt'
L2 = glob.glob('/content/drive/MyDrive/aqcat25-lab-data/runs/lora2/*/checkpoints/final/inference_ckpt.pt')
ARCHIVE = '/content/drive/MyDrive/oc20-lab-archive/runs/oc20-ood-both-error-map'
FORGET = '/content/drive/MyDrive/aqcat25-lab-data/forgetting'
DATASETS = {
    'omat24_val': 'omat',
    'omol25_val': 'omol',
    'oc25_val': 'oc25',
    'odac25_val_filtered': 'odac',
    'omc25_val': 'omc',
}
N_FORGET = 300


def settings():
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    return InferenceSettings(execution_mode='general', merge_mole=False)


def l2_unit(hybrid=False):
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
    from fairchem.core.common.utils import load_state_dict, match_state_dict
    from adapt_lora import add_task_lora

    unit = MLIPPredictUnit(R2, device='cuda', inference_settings=settings())
    add_task_lora(unit.model.module, 'aqcat25', 8, 8.0)
    base = {k: v.detach().clone() for k, v in unit.model.state_dict().items() if 'lora_' not in k}
    ck = torch.load(L2[0], map_location='cpu', weights_only=False)
    r2_tasks = torch.load(R2, map_location='cpu', weights_only=False).tasks_config
    pick = lambda tasks, n: json.dumps(
        {k: next(t for t in tasks if t['name'] == n).get(k) for k in ('element_references', 'normalizer')},
        sort_keys=True,
        default=str,
    )
    for n in ('aqcat25_energy', 'oc20_energy'):
        assert pick(r2_tasks, n) == pick(ck.tasks_config, n), f'{n} refs/normalizer of L2 differ from r2'
    model_sd, ema = unit.model.state_dict(), dict(ck.ema_state_dict)
    n_avg = ema.pop('n_averaged')
    model_sd.pop('n_averaged')
    sd = match_state_dict(model_sd, ema)
    sd['n_averaged'] = n_avg
    load_state_dict(unit.model, sd, strict=True)
    n_kept = 0
    if hybrid:
        mapping = torch.load(R2, map_location='cpu', weights_only=False).model_config['heads']['energyandforcehead'][
            'dataset_mapping'
        ]
        slot = sorted(set(mapping.values())).index(mapping['aqcat25'])
        cur, new = unit.model.state_dict(), {}
        for k, v in base.items():
            if k == 'n_averaged' or k.endswith('dataset_emb_dict.aqcat25.weight'):
                continue
            if 'energy_block' in k and k.endswith('.weights'):
                t = v.clone()
                t[slot] = cur[k][slot]
                new[k] = t
                n_kept += 1
            else:
                new[k] = v
        unit.model.load_state_dict(new, strict=False)
    unit.model.eval()
    lora_norm = sum(float(v.norm()) for k, v in unit.model.state_dict().items() if 'lora_B' in k)
    assert lora_norm > 0, 'LoRA B is zero: adapters not loaded'
    return unit, dict(
        l2_ckpt=L2[0],
        hybrid=hybrid,
        head_tensors_slot_kept=n_kept,
        lora_B_norm=lora_norm,
        n_lora_tensors=sum('lora_' in k for k in sd),
    )


def uma_unit():
    from huggingface_hub import hf_hub_download
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    path = hf_hub_download('facebook/UMA', 'checkpoints/uma-s-1p2.pt', cache_dir='/content/hf_cache')
    return MLIPPredictUnit(path, device='cuda', inference_settings=settings())


def predict(unit, task, atoms):
    from fairchem.core import FAIRChemCalculator

    calc = FAIRChemCalculator(unit, task_name=task)
    a = atoms.copy()
    a.info = dict(atoms.info)
    t = time.time()
    calc.calculate(a, ['energy', 'forces'], all_changes)
    return float(calc.results['energy']), np.asarray(calc.results['forces']), time.time() - t


def free_mask(atoms):
    from ase.constraints import FixAtoms

    fixed = np.zeros(len(atoms), bool)
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            fixed[c.index] = True
    if not fixed.any() and (atoms.get_tags() == 0).any() and (atoms.get_tags() != 0).any():
        fixed = atoms.get_tags() == 0
    return ~fixed


def row(key, atoms, e_dft, f_dft, e, f, dt, **extra):
    m = free_mask(atoms)
    d = np.abs(f - f_dft)
    return dict(
        key=key,
        n_atoms=len(atoms),
        n_free=int(m.sum()),
        dft_total_eV=e_dft,
        pred_total_eV=e,
        total_err_eV=e - e_dft,
        force_mae_free=float(d[m].mean()) if m.any() else None,
        force_mae_all=float(d.mean()),
        runtime_s=round(dt, 3),
        status='ok',
        **extra,
    )


class Out:
    def __init__(self, path):
        self.path = Path(path)
        rows = [json.loads(l) for l in open(self.path)] if self.path.exists() else []
        self.done = {r['key'] for r in rows if r.get('status') == 'ok'}
        self.f = open(self.path, 'a')

    def write(self, r):
        self.f.write(json.dumps(r) + '\n')
        self.f.flush()


def labels(atoms):
    if 'dft_total_eV' in atoms.info:
        return float(atoms.info['dft_total_eV']), np.asarray(atoms.arrays['dft_forces'], dtype=float)
    return float(atoms.calc.results['energy']), np.asarray(atoms.calc.results['forces'])


def run(units, jobs, out):
    n = 0
    for key, atoms, task, extra in jobs:
        for name, unit in units.items():
            k = f'{name}|{key}'
            if k in out.done:
                continue
            try:
                e_dft, f_dft = labels(atoms)
                e, f, dt = predict(unit, task, atoms)
                out.write(row(k, atoms, e_dft, f_dft, e, f, dt, model=name, task=task, **extra))
            except Exception as ex:
                out.write(dict(key=k, model=name, task=task, status=f'error: {type(ex).__name__}: {ex}'[:300], **extra))
        n += 1
        if n % 200 == 0:
            print('rows', n, flush=True)


def aq_val_jobs():
    for p in sorted(glob.glob('/content/data/aq_val/*.aselmdb')):
        db = connect(p, type='aselmdb', readonly=True)
        for r in db.select():
            yield f'aq_val:{os.path.basename(p)}:{r.id}', r.toatoms(), 'aqcat25', {}


def stage_u_jobs(limit=None):
    for s in ('discovery', 'confirm'):
        for k, a in enumerate(iread(f'{ARCHIVE}/{s}.extxyz', index=':')):
            if limit is not None and k >= limit:
                return
            a.info['dft_total_eV'], a.arrays['dft_forces'] = (
                float(a.info['raw_total_eV']),
                np.asarray(a.arrays['REF_forces']),
            )
            yield (
                f'{s}:adslab:{a.info["system_id"]}',
                a,
                'oc20',
                dict(set=s, kind='adslab', id=str(a.info['system_id'])),
            )
    for a in iread('/content/drive/MyDrive/oc20-uma/slabs_final.extxyz', index=':'):
        a.info['dft_total_eV'], a.arrays['dft_forces'] = (
            float(a.info['dft_energy_eV']),
            a.get_forces(apply_constraint=False),
        )
        yield f'slab:{a.info["slab_id"]}', a, 'oc20', dict(set='slab', kind='slab', id=str(a.info['slab_id']))


def forget_jobs(available):
    rng = np.random.default_rng(0)
    for name, task in DATASETS.items():
        if task not in available:
            print('skip', name, 'task', task, 'not in checkpoint', flush=True)
            continue
        files = sorted(glob.glob(f'{FORGET}/{name}/**/*.aselmdb', recursive=True))
        dbs = [connect(p, type='aselmdb', readonly=True) for p in files]
        sizes = np.array([len(d) for d in dbs])
        pick = np.sort(rng.choice(sizes.sum(), size=min(N_FORGET, sizes.sum()), replace=False))
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        print(name, task, len(files), 'files', int(sizes.sum()), 'rows', flush=True)
        for g in pick:
            i = int(np.searchsorted(starts, g, side='right') - 1)
            atoms = dbs[i].get_atoms(int(g - starts[i]) + 1, add_additional_information=True)
            atoms.info.update(atoms.info.pop('data', {}) or {})
            yield f'{name}:{os.path.basename(files[i])}:{int(g - starts[i])}', atoms, task, dict(dataset=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('gate', 'gateh', 'aq', 'aqz', 'oc20', 'forget'))
    ap.add_argument('--out', default='/content/final')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = Out(f'{a.out}/{a.step}.jsonl')
    final = os.environ.get('FINAL_MODEL', 'l2h')
    if a.step == 'forget':
        units = {'uma': uma_unit(), 'l2': l2_unit()[0], 'l2h': l2_unit(hybrid=True)[0]}
        run(units, forget_jobs(set(units['uma'].dataset_to_tasks)), out)
        print('step done', a.step, flush=True)
        return
    l2, info = l2_unit(hybrid=(a.step == 'gateh' or (a.step != 'gate' and final == 'l2h')))
    json.dump(info, open(f'{a.out}/{a.step}_model_info.json', 'w'), indent=1)
    print('L2 loaded', info, flush=True)
    if a.step == 'gateh':
        run({'l2h': l2}, aq_val_jobs(), out)
        run({'l2h': l2, 'uma': uma_unit()}, stage_u_jobs(limit=100), out)
    elif a.step == 'gate':
        run({'l2': l2}, aq_val_jobs(), out)
        run({'l2': l2}, stage_u_jobs(limit=100), out)
    elif a.step == 'aq':
        jobs = (
            (
                f'{x.info["key"]}',
                x,
                'aqcat25',
                dict(kind=x.info['kind'], split=x.info['split'], slab_id=x.info['slab_id']),
            )
            for x in iread('/content/aq_final.extxyz', index=':')
        )
        run({'final': l2}, jobs, out)
    elif a.step == 'aqz':
        jobs = (
            (x.info['key'], x, 'aqcat25', dict(kind=x.info['kind']))
            for x in iread('/content/uma_zeroshot_frames.extxyz', index=':')
        )
        run({'final': l2}, jobs, out)
    elif a.step == 'oc20':
        run({'final': l2, 'uma': uma_unit()}, stage_u_jobs(), out)
    print('step done', a.step, flush=True)


if __name__ == '__main__':
    main()
