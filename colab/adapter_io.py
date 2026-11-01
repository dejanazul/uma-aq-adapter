import argparse
import hashlib
import json

import torch

from adapt_lora import add_task_lora

TASK, RANK, ALPHA = 'aqcat25', 8, 8.0


def head_slot(model_config, task=TASK):
    mapping = model_config['heads']['energyandforcehead']['dataset_mapping']
    return sorted(set(mapping.values())).index(mapping[task])


def is_adapter_key(k, task=TASK):
    return (
        'lora_' in k
        or k.endswith(f'dataset_emb_dict.{task}.weight')
        or ('energy_block' in k and k.endswith('.weights'))
    )


def adapter_state(state_dict, slot):
    return {
        k: (v[slot] if k.endswith('.weights') and 'energy_block' in k else v).detach().cpu().clone()
        for k, v in state_dict.items()
        if is_adapter_key(k)
    }


def task_refs(tasks_config, task=TASK):
    t = next(t for t in tasks_config if t['name'] == f'{task}_energy')
    return [float(x) for x in t['element_references']['element_references']['_args_'][0]]


def export(unit, base_ckpt, out_pt, out_refs, meta):
    ck = torch.load(base_ckpt, map_location='cpu', weights_only=False)
    slot = head_slot(ck.model_config)
    tensors = adapter_state(unit.model.state_dict(), slot)
    n = sum(v.numel() for v in tensors.values())
    meta = dict(meta, task=TASK, rank=RANK, alpha=ALPHA, head_slot=slot, n_parameters=n, n_tensors=len(tensors))
    torch.save(dict(tensors=tensors, meta=json.dumps(meta, sort_keys=True)), out_pt)
    json.dump(task_refs(ck.tasks_config), open(out_refs, 'w'))
    return meta


def load_unit(base_ckpt, adapter_pt, device='cpu', inference_settings=None):
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    settings = inference_settings or InferenceSettings(execution_mode='general', merge_mole=False)
    assert not settings.merge_mole, 'merge_mole=True folds the MoLE layers and would bypass the adapters'
    saved = torch.load(adapter_pt, map_location='cpu', weights_only=True)
    meta = json.loads(saved['meta'])
    unit = MLIPPredictUnit(base_ckpt, device=device, inference_settings=settings)
    add_task_lora(unit.model.module, meta['task'], meta['rank'], meta['alpha'])
    sd = unit.model.state_dict()
    missing = [k for k in sd if is_adapter_key(k, meta['task']) and k not in saved['tensors']]
    assert not missing and set(saved['tensors']) <= set(sd), ('adapter does not match the model', missing[:3])
    for k, v in saved['tensors'].items():
        v = v.to(sd[k].device, sd[k].dtype)
        if 'energy_block' in k and k.endswith('.weights'):
            sd[k][meta['head_slot']] = v
        else:
            sd[k] = v
    unit.model.load_state_dict(sd, strict=True)
    unit.model.eval()
    return unit


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('base_ckpt')
    ap.add_argument('adapter_pt')
    ap.add_argument('extxyz')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    from ase.io import iread
    from fairchem.core import FAIRChemCalculator

    calc = FAIRChemCalculator(load_unit(args.base_ckpt, args.adapter_pt, args.device), task_name=TASK)
    for a in iread(args.extxyz):
        a.calc = calc
        print(a.info.get('key', ''), a.get_potential_energy())
