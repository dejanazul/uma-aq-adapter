import argparse
import copy
import json

import numpy as np
import torch

NEW, SRC = 'aqcat25', 'oc20'
HEAD_KEYS = [f'output_heads.energyandforcehead.head.energy_block.{i}.weights' for i in (0, 2, 4)]
EMB = 'backbone.dataset_embedding.dataset_emb_dict.{}.weight'


def surgery(ck, elem_refs=None):
    bb, head = ck.model_config['backbone'], ck.model_config['heads']['energyandforcehead']
    assert NEW not in bb['dataset_mapping'] and bb['dataset_mapping'] == head['dataset_mapping']
    old_targets = sorted(set(head['dataset_mapping'].values()))
    for mapping in (bb['dataset_mapping'], head['dataset_mapping']):
        mapping[NEW] = NEW
    new_targets = sorted(set(head['dataset_mapping'].values()))
    order = [old_targets.index(SRC if t == NEW else t) for t in new_targets]

    for sd, prefix in ((ck.model_state_dict, ''), (ck.ema_state_dict, 'module.')):
        src = prefix + EMB.format(SRC)
        if src in sd:
            sd[prefix + EMB.format(NEW)] = sd[src].clone()
        for key in HEAD_KEYS:
            if prefix + key in sd:
                w = sd[prefix + key]
                assert w.shape[0] == len(old_targets), (key, w.shape)
                sd[prefix + key] = w[order].clone()

    new_tasks = []
    for t in ck.tasks_config:
        if t.get('datasets') == [SRC]:
            n = copy.deepcopy(t)
            n['name'] = t['name'].replace(SRC, NEW)
            n['datasets'] = [NEW]
            if elem_refs is not None and 'element_references' in n:
                n['element_references']['element_references']['_args_'] = [list(map(float, elem_refs))]
            new_tasks.append(n)
    assert {t['name'] for t in new_tasks} == {f'{NEW}_energy', f'{NEW}_forces'}, [t['name'] for t in new_tasks]
    ck.tasks_config = list(ck.tasks_config) + new_tasks
    return ck, order


def predict(path, task, frames):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

    calc = FAIRChemCalculator(load_predict_unit(path, inference_settings='default', device='cpu'), task_name=task)
    out = []
    for a in frames:
        a = a.copy()
        a.calc = calc
        out.append((a.get_potential_energy(), a.get_forces().copy()))
    return out


def check(orig, new, frames_path):
    from ase.io import read

    frames = read(frames_path, ':')
    rows = []
    pairs = [(t, t) for t in ('oc20', 'oc22', 'oc25', 'omat', 'odac', 'omc')] + [(SRC, NEW)]
    for t_orig, t_new in pairs:
        a, b = predict(orig, t_orig, frames), predict(new, t_new, frames)
        de = max(abs(x[0] - y[0]) for x, y in zip(a, b))
        df = max(float(np.abs(x[1] - y[1]).max()) for x, y in zip(a, b))
        rows.append(dict(orig=t_orig, new=t_new, max_dE=de, max_dF=df, ok=de < 1e-4 and df < 1e-4))
        print(rows[-1], flush=True)
    assert all(r['ok'] for r in rows), 'gate S0 failed'
    print('gate S0 passed')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--elem-refs')
    ap.add_argument('--check')
    args = ap.parse_args()
    ck = torch.load(args.src, map_location='cpu', weights_only=False)
    refs = json.load(open(args.elem_refs)) if args.elem_refs else None
    ck, order = surgery(ck, refs)
    print('expert order (new slot -> old slot):', order)
    torch.save(ck, args.out)
    if args.check:
        assert refs is None, 'gate S0 needs the copied oc20 element references'
        check(args.src, args.out, args.check)


if __name__ == '__main__':
    main()
