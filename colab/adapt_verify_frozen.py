import json
import sys

import torch


def compare(start, final, emb_key, slot):
    start = {k.removeprefix('module.'): v for k, v in start.items()}
    final = {k.removeprefix('module.'): v for k, v in final.items() if k != 'n_averaged'}
    extra = set(final) - set(start)
    assert set(start) <= set(final) and all('lora_' in k for k in extra), set(start) ^ set(final)
    final = {k: v for k, v in final.items() if k in start}
    changed, bad = [], []
    for k, a in start.items():
        b = final[k].to(a.dtype)
        if torch.equal(a, b):
            continue
        changed.append(k)
        if k == emb_key:
            continue
        if k.startswith('output_heads') and k.endswith('.weights'):
            others = [i for i in range(a.shape[0]) if i != slot]
            if all(torch.equal(a[i], b[i]) for i in others):
                continue
        bad.append(k)
    return changed, bad


def main(start_path, final_path, task='aqcat25'):
    s = torch.load(start_path, map_location='cpu', weights_only=False)
    f = torch.load(final_path, map_location='cpu', weights_only=False)
    mapping = s.model_config['heads']['energyandforcehead']['dataset_mapping']
    slot = sorted(set(mapping.values())).index(mapping[task])
    emb_key = f'backbone.dataset_embedding.dataset_emb_dict.{task}.weight'
    res = {}
    for name in ('model_state_dict', 'ema_state_dict'):
        if getattr(f, name, None) is None:
            continue
        changed, bad = compare(s.model_state_dict, getattr(f, name), emb_key, slot)
        res[name] = dict(n_changed=len(changed), changed=changed, bad=bad)
    res['passed'] = all(not r['bad'] for r in res.values()) and bool(res)
    print(json.dumps(res, indent=1))
    assert res['passed'], 'V1 failed: shared parameters changed'


if __name__ == '__main__':
    assert compare({'w': torch.zeros(1)}, {'w': torch.zeros(1), 'n_averaged': torch.tensor(5)}, 'x', 0) == ([], [])
    assert compare({'w': torch.zeros(1)}, {'w': torch.zeros(1), 'x.lora_A': torch.ones(1)}, 'e', 0) == ([], [])
    a = {
        'w': torch.zeros(3, 2),
        'emb': torch.zeros(1, 2),
        'output_heads.x.weights': torch.zeros(3, 2, 2),
        'shared': torch.ones(2),
    }
    b = {k: v.clone() for k, v in a.items()}
    b['emb'] += 1
    b['output_heads.x.weights'][1] += 1
    assert compare(a, b, 'emb', 1) == (['emb', 'output_heads.x.weights'], [])
    b['shared'] += 1
    b['output_heads.x.weights'][0] += 1
    assert compare(a, b, 'emb', 1)[1] == ['output_heads.x.weights', 'shared']
    main(*sys.argv[1:])
