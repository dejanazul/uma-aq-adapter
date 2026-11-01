import json
import sys

import torch
from omegaconf import OmegaConf


def main(template, out, ckpt, aq_refs, *overrides):
    cfg = OmegaConf.load(template)
    for k in ('aq_common', 'oc20_common', 'energy_task', 'force_task'):
        cfg.pop(k)
    tasks = torch.load(ckpt, map_location='cpu', weights_only=False).tasks_config
    oc20 = next(t for t in tasks if t['name'] == 'oc20_energy')['element_references']['element_references']['_args_'][0]
    cfg.ckpt, cfg.aq_elem_refs, cfg.oc20_elem_refs = ckpt, json.load(open(aq_refs)), list(map(float, oc20))
    for kv in overrides:
        k, v = kv.split('=', 1)
        OmegaConf.update(
            cfg,
            k,
            OmegaConf.create({'v': v}).v
            if not v.replace('.', '', 1).replace('-', '', 1).isdigit()
            else (float(v) if '.' in v else int(v)),
        )
    OmegaConf.save(cfg, out)
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
