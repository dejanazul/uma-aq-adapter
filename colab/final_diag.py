import json

import numpy as np
from ase.io import iread

import final_eval as fe


def main():
    from huggingface_hub import hf_hub_download
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    frames = [a for _, a in zip(range(20), iread(f'{fe.ARCHIVE}/discovery.extxyz', index=':'))]
    path = hf_hub_download('facebook/UMA', 'checkpoints/uma-s-1p2.pt', cache_dir='/content/hf_cache')
    units = {
        'uma_default': load_predict_unit(path, device='cuda'),
        'uma_general': MLIPPredictUnit(path, device='cuda', inference_settings=fe.settings()),
        'r2_general': MLIPPredictUnit(fe.R2, device='cuda', inference_settings=fe.settings()),
    }
    units['l2_general'], _ = fe.l2_unit()
    print('uma_default settings:', units['uma_default'].inference_settings, flush=True)
    E = {n: [fe.predict(u, 'oc20', a)[0] for a in frames] for n, u in units.items()}
    bb = units['l2_general'].model.module.backbone
    hooks = list(bb._forward_pre_hooks.values())
    for h in hooks:
        for c in h.__closure__ or ():
            if c.cell_contents == 'aqcat25':
                c.cell_contents = '__none__'
    E['l2_gate_off'] = [fe.predict(units['l2_general'], 'oc20', a)[0] for a in frames]
    stage_u = {json.loads(l)['id']: json.loads(l)['pred_total_eV'] for l in open('/content/stageU_raw.jsonl')}
    E['stage_u_file'] = [stage_u[str(a.info['system_id'])] for a in frames]
    ref = np.array(E['uma_general'])
    for n, v in E.items():
        d = np.abs(np.array(v) - ref)
        print(f'{n:13s} vs uma_general: max {d.max():.2e} median {np.median(d):.2e} eV', flush=True)
    d = np.abs(np.array(E['uma_default']) - np.array(E['stage_u_file']))
    print(f'uma_default vs stage U file: max {d.max():.2e} median {np.median(d):.2e} eV')


if __name__ == '__main__':
    main()
