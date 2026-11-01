import json
import math
import os
import subprocess
import sys

import numpy as np

DATA = '/content/data'
CKPT = os.environ.get('ADAPT_CKPT', '/content/ckpt/uma-s-1p2-aqcat25.pt')
REFS = os.environ.get('ADAPT_REFS', f'{DATA}/aq_elem_refs.json')
REPLAY = float(os.environ.get('ADAPT_REPLAY', '1'))
WARMUP = os.environ.get('ADAPT_WARMUP_STEPS')


def ratios(sizes, w, replay=1.0, samples=0):
    pool = sizes['aq_train_low'] + w * sizes['aq_train_high'] + sizes['aq_train_slab']
    if replay == 0:
        k = math.ceil(samples / pool)
        return dict(ratio_low=k, ratio_high=w * k, ratio_slab=k, ratio_oc20=0.0)
    k = sizes['oc20_train'] / (replay * (sizes['aq_train_low'] + w * sizes['aq_train_high'] + sizes['aq_train_slab']))
    return dict(ratio_low=k, ratio_high=w * k, ratio_slab=k, ratio_oc20=1.0)


def main(variant, name, steps, batch_size, *overrides):
    sizes = {
        d: len(np.load(f'{DATA}/{d}/metadata.npz')['natoms'])
        for d in ('aq_train_low', 'aq_train_high', 'aq_train_slab', 'oc20_train')
    }
    r = ratios(sizes, {'A': 1, 'B': 2}[variant], REPLAY, int(steps) * int(batch_size))
    n_train = sum(int(r[f'ratio_{s}'] * sizes[f'aq_train_{s}']) for s in ('low', 'high', 'slab')) + int(
        r['ratio_oc20'] * sizes['oc20_train']
    )
    if WARMUP:
        overrides += (
            f'runner.train_eval_unit.cosine_lr_scheduler_fn.warmup_epochs={int(WARMUP) / (n_train / int(batch_size)):.8f}',
        )
    run_dir, log = f'/content/runs/{name}', f'/content/{name}.log'
    os.makedirs(run_dir, exist_ok=True)
    json.dump(
        dict(
            variant=variant,
            steps=int(steps),
            batch_size=int(batch_size),
            ckpt=CKPT,
            refs=REFS,
            replay=REPLAY,
            sizes=sizes,
            **r,
            n_train=n_train,
            warmup_steps=WARMUP,
            overrides=list(overrides),
        ),
        open(f'{run_dir}/launch.json', 'w'),
        indent=1,
    )
    sys.path.insert(0, '/content')
    from adapt_make_config import main as make_config

    make_config(
        '/content/adapt_train.yaml',
        f'{run_dir}/run.yaml',
        CKPT,
        REFS,
        f'data_dir={DATA}',
        f'job.run_dir={run_dir}',
        f'job.run_name={name}',
        f'steps={steps}',
        f'batch_size={batch_size}',
        *[f'{k}={v}' for k, v in r.items()],
        *overrides,
    )
    if REPLAY == 0:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(f'{run_dir}/run.yaml')
        cfg.train_dataset.dataset_configs.pop('oc20')
        cfg.train_dataset.combined_dataset_config.sampling.ratios.pop('oc20.train')
        cfg.pop('ratio_oc20')
        OmegaConf.save(cfg, f'{run_dir}/run.yaml')
    open(log, 'w').close()
    p = subprocess.Popen(
        ['bash', '/content/adapt_train_job.sh', name, f'{run_dir}/run.yaml'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps(dict(pid=p.pid, log=log, run_dir=run_dir, sizes=sizes, **r)))


if __name__ == '__main__':
    assert (
        abs(ratios(dict(oc20_train=40, aq_train_low=10, aq_train_high=5, aq_train_slab=5), 2)['ratio_low'] - 1.6)
        < 1e-12
    )
    assert (
        abs(ratios(dict(oc20_train=40, aq_train_low=10, aq_train_high=5, aq_train_slab=5), 2, 3)['ratio_low'] - 1.6 / 3)
        < 1e-12
    )
    assert ratios(dict(oc20_train=40, aq_train_low=10, aq_train_high=5, aq_train_slab=5), 2, 0, 51) == dict(
        ratio_low=3, ratio_high=6, ratio_slab=3, ratio_oc20=0.0
    )
    main(*sys.argv[1:])
