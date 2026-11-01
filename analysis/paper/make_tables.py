import ast
import json
import re

import numpy as np
import pandas as pd

A, M = 'results/adapter/', 'results/ev2_film/'
DRAWS = 2000
AQ_E, OC_E, OC_F = 'aqcat25.val,aqcat25_energy,mae', 'oc20.val,oc20_energy,mae', 'oc20.val,oc20_forces,mae'
CLASSES = ('metal', 'metalloid', 'metalloid+non-metal', 'non-metal')
md = lambda t: print(t.to_markdown(index=False), end='\n\n')


def evals(path):
    out, step = {}, 0
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("INFO:root:{'train/loss'"):
            step = int(ast.literal_eval(line[len('INFO:root:') :])['train/step']) + 10
        elif 'Finished aggregating metrics' in line:
            v = {}
            for l in lines[i + 1 :]:
                m = re.match(r'\s+val/(\S+): ([-\d.eE+]+)$', l)
                if not m:
                    break
                v[m[1]] = float(m[2])
            out[step] = v
    return out


def params(path, pattern):
    return next(int(m[1]) for m in re.finditer(pattern, open(path).read()))


def variants():
    runb, arm_a, arm_b = evals(A + 'runB/runB.log'), evals(A + 'u1/u1_lr.log'), evals(A + 'u1/u1_replay.log')
    o1, l1, l2 = evals(A + 'o1/opt1.log'), evals(A + 'l1/lora1.log'), evals(A + 'l2/lora2.log')
    base_e = json.load(open(A + 'offset_check.json'))['oc20_val']['mae_eV']
    base_f = evals(A + 'bench/bench8r2.log')[300][OC_F]
    aq0 = json.load(open(A + 'offset_check_r2.json'))['aq_val']['mae_eV']
    ratio = lambda arm, key, base: (arm[4000][key] - base) / (runb[4000][key] - base)
    r = [ratio(arm, OC_E, base_e) for arm in (arm_a, arm_b)] + [ratio(arm, OC_F, base_f) for arm in (arm_a, arm_b)]
    n_task = params(A + 'o1/opt1.log', r'task-only fine-tuning on aqcat25 \(head slot 0\): (\d+) trainable')
    n_lora = params(A + 'l1/lora1.log', r'task-gated LoRA on aqcat25: rank 8, alpha 8.0, (\d+) adapter')
    rows = [
        ('No adaptation', '---', f'{o1[1000][OC_E]:.3f} eV', f'{aq0:.2f} eV', 'starting point'),
        (
            'Full fine-tuning',
            'all weights',
            f'{runb[6000][OC_E]:.2f} eV (6k)',
            f'{runb[6000][AQ_E]:.2f} eV (6k)',
            'stopped, forgetting',
        ),
        (
            'Full fine-tuning, lower learning rate or more OC20 replay',
            'all weights',
            f'forgetting {min(r):.0%}--{max(r):.0%} of the above',
            '---',
            'not sufficient',
        ),
        (
            'Task-specific parts only',
            f'task embedding + head slot ({n_task:,})',
            f'{o1[4000][OC_E]:.3f} eV',
            f'{o1[4000][AQ_E]:.2f} eV (4k)',
            'promising',
        ),
        (
            '+ task-gated LoRA, short run',
            f'+ rank-8 adapters ({n_task + n_lora:,} total)',
            f'{l1[4000][OC_E]:.3f} eV',
            f'{l1[4000][AQ_E]:.2f} eV (4k)',
            'selected',
        ),
        (
            '+ task-gated LoRA, final run',
            'same, 40k steps',
            f'{l2[40000][OC_E]:.3f} eV',
            f'{l2[40000][AQ_E]:.3f} eV (40k)',
            'final model',
        ),
    ]
    print('## Table: selection of the adaptation method\n')
    md(pd.DataFrame(rows, columns=['Variant', 'Trainable parts', 'OC20 val E MAE', 'AQCat25 val E MAE', 'Outcome']))
    unchanged = {k: {v[OC_E] for v in d.values()} for k, d in (('o1', o1), ('l1', l1), ('l2', l2))}
    print(f'OC20 val energy MAE over all evals: {unchanged}. Forgetting ratios R_E A/B, R_F A/B: {np.round(r, 2)}\n')


def film():
    rel = pd.read_csv(A + 'final/aq_final_relaxations.csv')
    f = pd.read_csv(M + 'film_relaxations.csv')[['relaxation', 'split', 'film_err_on_midfilm', 'film_err_off_midfilm']]
    d = rel.merge(f, on=['relaxation', 'split'], validate='1:1')
    d['abs_film'] = np.abs(np.where(d.spin_on, d.film_err_on_midfilm, d.film_err_off_midfilm))
    d['cls'] = np.select([d[f'class_{c}'] == 1 for c in CLASSES[1:]], list(CLASSES[1:]), 'metal')
    rows = []
    for s in ('test', 'val'):
        x = d[d.split == s]
        diff = (x.abs_ft - x.abs_film).to_numpy()
        rng = np.random.default_rng(0)
        lo, hi = np.percentile([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(DRAWS)], [2.5, 97.5])
        rows.append(
            (s, len(x), x.abs_film.mean(), x.abs_zs.mean(), x.abs_ft.mean(), f'{diff.mean():.3f} [{lo:.3f}, {hi:.3f}]')
        )
    print('## Table: E_ads MAE (eV) on identical OOD-Both final frames\n')
    md(
        pd.DataFrame(
            rows,
            columns=[
                'split',
                'n',
                'EV2-in+midFiLM',
                'UMA-S-1p2 zero-shot',
                'UMA-AQ adapter',
                'ours - EV2-FiLM [95% CI, bootstrap over relaxations]',
            ],
        ).round(3)
    )
    t = (
        d[d.split == 'test']
        .groupby('cls')
        .agg(n=('abs_ft', 'size'), adapter=('abs_ft', 'mean'), ev2_film=('abs_film', 'mean'))
    )
    print('## Table: E_ads MAE (eV) by material class, test split\n')
    md(t.reindex(list(CLASSES)).reset_index().round(3))


if __name__ == '__main__':
    variants()
    film()
