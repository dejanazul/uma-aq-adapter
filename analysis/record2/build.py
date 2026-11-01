import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'analysis/record2')
from labels import FILES, is_aqcat25, rewrite, strip

SHIP = (
    'results/zeroshot/uma_zeroshot_raw.jsonl',
    'results/failure_analysis/sprint2/raw_test/*',
    'results/failure_analysis/sprint2/raw_val/*',
    'results/ev2_film/film/rows.jsonl',
    'results/ev2_film/film/info.json',
    'results/adapter/final/aq.jsonl',
    'results/adapter/final/aqz.jsonl',
    'results/adapter/final/oc20.jsonl',
    'results/adapter/final/forget.jsonl',
    'results/adapter/final/gate.jsonl',
    'results/adapter/final/gateh.jsonl',
    'results/adapter/final/*_info.json',
    'results/adapter/final/final_pip_freeze.txt',
    'results/adapter/aq_val_frames.csv',
    'results/adapter/*.json',
    'results/adapter/runB/*',
    'results/adapter/u1/*',
    'results/adapter/o1/*',
    'results/adapter/l1/*',
    'results/adapter/l2/*',
    'results/adapter/bench/*',
    'results/oc20_stage_u/nonmetal_debug/stageU_raw.jsonl',
    'results/oc20_stage_u/nonmetal_debug/stageU_slab_map.json',
    'results/oc20_stage_u/nonmetal_debug/stageU_*_table.csv',
    'results/oc20_stage_u/ood_both_error_map/discovery_table.csv',
    'results/oc20_stage_u/ood_both_error_map/confirm_table.csv',
)
SKIP = ('wandb',)


def ship(out):
    n = 0
    for pattern in SHIP:
        paths = [p for p in sorted(glob.glob(pattern)) if os.path.isfile(p) and not any(s in p for s in SKIP)]
        assert paths, f'nothing matches {pattern}'
        for p in paths:
            os.makedirs(os.path.join(out, os.path.dirname(p)), exist_ok=True)
            shutil.copy2(p, os.path.join(out, p))
            n += 1
    here = os.getcwd()
    os.chdir(out)
    for path in FILES:
        rewrite(path, strip)
        assert not any(
            is_aqcat25(d['key']) and d.get('dft_total_eV') is not None for d in map(json.loads, open(path))
        ), path
    os.chdir(here)
    return n


def predictions():
    rel = pd.read_csv('results/adapter/final/aq_final_relaxations.csv')
    film = pd.read_csv('results/ev2_film/film_relaxations.csv').set_index('key')
    spin_on = rel.key.map(film.spin_on).astype(bool)
    assert (spin_on == ~rel.is_spin_off.astype(bool)).all()
    return pd.DataFrame(
        dict(
            split=rel.split,
            relaxation=rel.relaxation,
            sid=rel.sid,
            frame_id=rel.frame_id,
            slab_id=rel.slab_id,
            bulk=rel.bulk,
            adsorbate=rel.adsorbate,
            is_spin_off=rel.is_spin_off,
            stratum=rel.stratum,
            adm_muB=rel.key.map(film.adm),
            eads_uma_zeroshot_eV=rel.eads_oc20,
            eads_uma_aq_adapter_eV=rel.adsorption_energy + rel.err_ft,
            eads_ev2_midfilm_eV=np.where(spin_on, rel.key.map(film.e_on_midfilm), rel.key.map(film.e_off_midfilm)),
            ev2_spin_mode=np.where(spin_on, 'on', 'off'),
        )
    ).sort_values(['split', 'relaxation'])


def original_tasks():
    rows = []
    for name in ('forget', 'oc20'):
        r = pd.read_json(f'results/adapter/final/{name}.jsonl', lines=True)
        r = r[r.status == 'ok'].drop_duplicates('key', keep='last')
        r[['model', 'k']] = r.key.str.split('|', n=1, expand=True)
        if name == 'oc20':
            r['dataset'], r['task'], r['model'] = 'oc20_ood_both_stage_u', 'oc20', r.model.replace('final', 'l2h')
        w = r.pivot_table(index=['dataset', 'task', 'k'], columns='model', values='pred_total_eV', aggfunc='first')
        w.columns = [f'pred_total_eV_{c}' for c in w.columns]
        dft = r.drop_duplicates('k').set_index(['dataset', 'task', 'k'])[['n_atoms', 'dft_total_eV']]
        rows.append(dft.join(w).reset_index().rename(columns={'k': 'structure'}))
    t = pd.concat(rows, ignore_index=True)
    assert t.columns.is_unique
    for m in ('l2h', 'l2'):
        if f'pred_total_eV_{m}' in t:
            t[f'dE_{m}_minus_uma_eV'] = t[f'pred_total_eV_{m}'] - t.pred_total_eV_uma
    return t


def splits(selection):
    s = pd.read_csv(selection)
    ids = s.drop(columns=['total_magnetization'])
    held = s[s.split == 'val'].groupby('bulk').size().rename('n_frames').reset_index()
    return ids, held


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checksums(out):
    paths = sorted(
        os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs if f != 'SHA256SUMS'
    )
    with open(os.path.join(out, 'SHA256SUMS'), 'w') as fh:
        for p in paths:
            fh.write(f'{sha256(os.path.join(out, p))}  {p}\n')
    return len(paths)


if __name__ == '__main__':
    out, selection, organic = sys.argv[1:4]
    n = ship(out)
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    predictions().to_csv(os.path.join(out, 'tables/aqcat25_ood_both_predictions.csv'), index=False)
    original_tasks().to_csv(os.path.join(out, 'tables/original_tasks_predictions.csv'), index=False)
    ids, held = splits(selection)
    ids.to_csv(os.path.join(out, 'tables/training_selection.csv'), index=False)
    held.to_csv(os.path.join(out, 'tables/heldout_bulks.csv'), index=False)
    shutil.copy2(organic, os.path.join(out, 'tables/removed_organic_frames.csv'))
    print(n, 'result files shipped,', checksums(out), 'files in SHA256SUMS')
