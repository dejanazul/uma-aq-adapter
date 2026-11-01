import json
import sys

import pandas as pd

TRAIN_E, TRAIN_F, TOL_REL, TOL_OC20 = 0.442, 0.0338, 0.02, 1e-3
STAGE_U = 'results/oc20_stage_u/nonmetal_debug/stageU_raw.jsonl'


def main(step='gate'):
    g = pd.read_json(f'results/adapter/final/{step}.jsonl', lines=True)
    if step == 'gateh':
        return hybrid(g)
    print('rows', len(g), 'status', g.status.value_counts().to_dict())
    g = g[g.status == 'ok']
    aq = g[g.key.str.startswith('l2|aq_val:')]
    e = aq.total_err_eV.abs().mean()
    f = (aq.force_mae_free * aq.n_free * 3).sum() / (aq.n_free * 3).sum()
    ok_a = len(aq) == 2117 and abs(e / TRAIN_E - 1) <= TOL_REL and abs(f / TRAIN_F - 1) <= TOL_REL
    print(
        f'G0a: n={len(aq)} energy MAE {e:.4f} eV (train {TRAIN_E}, {100 * (e / TRAIN_E - 1):+.2f} %), '
        f'force MAE {f:.5f} eV/A (train {TRAIN_F}, {100 * (f / TRAIN_F - 1):+.2f} %) -> {"PASS" if ok_a else "FAIL"}'
    )
    u = pd.read_json(STAGE_U, lines=True)
    u = u[u.status == 'ok'].set_index(['set', 'kind', 'id'])
    oc = g[g.task == 'oc20'].copy()
    oc['id'] = oc['id'].astype(str)
    oc = oc.join(u[['pred_total_eV', 'dft_total_eV']], on=['set', 'kind', 'id'], rsuffix='_uma')
    d = (oc.pred_total_eV - oc.pred_total_eV_uma).abs()
    label_same = (oc.dft_total_eV - oc.dft_total_eV_uma).abs().max()
    ok_b = len(oc) == 100 and d.notna().all() and d.max() <= TOL_OC20 and label_same < 1e-6
    print(
        f'G0b: n={len(oc)} max |E_L2 - E_UMA| {d.max():.2e} eV, median {d.median():.2e}; '
        f'labels identical (max diff {label_same:.1e}) -> {"PASS" if ok_b else "FAIL"}'
    )
    json.dump(
        dict(
            G0a=dict(n=len(aq), energy_mae=e, force_mae=f, passed=bool(ok_a)),
            G0b=dict(n=len(oc), max_abs_diff=float(d.max()), passed=bool(ok_b)),
        ),
        open('results/adapter/final/gate_summary.json', 'w'),
        indent=1,
    )
    print('G0', 'PASS' if ok_a and ok_b else 'FAIL')


def hybrid(g):
    print('rows', len(g), 'status', g.status.value_counts().to_dict())
    g = g[g.status == 'ok']
    aq = g[g.key.str.startswith('l2h|aq_val:')]
    e = aq.total_err_eV.abs().mean()
    f = (aq.force_mae_free * aq.n_free * 3).sum() / (aq.n_free * 3).sum()
    ok_c = len(aq) == 2117 and abs(e / TRAIN_E - 1) <= TOL_REL and abs(f / TRAIN_F - 1) <= TOL_REL
    print(
        f'G0c: n={len(aq)} energy MAE {e:.4f} eV ({100 * (e / TRAIN_E - 1):+.2f} %), force MAE {f:.5f} eV/A '
        f'({100 * (f / TRAIN_F - 1):+.2f} %) -> {"PASS" if ok_c else "FAIL"}'
    )
    oc = g[g.task == 'oc20'].assign(sk=lambda x: x.key.str.split('|', n=1).str[1])
    w = oc.pivot(index='sk', columns='model', values='pred_total_eV')
    d = (w['l2h'] - w['uma']).abs()
    ok_d = len(w) == 100 and d.notna().all() and d.max() <= TOL_OC20
    print(
        f'G0d: n={len(w)} max |E_l2h - E_UMA| {d.max():.2e} eV, median {d.median():.2e} -> {"PASS" if ok_d else "FAIL"}'
    )
    json.dump(
        dict(
            G0c=dict(n=len(aq), energy_mae=e, force_mae=f, passed=bool(ok_c)),
            G0d=dict(n=len(w), max_abs_diff=float(d.max()), passed=bool(ok_d)),
        ),
        open('results/adapter/final/gateh_summary.json', 'w'),
        indent=1,
    )
    print('final model:', 'l2h' if ok_c and ok_d else 'l2 (G0c/G0d failed)')


if __name__ == '__main__':
    main(*sys.argv[1:])
