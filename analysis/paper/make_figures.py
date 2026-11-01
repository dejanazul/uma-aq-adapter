import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D, OUT = 'results/adapter/final/', 'results/figures/'
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update(
    {
        'font.size': 8,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
    }
)


def boot_mae(d, col, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in d.groupby('bulk')]
    stats = [
        np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]).mean() for _ in range(draws)
    ]
    return d[col].mean(), np.quantile(stats, [0.025, 0.975])


def fig_strata(rel):
    strata = ['S0', 'S1', 'S2', 'S3']
    labels = ['S0\nspin-off', 'S1\n$\\leq$0.14 $\\mu_B$', 'S2\n$\\leq$1.09 $\\mu_B$', 'S3\n>1.09 $\\mu_B$']
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    rows = []
    for ax, split in zip(axes, ('test', 'val')):
        for col, color, marker, name, dx in (
            ('abs_zs', BLUE, 'o', 'UMA zero-shot', -0.12),
            ('abs_ft', ORANGE, 's', 'UMA-AQ adapter', 0.12),
        ):
            for i, s in enumerate(strata):
                d = rel[(rel.split == split) & (rel.stratum == s)]
                m, (lo, hi) = boot_mae(d, col)
                rows.append(dict(split=split, stratum=s, model=name, n=len(d), mae=m, lo=lo, hi=hi))
                ax.errorbar(
                    i + dx,
                    m,
                    yerr=[[m - lo], [hi - m]],
                    fmt=marker,
                    ms=5,
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=0,
                    mec='white',
                    mew=0.8,
                    label=name if i == 0 else None,
                )
        ax.axhline(0.1, color=MUTED, lw=0.8, ls='--')
        ax.set_xticks(range(4), labels)
        ax.set_title(f'OOD-Both {split}', color=INK, fontsize=8.5)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('$E_\\mathrm{ads}$ MAE (eV)')
    axes[0].text(3.45, 0.105, '0.1 eV', color=MUTED, fontsize=7, ha='right', va='bottom')
    axes[1].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT + 'fig_strata.pdf')
    return pd.DataFrame(rows)


def fig_forgetting(summary):
    names = {
        'oc22_val': 'OC22',
        'oc25_val': 'OC25',
        'odac25_val_filtered': 'ODAC25',
        'omat24_val': 'OMat24',
        'omc25_val': 'OMC25',
        'omol25_val': 'OMol25',
    }
    per = {names[d['dataset']]: d for d in summary['all_tasks']['per_dataset']}
    order = ['OC20'] + list(names.values())
    hybrid = [summary['oc20']['max_abs_diff_final_vs_uma_eV']] + [per[k]['max_abs_dE_l2h_eV'] for k in order[1:]]
    trained = [np.nan] + [per[k]['max_abs_dE_l2_eV'] for k in order[1:]]
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    x = np.arange(len(order))
    ax.scatter(
        x,
        trained,
        marker='s',
        s=28,
        color=BLUE,
        edgecolor='white',
        lw=0.8,
        zorder=3,
        label='final run as trained (non-EMA backbone)',
    )
    ax.scatter(
        x,
        hybrid,
        marker='o',
        s=28,
        color=ORANGE,
        edgecolor='white',
        lw=0.8,
        zorder=3,
        label='UMA-AQ adapter (released EMA backbone)',
    )
    ax.axhline(0.1, color=MUTED, lw=0.8, ls='--')
    ax.text(0.5, 0.125, '0.1 eV', color=MUTED, fontsize=7, ha='center', va='bottom')
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    ax.set_xticks(x, order)
    ax.set_ylabel('max $|E - E_\\mathrm{UMA}|$\nper structure (eV)')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    leg = ax.legend(
        frameon=True,
        loc='lower left',
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        fancybox=False,
        borderpad=0.5,
        handletextpad=0.4,
    )
    leg.get_frame().set(facecolor='#f1f0ec', edgecolor=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT + 'fig_forgetting.pdf')
    return pd.DataFrame(dict(dataset=order, hybrid=hybrid, trained=trained))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rel = pd.read_csv(D + 'aq_final_relaxations.csv')
    t = fig_strata(rel)
    s3 = t[(t.split == 'test') & (t.stratum == 'S3')].set_index('model').mae
    assert abs(s3['UMA zero-shot'] - 0.4575) < 1e-3 and abs(s3['UMA-AQ adapter'] - 0.2594) < 1e-3, s3
    print(t.round(4).to_string(index=False))
    print(fig_forgetting(json.load(open(D + 'final_summary.json'))).to_string(index=False))
