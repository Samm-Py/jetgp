"""
Comparison plots: Dense DEGP vs Sparse DEGP at various rho values.
Reads JSON results from data/ subdirectory; saves figures to figures/ subdirectory.

Dense results:  results_jetgp_{func}.json        (from jetgp_degp/jetgp_{func}.py)
Sparse results: results_sparse_degp_{func}.json  (from jetgp_degp/jetgp_sparse_{func}.py)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path
from collections import defaultdict

# ── Directories ──────────────────────────────────────────────────────────────
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')
FIG_DIR  = os.path.join(TESTS_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

data_dir = Path(DATA_DIR)
out_dir  = Path(FIG_DIR)

# ── Font / style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# ── File map ─────────────────────────────────────────────────────────────────
dense_files = {
    'borehole':    'results_jetgp_borehole.json',
    'otl_circuit': 'results_jetgp_otl_circuit.json',
    'morris':      'results_jetgp_morris.json',
}
sparse_files = {
    'borehole':    'results_sparse_degp_borehole.json',
    'otl_circuit': 'results_sparse_degp_otl_circuit.json',
    'morris':      'results_sparse_degp_morris.json',
}
func_meta = {
    'otl_circuit': {'dim': 6,  'label': 'OTL Circuit (6D)'},
    'borehole':    {'dim': 8,  'label': 'Borehole (8D)'},
    'morris':      {'dim': 20, 'label': 'Morris (20D)'},
}

RHO_VALUES = [1.0, 3.0]

COLORS = {
    'Dense DEGP':          '#1D9E75',
    'Sparse ($\\rho=1$)':  '#2171B5',
    'Sparse ($\\rho=3$)':  '#D85A30',
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def load(fname):
    path = data_dir / fname
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def agg(records, metric):
    d = defaultdict(list)
    for r in records:
        d[r['n_train']].append(r[metric])
    return {k: (np.mean(v), np.std(v)) for k, v in sorted(d.items())}


def agg_total(records):
    d = defaultdict(list)
    for r in records:
        d[r['n_train']].append(r['train_time'] + r['pred_time'])
    return {k: (np.mean(v), np.std(v)) for k, v in sorted(d.items())}


def build_summary(func, metric):
    dim = func_meta[func]['dim']
    n_sizes = [dim, 5 * dim, 10 * dim]
    summary = {n: {} for n in n_sizes}

    get_agg = agg_total if metric == 'total_time' else lambda recs: agg(recs, metric)

    # Dense DEGP
    dense_recs = load(dense_files[func])
    if dense_recs:
        for n, v in get_agg(dense_recs).items():
            if n in summary:
                summary[n]['Dense DEGP'] = v

    # Sparse DEGP at each rho
    sparse_recs = load(sparse_files[func])
    for rho in RHO_VALUES:
        label = f'Sparse ($\\rho={int(rho) if rho == int(rho) else rho}$)'
        subset = [r for r in sparse_recs if r.get('rho') == rho]
        if subset:
            for n, v in get_agg(subset).items():
                if n in summary:
                    summary[n][label] = v

    return summary, n_sizes


def log_tick_formatter(val, pos):
    exp = int(round(val))
    return r'$10^{' + str(exp) + r'}$'


# ── Main plot function ───────────────────────────────────────────────────────
def make_grid(metric, ylabel, fname, invert=True):
    funcs = ['otl_circuit', 'borehole', 'morris']
    col_labels = [r'$n = d$', r'$n = 5d$', r'$n = 10d$']

    fig, axes = plt.subplots(len(funcs), 3, figsize=(15, 5 * len(funcs)))
    fig.patch.set_facecolor('white')

    for row, func in enumerate(funcs):
        summary, n_sizes = build_summary(func, metric)

        for col, n in enumerate(n_sizes):
            ax = axes[row][col]
            ax.set_facecolor('#F8F8F6')
            data = summary.get(n, {})

            if row == 0:
                ax.set_title(col_labels[col] + f' ({n})', fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel(func_meta[func]['label'], fontsize=12,
                              labelpad=10)

            if not data:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=11)
                ax.spines[['top', 'right']].set_visible(False)
                continue

            labels = list(data.keys())
            means  = [data[l][0] for l in labels]
            stds   = [data[l][1] for l in labels]
            colors = [COLORS.get(l, '#888780') for l in labels]
            y_pos  = np.arange(len(labels))

            log_means = np.log10(np.maximum(means, 1e-9))
            lower = np.log10(np.maximum(np.array(means) - np.array(stds),
                                        1e-9))
            upper = np.log10(np.array(means) + np.array(stds))
            xerr  = [log_means - lower, upper - log_means]

            ax.barh(y_pos, log_means, xerr=xerr,
                    color=colors, edgecolor='white', linewidth=0.5,
                    height=0.6,
                    error_kw=dict(ecolor='#444441', elinewidth=1.0,
                                  capsize=3, capthick=1.0))

            lo = int(np.floor(min(log_means) - 0.5))
            hi = int(np.ceil(max(log_means) + 0.5))
            ax.set_xticks(list(range(lo, hi + 1)))
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(log_tick_formatter))

            if invert:
                ax.invert_xaxis()

            if row == len(funcs) - 1:
                ax.set_xlabel(f'{ylabel} (log scale)', fontsize=12)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=11)
            ax.tick_params(axis='x', labelsize=11)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#CCCCCC')
            ax.grid(axis='x', color='#E0E0E0', linewidth=0.5)
            ax.set_axisbelow(True)

    fig.suptitle(
        f'{ylabel} (log scale) \u2014 Dense vs Sparse DEGP '
        f'(mean $\\pm$ std over 5 macroreplicates)',
        fontsize=13, y=1.01)
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out}')


# ── Run ──────────────────────────────────────────────────────────────────────
make_grid('nrmse',      'NRMSE',
          'sparse_degp_nrmse_grid.pdf',      invert=True)
make_grid('train_time', 'Train time (s)',
          'sparse_degp_time_grid.pdf',       invert=True)
make_grid('total_time', 'Train + prediction time (s)',
          'sparse_degp_total_time_grid.pdf', invert=True)
print('All done.')
