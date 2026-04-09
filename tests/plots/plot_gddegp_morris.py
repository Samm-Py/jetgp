"""
Benchmark comparison plots for GDDEGP on Morris 20D.
Compares GDDEGP with 1, 2, 3 directions per point against GEKPLS and GPyTorch.
Reuses existing GEKPLS and GPyTorch Morris result files.

Reads JSON results from data/ subdirectory; saves figures to figures/ subdirectory.
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

# ── Font / style ──────────────────────────────────────────────────────────────
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

# ── File map ──────────────────────────────────────────────────────────────────
DIM          = 20
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
SHOW_ITERS   = [100, 200, 500, 1000]

degp_file = 'results_jetgp_morris.json'
gddegp_files = {
    k: f'results_jetgp_gddegp_{k}dirs_morris.json' for k in [1, 2, 3]
}
gekpls_file      = 'results_gekpls_morris.json'
gekpls_ndim_file = 'results_gekpls_ndim_morris.json'
gpt_files   = {it: f'results_gpytorch_morris_{it}iter.json' for it in SHOW_ITERS}

COLORS = {
    'DEGP':            '#1D9E75',
    'GDDEGP (1 dir)':  '#A8DDB5',
    'GDDEGP (2 dirs)': '#43A96D',
    'GDDEGP (3 dirs)': '#1D6B3A',
    'GEKPLS':          '#D85A30',
    'GEKPLS (ndim)':   '#E8944A',
    'GPyTorch (100)':  '#C6DCEF',
    'GPyTorch (200)':  '#9ECAE1',
    'GPyTorch (500)':  '#6BAED6',
    'GPyTorch (1000)': '#2171B5',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(fname):
    path = data_dir / fname
    if not path.exists():
        return None
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

def get_agg(records, metric):
    if records is None:
        return {}
    return agg_total(records) if metric == 'total_time' else agg(records, metric)

def build_summary(metric):
    summary = {n: {} for n in SAMPLE_SIZES}

    for n, v in get_agg(load(degp_file), metric).items():
        if n in summary: summary[n]['DEGP'] = v

    for k in [1, 2, 3]:
        label = f'GDDEGP ({k} dir{"s" if k > 1 else ""})'
        for n, v in get_agg(load(gddegp_files[k]), metric).items():
            if n in summary: summary[n][label] = v

    for n, v in get_agg(load(gekpls_file), metric).items():
        if n in summary: summary[n]['GEKPLS'] = v

    # GEKPLS with extra_points=DIM (optional)
    ndim_recs = load(gekpls_ndim_file)
    if ndim_recs is not None:
        for n, v in get_agg(ndim_recs, metric).items():
            if n in summary: summary[n]['GEKPLS (ndim)'] = v

    for it in SHOW_ITERS:
        label = f'GPyTorch ({it})'
        for n, v in get_agg(load(gpt_files[it]), metric).items():
            if n in summary: summary[n][label] = v

    return summary

def log_tick_formatter(val, pos):
    exp = int(round(val))
    return r'$10^{' + str(exp) + r'}$'

# ── Plot function ─────────────────────────────────────────────────────────────
def make_plot(metric, ylabel, fname, invert=True):
    col_labels = [rf'$n = d$ ({SAMPLE_SIZES[0]})',
                  rf'$n = 5d$ ({SAMPLE_SIZES[1]})',
                  rf'$n = 10d$ ({SAMPLE_SIZES[2]})']

    summary = build_summary(metric)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    for col, n in enumerate(SAMPLE_SIZES):
        ax = axes[col]
        ax.set_facecolor('#F8F8F6')
        ax.set_title(col_labels[col], fontsize=12, pad=8)
        data = summary.get(n, {})

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
        lower = np.log10(np.maximum(np.array(means) - np.array(stds), 1e-9))
        upper = np.log10(np.array(means) + np.array(stds))
        xerr  = [log_means - lower, upper - log_means]

        ax.barh(y_pos, log_means, xerr=xerr,
                color=colors, edgecolor='white', linewidth=0.5, height=0.6,
                error_kw=dict(ecolor='#444441', elinewidth=1.0,
                              capsize=3, capthick=1.0))

        lo = int(np.floor(min(log_means) - 0.5))
        hi = int(np.ceil(max(log_means) + 0.5))
        ax.set_xticks(list(range(lo, hi + 1)))
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(log_tick_formatter))

        if invert:
            ax.invert_xaxis()

        ax.set_xlabel(f'{ylabel} (log scale)', fontsize=12)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=11)
        ax.tick_params(axis='x', labelsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#CCCCCC')
        ax.grid(axis='x', color='#E0E0E0', linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        f'Morris 20D \u2014 {ylabel} (log scale)'
        f' \u2014 Mean $\\pm$ std over 5 macroreplicates',
        fontsize=13, y=1.02)
    plt.tight_layout()
    out = out_dir / fname
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out}')

# ── Run ───────────────────────────────────────────────────────────────────────
make_plot('nrmse',      'NRMSE',           'gddegp_morris_nrmse.pdf',      invert=True)
make_plot('train_time', 'Train time (s)',   'gddegp_morris_train_time.pdf', invert=True)
print('All done.')
