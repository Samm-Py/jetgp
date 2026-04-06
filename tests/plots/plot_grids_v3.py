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

base = Path(DATA_DIR)
extra = Path(DATA_DIR)

def load(fname, alt_base=None):
    p = (alt_base or base) / fname
    with open(p) as f:
        return json.load(f)

gpt_files = {
    'borehole': {
        100:  ('1774629006413_results_gpytorch_borehole_100iter.json', base),
        500:  ('1774629006413_results_gpytorch_borehole_500iter.json', base),
        1000: ('1774629006414_results_gpytorch_borehole_1000iter.json', base),
        2000: ('1774629006414_results_gpytorch_borehole_2000iter.json', base),
    },
    'otl_circuit': {
        100:  ('1774629006415_results_gpytorch_otl_circuit_100iter.json', base),
        500:  ('1774629006415_results_gpytorch_otl_circuit_500iter.json', base),
        1000: ('1774629006415_results_gpytorch_otl_circuit_1000iter.json', base),
        2000: ('1774629006415_results_gpytorch_otl_circuit_2000iter.json', base),
    },
    'morris': {
        100:  ('1774629006414_results_gpytorch_morris_100iter.json', base),
        500:  ('1774629006414_results_gpytorch_morris_500iter.json', base),
        1000: ('1774629006414_results_gpytorch_morris_1000iter.json', base),
        2000: ('results_gpytorch_morris_2000iter.json', extra),
    },
}
jetgp_files = {
    'borehole':    ('1774629006415_results_jetgp_borehole.json', base),
    'otl_circuit': ('1774629006416_results_jetgp_otl_circuit.json', base),
    'morris':      ('1774629006415_results_jetgp_morris.json', base),
}
gekpls_files = {
    'borehole':    ('1774629006416_results_gekpls_borehole.json', base),
    'otl_circuit': ('1774629006416_results_gekpls_otl_circuit.json', base),
    'morris':      ('1774629006416_results_gekpls_morris.json', base),
}
gekpls_ndim_files = {
    'borehole':    ('results_gekpls_ndim_borehole.json', base),
    'otl_circuit': ('results_gekpls_ndim_otl_circuit.json', base),
    'morris':      ('results_gekpls_ndim_morris.json', base),
}
func_meta = {
    'otl_circuit': {'dim': 6,  'label': 'OTL Circuit (6D)'},
    'borehole':    {'dim': 8,  'label': 'Borehole (8D)'},
    'morris':      {'dim': 20, 'label': 'Morris (20D)'},
}
SHOW_ITERS = {
    'borehole':    [100, 500, 1000, 2000],
    'otl_circuit': [100, 500, 1000, 2000],
    'morris':      [100, 500, 1000, 2000],
}
COLORS = {
    'JetGP':           '#1D9E75',
    'GEKPLS':          '#D85A30',
    'GEKPLS (ndim)':   '#E8944A',
    'GPyTorch (100)':  '#C6DCEF',
    'GPyTorch (500)':  '#6BAED6',
    'GPyTorch (1000)': '#2171B5',
    'GPyTorch (2000)': '#084594',
}

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
    n_sizes = [dim, 5*dim, 10*dim]
    summary = {n: {} for n in n_sizes}

    def get_agg(recs):
        return agg_total(recs) if metric == 'total_time' else agg(recs, metric)

    fname, b = jetgp_files[func]
    for n, v in get_agg(load(fname, b)).items():
        if n in summary: summary[n]['JetGP'] = v

    fname, b = gekpls_files[func]
    for n, v in get_agg(load(fname, b)).items():
        if n in summary: summary[n]['GEKPLS'] = v

    # GEKPLS with extra_points=DIM (optional — skip if file not found)
    if func in gekpls_ndim_files:
        ndim_fname, ndim_b = gekpls_ndim_files[func]
        ndim_path = (ndim_b) / ndim_fname
        if ndim_path.exists():
            for n, v in get_agg(load(ndim_fname, ndim_b)).items():
                if n in summary: summary[n]['GEKPLS (ndim)'] = v

    for it in SHOW_ITERS[func]:
        fname, b = gpt_files[func][it]
        label = f'GPyTorch ({it})'
        for n, v in get_agg(load(fname, b)).items():
            if n in summary: summary[n][label] = v

    return summary, n_sizes

def log_tick_formatter(val, pos):
    exp = int(round(val))
    return r'$10^{' + str(exp) + r'}$'

def make_grid(metric, ylabel, fname, invert=True):
    funcs = ['otl_circuit', 'borehole', 'morris']
    col_labels = [r'$n = d$', r'$n = 5d$', r'$n = 10d$']

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
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
                ax.set_ylabel(func_meta[func]['label'], fontsize=12, labelpad=10)

            if not data:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray', fontsize=11)
                ax.spines[['top','right']].set_visible(False)
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

            if row == 2:
                ax.set_xlabel(f'{ylabel} (log scale)', fontsize=12)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=11)
            ax.tick_params(axis='x', labelsize=11)
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color('#CCCCCC')
            ax.grid(axis='x', color='#E0E0E0', linewidth=0.5)
            ax.set_axisbelow(True)

    fig.suptitle(
        f'{ylabel} (log scale) \u2014 Mean $\\pm$ std over 5 macroreplicates',
        fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out}')

make_grid('nrmse',      'NRMSE',                         'benchmark_nrmse_grid.pdf',       invert=True)
make_grid('train_time', 'Train time (s)',                 'benchmark_time_grid.pdf',        invert=True)
make_grid('total_time', 'Train + prediction time (s)',    'benchmark_total_time_grid.pdf',  invert=True)
print('All done.')
