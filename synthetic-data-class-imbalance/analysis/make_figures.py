"""Genera las figuras 3 y 5-14 del artículo a partir de las 10 corridas (outputs/metrics_v2) y de results/stats_full.csv.
Uso (desde la raíz):  python analysis/stats_tests.py && python analysis/make_figures.py
Salida: results/figures/
"""
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', font='DejaVu Serif')
plt.rcParams.update({'figure.dpi': 300, 'savefig.facecolor': 'white'})

DSS = ['DS1', 'DS2', 'DS3', 'DS4', 'DS5']
RATIO = {'DS1': 6.18, 'DS2': 2.55, 'DS3': 3.10, 'DS4': 3.52, 'DS5': 29.99}
LAB = {'REAL': 'REAL', 'REAL_CW': 'REAL-CW', 'SMOTE': 'SMOTE-NC', 'CTGAN': 'CTGAN', 'TVAE': 'TVAE', 'HYBRID': 'HYBRID'}
LINE = {'REAL': ('#555555', 'D', '--'), 'REAL_CW': ('#9467bd', 'P', ':'), 'SMOTE': ('#1f77b4', 'o', '-'),
        'CTGAN': ('#ff7f0e', 's', '-'), 'TVAE': ('#2ca02c', '^', '-'), 'HYBRID': ('#d62728', 'X', '-')}
BAR = {'SMOTE': '#4C72B0', 'CTGAN': '#55A868', 'TVAE': '#C44E52', 'HYBRID': '#8172B2'}
names = {'cdc_diabetes': 'DS1', 'thyroid_cancer_recurrence': 'DS2', 'aids_clinical_trial_175': 'DS3',
         'credit_card_default': 'DS4', 'taiwanese_bankruptcy_prediction': 'DS5'}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER = {'DS1': 'dataset_01_cdc_diabetes', 'DS2': 'dataset_02_thyroid_cancer', 'DS3': 'dataset_03_aids_trial_175',
          'DS4': 'dataset_04_credit_default', 'DS5': 'dataset_05_taiwanese_bankruptcy'}
mv = lambda d, f: os.path.join(ROOT, FOLDER[d], 'outputs', 'metrics_v2', f)
U = pd.concat([pd.read_csv(mv(d, 'utility_runs.csv')).assign(DS=d) for d in DSS]).drop(columns=['dataset'])
F = pd.concat([pd.read_csv(mv(d, 'fidelity_runs.csv')).assign(DS=d) for d in DSS]).drop(columns=['dataset'])
ST = pd.read_csv(os.path.join(ROOT, 'results', 'stats_full.csv')); ST['DS'] = ST.dataset.map(names)
UA = U.groupby(['DS', 'model', 'train_set']).mean(numeric_only=True)
FA = F.groupby(['DS', 'method']).mean(numeric_only=True)
FS = F.groupby(['DS', 'method']).std(numeric_only=True)
OUT = os.path.join(ROOT, 'results', 'figures'); os.makedirs(OUT, exist_ok=True)
def m(d, k, met, c='CatBoost'): return UA.loc[(d, c, k), met]

# ---------------- Fig 3 ----------------
def fig3():
    order = ['SMOTE', 'CTGAN', 'TVAE', 'HYBRID']; d = 'DS1'
    ref = F[F.DS == d].real_diversity_ref.iloc[0]
    panels = [('wasserstein_std', 'Wasserstein Distance', 'Lower is better', 'min'),
              ('cat_diff_mean', 'Categorical Difference', 'Lower is better', 'min'),
              ('corr_diff_mean', 'Correlation Difference', 'Lower is better', 'min'),
              ('diversity', 'Sample Diversity', 'Closer to real is better', 'ref')]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (k, t, note, rule) in zip(axs.ravel(), panels):
        vals = [FA.loc[(d, o), k] for o in order]; sds = [FS.loc[(d, o), k] for o in order]
        best = int(np.argmin(vals)) if rule == 'min' else int(np.argmin([abs(v - ref) for v in vals]))
        y = np.arange(len(order))
        bars = ax.barh(y, vals, xerr=sds, color=[BAR[o] for o in order], alpha=.9, capsize=3, error_kw={'lw': .8})
        bars[best].set_edgecolor('black'); bars[best].set_linewidth(2)
        ax.set_yticks(y); ax.set_yticklabels([LAB[o] for o in order], fontsize=9); ax.invert_yaxis()
        ax.set_title(t, fontweight='bold', fontsize=11)
        xmax = max(np.array(vals) + np.array(sds)) * 1.35
        if k == 'diversity':
            ax.axvline(ref, color='black', ls='--', lw=1.2)
            ax.set_title(f'Sample Diversity (real = {ref:.3f})', fontweight='bold', fontsize=11)
            xmax = max(xmax, ref * 1.35)
        for yi, v, s in zip(y, vals, sds):
            ax.text(v + s + xmax * .01, yi, f'{v:.3f}', va='center', fontsize=8)
        ax.set_xlim(0, xmax)
        ax.text(.98, .03, note, transform=ax.transAxes, ha='right', fontsize=8, style='italic', color='grey')
    fig.suptitle('Fidelity and Diversity Assessment\nDataset 1  |  Imbalance Ratio 6.18:1  |  Mean ± SD of 10 runs',
                 fontweight='bold', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(f'{OUT}/fig3.png'); plt.close(fig)

# ---------------- Fig 5 / Fig 7 ----------------
def fig_line(met, fname, title, ylabel, shade=False):
    xs = sorted(DSS, key=lambda d: RATIO[d]); xr = [RATIO[d] for d in xs]
    fig, ax = plt.subplots(figsize=(8, 5))
    if shade:
        ax.axvspan(5, 35, color='#e8e8e8', alpha=.6, label='Extreme Imbalance Zone')
    for k in ['REAL', 'REAL_CW', 'SMOTE', 'CTGAN', 'TVAE', 'HYBRID']:
        c, mk, ls = LINE[k]
        ax.plot(xr, [m(d, k, met) for d in xs], color=c, marker=mk, ls=ls, lw=2.2 if k != 'REAL' else 2.4,
                ms=8, label=LAB[k], markeredgecolor='white' if k not in ('REAL',) else c)
    ax.set_xscale('log'); ax.set_xticks(xr); ax.set_xticklabels([f'{x:.2f}\n({d})' for x, d in zip(xr, xs)], fontsize=8, rotation=0); ax.minorticks_off()
    for lab_, d in zip(ax.get_xticklabels(), xs):
        if d == 'DS3': lab_.set_ha('center')
    ax.set_xlabel('Imbalance Ratio (Majority : Minority) [Log Scale]', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(title='Method', fontsize=9, title_fontsize=10, loc='upper right', frameon=True, shadow=True, ncol=2)
    fig.text(.5, .01, 'CatBoost classifier; values are means over 10 runs. XGBoost results in Table A2.',
             ha='center', fontsize=8, style='italic', color='#444444')
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(f'{OUT}/{fname}'); plt.close(fig)

# ---------------- Fig 6 ----------------
def fig6():
    mets = [('AUC_PR', 'AUC-PR\n(Ranking)'), ('F1', 'F1-Score\n(Balance)'), ('Recall', 'Recall\n(Minority Detection)')]
    meths = ['REAL', 'REAL_CW', 'SMOTE', 'CTGAN', 'TVAE', 'HYBRID']
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for row, clf in enumerate(['CatBoost', 'XGBoost']):
        for col, (met, t) in enumerate(mets):
            ax = axs[row, col]
            base = m('DS5', 'REAL', met, clf)
            vals = [m('DS5', k, met, clf) for k in meths]; cols = []
            for k in meths:
                if k == 'REAL': cols.append('grey'); continue
                s = ST[(ST.DS == 'DS5') & (ST.model == clf) & (ST.method == k) & (ST.metric == met)].iloc[0]
                cols.append('#2ca02c' if s.res == 'UP' else '#d62728' if s.res == 'DOWN' else 'grey')
            bars = ax.bar(range(len(meths)), vals, color=cols, edgecolor='black', lw=1)
            ax.axhline(base, color='black', ls='--', lw=1.3)
            for i, (b, v) in enumerate(zip(bars, vals)):
                ax.text(b.get_x() + b.get_width() / 2, v + .015, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
                pct = 0 if base == 0 else 100 * (v - base) / base
                ax.text(b.get_x() + b.get_width() / 2, v / 2, f'{pct:+.1f}%', ha='center', color='white', fontsize=9, fontweight='bold')
            ax.set_xticks(range(len(meths))); ax.set_xticklabels([LAB[k] for k in meths], fontsize=9, rotation=30, ha='right')
            ax.set_ylim(0, 1.05)
            ax.set_title(f'{t} — {clf}', fontweight='bold', fontsize=11)
            sns.despine(ax=ax)
    fig.suptitle('Performance Comparison - Dataset 5 (Imbalance 29.99:1), mean of 10 runs\n'
                 'green / red: significant increase / decrease vs REAL after correction; grey: not significant',
                 fontweight='bold', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(f'{OUT}/fig6.png'); plt.close(fig)

# ---------------- Figs 9-13: confusion matrices ----------------
PAIR = {'DS1': 'SMOTE', 'DS2': 'CTGAN', 'DS3': 'TVAE', 'DS4': 'SMOTE', 'DS5': 'SMOTE'}
def fig_cm(d, idx):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, k in zip(axs, ['REAL', PAIR[d]]):
        tn, fp, fn, tp = [m(d, k, c) for c in ['TN', 'FP', 'FN', 'TP']]
        cm = np.array([[tn, fp], [fn, tp]])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ann = np.array([[f'{cm[i, j]:,.0f}\n({pct[i, j]:.2f}%)' for j in range(2)] for i in range(2)])
        sns.heatmap(pct, annot=ann, fmt='', cmap='Blues', cbar=True, ax=ax, vmin=0, vmax=100,
                    annot_kws={'fontsize': 13}, linewidths=0, square=True,
                    xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        ax.add_patch(plt.Rectangle((0, 1), 2, 1, fill=False, edgecolor='red', lw=3))
        ax.set_title(f'{LAB[k]} + CatBoost', fontweight='bold', fontsize=13)
    fig.suptitle(f'Confusion Matrix Analysis — Dataset {d[-1]} (Imbalance {RATIO[d]:.2f}:1), mean of 10 runs',
                 fontweight='bold', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(f'{OUT}/fig{idx}.png'); plt.close(fig)

# ---------------- Fig 14: trade-off ----------------
def fig14():
    mk = {'DS1': 's', 'DS2': 'v', 'DS3': 'o', 'DS4': '^', 'DS5': 'D'}
    colm = {'SMOTE': '#2b3a4a', 'CTGAN': '#7f8c8d', 'TVAE': '#16a085', 'HYBRID': '#c0392b'}
    fig, ax = plt.subplots(figsize=(8, 5.6))
    xs, ys = [], []
    for d in DSS:
        for k in ['SMOTE', 'CTGAN', 'TVAE', 'HYBRID']:
            x = FA.loc[(d, k), 'wasserstein_std']; y = m(d, k, 'AUC_PR')
            ax.scatter(x, y, marker=mk[d], color=colm[k], s=110, edgecolor='black', lw=1, zorder=3)
            xs.append(x); ys.append(y)
    ax.axvline(np.median(xs), color='#8fa8c8', ls=':', lw=1.5); ax.axhline(np.median(ys), color='#8fa8c8', ls=':', lw=1.5)
    ax.set_xlabel('Fidelity Metric: Wasserstein Distance (↓)', fontsize=11)
    ax.set_ylabel('Predictive Metric: AUC-PR (↑), CatBoost', fontsize=11)
    ax.set_title('Experimental Trade-off Analysis: Fidelity vs. Model Utility', fontweight='bold', fontsize=13)
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], marker='o', ls='', color=colm[k], markeredgecolor='black', label=LAB[k], ms=8) for k in colm]
    h2 = [Line2D([], [], marker=mk[d], ls='', color='grey', markeredgecolor='black', label=d, ms=8) for d in DSS]
    l1 = ax.legend(handles=h1, title='Method', loc='upper right', fontsize=9); ax.add_artist(l1)
    ax.legend(handles=h2, title='Dataset', loc='lower right', ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig14.png'); plt.close(fig)

fig3()
fig_line('AUC_PR', 'fig5.png', 'Methodological Resilience across Imbalance Scales\nMetric: AUC-PR', 'Metric Score: AUC-PR')
fig6()
fig_line('F1', 'fig7.png', 'Methodological Resilience Analysis: F1-Score', 'Predictive Performance (F1-Score)', shade=True)
for i, d in enumerate(DSS):
    fig_cm(d, 9 + i)

def fig8():
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)); axs = axs.ravel()
    for ax, d in zip(axs, DSS):
        for fval in [.2, .4, .6, .8]:
            r = np.linspace(fval / 2 + 1e-3, 1, 200); p = fval * r / (2 * r - fval); ok = (p > 0) & (p <= 1)
            ax.plot(r[ok], p[ok], color='grey', lw=.7, alpha=.5)
            ax.text(r[ok][-1], p[ok][-1], f'F1={fval}', fontsize=7, color='grey', ha='right', va='bottom')
        for k in ['REAL', 'REAL_CW', 'SMOTE', 'CTGAN', 'TVAE', 'HYBRID']:
            c, mk, _ = LINE[k]
            ax.scatter(m(d, k, 'Recall'), m(d, k, 'Precision'), color=c, marker=mk, s=110, edgecolor='black', lw=.8, label=LAB[k], zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'Dataset {d[-1]} (Imbalance {RATIO[d]:.2f}:1)', fontweight='bold', fontsize=11)
    h, l = axs[0].get_legend_handles_labels(); axs[5].axis('off')
    axs[5].legend(h, l, loc='center', fontsize=11, title='Method', title_fontsize=12, frameon=True, shadow=True)
    fig.suptitle('Precision-Recall Operating Points at the Default Threshold (CatBoost, mean of 10 runs)', fontweight='bold', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(f'{OUT}/fig8.png'); plt.close(fig)
fig8()
fig14()
print('ok')
