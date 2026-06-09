#!/usr/bin/env python3
"""
Phase 3a — Time Series Cell Type Composition
==============================================
Publication-quality figures for cell type dynamics across timepoints.

Reads the annotated AnnData from Phase 2 and produces:
  1. Proportion trajectory panels (Tier 1, Tier 2 subtypes)
  2. Stacked bar composition per treatment × week
  3. Proportion fold-change from Week 5 baseline
  4. Focused biology panel (Mac polarization, SIV burden, IDO, CD4 scores)
  5. Underlying data tables for all figures

Treatment groups:
  - D1MT + Late cART : IDO inhibitor + late antiretroviral therapy
  - Late cART        : Late cART only (no IDO inhibition)

Timepoints: Week 5, 11, 15 post-infection

Macrophage subtypes (5): AM-like, TREM2+, M2 IM-like, IFN-responsive, M1 IM-like

Formatting: Publication-ready (28pt axis labels, 24pt ticks, no titles,
            large markers, clean spines). Titles added manually in figure
            assembly.

Run in Spyder (interactive cells with # %%) or standalone.
Output dir: analysis/phase3a_timeseries/

Author: Jake Lehle / Kaushal Lab
Date: May 2026
"""

# %% Cell 1 — Configuration, Load, and Compute Per-Animal Proportions
# =========================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === PUBLICATION FORMATTING ===
AXIS_LABEL_SIZE = 28
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 20
MARKER_SIZE = 12
LINE_WIDTH = 3
CAP_SIZE = 6
CAP_THICK = 2
ERROR_LINE_WIDTH = 2

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,       # Editable text in Illustrator
    'ps.fonttype': 42,
    'axes.labelsize': AXIS_LABEL_SIZE,
    'axes.titlesize': AXIS_LABEL_SIZE,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'figure.dpi': 150,
})

# === PATHS ===
ADATA_IN = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_focal_annotated.h5ad'
OUT_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/phase3a_timeseries/'
os.makedirs(OUT_DIR, exist_ok=True)

# === TREATMENT CONFIG ===
TREATMENT_CONFIG = {
    'D1MT + Late cART': {'color': '#D32F2F', 'marker': 'o'},
    'Late cART':        {'color': '#1976D2', 'marker': 's'},
}
TREATMENT_COL = 'treatment'
ANIMAL_COL = 'animal_id'
WEEK_COL = 'week'
WEEKS = [5, 11, 15]

# === LOAD ===
print("Loading Phase 2 annotated data...")
adata = sc.read_h5ad(ADATA_IN)
adata.obs[WEEK_COL] = pd.to_numeric(adata.obs[WEEK_COL], errors='coerce').astype(int)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  Treatments: {sorted(adata.obs[TREATMENT_COL].unique())}")
print(f"  Weeks: {sorted(adata.obs[WEEK_COL].unique())}")

# === COMPUTE PER-ANIMAL PROPORTIONS ===
def compute_proportions(adata_obj, celltype_col):
    """Per-animal proportions → mean ± SEM per treatment × week × celltype."""
    counts = (
        adata_obj.obs
        .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL, celltype_col], observed=True)
        .size()
        .reset_index(name='n_cells')
    )
    totals = (
        counts
        .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL], observed=True)['n_cells']
        .sum()
        .reset_index(name='total_cells')
    )
    counts = counts.merge(totals, on=[TREATMENT_COL, ANIMAL_COL, WEEK_COL])
    counts['proportion'] = counts['n_cells'] / counts['total_cells']

    # Zero-fill missing cell types
    all_types = sorted(adata_obj.obs[celltype_col].unique())
    all_combos = (
        counts[[TREATMENT_COL, ANIMAL_COL, WEEK_COL]]
        .drop_duplicates()
        .assign(key=1)
        .merge(pd.DataFrame({celltype_col: all_types, 'key': 1}), on='key')
        .drop('key', axis=1)
    )
    counts = all_combos.merge(
        counts, on=[TREATMENT_COL, ANIMAL_COL, WEEK_COL, celltype_col], how='left'
    )
    counts['proportion'] = counts['proportion'].fillna(0)
    counts['n_cells'] = counts['n_cells'].fillna(0)

    agg = (
        counts
        .groupby([TREATMENT_COL, WEEK_COL, celltype_col], observed=True)['proportion']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    agg['sem'] = agg['std'] / np.sqrt(agg['count'])
    agg[WEEK_COL] = pd.to_numeric(agg[WEEK_COL], errors='coerce')
    return counts, agg


def compute_foldchange(counts_df, celltype_col):
    """Proportion fold-change from Week 5 per animal, then aggregate."""
    animal_props = (
        counts_df
        .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL, celltype_col], observed=True)
        ['proportion'].first()
        .reset_index()
    )

    baseline = (
        animal_props[animal_props[WEEK_COL] == 5]
        [[TREATMENT_COL, ANIMAL_COL, celltype_col, 'proportion']]
        .rename(columns={'proportion': 'baseline_prop'})
    )

    fc = animal_props.merge(
        baseline, on=[TREATMENT_COL, ANIMAL_COL, celltype_col], how='inner'
    )
    fc['fold_change'] = np.where(
        fc['baseline_prop'] > 0,
        fc['proportion'] / fc['baseline_prop'],
        np.nan
    )

    fc_agg = (
        fc.groupby([TREATMENT_COL, WEEK_COL, celltype_col])['fold_change']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    fc_agg['sem'] = fc_agg['std'] / np.sqrt(fc_agg['count'])
    fc_agg[WEEK_COL] = pd.to_numeric(fc_agg[WEEK_COL], errors='coerce')
    return fc, fc_agg


# === PLOTTING HELPERS ===
def _style_ax(ax, ylabel, show_legend=False):
    """Apply publication formatting to an axis."""
    ax.set_xlabel('Week post-infection')
    ax.set_ylabel(ylabel)
    ax.set_xlim(3, 17)
    ax.xaxis.set_major_locator(mticker.FixedLocator(WEEKS))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    if show_legend:
        ax.legend(frameon=True, fancybox=False, edgecolor='black')


def plot_treatment_lines(ax, data_df, celltype_col, celltype_val, value_col='mean',
                         sem_col='sem', scale=1.0):
    """Plot treatment trajectory lines on an axis."""
    for treatment, cfg in TREATMENT_CONFIG.items():
        subset = data_df[
            (data_df[TREATMENT_COL] == treatment) &
            (data_df[celltype_col] == celltype_val)
        ].sort_values(WEEK_COL)

        if subset.empty:
            continue

        weeks = subset[WEEK_COL].values.astype(float)
        means = subset[value_col].values.astype(float) * scale
        sems = np.nan_to_num(subset[sem_col].values.astype(float) * scale, nan=0.0)
        valid = ~np.isnan(means)

        if valid.sum() == 0:
            continue

        ax.plot(weeks[valid], means[valid],
                marker=cfg['marker'], markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, color=cfg['color'],
                label=treatment, zorder=3,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.errorbar(weeks[valid], means[valid], yerr=sems[valid],
                    fmt='none', ecolor=cfg['color'],
                    capsize=CAP_SIZE, capthick=CAP_THICK,
                    elinewidth=ERROR_LINE_WIDTH, zorder=2)


def plot_panel(populations, celltype_col, agg_df, ylabel, filename,
               ncols=3, scale=100.0):
    """Multi-subplot panel for a set of cell types."""
    n = len(populations)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8 * ncols, 7 * nrows), squeeze=False)

    for idx, pop in enumerate(populations):
        ax = axes[idx // ncols, idx % ncols]
        plot_treatment_lines(ax, agg_df, celltype_col, pop, scale=scale)

        # Short label for y-axis context
        short = pop.split(': ')[-1] if ': ' in pop else pop
        ax.text(0.05, 0.95, short, transform=ax.transAxes,
                fontsize=AXIS_LABEL_SIZE, fontweight='bold',
                va='top', ha='left')
        _style_ax(ax, ylabel, show_legend=(idx == 0))

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, f'{filename}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{filename}.png'), bbox_inches='tight', dpi=300)
    print(f"  Saved: {filename}")
    plt.close(fig)


def plot_fc_panel(populations, celltype_col, fc_agg_df, filename, ncols=3):
    """Multi-subplot panel for fold-change from W5."""
    n = len(populations)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(8 * ncols, 7 * nrows), squeeze=False)

    for idx, pop in enumerate(populations):
        ax = axes[idx // ncols, idx % ncols]

        for treatment, cfg in TREATMENT_CONFIG.items():
            subset = fc_agg_df[
                (fc_agg_df[TREATMENT_COL] == treatment) &
                (fc_agg_df[celltype_col] == pop)
            ].sort_values(WEEK_COL)

            if subset.empty:
                continue

            weeks = subset[WEEK_COL].values.astype(float)
            means = subset['mean'].values.astype(float)
            sems = np.nan_to_num(subset['sem'].values.astype(float), nan=0.0)
            valid = ~np.isnan(means)

            if valid.sum() == 0:
                continue

            ax.plot(weeks[valid], means[valid],
                    marker=cfg['marker'], markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, color=cfg['color'],
                    label=treatment, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.errorbar(weeks[valid], means[valid], yerr=sems[valid],
                        fmt='none', ecolor=cfg['color'],
                        capsize=CAP_SIZE, capthick=CAP_THICK,
                        elinewidth=ERROR_LINE_WIDTH, zorder=2)

        ax.axhline(1.0, color='grey', linestyle=':', linewidth=1.5, alpha=0.6)

        short = pop.split(': ')[-1] if ': ' in pop else pop
        ax.text(0.05, 0.95, short, transform=ax.transAxes,
                fontsize=AXIS_LABEL_SIZE, fontweight='bold',
                va='top', ha='left')
        _style_ax(ax, 'Fold-change from W5', show_legend=(idx == 0))

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, f'{filename}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f'{filename}.png'), bbox_inches='tight', dpi=300)
    print(f"  Saved: {filename}")
    plt.close(fig)


print("Setup complete.\n")


# %% Cell 2 — Tier 1 Broad Lineage Trajectories (Proportions + Fold-Change)
# =========================================================================
print("=" * 80)
print("TIER 1 — Broad Lineage Proportions and Fold-Change")
print("=" * 80)

counts_t1, agg_t1 = compute_proportions(adata, 'tier1_celltype')

t1_pops = [t for t in ['T cells', 'Macrophages', 'Monocytes', 'DCs',
                        'NK cells', 'B cells', 'Mast cells', 'Ciliated epithelial']
           if t in agg_t1['tier1_celltype'].unique()]

# Proportions
plot_panel(t1_pops, 'tier1_celltype', agg_t1,
           '% of cells', 'tier1_proportions', ncols=3, scale=100.0)

# Fold-change
_, fc_agg_t1 = compute_foldchange(counts_t1, 'tier1_celltype')
fc_pops_t1 = ['T cells', 'Macrophages', 'Monocytes']
plot_fc_panel(fc_pops_t1, 'tier1_celltype', fc_agg_t1,
              'tier1_foldchange', ncols=3)


# %% Cell 3 — Tier 2 Subtype Trajectories
# =========================================================================
print("\n" + "=" * 80)
print("TIER 2 — Subtype Proportions and Fold-Change")
print("=" * 80)

counts_t2, agg_t2 = compute_proportions(adata, 'tier2_full')

# --- Macrophage subtypes (5: AM-like, TREM2+, M2 IM-like, IFN-responsive, M1 IM-like) ---
mac_pops = sorted([t for t in agg_t2['tier2_full'].unique() if 'Macrophage' in str(t)])
if mac_pops:
    plot_panel(mac_pops, 'tier2_full', agg_t2,
               '% of cells', 'tier2_macrophage_proportions', ncols=3)
    _, fc_agg_mac = compute_foldchange(counts_t2, 'tier2_full')
    plot_fc_panel(mac_pops, 'tier2_full', fc_agg_mac,
                  'tier2_macrophage_foldchange', ncols=3)

# --- T cell subtypes ---
tc_pops = sorted([t for t in agg_t2['tier2_full'].unique() if 'T cells' in str(t)])
if tc_pops:
    plot_panel(tc_pops, 'tier2_full', agg_t2,
               '% of cells', 'tier2_tcell_proportions', ncols=2)
    _, fc_agg_tc = compute_foldchange(counts_t2, 'tier2_full')
    plot_fc_panel(tc_pops, 'tier2_full', fc_agg_tc,
                  'tier2_tcell_foldchange', ncols=2)

# --- DC subtypes ---
dc_pops = sorted([t for t in agg_t2['tier2_full'].unique() if 'DC' in str(t)])
if dc_pops:
    plot_panel(dc_pops, 'tier2_full', agg_t2,
               '% of cells', 'tier2_dc_proportions', ncols=3)

# --- Monocyte subtypes ---
mono_pops = sorted([t for t in agg_t2['tier2_full'].unique() if 'Monocyte' in str(t)])
if mono_pops:
    plot_panel(mono_pops, 'tier2_full', agg_t2,
               '% of cells', 'tier2_monocyte_proportions', ncols=2)


# %% Cell 4 — Stacked Bar Composition
# =========================================================================
print("\n" + "=" * 80)
print("STACKED BAR — Cell Type Composition per Treatment × Week")
print("=" * 80)

# Build display order
display_order = []
for treatment in ['D1MT + Late cART', 'Late cART']:
    for wk in WEEKS:
        display_order.append((treatment, wk))

# Stacking order: macrophages first, then monocytes, T cells, minor pops
STACKED_ORDER = [
    'Macrophages: AM-like', 'Macrophages: TREM2+',
    'Macrophages: M2 IM-like', 'Macrophages: IFN-responsive',
    'Macrophages: M1 IM-like',
    'Monocytes: Classical', 'Monocytes: Non-classical',
    'T cells: CD4', 'T cells: CD8',
    'DCs: cDC1', 'DCs: cDC2', 'DCs: pDC',
    'NK cells', 'B cells', 'Mast cells',
    'Ciliated epithelial', 'Unassigned',
]

STACKED_COLORS = {
    'Macrophages: AM-like':        '#D32F2F',
    'Macrophages: TREM2+':         '#1976D2',
    'Macrophages: M2 IM-like':       '#F9A825',
    'Macrophages: IFN-responsive': '#7B1FA2',
    'Macrophages: M1 IM-like':     '#FF6F00',
    'Monocytes: Classical':        '#FF7043',
    'Monocytes: Non-classical':    '#00897B',
    'T cells: CD4':                '#42A5F5',
    'T cells: CD8':                '#E64B35',
    'DCs: cDC1':                   '#78909C',
    'DCs: cDC2':                   '#CDDC39',
    'DCs: pDC':                    '#2E7D32',
    'NK cells':                    '#5D4037',
    'B cells':                     '#EC407A',
    'Mast cells':                  '#8D6E63',
    'Ciliated epithelial':         '#546E7A',
    'Unassigned':                  '#BDBDBD',
}

stacked_data = []
bar_labels = []

for treatment, wk in display_order:
    subset = agg_t2[
        (agg_t2[TREATMENT_COL] == treatment) & (agg_t2[WEEK_COL] == wk)
    ]
    if subset.empty:
        continue
    props = {}
    for ct in STACKED_ORDER:
        ct_row = subset[subset['tier2_full'] == ct]
        props[ct] = ct_row['mean'].values[0] * 100 if len(ct_row) > 0 else 0
    stacked_data.append(props)

    short_treat = TREATMENT_CONFIG[treatment]['marker']
    bar_labels.append(f'{treatment}\nW{wk}')

df_stacked = pd.DataFrame(stacked_data, index=range(len(stacked_data)))

fig, ax = plt.subplots(figsize=(14, 9))
bottom = np.zeros(len(df_stacked))

for ct in STACKED_ORDER:
    if ct not in df_stacked.columns:
        continue
    values = df_stacked[ct].values
    color = STACKED_COLORS.get(ct, '#999999')
    ax.bar(range(len(df_stacked)), values, bottom=bottom,
           label=ct, color=color, edgecolor='white', linewidth=0.8, width=0.75)
    bottom += values

ax.set_xticks(range(len(df_stacked)))
ax.set_xticklabels(bar_labels, fontsize=18, ha='center')
ax.set_ylabel('% of cells', fontsize=AXIS_LABEL_SIZE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE, width=1.5, length=6)
ax.tick_params(axis='x', width=1.5, length=6)
ax.set_ylim(0, 105)

# Group separator
ax.axvline(2.5, color='black', linewidth=1.5, linestyle='--', alpha=0.5)

# Legend outside right
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14,
          ncol=1, frameon=True, fancybox=False, edgecolor='black')

plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()
fig.savefig(os.path.join(OUT_DIR, 'stacked_bar_composition.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'stacked_bar_composition.png'), bbox_inches='tight', dpi=300)
print("  Saved: stacked_bar_composition")
plt.close(fig)


# %% Cell 5 — Focused Biology Panel
# =========================================================================
print("\n" + "=" * 80)
print("FOCUSED BIOLOGY — Mac Polarization, SIV, IDO, CD4 Scores")
print("=" * 80)

# --- 5A: Macrophage AM-like / IFN-responsive ratio ---
print("  Computing macrophage polarization ratio...")

mac_types = ['Macrophages: AM-like', 'Macrophages: IFN-responsive']
mac_counts = counts_t2[counts_t2['tier2_full'].isin(mac_types)].copy()
mac_pivot = mac_counts.pivot_table(
    index=[TREATMENT_COL, ANIMAL_COL, WEEK_COL],
    columns='tier2_full',
    values='proportion',
    fill_value=0
).reset_index()

mac_sum = (mac_pivot.get('Macrophages: AM-like', 0) +
           mac_pivot.get('Macrophages: IFN-responsive', 0))
mac_pivot['am_ifn_ratio'] = np.where(mac_sum > 0,
    mac_pivot.get('Macrophages: AM-like', 0) / mac_sum, np.nan)

ratio_agg = (
    mac_pivot
    .groupby([TREATMENT_COL, WEEK_COL])['am_ifn_ratio']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
ratio_agg['sem'] = ratio_agg['std'] / np.sqrt(ratio_agg['count'])
ratio_agg[WEEK_COL] = pd.to_numeric(ratio_agg[WEEK_COL], errors='coerce')

fig, ax = plt.subplots(figsize=(8, 7))
for treatment, cfg in TREATMENT_CONFIG.items():
    subset = ratio_agg[ratio_agg[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
    if subset.empty:
        continue
    weeks = subset[WEEK_COL].values.astype(float)
    means = subset['mean'].values.astype(float)
    sems = np.nan_to_num(subset['sem'].values.astype(float), nan=0.0)
    valid = ~np.isnan(means)
    ax.plot(weeks[valid], means[valid], marker=cfg['marker'], markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH, color=cfg['color'], label=treatment, zorder=3,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                ecolor=cfg['color'], capsize=CAP_SIZE, capthick=CAP_THICK,
                elinewidth=ERROR_LINE_WIDTH, zorder=2)

ax.axhline(0.5, color='grey', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_ylim(0, 1.05)
_style_ax(ax, 'AM-like / (AM-like + IFN-responsive)', show_legend=True)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(OUT_DIR, 'mac_polarization_ratio.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT_DIR, 'mac_polarization_ratio.png'), bbox_inches='tight', dpi=300)
print("  Saved: mac_polarization_ratio")
plt.close(fig)

# --- 5B: SIV+ cell fraction ---
print("  Computing SIV+ cell fraction...")

if 'total_counts_is_siv' in adata.obs.columns:
    siv_data = (
        adata.obs
        .assign(siv_positive=(adata.obs['total_counts_is_siv'] > 0).astype(int))
        .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL], observed=True)
        .agg(total_cells=('siv_positive', 'size'), siv_pos=('siv_positive', 'sum'))
        .reset_index()
    )
    siv_data['siv_pct'] = siv_data['siv_pos'] / siv_data['total_cells'] * 100

    siv_agg = (
        siv_data
        .groupby([TREATMENT_COL, WEEK_COL])['siv_pct']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    siv_agg['sem'] = siv_agg['std'] / np.sqrt(siv_agg['count'])
    siv_agg[WEEK_COL] = pd.to_numeric(siv_agg[WEEK_COL], errors='coerce')

    fig, ax = plt.subplots(figsize=(8, 7))
    for treatment, cfg in TREATMENT_CONFIG.items():
        subset = siv_agg[siv_agg[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
        if subset.empty:
            continue
        weeks = subset[WEEK_COL].values.astype(float)
        means = subset['mean'].values.astype(float)
        sems = np.nan_to_num(subset['sem'].values.astype(float), nan=0.0)
        valid = ~np.isnan(means)
        ax.plot(weeks[valid], means[valid], marker=cfg['marker'], markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, color=cfg['color'], label=treatment, zorder=3,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                    ecolor=cfg['color'], capsize=CAP_SIZE, capthick=CAP_THICK,
                    elinewidth=ERROR_LINE_WIDTH, zorder=2)

    _style_ax(ax, '% SIV+ cells', show_legend=True)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, 'siv_positive_fraction.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'siv_positive_fraction.png'), bbox_inches='tight', dpi=300)
    print("  Saved: siv_positive_fraction")
    plt.close(fig)
else:
    print("  *** total_counts_is_siv not found — skipping SIV plot ***")
    siv_data = None

# --- 5C: IDO pathway score in macrophages ---
print("  Computing IDO pathway score in macrophages...")

mac_mask = adata.obs['tier1_celltype'] == 'Macrophages'
if 'IDO_pathway_score' in adata.obs.columns and mac_mask.sum() > 0:
    ido_data = (
        adata.obs.loc[mac_mask]
        .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL], observed=True)['IDO_pathway_score']
        .mean()
        .reset_index()
    )
    ido_agg = (
        ido_data
        .groupby([TREATMENT_COL, WEEK_COL])['IDO_pathway_score']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    ido_agg['sem'] = ido_agg['std'] / np.sqrt(ido_agg['count'])
    ido_agg[WEEK_COL] = pd.to_numeric(ido_agg[WEEK_COL], errors='coerce')

    fig, ax = plt.subplots(figsize=(8, 7))
    for treatment, cfg in TREATMENT_CONFIG.items():
        subset = ido_agg[ido_agg[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
        if subset.empty:
            continue
        weeks = subset[WEEK_COL].values.astype(float)
        means = subset['mean'].values.astype(float)
        sems = np.nan_to_num(subset['sem'].values.astype(float), nan=0.0)
        valid = ~np.isnan(means)
        ax.plot(weeks[valid], means[valid], marker=cfg['marker'], markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH, color=cfg['color'], label=treatment, zorder=3,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                    ecolor=cfg['color'], capsize=CAP_SIZE, capthick=CAP_THICK,
                    elinewidth=ERROR_LINE_WIDTH, zorder=2)

    _style_ax(ax, 'IDO pathway score', show_legend=True)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, 'ido_score_macrophages.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'ido_score_macrophages.png'), bbox_inches='tight', dpi=300)
    print("  Saved: ido_score_macrophages")
    plt.close(fig)

# --- 5D: CD4 Effector continuous scores ---
print("  Computing CD4 Effector scores...")

cd4_eff = adata[adata.obs['tier3_state'] == 'CD4 Effector'].copy()

if cd4_eff.n_obs > 100:
    score_panels = [
        ('score_Th1', 'Th1 score'),
        ('score_Th17', 'Th17 score'),
        ('exhaustion_score', 'Exhaustion score'),
        ('SIV_host_factor_score', 'SIV host factor score'),
        ('IDO_pathway_score', 'IDO pathway score'),
        ('granuloma_trafficking_score', 'Granuloma trafficking score'),
    ]
    score_panels = [(c, l) for c, l in score_panels if c in cd4_eff.obs.columns]
    n = len(score_panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)

    for idx, (score_col, ylabel) in enumerate(score_panels):
        ax = axes[idx // ncols, idx % ncols]

        score_data = (
            cd4_eff.obs
            .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL], observed=True)[score_col]
            .mean()
            .reset_index()
        )
        score_agg = (
            score_data
            .groupby([TREATMENT_COL, WEEK_COL])[score_col]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        score_agg['sem'] = score_agg['std'] / np.sqrt(score_agg['count'])
        score_agg[WEEK_COL] = pd.to_numeric(score_agg[WEEK_COL], errors='coerce')

        for treatment, cfg in TREATMENT_CONFIG.items():
            subset = score_agg[score_agg[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
            if subset.empty:
                continue
            weeks = subset[WEEK_COL].values.astype(float)
            means = subset['mean'].values.astype(float)
            sems = np.nan_to_num(subset['sem'].values.astype(float), nan=0.0)
            valid = ~np.isnan(means)
            if valid.sum() == 0:
                continue
            ax.plot(weeks[valid], means[valid], marker=cfg['marker'],
                    markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                    color=cfg['color'], label=treatment, zorder=3,
                    markeredgecolor='white', markeredgewidth=1.5)
            ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                        ecolor=cfg['color'], capsize=CAP_SIZE, capthick=CAP_THICK,
                        elinewidth=ERROR_LINE_WIDTH, zorder=2)

        _style_ax(ax, ylabel, show_legend=(idx == 0))

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, 'cd4_effector_scores.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'cd4_effector_scores.png'), bbox_inches='tight', dpi=300)
    print("  Saved: cd4_effector_scores")
    plt.close(fig)


# %% Cell 6 — Export Data Tables and Summary
# =========================================================================
print("\n" + "=" * 80)
print("EXPORTING DATA TABLES")
print("=" * 80)

# Per-animal proportions (Tier 1 and Tier 2)
counts_t1.to_csv(os.path.join(OUT_DIR, 'per_animal_tier1_proportions.csv'), index=False)
counts_t2.to_csv(os.path.join(OUT_DIR, 'per_animal_tier2_proportions.csv'), index=False)
print("  Saved: per_animal_tier1_proportions.csv, per_animal_tier2_proportions.csv")

# Aggregated stats
agg_t1.to_csv(os.path.join(OUT_DIR, 'agg_tier1_stats.csv'), index=False)
agg_t2.to_csv(os.path.join(OUT_DIR, 'agg_tier2_stats.csv'), index=False)
print("  Saved: agg_tier1_stats.csv, agg_tier2_stats.csv")

# Fold-change data
if 'fc_agg_t1' in dir():
    fc_agg_t1.to_csv(os.path.join(OUT_DIR, 'foldchange_tier1.csv'), index=False)
if 'fc_agg_mac' in dir():
    fc_agg_mac.to_csv(os.path.join(OUT_DIR, 'foldchange_tier2_macrophages.csv'), index=False)
if 'fc_agg_tc' in dir():
    fc_agg_tc.to_csv(os.path.join(OUT_DIR, 'foldchange_tier2_tcells.csv'), index=False)
print("  Saved: foldchange CSVs")

# Focused biology data
mac_pivot.to_csv(os.path.join(OUT_DIR, 'per_animal_mac_polarization.csv'), index=False)
if siv_data is not None:
    siv_data.to_csv(os.path.join(OUT_DIR, 'per_animal_siv_burden.csv'), index=False)
print("  Saved: focused biology CSVs")

# Summary table
print("\n--- Treatment × Week Summary ---")
summary = (
    adata.obs
    .groupby([TREATMENT_COL, WEEK_COL], observed=True)
    .agg(
        n_animals=(ANIMAL_COL, 'nunique'),
        n_cells=('tier1_celltype', 'size'),
    )
    .reset_index()
    .sort_values([TREATMENT_COL, WEEK_COL])
)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, 'summary_treatment_week.csv'), index=False)

print(f"\n{'=' * 80}")
print("PHASE 3a COMPLETE")
print(f"{'=' * 80}")
print(f"  Output directory: {OUT_DIR}")
print(f"  All figures saved as PDF (vector) and PNG (300 dpi)")
print(f"  Formatting: {AXIS_LABEL_SIZE}pt axis labels, {TICK_LABEL_SIZE}pt ticks, no titles")
print(f"\n  Next: Phase 3b — DEG analysis")
