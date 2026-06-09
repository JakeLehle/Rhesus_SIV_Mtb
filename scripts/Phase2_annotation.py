#!/usr/bin/env python3
"""
Phase 2 — Tiered Annotation, Scoring, and Diagnostics
=======================================================
Standalone pipeline for Rhesus Mtb/SIV coinfection BAL scRNA-seq.

Takes the QC'd, embedded AnnData from Phase 1 and performs:
  1. Hierarchical cell type annotation (Tier 1 → 2 → 3)
  2. Biological gene set scoring (IDO, exhaustion, SIV, Th1/Th17, etc.)
  3. Diagnostic UMAPs and dotplots
  4. Proportion trajectories and fold-change from Week 5

Treatment groups (from Phase 1):
  - D1MT + Late cART : IDO inhibitor (1-methyl-D-tryptophan) + late cART
  - Late cART        : Late cART only (no IDO inhibition)

Tiered annotation system:
  Tier 1: 8 broad lineages (scored from scratch, winner-take-all)
  Tier 2: Subtypes within lineage (CD4/CD8, Mac subtypes, DC subtypes, etc.)
          Macrophages now include 5 subtypes: AM-like, TREM2+, M2 IM-like,
          IFN-responsive, and M1 IM-like (inflammatory antigen-presenting,
          MRC1-low IDO1-low, scores on STAT1/IRF1/TNF/IL1B/CCL2/CCL3).
  Tier 3: T cell functional states — Naive vs Effector (discrete)
          + continuous scores (Th1, Th17, exhaustion, IDO pathway, etc.)

New columns added to adata.obs:
  tier1_celltype, tier1_score
  tier2_celltype, tier2_score, tier2_full
  tier3_state, tier3_naive_score
  score_Th1, score_Th17, exhaustion_score, IDO_pathway_score,
  SIV_host_factor_score, granuloma_trafficking_score,
  AM_like_score, IFN_responsive_score, M1_IM_like_score

Run in Spyder (interactive cells with # %%) or as a standalone script.
Output dirs: analysis/phase2_annotation/, analysis/phase2_timeseries/

Author: Jake Lehle / Kaushal Lab
Date: May 2026
"""

# %% Cell 1 — Configuration and Load Phase 1 Output
# =========================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import sparse
import shutil
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === ADJUSTABLE PARAMETERS ===
SCORE_THRESHOLD = 0.1       # Minimum score to assign a cell type (Tier 1 and 2)
NAIVE_THRESHOLD = 0.1       # TCF7/LEF1 score above this = Naive T cell

# === PATHS ===
ADATA_IN = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_focal_qcd.h5ad'
ADATA_OUT = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_focal_annotated.h5ad'
OUT_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/phase2_annotation/'
TS_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/phase2_timeseries/'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TS_DIR, exist_ok=True)

# === TREATMENT CONFIG ===
# Two focal groups — no group numbers, treatment labels only
TREATMENT_CONFIG = {
    'D1MT + Late cART': {'color': '#D32F2F', 'short': 'D1MT+cART'},
    'Late cART':        {'color': '#1976D2', 'short': 'cART only'},
}
TREATMENT_PALETTE = {k: v['color'] for k, v in TREATMENT_CONFIG.items()}
TREATMENT_COL = 'treatment'
ANIMAL_COL = 'animal_id'
WEEK_COL = 'week'
SAMPLE_COL = 'sample_id'

# === LOAD ===
print("Loading Phase 1 output...")
adata = sc.read_h5ad(ADATA_IN)
adata.obs[WEEK_COL] = pd.to_numeric(adata.obs[WEEK_COL], errors='coerce').astype(int)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  Treatments: {sorted(adata.obs[TREATMENT_COL].unique())}")
print(f"  Weeks: {sorted(adata.obs[WEEK_COL].unique())}")
print(f"  Animals: {sorted(adata.obs[ANIMAL_COL].unique())}")


# %% Cell 2 — Tier 1: Broad Lineage Scoring and Assignment
# =========================================================================
# Score 8 lineages from scratch using sc.tl.score_genes().
# Winner-take-all assignment with 0.1 minimum threshold.
# =========================================================================
print("\n" + "=" * 80)
print("TIER 1: Broad Lineage Assignment (8 populations)")
print("=" * 80)

TIER1_MARKERS = {
    'T cells': {
        'markers': ['CD3D', 'CD3E'],
        'note': 'Minimal and definitive. CD3 complex is T-cell exclusive.'
    },
    'Macrophages': {
        'markers': ['CD163', 'MRC1', 'MARCO', 'MERTK', 'PPARG'],
        'note': 'Pan-macrophage. PPARG is AM master regulator.'
    },
    'Monocytes': {
        'markers': ['S100A8', 'S100A9', 'VCAN', 'CAMP', 'CD14', 'LYZ'],
        'note': 'S100A8/A9 and VCAN are the most monocyte-specific.'
    },
    'DCs': {
        'markers': ['FLT3', 'CLEC9A', 'CLEC10A', 'IL3RA', 'CLEC4C'],
        'note': 'Pan-DC markers spanning cDC1, cDC2, pDC.'
    },
    'NK cells': {
        'markers': ['KLRC3', 'KLRD1', 'GNLY', 'PRF1', 'GZMB'],
        'note': 'KLRC3/KLRD1 are NK-specific anchors.'
    },
    'B cells': {
        'markers': ['CD79A', 'MS4A1', 'CD19', 'BANK1'],
        'note': 'All excellent specificity from marker audit.'
    },
    'Mast cells': {
        'markers': ['CPA3', 'KIT', 'HPGD', 'ENPP3', 'FCER1A'],
        'note': 'CPA3/KIT are definitive mast anchors.'
    },
    'Ciliated epithelial': {
        'markers': ['DNAH9', 'HYDIN', 'MUC16', 'TSPAN1'],
        'note': 'Ciliary/dynein genes, airway lining cells in BAL.'
    },
}

tier1_score_cols = {}
for lineage, info in TIER1_MARKERS.items():
    present = [g for g in info['markers'] if g in adata.var_names]
    score_key = f"tier1_score_{lineage.replace(' ', '_')}"

    if len(present) >= 2:
        sc.tl.score_genes(adata, gene_list=present, score_name=score_key)
        tier1_score_cols[lineage] = score_key
        mean_s = adata.obs[score_key].mean()
        max_s = adata.obs[score_key].max()
        print(f"  {lineage}: {len(present)}/{len(info['markers'])} markers, "
              f"mean={mean_s:.3f}, max={max_s:.3f}")
    else:
        print(f"  SKIP {lineage}: only {len(present)} markers present")

# Assign Tier 1: argmax across lineage scores, threshold applied
lineage_names = list(tier1_score_cols.keys())
lineage_cols = [tier1_score_cols[n] for n in lineage_names]
tier1_matrix = adata.obs[lineage_cols].values

best_idx = np.argmax(tier1_matrix, axis=1)
best_score = np.max(tier1_matrix, axis=1)

tier1_labels = np.where(
    best_score >= SCORE_THRESHOLD,
    [lineage_names[i] for i in best_idx],
    'Unassigned'
)

adata.obs['tier1_celltype'] = pd.Categorical(tier1_labels)
adata.obs['tier1_score'] = best_score

print(f"\n  Tier 1 assignment:")
print(adata.obs['tier1_celltype'].value_counts().to_string())


# %% Cell 3 — Tier 2: Subtype Within Lineage
# =========================================================================
# Score subtypes using markers specific to each lineage.
# Only cells assigned at Tier 1 receive a Tier 2 label.
#
# CHANGE LOG (May 2026):
#   Added M1 IM-like macrophage subtype. These are inflammatory antigen-
#   presenting macrophages (STAT1/IRF1/TNF/IL1B/CCL2/CCL3) that represent
#   a transitional state between M2 IM-like (MRC1+) and IFN-responsive
#   (IDO1+/CXCL9+). They have entered the inflammatory activation program
#   but have not yet upregulated the IDO1 immunosuppressive axis.
#   The gene set scoring lets them compete naturally with other subtypes
#   via winner-take-all — cells with high STAT1/IRF1/TNF but low IDO1/CXCL9
#   will score higher for M1 IM-like than for IFN-responsive, and cells
#   with high MRC1 will still score highest for AM-like or M2 IM-like.
# =========================================================================
print("\n" + "=" * 80)
print("TIER 2: Subtype Assignment Within Each Lineage")
print("=" * 80)

TIER2_MARKERS = {
    'T cells': {
        'CD4': ['CD4', 'BCL11B', 'LTB', 'IL7R'],
        'CD8': ['CD8A', 'CD8B'],
    },
    'Macrophages': {
        'AM-like':        ['PPARG', 'COLEC12', 'MARCO', 'MERTK', 'MRC1', 'CD163'],
        'TREM2+':         ['TREM2', 'C1QA', 'C1QB', 'C1QC', 'MAMU-DRA', 'CD74'],
        'M2 IM-like':     ['CD163', 'MAMU-DRA', 'MAMU-DRB1'],
        'IFN-responsive': ['IDO1', 'CXCL9', 'CXCL10', 'CXCL11', 'CD274', 'SOD2'],
        'M1 IM-like':     ['STAT1', 'IRF1', 'TNF', 'IL1B', 'CCL2', 'CCL3'],
    },
    'Monocytes': {
        'Classical':     ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'CAMP'],
        'Non-classical': ['ITGAX', 'MAMU-DRA'],
    },
    'DCs': {
        'cDC1': ['CLEC9A', 'XCR1', 'BATF3', 'ZBTB46'],
        'cDC2': ['FLT3', 'CD1C', 'CLEC10A', 'CPVL'],
        'pDC':  ['IL3RA', 'CLEC4C', 'TCF4', 'BCL2', 'TLR7', 'PACSIN1', 'COBLL1'],
    },
}

tier2_score_cols = {}

for lineage, subtypes in TIER2_MARKERS.items():
    tier2_score_cols[lineage] = {}
    print(f"\n  --- {lineage} subtypes ---")
    for subtype, markers in subtypes.items():
        present = [g for g in markers if g in adata.var_names]
        score_key = (f"tier2_score_{lineage.replace(' ', '_')}_"
                     f"{subtype.replace(' ', '_').replace('-', '_').replace('+', 'pos')}")

        if len(present) >= 2:
            sc.tl.score_genes(adata, gene_list=present, score_name=score_key)
            tier2_score_cols[lineage][subtype] = score_key
            print(f"    {subtype}: {len(present)} markers scored")
        elif len(present) == 1:
            gene = present[0]
            expr = adata[:, gene].X
            if sparse.issparse(expr):
                expr = expr.toarray().flatten()
            else:
                expr = np.array(expr).flatten()
            adata.obs[score_key] = expr
            tier2_score_cols[lineage][subtype] = score_key
            print(f"    {subtype}: 1 marker ({gene}), using direct expression")
        else:
            print(f"    SKIP {subtype}: no markers present")

# Assign Tier 2 for each cell based on its Tier 1 lineage
tier2_labels = []
tier2_scores = []

for i in range(adata.n_obs):
    t1 = adata.obs['tier1_celltype'].iloc[i]

    if t1 in TIER2_MARKERS and t1 in tier2_score_cols and tier2_score_cols[t1]:
        subtypes = tier2_score_cols[t1]
        subtype_names = list(subtypes.keys())
        subtype_vals = [adata.obs[subtypes[s]].iloc[i] for s in subtype_names]
        best = np.argmax(subtype_vals)
        tier2_labels.append(subtype_names[best])
        tier2_scores.append(subtype_vals[best])
    elif t1 == 'Unassigned':
        tier2_labels.append('Unassigned')
        tier2_scores.append(0.0)
    else:
        tier2_labels.append(t1)
        tier2_scores.append(adata.obs['tier1_score'].iloc[i])

adata.obs['tier2_celltype'] = pd.Categorical(tier2_labels)
adata.obs['tier2_score'] = tier2_scores

# Combined label: "Lineage: Subtype"
adata.obs['tier2_full'] = [
    f"{t1}: {t2}" if t1 != t2 and t1 != 'Unassigned' else t1
    for t1, t2 in zip(adata.obs['tier1_celltype'], adata.obs['tier2_celltype'])
]

print(f"\n  Tier 2 assignment:")
print(adata.obs['tier2_full'].value_counts().to_string())


# %% Cell 4 — Biological Gene Set Scoring
# =========================================================================
# Score key biological pathways per cell. These feed into Tier 3
# continuous scores and are used in downstream DEG interpretation.
#
# Gene sets from Kaushal Lab publications (CHM 2021, Sharan et al. 2025).
# =========================================================================
print("\n" + "=" * 80)
print("BIOLOGICAL GENE SET SCORING")
print("=" * 80)

def score_gene_set(adata_obj, gene_list, score_name):
    """Score a gene set, handling missing genes gracefully."""
    present = [g for g in gene_list if g in adata_obj.var_names]
    if len(present) < 2:
        print(f"  SKIP {score_name}: only {len(present)}/{len(gene_list)} genes present")
        adata_obj.obs[score_name] = 0.0
        return 0
    sc.tl.score_genes(adata_obj, gene_list=present, score_name=score_name)
    print(f"  {score_name}: scored with {len(present)}/{len(gene_list)} genes")
    return len(present)

# --- Th1 polarization ---
TH1_GENES = ['TBX21', 'TNF', 'BHLHE40', 'IFNG']
score_gene_set(adata, TH1_GENES, 'score_Th1')

# --- Th17 polarization ---
TH17_GENES = ['RORC', 'STAT3', 'RORA']
score_gene_set(adata, TH17_GENES, 'score_Th17')

# --- T cell exhaustion ---
EXHAUSTION_GENES = [
    'PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'CTLA4',
    'TOX', 'ENTPD1', 'CD160', 'CD244',
]
score_gene_set(adata, EXHAUSTION_GENES, 'exhaustion_score')

# --- IDO / tryptophan catabolism pathway ---
IDO_PATHWAY_GENES = [
    'IDO1', 'IDO2', 'TDO2',
    'KMO', 'KYNU', 'HAAO',
    'IFNG', 'STAT1', 'IRF1',
    'CD274', 'CD38', 'AHR',
]
score_gene_set(adata, IDO_PATHWAY_GENES, 'IDO_pathway_score')

# --- SIV host factors (receptor, co-receptors, restriction factors) ---
SIV_HOST_GENES = [
    'CD4', 'CCR5', 'CXCR4',
    'APOBEC3G', 'APOBEC3F',
    'TRIM5', 'BST2', 'SAMHD1',
    'IFITM1', 'IFITM2', 'IFITM3', 'MX2',
]
score_gene_set(adata, SIV_HOST_GENES, 'SIV_host_factor_score')

# --- Granuloma trafficking ---
GRANULOMA_GENES = [
    'CCR7', 'CCL19', 'CCL21',
    'SELL', 'CXCL13', 'LTB', 'CD27',
]
score_gene_set(adata, GRANULOMA_GENES, 'granuloma_trafficking_score')

# --- Macrophage polarization subscores ---
AM_GENES = ['MRC1', 'CD163', 'MARCO', 'MERTK', 'TREM2', 'C1QA', 'C1QB', 'C1QC']
score_gene_set(adata, AM_GENES, 'AM_like_score')

IFN_GENES = [
    'CXCL9', 'CXCL10', 'CXCL11', 'IDO1', 'CD274',
    'GBP1', 'GBP2', 'GBP5', 'STAT1', 'IRF1', 'IRF7',
]
score_gene_set(adata, IFN_GENES, 'IFN_responsive_score')

# --- M1 IM-like inflammatory macrophage score ---
# Inflammatory antigen-presenting macrophages: early IFN signaling (STAT1/IRF1)
# + pro-inflammatory cytokines (TNF/IL1B) + chemokines (CCL2/CCL3),
# but WITHOUT the IDO1/CXCL9-11 immunosuppressive program.
# Matches the Tier 2 M1 IM-like gene set for consistency.
M1_IM_GENES = ['STAT1', 'IRF1', 'TNF', 'IL1B', 'CCL2', 'CCL3']
score_gene_set(adata, M1_IM_GENES, 'M1_IM_like_score')


# %% Cell 5 — Tier 3: T Cell Functional States
# =========================================================================
# Discrete: Naive (TCF7/LEF1 > threshold) vs Effector
# Continuous scores from Cell 4 are already attached.
# =========================================================================
print("\n" + "=" * 80)
print("TIER 3: T Cell Functional States")
print("=" * 80)

NAIVE_MARKERS = ['TCF7', 'LEF1']
naive_present = [g for g in NAIVE_MARKERS if g in adata.var_names]

if len(naive_present) >= 2:
    sc.tl.score_genes(adata, gene_list=naive_present, score_name='tier3_naive_score')
    print(f"  Naive score computed with {len(naive_present)} markers")
elif len(naive_present) == 1:
    gene = naive_present[0]
    expr = adata[:, gene].X
    if sparse.issparse(expr):
        expr = expr.toarray().flatten()
    else:
        expr = np.array(expr).flatten()
    adata.obs['tier3_naive_score'] = expr
    print(f"  Naive score from single marker: {gene}")
else:
    adata.obs['tier3_naive_score'] = 0.0
    print("  WARNING: No naive markers found")

# Assign Tier 3 for T cells
t_cell_mask = adata.obs['tier1_celltype'] == 'T cells'
print(f"  Total T cells: {t_cell_mask.sum():,}")

tier3_labels = pd.Series('N/A', index=adata.obs.index)

for i in adata.obs.index[t_cell_mask]:
    t2 = adata.obs.loc[i, 'tier2_celltype']
    naive_score = adata.obs.loc[i, 'tier3_naive_score']

    if naive_score >= NAIVE_THRESHOLD:
        tier3_labels[i] = f'{t2} Naive'
    else:
        tier3_labels[i] = f'{t2} Effector'

adata.obs['tier3_state'] = pd.Categorical(tier3_labels)

print(f"\n  Tier 3 T cell states:")
print(adata.obs.loc[t_cell_mask, 'tier3_state'].value_counts().to_string())

# Summary of continuous scores in CD4 Effector
cd4_eff_mask = adata.obs['tier3_state'] == 'CD4 Effector'
print(f"\n  CD4 Effector cells: {cd4_eff_mask.sum():,}")

continuous_scores = {
    'Th1':                   'score_Th1',
    'Th17':                  'score_Th17',
    'Exhaustion':            'exhaustion_score',
    'Granuloma trafficking': 'granuloma_trafficking_score',
    'SIV host factors':      'SIV_host_factor_score',
    'IDO pathway':           'IDO_pathway_score',
}

if cd4_eff_mask.sum() > 0:
    print(f"\n  Continuous scores within CD4 Effector:")
    for label, col in continuous_scores.items():
        if col in adata.obs.columns:
            vals = adata.obs.loc[cd4_eff_mask, col]
            print(f"    {label}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
                  f"range=[{vals.min():.3f}, {vals.max():.3f}]")


# %% Cell 6 — UMAPs
# =========================================================================
print("\n" + "=" * 80)
print("UMAPS — Tiered Cell Type Annotation")
print("=" * 80)

# --- 6A: Tier 1 overview ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
sc.pl.umap(adata, color='tier1_celltype', ax=axes[0], show=False,
           title='Tier 1: Broad Lineages', frameon=False, legend_loc='right margin')
sc.pl.umap(adata, color=TREATMENT_COL, ax=axes[1], show=False,
           title='Treatment', frameon=False, legend_loc='right margin',
           palette=TREATMENT_PALETTE)
sc.pl.umap(adata, color=WEEK_COL, ax=axes[2], show=False,
           title='Week Post-Infection', frameon=False, cmap='viridis')
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(OUT_DIR, 'umap_tier1_overview.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(OUT_DIR, 'umap_tier1_overview.png'), bbox_inches='tight', dpi=300)
print("  Saved: umap_tier1_overview")
plt.close(fig)

# --- 6B: Tier 2 breakout UMAPs per lineage ---
breakout_lineages = {
    'T cells':     {'extra': 'tier3_state'},
    'Macrophages': {'extra': None},
    'Monocytes':   {'extra': None},
    'DCs':         {'extra': None},
}

for lineage, config in breakout_lineages.items():
    mask = adata.obs['tier1_celltype'] == lineage
    n_cells = mask.sum()
    if n_cells < 100:
        print(f"  Skip {lineage}: only {n_cells} cells")
        continue

    sub = adata[mask].copy()
    safe_name = lineage.replace(' ', '_').lower()

    panels = ['tier2_celltype']
    panel_titles = [f'{lineage}: Tier 2 Subtypes']
    if config['extra']:
        panels.append(config['extra'])
        panel_titles.append(f'{lineage}: Tier 3 States')
    panels.append(TREATMENT_COL)
    panel_titles.append(f'{lineage}: Treatment')

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    for i, (panel_col, panel_title) in enumerate(zip(panels, panel_titles)):
        kw = {}
        if panel_col == TREATMENT_COL:
            kw['palette'] = TREATMENT_PALETTE
        sc.pl.umap(sub, color=panel_col, ax=axes[i], show=False,
                   title=panel_title, frameon=False, legend_loc='right margin', **kw)

    plt.suptitle(f'{lineage} Breakout ({n_cells:,} cells)', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, f'umap_breakout_{safe_name}.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, f'umap_breakout_{safe_name}.png'),
                bbox_inches='tight', dpi=300)
    print(f"  Saved: umap_breakout_{safe_name}")
    plt.close(fig)

# --- 6C: Full Tier 2 UMAP ---
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(adata, color='tier2_full', ax=ax, show=False,
           title='All Tier 2 Subtypes', frameon=False,
           legend_loc='right margin', legend_fontsize=7)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(OUT_DIR, 'umap_tier2_full.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(OUT_DIR, 'umap_tier2_full.png'), bbox_inches='tight', dpi=300)
print("  Saved: umap_tier2_full")
plt.close(fig)

# --- 6D: Biological score UMAPs ---
sig_cols = ['IDO_pathway_score', 'exhaustion_score', 'SIV_host_factor_score',
            'AM_like_score', 'IFN_responsive_score', 'M1_IM_like_score',
            'granuloma_trafficking_score']
sig_present = [s for s in sig_cols if s in adata.obs.columns]

if sig_present:
    ncols = min(3, len(sig_present))
    nrows = int(np.ceil(len(sig_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    for i, sig in enumerate(sig_present):
        sc.pl.umap(adata, color=sig, ax=axes[i], show=False,
                   title=sig.replace('_', ' '), cmap='RdYlBu_r', frameon=False)
    for i in range(len(sig_present), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(OUT_DIR, 'umap_biological_scores.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'umap_biological_scores.png'),
                bbox_inches='tight', dpi=300)
    print("  Saved: umap_biological_scores")
    plt.close(fig)


# %% Cell 7 — Dotplots: Marker Gene Specificity
# =========================================================================
print("\n" + "=" * 80)
print("DOTPLOTS — Marker Gene Specificity")
print("=" * 80)

TIER1_DOTPLOT_MARKERS = {
    'T cells':             ['CD3D', 'CD3E'],
    'Macrophages':         ['CD163', 'MRC1', 'MARCO', 'MERTK', 'PPARG'],
    'Monocytes':           ['S100A8', 'S100A9', 'VCAN', 'CD14', 'LYZ'],
    'DCs':                 ['FLT3', 'CLEC9A', 'CLEC10A', 'IL3RA'],
    'NK cells':            ['KLRC3', 'KLRD1', 'GNLY', 'GZMB'],
    'B cells':             ['CD79A', 'MS4A1', 'CD19', 'BANK1'],
    'Mast cells':          ['CPA3', 'KIT', 'HPGD', 'ENPP3'],
    'Ciliated epithelial': ['DNAH9', 'HYDIN', 'MUC16', 'TSPAN1'],
}

TIER2_DOTPLOT_MARKERS = {
    'T cells: CD4':                  ['CD4', 'BCL11B', 'LTB', 'IL7R'],
    'T cells: CD8':                  ['CD8A', 'CD8B'],
    'Macrophages: AM-like':          ['PPARG', 'COLEC12', 'MARCO', 'MERTK'],
    'Macrophages: TREM2+':           ['TREM2', 'C1QA', 'C1QB', 'CD74'],
    'Macrophages: M2 IM-like':         ['CD163', 'MAMU-DRA', 'MAMU-DRB1'],
    'Macrophages: IFN-responsive':   ['IDO1', 'CXCL9', 'CXCL10', 'CD274'],
    'Macrophages: M1 IM-like':       ['STAT1', 'IRF1', 'TNF', 'IL1B', 'CCL2', 'CCL3'],
    'Monocytes: Classical':          ['CD14', 'S100A8', 'S100A9', 'VCAN'],
    'Monocytes: Non-classical':      ['ITGAX', 'MAMU-DRA'],
    'DCs: cDC1':                     ['CLEC9A', 'XCR1', 'BATF3'],
    'DCs: cDC2':                     ['CD1C', 'CLEC10A', 'CPVL'],
    'DCs: pDC':                      ['IL3RA', 'CLEC4C', 'TCF4', 'PACSIN1'],
    'NK cells':                      ['KLRC3', 'KLRD1', 'GNLY', 'PRF1', 'GZMB'],
    'B cells':                       ['CD79A', 'MS4A1', 'CD19', 'BANK1'],
    'Mast cells':                    ['CPA3', 'KIT', 'HPGD', 'ENPP3'],
    'Ciliated epithelial':           ['DNAH9', 'HYDIN', 'MUC16', 'TSPAN1'],
}

# --- Tier 1 dotplot ---
print("\n  Generating Tier 1 marker dotplot...")
t1_order = list(TIER1_DOTPLOT_MARKERS.keys())
t1_mask = adata.obs['tier1_celltype'].isin(t1_order)
adata_t1 = adata[t1_mask].copy()

t1_gene_groups = {}
for ct, markers in TIER1_DOTPLOT_MARKERS.items():
    present = [g for g in markers if g in adata_t1.var_names]
    if present:
        t1_gene_groups[ct] = present

sc.pl.dotplot(adata_t1, var_names=t1_gene_groups, groupby='tier1_celltype',
              categories_order=t1_order, standard_scale='var',
              save='_tier1_markers.png', show=True)
sc.pl.dotplot(adata_t1, var_names=t1_gene_groups, groupby='tier1_celltype',
              categories_order=t1_order, standard_scale='var',
              save='_tier1_markers.pdf', show=False)
plt.close('all')

for ext in ['png', 'pdf']:
    src = os.path.join(sc.settings.figdir, f'dotplot_tier1_markers.{ext}')
    dst = os.path.join(OUT_DIR, f'dotplot_tier1_markers.{ext}')
    if os.path.exists(src):
        shutil.copy2(src, dst)
print("  Saved: dotplot_tier1_markers")

# --- Tier 2 dotplot ---
print("\n  Generating Tier 2 marker dotplot...")

has_tier2 = {'T cells', 'Macrophages', 'Monocytes', 'DCs'}
adata.obs['dotplot_group'] = adata.obs.apply(
    lambda row: row['tier2_full'] if row['tier1_celltype'] in has_tier2
    else row['tier1_celltype'], axis=1
)

t2_order = list(TIER2_DOTPLOT_MARKERS.keys())
t2_mask = adata.obs['dotplot_group'].isin(t2_order)
adata_t2 = adata[t2_mask].copy()
print(f"  Cells included: {t2_mask.sum():,} (excluded {(~t2_mask).sum():,} Unassigned)")

t2_gene_groups = {}
for ct, markers in TIER2_DOTPLOT_MARKERS.items():
    present = [g for g in markers if g in adata_t2.var_names]
    if present:
        t2_gene_groups[ct] = present

sc.pl.dotplot(adata_t2, var_names=t2_gene_groups, groupby='dotplot_group',
              categories_order=t2_order, standard_scale='var',
              save='_tier2_markers.png', show=True)
sc.pl.dotplot(adata_t2, var_names=t2_gene_groups, groupby='dotplot_group',
              categories_order=t2_order, standard_scale='var',
              save='_tier2_markers.pdf', show=False)
plt.close('all')

for ext in ['png', 'pdf']:
    src = os.path.join(sc.settings.figdir, f'dotplot_tier2_markers.{ext}')
    dst = os.path.join(OUT_DIR, f'dotplot_tier2_markers.{ext}')
    if os.path.exists(src):
        shutil.copy2(src, dst)
print("  Saved: dotplot_tier2_markers")


# %% Cell 8 — Diagnostic Tables
# =========================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC TABLES")
print("=" * 80)

# --- 8A: Tier 1 proportions per treatment × week ---
print("\n--- Tier 1 cell type proportions (%) ---")
t1_props = pd.crosstab(
    [adata.obs[TREATMENT_COL], adata.obs[WEEK_COL]],
    adata.obs['tier1_celltype'],
    normalize='index'
) * 100
print(t1_props.round(1).to_string())
t1_props.to_csv(os.path.join(OUT_DIR, 'tier1_proportions.csv'))

# --- 8B: Tier 2 proportions per treatment × week ---
print("\n\n--- Tier 2 cell type proportions (%) ---")
t2_props = pd.crosstab(
    [adata.obs[TREATMENT_COL], adata.obs[WEEK_COL]],
    adata.obs['tier2_full'],
    normalize='index'
) * 100
print(t2_props.round(1).to_string())
t2_props.to_csv(os.path.join(OUT_DIR, 'tier2_proportions.csv'))

# --- 8C: CD4 Effector continuous scores per treatment × week ---
print("\n\n--- CD4 Effector continuous scores (mean ± std) ---")
cd4_eff_obs = adata.obs.loc[adata.obs['tier3_state'] == 'CD4 Effector']

score_cols_present = [c for _, c in continuous_scores.items() if c in cd4_eff_obs.columns]
if score_cols_present and len(cd4_eff_obs) > 0:
    cd4_scores = (
        cd4_eff_obs
        .groupby([TREATMENT_COL, WEEK_COL])[score_cols_present]
        .agg(['mean', 'std', 'count'])
        .round(3)
    )
    print(cd4_scores.to_string())
    cd4_scores.to_csv(os.path.join(OUT_DIR, 'cd4_effector_scores.csv'))

# --- 8D: Key population summary (per-animal mean ± SEM) ---
print("\n\n--- Key population summary (per-animal mean ± SEM) ---")

key_pops = ['T cells: CD4', 'T cells: CD8',
            'Macrophages: AM-like', 'Macrophages: IFN-responsive',
            'Macrophages: TREM2+', 'Macrophages: M2 IM-like',
            'Macrophages: M1 IM-like']

pop_summary_rows = []

for pop in key_pops:
    pop_mask = adata.obs['tier2_full'] == pop
    if pop_mask.sum() == 0:
        continue

    animal_props = (
        adata.obs
        .assign(is_pop=pop_mask.astype(int))
        .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL], observed=True)
        .agg(total=('is_pop', 'size'), pop_count=('is_pop', 'sum'))
        .reset_index()
    )
    animal_props['pct'] = animal_props['pop_count'] / animal_props['total'] * 100

    agg = (
        animal_props
        .groupby([TREATMENT_COL, WEEK_COL])['pct']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    agg['sem'] = (agg['std'] / np.sqrt(agg['count'])).round(2)
    agg['mean'] = agg['mean'].round(2)
    agg = agg.sort_values([TREATMENT_COL, WEEK_COL])

    print(f"\n  {pop} (% of total cells):")
    for _, row in agg.iterrows():
        print(f"    {row[TREATMENT_COL]:25s} W{int(row[WEEK_COL]):2d}: "
              f"{row['mean']:6.2f} ± {row['sem']:5.2f}% "
              f"(n={int(row['count'])} animals)")

    agg['population'] = pop
    pop_summary_rows.append(agg)

if pop_summary_rows:
    df_pop_summary = pd.concat(pop_summary_rows, ignore_index=True)
    df_pop_summary.to_csv(os.path.join(OUT_DIR, 'key_population_summary.csv'), index=False)

# --- 8E: Annotation sanity check ---
print("\n\n--- Annotation sanity check ---")
n_unassigned = (adata.obs['tier1_celltype'] == 'Unassigned').sum()
pct_unassigned = n_unassigned / adata.n_obs * 100
print(f"  Unassigned cells: {n_unassigned:,} ({pct_unassigned:.1f}%)")

for lineage in TIER1_MARKERS:
    n = (adata.obs['tier1_celltype'] == lineage).sum()
    pct = n / adata.n_obs * 100
    flag = ""
    if pct < 0.1 and lineage not in ['Ciliated epithelial', 'Mast cells']:
        flag = " *** LOW"
    elif pct > 60:
        flag = " *** DOMINANT"
    print(f"  {lineage}: {n:>8,} ({pct:5.1f}%){flag}")

print(f"\n  Diagnostic tables saved to {OUT_DIR}")


# %% Cell 9 — Proportion Trajectories and Fold-Change
# =========================================================================
print("\n" + "=" * 80)
print("TIME SERIES — Proportions and Fold-Change")
print("=" * 80)

# --- Helper functions ---
def compute_proportions(adata_obj, celltype_col):
    """Compute per-animal proportions, aggregate to mean ± SEM."""
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

    agg = (
        counts
        .groupby([TREATMENT_COL, WEEK_COL, celltype_col], observed=True)['proportion']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    agg['sem'] = agg['std'] / np.sqrt(agg['count'])
    agg[WEEK_COL] = pd.to_numeric(agg[WEEK_COL], errors='coerce')
    return agg


def plot_trajectories(agg, celltypes, celltype_col, title, filename, ncols=3):
    """Plot proportion trajectory subplots."""
    n = len(celltypes)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    for idx, ct in enumerate(celltypes):
        ax = axes[idx // ncols, idx % ncols]
        ct_data = agg[agg[celltype_col] == ct]

        for treatment, cfg in TREATMENT_CONFIG.items():
            grp_data = ct_data[ct_data[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
            if grp_data.empty:
                continue
            weeks = grp_data[WEEK_COL].values.astype(float)
            means = grp_data['mean'].values.astype(float) * 100
            sems = np.nan_to_num(grp_data['sem'].values.astype(float) * 100, nan=0.0)
            valid = ~np.isnan(means)
            if valid.sum() == 0:
                continue

            ax.plot(weeks[valid], means[valid], marker='o', markersize=5,
                    linewidth=1.8, color=cfg['color'], label=treatment, zorder=3)
            ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                        ecolor=cfg['color'], capsize=3, capthick=1, zorder=2)

        ax.set_title(ct, fontsize=10, fontweight='bold')
        ax.set_xlabel('Week', fontsize=9)
        ax.set_ylabel('% of cells', fontsize=9)
        ax.set_xlim(3, 17)
        ax.xaxis.set_major_locator(mticker.FixedLocator([5, 11, 15]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.show()
    fig.savefig(os.path.join(TS_DIR, f'{filename}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(TS_DIR, f'{filename}.png'), bbox_inches='tight', dpi=300)
    print(f"  Saved: {filename}")
    plt.close(fig)


# --- Tier 1 trajectories ---
print("\n  Computing Tier 1 proportions...")
agg_t1 = compute_proportions(adata, 'tier1_celltype')
t1_types = [t for t in ['T cells', 'Macrophages', 'Monocytes', 'DCs',
                         'NK cells', 'B cells', 'Mast cells', 'Ciliated epithelial']
            if t in agg_t1['tier1_celltype'].unique()]
plot_trajectories(agg_t1, t1_types, 'tier1_celltype',
                  'Tier 1 — Broad Lineage Proportions Over Time (mean ± SEM)',
                  'tier1_lineage_trajectories')

# --- Tier 2 trajectories ---
print("\n  Computing Tier 2 proportions...")
agg_t2 = compute_proportions(adata, 'tier2_full')

mac_types = sorted([t for t in agg_t2['tier2_full'].unique() if 'Macrophage' in str(t)])
if mac_types:
    plot_trajectories(agg_t2, mac_types, 'tier2_full',
                      'Tier 2 — Macrophage Subtypes Over Time',
                      'tier2_macrophage_trajectories', ncols=3)

tcell_types = sorted([t for t in agg_t2['tier2_full'].unique() if 'T cells' in str(t)])
if tcell_types:
    plot_trajectories(agg_t2, tcell_types, 'tier2_full',
                      'Tier 2 — T Cell Subtypes Over Time',
                      'tier2_tcell_trajectories', ncols=2)

dc_types = sorted([t for t in agg_t2['tier2_full'].unique() if 'DC' in str(t)])
if dc_types:
    plot_trajectories(agg_t2, dc_types, 'tier2_full',
                      'Tier 2 — DC Subtypes Over Time',
                      'tier2_dc_trajectories', ncols=3)

# --- Tier 3 T cell state trajectories ---
print("\n  Computing Tier 3 proportions (T cells only)...")
t_only = adata[adata.obs['tier1_celltype'] == 'T cells'].copy()
agg_t3 = compute_proportions(t_only, 'tier3_state')
t3_types = sorted([t for t in agg_t3['tier3_state'].unique() if t != 'N/A'])
if t3_types:
    plot_trajectories(agg_t3, t3_types, 'tier3_state',
                      'Tier 3 — T Cell States Over Time (% of T cells)',
                      'tier3_tcell_states', ncols=2)

# --- CD4 Effector continuous score trajectories ---
print("\n  Computing CD4 Effector continuous scores over time...")
cd4_eff = adata[adata.obs['tier3_state'] == 'CD4 Effector'].copy()

if cd4_eff.n_obs > 100:
    score_panels = [
        ('score_Th1', 'Th1 Score'),
        ('score_Th17', 'Th17 Score'),
        ('exhaustion_score', 'Exhaustion Score'),
        ('granuloma_trafficking_score', 'Granuloma Trafficking'),
        ('SIV_host_factor_score', 'SIV Host Factors'),
        ('IDO_pathway_score', 'IDO Pathway'),
    ]
    score_panels = [(c, l) for c, l in score_panels if c in cd4_eff.obs.columns]

    ncols = min(3, len(score_panels))
    nrows = int(np.ceil(len(score_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('CD4 Effector — Continuous Functional Scores Over Time',
                 fontsize=14, fontweight='bold', y=1.01)

    for idx, (score_col, label) in enumerate(score_panels):
        ax = axes[idx // ncols, idx % ncols]

        # Per-animal means, then aggregate
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
            grp_data = score_agg[score_agg[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
            if grp_data.empty:
                continue
            weeks = grp_data[WEEK_COL].values.astype(float)
            means = grp_data['mean'].values.astype(float)
            sems = np.nan_to_num(grp_data['sem'].values.astype(float), nan=0.0)
            valid = ~np.isnan(means)
            if valid.sum() == 0:
                continue

            ax.plot(weeks[valid], means[valid], marker='o', markersize=5,
                    linewidth=1.8, color=cfg['color'], label=treatment, zorder=3)
            ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                        ecolor=cfg['color'], capsize=3, capthick=1, zorder=2)

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Mean score')
        ax.set_xlim(3, 17)
        ax.xaxis.set_major_locator(mticker.FixedLocator([5, 11, 15]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(len(score_panels), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()
    fig.savefig(os.path.join(TS_DIR, 'cd4_effector_continuous_scores.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(TS_DIR, 'cd4_effector_continuous_scores.png'),
                bbox_inches='tight', dpi=300)
    print("  Saved: cd4_effector_continuous_scores")
    plt.close(fig)

# --- Fold-change from Week 5 (proportion-based) ---
print("\n  Computing proportion fold-change from Week 5...")

FC_POPULATIONS = {
    'Tier 1': {
        'col': 'tier1_celltype',
        'pops': ['T cells', 'Macrophages', 'Monocytes'],
    },
    'Tier 2 Macrophages': {
        'col': 'tier2_full',
        'pops': ['Macrophages: AM-like', 'Macrophages: TREM2+',
                 'Macrophages: M2 IM-like', 'Macrophages: IFN-responsive',
                 'Macrophages: M1 IM-like'],
    },
    'Tier 2 T cells': {
        'col': 'tier2_full',
        'pops': ['T cells: CD4', 'T cells: CD8'],
    },
}

all_fc_results = []

for panel_name, panel_info in FC_POPULATIONS.items():
    celltype_col = panel_info['col']
    populations = panel_info['pops']
    panel_fc = []

    for pop in populations:
        pop_mask = adata.obs[celltype_col] == pop

        animal_data = (
            adata.obs
            .assign(is_pop=pop_mask.astype(int))
            .groupby([TREATMENT_COL, ANIMAL_COL, WEEK_COL], observed=True)
            .agg(total=('is_pop', 'size'), pop_count=('is_pop', 'sum'))
            .reset_index()
        )
        animal_data['proportion'] = animal_data['pop_count'] / animal_data['total']

        baseline = (
            animal_data[animal_data[WEEK_COL] == 5]
            [[TREATMENT_COL, ANIMAL_COL, 'proportion']]
            .rename(columns={'proportion': 'baseline_prop'})
        )

        fc = animal_data.merge(baseline, on=[TREATMENT_COL, ANIMAL_COL], how='inner')
        fc['fold_change'] = np.where(
            fc['baseline_prop'] > 0,
            fc['proportion'] / fc['baseline_prop'],
            np.nan
        )

        if fc.empty:
            continue

        fc_agg = (
            fc.groupby([TREATMENT_COL, WEEK_COL])['fold_change']
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        fc_agg['sem'] = fc_agg['std'] / np.sqrt(fc_agg['count'])
        fc_agg[WEEK_COL] = pd.to_numeric(fc_agg[WEEK_COL], errors='coerce')
        fc_agg['population'] = pop
        fc_agg['panel'] = panel_name

        panel_fc.append(fc_agg)
        all_fc_results.append(fc_agg)

    if not panel_fc:
        continue

    df_panel = pd.concat(panel_fc, ignore_index=True)
    pops_in_panel = [p for p in populations if p in df_panel['population'].unique()]
    n = len(pops_in_panel)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f'{panel_name} — Proportion Fold-Change from Week 5 (mean ± SEM)',
                 fontsize=13, fontweight='bold', y=1.02)

    for pidx, pop in enumerate(pops_in_panel):
        ax = axes[pidx // ncols, pidx % ncols]
        pop_data = df_panel[df_panel['population'] == pop]

        for treatment, cfg in TREATMENT_CONFIG.items():
            grp_data = pop_data[pop_data[TREATMENT_COL] == treatment].sort_values(WEEK_COL)
            if grp_data.empty:
                continue
            weeks = grp_data[WEEK_COL].values.astype(float)
            means = grp_data['mean'].values.astype(float)
            sems = np.nan_to_num(grp_data['sem'].values.astype(float), nan=0.0)
            valid = ~np.isnan(means)
            if valid.sum() == 0:
                continue

            ax.plot(weeks[valid], means[valid], marker='o', markersize=6,
                    linewidth=2, color=cfg['color'], label=treatment, zorder=3)
            ax.errorbar(weeks[valid], means[valid], yerr=sems[valid], fmt='none',
                        ecolor=cfg['color'], capsize=4, capthick=1.2, zorder=2)

        ax.axhline(1.0, color='grey', linestyle=':', linewidth=1, alpha=0.6)
        short_label = pop.split(': ')[-1] if ': ' in pop else pop
        ax.set_title(short_label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Week post-infection')
        ax.set_ylabel('Fold-change from W5')
        ax.set_xlim(3, 17)
        ax.xaxis.set_major_locator(mticker.FixedLocator([5, 11, 15]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for pidx in range(n, nrows * ncols):
        axes[pidx // ncols, pidx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06), frameon=True)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.show()

    safe_panel = panel_name.lower().replace(' ', '_')
    fig.savefig(os.path.join(TS_DIR, f'foldchange_{safe_panel}.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(TS_DIR, f'foldchange_{safe_panel}.png'),
                bbox_inches='tight', dpi=300)
    print(f"  Saved: foldchange_{safe_panel}")
    plt.close(fig)

# Export fold-change data
if all_fc_results:
    df_all_fc = pd.concat(all_fc_results, ignore_index=True)
    df_all_fc.to_csv(os.path.join(TS_DIR, 'foldchange_stats.csv'), index=False)
    print("  Saved: foldchange_stats.csv")

    print("\n--- Fold-change summary ---")
    for _, row in df_all_fc.sort_values(['population', TREATMENT_COL, WEEK_COL]).iterrows():
        if pd.isna(row['mean']):
            continue
        print(f"  {row['population']:30s} | {row[TREATMENT_COL]:25s} | "
              f"W{int(row[WEEK_COL]):2d}: {row['mean']:.2f}x ± {row['sem']:.2f} "
              f"(n={int(row['count'])})")


# %% Cell 10 — Save Annotated AnnData
# =========================================================================
print("\n" + "=" * 80)
print("SAVING ANNOTATED ADATA")
print("=" * 80)

# Save annotated adata
adata.write_h5ad(ADATA_OUT)
print(f"  Saved: {ADATA_OUT}")
print(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Export tier count summaries
tier1_summary = adata.obs['tier1_celltype'].value_counts().to_frame('n_cells')
tier1_summary['pct'] = (tier1_summary['n_cells'] / adata.n_obs * 100).round(1)
tier1_summary.to_csv(os.path.join(OUT_DIR, 'tier1_counts.csv'))

tier2_summary = adata.obs['tier2_full'].value_counts().to_frame('n_cells')
tier2_summary['pct'] = (tier2_summary['n_cells'] / adata.n_obs * 100).round(1)
tier2_summary.to_csv(os.path.join(OUT_DIR, 'tier2_counts.csv'))

if t_cell_mask.any():
    tier3_summary = adata.obs.loc[t_cell_mask, 'tier3_state'].value_counts().to_frame('n_cells')
    tier3_summary.to_csv(os.path.join(OUT_DIR, 'tier3_tcell_counts.csv'))

# Export new columns list for reference
new_cols = [c for c in adata.obs.columns if c.startswith(('tier', 'score_', 'exhaustion',
            'IDO_', 'SIV_', 'granuloma', 'AM_', 'IFN_', 'M1_', 'dotplot'))]
print(f"\n  New annotation columns added ({len(new_cols)}):")
for c in sorted(new_cols):
    print(f"    {c}")

print(f"\n{'=' * 80}")
print("PHASE 2 COMPLETE")
print(f"{'=' * 80}")
print(f"  Annotation output: {OUT_DIR}")
print(f"  Time series output: {TS_DIR}")
print(f"  Annotated AnnData: {ADATA_OUT}")
print(f"  Treatments: {sorted(adata.obs[TREATMENT_COL].unique())}")
print(f"  Tier 1 lineages: {adata.obs['tier1_celltype'].nunique()}")
print(f"  Tier 2 subtypes: {adata.obs['tier2_full'].nunique()}")
print(f"  Tier 3 T cell states: {adata.obs.loc[t_cell_mask, 'tier3_state'].nunique()}")
print(f"\n  Next: Phase 3 — DEG, Pathway Analysis, and SIV Profiling")
