#!/usr/bin/env python3
"""
Phase 1 — Load, QC, and Embed
===============================
Standalone pipeline for Rhesus Mtb/SIV coinfection BAL scRNA-seq.

Reads cellranger count matrices for two late cART treatment groups,
performs mitochondrial and expression-based QC, normalizes, batch-corrects
(BBKNN), and produces a UMAP embedding ready for annotation.

Treatment groups:
  - D1MT + Late cART : IDO inhibitor (1-methyl-D-tryptophan) + late cART
  - Late cART        : Late cART only (no IDO inhibition)

Custom reference: Mmul10 (rhesus) + MtbCDC1551 + SIVmac239

Gene identification notes:
  - Rhesus MT genes use KEG06_ locus tags (NOT MT- prefix)
  - SIV genes: gene_id = "gene-gag", var_name = "gag" (no prefix on var_names)
  - Mtb genes: gene_id = "MT_RS#####", var_name = "MT_RS#####" or named (dnaA)

Run in Spyder (interactive cells with # %%) or as a standalone script.
Output dir: analysis/phase1_qc/

Author: Jake Lehle / Kaushal Lab
Date: May 2026
"""

# %% Cell 1 — Configuration, Paths, and Sample Registry
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === ADJUSTABLE QC PARAMETERS ===
MT_THRESHOLD = None
MIN_GENES = 200
MIN_CELLS_PER_GENE = 20
MAD_MULTIPLIER = 5
COMPROMISED_MT_FLOOR = 20
TOP_N_GENE_PCT = 20

# === PATHOGEN GENE IDENTIFICATION ===
# SIV genes have no var_names prefix (stored as gag, pol, etc.)
# Must use explicit list or gene_ids column (gene-gag, gene-pol)
SIV_GENE_LIST = ['gag', 'pol', 'vif', 'vpx', 'vpr', 'tat', 'rev', 'env', 'nef']

# === PATHS ===
CELLRANGER_BASE = '/master/jlehle/WORKING/SC/fastq/Rhesus/cellranger_counts'
OUT_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/phase1_qc/'
ADATA_OUT = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_focal_qcd.h5ad'
os.makedirs(OUT_DIR, exist_ok=True)

# === SAMPLE REGISTRY ===
SAMPLE_REGISTRY = [
    {'sample_id': '44129-Week5',  'animal_id': '44129', 'week': 5,  'treatment': 'D1MT + Late cART'},
    {'sample_id': '44129-Week11', 'animal_id': '44129', 'week': 11, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44129-Week15', 'animal_id': '44129', 'week': 15, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44132-Week5',  'animal_id': '44132', 'week': 5,  'treatment': 'D1MT + Late cART'},
    {'sample_id': '44132-Week11', 'animal_id': '44132', 'week': 11, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44132-Week15', 'animal_id': '44132', 'week': 15, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44137-Week5',  'animal_id': '44137', 'week': 5,  'treatment': 'D1MT + Late cART'},
    {'sample_id': '44137-Week11', 'animal_id': '44137', 'week': 11, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44137-Week15', 'animal_id': '44137', 'week': 15, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44154-Week5',  'animal_id': '44154', 'week': 5,  'treatment': 'D1MT + Late cART'},
    {'sample_id': '44154-Week11', 'animal_id': '44154', 'week': 11, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44154-Week15', 'animal_id': '44154', 'week': 15, 'treatment': 'D1MT + Late cART'},
    {'sample_id': '44120-Week5',  'animal_id': '44120', 'week': 5,  'treatment': 'Late cART'},
    {'sample_id': '44120-Week11', 'animal_id': '44120', 'week': 11, 'treatment': 'Late cART'},
    {'sample_id': '44120-Week15', 'animal_id': '44120', 'week': 15, 'treatment': 'Late cART'},
    {'sample_id': '44156-Week5',  'animal_id': '44156', 'week': 5,  'treatment': 'Late cART'},
    {'sample_id': '44156-Week11', 'animal_id': '44156', 'week': 11, 'treatment': 'Late cART'},
    {'sample_id': '44156-Week15', 'animal_id': '44156', 'week': 15, 'treatment': 'Late cART'},
    {'sample_id': 'KV48-Week5',   'animal_id': 'KV48',  'week': 5,  'treatment': 'Late cART'},
    {'sample_id': 'KV48-Week15',  'animal_id': 'KV48',  'week': 15, 'treatment': 'Late cART'},
    {'sample_id': 'LE24-Week5',   'animal_id': 'LE24',  'week': 5,  'treatment': 'Late cART'},
    {'sample_id': 'LE24-Week15',  'animal_id': 'LE24',  'week': 15, 'treatment': 'Late cART'},
]

print(f"Sample registry: {len(SAMPLE_REGISTRY)} samples")
print(f"  D1MT + Late cART: {sum(1 for s in SAMPLE_REGISTRY if s['treatment'] == 'D1MT + Late cART')}")
print(f"  Late cART:        {sum(1 for s in SAMPLE_REGISTRY if s['treatment'] == 'Late cART')}")


# %% Cell 2 — Read cellranger matrices and build combined AnnData
print("\n" + "=" * 80)
print("LOADING CELLRANGER COUNT MATRICES")
print("=" * 80)

adatas = []
load_summary = []

for entry in SAMPLE_REGISTRY:
    sid = entry['sample_id']
    mtx_path = os.path.join(CELLRANGER_BASE, sid, 'outs', 'filtered_feature_bc_matrix')

    if not os.path.exists(mtx_path):
        print(f"  *** MISSING: {mtx_path}")
        load_summary.append({**entry, 'n_cells': 0, 'n_genes': 0, 'status': 'MISSING'})
        continue

    try:
        a = sc.read_10x_mtx(mtx_path, var_names='gene_symbols', cache=False)
    except Exception as e:
        print(f"  *** ERROR reading {sid}: {e}")
        load_summary.append({**entry, 'n_cells': 0, 'n_genes': 0, 'status': f'ERROR: {e}'})
        continue

    a.obs_names = [f"{sid}_{bc}" for bc in a.obs_names]
    a.obs['sample_id'] = sid
    a.obs['animal_id'] = entry['animal_id']
    a.obs['week'] = entry['week']
    a.obs['treatment'] = entry['treatment']

    n_cells = a.shape[0]
    n_genes = a.shape[1]
    adatas.append(a)
    load_summary.append({**entry, 'n_cells': n_cells, 'n_genes': n_genes, 'status': 'OK'})
    print(f"  {sid:20s}  {n_cells:>7,} cells  {n_genes:>6,} genes  OK")

print(f"\nConcatenating {len(adatas)} samples...")
adata = sc.concat(adatas, join='outer', merge='same')
adata.obs_names_make_unique()
adata.obs['week'] = pd.to_numeric(adata.obs['week'], errors='coerce').astype(int)
print(f"\nCombined AnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

df_load = pd.DataFrame(load_summary)
print("\n--- Load Summary ---")
print(df_load[['sample_id', 'treatment', 'animal_id', 'week', 'n_cells', 'status']].to_string(index=False))

missing = df_load[df_load['status'] != 'OK']
if len(missing) > 0:
    print(f"\n*** WARNING: {len(missing)} samples failed to load ***")
    print(missing[['sample_id', 'status']].to_string(index=False))

df_load.to_csv(os.path.join(OUT_DIR, 'load_summary.csv'), index=False)


# %% Cell 3 — Gene Class Identification and QC Metrics
print("\n" + "=" * 80)
print("GENE CLASS IDENTIFICATION AND QC METRICS")
print("=" * 80)

# --- [FIX 1] Mitochondrial genes ---
# Rhesus macaque Mmul10 uses KEG06_ locus tags for mitochondrial genes
# (13 protein-coding: KEG06_p01-p13, plus tRNAs and rRNAs)
adata.var['mt'] = adata.var_names.str.startswith('KEG06_')
n_mt = adata.var['mt'].sum()

if n_mt == 0:
    # Fallback: try human-style MT- prefix
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    n_mt = adata.var['mt'].sum()
    if n_mt > 0:
        print(f"  NOTE: Using MT- prefix fallback ({n_mt} genes)")
    else:
        print("  *** WARNING: No MT genes found (tried KEG06_ and MT-) ***")
        print(f"  First 20 genes: {adata.var_names[:20].tolist()}")

# --- [FIX 2] Pathogen gene identification ---
# Use gene_ids column (features.tsv col 1) for definitive identification:
#   Mtb CDC1551: gene_id starts with "MT_RS"
#   SIV mac239:  gene_id starts with "gene-"
# Fallback to var_names patterns if gene_ids not available
if 'gene_ids' in adata.var.columns:
    print("  Using gene_ids column for pathogen identification")
    adata.var['pathogen_mtb'] = adata.var['gene_ids'].str.startswith('MT_RS')
    adata.var['pathogen_siv'] = adata.var['gene_ids'].str.startswith('gene-')
else:
    print("  WARNING: gene_ids not found — using var_names fallback")
    adata.var['pathogen_mtb'] = adata.var_names.str.startswith('MT_RS')
    adata.var['pathogen_siv'] = adata.var_names.isin(SIV_GENE_LIST)

adata.var['pathogen'] = adata.var['pathogen_mtb'] | adata.var['pathogen_siv']
adata.var['host'] = ~adata.var['pathogen']

n_mtb = adata.var['pathogen_mtb'].sum()
n_siv = adata.var['pathogen_siv'].sum()
n_host = adata.var['host'].sum()
siv_found = adata.var_names[adata.var['pathogen_siv']].tolist()

print(f"\nGene classes:")
print(f"  Host (Mmul10):     {n_host:,} genes")
print(f"  Mitochondrial:     {n_mt} genes (KEG06_ prefix)")
print(f"  Pathogen (Mtb):    {n_mtb} genes")
print(f"  Pathogen (SIV):    {n_siv} genes: {siv_found}")

if n_siv == 0:
    print(f"  *** WARNING: 0 SIV genes found — expected: {SIV_GENE_LIST} ***")

# --- Calculate QC metrics ---
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt', 'pathogen_mtb', 'pathogen_siv'],
    percent_top=[TOP_N_GENE_PCT],
    inplace=True
)

# --- Pathogen read detection verification ---
cells_with_siv = (adata.obs['total_counts_pathogen_siv'] > 0).sum()
cells_with_mtb = (adata.obs['total_counts_pathogen_mtb'] > 0).sum()
print(f"\n  Cells with SIV reads: {cells_with_siv:,} ({100*cells_with_siv/adata.n_obs:.2f}%)")
print(f"  Cells with Mtb reads: {cells_with_mtb:,} ({100*cells_with_mtb/adata.n_obs:.2f}%)")

# --- Per-sample MT summary ---
mt_summary = (
    adata.obs.groupby('sample_id')
    .agg(
        treatment=('treatment', 'first'),
        animal_id=('animal_id', 'first'),
        week=('week', 'first'),
        n_cells=('pct_counts_mt', 'size'),
        median_mt=('pct_counts_mt', 'median'),
        mean_mt=('pct_counts_mt', 'mean'),
        pct95_mt=('pct_counts_mt', lambda x: np.percentile(x, 95)),
        pct10_mt=('pct_counts_mt', lambda x: np.percentile(x, 10)),
        median_genes=('n_genes_by_counts', 'median'),
        median_umis=('total_counts', 'median'),
    )
    .sort_values(['treatment', 'week', 'animal_id'])
    .reset_index()
)
mt_summary['compromised'] = mt_summary['median_mt'] > COMPROMISED_MT_FLOOR

print("\n--- Per-Sample QC Summary ---")
print(mt_summary.to_string(index=False))

n_compromised = mt_summary['compromised'].sum()
if n_compromised > 0:
    print(f"\n*** {n_compromised} COMPROMISED SAMPLE(S) (median MT% > {COMPROMISED_MT_FLOOR}%) ***")
    print(mt_summary[mt_summary['compromised']][
        ['sample_id', 'median_mt', 'pct10_mt', 'median_genes']].to_string(index=False))
else:
    print(f"\nNo compromised samples (all median MT% < {COMPROMISED_MT_FLOOR}%)")

mt_summary.to_csv(os.path.join(OUT_DIR, 'mt_qc_summary.csv'), index=False)

# --- MT% distribution plots ---
print("\nGenerating MT% distribution plots...")

samples_sorted = mt_summary['sample_id'].tolist()
n_cols = 4
n_rows = int(np.ceil(len(samples_sorted) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = axes.flatten()

for i, sid in enumerate(samples_sorted):
    ax = axes[i]
    subset = adata.obs[adata.obs['sample_id'] == sid]
    info = mt_summary[mt_summary['sample_id'] == sid].iloc[0]
    ax.hist(subset['pct_counts_mt'], bins=50, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(info['median_mt'], color='red', linestyle='--', linewidth=1.5,
               label=f"median={info['median_mt']:.1f}%")
    if MT_THRESHOLD is not None:
        ax.axvline(MT_THRESHOLD, color='orange', linestyle='-', linewidth=1.5)
    title_color = 'red' if info['compromised'] else 'black'
    ax.set_title(f"{sid}\nn={len(subset):,}  med_genes={info['median_genes']:.0f}",
                 fontsize=8, color=title_color)
    ax.set_xlabel('MT %', fontsize=8)
    ax.set_ylabel('Cells', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_xlim(0, 100)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Mitochondrial % Distribution per Sample', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mt_distribution_per_sample.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'mt_distribution_per_sample.pdf'), bbox_inches='tight')
plt.show()
plt.close()

# --- MT% vs n_genes scatter ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for treatment, color in [('D1MT + Late cART', 'tab:blue'), ('Late cART', 'tab:red')]:
    mask = adata.obs['treatment'] == treatment
    axes[0].scatter(adata.obs.loc[mask, 'pct_counts_mt'],
                    adata.obs.loc[mask, 'n_genes_by_counts'],
                    s=0.5, alpha=0.1, c=color, label=treatment, rasterized=True)
axes[0].set_xlabel('MT %'); axes[0].set_ylabel('Genes detected')
axes[0].set_title('MT% vs Genes by Treatment'); axes[0].legend(markerscale=10)

for week, color in [(5, 'tab:green'), (11, 'tab:orange'), (15, 'tab:purple')]:
    mask = adata.obs['week'] == week
    axes[1].scatter(adata.obs.loc[mask, 'pct_counts_mt'],
                    adata.obs.loc[mask, 'n_genes_by_counts'],
                    s=0.5, alpha=0.1, c=color, label=f'Week {week}', rasterized=True)
axes[1].set_xlabel('MT %'); axes[1].set_ylabel('Genes detected')
axes[1].set_title('MT% vs Genes by Week'); axes[1].legend(markerscale=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mt_vs_genes_scatter.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'mt_vs_genes_scatter.pdf'), bbox_inches='tight')
plt.show()
plt.close()


# %% Cell 4 — QC Filtering
print("\n" + "=" * 80)
print("QC FILTERING")
print("=" * 80)

n_before = adata.shape[0]
print(f"Starting cells: {n_before:,}")

# Step 1: Exclude compromised samples
compromised_ids = mt_summary.loc[mt_summary['compromised'], 'sample_id'].tolist()
if compromised_ids:
    print(f"\n  Excluding {len(compromised_ids)} compromised sample(s): {compromised_ids}")
    adata = adata[~adata.obs['sample_id'].isin(compromised_ids)].copy()
    print(f"  Cells remaining: {adata.shape[0]:,}")
else:
    print("\n  No compromised samples to exclude.")

# Step 2: MT threshold filter
if MT_THRESHOLD is not None:
    n_pre = adata.shape[0]
    adata = adata[adata.obs['pct_counts_mt'] <= MT_THRESHOLD].copy()
    print(f"\n  MT filter (>{MT_THRESHOLD}%): removed {n_pre - adata.shape[0]:,}")
else:
    print(f"\n  MT threshold: None (tracked as metadata only)")

# Step 3: Min genes
n_pre = adata.shape[0]
adata = adata[adata.obs['n_genes_by_counts'] >= MIN_GENES].copy()
print(f"  Min genes filter (<{MIN_GENES}): removed {n_pre - adata.shape[0]:,}")

# Step 4: Per-sample MAD
print(f"\n  Per-sample MAD filtering (multiplier={MAD_MULTIPLIER})...")
cells_to_keep = pd.Series(True, index=adata.obs_names)
mad_report = []

for sid in adata.obs['sample_id'].unique():
    mask = adata.obs['sample_id'] == sid
    subset = adata.obs.loc[mask]
    n_start = mask.sum()
    sample_keep = pd.Series(True, index=subset.index)

    genes = subset['n_genes_by_counts']
    med_g = genes.median()
    mad_g = np.median(np.abs(genes - med_g)) * 1.4826
    sample_keep &= (genes >= max(med_g - MAD_MULTIPLIER * mad_g, MIN_GENES))

    umis = subset['total_counts']
    med_u = umis.median()
    mad_u = np.median(np.abs(umis - med_u)) * 1.4826
    sample_keep &= (umis >= max(med_u - MAD_MULTIPLIER * mad_u, 0))

    top_col = f'pct_counts_in_top_{TOP_N_GENE_PCT}_genes'
    if top_col in subset.columns:
        topgene = subset[top_col]
        med_t = topgene.median()
        mad_t = np.median(np.abs(topgene - med_t)) * 1.4826
        high_thresh = med_t + MAD_MULTIPLIER * mad_t
        sample_keep &= (topgene <= high_thresh)
    else:
        high_thresh = np.nan

    cells_to_keep.loc[subset.index] = sample_keep
    mad_report.append({'sample_id': sid, 'n_before': n_start,
                       'n_removed_mad': n_start - sample_keep.sum(),
                       'n_after': sample_keep.sum()})

n_pre = adata.shape[0]
adata = adata[cells_to_keep].copy()
print(f"  MAD removed {n_pre - adata.shape[0]:,} cells")
print(f"  Cells remaining: {adata.shape[0]:,}")

df_mad = pd.DataFrame(mad_report).sort_values('sample_id')
print("\n--- MAD QC Report ---")
print(df_mad.to_string(index=False))
df_mad.to_csv(os.path.join(OUT_DIR, 'mad_qc_report.csv'), index=False)

# Step 5: Gene filtering (host only, pathogen protected)
print(f"\n  Gene filtering (min_cells={MIN_CELLS_PER_GENE}, host only)...")
n_genes_before = adata.shape[1]

host_mask = adata.var['host']
pathogen_mask = adata.var['pathogen']

host_genes = adata[:, host_mask].copy()
sc.pp.filter_genes(host_genes, min_cells=MIN_CELLS_PER_GENE)
host_genes_kept = host_genes.var_names.tolist()

pathogen_genes_kept = adata.var_names[pathogen_mask].tolist()
all_genes_kept = host_genes_kept + pathogen_genes_kept
adata = adata[:, adata.var_names.isin(all_genes_kept)].copy()

n_siv_kept = adata.var['pathogen_siv'].sum() if 'pathogen_siv' in adata.var.columns else 0
n_mtb_kept = adata.var['pathogen_mtb'].sum() if 'pathogen_mtb' in adata.var.columns else 0
print(f"  Host: {host_mask.sum():,} → {len(host_genes_kept):,}")
print(f"  Pathogen kept: {len(pathogen_genes_kept)} (Mtb={n_mtb_kept}, SIV={n_siv_kept})")
print(f"  Total: {n_genes_before:,} → {adata.shape[1]:,}")

n_after = adata.shape[0]
print(f"\n{'=' * 60}")
print(f"QC SUMMARY: {n_before:,} → {n_after:,} cells "
      f"(removed {n_before-n_after:,}, {(n_before-n_after)/n_before*100:.1f}%)")
print(f"{'=' * 60}")


# %% Cell 5 — Post-QC Diagnostics
print("\n" + "=" * 80)
print("POST-QC DIAGNOSTICS")
print("=" * 80)

ct_tw = pd.crosstab(adata.obs['treatment'], adata.obs['week'], margins=True)
ct_tw.columns = [f'W{int(c)}' if isinstance(c, (int, float, np.integer)) and c == c
                 else str(c) for c in ct_tw.columns]
print("\nCell counts — Treatment × Week:")
print(ct_tw.to_string())

ct_detail = pd.crosstab([adata.obs['treatment'], adata.obs['animal_id']],
                         adata.obs['week'], margins=True)
ct_detail.columns = [f'W{int(c)}' if isinstance(c, (int, float, np.integer)) and c == c
                     else str(c) for c in ct_detail.columns]
print("\n\nCell counts — Treatment × Week × Animal:")
print(ct_detail.to_string())

# Data gap analysis
print("\n--- DATA GAP ANALYSIS ---")
expected = {
    'D1MT + Late cART': {'animals': ['44129', '44132', '44137', '44154'], 'weeks': [5, 11, 15]},
    'Late cART': {'animals': ['44120', '44156', 'KV48', 'LE24'], 'weeks': [5, 11, 15]},
}
for treatment, design in expected.items():
    print(f"\n  {treatment}:")
    for week in design['weeks']:
        mask = (adata.obs['treatment'] == treatment) & (adata.obs['week'] == week)
        present = sorted(adata.obs.loc[mask, 'animal_id'].unique())
        gaps = [a for a in design['animals'] if a not in present]
        status = "OK" if not gaps else f"GAPS: {gaps}"
        print(f"    Week {week:2d}: {mask.sum():>7,} cells, "
              f"{len(present)}/{len(design['animals'])} animals  [{status}]")

# Animal dominance
print("\n--- ANIMAL DOMINANCE CHECK ---")
for treatment in sorted(adata.obs['treatment'].unique()):
    print(f"  {treatment}:")
    for week in sorted(adata.obs['week'].unique()):
        mask = (adata.obs['treatment'] == treatment) & (adata.obs['week'] == week)
        subset = adata.obs.loc[mask]
        if len(subset) == 0:
            continue
        animal_counts = subset['animal_id'].value_counts()
        total = animal_counts.sum()
        print(f"    Week {week} (n={total:,}):")
        for animal, count in animal_counts.items():
            pct = count / total * 100
            flag = " *** DOMINANT" if pct > 60 else ""
            print(f"      {animal}: {count:>7,} ({pct:5.1f}%){flag}")

# Post-QC pathogen verification
print("\n--- POST-QC PATHOGEN VERIFICATION ---")
siv_post = (adata.obs['total_counts_pathogen_siv'] > 0).sum()
mtb_post = (adata.obs['total_counts_pathogen_mtb'] > 0).sum()
print(f"  SIV+ cells: {siv_post:,} / {adata.n_obs:,} ({100*siv_post/adata.n_obs:.2f}%)")
print(f"  Mtb+ cells: {mtb_post:,} / {adata.n_obs:,} ({100*mtb_post/adata.n_obs:.2f}%)")

ct_tw.to_csv(os.path.join(OUT_DIR, 'postqc_treatment_week.csv'))
ct_detail.to_csv(os.path.join(OUT_DIR, 'postqc_treatment_week_animal.csv'))


# %% Cell 6 — Normalize, Batch Correct, Embed, and Save
print("\n" + "=" * 80)
print("NORMALIZATION AND EMBEDDING")
print("=" * 80)

print("\n  Log-normalizing (target_sum=1e4)...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

print("  Finding highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key='sample_id')
print(f"  HVGs selected: {adata.var['highly_variable'].sum()}")

print("  Running PCA (n_comps=50)...")
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)

print("  Running BBKNN (batch_key='sample_id')...")
try:
    import bbknn
    bbknn.bbknn(adata, batch_key='sample_id')
    print("  BBKNN complete.")
except ImportError:
    print("  *** BBKNN not installed — falling back to standard neighbors ***")
    sc.pp.neighbors(adata, n_pcs=30)

print("  Computing UMAP...")
sc.tl.umap(adata)

# QC UMAPs
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
sc.pl.umap(adata, color='treatment', ax=axes[0, 0], show=False, frameon=False,
           title='Treatment', palette=['tab:blue', 'tab:red'])
sc.pl.umap(adata, color='week', ax=axes[0, 1], show=False, frameon=False, title='Week')
sc.pl.umap(adata, color='animal_id', ax=axes[0, 2], show=False, frameon=False, title='Animal ID')
sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1, 0], show=False,
           frameon=False, title='Genes Detected', cmap='viridis')
sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 1], show=False,
           frameon=False, title='MT %', cmap='inferno')
sc.pl.umap(adata, color='total_counts', ax=axes[1, 2], show=False,
           frameon=False, title='Total UMIs', cmap='viridis')
plt.suptitle(f'Post-QC UMAP — {adata.shape[0]:,} cells', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'umap_qc_overview.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'umap_qc_overview.pdf'), bbox_inches='tight')
plt.show()
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, treatment in enumerate(['D1MT + Late cART', 'Late cART']):
    mask = adata.obs['treatment'] == treatment
    sc.pl.umap(adata[mask], color='week', ax=axes[i], show=False, frameon=False,
               title=f'{treatment}\n({mask.sum():,} cells)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'umap_by_treatment_week.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'umap_by_treatment_week.pdf'), bbox_inches='tight')
plt.show()
plt.close()

# Save
print(f"\n  Saving QC'd AnnData to: {ADATA_OUT}")
adata.write_h5ad(ADATA_OUT)
print(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

print(f"\n{'=' * 80}")
print("PHASE 1 COMPLETE")
print(f"{'=' * 80}")
print(f"  Output directory: {OUT_DIR}")
print(f"  AnnData file:     {ADATA_OUT}")
print(f"  Total cells:      {adata.shape[0]:,}")
print(f"  Total genes:      {adata.shape[1]:,}")
print(f"  Treatments:       {sorted(adata.obs['treatment'].unique())}")
print(f"  Weeks:            {sorted(adata.obs['week'].unique())}")
print(f"  Animals:          {sorted(adata.obs['animal_id'].unique())}")
print(f"  MT genes:         {adata.var['mt'].sum()} (KEG06_ prefix)")
print(f"  SIV genes:        {adata.var['pathogen_siv'].sum()}: "
      f"{adata.var_names[adata.var['pathogen_siv']].tolist()}")
print(f"  Mtb genes:        {adata.var['pathogen_mtb'].sum()}")
print(f"\n  Next: Phase 2 — Tiered Annotation")
