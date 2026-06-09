#!/usr/bin/env python3
"""
Phase 3b — DEG Analysis and Pathway Enrichment
================================================
Rhesus Mtb/SIV Coinfection scRNA-seq

Focused comparisons informed by LIANA cell-cell communication and
composition findings from Phases 2-3a-3c:

  PRIORITY 1: M1 IM-like D1MT vs Late cART at W11
    - Peak M1 expansion (10.4x fold-change in D1MT)
    - Strongest TNF→TNFRSF1B signaling to T cells
    - Defines the inflammatory gene program under IDO blockade

  PRIORITY 2: CD4 Effector D1MT vs Late cART at W15
    - CD4 collapse timepoint (D1MT 0.93x vs Late cART 0.40x)
    - Higher Th1 scores in D1MT, lower exhaustion
    - Should reveal what drives CD4 loss in Late cART

  PRIORITY 3: IFN-responsive Macs D1MT vs Late cART at W11
    - Hub for CD80→CD28 co-stimulation (spec_rank=0.000003)
    - Power-limited (66 vs 30 cells) but biologically critical

  PRIORITY 4: M2 IM-like D1MT vs Late cART at W11
    - Dominant population with good power (9,147 vs 7,512 cells)
    - Primary antigen presenters (HLA-DRA→CD4 mag_rank 0.001-0.006)

  PRIORITY 5: M1 IM-like vs M2 IM-like within D1MT W11
    - Defines M1/M2 transcriptional boundary
    - Validates the subtype separation for the paper

  PRIORITY 6: M1 IM-like vs IFN-responsive within D1MT W11
    - Defines transitional vs full immunosuppressive state
    - Should show IDO1/CXCL enrichment in IFN-resp, TNF/IL1B in M1

  SUPPLEMENTARY: CD4 Effector at W5, W11 (baseline and mid-infection)

Methods:
  - Wilcoxon rank-sum on pre-normalized data (no re-normalization)
  - MHC-I genes excluded (animal haplotype confounder)
  - gseapy enrichment (KEGG + GO_BP) on significant DEGs
  - Hierarchical heatmaps of top DEGs per comparison

Treatment groups (focal analysis only):
  - D1MT + Late cART : IDO inhibitor + late antiretroviral therapy
  - Late cART        : Late cART only (no IDO inhibition)

Run in Spyder (interactive cells with # %%) or standalone.
Output dir: analysis/phase3b_deg/

Author: Jake Lehle / Kaushal Lab
Date: May 2026
"""

# %% Cell 1 — Load, filter MHC-I, define comparisons
# =========================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import os
import warnings
warnings.filterwarnings('ignore')

# === PATHS ===
ADATA_PATH = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_focal_annotated.h5ad'
OUT_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/phase3b_deg/'
os.makedirs(OUT_DIR, exist_ok=True)

# === COLUMN NAMES (matching Phase 2 output) ===
TREATMENT_COL = 'treatment'
WEEK_COL = 'week'
TIER2_COL = 'tier2_full'
TIER3_COL = 'tier3_state'

# === TREATMENT LABELS ===
D1MT = 'D1MT + Late cART'
CART = 'Late cART'

# === THRESHOLDS ===
PADJ_THRESHOLD = 0.05
LOGFC_THRESHOLD = 0.25     # Minimum absolute log fold-change for reporting
MIN_CELLS = 30             # Lowered from 50 to accommodate small populations (M1, IFN-resp)

# === LOAD ===
print("Loading Phase 2 annotated data...")
adata = sc.read_h5ad(ADATA_PATH)
adata.obs[WEEK_COL] = pd.to_numeric(adata.obs[WEEK_COL], errors='coerce').astype(int)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"  Treatments: {sorted(adata.obs[TREATMENT_COL].unique())}")
print(f"  Weeks: {sorted(adata.obs[WEEK_COL].unique())}")

# --- Exclude MHC class I genes (haplotype confounder) ---
# MAMU-A and MAMU-B are MHC-I; MAMU-D/DR/DP/DQ are MHC-II (keep those)
mhc1_mask = (adata.var_names.str.startswith('MAMU-A') |
             adata.var_names.str.startswith('MAMU-B'))
mhc1_genes = adata.var_names[mhc1_mask].tolist()
print(f"  Excluding {len(mhc1_genes)} MHC-I genes: {mhc1_genes}")

adata = adata[:, ~mhc1_mask].copy()
print(f"  After MHC-I exclusion: {adata.shape[1]:,} genes")

# --- Print cell counts for key populations ---
print(f"\n  Key population sizes:")
for pop in ['Macrophages: M1 IM-like', 'Macrophages: M2 IM-like',
            'Macrophages: IFN-responsive', 'Macrophages: AM-like']:
    for treat in [D1MT, CART]:
        for wk in [5, 11, 15]:
            n = ((adata.obs[TREATMENT_COL] == treat) &
                 (adata.obs[WEEK_COL] == wk) &
                 (adata.obs[TIER2_COL] == pop)).sum()
            if n > 0:
                flag = " ***" if n < MIN_CELLS else ""
                print(f"    {pop:35s} | {treat:20s} W{wk}: {n:>5,}{flag}")


# %% Cell 2 — DEG helper function
# =========================================================================
print("\n" + "=" * 80)
print("DEG ANALYSIS (Wilcoxon, no re-normalization, MHC-I excluded)")
print("=" * 80)

all_deg = {}

def run_deg(adata_obj, mask_a, mask_b, name, description,
            test_label='test', ref_label='reference'):
    """Run Wilcoxon DEG on pre-normalized data."""
    n_a, n_b = mask_a.sum(), mask_b.sum()
    print(f"\n--- {name} ---")
    print(f"  {description}")
    print(f"  Test ({test_label}): {n_a:,} cells | Reference ({ref_label}): {n_b:,} cells")

    if n_a < MIN_CELLS or n_b < MIN_CELLS:
        print(f"  *** SKIPPED: <{MIN_CELLS} cells in one group "
              f"(test={n_a}, ref={n_b}) ***")
        all_deg[name] = None
        return None

    sub = adata_obj[mask_a | mask_b].copy()
    sub.obs['deg_group'] = ref_label
    sub.obs.loc[mask_a[mask_a | mask_b], 'deg_group'] = test_label

    sc.tl.rank_genes_groups(
        sub, groupby='deg_group', groups=[test_label], reference=ref_label,
        method='wilcoxon', key_added='deg', use_raw=False
    )

    result = sc.get.rank_genes_groups_df(sub, group=test_label, key='deg')
    result['comparison'] = name
    result['test_label'] = test_label
    result['ref_label'] = ref_label
    result['n_test'] = n_a
    result['n_ref'] = n_b

    sig = result[result['pvals_adj'] < PADJ_THRESHOLD]
    sig_up = sig[sig['logfoldchanges'] > 0]
    sig_down = sig[sig['logfoldchanges'] < 0]
    # Also filter by logFC for reporting
    sig_strong = sig[sig['logfoldchanges'].abs() >= LOGFC_THRESHOLD]

    print(f"  Significant (padj<{PADJ_THRESHOLD}): {len(sig):,} "
          f"({len(sig_up)} up in {test_label}, {len(sig_down)} down)")
    print(f"  Strong (|logFC|>={LOGFC_THRESHOLD}): {len(sig_strong):,}")
    if len(sig_up) > 0:
        print(f"  Top 5 up:   {sig_up.head(5)['names'].tolist()}")
    if len(sig_down) > 0:
        print(f"  Top 5 down: {sig_down.head(5)['names'].tolist()}")

    # Save results
    result.to_csv(os.path.join(OUT_DIR, f'deg_{name}_all.csv'), index=False)
    if len(sig) > 0:
        sig.to_csv(os.path.join(OUT_DIR, f'deg_{name}_sig.csv'), index=False)

    all_deg[name] = result
    return result


# %% Cell 3 — Priority DEG Comparisons
# =========================================================================

# ===================================================================
# PRIORITY 1: M1 IM-like — D1MT vs Late cART at W11
# Peak expansion (10.4x in D1MT), strongest TNF signaling
# ===================================================================
print("\n" + "=" * 80)
print("PRIORITY 1: M1 IM-like — D1MT Effect at Peak Expansion")
print("=" * 80)

m1_mask = adata.obs[TIER2_COL] == 'Macrophages: M1 IM-like'

mask_a = m1_mask & (adata.obs[TREATMENT_COL] == D1MT) & (adata.obs[WEEK_COL] == 11)
mask_b = m1_mask & (adata.obs[TREATMENT_COL] == CART) & (adata.obs[WEEK_COL] == 11)
run_deg(adata, mask_a, mask_b, 'M1_D1MT_v_cART_W11',
        'M1 IM-like: D1MT vs Late cART at W11 (peak M1 expansion)',
        test_label=D1MT, ref_label=CART)

# Also run W5 and W15 for M1 if sufficient cells
for wk in [5, 15]:
    mask_a = m1_mask & (adata.obs[TREATMENT_COL] == D1MT) & (adata.obs[WEEK_COL] == wk)
    mask_b = m1_mask & (adata.obs[TREATMENT_COL] == CART) & (adata.obs[WEEK_COL] == wk)
    run_deg(adata, mask_a, mask_b, f'M1_D1MT_v_cART_W{wk}',
            f'M1 IM-like: D1MT vs Late cART at W{wk}',
            test_label=D1MT, ref_label=CART)


# ===================================================================
# PRIORITY 2: CD4 Effector — D1MT vs Late cART at W5, W11, W15
# CD4 collapse at W15 (D1MT 0.93x vs cART 0.40x)
# ===================================================================
print("\n" + "=" * 80)
print("PRIORITY 2: CD4 Effector — D1MT Effect Across Timepoints")
print("=" * 80)

for wk in [5, 11, 15]:
    mask_a = ((adata.obs[TREATMENT_COL] == D1MT) &
              (adata.obs[WEEK_COL] == wk) &
              (adata.obs[TIER3_COL] == 'CD4 Effector'))
    mask_b = ((adata.obs[TREATMENT_COL] == CART) &
              (adata.obs[WEEK_COL] == wk) &
              (adata.obs[TIER3_COL] == 'CD4 Effector'))
    run_deg(adata, mask_a, mask_b, f'CD4eff_D1MT_v_cART_W{wk}',
            f'CD4 Effector: D1MT vs Late cART at W{wk}',
            test_label=D1MT, ref_label=CART)


# ===================================================================
# PRIORITY 3: IFN-responsive Macs — D1MT vs Late cART at W11
# Co-stimulation hub (CD80→CD28 spec_rank=0.000003)
# ===================================================================
print("\n" + "=" * 80)
print("PRIORITY 3: IFN-responsive Macs — D1MT Effect")
print("=" * 80)

ifn_mask = adata.obs[TIER2_COL] == 'Macrophages: IFN-responsive'

for wk in [5, 11, 15]:
    mask_a = ifn_mask & (adata.obs[TREATMENT_COL] == D1MT) & (adata.obs[WEEK_COL] == wk)
    mask_b = ifn_mask & (adata.obs[TREATMENT_COL] == CART) & (adata.obs[WEEK_COL] == wk)
    run_deg(adata, mask_a, mask_b, f'IFNresp_D1MT_v_cART_W{wk}',
            f'IFN-responsive Mac: D1MT vs Late cART at W{wk}',
            test_label=D1MT, ref_label=CART)


# ===================================================================
# PRIORITY 4: M2 IM-like — D1MT vs Late cART at W11
# Dominant population, primary antigen presenters
# ===================================================================
print("\n" + "=" * 80)
print("PRIORITY 4: M2 IM-like — D1MT Effect (Dominant Antigen Presenters)")
print("=" * 80)

m2_mask = adata.obs[TIER2_COL] == 'Macrophages: M2 IM-like'

for wk in [5, 11, 15]:
    mask_a = m2_mask & (adata.obs[TREATMENT_COL] == D1MT) & (adata.obs[WEEK_COL] == wk)
    mask_b = m2_mask & (adata.obs[TREATMENT_COL] == CART) & (adata.obs[WEEK_COL] == wk)
    run_deg(adata, mask_a, mask_b, f'M2_D1MT_v_cART_W{wk}',
            f'M2 IM-like: D1MT vs Late cART at W{wk}',
            test_label=D1MT, ref_label=CART)


# ===================================================================
# PRIORITY 5: M1 IM-like vs M2 IM-like within D1MT W11
# Defines M1/M2 transcriptional boundary
# ===================================================================
print("\n" + "=" * 80)
print("PRIORITY 5: M1 vs M2 IM-like — Subtype Boundary (D1MT W11)")
print("=" * 80)

d1mt_w11 = (adata.obs[TREATMENT_COL] == D1MT) & (adata.obs[WEEK_COL] == 11)

mask_a = m1_mask & d1mt_w11
mask_b = m2_mask & d1mt_w11
run_deg(adata, mask_a, mask_b, 'M1_v_M2_D1MT_W11',
        'M1 IM-like vs M2 IM-like within D1MT at W11 '
        '(defines inflammatory vs anti-inflammatory boundary)',
        test_label='M1 IM-like', ref_label='M2 IM-like')

# Also within Late cART W11 for comparison
cart_w11 = (adata.obs[TREATMENT_COL] == CART) & (adata.obs[WEEK_COL] == 11)
mask_a = m1_mask & cart_w11
mask_b = m2_mask & cart_w11
run_deg(adata, mask_a, mask_b, 'M1_v_M2_cART_W11',
        'M1 IM-like vs M2 IM-like within Late cART at W11',
        test_label='M1 IM-like', ref_label='M2 IM-like')


# ===================================================================
# PRIORITY 6: M1 IM-like vs IFN-responsive within D1MT W11
# Transitional vs full immunosuppressive state
# ===================================================================
print("\n" + "=" * 80)
print("PRIORITY 6: M1 IM-like vs IFN-responsive — Activation Boundary (D1MT W11)")
print("=" * 80)

mask_a = m1_mask & d1mt_w11
mask_b = ifn_mask & d1mt_w11
run_deg(adata, mask_a, mask_b, 'M1_v_IFNresp_D1MT_W11',
        'M1 IM-like vs IFN-responsive within D1MT at W11 '
        '(transitional vs immunosuppressive)',
        test_label='M1 IM-like', ref_label='IFN-responsive')


# ===================================================================
# SUPPLEMENTARY: AM-like — D1MT vs Late cART (global, all timepoints)
# ===================================================================
print("\n" + "=" * 80)
print("SUPPLEMENTARY: AM-like Macrophages — Global Treatment Comparison")
print("=" * 80)

am_mask = adata.obs[TIER2_COL] == 'Macrophages: AM-like'

mask_a = am_mask & (adata.obs[TREATMENT_COL] == D1MT)
mask_b = am_mask & (adata.obs[TREATMENT_COL] == CART)
run_deg(adata, mask_a, mask_b, 'AMlike_D1MT_v_cART_global',
        'AM-like: D1MT vs Late cART — all timepoints pooled',
        test_label=D1MT, ref_label=CART)


# %% Cell 4 — DEG Summary Table
# =========================================================================
print("\n" + "=" * 80)
print("DEG SUMMARY")
print("=" * 80)

summary_rows = []
for name, res in all_deg.items():
    if res is None:
        summary_rows.append({'comparison': name, 'status': 'SKIPPED',
                             'n_sig': 0, 'n_up': 0, 'n_down': 0,
                             'n_strong': 0, 'n_test': 0, 'n_ref': 0})
    else:
        sig = res[res['pvals_adj'] < PADJ_THRESHOLD]
        sig_strong = sig[sig['logfoldchanges'].abs() >= LOGFC_THRESHOLD]
        summary_rows.append({
            'comparison': name,
            'status': 'OK',
            'n_test': res['n_test'].iloc[0] if 'n_test' in res.columns else 0,
            'n_ref': res['n_ref'].iloc[0] if 'n_ref' in res.columns else 0,
            'n_sig': len(sig),
            'n_up': (sig['logfoldchanges'] > 0).sum(),
            'n_down': (sig['logfoldchanges'] < 0).sum(),
            'n_strong': len(sig_strong),
        })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(OUT_DIR, 'deg_summary.csv'), index=False)


# %% Cell 5 — Pathway Enrichment (logFC-filtered for high-power comparisons)
# =========================================================================
print("\n" + "=" * 80)
print("PATHWAY ENRICHMENT (logFC-filtered)")
print("=" * 80)

try:
    import gseapy as gp
    GSEA_OK = True
except ImportError:
    print("  gseapy not available — skipping enrichment")
    GSEA_OK = False

if GSEA_OK:
    import time
    pathway_dir = os.path.join(OUT_DIR, 'pathways/')
    os.makedirs(pathway_dir, exist_ok=True)

    def enrichr_with_retry(gene_list, lib, max_retries=3, base_delay=30):
        """Run enrichr with retry on 429 rate limit errors."""
        for attempt in range(max_retries):
            try:
                enr = gp.enrichr(gene_list=gene_list, gene_sets=lib,
                                 organism='human', outdir=None, no_plot=True)
                return enr
            except Exception as e:
                if '429' in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    print(f"      Rate limited (429), waiting {delay}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise e

    for name, res in all_deg.items():
        if res is None:
            continue

        sig = res[res['pvals_adj'] < PADJ_THRESHOLD]
        n_test = res['n_test'].iloc[0] if 'n_test' in res.columns else 0
        n_ref = res['n_ref'].iloc[0] if 'n_ref' in res.columns else 0

        # Apply stricter logFC filter for high-power comparisons
        # (>5000 total cells detect tiny differences as significant)
        if (n_test + n_ref) > 5000:
            lfc_cutoff = 0.5
            sig = sig[sig['logfoldchanges'].abs() >= lfc_cutoff]
            print(f"\n  {name}: applying |logFC|>={lfc_cutoff} filter "
                  f"(high-power: {n_test+n_ref:,} cells) → {len(sig)} DEGs")
        else:
            lfc_cutoff = LOGFC_THRESHOLD

        sig_up = sig[sig['logfoldchanges'] > 0]['names'].tolist()
        sig_down = sig[sig['logfoldchanges'] < 0]['names'].tolist()

        if len(sig_up) < 10 and len(sig_down) < 10:
            print(f"\n  {name}: too few DEGs for enrichment "
                  f"(up={len(sig_up)}, down={len(sig_down)})")
            continue

        print(f"\n  --- {name} (up={len(sig_up)}, down={len(sig_down)}) ---")
        all_terms = []

        for direction, gene_list in [('up', sig_up), ('down', sig_down)]:
            if len(gene_list) < 10:
                continue
            for lib in ['KEGG_2021_Human', 'GO_Biological_Process_2023']:
                try:
                    enr = enrichr_with_retry(gene_list, lib)
                    enr_sig = enr.results[enr.results['Adjusted P-value'] < 0.05]
                    if len(enr_sig) > 0:
                        top = enr_sig.head(10).copy()
                        top['direction'] = direction
                        top['library'] = lib
                        top['comparison'] = name
                        all_terms.append(top)
                        enr_sig.to_csv(
                            os.path.join(pathway_dir, f'{name}_{direction}_{lib}.csv'),
                            index=False)
                        print(f"    {lib} {direction}: {len(enr_sig)} terms")
                        top5 = enr_sig.nsmallest(5, 'Adjusted P-value')
                        for _, row in top5.iterrows():
                            print(f"      {row['Adjusted P-value']:.2e}  "
                                  f"{row['Term'][:60]}")
                    else:
                        print(f"    {lib} {direction}: no significant terms")
                except Exception as e:
                    print(f"    {lib} {direction}: {str(e)[:80]}")
                time.sleep(2)

        # Enrichment barplot (kept as supplementary)
        if all_terms:
            df_terms = pd.concat(all_terms, ignore_index=True)
            df_terms['neg_log10_padj'] = -np.log10(
                df_terms['Adjusted P-value'].clip(lower=1e-50))
            df_terms['Term_short'] = df_terms['Term'].str[:55]
            df_terms['signed_score'] = df_terms.apply(
                lambda r: r['neg_log10_padj'] if r['direction'] == 'up'
                else -r['neg_log10_padj'], axis=1)
            df_terms = df_terms.sort_values('signed_score', ascending=True)

            fig, ax = plt.subplots(figsize=(10, max(4, len(df_terms) * 0.35)))
            colors = ['#D32F2F' if d == 'up' else '#1976D2'
                      for d in df_terms['direction']]
            ax.barh(range(len(df_terms)), df_terms['signed_score'],
                    color=colors, edgecolor='white')
            ax.set_yticks(range(len(df_terms)))
            ax.set_yticklabels(df_terms['Term_short'], fontsize=7)
            ax.set_xlabel('-log10(padj) × direction', fontsize=10)
            test_lbl = res['test_label'].iloc[0] if 'test_label' in res.columns else 'test'
            ref_lbl = res['ref_label'].iloc[0] if 'ref_label' in res.columns else 'ref'
            ax.set_title(f'{name}\nRed = Up in {test_lbl} | Blue = Up in {ref_lbl}',
                        fontsize=11, fontweight='bold')
            ax.axvline(0, color='black', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f'enrichment_{name}.pdf'),
                        bbox_inches='tight', dpi=300)
            fig.savefig(os.path.join(OUT_DIR, f'enrichment_{name}.png'),
                        bbox_inches='tight', dpi=300)
            print(f"    Saved: enrichment_{name}.pdf/png")
            plt.close(fig)


# %% Cell 6 — Paper Question Dotplots (Curated Gene Panels)
# =========================================================================
print("\n" + "=" * 80)
print("PAPER QUESTION DOTPLOTS — Curated Gene Panels")
print("=" * 80)

import re

# Helper: filter to genes present in adata
def valid_genes(gene_list):
    return [g for g in gene_list if g in adata.var_names]

# Helper: make dotplot with scanpy and save
def make_dotplot(sub, genes, groupby, title, filename, figsize=None, swap_axes=False):
    """Make and save a dotplot."""
    genes = valid_genes(genes)
    if len(genes) < 2 or sub.n_obs < 10:
        print(f"    Skipped {filename}: {len(genes)} genes, {sub.n_obs} cells")
        return
    print(f"    {filename}: {len(genes)} genes, {sub.n_obs:,} cells")
    try:
        dp = sc.pl.dotplot(
            sub, var_names=genes, groupby=groupby,
            standard_scale='var', swap_axes=swap_axes,
            title=title, show=False, return_fig=True
        )
        dp.savefig(os.path.join(OUT_DIR, f'{filename}.pdf'),
                   bbox_inches='tight', dpi=300)
        dp.savefig(os.path.join(OUT_DIR, f'{filename}.png'),
                   bbox_inches='tight', dpi=300)
        print(f"    Saved: {filename}.pdf/png")
        plt.close('all')
    except Exception as e:
        print(f"    Failed {filename}: {e}")
        plt.close('all')


# ===================================================================
# Q1: Does D1MT preserve CD4 T cells?
# Comparison: CD4eff D1MT vs cART across timepoints
# Figure: Dotplot of Th1/Th17/exhaustion/survival markers
# ===================================================================
print("\n--- Q1: CD4 Preservation (Effector Gene Panel) ---")

Q1_GENES = {
    'Th1': ['TBX21', 'IFNG', 'TNF', 'IL2', 'STAT4', 'IL12RB2'],
    'Th17': ['RORC', 'IL17A', 'IL17F', 'IL23R', 'CCR6', 'IL22'],
    'Activation': ['CD69', 'CD44', 'IL2RA', 'HLA-DRA', 'ICOS'],
    'Exhaustion': ['PDCD1', 'LAG3', 'HAVCR2', 'TIGIT', 'CTLA4', 'TOX'],
    'Survival': ['BCL2', 'MCL1', 'IL7R', 'BCL2L1'],
    'Cytotoxicity': ['GZMA', 'GZMB', 'PRF1', 'GNLY', 'NKG7'],
    'TCR signaling': ['LCK', 'ZAP70', 'CD3E', 'CD3D', 'NFKB1', 'STAT1'],
}
q1_flat = [g for panel in Q1_GENES.values() for g in panel]

# Create a combined grouping column: treatment × week for CD4 effectors
cd4_mask = adata.obs[TIER3_COL] == 'CD4 Effector'
cd4_sub = adata[cd4_mask].copy()
cd4_sub.obs['treat_week'] = (cd4_sub.obs[TREATMENT_COL].astype(str).str.replace(' ', '_') +
                              '_W' + cd4_sub.obs[WEEK_COL].astype(str))
# Order: D1MT W5/W11/W15 first, then Late cART W5/W11/W15
tw_order = ['D1MT_+_Late_cART_W5', 'D1MT_+_Late_cART_W11', 'D1MT_+_Late_cART_W15',
            'Late_cART_W5', 'Late_cART_W11', 'Late_cART_W15']
cd4_sub.obs['treat_week'] = pd.Categorical(
    cd4_sub.obs['treat_week'], categories=tw_order, ordered=True)

make_dotplot(cd4_sub, q1_flat, 'treat_week',
             'Q1: CD4 Effector Gene Panel — D1MT vs Late cART across Timepoints',
             'Q1_cd4_effector_panel', swap_axes=True)


# ===================================================================
# Q2: What is the M1 IM-like inflammatory program?
# Comparison: M1 vs IFN-resp (Priority 6), M1 vs M2 (Priority 5)
# Figure: Dotplot of the 7 boundary DEGs across all 3 mac subtypes
# ===================================================================
print("\n--- Q2: M1 Inflammatory Program (Boundary DEGs) ---")

Q2_BOUNDARY_GENES = ['TNF', 'IL1B', 'IDO1', 'CD274', 'SOD2', 'CYTH1', 'LPIN2']

# Show across all mac subtypes at D1MT W11
mac_subtypes = ['Macrophages: M1 IM-like', 'Macrophages: M2 IM-like',
                'Macrophages: IFN-responsive']
d1mt_w11_mac = adata[
    (adata.obs[TREATMENT_COL] == D1MT) &
    (adata.obs[WEEK_COL] == 11) &
    (adata.obs[TIER2_COL].isin(mac_subtypes))
].copy()
d1mt_w11_mac.obs[TIER2_COL] = pd.Categorical(
    d1mt_w11_mac.obs[TIER2_COL], categories=mac_subtypes, ordered=True)

make_dotplot(d1mt_w11_mac, Q2_BOUNDARY_GENES, TIER2_COL,
             'Q2: M1 vs IFN-resp Boundary DEGs — D1MT W11',
             'Q2_m1_boundary_degs')

# Extended M1 program: top DEGs from M1 D1MT vs cART W11
Q2_M1_PROGRAM = ['ITGAV', 'CXCL8', 'CD44', 'NFKB1', 'VPS13B',
                  'TNF', 'IL1B', 'STAT1', 'IRF1', 'CCL2', 'CCL3',
                  'MX1', 'MX2']

# Show across M1/M2/IFN at D1MT W11 + cART W11
all_mac_w11 = adata[
    (adata.obs[WEEK_COL] == 11) &
    (adata.obs[TIER2_COL].isin(mac_subtypes))
].copy()
all_mac_w11.obs['subtype_treat'] = (
    all_mac_w11.obs[TIER2_COL].astype(str).str.replace('Macrophages: ', '') +
    ' | ' + all_mac_w11.obs[TREATMENT_COL].astype(str))
make_dotplot(all_mac_w11, Q2_M1_PROGRAM, 'subtype_treat',
             'Q2: M1 IM-like Program — All Mac Subtypes × Treatment at W11',
             'Q2_m1_program_extended', swap_axes=True)


# ===================================================================
# Q3: How does D1MT reshape macrophage activation?
# Comparison: M2/IFN-resp/AM-like D1MT vs cART
# Figure: Dotplot of activation/antigen presentation/inflammatory
#         markers across all 5 mac subtypes × 2 treatments at W11
# ===================================================================
print("\n--- Q3: Macrophage Activation Landscape ---")

Q3_GENES = {
    'Antigen presentation': ['HLA-DRA', 'HLA-DRB1', 'CD74', 'CD80', 'CD86'],
    'Inflammatory': ['TNF', 'IL1B', 'IL6', 'CXCL8', 'CCL2', 'CCL3', 'NFKB1'],
    'IDO/checkpoint': ['IDO1', 'CD274', 'PDCD1LG2'],
    'IFN response': ['MX1', 'MX2', 'ISG15', 'IFIT1', 'STAT1', 'IRF1'],
    'M2 markers': ['CD163', 'MRC1', 'TGFB1', 'IL10'],
    'Homeostatic': ['FABP4', 'PPARG', 'MARCO'],
}
q3_flat = [g for panel in Q3_GENES.values() for g in panel]

all_mac_subtypes = ['Macrophages: AM-like', 'Macrophages: M2 IM-like',
                    'Macrophages: M1 IM-like', 'Macrophages: IFN-responsive',
                    'Macrophages: TREM2+']

mac_w11 = adata[
    (adata.obs[WEEK_COL] == 11) &
    (adata.obs[TIER2_COL].isin(all_mac_subtypes))
].copy()
mac_w11.obs['subtype_treat'] = (
    mac_w11.obs[TIER2_COL].astype(str).str.replace('Macrophages: ', '') +
    ' | ' + mac_w11.obs[TREATMENT_COL].astype(str))

make_dotplot(mac_w11, q3_flat, 'subtype_treat',
             'Q3: Macrophage Activation Landscape — W11 All Subtypes × Treatment',
             'Q3_mac_activation_w11', swap_axes=True)


# ===================================================================
# Q4: Where is the SIV reservoir?
# (Built in Cell 7 below — normalized UMI analysis)
# ===================================================================


# ===================================================================
# Q5: Macrophage-T cell signaling (Phase 3c LIANA — already done)
# ===================================================================
print("\n--- Q5: Mac-T Cell Signaling — see Phase 3c LIANA output ---")


# ===================================================================
# Q6: What pathways define the D1MT effect?
# Instead of KEGG barplots, show focused gene panels from enrichment
# ===================================================================
print("\n--- Q6: D1MT Pathway Gene Panels ---")

# CD4 W5+W11: Ribosome/translation UP in D1MT → active proliferation
Q6_RIBOSOME = ['RPL3', 'RPL23', 'RPS19', 'FAU', 'RPL11', 'RPS26',
               'EEF1A1', 'EEF2', 'RPS3', 'RPL5']

for wk in [5, 11]:
    cd4_wk = adata[
        (adata.obs[TIER3_COL] == 'CD4 Effector') &
        (adata.obs[WEEK_COL] == wk)
    ].copy()
    make_dotplot(cd4_wk, Q6_RIBOSOME, TREATMENT_COL,
                 f'Q6: Ribosome/Translation — CD4 Effector W{wk}',
                 f'Q6_cd4_ribosome_W{wk}')

# CD4 W11: Defense Response to Virus DOWN in D1MT (= UP in Late cART)
Q6_ANTIVIRAL = ['MX1', 'MX2', 'ISG15', 'IFIT1', 'IFIT3', 'OAS1',
                'OAS2', 'RSAD2', 'BST2', 'IFI44', 'IFI44L']

cd4_w11 = adata[
    (adata.obs[TIER3_COL] == 'CD4 Effector') &
    (adata.obs[WEEK_COL] == 11)
].copy()
make_dotplot(cd4_w11, Q6_ANTIVIRAL, TREATMENT_COL,
             'Q6: Antiviral Defense Genes — CD4 Effector W11\n'
             '(Up in Late cART = antiviral defense state)',
             'Q6_cd4_antiviral_W11')

# CD4 W15: TCR signaling + Th17 UP in D1MT → mature effector shift
Q6_TCR_TH17 = ['LCK', 'ZAP70', 'CD3E', 'CD3D', 'NFKB1', 'STAT1',
               'RORC', 'IL23R', 'CCR6', 'FKBP5', 'RNF213', 'EPSTI1']

cd4_w15 = adata[
    (adata.obs[TIER3_COL] == 'CD4 Effector') &
    (adata.obs[WEEK_COL] == 15)
].copy()
make_dotplot(cd4_w15, Q6_TCR_TH17, TREATMENT_COL,
             'Q6: TCR Signaling + Th17 Genes — CD4 Effector W15\n'
             '(D1MT mature effector shift)',
             'Q6_cd4_tcr_th17_W15')

# M2 W11: Spliceosome + Lysosome UP in D1MT
Q6_SPLICEOSOME = ['SRSF1', 'SRSF3', 'HNRNPA1', 'HNRNPC', 'SNRPD1',
                  'SF3B1', 'U2AF2', 'PRPF8']
Q6_LYSOSOME = ['CTSS', 'CTSL', 'LAMP1', 'LAMP2', 'LGMN',
               'GLA', 'HEXA', 'ATP6V1A']

m2_w11 = adata[
    (adata.obs[TIER2_COL] == 'Macrophages: M2 IM-like') &
    (adata.obs[WEEK_COL] == 11)
].copy()
make_dotplot(m2_w11, Q6_SPLICEOSOME + Q6_LYSOSOME, TREATMENT_COL,
             'Q6: Spliceosome + Lysosome — M2 IM-like W11',
             'Q6_m2_spliceosome_lysosome_W11')


# %% Cell 7 — SIV Reservoir Analysis (Normalized UMI Counts)
# =========================================================================
print("\n" + "=" * 80)
print("Q4: SIV RESERVOIR — Normalized UMI Counts and Cell Type Breakdown")
print("=" * 80)

from scipy import sparse

# SIV genes use individual names in the custom reference
SIV_GENE_NAMES = ['gag', 'pol', 'vif', 'vpx', 'vpr', 'tat', 'rev', 'env', 'nef']
siv_genes = [g for g in SIV_GENE_NAMES if g in adata.var_names]
siv_missing = [g for g in SIV_GENE_NAMES if g not in adata.var_names]
print(f"  SIV genes found: {siv_genes}")
if siv_missing:
    print(f"  SIV genes missing: {siv_missing}")

if siv_genes:
    # --- Compute SIV metrics per cell ---
    def get_expr(adata_obj, gene):
        x = adata_obj[:, gene].X
        return x.toarray().ravel() if hasattr(x, 'toarray') else np.array(x).ravel()

    siv_matrix = np.column_stack([get_expr(adata, g) for g in siv_genes])
    adata.obs['siv_total_umi'] = siv_matrix.sum(axis=1)
    adata.obs['siv_n_genes'] = (siv_matrix > 0).sum(axis=1)
    adata.obs['is_siv_positive'] = (adata.obs['siv_total_umi'] > 0).astype(int)

    n_siv_pos = adata.obs['is_siv_positive'].sum()
    total_siv_umi = adata.obs['siv_total_umi'].sum()
    print(f"\n  Total SIV+ cells: {n_siv_pos:,} / {adata.n_obs:,} "
          f"({100*n_siv_pos/adata.n_obs:.2f}%)")
    print(f"  Total SIV UMIs: {total_siv_umi:,.0f}")

    # --- Per-gene expression table ---
    siv_results = []
    for treat in [D1MT, CART]:
        for wk in [5, 11, 15]:
            mask = ((adata.obs[TREATMENT_COL] == treat) &
                    (adata.obs[WEEK_COL] == wk))
            n_total = mask.sum()
            if n_total == 0:
                continue
            sub = adata[mask]
            for gene in siv_genes:
                expr = get_expr(sub, gene)
                n_pos = (expr > 0).sum()
                total_umi = expr.sum()
                siv_results.append({
                    'treatment': treat, 'week': wk, 'gene': gene,
                    'n_cells': n_total, 'n_positive': n_pos,
                    'pct_positive': round(100 * n_pos / n_total, 3),
                    'total_umi': round(total_umi, 2),
                    'umi_per_10k_cells': round(total_umi / n_total * 10000, 2),
                })
    df_siv = pd.DataFrame(siv_results)
    df_siv.to_csv(os.path.join(OUT_DIR, 'siv_gene_expression.csv'), index=False)

    # --- FIGURE Q4a: Normalized SIV UMIs per 10K cells by treatment × week ---
    print("\n  SIV UMI burden by treatment × week (normalized per 10K cells):")
    siv_burden = []
    for treat in [D1MT, CART]:
        for wk in [5, 11, 15]:
            mask = ((adata.obs[TREATMENT_COL] == treat) &
                    (adata.obs[WEEK_COL] == wk))
            n_total = mask.sum()
            umi_total = adata.obs.loc[mask, 'siv_total_umi'].sum()
            n_pos = adata.obs.loc[mask, 'is_siv_positive'].sum()
            norm_umi = umi_total / n_total * 10000 if n_total > 0 else 0
            siv_burden.append({
                'treatment': treat, 'week': wk,
                'n_cells': n_total, 'n_siv_pos': n_pos,
                'total_siv_umi': umi_total,
                'siv_umi_per_10k': round(norm_umi, 2),
            })
            print(f"    {treat:25s} W{wk}: {umi_total:>8,.0f} UMIs, "
                  f"{n_pos:>4,} cells, {norm_umi:.1f} UMI/10K cells")

    df_burden = pd.DataFrame(siv_burden)
    df_burden.to_csv(os.path.join(OUT_DIR, 'siv_umi_burden.csv'), index=False)

    # Bar chart: SIV UMIs per 10K cells
    fig, ax = plt.subplots(figsize=(8, 5))
    weeks = [5, 11, 15]
    x = np.arange(len(weeks))
    width = 0.35
    d1mt_vals = [df_burden[(df_burden['treatment'] == D1MT) &
                           (df_burden['week'] == w)]['siv_umi_per_10k'].values[0]
                 for w in weeks]
    cart_vals = [df_burden[(df_burden['treatment'] == CART) &
                           (df_burden['week'] == w)]['siv_umi_per_10k'].values[0]
                 for w in weeks]

    bars1 = ax.bar(x - width/2, d1mt_vals, width, label=D1MT,
                   color='#D32F2F', edgecolor='white')
    bars2 = ax.bar(x + width/2, cart_vals, width, label=CART,
                   color='#1976D2', edgecolor='white')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('SIV UMIs per 10,000 cells', fontsize=12)
    ax.set_title('Q4a: SIV Viral Burden (Normalized UMI Counts)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'W{w}' for w in weeks])
    ax.legend()
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'Q4a_siv_umi_burden.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'Q4a_siv_umi_burden.png'),
                bbox_inches='tight', dpi=300)
    print("  Saved: Q4a_siv_umi_burden.pdf/png")
    plt.close(fig)

    # --- FIGURE Q4b: SIV UMI burden by cell type at W11 ---
    print("\n  SIV UMI burden by cell type at W11:")
    ct_umi_rows = []
    for treat in [D1MT, CART]:
        mask_tw = ((adata.obs[TREATMENT_COL] == treat) &
                   (adata.obs[WEEK_COL] == 11))
        for ct in sorted(adata.obs[TIER2_COL].unique()):
            ct_mask = mask_tw & (adata.obs[TIER2_COL] == ct)
            n_total = ct_mask.sum()
            if n_total == 0:
                continue
            umi_total = adata.obs.loc[ct_mask, 'siv_total_umi'].sum()
            n_pos = adata.obs.loc[ct_mask, 'is_siv_positive'].sum()
            ct_umi_rows.append({
                'treatment': treat, 'cell_type': ct,
                'n_cells': n_total, 'n_siv_pos': n_pos,
                'total_siv_umi': umi_total,
                'siv_umi_per_10k': round(umi_total / n_total * 10000, 2),
                'pct_of_total_umi': 0,  # filled below
            })

    df_ct_umi = pd.DataFrame(ct_umi_rows)
    # Calculate % of total SIV UMIs within each treatment
    for treat in [D1MT, CART]:
        treat_total = df_ct_umi.loc[df_ct_umi['treatment'] == treat,
                                     'total_siv_umi'].sum()
        if treat_total > 0:
            df_ct_umi.loc[df_ct_umi['treatment'] == treat, 'pct_of_total_umi'] = (
                df_ct_umi.loc[df_ct_umi['treatment'] == treat, 'total_siv_umi'] /
                treat_total * 100).round(1)
    df_ct_umi.to_csv(os.path.join(OUT_DIR, 'siv_umi_by_celltype_W11.csv'), index=False)

    # Print summary
    for treat in [D1MT, CART]:
        sub = df_ct_umi[df_ct_umi['treatment'] == treat].sort_values(
            'total_siv_umi', ascending=False)
        print(f"\n  {treat} W11:")
        for _, row in sub.head(8).iterrows():
            print(f"    {row['cell_type']:35s}: {row['total_siv_umi']:>8,.0f} UMIs "
                  f"({row['pct_of_total_umi']:.1f}%), "
                  f"{row['siv_umi_per_10k']:.0f}/10K cells, "
                  f"{row['n_siv_pos']} SIV+ cells")

    # Stacked bar: % of total SIV UMIs by cell type for D1MT vs cART at W11
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, treat in enumerate([D1MT, CART]):
        ax = axes[idx]
        sub = df_ct_umi[(df_ct_umi['treatment'] == treat) &
                         (df_ct_umi['total_siv_umi'] > 0)].sort_values(
                            'total_siv_umi', ascending=True)
        if sub.empty:
            ax.set_title(f'{treat}: No SIV UMIs')
            continue
        ct_labels = sub['cell_type'].str.replace('Macrophages: ', 'Mac: ')
        ct_labels = ct_labels.str.replace('Monocytes: ', 'Mono: ')
        colors_ct = plt.cm.Set3(np.linspace(0, 1, len(sub)))
        ax.barh(range(len(sub)), sub['total_siv_umi'], color=colors_ct,
                edgecolor='white')
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(ct_labels, fontsize=8)
        ax.set_xlabel('Total SIV UMIs', fontsize=10)
        total_umi = sub['total_siv_umi'].sum()
        ax.set_title(f'{treat}\n({total_umi:,.0f} total SIV UMIs)',
                    fontsize=11, fontweight='bold')
        # Add percentage labels
        for i, (_, row) in enumerate(sub.iterrows()):
            if row['pct_of_total_umi'] >= 2:
                ax.text(row['total_siv_umi'] + total_umi*0.01, i,
                        f"{row['pct_of_total_umi']:.0f}%",
                        ha='left', va='center', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Q4b: SIV UMI Distribution by Cell Type — W11',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'Q4b_siv_celltype_breakdown_W11.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'Q4b_siv_celltype_breakdown_W11.png'),
                bbox_inches='tight', dpi=300)
    print("\n  Saved: Q4b_siv_celltype_breakdown_W11.pdf/png")
    plt.close(fig)

    # --- FIGURE Q4c: Per-gene SIV heatmap (normalized UMI per 10K cells) ---
    siv_umi_pivot = df_siv.pivot_table(
        index='gene', columns=['treatment', 'week'],
        values='umi_per_10k_cells', fill_value=0
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(siv_umi_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': 'UMIs per 10K cells'})
    ax.set_title('Q4c: SIV Gene Expression (Normalized UMI per 10K cells)\n'
                 'by Treatment × Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('SIV gene')
    ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'Q4c_siv_gene_heatmap.pdf'),
                bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(OUT_DIR, 'Q4c_siv_gene_heatmap.png'),
                bbox_inches='tight', dpi=300)
    print("  Saved: Q4c_siv_gene_heatmap.pdf/png")
    plt.close(fig)

    # --- Full cell type × treatment × week table ---
    siv_ct_all = []
    for treat in [D1MT, CART]:
        for wk in [5, 11, 15]:
            treat_wk = ((adata.obs[TREATMENT_COL] == treat) &
                        (adata.obs[WEEK_COL] == wk))
            for ct in sorted(adata.obs[TIER2_COL].unique()):
                ct_mask = treat_wk & (adata.obs[TIER2_COL] == ct)
                n_total = ct_mask.sum()
                if n_total == 0:
                    continue
                umi_total = adata.obs.loc[ct_mask, 'siv_total_umi'].sum()
                n_siv = adata.obs.loc[ct_mask, 'is_siv_positive'].sum()
                siv_ct_all.append({
                    'treatment': treat, 'week': wk, 'cell_type': ct,
                    'n_total': n_total, 'n_siv_pos': n_siv,
                    'pct_siv_pos': round(100 * n_siv / n_total, 3),
                    'total_siv_umi': round(umi_total, 2),
                    'siv_umi_per_10k': round(umi_total / n_total * 10000, 2),
                })
    df_siv_ct_all = pd.DataFrame(siv_ct_all)
    df_siv_ct_all.to_csv(os.path.join(OUT_DIR, 'siv_full_celltype_table.csv'),
                          index=False)
    print("  Saved: siv_full_celltype_table.csv")

else:
    print("  No SIV genes found — check features.tsv for gene naming")


# %% Cell 8 — Final Summary
# =========================================================================
print("\n" + "=" * 80)
print("PHASE 3b COMPLETE — DEG ANALYSIS AND PAPER FIGURES")
print("=" * 80)

print(f"\nAll outputs in: {OUT_DIR}")
print(f"\nDEG Summary ({len(all_deg)} comparisons):")
for name, res in all_deg.items():
    if res is None:
        print(f"  {name:40s}: SKIPPED")
    else:
        sig = res[res['pvals_adj'] < PADJ_THRESHOLD]
        n_up = (sig['logfoldchanges'] > 0).sum()
        n_down = (sig['logfoldchanges'] < 0).sum()
        print(f"  {name:40s}: {len(sig):>5,} sig ({n_up} up, {n_down} down)")

print("""
===================================================================
PAPER QUESTIONS → COMPARISONS → FIGURES
===================================================================

Q1: Does D1MT preserve CD4 T cells during Mtb/SIV coinfection?
    Comparisons: CD4eff D1MT vs cART at W5, W11, W15
    Finding: CD4 counts preserved (0.93x vs 0.40x at W15)
             W5+W11: ribosome/translation UP → active proliferation
             W15: TCR signaling + Th17 UP → mature effector shift
             W11: antiviral defense DOWN in D1MT (UP in Late cART)
    Figures:
      Q1_cd4_effector_panel.pdf    — Dotplot: Th1/Th17/exhaustion/survival
      Q6_cd4_ribosome_W5.pdf       — Dotplot: ribosome genes W5
      Q6_cd4_ribosome_W11.pdf      — Dotplot: ribosome genes W11
      Q6_cd4_antiviral_W11.pdf     — Dotplot: antiviral defense W11
      Q6_cd4_tcr_th17_W15.pdf      — Dotplot: TCR/Th17 W15
      enrichment_CD4eff_*.pdf      — Supplementary pathway barplots

Q2: What is the M1 IM-like inflammatory program?
    Comparisons: M1 vs IFN-resp (Priority 6), M1 vs M2 (Priority 5)
    Finding: Only 7 DEGs between M1 and IFN-resp (perfect boundary)
             M1 = TNF + IL1B (inflammatory)
             IFN-resp = IDO1 + CD274/PD-L1 + SOD2 (immunosuppressive)
    Figures:
      Q2_m1_boundary_degs.pdf      — Dotplot: 7 boundary genes
      Q2_m1_program_extended.pdf   — Dotplot: full M1 program across subtypes

Q3: How does D1MT reshape macrophage activation?
    Comparisons: All mac subtypes D1MT vs cART at W11
    Finding: M1 expansion 10.4x in D1MT; M2 spliceosome/lysosome UP;
             IFN-resp NF-κB signaling UP in D1MT
    Figures:
      Q3_mac_activation_w11.pdf    — Dotplot: all markers × 5 subtypes
      Q6_m2_spliceosome_*.pdf      — Dotplot: spliceosome/lysosome M2

Q4: Where is the SIV reservoir and how does D1MT affect it?
    Comparisons: SIV UMI counts by cell type × treatment
    Finding: D1MT W11 has 3x more SIV UMIs (more targets available);
             CD4 T cells are the dominant reservoir;
             Both groups suppress to near-zero by W15
    Figures:
      Q4a_siv_umi_burden.pdf       — Bar: normalized UMIs per 10K cells
      Q4b_siv_celltype_breakdown.pdf — Bar: cell type UMI distribution
      Q4c_siv_gene_heatmap.pdf     — Heatmap: per-gene normalized UMIs

Q5: Does D1MT alter macrophage-T cell communication?
    → Phase 3c LIANA output (already complete)

Q6: What pathways define the D1MT effect?
    → Focused gene panels replace opaque KEGG terms (see above)
    → Supplementary enrichment barplots in enrichment_*.pdf

Key outputs:
  deg_summary.csv                — Overview of all comparisons
  deg_*_sig.csv / deg_*_all.csv  — Significant / full DEG tables
  Q[1-4]_*.pdf                   — Paper question figures (dotplots/bars)
  Q6_*.pdf                       — Pathway gene panel dotplots
  enrichment_*.pdf               — Supplementary pathway barplots
  pathways/*.csv                 — Full enrichment results
  siv_*.csv                      — SIV expression and burden tables
""")
