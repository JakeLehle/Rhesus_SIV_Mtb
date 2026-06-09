#!/usr/bin/env python3
"""
Phase 3c — Cell-Cell Communication (LIANA+)
=============================================
Infers ligand-receptor interactions across macrophage subtypes, T cells,
monocytes, and NK cells, comparing D1MT + Late cART vs Late cART at
each timepoint to identify treatment-driven shifts in communication.

Strategy:
  - Run LIANA rank_aggregate on each treatment × timepoint subset (6 runs)
  - Compare interaction landscapes across conditions and timepoints
  - Track key biological axes: IFN-γ signaling, PD-L1/PD-1 suppression,
    chemokine recruitment (CXCL/CXCR3), antigen presentation

Key biological question:
  Does IDO inhibition (D1MT) alter the communication between macrophages
  and T cells in the granuloma, preserving activation signals and reducing
  suppressive signals compared to cART alone?

Note on gene symbols:
  Mmul10 uses MAMU- prefix for MHC genes (MAMU-DRA, MAMU-DRB1).
  LIANA's consensus resource uses human HLA- symbols. This script
  temporarily renames MAMU→HLA in a working copy so the resource
  matches correctly.

Run in Spyder (interactive cells with # %%) or standalone.
Output dir: analysis/phase3c_cellcomm/

Author: Jake Lehle / Kaushal Lab
Date: May 2026
"""

# %% Cell 1 — Configuration, Load, and Diagnostics
# =========================================================================
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# === PATHS ===
ADATA_IN = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_focal_annotated.h5ad'
OUT_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/phase3c_cellcomm/'
os.makedirs(OUT_DIR, exist_ok=True)

# === LIANA PARAMETERS ===
RESOURCE_NAME = 'consensus'    # LIANA's default curated LR resource
EXPR_PROP = 0.1                # Min proportion of cells expressing L or R
MIN_CELLS = 5                  # Min cells per cell type in a subset
N_PERMS = 1000                 # Permutations for p-value calculation
N_JOBS = 4                     # Parallel jobs for permutation tests

# === CELL TYPES TO INCLUDE ===
# Using tier2_full labels. Adjust this list to add/remove populations.
CELLTYPES_TO_INCLUDE = [
    'Macrophages: AM-like',
    'Macrophages: TREM2+',
    'Macrophages: M2 IM-like',
    'Macrophages: IFN-responsive',
    'Macrophages: M1 IM-like',
    'T cells: CD4',
    'T cells: CD8',
    'Monocytes: Classical',
    'NK cells',
]

# === MAMU → HLA GENE RENAMING ===
# Mmul10 uses MAMU- prefix for MHC genes; LIANA resource expects HLA-
MAMU_TO_HLA = {
    'MAMU-DRA':  'HLA-DRA',
    'MAMU-DRB1': 'HLA-DRB1',
    'MAMU-DRB5': 'HLA-DRB5',
    'MAMU-DPA1': 'HLA-DPA1',
    'MAMU-DPB1': 'HLA-DPB1',
    'MAMU-DQA1': 'HLA-DQA1',
    'MAMU-DQB1': 'HLA-DQB1',
    'MAMU-A':    'HLA-A',
    'MAMU-B':    'HLA-B',
    'MAMU-E':    'HLA-E',
    'MAMU-F':    'HLA-F',
}

# === KEY BIOLOGICAL INTERACTIONS TO TRACK ===
# These are the axes we expect D1MT to modulate
KEY_INTERACTIONS = [
    # IFN-γ activation axis (T cell → Macrophage)
    ('IFNG', 'IFNGR1'),
    ('IFNG', 'IFNGR2'),
    # PD-L1/PD-1 suppression axis (Macrophage → T cell)
    ('CD274', 'PDCD1'),
    # Chemokine recruitment (Macrophage → T cell)
    ('CXCL9', 'CXCR3'),
    ('CXCL10', 'CXCR3'),
    ('CXCL11', 'CXCR3'),
    # Antigen presentation (Macrophage → T cell)
    ('HLA-DRA', 'CD4'),       # Note: uses HLA name after renaming
    ('HLA-DRB1', 'CD4'),
    ('CD74', 'CD4'),
    # TNF signaling
    ('TNF', 'TNFRSF1A'),
    ('TNF', 'TNFRSF1B'),
    # Co-stimulation / co-inhibition
    ('CD80', 'CD28'),
    ('CD86', 'CD28'),
    ('CD80', 'CTLA4'),
    ('CD86', 'CTLA4'),
]

# === TREATMENT / METADATA COLS ===
TREATMENT_COL = 'treatment'
WEEK_COL = 'week'
CELLTYPE_COL = 'tier2_full'

TREATMENT_CONFIG = {
    'D1MT + Late cART': {'color': '#D32F2F'},
    'Late cART':        {'color': '#1976D2'},
}

# === LOAD ===
print("Loading annotated data...")
adata = sc.read_h5ad(ADATA_IN)
adata.obs[WEEK_COL] = pd.to_numeric(adata.obs[WEEK_COL], errors='coerce').astype(int)
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# === VERIFY LIANA ===
try:
    import liana as li
    print(f"  LIANA version: {li.__version__}")
except ImportError:
    raise ImportError("LIANA not installed. Run: pip install liana --break-system-packages")


# %% Cell 2 — Prepare AnnData for LIANA
# =========================================================================
# 1. Subset to selected cell types
# 2. Rename MAMU→HLA genes so LIANA resource matches
# 3. Diagnostic: check which key genes and interactions are available
# =========================================================================
print("\n" + "=" * 80)
print("PREPARING DATA FOR LIANA")
print("=" * 80)

# --- Subset to selected cell types ---
ct_mask = adata.obs[CELLTYPE_COL].isin(CELLTYPES_TO_INCLUDE)
adata_liana = adata[ct_mask].copy()
print(f"\n  Subset to {len(CELLTYPES_TO_INCLUDE)} cell types: "
      f"{adata_liana.n_obs:,} cells (from {adata.n_obs:,})")

print(f"\n  Cell type counts:")
ct_counts = adata_liana.obs[CELLTYPE_COL].value_counts()
for ct, n in ct_counts.items():
    print(f"    {ct}: {n:,}")

# --- Per treatment × week cell type counts (check for underpowered bins) ---
print(f"\n  Cell counts per treatment × week × celltype:")
per_bin = (
    adata_liana.obs
    .groupby([TREATMENT_COL, WEEK_COL, CELLTYPE_COL], observed=True)
    .size()
    .reset_index(name='n_cells')
)
per_bin_wide = per_bin.pivot_table(
    index=CELLTYPE_COL, columns=[TREATMENT_COL, WEEK_COL],
    values='n_cells', fill_value=0
)
print(per_bin_wide.to_string())

# Flag bins below MIN_CELLS
low_bins = per_bin[per_bin['n_cells'] < MIN_CELLS]
if len(low_bins) > 0:
    print(f"\n  *** WARNING: {len(low_bins)} bins have < {MIN_CELLS} cells ***")
    print(low_bins.to_string(index=False))
    print("  These cell types will be excluded from those specific runs.")
else:
    print(f"\n  All bins have >= {MIN_CELLS} cells.")

per_bin_wide.to_csv(os.path.join(OUT_DIR, 'celltype_counts_per_bin.csv'))

# --- Rename MAMU → HLA genes ---
print(f"\n  Renaming MAMU → HLA genes for LIANA resource compatibility...")
rename_map = {}
for mamu, hla in MAMU_TO_HLA.items():
    if mamu in adata_liana.var_names:
        rename_map[mamu] = hla
        print(f"    {mamu} → {hla}")
    else:
        print(f"    {mamu} — not in var_names, skip")

if rename_map:
    adata_liana.var_names = [rename_map.get(g, g) for g in adata_liana.var_names]
    # Also rename in .raw if it exists
    if adata_liana.raw is not None:
        raw_adata = adata_liana.raw.to_adata()
        raw_adata.var_names = [rename_map.get(g, g) for g in raw_adata.var_names]
        adata_liana.raw = raw_adata
    print(f"  Renamed {len(rename_map)} genes.")
else:
    print("  No MAMU genes found to rename.")

# --- Check which key interaction genes are present ---
print(f"\n  Checking key interaction genes in dataset:")
all_key_genes = set()
for lig, rec in KEY_INTERACTIONS:
    all_key_genes.add(lig)
    all_key_genes.add(rec)

present_key = [g for g in sorted(all_key_genes) if g in adata_liana.var_names]
missing_key = [g for g in sorted(all_key_genes) if g not in adata_liana.var_names]

print(f"    Present: {len(present_key)}/{len(all_key_genes)}")
for g in present_key:
    n_expr = (adata_liana[:, g].X.toarray().flatten() > 0).sum() if hasattr(adata_liana[:, g].X, 'toarray') else (adata_liana[:, g].X.flatten() > 0).sum()
    print(f"      {g}: {n_expr:,} cells expressing ({100*n_expr/adata_liana.n_obs:.1f}%)")
if missing_key:
    print(f"    Missing: {missing_key}")
    print("    Interactions involving these genes will not be scored.")

# --- Check what LIANA resource contains for our key interactions ---
print(f"\n  Checking LIANA '{RESOURCE_NAME}' resource for key interactions...")
resource = li.rs.select_resource(RESOURCE_NAME)
print(f"    Resource has {len(resource)} ligand-receptor pairs")

resource_pairs = set(zip(resource['ligand'].values, resource['receptor'].values))
for lig, rec in KEY_INTERACTIONS:
    in_resource = (lig, rec) in resource_pairs
    both_present = lig in adata_liana.var_names and rec in adata_liana.var_names
    status = "OK" if in_resource and both_present else ""
    if not in_resource:
        status = "NOT IN RESOURCE"
    elif not both_present:
        missing_gene = lig if lig not in adata_liana.var_names else rec
        status = f"GENE MISSING: {missing_gene}"
    print(f"    {lig:12s} → {rec:12s}  [{status}]")

# --- Also check for complex-based interactions ---
# LIANA handles complexes — check if any of our genes appear as subunits
print(f"\n  Checking for complex interactions involving key genes...")
for gene in sorted(present_key)[:10]:
    as_ligand = resource[resource['ligand'].str.contains(gene, na=False)]
    as_receptor = resource[resource['receptor'].str.contains(gene, na=False)]
    n_interactions = len(as_ligand) + len(as_receptor)
    if n_interactions > 0:
        print(f"    {gene}: {len(as_ligand)} as ligand, {len(as_receptor)} as receptor")


# %% Cell 3 — Run LIANA on each treatment × timepoint subset
# =========================================================================
# 6 runs total: 2 treatments × 3 timepoints
# Results stored in a dict and also in adata.uns per run
# =========================================================================
print("\n" + "=" * 80)
print("RUNNING LIANA RANK AGGREGATE")
print("=" * 80)

treatments = sorted(adata_liana.obs[TREATMENT_COL].unique())
weeks = sorted(adata_liana.obs[WEEK_COL].unique())

liana_results = {}  # {(treatment, week): DataFrame}
run_summary = []

for treatment in treatments:
    for week in weeks:
        run_key = (treatment, week)
        mask = (
            (adata_liana.obs[TREATMENT_COL] == treatment) &
            (adata_liana.obs[WEEK_COL] == week)
        )
        sub = adata_liana[mask].copy()
        n_cells = sub.n_obs

        # Check cell type coverage
        ct_in_sub = sub.obs[CELLTYPE_COL].value_counts()
        ct_above_min = ct_in_sub[ct_in_sub >= MIN_CELLS]
        ct_below_min = ct_in_sub[ct_in_sub < MIN_CELLS]

        print(f"\n  --- {treatment} | Week {week} ---")
        print(f"  Cells: {n_cells:,}")
        print(f"  Cell types with >= {MIN_CELLS} cells: {len(ct_above_min)}")
        if len(ct_below_min) > 0:
            print(f"  Cell types BELOW threshold (excluded by LIANA):")
            for ct, n in ct_below_min.items():
                print(f"    {ct}: {n}")

        if n_cells < 100:
            print(f"  *** SKIPPING: only {n_cells} cells ***")
            run_summary.append({
                'treatment': treatment, 'week': week,
                'n_cells': n_cells, 'n_celltypes': len(ct_above_min),
                'n_interactions': 0, 'status': 'SKIPPED'
            })
            continue

        # Run LIANA rank_aggregate
        try:
            li.mt.rank_aggregate(
                sub,
                groupby=CELLTYPE_COL,
                resource_name=RESOURCE_NAME,
                expr_prop=EXPR_PROP,
                min_cells=MIN_CELLS,
                n_perms=N_PERMS,
                n_jobs=N_JOBS,
                use_raw=True,
                verbose=True,
                inplace=True,
                key_added='liana_res',
            )

            res = sub.uns['liana_res'].copy()
            res['treatment'] = treatment
            res['week'] = week
            liana_results[run_key] = res

            n_interactions = len(res)
            n_sig = (res['specificity_rank'] <= 0.05).sum() if 'specificity_rank' in res.columns else 0

            print(f"  Results: {n_interactions:,} interactions, {n_sig:,} significant (spec_rank <= 0.05)")

            run_summary.append({
                'treatment': treatment, 'week': week,
                'n_cells': n_cells, 'n_celltypes': len(ct_above_min),
                'n_interactions': n_interactions, 'n_significant': n_sig,
                'status': 'OK'
            })

        except Exception as e:
            print(f"  *** ERROR: {e} ***")
            run_summary.append({
                'treatment': treatment, 'week': week,
                'n_cells': n_cells, 'n_celltypes': len(ct_above_min),
                'n_interactions': 0, 'status': f'ERROR: {e}'
            })

# Run summary
df_runs = pd.DataFrame(run_summary)
print(f"\n\n--- Run Summary ---")
print(df_runs.to_string(index=False))
df_runs.to_csv(os.path.join(OUT_DIR, 'liana_run_summary.csv'), index=False)

# Combine all results into one DataFrame
if liana_results:
    df_all = pd.concat(liana_results.values(), ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, 'liana_all_results.csv'), index=False)
    print(f"\n  Combined results: {len(df_all):,} rows across {len(liana_results)} runs")
else:
    print("\n  *** No LIANA results generated ***")
    df_all = pd.DataFrame()


# %% Cell 4 — Track Key Biological Interactions Across Conditions
# =========================================================================
# Extract the specific interactions we care about and compare
# across the 2×3 treatment-timepoint grid
# =========================================================================
print("\n" + "=" * 80)
print("KEY BIOLOGICAL INTERACTIONS — Cross-Condition Comparison")
print("=" * 80)

if df_all.empty:
    print("  No results to analyze. Check Cell 3 for errors.")
else:
    # Build a focused lookup for key interactions
    key_rows = []

    for (treatment, week), res_df in liana_results.items():
        for lig, rec in KEY_INTERACTIONS:
            # LIANA may store complexes — match on ligand_complex and receptor_complex too
            match = res_df[
                ((res_df['ligand_complex'].str.contains(lig, na=False)) &
                 (res_df['receptor_complex'].str.contains(rec, na=False)))
            ]

            if len(match) > 0:
                for _, row in match.iterrows():
                    key_rows.append({
                        'treatment': treatment,
                        'week': week,
                        'ligand': lig,
                        'receptor': rec,
                        'ligand_complex': row.get('ligand_complex', ''),
                        'receptor_complex': row.get('receptor_complex', ''),
                        'source': row['source'],
                        'target': row['target'],
                        'magnitude_rank': row.get('magnitude_rank', np.nan),
                        'specificity_rank': row.get('specificity_rank', np.nan),
                        'lr_means': row.get('lr_means', np.nan),
                    })

    if key_rows:
        df_key = pd.DataFrame(key_rows)
        df_key = df_key.sort_values(['ligand', 'receptor', 'source', 'target',
                                      'treatment', 'week'])

        print(f"\n  Found {len(df_key)} instances of key interactions across all runs")

        # Print summary
        for (lig, rec), grp in df_key.groupby(['ligand', 'receptor']):
            print(f"\n  {lig} → {rec}:")
            for _, row in grp.iterrows():
                sig_flag = "*" if row['magnitude_rank'] <= 0.05 else " "
                print(f"    {sig_flag} {row['treatment']:25s} W{row['week']:2d} | "
                      f"{row['source']:30s} → {row['target']:20s} | "
                      f"mag_rank={row['magnitude_rank']:.4f}  "
                      f"spec_rank={row['specificity_rank']:.4f}")

        df_key.to_csv(os.path.join(OUT_DIR, 'key_interactions_tracked.csv'), index=False)
        print(f"\n  Saved: key_interactions_tracked.csv")
    else:
        print("  *** None of the key interactions found in results ***")
        print("  This may indicate gene symbol mismatch or low expression.")
        print("  Check Cell 2 diagnostics for missing genes.")
        df_key = pd.DataFrame()


# %% Cell 5 — Dotplots: Top Interactions per Condition-Timepoint
# =========================================================================
print("\n" + "=" * 80)
print("DOTPLOTS — Top Interactions per Condition × Timepoint")
print("=" * 80)

if df_all.empty:
    print("  No results to plot.")
else:
    # Focus on macrophage ↔ T cell interactions
    mac_labels = [ct for ct in CELLTYPES_TO_INCLUDE if 'Macrophage' in ct]
    tc_labels = [ct for ct in CELLTYPES_TO_INCLUDE if 'T cells' in ct]

    for (treatment, week), res_df in liana_results.items():
        safe_name = f"{treatment.replace(' ', '_').replace('+', '')}__W{week}"

        # Store results in a temporary adata for LIANA plotting
        # LIANA's plotting functions expect results in adata.uns
        sub_mask = (
            (adata_liana.obs[TREATMENT_COL] == treatment) &
            (adata_liana.obs[WEEK_COL] == week)
        )
        sub = adata_liana[sub_mask].copy()
        sub.uns['liana_res'] = res_df

        try:
            p = li.pl.dotplot(
                adata=sub,
                colour='magnitude_rank',
                size='specificity_rank',
                inverse_size=True,
                inverse_colour=True,
                source_labels=mac_labels,
                target_labels=tc_labels,
                top_n=20,
                orderby='magnitude_rank',
                orderby_ascending=True,
                figure_size=(12, 10),
                filter_fun=lambda x: x['specificity_rank'] <= 0.05,
            )
            p.save(os.path.join(OUT_DIR, f'dotplot_mac_to_tcell_{safe_name}.pdf'))
            print(f"  Saved: dotplot_mac_to_tcell_{safe_name}.pdf")

        except Exception as e:
            print(f"  *** Dotplot failed for {treatment} W{week}: {e} ***")

        # Also do T cell → Macrophage direction
        try:
            p2 = li.pl.dotplot(
                adata=sub,
                colour='magnitude_rank',
                size='specificity_rank',
                inverse_size=True,
                inverse_colour=True,
                source_labels=tc_labels,
                target_labels=mac_labels,
                top_n=20,
                orderby='magnitude_rank',
                orderby_ascending=True,
                figure_size=(12, 10),
                filter_fun=lambda x: x['specificity_rank'] <= 0.05,
            )
            p2.save(os.path.join(OUT_DIR, f'dotplot_tcell_to_mac_{safe_name}.pdf'))
            print(f"  Saved: dotplot_tcell_to_mac_{safe_name}.pdf")

        except Exception as e:
            print(f"  *** Dotplot failed for T→Mac {treatment} W{week}: {e} ***")


# %% Cell 6 — Interaction Trajectory Heatmaps
# =========================================================================
# For key interactions, show how magnitude_rank changes across the
# 2 treatments × 3 timepoints grid
# =========================================================================
print("\n" + "=" * 80)
print("INTERACTION TRAJECTORIES — Key Axes Over Time")
print("=" * 80)

if df_all.empty or (isinstance(df_key, pd.DataFrame) and df_key.empty):
    print("  No key interaction data to plot.")
else:
    # Build a pivoted summary: for each key interaction + source→target pair,
    # show magnitude_rank across all condition-timepoints
    trajectory_rows = []

    for (lig, rec), inter_group in df_key.groupby(['ligand', 'receptor']):
        for (src, tgt), pair_group in inter_group.groupby(['source', 'target']):
            row = {
                'interaction': f'{lig}→{rec}',
                'source_target': f'{src} → {tgt}',
            }
            for _, r in pair_group.iterrows():
                col_name = f"{r['treatment']}|W{r['week']}"
                row[f'mag_{col_name}'] = r['magnitude_rank']
                row[f'spec_{col_name}'] = r['specificity_rank']
            trajectory_rows.append(row)

    if trajectory_rows:
        df_traj = pd.DataFrame(trajectory_rows)
        df_traj.to_csv(os.path.join(OUT_DIR, 'key_interaction_trajectories.csv'), index=False)
        print(f"  Saved trajectory table: {len(df_traj)} interaction-pairs tracked")
        print(df_traj.to_string(index=False))

    # --- Plot: magnitude rank trajectories for key axes ---
    # Group interactions by biological axis
    AXIS_GROUPS = {
        'IFN-γ signaling': [('IFNG', 'IFNGR1'), ('IFNG', 'IFNGR2')],
        'PD-L1 / PD-1 suppression': [('CD274', 'PDCD1')],
        'Chemokine recruitment': [('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'), ('CXCL11', 'CXCR3')],
        'Antigen presentation': [('HLA-DRA', 'CD4'), ('HLA-DRB1', 'CD4'), ('CD74', 'CD4')],
        'Co-stimulation': [('CD80', 'CD28'), ('CD86', 'CD28')],
    }

    for axis_name, axis_pairs in AXIS_GROUPS.items():
        # Filter to interactions in this axis
        axis_data = df_key[
            df_key.apply(lambda r: (r['ligand'], r['receptor']) in axis_pairs, axis=1)
        ]

        if axis_data.empty:
            print(f"\n  {axis_name}: no interactions found, skipping")
            continue

        # Get unique source→target pairs for this axis
        st_pairs = axis_data.groupby(['source', 'target']).size().reset_index(name='n')
        n_pairs = len(st_pairs)

        if n_pairs == 0:
            continue

        print(f"\n  {axis_name}: {len(axis_data)} entries across {n_pairs} cell-type pairs")

        ncols = min(3, n_pairs)
        nrows = int(np.ceil(n_pairs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle(f'{axis_name}', fontsize=16, fontweight='bold', y=1.02)

        for pidx, (_, pair_row) in enumerate(st_pairs.iterrows()):
            ax = axes[pidx // ncols, pidx % ncols]
            src, tgt = pair_row['source'], pair_row['target']

            pair_data = axis_data[
                (axis_data['source'] == src) & (axis_data['target'] == tgt)
            ]

            # Average across all LR pairs in this axis for this cell-type pair
            for treatment, cfg in TREATMENT_CONFIG.items():
                treat_data = pair_data[pair_data['treatment'] == treatment]
                if treat_data.empty:
                    continue

                # Average magnitude_rank across LR pairs per week
                week_avg = (
                    treat_data
                    .groupby('week')['magnitude_rank']
                    .mean()
                    .reset_index()
                    .sort_values('week')
                )

                ax.plot(week_avg['week'], week_avg['magnitude_rank'],
                        marker='o', markersize=10, linewidth=2.5,
                        color=cfg['color'], label=treatment,
                        markeredgecolor='white', markeredgewidth=1.5)

            # Shorten labels for display
            src_short = src.replace('Macrophages: ', 'Mac:').replace('T cells: ', 'T:')
            tgt_short = tgt.replace('Macrophages: ', 'Mac:').replace('T cells: ', 'T:')
            ax.set_title(f'{src_short} → {tgt_short}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Week', fontsize=11)
            ax.set_ylabel('Magnitude rank\n(lower = stronger)', fontsize=11)
            ax.set_xlim(3, 17)
            ax.invert_yaxis()  # Lower rank = stronger interaction
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if pidx == 0:
                ax.legend(fontsize=10, frameon=True)

        for pidx in range(n_pairs, nrows * ncols):
            axes[pidx // ncols, pidx % ncols].set_visible(False)

        plt.tight_layout()
        plt.show()
        safe_axis = axis_name.lower().replace(' ', '_').replace('/', '_')
        fig.savefig(os.path.join(OUT_DIR, f'trajectory_{safe_axis}.pdf'),
                    bbox_inches='tight')
        fig.savefig(os.path.join(OUT_DIR, f'trajectory_{safe_axis}.png'),
                    bbox_inches='tight', dpi=300)
        print(f"  Saved: trajectory_{safe_axis}")
        plt.close(fig)


# %% Cell 7 — Top Differential Interactions Between Treatments
# =========================================================================
# For each timepoint, find interactions that differ most between
# D1MT + Late cART and Late cART
# =========================================================================
print("\n" + "=" * 80)
print("DIFFERENTIAL INTERACTIONS — D1MT + Late cART vs Late cART")
print("=" * 80)

if len(liana_results) < 2:
    print("  Need results from both treatments to compare.")
else:
    diff_tables = []

    for week in weeks:
        key_d1mt = ('D1MT + Late cART', week)
        key_cart = ('Late cART', week)

        if key_d1mt not in liana_results or key_cart not in liana_results:
            print(f"\n  Week {week}: missing one or both treatments, skipping")
            continue

        res_d1mt = liana_results[key_d1mt]
        res_cart = liana_results[key_cart]

        # Merge on interaction + cell type pair
        merge_cols = ['ligand_complex', 'receptor_complex', 'source', 'target']
        merged = res_d1mt[merge_cols + ['magnitude_rank', 'specificity_rank']].merge(
            res_cart[merge_cols + ['magnitude_rank', 'specificity_rank']],
            on=merge_cols,
            suffixes=('_d1mt', '_cart'),
            how='inner'
        )

        # Compute rank difference (negative = stronger in D1MT)
        merged['mag_rank_diff'] = merged['magnitude_rank_d1mt'] - merged['magnitude_rank_cart']
        merged['week'] = week

        # Top interactions stronger in D1MT (lower rank = stronger)
        stronger_d1mt = merged.nsmallest(20, 'mag_rank_diff')
        # Top interactions stronger in Late cART
        stronger_cart = merged.nlargest(20, 'mag_rank_diff')

        print(f"\n  --- Week {week} ---")
        print(f"  Total comparable interactions: {len(merged):,}")
        print(f"\n  Top 10 STRONGER in D1MT + Late cART (negative diff):")
        for _, row in stronger_d1mt.head(10).iterrows():
            print(f"    {row['ligand_complex']:15s} → {row['receptor_complex']:15s} | "
                  f"{row['source']:25s} → {row['target']:20s} | "
                  f"diff={row['mag_rank_diff']:.4f}")

        print(f"\n  Top 10 STRONGER in Late cART (positive diff):")
        for _, row in stronger_cart.head(10).iterrows():
            print(f"    {row['ligand_complex']:15s} → {row['receptor_complex']:15s} | "
                  f"{row['source']:25s} → {row['target']:20s} | "
                  f"diff={row['mag_rank_diff']:.4f}")

        diff_tables.append(merged)

    if diff_tables:
        df_diff = pd.concat(diff_tables, ignore_index=True)
        df_diff.to_csv(os.path.join(OUT_DIR, 'differential_interactions.csv'), index=False)
        print(f"\n  Saved: differential_interactions.csv ({len(df_diff):,} rows)")


# %% Cell 8 — Summary and Export
# =========================================================================
print("\n" + "=" * 80)
print("SUMMARY AND EXPORT")
print("=" * 80)

# Export individual run results
for (treatment, week), res_df in liana_results.items():
    safe = f"{treatment.replace(' ', '_').replace('+', '')}__W{week}"
    res_df.to_csv(os.path.join(OUT_DIR, f'liana_results_{safe}.csv'), index=False)

print(f"  Individual run CSVs saved to {OUT_DIR}")

# Print overall summary
print(f"\n  === LIANA Analysis Summary ===")
print(f"  Runs completed: {sum(1 for r in run_summary if r['status'] == 'OK')}/{len(run_summary)}")
print(f"  Cell types included: {len(CELLTYPES_TO_INCLUDE)}")
print(f"  Resource: {RESOURCE_NAME}")
print(f"  Permutations: {N_PERMS}")
if liana_results:
    total_interactions = sum(len(r) for r in liana_results.values())
    print(f"  Total interactions scored: {total_interactions:,}")

print(f"\n  Output directory: {OUT_DIR}")
print(f"\n  Key output files:")
print(f"    liana_run_summary.csv             — run diagnostics")
print(f"    liana_all_results.csv             — combined results from all runs")
print(f"    key_interactions_tracked.csv      — biological axes tracked")
print(f"    key_interaction_trajectories.csv  — trajectories across conditions")
print(f"    differential_interactions.csv     — D1MT vs Late cART comparisons")
print(f"    dotplot_*.pdf                     — per-run interaction dotplots")
print(f"    trajectory_*.pdf                  — biological axis trajectory plots")

print(f"\n  Next steps:")
print(f"    1. Review dotplots for unexpected or missing interactions")
print(f"    2. Check if MAMU→HLA renaming captured all MHC interactions")
print(f"    3. Examine differential_interactions.csv for DEG targets")
print(f"    4. Consider adding more cell types if key interactions are missing")
print(f"    5. Adjust EXPR_PROP or MIN_CELLS if too many interactions filtered")
