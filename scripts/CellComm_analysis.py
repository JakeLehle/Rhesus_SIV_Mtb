#!/usr/bin/env python3
"""
Cell-Cell Communication Analysis — Macrophage ↔ T Cell Interactions
=====================================================================
Rhesus Mtb/SIV Coinfection scRNA-seq

Uses LIANA (LIgand-receptor ANalysis framework) to identify ligand-receptor
interactions between macrophage subtypes and CD4/CD8 T cells.

Key biological axes to examine:
  - IFN-γ signaling: T cell IFNG → Mac IFNGR1/IFNGR2 (activating)
  - IDO suppression: Mac CD274 (PD-L1) → T cell PDCD1 (PD-1) (suppressive)
  - Chemokine recruitment: Mac CXCL9/10/11 → T cell CXCR3
  - Antigen presentation: Mac MHC-II → T cell CD4/TCR
  - TNF/death signaling: bidirectional

Comparisons:
  - G1 vs G2 at W11 (D1MT effect on mac-T cell crosstalk)
  - G1 vs G4 at W11 (treatment strategy)
  - G4 W5 vs W13 vs W24 (resolution of interactions over time)

Install: pip install liana --break-system-packages

Run in Spyder (interactive) — 5 cells.
"""

# %% Cell 1 — Load and set up
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

ADATA_PATH = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/adata_annotated.h5ad'
OUT_DIR = '/master/jlehle/WORKING/SC/fastq/Rhesus/analysis/cellcomm/'
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading adata...")
adata = sc.read_h5ad(ADATA_PATH)
adata.obs['week'] = pd.to_numeric(adata.obs['week'], errors='coerce')
print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# Check LIANA installation
try:
    import liana as li
    from liana.mt import rank_aggregate
    print(f"  LIANA version: {li.__version__}")
    LIANA_AVAILABLE = True
except ImportError:
    print("  *** LIANA not installed ***")
    print("  Install with: pip install liana --break-system-packages")
    print("  Then restart Python and rerun.")
    LIANA_AVAILABLE = False

GROUP_COL = 'treatment_group'
ANIMAL_COL = 'animal_id'
WEEK_COL = 'week'

# Cell types to include in the interaction analysis
# Use tier2_full for subtypes, keep only mac subtypes + T cell subtypes
COMM_CELLTYPES = [
    'Macrophages: AM-like', 'Macrophages: TREM2+',
    'Macrophages: IM-like', 'Macrophages: IFN-responsive',
    'T cells: CD4', 'T cells: CD8',
]

# Biologically relevant L-R pairs to highlight
HIGHLIGHT_INTERACTIONS = {
    'IFN-γ activation': [
        ('IFNG', 'IFNGR1'), ('IFNG', 'IFNGR2'),
    ],
    'IDO/PD-L1 suppression': [
        ('CD274', 'PDCD1'),  # PD-L1 → PD-1
        ('CD274', 'CD80'),
    ],
    'Chemokine recruitment': [
        ('CXCL9', 'CXCR3'), ('CXCL10', 'CXCR3'), ('CXCL11', 'CXCR3'),
        ('CCL5', 'CCR5'), ('CCL3', 'CCR5'),
    ],
    'Antigen presentation': [
        ('CD74', 'CD4'),
    ],
    'TNF signaling': [
        ('TNF', 'TNFRSF1A'), ('TNF', 'TNFRSF1B'),
        ('TNFSF10', 'TNFRSF10A'), ('TNFSF10', 'TNFRSF10B'),  # TRAIL
    ],
    'Co-stimulation': [
        ('CD80', 'CD28'), ('CD80', 'CTLA4'),
        ('CD86', 'CD28'), ('CD86', 'CTLA4'),
    ],
}

# Flatten for quick lookup
highlight_pairs = set()
for pairs in HIGHLIGHT_INTERACTIONS.values():
    for l, r in pairs:
        highlight_pairs.add((l, r))

# Define run conditions: (name, group, week)
RUN_CONDITIONS = [
    ('G1_W5',  'D1MT_late_cART',  5),
    ('G1_W11', 'D1MT_late_cART', 11),
    ('G1_W15', 'D1MT_late_cART', 15),
    ('G2_W5',  'late_cART',       5),
    ('G2_W11', 'late_cART',      11),
    ('G2_W15', 'late_cART',      15),
    ('G4_W5',  'early_cART_3HP',  5),
    ('G4_W11', 'early_cART_3HP', 11),
    ('G4_W13', 'early_cART_3HP', 13),
    ('G4_W24', 'early_cART_3HP', 24),
]

# Pre-check cell counts for each condition
print("\n--- Cell counts per condition × cell type ---")
for name, grp, wk in RUN_CONDITIONS:
    mask = (adata.obs[GROUP_COL] == grp) & (adata.obs[WEEK_COL] == wk)
    ct_mask = adata.obs['tier2_full'].isin(COMM_CELLTYPES)
    n_total = (mask & ct_mask).sum()
    ct_counts = adata.obs.loc[mask & ct_mask, 'tier2_full'].value_counts()
    types_above_50 = (ct_counts >= 50).sum()
    print(f"  {name:8s}: {n_total:>6,} cells, {types_above_50}/{len(COMM_CELLTYPES)} "
          f"types with ≥50 cells")


# %% Cell 2 — Run LIANA for each condition
if not LIANA_AVAILABLE:
    print("\n*** LIANA not available — install and rerun ***")
else:
    print("\n" + "=" * 80)
    print("LIANA CELL-CELL COMMUNICATION ANALYSIS")
    print("=" * 80)

    MIN_CELLS_PER_TYPE = 30
    all_results = {}

    for name, grp, wk in RUN_CONDITIONS:
        print(f"\n--- {name} ({grp}, Week {wk}) ---")

        # Subset to this condition and relevant cell types
        mask = ((adata.obs[GROUP_COL] == grp) &
                (adata.obs[WEEK_COL] == wk) &
                (adata.obs['tier2_full'].isin(COMM_CELLTYPES)))
        sub = adata[mask].copy()

        if sub.n_obs < 200:
            print(f"  SKIP: only {sub.n_obs} cells total")
            all_results[name] = None
            continue

        # Check per-type counts, drop types with too few cells
        type_counts = sub.obs['tier2_full'].value_counts()
        valid_types = type_counts[type_counts >= MIN_CELLS_PER_TYPE].index.tolist()
        dropped = [t for t in COMM_CELLTYPES if t not in valid_types]

        if dropped:
            print(f"  Dropped (< {MIN_CELLS_PER_TYPE} cells): {dropped}")
            sub = sub[sub.obs['tier2_full'].isin(valid_types)].copy()

        # Need at least 2 cell types for interaction analysis
        if sub.obs['tier2_full'].nunique() < 2:
            print(f"  SKIP: <2 cell types remaining")
            all_results[name] = None
            continue

        # Ensure we have normalized data
        if 'raw_counts' in sub.layers:
            sub.X = sub.layers['raw_counts'].copy()
            sc.pp.normalize_total(sub, target_sum=1e4)
            sc.pp.log1p(sub)

        print(f"  Running LIANA on {sub.n_obs:,} cells, "
              f"{sub.obs['tier2_full'].nunique()} cell types...")

        try:
            li.mt.rank_aggregate(
                sub,
                groupby='tier2_full',
                resource_name='consensus',
                expr_prop=0.1,
                n_perms=1000,
                seed=42,
                verbose=False,
                use_raw=False,
            )

            # Extract results
            liana_res = sub.uns['liana_res']
            print(f"  Results: {len(liana_res):,} interaction-pair tests")

            # Filter to significant interactions
            # LIANA's aggregate rank: lower = more significant
            # magnitude_rank and specificity_rank are 0-1, lower = better
            sig = liana_res[liana_res['magnitude_rank'] <= 0.05].copy()
            print(f"  Significant (magnitude_rank ≤ 0.05): {len(sig)}")

            # Save full and significant results
            liana_res.to_csv(os.path.join(OUT_DIR, f'liana_{name}_all.csv'), index=False)
            if len(sig) > 0:
                sig.to_csv(os.path.join(OUT_DIR, f'liana_{name}_significant.csv'), index=False)

            all_results[name] = liana_res

        except Exception as e:
            print(f"  ERROR: {str(e)[:150]}")
            all_results[name] = None


# %% Cell 3 — Extract and compare key interactions across conditions
if LIANA_AVAILABLE and any(v is not None for v in all_results.values()):
    print("\n" + "=" * 80)
    print("KEY INTERACTION COMPARISON ACROSS CONDITIONS")
    print("=" * 80)

    # For each biological axis, find the interactions across conditions
    for axis_name, lr_pairs in HIGHLIGHT_INTERACTIONS.items():
        print(f"\n--- {axis_name} ---")

        for ligand, receptor in lr_pairs:
            row_data = []
            for name, grp, wk in RUN_CONDITIONS:
                res = all_results.get(name)
                if res is None:
                    continue

                # Find this L-R pair in results
                # LIANA stores ligand/receptor as complex names, try exact match first
                pair_mask = (
                    (res['ligand_complex'].str.contains(ligand, case=True, na=False)) &
                    (res['receptor_complex'].str.contains(receptor, case=True, na=False))
                )
                pair_res = res[pair_mask]

                if pair_res.empty:
                    continue

                # Get interactions between mac and T cell types
                for _, row in pair_res.iterrows():
                    source = row.get('source', '')
                    target = row.get('target', '')
                    mag_rank = row.get('magnitude_rank', 1.0)
                    spec_rank = row.get('specificity_rank', 1.0)

                    # Only report mac ↔ T cell interactions
                    is_mac = 'Macrophage' in str(source) or 'Macrophage' in str(target)
                    is_t = 'T cells' in str(source) or 'T cells' in str(target)

                    if is_mac and is_t:
                        sig_flag = '*' if mag_rank <= 0.05 else ' '
                        row_data.append({
                            'condition': name,
                            'source': source,
                            'target': target,
                            'ligand': ligand,
                            'receptor': receptor,
                            'magnitude_rank': mag_rank,
                            'specificity_rank': spec_rank,
                            'significant': sig_flag,
                        })

            if row_data:
                df_pair = pd.DataFrame(row_data)
                print(f"\n  {ligand} → {receptor}:")
                for _, r in df_pair.sort_values('condition').iterrows():
                    print(f"    {r['condition']:8s} | {r['source']:30s} → "
                          f"{r['target']:15s} | mag={r['magnitude_rank']:.3f} "
                          f"spec={r['specificity_rank']:.3f} {r['significant']}")
            else:
                print(f"  {ligand} → {receptor}: not detected in any condition")

    # --- Summary: top interactions per condition ---
    print("\n\n--- Top 15 Mac ↔ T cell interactions per condition ---")
    for name, grp, wk in RUN_CONDITIONS:
        res = all_results.get(name)
        if res is None:
            continue

        # Filter to mac-T interactions
        mac_t_mask = (
            ((res['source'].str.contains('Macrophage', na=False)) &
             (res['target'].str.contains('T cells', na=False))) |
            ((res['source'].str.contains('T cells', na=False)) &
             (res['target'].str.contains('Macrophage', na=False)))
        )
        mac_t = res[mac_t_mask].sort_values('magnitude_rank')

        print(f"\n  {name} (top 15 mac ↔ T cell):")
        for _, row in mac_t.head(15).iterrows():
            sig = '*' if row['magnitude_rank'] <= 0.05 else ' '
            print(f"    {sig} {row['source']:30s} → {row['target']:15s} | "
                  f"{row.get('ligand_complex', '?'):12s} → "
                  f"{row.get('receptor_complex', '?'):15s} | "
                  f"mag={row['magnitude_rank']:.3f}")


# %% Cell 4 — Visualization: interaction heatmaps
if LIANA_AVAILABLE and any(v is not None for v in all_results.values()):
    print("\n" + "=" * 80)
    print("INTERACTION HEATMAPS")
    print("=" * 80)

    # Compare G1 vs G2 vs G4 at Week 11 (the key divergence timepoint)
    w11_conditions = ['G1_W11', 'G2_W11', 'G4_W11']
    w11_results = {name: all_results.get(name) for name in w11_conditions
                   if all_results.get(name) is not None}

    if len(w11_results) >= 2:
        # Get all mac→T interactions across W11 conditions
        all_interactions = set()
        for name, res in w11_results.items():
            mac_t_mask = (
                ((res['source'].str.contains('Macrophage', na=False)) &
                 (res['target'].str.contains('T cells', na=False))) |
                ((res['source'].str.contains('T cells', na=False)) &
                 (res['target'].str.contains('Macrophage', na=False)))
            )
            sig_mask = res['magnitude_rank'] <= 0.05
            sig_pairs = res[mac_t_mask & sig_mask]
            for _, row in sig_pairs.iterrows():
                lr = f"{row.get('ligand_complex', '?')}_{row.get('receptor_complex', '?')}"
                all_interactions.add(lr)

        if all_interactions:
            print(f"  Total unique significant mac↔T interactions at W11: "
                  f"{len(all_interactions)}")

            # Build comparison matrix: interaction × condition
            heatmap_data = []
            for lr in sorted(all_interactions):
                row = {'interaction': lr}
                for name, res in w11_results.items():
                    mac_t_mask = (
                        ((res['source'].str.contains('Macrophage', na=False)) &
                         (res['target'].str.contains('T cells', na=False))) |
                        ((res['source'].str.contains('T cells', na=False)) &
                         (res['target'].str.contains('Macrophage', na=False)))
                    )
                    lr_parts = lr.split('_', 1)
                    if len(lr_parts) == 2:
                        pair_mask = (
                            mac_t_mask &
                            (res['ligand_complex'] == lr_parts[0]) &
                            (res['receptor_complex'] == lr_parts[1])
                        )
                        matches = res[pair_mask]
                        if not matches.empty:
                            row[name] = -np.log10(
                                matches['magnitude_rank'].values[0] + 1e-10)
                        else:
                            row[name] = 0
                    else:
                        row[name] = 0
                heatmap_data.append(row)

            df_heatmap = pd.DataFrame(heatmap_data).set_index('interaction')

            # Filter to top 30 most variable interactions
            df_heatmap['variance'] = df_heatmap.var(axis=1)
            df_top = df_heatmap.nlargest(30, 'variance').drop('variance', axis=1)

            if len(df_top) > 0:
                fig, ax = plt.subplots(figsize=(8, max(6, len(df_top) * 0.3)))
                im = ax.imshow(df_top.values, aspect='auto', cmap='YlOrRd')
                ax.set_xticks(range(len(df_top.columns)))
                ax.set_xticklabels(df_top.columns, rotation=45, ha='right')
                ax.set_yticks(range(len(df_top)))
                ax.set_yticklabels(df_top.index, fontsize=7)
                plt.colorbar(im, ax=ax, label='-log10(magnitude_rank)')
                ax.set_title('Mac ↔ T Cell Interactions at Week 11\n(top 30 most variable)',
                            fontsize=11, fontweight='bold')
                plt.tight_layout()
                plt.show()
                fig.savefig(os.path.join(OUT_DIR, 'heatmap_w11_mac_tcell_interactions.pdf'),
                            bbox_inches='tight', dpi=300)
                fig.savefig(os.path.join(OUT_DIR, 'heatmap_w11_mac_tcell_interactions.png'),
                            bbox_inches='tight', dpi=300)
                print("  Saved: heatmap_w11_mac_tcell_interactions.pdf/.png")
                plt.close(fig)

    # Export all results summary
    summary_rows = []
    for name, grp, wk in RUN_CONDITIONS:
        res = all_results.get(name)
        if res is None:
            summary_rows.append({'condition': name, 'status': 'SKIPPED',
                                'n_total': 0, 'n_sig': 0, 'n_mac_t_sig': 0})
        else:
            sig = res[res['magnitude_rank'] <= 0.05]
            mac_t = sig[
                ((sig['source'].str.contains('Macrophage', na=False)) &
                 (sig['target'].str.contains('T cells', na=False))) |
                ((sig['source'].str.contains('T cells', na=False)) &
                 (sig['target'].str.contains('Macrophage', na=False)))
            ]
            summary_rows.append({
                'condition': name, 'status': 'OK',
                'n_total': len(res), 'n_sig': len(sig),
                'n_mac_t_sig': len(mac_t),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, 'cellcomm_summary.csv'), index=False)
    print(f"\n  Summary saved: cellcomm_summary.csv")


# %% Cell 5 — Summary
print("\n" + "=" * 80)
print("CELL-CELL COMMUNICATION ANALYSIS COMPLETE")
print("=" * 80)

if LIANA_AVAILABLE:
    print(f"\nResults in: {OUT_DIR}")
    print(f"\nSummary:")
    print(summary_df.to_string(index=False))

print("""
Key outputs to review:
  1. liana_*_significant.csv — Significant L-R interactions per condition
  2. heatmap_w11_*.png — Comparison of interactions at the key timepoint
  3. Diagnostic text above — Biology-relevant L-R pairs across conditions

Key questions answered:
  - Does D1MT (G1 vs G2) change the PD-L1/PD-1 suppression axis?
  - Does IFN-γ signaling from T cells → macs differ between groups?
  - Does G4 (3HP) show resolution of inflammatory interactions over time?
  - Which mac subtype is the primary communicator with CD4 T cells?

Integration with DEG results:
  - Cross-reference top DEGs with ligands/receptors identified here
  - If CXCL9/10 are DEGs AND significant L-R interactions, that's convergent evidence
  - IDO1 won't appear as a ligand (it's an enzyme), but CD274 (PD-L1) will
""")
