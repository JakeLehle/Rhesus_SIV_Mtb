#!/bin/bash
#SBATCH -J Rhesus_scRNA_pipeline
#SBATCH -o /master/jlehle/WORKING/LOGS/Rhesus_scRNA_pipeline.o.log
#SBATCH -e /master/jlehle/WORKING/LOGS/Rhesus_scRNA_pipeline.e.log
#SBATCH -t 5-00:00:00
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 80
#SBATCH --mem 800GB

set -euo pipefail

# ============================================
# Rhesus Mtb/SIV Coinfection scRNA-seq Pipeline
# ============================================
# Focal analysis: D1MT + Late cART vs Late cART
# Runs Phase 1 → 2 → 3a sequentially.
# Comment out completed phases to re-run from a specific point.
#
# Phase 1: Load cellranger output, QC, normalize, BBKNN, UMAP
# Phase 2: Tiered annotation, biological scoring, diagnostics
# Phase 3a: Time series composition and fold-change
# ============================================

# --- Environment ---
source $HOME/anaconda3/bin/activate
conda activate sc_pre
export MPLBACKEND=Agg    # Headless matplotlib for SLURM

WORKDIR="/master/jlehle/WORKING/SC/fastq/Rhesus"
cd "${WORKDIR}"

echo "============================================"
echo "RHESUS scRNA-seq PIPELINE — $(date)"
echo "Working dir: ${WORKDIR}"
echo "Conda env:   ${CONDA_DEFAULT_ENV}"
echo "============================================"

# ============================================
# PHASE 1: Load, QC, Normalize, Embed
# ============================================
echo ""
echo "============================================"
echo "PHASE 1: Load, QC, Normalize, Embed"
echo "Start: $(date)"
echo "============================================"

#python "${WORKDIR}/Phase1_scanpy_qc.py"

echo "PHASE 1 COMPLETE: $(date)"

# ============================================
# PHASE 2: Tiered Annotation and Diagnostics
# ============================================
echo ""
echo "============================================"
echo "PHASE 2: Tiered Annotation and Diagnostics"
echo "Start: $(date)"
echo "============================================"

#python "${WORKDIR}/Phase2_annotation.py"

echo "PHASE 2 COMPLETE: $(date)"

# ============================================
# PHASE 3a: Time Series Composition
# ============================================
echo ""
echo "============================================"
echo "PHASE 3a: Time Series Composition"
echo "Start: $(date)"
echo "============================================"

#python "${WORKDIR}/Phase3a_timeseries.py"
#python "${WORKDIR}/Phase3b_LIANA.py"
echo "PHASE 3a COMPLETE: $(date)"

# ============================================
# PHASE 3b: DEG and Pathway Analysis (TODO)
# ============================================
# echo ""
# echo "============================================"
# echo "PHASE 4: DEG and Pathway Analysis"
# echo "Start: $(date)"
# echo "============================================"
conda run -n sc_pre python "${WORKDIR}/Phase3b_DEG.py"
# echo "PHASE 3b COMPLETE: $(date)"

# ============================================
# PHASE 3c: Cell-Cell Communication (TODO)
# ============================================
# echo ""
# echo "============================================"
# echo "PHASE 3c: Cell-Cell Communication (LIANA)"
# echo "Start: $(date)"
# echo "============================================"
# python "${WORKDIR}/Phase3c_cellcomm.py"
# echo "PHASE 3c COMPLETE: $(date)"

# ============================================
# PHASE 3d: SIV Dynamics Profiling (TODO)
# ============================================
# echo ""
# echo "============================================"
# echo "PHASE 3d: SIV Dynamics Profiling"
# echo "Start: $(date)"
# echo "============================================"
# python "${WORKDIR}/Phase3d_siv.py"
# echo "PHASE 3d COMPLETE: $(date)"

echo ""
echo "============================================"
echo "PIPELINE COMPLETE — $(date)"
echo "============================================"
echo "Outputs:"
echo "  Phase 1: analysis/phase1_qc/"
echo "           analysis/adata_focal_qcd.h5ad"
echo "  Phase 2: analysis/phase2_annotation/"
echo "           analysis/phase2_timeseries/"
echo "           analysis/adata_focal_annotated.h5ad"
echo "  Phase 3a: analysis/phase3a_timeseries/"
echo "============================================"
