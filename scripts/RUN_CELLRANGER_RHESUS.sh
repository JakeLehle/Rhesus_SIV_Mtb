#!/bin/bash
#SBATCH -J Rhesus_MtbSIV
#SBATCH -o /master/jlehle/WORKING/LOGS/Rhesus_MtbSIV.o.log
#SBATCH -e /master/jlehle/WORKING/LOGS/Rhesus_MtbSIV.e.log
#SBATCH -t 7-00:00:00
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32                                                    # Reduced from 100 to avoid OOM with 2941-contig reference
#SBATCH --mem 800GB
 
set -euo pipefail
 
# Activate environment
source $HOME/anaconda3/bin/activate
conda activate sc_pre
export PATH=$HOME/cellranger-10.0.0/bin:$PATH
 
WORKDIR="/master/jlehle/WORKING/SC/fastq/Rhesus"
mkdir -p "${WORKDIR}/metadata"
mkdir -p "${WORKDIR}/logs"
mkdir -p "${WORKDIR}/cellranger_counts"
 
cd "${WORKDIR}"
 
# Step 1: Build unified metadata from both projects
echo "============================================"
echo "STEP 1: Building unified metadata"
echo "============================================"
#python "${WORKDIR}/build_unified_metadata.py"
 
# Step 2: Run cellranger count on all samples
echo ""
echo "============================================"
echo "STEP 2: Running cellranger count"
echo "============================================"
#python "${WORKDIR}/cellranger_count_all.py"

python rhesus_bal_scrna_pipeline.py \
    --all_samples /master/jlehle/WORKING/SC/fastq/Rhesus/cellranger_counts/ \
    --metadata /master/jlehle/WORKING/SC/fastq/Rhesus/metadata/unified_metadata.pkl \
    --working_dir /master/jlehle/WORKING/SC/fastq/Rhesus/analysis

echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "============================================"
exit
