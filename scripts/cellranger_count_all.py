#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell Ranger Count - Unified Processing Script
Rhesus Mtb/SIV Coinfection Project

Reads unified_metadata.pkl and runs cellranger count for all samples
using the custom Mmul10 + MtbCDC1551 + SIVmac239 reference.

Handles:
- Greehey samples (some split across two sequencing runs -> merged via --fastqs)
- GSE293173 SRA samples
- Resume capability: skips samples with completed outputs
- Chemistry auto-detection with fallback cascade
"""

import os
import sys
import yaml
import shutil
import subprocess
import pickle
import json
import traceback
import glob
import re
from os import cpu_count
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = "/master/jlehle/WORKING/SC/fastq/Rhesus"
TRANSCRIPTOME = "/master/jlehle/WORKING/SC/REF/Mmul10_MtbCDC1551_SIVmac239_v10"
METADATA_PKL = os.path.join(BASE_DIR, "metadata", "unified_metadata.pkl")
OUTPUT_BASE = os.path.join(BASE_DIR, "cellranger_counts")
LOG_DIR = os.path.join(BASE_DIR, "logs")
NCPUS = 32  # Reduced from cpu_count() to avoid OOM with large reference

os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# LOAD UNIFIED METADATA
# =============================================================================
print("=" * 80)
print("LOADING UNIFIED METADATA")
print("=" * 80)

with open(METADATA_PKL, 'rb') as f:
    unified_metadata = pickle.load(f)
    print(f"Loaded {len(unified_metadata)} samples")

greehey_count = sum(1 for m in unified_metadata.values() if m['project'] == 'Greehey')
gse_count = sum(1 for m in unified_metadata.values() if m['project'] == 'GSE293173')
print(f"  Greehey:   {greehey_count}")
print(f"  GSE293173: {gse_count}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_file_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def cleanup_directory(directory_path):
    """Safely remove a directory if it exists"""
    if os.path.exists(directory_path):
        try:
            print(f"  Cleaning up: {directory_path}")
            shutil.rmtree(directory_path)
            return True
        except Exception as e:
            print(f"  WARNING: Could not remove {directory_path}: {e}")
            return False
    return True


def check_cellranger_complete(output_dir):
    """
    Check if cellranger count completed successfully.
    Validates filtered_feature_bc_matrix has all 3 required files.
    
    IMPORTANT: barcodes.tsv.gz must be >100 bytes to be considered valid.
    A 33-byte file is just an empty gzip header (zero cells called).
    
    Returns dict with status info.
    """
    result = {
        'completed': False,
        'has_barcodes': False,
        'has_features': False,
        'has_matrix': False,
        'sizes': {},
        'error': None
    }

    filtered_dir = os.path.join(output_dir, 'outs', 'filtered_feature_bc_matrix')
    if not os.path.exists(filtered_dir):
        result['error'] = "filtered_feature_bc_matrix not found"
        return result

    required_files = {
        'barcodes': 'barcodes.tsv.gz',
        'features': 'features.tsv.gz',
        'matrix': 'matrix.mtx.gz'
    }

    # Minimum size thresholds - empty gzip is ~33 bytes
    min_sizes = {
        'barcodes': 100,  # Zero-cell results produce 33-byte empty gzip
        'features': 100,
        'matrix': 100
    }

    for key, filename in required_files.items():
        filepath = os.path.join(filtered_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > min_sizes[key]:
                result[f'has_{key}'] = True
                result['sizes'][key] = size
            else:
                result['error'] = (f"{filename} too small ({size} bytes) - "
                                   f"likely zero cells called")
        else:
            result['error'] = f"{filename} not found"

    result['completed'] = (result['has_barcodes'] and
                           result['has_features'] and
                           result['has_matrix'])
    if result['completed']:
        result['error'] = None

    return result


def validate_fastqs(fastq_dirs, sample_name):
    """
    Validate FASTQ files exist and have R1+R2 across all fastq directories.
    """
    validation = {
        'valid': False,
        'has_r1': False,
        'has_r2': False,
        'has_i1': False,
        'has_i2': False,
        'r1_total_size': 0,
        'r2_total_size': 0,
        'error': None
    }

    for fq_dir in fastq_dirs:
        if not os.path.exists(fq_dir):
            validation['error'] = f"Directory not found: {fq_dir}"
            return validation

        # Look for R1/R2 matching this sample name
        r1_files = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_R1_*.fastq.gz"))
        r2_files = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_R2_*.fastq.gz"))
        i1_files = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_I1_*.fastq.gz"))
        i2_files = glob.glob(os.path.join(fq_dir, f"{sample_name}_S*_L*_I2_*.fastq.gz"))

        if r1_files:
            validation['has_r1'] = True
            validation['r1_total_size'] += sum(os.path.getsize(f) for f in r1_files)
        if r2_files:
            validation['has_r2'] = True
            validation['r2_total_size'] += sum(os.path.getsize(f) for f in r2_files)
        if i1_files:
            validation['has_i1'] = True
        if i2_files:
            validation['has_i2'] = True

    if not validation['has_r1']:
        validation['error'] = "No R1 files found"
    elif not validation['has_r2']:
        validation['error'] = "No R2 files found (single-end not supported by cellranger)"

    validation['valid'] = validation['has_r1'] and validation['has_r2']
    return validation


def format_error_output(result):
    """Extract last 20 lines of stdout/stderr for error reporting"""
    lines = []
    if result.stdout:
        lines.append("=== STDOUT (last 20 lines) ===")
        lines.extend(result.stdout.split('\n')[-20:])
    if result.stderr:
        lines.append("=== STDERR (last 20 lines) ===")
        lines.extend(result.stderr.split('\n')[-20:])
    return "\n".join(lines)


def check_if_multiplexed(fastq_dirs):
    """Check if sample has index files (I1/I2) - used for logging only"""
    for fq_dir in fastq_dirs:
        try:
            files = os.listdir(fq_dir)
            if any('_I1_' in f for f in files) and any('_I2_' in f for f in files):
                return True
        except Exception:
            pass
    return False


# =============================================================================
# PRE-FLIGHT VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("PRE-FLIGHT VALIDATION")
print("=" * 80)

validation_results = {}
valid_samples = []
invalid_samples = []

for sample_id, meta in sorted(unified_metadata.items()):
    sample_name = meta['sample_name']
    fastq_dirs = meta['fastq_dirs']

    print(f"\n  Validating {sample_id} (project={meta['project']})...")
    print(f"    Sample prefix: {sample_name}")
    print(f"    FASTQ dirs: {len(fastq_dirs)}")
    for d in fastq_dirs:
        print(f"      - {d}")

    val = validate_fastqs(fastq_dirs, sample_name)
    validation_results[sample_id] = val

    if val['valid']:
        valid_samples.append(sample_id)
        print(f"    VALID - R1: {format_file_size(val['r1_total_size'])}, "
              f"R2: {format_file_size(val['r2_total_size'])}")
        if val['has_i1'] or val['has_i2']:
            print(f"    Index files: I1={'yes' if val['has_i1'] else 'no'}, "
                  f"I2={'yes' if val['has_i2'] else 'no'}")
    else:
        invalid_samples.append(sample_id)
        print(f"    INVALID - {val['error']}")

# Save validation
val_file = os.path.join(BASE_DIR, "metadata", "cellranger_validation.json")
with open(val_file, 'w') as f:
    json.dump(validation_results, f, indent=2, default=str)

print(f"\n{'=' * 80}")
print(f"VALIDATION SUMMARY")
print(f"{'=' * 80}")
print(f"Total samples: {len(unified_metadata)}")
print(f"Valid (paired-end): {len(valid_samples)}")
print(f"Invalid: {len(invalid_samples)}")

if len(valid_samples) == 0:
    print("\nERROR: No valid samples found!")
    sys.exit(1)

print(f"\nProceeding with {len(valid_samples)} valid samples...")

# =============================================================================
# CELL RANGER COUNT PROCESSING
# =============================================================================
print("\n" + "=" * 80)
print("CELL RANGER COUNT PROCESSING")
print(f"Reference: {TRANSCRIPTOME}")
print(f"Output base: {OUTPUT_BASE}")
print("=" * 80)

# Chemistry options list - auto first, then specific fallbacks
chemistry_options = [
    "auto",
    "threeprime",
    "fiveprime",
    "SC3Pv4-polyA",
    "SC3Pv4",
    "SC3Pv3-polyA",
    "SC3Pv3",
    "SC3Pv2",
    "SC3Pv3HT",
    "SC3Pv3HT-polyA",
    "SC5P-PE-v3",
    "SC5P-PE",
    "SC5P-R2-v3",
    "SC5P-R2",
    "SC3Pv1",
    "ARC-v1"
]

successful_samples = []
skipped_samples = []
failed_samples = []
already_completed = []

for sample_id in sorted(valid_samples):
    meta = unified_metadata[sample_id]
    sample_name = meta['sample_name']
    fastq_dirs = meta['fastq_dirs']
    project = meta['project']

    # cellranger output ID - use sample_id as the run ID
    cr_id = sample_id
    cr_output_dir = os.path.join(OUTPUT_BASE, cr_id)

    # Detect multiplexing for chemistry suggestion
    is_multiplexed = check_if_multiplexed(fastq_dirs)

    print(f"\n{'=' * 60}")
    print(f"Sample: {sample_id} ({project})")
    if meta.get('treatment_group') and meta['treatment_group'] != 'TBD':
        print(f"  Animal: {meta['animal_id']} | {meta['timepoint']} | "
              f"Group {meta['group_number']}: {meta['treatment_group']}")
    print(f"  --sample={sample_name}")
    print(f"  --fastqs={','.join(fastq_dirs)}")
    print(f"  Multiplexed: {'Yes' if is_multiplexed else 'No'}")
    print(f"  Output: {cr_output_dir}")
    print(f"{'=' * 60}")

    # --- CHECK IF ALREADY COMPLETED ---
    check = check_cellranger_complete(cr_output_dir)
    if check['completed']:
        print(f"  ALREADY COMPLETED - skipping")
        for key, size in check['sizes'].items():
            print(f"    {key}: {format_file_size(size)}")
        already_completed.append(sample_id)
        successful_samples.append(sample_id)
        continue

    # --- CLEAN UP INCOMPLETE PREVIOUS RUN ---
    if os.path.exists(cr_output_dir):
        print(f"  Incomplete previous run detected: {check['error']}")
        cleanup_directory(cr_output_dir)

    # --- RUN CELLRANGER COUNT ---
    # Build the --fastqs argument (comma-separated for multi-run samples)
    fastqs_arg = ",".join(fastq_dirs)

    # Change to output base so cellranger creates its directory there
    os.chdir(OUTPUT_BASE)

    success = False
    last_errors = {}

    # Always try auto first - let cellranger detect the chemistry.
    # Only fall back to specific chemistries if auto fails.
    trial_chemistries = chemistry_options  # auto is first in the list

    for chemistry in trial_chemistries:
        # Clean up any partial attempt
        cleanup_directory(cr_output_dir)

        try:
            print(f"\n  Trying chemistry: {chemistry}")
            cmd = [
                "cellranger", "count",
                f"--id={cr_id}",
                f"--fastqs={fastqs_arg}",
                f"--sample={sample_name}",
                f"--transcriptome={TRANSCRIPTOME}",
                f"--localcores={NCPUS}",
                "--create-bam=true",
                "--expect-cells=10000",
                f"--chemistry={chemistry}"
            ]

            print(f"  Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0:
                # Verify outputs
                final_check = check_cellranger_complete(cr_output_dir)
                if final_check['completed']:
                    print(f"  SUCCESS with chemistry={chemistry}")
                    for key, size in final_check['sizes'].items():
                        print(f"    {key}: {format_file_size(size)}")
                    success = True
                    successful_samples.append(sample_id)
                    break
                else:
                    print(f"  Returned 0 but outputs incomplete: {final_check['error']}")
                    last_errors[chemistry] = f"Incomplete: {final_check['error']}"
                    cleanup_directory(cr_output_dir)
            else:
                error_out = format_error_output(result)
                last_errors[chemistry] = error_out
                first_line = error_out.split('\n')[0] if error_out else 'Unknown'
                print(f"  Failed: {first_line}")
                cleanup_directory(cr_output_dir)

        except Exception as e:
            last_errors[chemistry] = traceback.format_exc()
            print(f"  Exception: {e}")
            cleanup_directory(cr_output_dir)

    if not success:
        print(f"\n  ALL CHEMISTRIES FAILED for {sample_id}")
        for chem, err in list(last_errors.items())[-3:]:
            preview = err.splitlines()[0] if err else 'Unknown'
            print(f"    {chem}: {preview}")
        failed_samples.append({
            'sample_id': sample_id,
            'project': project,
            'last_errors': {k: v[:500] for k, v in last_errors.items()}
        })

# =============================================================================
# SAVE PROCESSING SUMMARY
# =============================================================================
summary = {
    'reference': TRANSCRIPTOME,
    'output_base': OUTPUT_BASE,
    'validation': {
        'total': len(unified_metadata),
        'valid': len(valid_samples),
        'invalid': len(invalid_samples),
        'invalid_samples': invalid_samples
    },
    'processing': {
        'successful': len(successful_samples),
        'already_completed': len(already_completed),
        'newly_processed': len(successful_samples) - len(already_completed),
        'failed': len(failed_samples),
    },
    'successful_samples': successful_samples,
    'already_completed_samples': already_completed,
    'failed_details': [
        {'sample_id': f['sample_id'], 'project': f['project'],
         'chemistries_tried': list(f['last_errors'].keys())}
        for f in failed_samples
    ]
}

summary_file = os.path.join(BASE_DIR, "metadata", "cellranger_processing_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

# Also save a simple mapping of sample_id -> output path for downstream
output_map = {}
for sid in successful_samples:
    cr_dir = os.path.join(OUTPUT_BASE, sid)
    mtx_dir = os.path.join(cr_dir, 'outs', 'filtered_feature_bc_matrix')
    meta = unified_metadata[sid]
    output_map[sid] = {
        'cellranger_dir': cr_dir,
        'matrix_dir': mtx_dir,
        'project': meta['project'],
        'animal_id': meta.get('animal_id'),
        'timepoint': meta.get('timepoint'),
        'treatment_group': meta.get('treatment_group'),
        'group_number': meta.get('group_number'),
        'd1mt': meta.get('d1mt'),
        'cart_timing': meta.get('cart_timing'),
    }

map_file = os.path.join(BASE_DIR, "metadata", "sample_output_map.json")
with open(map_file, 'w') as f:
    json.dump(output_map, f, indent=2)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PROCESSING COMPLETE")
print("=" * 80)

print(f"\nValidation:")
print(f"  Total samples:  {len(unified_metadata)}")
print(f"  Valid:           {len(valid_samples)}")
print(f"  Invalid:         {len(invalid_samples)}")

print(f"\nCell Ranger Count:")
print(f"  Already done:    {len(already_completed)}")
print(f"  Newly processed: {len(successful_samples) - len(already_completed)}")
print(f"  Total successful:{len(successful_samples)}")
print(f"  Failed:          {len(failed_samples)}")

if already_completed:
    print(f"\nAlready completed:")
    for sid in already_completed[:10]:
        print(f"  - {sid}")
    if len(already_completed) > 10:
        print(f"  ... and {len(already_completed) - 10} more")

if failed_samples:
    print(f"\nFailed samples:")
    for f in failed_samples:
        print(f"  - {f['sample_id']} ({f['project']})")

print(f"\nOutputs saved to:")
print(f"  Count matrices: {OUTPUT_BASE}/")
print(f"  Summary:        {summary_file}")
print(f"  Output map:     {map_file}")
print(f"  Validation:     {val_file}")

print("\n" + "=" * 80)
if len(failed_samples) == 0 and len(valid_samples) > 0:
    print("All valid samples processed successfully!")
elif len(failed_samples) > 0:
    print(f"{len(failed_samples)} sample(s) failed - check summary for details")
print("=" * 80)

sys.exit(0)
