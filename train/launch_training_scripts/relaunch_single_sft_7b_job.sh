#!/bin/bash

# Script to re-launch a single failed job from the sweep
# Run this script from the qwen_training directory: bash scripts/relaunch_single_sft_7b_job.sh
#
# Usage:
#   bash scripts/relaunch_single_sft_7b_job.sh MODEL_PATH DATASET_NAME
#
# Example to re-launch the failed job you mentioned:
#   bash scripts/relaunch_single_sft_7b_job.sh "./custom_token_models/Qwen2.5-VL-7B-Instruct-updated-tokens" "visualprm_data_no_perception_v1%100"

# Check arguments
if [ $# -ne 2 ]; then
    echo "Error: This script requires exactly 2 arguments"
    echo "Usage: bash scripts/relaunch_single_sft_7b_job.sh MODEL_PATH DATASET_NAME"
    echo ""
    echo "Available models from sweep:"
    echo "  - Qwen/Qwen2.5-VL-7B-Instruct"
    echo "  - ./custom_token_models/Qwen2.5-VL-7B-Instruct-updated-tokens"
    echo "  - ./custom_token_models/Qwen2.5-VL-7B-Instruct-updated-tokens-random-init-vals"
    echo ""
    echo "Available datasets from sweep:"
    echo "  - visualprm_data_no_perception_v1%100"
    echo "  - visualprm_data_balanced_v1%100"
    echo ""
    echo "Example:"
    echo "  bash scripts/relaunch_single_sft_7b_job.sh \"./custom_token_models/Qwen2.5-VL-7B-Instruct-updated-tokens\" \"visualprm_data_no_perception_v1%100\""
    exit 1
fi

# Get arguments
model="$1"
dataset="$2"

# Base job name for retries
base_job_name="retry-q7b-swp"

echo "Starting PBS job re-launch at $(date)"
echo "Model: ${model}"
echo "Dataset: ${dataset}"
echo ""

# Create logs directory if it doesn't exist
mkdir -p scripts/pbs_queue_logs

# Create short names for job naming (same logic as sweep script)
model_short=$(basename "${model}" | sed 's/Qwen2.5-VL-7B-Instruct/Q7B/g' | sed 's/updated-tokens/UT/g')

# Build an informative dataset_short (same logic as sweep script)
ds_work="${dataset}"
# Strip known prefix if present
ds_work="${ds_work#visualprm_data_}"
# Separate sampling percent (default 100 if absent)
if [[ "${ds_work}" == *%* ]]; then
    ds_pct="${ds_work##*%}"
    ds_core="${ds_work%%%*}"
else
    ds_pct="100"
    ds_core="${ds_work}"
fi
# Split name and version (version is the suffix after last '_', e.g., v1)
ds_ver="${ds_core##*_}"
ds_name_raw="${ds_core%_*}"
# Normalize descriptor: underscores -> hyphens
ds_name="${ds_name_raw//_/-}"
# Friendly aliases
case "${ds_name}" in
    full-non-balanced)
        ds_name="full-dataset"
        ;;
esac
dataset_short="${ds_name}-${ds_ver}-p${ds_pct}"

# Create unique job name
datetime=$(date +"%Y%m%d-%H%M%S")
job_name="${base_job_name}-${model_short}-${dataset_short}-${datetime}"
output_file="scripts/pbs_queue_logs/${job_name}.out"

echo "Submitting retry job: ${job_name}"
echo "  Model: ${model}"
echo "  Dataset: ${dataset}"
echo "  Output log: qwen_training/${output_file}"
echo ""

# Submit job with environment variables (submitting from parent directory)
cd ..
job_id=$(qsub -v MODEL_PATH="${model}",DATASET_NAME="${dataset}",PBS_OUTPUT_FILE="${output_file}" \
        -N "${job_name}" -o "qwen_training/${output_file}" qwen_training/scripts/sft_7b_parameterized.pbs)
cd qwen_training

echo "Job re-launched successfully"
echo "  Job ID: ${job_id}"
echo ""
echo "Monitor job status with: qstat -u $USER"
echo "View output with: tail -f qwen_training/${output_file}"