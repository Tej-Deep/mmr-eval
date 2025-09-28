#!/bin/bash

# Full sweep launch script for multiple model/dataset combinations
# Run the test script first (launch_test_32b_sft.sh) before using this
# Run this script from the qwen_training directory: bash scripts/launch_sweep_32b_sft.sh

# Arrays of models to sweep
models=(
    "Qwen/Qwen2.5-VL-32B-Instruct"
    # "./custom_token_models/Qwen2.5-VL-32B-Instruct-updated-tokens"
    # "./custom_token_models/Qwen2.5-VL-32B-Instruct-updated-tokens-random-init-vals"
)

# Arrays of datasets to sweep (from qwenvl/data/__init__.py)
datasets=(
    # "mc0_visualprm_data_full_non_balanced_v0%100"
    # "mc0_visualprm_data_errors_balanced_v2%100"
    # "mc0_visualprm_data_errors_no_perception_v2%100"
    # "mc0_visualprm_data_custom_tok_full_non_balanced_v1%100"
    # "mc0_visualprm_data_custom_tok_errors_balanced_v3%100"
    # "mc0_visualprm_data_normal_tok_errors_balanced_v1%100"
    # "mc0_visualprm_data_normal_tok_full_non_balanced_v1%100"
    "mc0_visualprm_data_normal_tok_ablation_only_reasoning_errors_v1%100"
)

# TODO: Check if using image or image_qwen_smart_resize in data_qwen.py

# Vision tuning options to sweep
tune_mm_vision_options=(
    # "False"  # Default: Vision tower frozen
    "True"   # Vision tower trainable
)

# Base job name
base_job_name="q32b-qsize"

# Counter for job submissions
job_count=0

echo "Starting PBS job sweep at $(date)"
echo "Models: ${models[@]}"
echo "Datasets: ${datasets[@]}"
echo "Vision tuning options: ${tune_mm_vision_options[@]}"
echo "Total combinations: $((${#models[@]} * ${#datasets[@]} * ${#tune_mm_vision_options[@]}))"
echo ""

# Create logs directory if it doesn't exist
mkdir -p scripts/pbs_queue_logs

# Loop through all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for tune_mm_vision in "${tune_mm_vision_options[@]}"; do
        # Create short names for job naming
        model_short=$(basename "${model}" | sed 's/Qwen2.5-VL-32B-Instruct/Q32B/g' | sed 's/updated-tokens/UT/g')

        # Build an informative dataset_short, e.g., "no-perception-v1-p100", "full-dataset-v1-p100"
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
        
        # Add vision tuning suffix
        if [ "${tune_mm_vision}" = "True" ]; then
            vision_suffix="vit_trained"
        else
            vision_suffix="vit_frozen"
        fi
        
        # Create unique job name
        datetime=$(date +"%Y%m%d-%H%M%S")
        job_name="${base_job_name}-${model_short}-${dataset_short}-${vision_suffix}-${datetime}"
        output_file="scripts/pbs_queue_logs/${job_name}.out"
        
        echo "Submitting job $((job_count+1)): ${job_name}"
        echo "  Model: ${model}"
        echo "  Dataset: ${dataset}"
        echo "  Vision tuning: ${tune_mm_vision}"
        
        # Submit job with environment variables (submitting from parent directory)
        # Add -r y to enable automatic requeue on exit code 99 (CUDA failure)
        cd ..
        job_id=$(qsub -r y -v MODEL_PATH="${model}",DATASET_NAME="${dataset}",TUNE_MM_VISION="${tune_mm_vision}",PBS_OUTPUT_FILE="${output_file}" \
                -N "${job_name}" -o "qwen_training/${output_file}" qwen_training/scripts/sft_32b_parameterized.pbs)
        cd qwen_training
        
        echo "  Job ID: ${job_id}"
        echo "  Note: Job will automatically requeue if CUDA is unavailable (exit code 99)"
        echo ""
        
        ((job_count++))
        
        # Add delay between submissions to avoid overwhelming the scheduler
        sleep 2
        done
    done
done

echo "Submitted ${job_count} jobs total"
echo "Use 'qstat -u $USER' to check job status"
echo ""
echo "Log files are in: qwen_training/scripts/pbs_queue_logs/"
echo "Output models will be in: ./outputs/"