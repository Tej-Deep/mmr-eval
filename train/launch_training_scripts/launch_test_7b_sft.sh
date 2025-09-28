#!/bin/bash

# Test launch script for single model/dataset combination
# This tests the PBS parameterized system before running the full sweep
# Run this script from the qwen_training directory: bash scripts/launch_test_7b_sft.sh

# Test configuration
# test_model="Qwen/Qwen2.5-VL-7B-Instruct"
test_model="./custom_token_models/Qwen2.5-VL-7B-Instruct-updated-tokens-random-init-vals"
test_dataset="visualprm_data_balanced%100"

# Base job name
base_job_name="test-qwen7b-sft"

echo "Starting PBS test job submission at $(date)"
echo "Test Model: ${test_model}"
echo "Test Dataset: ${test_dataset}"

# Create logs directory if it doesn't exist
mkdir -p scripts/pbs_queue_logs

# Create unique job name
datetime=$(date +"%Y%m%d-%H%M%S")
job_name="${base_job_name}-${datetime}"
output_file="scripts/pbs_queue_logs/${job_name}.out"

echo "Submitting test job: ${job_name}"
echo "  Model: ${test_model}"
echo "  Dataset: ${test_dataset}"
echo "  Output log: qwen_training/${output_file}"

# Submit job with environment variables (submitting from parent directory)
cd ..
job_id=$(qsub -v MODEL_PATH="${test_model}",DATASET_NAME="${test_dataset}",PBS_OUTPUT_FILE="${output_file}" \
        -N "${job_name}" -o "qwen_training/${output_file}" qwen_training/scripts/sft_7b_parameterized.pbs)
cd qwen_training

echo "Test job submitted successfully"
echo "  Job ID: ${job_id}"
echo ""
echo "Monitor job status with: qstat -u $USER"
echo "View output with: tail -f qwen_training/${output_file}"
echo ""
echo "Once this test succeeds, you can run the full sweep with scripts/launch_sweep_7b_sft.sh"