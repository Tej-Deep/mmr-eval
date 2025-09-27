# Rolling Out

1. Transfer all images and their absolute paths to preprocessing_scripts/{dataset_name}
2. Specify the endpoints, deployment and config (for input file path, 1-indexed sample_start_idx, sample_end_idx and output directory) in rollout.py THEN ./run_rollout.sh (no parameters) to generate rollouts
    - edit check_answer function in rollout.py to match the answer format of the dataset (for RAVEN, option 1-8 ONLY, integer only matching for MMPR correctness prompts, GPT answer checking for open text)
    - check_answer (set prompt_format_version), parse_answer (set scoring_mode)
3. Transfer completed rollouts to generated_rollouts/soft_estimation/{dataset_name}/final_output/{split_name}
4. run ./run_batch_processor.sh to verify the rollouts, checking for parameters in batch_processor.py
    - use test_batch_processor.py to test the batch processor
    - use test_timeout monitor to test recovery for stale Azure Batch Request
    - Use prepare_and_check_batch.ipynb to check status and errors of batches manually based on deployment

AI2D is content filtered error.

If content filter error thrown, API will return response "Content filter error", which results in a "fail to parse" error, and the prompt will be skipped, with error "fail to parse" in the log.

Example: infovqa_run1_open_ans_9K_v1_subset_raven_rollouts_1289_1610_streaming has 8 "failed to parse" errors, which are all content filter errors, as can see in log, "failed to parse: 4/60 rollouts" twice. 