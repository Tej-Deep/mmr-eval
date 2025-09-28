# Rolling Out

1. Transfer all images and their absolute paths to input_data_files/{dataset_name} (refer to VQAv2 example provided)
2. Specify the Azure endpoints, deployment and config in rollout.py
    - Edit the parse_answer and check_answer functions in rollout.py to match the answer format of the dataset
        - parse_answer is configured by setting scoring_mode, depending on the format of the answer. See parsing functions in accuracy_reward.py
        - check_answer is configured by setting prompt_format_version, depending on the data type of the answer (e.g. integer only, string only exact match, LLM as judge string match). Trace accuracy_reward.py to see how different modes are configured.
    - run ./run_rollout.sh (no parameters) to generate rollouts
3. Transfer completed rollouts to generated_rollouts/soft_estimation/{dataset_name}/final_output/{split_name}
4. run ./run_batch_processor.sh to verify the rollouts, checking for parameters in batch_processor.py
    - use test_batch_processor.py to test the batch processor
    - use test_timeout monitor to test recovery for stale Azure Batch Request
    - Use prepare_and_check_batch.ipynb to check status and errors of batches manually based on deployment

AI2D is content filtered error.

If content filter error thrown, API will return response "Content filter error", which results in a "fail to parse" error, and the prompt will be skipped, with error "fail to parse" in the log.

Example: infovqa_run1_open_ans_9K_v1_subset_raven_rollouts_1289_1610_streaming has 8 "failed to parse" errors, which are all content filter errors, as can see in log, "failed to parse: 4/60 rollouts" twice. 