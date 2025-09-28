# Rolling Out

1. Transfer all images and their absolute paths to input_data_files/{dataset_name} (refer to VQAv2 example provided)
2. Specify the Azure endpoints, deployment and config in rollout.py
    - Edit the parse_answer and check_answer functions in rollout.py to match the answer format of the dataset
        - parse_answer is configured by setting scoring_mode, depending on the format of the answer. See parsing functions in accuracy_reward.py
        - check_answer is configured by setting prompt_format_version, depending on the data type of the answer (e.g. integer only, string only exact match, LLM as judge string match). Trace accuracy_reward.py to see how different modes are configured.
    - run ```./run_rollout.sh (no parameters)``` to generate rollouts
3. Transfer rollouts outputs to rollout_data_generation/llm_judge_verification/rollout_output_data_files/{dataset_name}/{split_name}
    - we recommend to keep to raw rollout outputs untouched as a backup
4. Flatten all rollout JSONL files into a single JSONL file using merge_and_map_batch.py, then run ```./run_batch_processor.sh``` in rollout_data_generation/llm_judge_verification to verify the rollouts, checking for parameters in batch_processor.py
    - Usage: ```python merge_and_map_batch.py --split="vqav2_4k" --model="gpt-4.1-nano-3"```
        - Azure Batch API expects the model to be specified in the JSONL file along with prompt*
    - Usage: ```AZURE_API_KEY='your-key' ./run_batch_processor.sh [screen_session_name] [batch_start_index] [batch_end_index] [split] [azure_endpoint] [check_interval] [model]```

# Known Issues
For datasets like AI2D, InfoVQA, PuzzleTest, content filter errors are common for certain questions, and are avoidable. We drop these rollouts in the training data.

If content filter error thrown, API will return response "Content filter error", which results in a "fail to parse" error, and the prompt will be skipped, with error "fail to parse" in the log.

Example: infovqa_run1_open_ans_9K_v1_subset_raven_rollouts_1289_1610_streaming has 8 "failed to parse" errors, which are all content filter errors, as can see in log, "failed to parse: 4/60 rollouts" twice. 