# TODO: Set manually corresponding to data-dir set for extract_ans.py (TODO: Also set in extract_ans.py)
policy_model="g27b"

# Extract Answer
python extract_ans.py \
    --data-dir ./outputs/g27b_policy/q3b_prm/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_MMMU_DEV_VAL_result-merged-0-900-20250901_115456/step_agg/prm_V8B
# Compute Score
python compute_score.py \
    --data-dir ./outputs/extracted_ans_outputs/PRM_V8B/${policy_model}_policy/