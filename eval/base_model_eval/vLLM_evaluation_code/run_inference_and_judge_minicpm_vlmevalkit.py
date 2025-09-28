import os
import argparse
import json
import datetime
import random
import math
import torch
import glob
import pandas as pd
import numpy as np
import string
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    GenerationConfig,
)
from evaluation.reward_guided_search.utils import (
    sample_to_images_list,
    convert_images_to_base64,
    prepare_question_array_with_base64_image_strings,
    prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm,
)
from evaluation.reward_guided_search.mathvista_helper_functions import (
    calculate_mathvista_judge_evaluation_score,
)

from PIL import Image
import base64
from io import BytesIO
from vllm import LLM, SamplingParams

from evaluation.common.logger import log_info

import json
import ast
import traceback

from evaluation.common.send_telegram_notifications_helper import (
    send_telegram_error_notification,
    send_telegram_job_summary,
)
from evaluation.reward_guided_search.collate_final_eval_results import (
    calculate_evaluation_score_direct,
)
from evaluation.base_model_eval_code.judge_mathvista import (
    calculate_mathvista_judge_evaluation_score,
)
# from evaluation.reward_guided_search.mathvista_helper_functions import (
    # load_mathvista_dataset,
    # build_mathvista_prompt,
# )
import re
NEWLINE_STEP_SEPERATOR = "\n\n"
BOXED_ANSWER_STR = r"\boxed{"

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def extract_boxed(s):
    # Match both \boxed{...} and \\boxed{...} patterns to handle escape sequence issues
    # Use raw string in pattern but flexible matching for corrupted backslashes
    patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Standard \boxed{...}
        r"\x08oxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Handle corrupted \b -> \x08 from LLM string output, as LLM may not know to escape the backslash all the time
    ]

    all_matches = []
    for pattern in patterns:
        matches = list(re.finditer(pattern, s))
        all_matches.extend(matches)

    if not all_matches:
        return None

    # Sort by position and return the LATEST match
    all_matches.sort(key=lambda m: m.start())
    return all_matches[-1].group(1).strip()

def build_mathvista_prompt(line_dict): # no CoT for MathVista for MiniCPM
    # short_ans_cot_prompt = ('''Read the following question carefully, solve it step by step, and '''
    #                                  '''then output the final answer in the format of "Answer: single number '''
    #                                  '''or single word or phrase".\n\n''')
    # prompt = short_ans_cot_prompt + line_dict["question"] 
    return line_dict["question"] 

def toliststr(s):
    """Convert string representation of list to actual list"""
    if isinstance(s, str):
        try:
            import ast
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
    return s if isinstance(s, list) else [s]

def load_mathvista_dataset(data_dir, dataset_name='MathVista_MINI'):
    """Load the MathVista dataset from TSV file."""
    data_path = os.path.join(data_dir, f"{dataset_name}.tsv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    data = pd.read_csv(data_path, sep="\t")
    
    data['index'] = [str(x) for x in data['index']]
    
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]
    
    # Handle image paths
    if 'image_path' in data:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to int if possible
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

def inference_single_step(
    policy_model, image_data_base64_list, policy_prompt_array, eos_token, tokenizer=None
):
    if tokenizer.name_or_path == "openbmb/MiniCPM-V-2_6":
        policy_model_name = "MiniCPM"
    else: 
        model_config = policy_model.llm_engine.get_model_config()
        policy_model_name = model_config.model
    if "minicpm" in policy_model_name.lower(): 
        # max_new_tokens = 2048
        # sampling_params = dict(
        #     max_new_tokens=max_new_tokens,
        #     sampling=False,
        #     num_beams=3,
        # )
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            # top_p=0.95,
            # top_k=64,
            # stop=['<|im_end|>', '<|endoftext|>'],
            # include_stop_str_in_output=True,
            max_tokens=16384,
        )

    else:
        raise ValueError(f"Unsupported policy model to set SamplingParams in get_next_step: {policy_model_name}")
    log_info(f"Policy Model Sampling parameters: {sampling_params}")
    log_info(f"Sending prompt to policy model: {policy_prompt_array}")
    log_info(f"Length of image data base64 list: {len(image_data_base64_list)}")
    # Convert base64 images to PIL Images for VLLM
    images = []
    for base64_img in image_data_base64_list:
        img_data = base64.b64decode(base64_img)
        if "minicpm" in policy_model_name.lower():
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_width, img_height = img.width, img.height # resizing https://github.com/OpenBMB/MiniCPM-V/blob/main/eval_mm/vlmevalkit/vlmeval/vlm/minicpm_v.py#L260
            if (img_width * img_height) >= (1344 * 1344):
                log_info(f"Image is too large, skipping resize: {img_width}x{img_height}")
                img = img
            else:
                ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                max_img_width = int(img_width * ratio)
                new_img_width = random.randint(img_width, max_img_width)
                new_img_height = int(new_img_width / img_width * img_height)
                log_info(f"Resizing image from {img_width}x{img_height} to {new_img_width}x{new_img_height}")
                img = img.resize((new_img_width, new_img_height))
        elif "internvl" in policy_model_name.lower():
            img = base64_img
        else:
            img = Image.open(BytesIO(img_data))
            img.load()
        images.append(img)
                
    if "minicpm" in policy_model_name.lower():
        sys_prompt = policy_prompt_array[0]['content']
        user_prompt = policy_prompt_array[1]['content']
        combined_user_prompt = sys_prompt + "\n" + user_prompt
        policy_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": combined_user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        log_info(f"Policy prompt: {policy_prompt}")
        # user_obj['content'] = user_obj['content'].replace("(<image>./</image>)\n\n", "")
        
        # user_obj['content'] = [sys_prompt, images[0], user_obj['content']]
        # msgs = [user_obj]
        # log_info(f"MiniCPM Messages: {msgs}")

        # max_slice_nums = None
        # use_image_id = True
        # max_inp_length = 8192
        # llm_output = policy_model.chat(
        #     image=None,
        #     msgs=msgs,
        #     context=None,
        #     tokenizer=tokenizer,
        #     max_inp_length=max_inp_length,
        #     use_image_id=use_image_id,
        #     max_slice_nums=max_slice_nums,
        #     **sampling_params
        # )
        # log_info(f"Policy model returned: {llm_output}")
        # return llm_output
        llm_output = policy_model.generate(
            {
                "prompt": policy_prompt,
                "multi_modal_data": {"image": images} if images else None,
            },
            sampling_params,
        )
        log_info(f"Policy model returned: {llm_output}")

        return llm_output[0].outputs[0].text.strip()
    else:
        raise ValueError(f"Unsupported policy model to set SamplingParams in get_next_step: {policy_model_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Base Model Evaluation and Judgement pipeline"
    )
    parser.add_argument(
        "--policy_model_path", type=str, required=True, help="Path to the policy model."
    )
    dataset_choices = ["MMMU_DEV_VAL", "puzzleVQA_1K_subset", "AlgoPuzzleVQA_900_subset", "mathvista_testmini"]
    parser.add_argument("--data", type=str, required=True, help="Dataset to Evaluate on", 
                        choices=dataset_choices)
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=0.7, help="the temperature of the policy model.")
    parser.add_argument(
        "--data_begin",
        type=int,
        default=0,
        help="Starting index of the dataset to process.",
    )
    parser.add_argument(
        "--data_end",
        type=int,
        default=None,
        help="Ending index of the dataset to process. Defaults: mmmu_dev=150, mmmu_validation=900, mmmu_test_full=10500, others=10.",
    )
    parser.add_argument(
        "--development_mode",
        action="store_true",
        default=False,
        help="If True, filter dataset to target_ids after loading.",
    )
    parser.add_argument(
        "--policy_gpu",
        type=int,
        default=0,
        help="GPU device ID for policy model (default: 0)",
    )
    parser.add_argument(
        "--partition_id",
        type=int,
        default=None,
        help="Partition ID for parallel runs (adds partition suffix to output file)",
    )
    parser.add_argument(
        "--run_datetime",
        type=str,
        default=None,
        help="Run datetime string for consistent naming across parallel partitions",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/mmmu/data",
        help="The absolute path of MMMU dataset directory",
    )
    args = parser.parse_args()
    
    if args.data_end is None:
        if args.data == "MMMU_DEV_VAL":
            args.data_end = 900  # Default for MMMU_DEV_VAL
        elif args.data == "puzzleVQA_1K_subset":
            args.data_end = 1000  # PuzzleVQA 1K subset has 1000 samples (50 × 20 puzzle types)
        elif args.data == "AlgoPuzzleVQA_900_subset":
            args.data_end = 900  # PuzzleVQA 1K subset has 1000 samples (50 × 20 puzzle types)
        # elif args.data.startswith("puzzle_"):
        #     args.data_end = 100  # Individual PuzzleVQA datasets have 100 samples
        elif args.data == "mathvista_testmini":
            args.data_end = 1000
        else:
            raise ValueError(f"Unsupported dataset: {args.data}")
    
    log_info(f"Using data_end={args.data_end} for dataset {args.data}")

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.data == "mathvista_testmini":
            log_info("Loading MathVista dataset...")
            dataset_df = load_mathvista_dataset('data', 'MathVista_MINI')
            dataset_info = {
                "name": args.data,
                "answer_key": "answer",
                "interleave_image_tokens": False,
                "is_puzzle": False
            }
        else:
            raise ValueError(f"Unsupported dataset: {args.data}")

        ans_key = dataset_info["answer_key"]
        interleave_image_tokens = dataset_info["interleave_image_tokens"]

        log_info(f"Loaded Policy Model: {args.policy_model_path}")

        dataset = dataset_df

        if args.run_datetime:
            run_datetime = args.run_datetime
        else:
            run_datetime = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )  # for filename below
        log_info(f"Result file suffix: {run_datetime}")
        log_info(f"Evaluating dataset: {args.data}")
        
        # Dataset is a pandas DataFrame
        log_info(f"Dataset columns: {list(dataset.columns)}")

        # Dev mode not working for MathVista from vlmevalkit load TSV file
        # if args.development_mode:
        #     log_info("Development mode is True, filtering dataset for testing")
        #     if dataset_info["name"] == "mathvista_testmini":
        #         # Use first 8 samples for MathVista development testing
        #         target_ids = ["1", "2", "3", "4", "5", "6", "7", "8"]
        #         dataset = dataset[dataset["id"].isin(target_ids)]
        #         log_info(f"MathVista development mode: Using target_ids: {target_ids}")
        # else:
        #     log_info("Development mode is False, evaluating on full dataset")

        if not isinstance(args.data_begin, int) or not isinstance(
            args.data_end, int
        ):
            raise ValueError("data_begin and data_end must be integers")
        if args.data_end <= args.data_begin:
            raise ValueError(
                f"data_end ({args.data_end}) must be greater than data_begin ({args.data_begin})"
            )
        if args.data_begin < 0:
            raise ValueError(f"data_begin ({args.data_begin}) must be non-negative")

        start_idx = args.data_begin
        end_idx = min(args.data_end, len(dataset))
        
        data = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
            
        log_info(
            f"Using range [{args.data_begin}, {args.data_end}): Data size after selection: {len(data)}"
        )

        # Ensure CUDA device is available before initializing, sometimes fails because of failed CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        available_gpus = torch.cuda.device_count()
        log_info(f"Available GPUs: {available_gpus}")
        log_info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        if args.policy_gpu >= available_gpus:
            raise ValueError(f"Policy GPU {args.policy_gpu} not available. Only {available_gpus} GPUs found.")
        
        log_info(f"Initializing policy model on GPU index {args.policy_gpu} (cuda:{args.policy_gpu})")
        
        if args.policy_model_path == "openbmb/MiniCPM-V-2_6":
            # policy_model = AutoModel.from_pretrained(args.policy_model_path, trust_remote_code=True)
            # policy_model = policy_model.to(dtype=torch.bfloat16)
            # policy_model.eval().cuda()
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=128,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.80,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported policy model: {args.policy_model_path}")

        if "minicpm" in args.policy_model_path.lower():
            policy_model_processor = AutoTokenizer.from_pretrained(args.policy_model_path, trust_remote_code=True)
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unsupported policy model: {args.policy_model_path}")

        # Override for Gemma models due to incorrect tokenizer tokens
        if "gemma" in args.policy_model_path.lower():
            policy_model_processor.tokenizer.eos_token = "<end_of_turn>"
            policy_model_processor.tokenizer.bos_token = "[BOS]"
            log_info(
                f"Overriding EOS token for Gemma model {args.policy_model_path}: {policy_model_processor.tokenizer.eos_token}"
            )
            log_info(
                f"Overriding BOS token for Gemma model {args.policy_model_path}: {policy_model_processor.tokenizer.bos_token}"
            )
        elif "cpm" in args.policy_model_path.lower() or "internvl" in args.policy_model_path.lower(): 
            log_info(
                f"Using EOS token for policy model {args.policy_model_path}: {policy_model_processor.eos_token}"
            )
            log_info(
                f"Using BOS token for policy model {args.policy_model_path}: {policy_model_processor.bos_token}"
            )
        else:
            log_info(
                f"Using EOS token for policy model {args.policy_model_path}: {policy_model_processor.tokenizer.eos_token}"
            )
            log_info(
                f"Using BOS token for policy model {args.policy_model_path}: {policy_model_processor.tokenizer.bos_token}"
            )

        if "cpm" in args.policy_model_path.lower() or "internvl" in args.policy_model_path.lower():
            EOS_TOKEN = policy_model_processor.eos_token
        else:
            EOS_TOKEN = policy_model_processor.tokenizer.eos_token

        new_dataset = []
        
        for i in tqdm(range(len(data)), desc="Running inference"):
            line = data.iloc[i]
            line_dict = line.to_dict()
            for k, v in line_dict.items():
                if isinstance(v, np.integer):
                    line_dict[k] = int(v)
                elif isinstance(v, np.floating):
                    line_dict[k] = float(v)

            image_data_base64_list = []
            
            if dataset_info["is_puzzle"]:
                # PuzzleVQA format: single base64 image in "image" field, because we already wrote the processing in load_puzzlevqa_1k_subset function (original PuzzleVQA JSONL file is a path to local image)
                if "image" in line_dict and line_dict["image"]:
                    image_data_base64_list.append(line_dict["image"])
            else:
                # MMMU format: potentially multiple images in "image" field
                if "image" in line_dict and line_dict["image"]:
                    # Handle single image case
                    if isinstance(line_dict["image"], str):
                        image_data_base64_list.append(line_dict["image"])
                    # Handle multiple images case
                    elif isinstance(line_dict["image"], list):
                        image_data_base64_list.extend(
                            [img for img in line_dict["image"] if img]
                        )
            
            log_info(f"Found {len(image_data_base64_list)} base64 images in data")
            if len(image_data_base64_list) == 0:
                raise ValueError(f"No image data found for data sample: {line_dict}")

            question = line["question"] # For AlgoPuzzle which has Yes/No questions but we want it to be MCQ consistent format.
            if "Answer Yes or No." in question:
                question = question.replace("Answer Yes or No.", "")
            
            if dataset_info["name"] == "mathvista_testmini":
                question = line_dict["question"]
                user_prompt = build_mathvista_prompt(line_dict)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_info['name']}")

            log_info(f"Built prompt before interleave_image_tokens: {user_prompt}")

            if "internvl" in args.policy_model_path.lower() or "minicpm" in args.policy_model_path.lower():
                question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                    prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm(
                        user_prompt,
                        image_data_base64_list,
                        interleave_image_tokens=interleave_image_tokens,
                        model_type="internvl" if "internvl" in args.policy_model_path.lower() else "minicpm"
                    )
                )
            else:
                # TODO: For now we use the same function for both MMMU and PuzzleVQA, but be careful when using for MathVista
                question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                    prepare_question_array_with_base64_image_strings(
                        user_prompt,
                        image_data_base64_list,
                        interleave_image_tokens=interleave_image_tokens,
                    )
                )
            log_info(f"length of question_corresponding_image_data_base64_list: {len(question_corresponding_image_data_base64_list)}")
            
            system_prompt = "Answer the question using a single word or phrase."
            sys_and_user_prompt_messages_array = (
                [
                    {"role": "system", "content": system_prompt},
                ]
                + question_in_messages_array_format
            )
            log_info(f"Input prompt for generations: {sys_and_user_prompt_messages_array}")
            pred_answer = inference_single_step(
                policy_model, image_data_base64_list, sys_and_user_prompt_messages_array, EOS_TOKEN, policy_model_processor
            )

            result_record = {
                "question": question,
                "index": int(line["index"])
                if isinstance(line["index"], np.integer)
                else line["index"],
                "annotation": line_dict,
                "task": args.data,
                "gt_answer": line[ans_key],
                "raw_full_prediction": pred_answer,
                "pred_answer": None,  # as we are using non-CoT and want model to answer directly
                # "pred_answer": extract_boxed(pred_answer),  # use this for LLM Judge
            }
            
            # Add puzzle_type for PuzzleVQA 1K subset analysis
            if dataset_info["is_puzzle"] and "puzzle_type" in line_dict:
                result_record["puzzle_type"] = line_dict["puzzle_type"]
            
            new_dataset.append(result_record)

        # Write results to output file after processing all data
        # Add partition suffix if running in parallel mode
        if args.partition_id is not None:
            output_file = os.path.join(
                args.output_dir,
                f"result-p{args.partition_id}-{args.data_begin}-{args.data_end}-{run_datetime}.json",
            )
        else:
            output_file = os.path.join(
                args.output_dir,
                f"result-{args.data_begin}-{args.data_end}-{run_datetime}.json",
            )
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)

        log_info(f"Done! Results are saved to {output_file}.")

        # Calculate evaluation score based on dataset type
        log_info("Calculating basic evaluation score...")
        eval_score, correct_count, total_count = calculate_evaluation_score_direct(output_file)
        if eval_score is not None:
            eval_score_percent = f"{eval_score:.2%}"
            eval_summary = (
                f"Score: {eval_score_percent} ({correct_count}/{total_count} correct)"
            )
        else:
            eval_score_percent = "N/A"
            eval_summary = "Score calculation failed"

        log_info(f"Evaluation summary: {eval_summary}")

            # Run MathVista judge evaluation if dataset is mathvista
        judge_score = None
        judge_score_percent = "N/A"
        num_correct_samples_after_llm_judgement = None
        num_samples_after_llm_judgement = None

        if "mathvista" in args.data.lower():
            try:
                log_info("Running MathVista-specific LLM judge evaluation...")
                judge_results = calculate_mathvista_judge_evaluation_score(output_file)

                if judge_results is not None:
                    judge_score = judge_results["overall_accuracy"]
                    judge_score_percent = f"{judge_score:.2%}"
                    num_correct_samples_after_llm_judgement = judge_results["num_correct_samples_after_llm_judgement"]
                    num_samples_after_llm_judgement = judge_results["num_samples_after_llm_judgement"]
                    log_info(f"MathVista judge evaluation completed: {judge_score_percent}")
                    log_info(f"Judge correct samples: {num_correct_samples_after_llm_judgement}/{num_samples_after_llm_judgement}")
                else:
                    log_info("MathVista judge evaluation failed - no results returned")
            except Exception as e:
                log_info(f"Judge evaluation error: {e}")

        # Send success telegram notification
        try:
            eval_score_percent_str = f"{eval_score:.2%}" if eval_score is not None else "N/A"

            # Prepare judge breakdown string
            judge_breakdown = ""
            if num_correct_samples_after_llm_judgement is not None and num_samples_after_llm_judgement is not None:
                judge_breakdown = f" [{num_correct_samples_after_llm_judgement}/{num_samples_after_llm_judgement}]"

            # Create evaluation summary
            extraction_description = "LLM Answer Extraction tries to extract answer from the text if not given a boxed answer."
            eval_summary_msg = f"""{args.data}:
Basic: {correct_count}/{total_count} = {eval_score_percent_str}
With LLM Answer Extraction: {judge_breakdown} = {judge_score_percent}

Basic Score includes null answer samples which are considered as incorrect;
{extraction_description}"""

            # Send telegram notification
            send_telegram_job_summary(
                model_path_name=args.policy_model_path,
                evaluation_results_json_file=output_file,
                evaluation_run_logs_file=os.getenv("EVAL_RUN_LOG_FILE", ""),
                extra_fields={
                    "policy_model_path": args.policy_model_path,
                    "dataset": args.data,
                    "total_samples": total_count,
                    "correct_answers": correct_count,
                    "basic_accuracy": eval_score_percent_str,
                    "judge_accuracy": judge_score_percent,
                    "judge_correct_samples": num_correct_samples_after_llm_judgement if num_correct_samples_after_llm_judgement is not None else "N/A",
                    "judge_total_samples": num_samples_after_llm_judgement if num_samples_after_llm_judgement is not None else "N/A",
                    "data_begin": args.data_begin,
                    "data_end": args.data_end,
                    "partition_id": args.partition_id if args.partition_id is not None else "N/A",
                    "run_datetime": run_datetime,
                },
                separator="\t",
                include_header=True,
                send_files=True,
                message_prefix=f"✅[Eval Success]\n{eval_summary_msg}",
            )
            log_info("Telegram success notification sent")
        except Exception as e:
            log_info(f"Failed to send success notification to Telegram: {e}")

    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        log_info(f"Evaluation failed with error: {error_message}")
        log_info(f"Full traceback: {error_traceback}")

        # Check if this is an OOM error
        is_oom = any(
            oom_indicator in str(e).lower() or oom_indicator in error_traceback.lower()
            for oom_indicator in [
                "out of memory",
                "oom",
                "cuda out of memory",
                "memory error",
            ]
        )

        # Send error notification via Telegram
        # try:
        #     # Add OOM indicator and partition info to the message
        #     error_msg_prefix = ""
        #     if is_oom:
        #         error_msg_prefix = "🚨 OOM ERROR: "
        #     if args.partition_id is not None:
        #         error_msg_prefix += f"Partition {args.partition_id} failed - "

        #     send_telegram_error_notification(
        #         model_path_name=args.policy_model_path,
        #         error_message=error_msg_prefix + error_message,
        #         error_traceback=error_traceback,
        #         evaluation_run_logs_file=os.getenv("EVAL_RUN_LOG_FILE", ""),
        #         extra_fields={
        #             "data": args.data,
        #             "data_begin": args.data_begin,
        #             "data_end": args.data_end,
        #             "development_mode": args.development_mode,
        #             "partition_id": args.partition_id,
        #             "is_oom_error": is_oom,
        #             "policy_gpu": args.policy_gpu,
        #         },
        #         send_files=True,
        #     )
        # except Exception as telegram_error:
        #     log_info(f"Failed to send error notification to Telegram: {telegram_error}")

        # # Re-raise the original exception to maintain proper exit code
        # raise


if __name__ == "__main__":
    main()