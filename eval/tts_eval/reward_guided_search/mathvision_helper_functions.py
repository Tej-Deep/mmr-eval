try:
    from .logger import log_info
except ImportError:
    # Handle case when run as script
    from utils.logger import log_info
from datasets import load_dataset
import pandas as pd
import re

import json
from tqdm import tqdm
# from latex2sympy2 import latex2sympy # TODO: remember to activate when using MathVision as Judge
import time 
from math import *

import json
from tqdm import tqdm  # Assuming tqdm is imported to show progress bar
import time

import os
import math
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dotenv
import json
dotenv.load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    log_info(os.environ["OPENAI_API_KEY"][:5])
else:
    log_info("OPENAI_API_KEY not found")
    exit()  

def load_mathvision_dataset(dataset_name='MathLLMs/MathVision', test_split_name='testmini'):
    """Load MathVista dataset from HuggingFace and convert to DataFrame format compatible with existing pipeline."""
    log_info(f"Loading MathVision dataset {dataset_name}, split {test_split_name}...")
    
    # Load from HuggingFace (following MathVista generate_response.py:108-113)
    data_list = load_dataset(dataset_name, split=test_split_name)
    
    # Convert to DataFrame format compatible with existing pipeline
    data_records = []
    for item in data_list:

        # Convert HF dataset item to record format
        record = {
            "index": item["id"],  # Use pid as index
            "id": item["id"],  # Also store as id for consistency
            "question": item["question"],
            "answer": item['answer'],
            "options": item.get("options", []),
            "level": item.get("level", ""),
            "subject": item.get("subject", ""),
            "solution": item.get("solution", ""),
        }
        
        # Convert PIL image to base64 string
        if 'decoded_image' in item and item['decoded_image'] is not None:
            # Convert PIL image to base64 string
            import io
            import base64
            buffered = io.BytesIO()
            item['decoded_image'].save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            record['image'] = img_base64
        else:
            record['image'] = None
            
        data_records.append(record)
    
    # Convert to DataFrame
    data_df = pd.DataFrame(data_records)
    log_info(f"Loaded {len(data_df)} MathVista samples")
    
    return data_df

# Define the function to process each example
def build_mathvision_prompt_original(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    return input

def build_mathvision_prompt_minicpm(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and then output the final answer in the format of "Answer: single number or single word or phrase"\n\nWhen working out the reasoning / intermediate steps, 请一步步推理, especially when that helps clarity.\n\n' + f"{question}\n{options}"
    return input

def build_mathvision_prompt(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    user_prompt = ""

    user_prompt += f"Question: {question}\n"

    if options and len(options) > 0:
        user_prompt += f"Options:{options}"
        user_prompt += "Please select the correct answer from the options above.\n"

    # user_prompt += "If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. Then, work through your reasoning step by step to arrive at your answer. **Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** Put your final answer within \\boxed{}. If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer."

    # user_prompt = user_prompt.rstrip()
    return user_prompt

# Define the function to process each example
def build_mathvision_prompt_notags(example):
    question = example['question']
    cleaned_question = re.sub(r"<image\d+>", "", question)
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{cleaned_question}\n{options}"
    return input

# start copying from here
def is_nan_value(value):
    """Check if value is any form of NaN/null"""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() in ["nan", "null", ""]:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


class JudgeExtractorVLMEvalKit:
    """
    LLM Judge model for MathVision answer extraction fallback.

    Implements the exact ChainThoughtMultiExtractPrompter logic from the
    reference MathVista implementation.
    """

    # def __init__(self, model_path: str = "gpt-4.o-mini", temperature: float = 0.0):
    def __init__(self, model_path: str = "gpt-4.1-mini", temperature: float = 0.0):
    # def __init__(self, model_path: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the MathVision judge model.

        Args:
            model_path: Path to the policy model for judge inference
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.model_path = model_path
        self.temperature = temperature
        self.model = None

    def load_model(self):
        """Load the LLM model for judge inference."""
        if self.model is None:
            self.model = OpenAI()

    def judge_single_sample(self, sample_data: Dict) -> Tuple[str, bool]:
        """
        Run LLM judge on a single sample to extract the answer.

        Args:
            sample_data: Sample data dictionary from results file

        Returns:
            Tuple of (extracted_answer, success_flag)
        """
        self.load_model()

        try:
            # Extract information from sample data
            original_question = sample_data.get("question", "")
            # final_steps = sample_data.get("final_steps", [])
            annotation = sample_data.get("annotation", {})
            reasoning_trajectory = sample_data.get("prediction_full_text", "")

            # Get options from annotation
            choices_raw = annotation.get("options", [])

            # Parse choices from JSON string format
            if isinstance(choices_raw, str) and not is_nan_value(choices_raw):
                try:
                    import ast

                    options = ast.literal_eval(choices_raw) # parse the string to an actual list of options
                except (ValueError, SyntaxError):
                    raise ValueError(f"Failed to parse a valid list of options: {choices_raw}")
            elif isinstance(choices_raw, list):
                options = choices_raw
            elif is_nan_value(choices_raw):  # Check for NaN
                options = []
            else:
                raise ValueError(f"choices_raw is not a list or a string and not NaN: {choices_raw}")

            judge_prompt = self._construct_judge_prompt(
                original_question, options, reasoning_trajectory
            )
            response = self.model.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "text"},
                temperature=self.temperature,
                max_completion_tokens=256,
            )
            judge_output = response.choices[0].message.content.strip()

            if options:
                alphabetical_options = [f"{chr(65 + i)}" for i in range(len(options))]
                valid_responses = alphabetical_options + ["Z"]
                if judge_output in valid_responses:
                    extraction_method = "Valid alphabetical option"
                else:
                    extraction_method = "Invalid alphabetical option"
            elif not options:
                extraction_method = "Open-ended answer extraction"

            output = {
                "judge_output": judge_output,
                "extraction_method": extraction_method,
                "reasoning_trajectory": reasoning_trajectory,
            }

            return output

        except Exception as e:
            log_info(f"Error in judge_single_sample: {e}")
            return {
                "judge_output": f"Judge error: {str(e)}",
                "extraction_method": "Judge error",
                "reasoning_trajectory": "",
            }

    def _construct_judge_prompt(
        self, question: str, options: List[str], reasoning_trajectory: str
    ) -> str:
        """
        Reconstruct the original MathVista prompt format.

        Args:
            question: The question text
            options: List of answer options

        Returns:
            Reconstructed prompt in MathVista format
        """
        example_1 = """
    Hint: Please answer the question requiring an integer answer and provide the final value,
    e.g., 1, 2, 3, at the end.\n
    Question: Which number is missing?\n
    Model response: The number missing in the sequence is 14.\n
    Extracted answer: 14
    """

        example_2 = """
    Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
    e.g., 1.2, 1.3, 1.4, at the end.\n
    Question: What is the fraction of females facing the camera?\n
    Model response: The fraction of females facing the camera is 0.6,
    which means that six out of ten females in the group are facing the camera.\n
    Extracted answer: 0.6
    """

        example_3 = """
    Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
    e.g., 1.23, 1.34, 1.45, at the end.\n
    Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
    Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
    Extracted answer: 1.45
    """

        example_4 = """
    Hint: Please answer the question requiring a Python list as an answer and provide the final list,
    e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
    Question: Between which two years does the line graph saw its maximum peak?\n
    Model response: The line graph saw its maximum peak between 2007 and 2008.\n
    Extracted answer: [2007, 2008]
    """

        example_5 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: What fraction of the shape is blue?\n
    Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
    Model response: The correct answer is (B) 8/11.\n
    Extracted answer: B
    """

        example_6 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: What are the equivalent units for conversion costs for each quarter using the weighted-average method? Assume that the quarters are independent.\n
    Choices: (A) 132,625 (B) 134,485 (C) 135,332 (D) 132,685\n
    Model response: Therefore, the correct option is 132,685.\n\nAnswer: D.\n
    Extracted answer: D
    """

        example_7 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: Calculate the missing values of company 2.\n
    Choices: (A) $1,620 (B) $12,000 (C) $51,180 (D) $0\n
    Model response: Therefore, the missing value for Co.2 is $51,180.\n\nAnswer: C.\n
    Extracted answer: C
    """

        example_8 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: The primary disaccharide digestion product of starch is\n
    Choices: (A) <image 1> (B) <image 2> (C) <image 3> (D) <image 4>\n
    Model response: Based on this analysis, the correct answer is the one that shows the structure of maltose.\n\nAnswer: A.\n
    Extracted answer: A
    """
        
        example_9 = """
    Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
    Question: If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?\n
    Choices: (A) $3 (B) $4 (C) $5 (D) $6\n
    Model response: Therefore, the per unit manufacturing overhead costs incurred when 30,000 units are produced is $5.\n\nAnswer: C.\n
    Extracted answer: C
    """

        ice_examples = [
            example_1,
            example_2,
            example_3,
            example_4,
            example_5,
            example_6,
            example_7,
            example_8,
            example_9,
        ]

        task_description = """
    Please read the following example.
    Then extract the answer from the model response and type it at the end of the prompt.\n
    """
        # Notice above examples have hints in them, not always true. Also they have Choices in their questions, not always true so we have to add the options string manually if so below, TODO: To check for every dataset how options work.
        judge_prompt = task_description
        for example in ice_examples:
            judge_prompt += example + '\n'
        judge_prompt += f"Question: {question}\n"

        if options and ''.join(options) != 'ABCDE':
            options_str = ""
            for i, option in enumerate(options):
                letter = chr(ord("A") + i)
                options_str += f"({letter}) {option} " # format for MathVision
            if len(options_str) > 0:
                judge_prompt += f"Choices: {options_str} \n"
                print(f"judge_prompt with choices: {judge_prompt}")

        judge_prompt += f"Model response: {reasoning_trajectory}\n"
        judge_prompt += 'Extracted answer:'
        return judge_prompt


def eval_single_sample_vlmevalkit(item):
    """Evaluate a single sample."""

    judge_model = JudgeExtractorVLMEvalKit(model_path="gpt-4.1-mini", temperature=0.0)

    try:
        output = judge_model.judge_single_sample(item)
        extracted_answer = output["judge_output"]
        extraction_method = output["extraction_method"]
    except Exception as e:
        log_info(f"Error in judge_single_sample: {e}")
        return {
            "judge_output": f"Judge error: {str(e)}",
            "extraction_method": "Judge error",
            "reasoning_trajectory": "",
        }

    # Determine if the answer is correct
    hit = 1 if extracted_answer == item['gt_answer'] else 0
    # hit = 1 if is_answer_match(extracted_answer, item["gt_answer"]) else 0
    log_info(
        f"extracted_answer: {extracted_answer}, item['gt_answer']: {item['gt_answer']}, hit: {hit}"
    )
    log_info(f"reasoning_trajectory: {output['reasoning_trajectory']}")

    return {
        "index": item["index"],
        "split": item.get("split")
        or "Judge Evaluation (MCQ and Open-Ended Float/Integer)",  # Mathvista open-ended is ONLY float or Integer
        "question": item["question"],
        "prediction": item["prediction_full_text"],
        "extracted_answer": extracted_answer,
        "extraction_model": judge_model.model_path
        + "-temperature-"
        + str(judge_model.temperature)
        if judge_model.model_path and judge_model.temperature is not None
        else "gpt-4.1-judge-temperature-0.0",
        # "extraction_success": success,
        "extraction_method": extraction_method,
        "extraction_log": "Judge output: " + extracted_answer,
        "gt": item["gt_answer"],
        "hit": hit,
    }

def evaluate_null_results(null_result_items):
    complex_cases = []

    for i, sample in enumerate(null_result_items):
        if i % 50 == 0:  # Progress logging
            log_info(f"Processing sample {i + 1}/{len(null_result_items)}")

        gt_answer = sample.get("gt_answer", "").strip()

        choices_raw = sample.get("annotation", {}).get("options", [])
        print(f"choices_raw: {choices_raw}")

        # Parse choices from JSON string format
        if isinstance(choices_raw, str) and not is_nan_value(choices_raw):
            try:
                import ast

                choices = ast.literal_eval(choices_raw)
            except (ValueError, SyntaxError):
                raise ValueError(f"Failed to parse a valid list of options: {choices_raw}")
        elif isinstance(choices_raw, list) and len(choices_raw) > 0:
            choices = choices_raw
        elif any(
            sample.get("annotation", {}).get(key)
            for key in ["A", "B", "C", "D", "E", "F"]
        ):
            choices = []
            for key in ["A", "B", "C", "D", "E", "F"]:
                value = sample.get("annotation", {}).get(key)
                if (
                    value and str(value).strip() and not is_nan_value(value)
                ):  # Check if non-empty and not just whitespace
                    choices.append(value)
        elif is_nan_value(choices_raw):  # Check for NaN
            choices = []
        else:
            raise ValueError(f"choices_raw is not a list or a string and not NaN: {choices_raw}")
        print(f"sample.annotation.keys(): {sample['annotation'].keys()}")
        print(f"parsed choices: {choices}")
        complex_cases.append(sample)
    print("number of complex cases: ", len(complex_cases))
    print("total number of cases processed: ", len(complex_cases))
    judge_samples = []
    for result in complex_cases:
        if result.get("raw_full_prediction"):
            prediction_text = str(result["raw_full_prediction"])
        else:
            raise ValueError(f"No prediction text found: {result}")

        gt_answer = result.get("gt_answer", "").strip()

        judge_item = result.copy()
        judge_item["prediction_full_text"] = prediction_text.strip()
        judge_item["gt_answer"] = gt_answer
        judge_item["split"] = "null result cases (using judge extraction)"
    
        judge_samples.append(judge_item)

    # Initialize judge_eval_results as empty list
    judge_eval_results = []

    # Only process complex cases if they exist
    if judge_samples:
        # Run threaded evaluation
        with ThreadPoolExecutor(max_workers=200) as executor:
            for result in tqdm(
                executor.map(eval_single_sample_vlmevalkit, judge_samples),
                total=len(judge_samples),
                desc="Judge Evaluation",
            ):
                judge_eval_results.append(result)
    else:
        log_info(
            "No complex cases requiring judge evaluation - all samples handled by simple string matching"
        )
    return judge_eval_results

def calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(results_file: str) -> Optional[Dict]:
    """
    Calculate MathVision evaluation scores with LLM Judge fallback mechanism.

    This function implements the two-stage evaluation approach:
    1. Count initial regex-based extraction results
    2. Apply LLM Judge to failed extractions
    3. Calculate and return both sets of statistics

    Args:
        results_file: Path to JSON results file from evaluation

    Returns:
        Dictionary with evaluation statistics matching MMMU judge interface:
        {
            "overall_accuracy": float,
            "num_correct_samples_after_llm_judgement": int,
            "num_samples_after_llm_judgement": int,
            "initial_correct_count": int,
            "initial_total_count": int,
            "judge_improved_count": int,
            "judge_processed_count": int
        }
    """
    log_info("Starting LLM Judge evaluation...")

    try:
        # Load results file
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            log_info("Warning: Empty results file")
            return None

        # Initialize counters
        initial_total = len(results)

        # Process each sample
        log_info(f"Processing {initial_total} samples for judge evaluation...")
        null_results = [
            result for result in results if result.get("pred_answer") is None
        ]

        null_eval_results = evaluate_null_results(null_results)
        log_info(f"Null judge evaluation results: {len(null_eval_results)}")
        split_accuracy = sum(r["hit"] for r in null_eval_results) / len(null_eval_results)
        log_info(f"Null judge evaluation split accuracy: {split_accuracy:.4f} ({sum(r['hit'] for r in null_eval_results)}/{len(null_eval_results)})")
        
        return {
            "overall_accuracy": split_accuracy,
            "accuracy_by_split": {
                "null_result_cases_using_judge_extraction": split_accuracy
            },
            "eval_results": null_eval_results,
            "num_correct_samples_after_llm_judgement": sum(r['hit'] for r in null_eval_results),
            "num_samples_after_llm_judgement": len(results) # careful with this, make sure check how NULL values are handled and sums to full dataset
        }
    except Exception as e:
        log_info(f"Error in judge evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    """Test the MathVision judge evaluation on a results file."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python mathvision_helper_functions.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    print(f"Testing MathVision judge evaluation on: {results_file}")
    judge_results = (
        calculate_mathvision_judge_evaluation_score_all_sample_gpt_extraction(
            results_file
        )
    )

    if judge_results:
        print("\nJudge evaluation completed successfully!")
    else:
        print("Judge evaluation failed!")
        sys.exit(1)