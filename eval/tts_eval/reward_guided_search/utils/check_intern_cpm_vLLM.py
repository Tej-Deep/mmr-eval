from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from transformers import AutoTokenizer
from typing import List
import torch
import sys
sys.path.append('/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation')
from reward_guided_search.utils import prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm

QUESTION = "What is the content of each image?"
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
]

models = [
    # "OpenGVLab/InternVL2_5-8B",
    # "openbmb/MiniCPM-V-2_6",
]

for model in models:
    llm = LLM(
        model=model,
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": len(IMAGE_URLS)}
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    placeholders = "".join("(<image>./</image>)\n" for _ in IMAGE_URLS) # MiniCPM
    # placeholders = "\n".join(
    #     f"Image-{i}: <image>\n" for i, _ in enumerate(IMAGE_URLS, start=1)
    # ) # InternVL
    messages = [{
        'role': 'user',
        'content': f'{placeholders}\n{QUESTION}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # internVL_stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    # miniCPM_stop_tokens = ["<|im_end|>", "<|endoftext|>"]
 
    print(f"{model}: {prompt}")
    print("-"*100)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [fetch_image(url) for url in IMAGE_URLS]
            },
        },
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
        )
    )
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
