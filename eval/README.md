# Instructions to set up virtual environment for evaluation
- We recommend using uv to create a virtual environment and following the sequence below to install the dependencies. The sequence of installations are important as it ensures compatibility of dependencies.
    - We had the most success with uv's standalone installer (not the pip version), and recommend this for this project as well.
    - ```curl -LsSf https://astral.sh/uv/install.sh | sh```

# Set Up Virtual Environment
- We recommend you first try syncing the dependencies using our pyproject.toml file.
```bash
module load CUDA/12.6.0
uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip sync
```

- If the above fails, you can try syncing the dependencies using our requirements.txt file. You will need to install flash-attn manually after syncing.
```bash
module load CUDA/12.6.0
uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip sync requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
uv add flash-attn --no-build-isolation
```

## Testing Environment
- To test if the environment is set up correctly, you can run the following command:
```bash
cd base_model_eval/vLLM_evaluation_code
./test_run_base_model_eval.sh
```

- Set an OpenAI API key in a .env file in the parent directory, as this evaluation requires a LLM Judge.
- This will run the evaluation script for a Qwen2.5-VL-3B-Instruct base model.
- If everything is installed correctly, you should get a score of 6/8 on the development set of MathVista evaluation
- If you get a score of less than 6/8, please check that flash-attn is installed correctly, because you will get accuracy depreciation without it