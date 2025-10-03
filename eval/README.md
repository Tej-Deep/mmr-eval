# Instructions to set up virtual environment for evaluation
- We recommend using uv to create a virtual environment and following the sequence below to install the dependencies. The sequence of installations are important as it ensures compatibility of dependencies.

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