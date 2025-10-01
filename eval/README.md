# Instructions to set up virtual environment for evaluation
- We recommend using uv to create a virtual environment and following the sequence below to install the dependencies. The sequence of installations are important as it ensures compatibility of dependencies.

```bash
uv venv -p 3.12 --seed
uv pip install torch --no-build-isolation
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation
uv pip sync requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```