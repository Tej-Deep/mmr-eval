<div align="center">
  <h1>VL-PRMs: Vision-Language Process Reward Models</h1>
  <p>Training VL-PRMs with abstract reasoning problems results in strong generalization and reasoning performance improvements for Qwen2.5-VL and Gemma 3 family of models when used in test-time scaling settings.
 </p>
</div>
<br>

![](visuals/)

****************************************************************

**Updates:**

* 2025-09: Released VL-PRMs: [Arxiv]()

****************************************************************

This repository provides an overview of all resources for the paper ["Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning: Key Insights and Lessons Learned"]().

- [Artifacts](#artifacts)
- [Structure](#structure)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
    - [vLLM](#vllm)
    - [transformers](#transformers)
- [Visuals](#visuals)
- [Known Issues](#known-issues)
- [Citation](#citation)

### Artifacts

- **Paper**:
- **VL-PRM300K Dataset**: 
    - [Raw Version](https://huggingface.co/datasets/ob11/VL-PRM300K)
    - [Training Version](https://huggingface.co/datasets/ob11/VL-PRM300K-V1-train) 
- **Model**:
    - [VL-PRM-3B](https://huggingface.co/ob11/Qwen-VL-PRM-3B)
    - [VL-PRM-7B](https://huggingface.co/ob11/Qwen-VL-PRM-7B)
- **Weight and Biases Training Logs**:
    - [VL-PRM-3B](https://wandb.ai/aisg-arf/multimodal-reasoning/runs/pnsncs80)
    - [VL-PRM-7B](https://wandb.ai/aisg-arf/multimodal-reasoning/runs/pj4oc0qh)
- **Evaluation Files**: 
    - All our evaluation result files are at: https://huggingface.co/datasets/ob11/VL-PRM-Evaluation-Results

### Structure

- `rollout_data_generation/`: Synthetic data creation code
- `train/`: Training scripts
- `eval/`: Evaluation scripts

### Training


To run training, you can find the script `train/sft_qwen.py` by using the `train/sft_qwen.sh` script, or you can launch a sweep via `train/launch.sh` if you are on a PBS cluster.

To train VL-PRM-3B/VL-PRM-7B, we recommend 8 H100/H200 GPUs i.e. 1 node with 8 GPUs. We used 8 H200 GPUs for most of our jobs.

Quick start:
```
git clone https://github.com/ob11/VL-PRMs.git
cd VL-PRMs
uv pip install -r requirements.txt
bash train/sft_qwen.sh
```
*Note: If you encounter an out-of-memory (OOM) issues, consider reducing the batch size and gradient accumulation steps.*

### Evaluation

We cloned [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) at commit `<commit_hash>` and modified it accordingly based on the base policy model and evaluation task selected. Setup:

You can launch the evaluation using `eval/launch_eval.sh`. 

All our evaluation result files are at: https://huggingface.co/datasets/ob11/VL-PRM-Evaluation-Results/

### Data

To recreate VL-PRM-300K follow the steps below. 
1. 

### Known Issues

- Reproducing the results in the paper requires using the exact same seed for the synthetic data generation and training the VL-PRMs.

### Citation

```bibtex
@misc{ong2025vlprms,
      title={VL-PRMs: Vision-Language Process Reward Models}, 
      author={Brandon Ong, Tej Deep Pala, Vernon Toh, William Chandra Tjhi and Soujanya Poria},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}, 
}
```