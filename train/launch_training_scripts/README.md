```
cd qwen_training
bash scripts/launch_test_7b_sft.sh
bash scripts/launch_sweep_7b_sft.sh
```

Important to ensure env is specific to the requirements.txt file and the updated token models follow transformers=4.50.0 format because qwen SFT code will not working with other transformers versions