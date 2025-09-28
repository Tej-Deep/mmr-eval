#!/bin/bash

DIR="/home/ubuntu/porialab-us-midwest-1/Tej/mmr-eval/math_vision_outputs/M26_policy_mathvision"
# DIR="/home/ubuntu/porialab-us-midwest-1/Tej/mmr-eval/math_vision_outputs/Q32B_MathV"

# loop over all .json files in DIR and subdirs
find "$DIR" -type f -name "*.json" | while read -r file; do
    echo "Found: $file"
    
    python calc_score_mathvision.py \
      --data-path $file
done