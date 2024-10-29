#!/bin/bash
# chmod +x evaluate_adapt_baselines.sh

dataset="drebin" # "malradar" 

# Array of feature names
features=("drebin" "damd" "mamadroid") #  "damd" "mamadroid"

samplers=("entropy+") # "entropy" "cade" "transcendent" "hcc" "hcc+" "cade+" "transcendent+" "uncertainty+" "uncertainty"

# Array of budgets
budgets=(10 20 30 40 100) # 

# models are already trained and saved
gpu_id="3"
# Loop through each feature
for feature in "${features[@]}"
do
    if [ "${feature}" = "damd" ]; then
        batch_size="20"
    else
        batch_size="32"
    fi
    # Nested loop for each budget
    for sampler in "${samplers[@]}"
    do
        for budget in "${budgets[@]}"
        do
            python adapt_baselines.py --budget "$budget" --feature_name "$feature" --data_name "$dataset" --gpu_id "$gpu_id" --batch_size "$batch_size" --sampler_name "$sampler" 
        done
    done
    python adapt_baselines.py --feature_name "$feature" --data_name "$dataset" --display
done
