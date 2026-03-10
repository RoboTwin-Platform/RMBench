#!/bin/bash

policy_name=Mem-0

export CUDA_VISIBLE_DEVICES=0
echo -e "\033[33mGPU to use: 0\033[0m"

cd ../..  # move to project root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
    --task_name # TODO
    --execution_ckpt # TODO: path to checkpoint
    --state_stats_path ./policy/Mem-0/assets/x/norm_stats.json # TODO: replace x with correct path
    --global_task "..." # For Mn Tasks
    --vllm_url http:// # For Mn Tasks: replace with correct URL; not needed for M1 Tasks
    --action_horizon 30 # Changeable