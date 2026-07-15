#!/bin/bash
set -euo pipefail

gpu_id=${1:-0}
checkpoint=${2:-../smctm_robot_policy/outputs/dinov3_smctm/latest.pt}
test_num=${3:-10}

cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES="$gpu_id" .venv/bin/python script/eval_policy.py \
  --config policy/SMCTM/deploy_policy.yml \
  --overrides \
  --device cuda:0 \
  --checkpoint "$checkpoint" \
  --test_num "$test_num" \
  --ckpt_setting "$(basename "$(dirname "$checkpoint")")"
