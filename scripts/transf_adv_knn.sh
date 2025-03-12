#!/bin/bash


# Default values for arguments
EXP_NUMBER=${1:-1}

if [ $EXP_NUMBER -eq 1 ]; then

  # Run evaluation scripts
  echo "Running evaluation scripts for experiment 1 source models"
  bash scripts/eval_transf_adv_knn.sh resnet50 1 Results/Baseline/resnet50_exp1 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 1 Results/Baseline/resnet50_timm_pretrained_exp1 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_at 1 Results/Baseline/resnet50_at_exp1 8 10
  echo "All evaluation scripts executed successfully!"


fi


if [ $EXP_NUMBER -eq 2 ]; then

  echo "Running evaluation scripts for experiment 2 source models"

  bash scripts/eval_transf_adv_knn.sh resnet50 2 Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_at 2 Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 2 Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2 8 10


fi


if [ $EXP_NUMBER -eq 3 ]; then

  echo "Running evaluation scripts for experiment 3 source models"
  bash scripts/eval_transf_adv_knn.sh resnet50 3 Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp3 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_at 3 Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp3 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 3 Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp3 8 10


fi

