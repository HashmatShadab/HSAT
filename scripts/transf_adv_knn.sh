#!/bin/bash


# Default values for arguments
EXP_NUMBER=${1:-18}

if [ $EXP_NUMBER -eq 18 ]; then

  # Run evaluation scripts
  echo "Running evaluation scripts for experiment 18 source models"
  bash scripts/eval_transf_adv_knn.sh resnet50 18 Results/Baseline/resnet50_exp18 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 18 Results/Baseline/resnet50_timm_pretrained_exp18 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_at 18 Results/Baseline/resnet50_at_exp18 8 10

  bash scripts/eval_transf_adv_knn.sh wresnet50_normal 18 Results/Baseline/wresnet50_normal_exp18 8 10
  bash scripts/eval_transf_adv_knn.sh wresnet50_at 18 Results/Baseline/wresnet50_at_exp18 8 10
  bash scripts/eval_transf_adv_knn.sh resnet101_normal 18 Results/Baseline/resnet101_normal_exp18 8 10
  bash scripts/eval_transf_adv_knn.sh resnet101_at 18 Results/Baseline/resnet101_at_exp18 8 10
  echo "All evaluation scripts executed successfully!"


fi


if [ $EXP_NUMBER -eq 19 ]; then

  echo "Running evaluation scripts for experiment 19 source models"

  bash scripts/eval_transf_adv_knn.sh resnet50 19 Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_at 19 Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 19 Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10


#  bash scripts/eval_transf_adv_knn.sh wresnet50_normal 19 Results/Adv/wresnet50_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10
#  bash scripts/eval_transf_adv_knn.sh wresnet50_at 19 Results/Adv/wresnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10
#  bash scripts/eval_transf_adv_knn.sh resnet101_normal 19 Results/Adv/resnet101_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10
#  bash scripts/eval_transf_adv_knn.sh resnet101_at 19 Results/Adv/resnet101_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 8 10
#

fi


if [ $EXP_NUMBER -eq 20 ]; then

  echo "Running evaluation scripts for experiment 20 source models"
  bash scripts/eval_transf_adv_knn.sh resnet50 20 Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp20 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_at 20 Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20 8 10
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 20 Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp20 8 10


#  bash scripts/eval_transf_adv_knn.sh wresnet50_at 20 Results/Adv/wresnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20 8 10
#  bash scripts/eval_transf_adv_knn.sh wresnet50_normal 20 Results/Adv/wresnet50_normal_dynamicaug_true_epsilon_warmup_5000_exp20 8 10
#  bash scripts/eval_transf_adv_knn.sh resnet101_at 20 Results/Adv/resnet101_at_dynamicaug_true_epsilon_warmup_5000_exp20 8 10
#  bash scripts/eval_transf_adv_knn.sh resnet101_normal 20 Results/Adv/resnet101_normal_dynamicaug_true_epsilon_warmup_5000_exp20 8 10
#

fi

if [ $EXP_NUMBER -eq 0 ]; then

  echo "Running evaluation scripts for experiment 00 source models (ImageNet Models)"
  bash scripts/eval_transf_adv_knn.sh resnet50_timm_pretrained 18 Results/Baseline/resnet50_timm_pretrained_exp18 8 10 false
  bash scripts/eval_transf_adv_knn.sh resnet50_at 18 Results/Baseline/resnet50_at_exp18 8 10 false
  bash scripts/eval_transf_adv_knn.sh wresnet50_at 18 Results/Baseline/resnet50_at_exp18 8 10 false
  bash scripts/eval_transf_adv_knn.sh wresnet50_normal 18 Results/Baseline/resnet50_at_exp18 8 10 false
  bash scripts/eval_transf_adv_knn.sh resnet101_at 18 Results/Baseline/resnet50_at_exp18 8 10 false
  bash scripts/eval_transf_adv_knn.sh resnet101_normal 18 Results/Baseline/resnet50_at_exp18 8 10 false


fi