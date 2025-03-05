#!/bin/bash


# Default values for arguments
EXP_NUMBER=${1:-1}


if [ $EXP_NUMBER -eq 1 ]; then

  echo "Running evaluation scripts for experiment CLIP source models"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-B/16" 8 10 "pre_projection_features_all_layers"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-B/16" 8 10 "pre_projection_features"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-B/16" 8 10 "projection_features"


  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-L/14" 8 10 "pre_projection_features_all_layers"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-L/14" 8 10 "pre_projection_features"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-L/14" 8 10 "projection_features"

  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-B/16" 8 20 "pre_projection_features_all_layers"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-B/16" 8 20 "pre_projection_features"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-B/16" 8 20 "projection_features"

  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-L/14" 8 20 "pre_projection_features_all_layers"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-L/14" 8 20 "pre_projection_features"
  bash scripts/eval_transf_adv_clip_knn.sh "CLIP-ViT-L/14" 8 20 "projection_features"

fi

if [ $EXP_NUMBER -eq 2 ]; then

  echo "Running evaluation scripts for experiment MedCLIP source models"
  bash scripts/eval_transf_adv_clip_knn.sh "MedCLIP-ViT-B-16-quickgelu" 8 10 "pre_projection_features_all_layers"
  bash scripts/eval_transf_adv_clip_knn.sh "MedCLIP-ViT-B-16-quickgelu" 8 10 "pre_projection_features"
  bash scripts/eval_transf_adv_clip_knn.sh "MedCLIP-ViT-B-16-quickgelu" 8 10 "projection_features"



  bash scripts/eval_transf_adv_clip_knn.sh "MedCLIP-ViT-B-16-quickgelu" 8 20 "pre_projection_features_all_layers"
  bash scripts/eval_transf_adv_clip_knn.sh "MedCLIP-ViT-B-16-quickgelu" 8 20 "pre_projection_features"
  bash scripts/eval_transf_adv_clip_knn.sh "MedCLIP-ViT-B-16-quickgelu" 8 20 "projection_features"


fi