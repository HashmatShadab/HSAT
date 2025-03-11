#!/bin/bash

# This script performs evaluation of K-Nearest Neighbors (KNN) models with or without adversarial attacks.
# It takes several parameters to configure the evaluation process.

# Set the path to the data directory
DATA_PATH=/path/to//data

# Get the experiment number from the first argument
EXP_NUMBER=${1:-1}
MODEL_NAME=${2:-"resnet50_timm_pretrained"}
# Set the adversarial evaluation flag (default: true)
adv_eval=${3:-"true"}

# Set the epsilon value for adversarial attacks (default: 8)
epsilon=${4:-8}

# Set the number of steps for adversarial attacks (default: 7)
steps=${5:-7}

# Set the batch size for evaluation (default: 64)
batch_size=${6:-64}

# Set the name of the adversarial attack (default: "pgd_knn")
attack_name=${7:-"pgd_knn"}

###################### Exp 1 ########################################
if [ $EXP_NUMBER -eq 1 ]; then
  
  # Set model name and checkpoint directory for the first model
  model_name=$MODEL_NAME
  ckpt_dir="Results/Baseline/${model_name}_exp1"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth
  echo "Exp 1 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path: $ckpt_path"

  # loop over all the checkpoints in the directory ending with .pth

    if [ $adv_eval == "true" ]; then
      python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
      --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name --eval_predict_batch_size $batch_size --attack_name $attack_name
    else
      python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
      --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
    fi

 

fi





if [ $EXP_NUMBER -eq 2 ]; then


  ###################### Exp 2 ########################################
  model_name=$MODEL_NAME
  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth
  echo "Exp 2 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path: $ckpt_path"


      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi


if [ $EXP_NUMBER -eq 3 ]; then


  ###################### Exp 3 ########################################
  model_name=$MODEL_NAME
  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_exp3"
  ckpt_path=$ckpt_dir/checkpoint_80000.pth
  echo "Exp 3 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi




fi



if [ $EXP_NUMBER -eq 4 ]; then


  ###################### Exp 4 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp4_with_embedding256"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth
  echo "Exp 4 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi




fi

if [ $EXP_NUMBER -eq 5 ]; then


  ###################### Exp 5 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp5_with_embedding128"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth
  echo "Exp 5 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 128   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 128   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


fi


if [ $EXP_NUMBER -eq 6 ]; then

  ###################### Exp 6 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp6_with_embedding512"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 6 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"


      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi




fi


if [ $EXP_NUMBER -eq 7 ]; then

  ###################### Exp 7 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp7_with_embedding768"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 7 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi

if [ $EXP_NUMBER -eq 8 ]; then

  ###################### Exp 8 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp8_with_embedding1024"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 8 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi




if [ $EXP_NUMBER -eq 9 ]; then



  ###################### Exp 9 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp9"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 9 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi

if [ $EXP_NUMBER -eq 10 ]; then


  ###################### Exp 10 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp10"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 10 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"


      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi


if [ $EXP_NUMBER -eq 11 ]; then

 
  ###################### Exp 11 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp11_with_adv_loss_pt"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 11 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"


      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi


if [ $EXP_NUMBER -eq 12 ]; then



  ###################### Exp 12 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp12_with_adv_loss_s_pt"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 12 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi




fi



if [ $EXP_NUMBER -eq 13 ]; then

 

  ###################### Exp 13 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp13"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 13 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi

if [ $EXP_NUMBER -eq 14 ]; then

  

  ###################### Exp 14 ########################################
  model_name=$MODEL_NAME

  ckpt_dir="Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp14"
  ckpt_path=$ckpt_dir/checkpoint_40000.pth

  echo "Exp 14 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"
  echo "ckpt_path $ckpt_path"



      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --attack_name $attack_name
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi



fi


