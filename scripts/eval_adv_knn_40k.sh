#!/bin/bash

DATA_PATH=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data
EXP_NUMBER=$1
adv_eval=${2:-"true"}
epsilon=${3:-8}
steps=${4:-7}
batch_size=${5:-64}




######################################################################################################################
#Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20
#Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt
#Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt
#Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28
#Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29
#Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30
#Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp20
#Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28
#Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp20
#Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt
#Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt
#Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28
#Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29
#Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30

###################### Exp 18 ########################################
if [ $EXP_NUMBER -eq 18 ]; then

  model_name="resnet50_timm_pretrained"
  ckpt_dir="Results/Baseline/resnet50_timm_pretrained_exp18"
  echo "Exp 18 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done
    ###################### Exp 18 ########################################
  model_name="resnet50"
  ckpt_dir="Results/Baseline/resnet50_exp18"

  echo "Exp 18 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

       if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

    ###################### Exp 18 ########################################
  model_name="resnet50_at"
  ckpt_dir="Results/Baseline/resnet50_at_exp18"

  echo "Exp 18 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

       if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done


fi


if [ $EXP_NUMBER -eq 19 ]; then


  ###################### Exp 19 ########################################
  model_name="resnet50_timm_pretrained"
  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"
  echo "Exp 19 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

  ###################### Exp 19 ########################################
  model_name="resnet50"
  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"

  echo "Exp 19 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

  ###################### Exp 19 ########################################
  model_name="resnet50_at"
  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"
  echo "Exp 19 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path
      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

fi


if [ $EXP_NUMBER -eq 20 ]; then


  ###################### Exp 20 ########################################
  model_name="resnet50_timm_pretrained"
  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp20"
  echo "Exp 20 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

  ###################### Exp 20 ########################################
  model_name="resnet50"
  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp20"

  echo "Exp 20 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

  ###################### Exp 20 ########################################
  model_name="resnet50_at"
  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20"
  echo "Exp 20 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

fi


if [ $EXP_NUMBER -eq 24 ]; then


  ###################### Exp 24 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256"
  echo "Exp 24 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

  ###################### Exp 24 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256"
  echo "Exp 24 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

fi

if [ $EXP_NUMBER -eq 244 ]; then


  ###################### Exp 244 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp244_with_embedding128"
  echo "Exp 244 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 128   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 128   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

  ###################### Exp 244 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp244_with_embedding128"
  echo "Exp 244 with $model_name"

  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 128   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 128   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

fi


if [ $EXP_NUMBER -eq 25 ]; then

  ###################### Exp 25 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512"

  echo "Exp 25 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

  ###################### Exp 25 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512"

  echo "Exp 25 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

fi


if [ $EXP_NUMBER -eq 26 ]; then

  ###################### Exp 26 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp26_with_embedding768"

  echo "Exp 26 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

 ###################### Exp 26 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp26_with_embedding768"

  echo "Exp 26 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi

  done

fi

if [ $EXP_NUMBER -eq 27 ]; then

  ###################### Exp 27 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp27_with_embedding1024"

  echo "Exp 27 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

  ###################### Exp 27 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp27_with_embedding1024"

  echo "Exp 27 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

fi

if [ $EXP_NUMBER -eq 28 ]; then

  ###################### Exp 28 ########################################
  model_name="resnet50"

  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28"

  echo "Exp 28 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

  ###################### Exp 28 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28"

  echo "Exp 28 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

    ###################### Exp 28 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28"

  echo "Exp 28 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done




fi



if [ $EXP_NUMBER -eq 29 ]; then

  ###################### Exp 29 ########################################
  model_name="resnet50"

  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29"

  echo "Exp 29 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

  ###################### Exp 29 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29"

  echo "Exp 29 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

    ###################### Exp 29 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29"

  echo "Exp 29 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.proj_head=True model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done


fi

if [ $EXP_NUMBER -eq 30 ]; then

  ###################### Exp 30 ########################################
  model_name="resnet50"

  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30"

  echo "Exp 30 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

  ###################### Exp 30 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30"

  echo "Exp 30 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

    ###################### Exp 30 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30"

  echo "Exp 30 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048   \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done


fi


if [ $EXP_NUMBER -eq 31 ]; then

  ###################### Exp 31 ########################################
  model_name="resnet50"

  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt"

  echo "Exp 31 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

   ###################### Exp 31 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt"

  echo "Exp 31 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done


   ###################### Exp 31 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt"

  echo "Exp 31 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

fi


if [ $EXP_NUMBER -eq 32 ]; then

  ###################### Exp 32 ########################################
  model_name="resnet50"

  ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt"

  echo "Exp 32 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

   ###################### Exp 32 ########################################
  model_name="resnet50_timm_pretrained"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt"

  echo "Exp 32 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done


   ###################### Exp 32 ########################################
  model_name="resnet50_at"

  ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt"

  echo "Exp 32 with $model_name"
  echo "ckpt_dir: $ckpt_dir"
  echo "saving results to: $ckpt_dir/eval_knn_results"

  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do

      echo $ckpt_path

      if [ $adv_eval == "true" ]; then
        python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name     \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results --eps $epsilon --steps $steps --eval_predict_batch_size $batch_size
      else
        python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name    \
        --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
      fi


  done

fi


#
#
#
#
#
#ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256"
#
#echo "ckpt_dir: $ckpt_dir"
#echo "saving results to: $ckpt_dir/eval_knn_results"
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do
#
#    echo $ckpt_path
#    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#
#done
#
#
#
#ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512/"
#
#echo "ckpt_dir: $ckpt_dir"
#echo "saving results to: $ckpt_dir/eval_knn_results"
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in $ckpt_dir/checkpoint_40000.pth; do
#
#    echo $ckpt_path
#    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#
#done
