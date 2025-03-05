#!/bin/bash

exp_num=$1
BATCH_SIZE=${2:-128}
pgd_steps=${3:-7}
eps=${4:-8}
train_attack=${5:-pgd} # pgd, none
finetuning_type=${6:-linear} # linear, full
class_balance=${7:-false} # true, false

# get the parent name of ssl_ckpt

DATA_PATH=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data


###################### Exp 18 ########################################
if [ $exp_num -eq 1 ]; then

  model_name="resnet50_timm_pretrained"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Baseline/resnet50_timm_pretrained_exp18"

  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"


  
  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do

       echo "Starting finetuning with checkpoint: $ckpt_path"
       ckpt_name=$(basename "$ckpt_path" .pth)
       wandb_exp_name="FT_Baseline_${model_name}_exp18_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
       out_dir="FT_Results/ssl_Baseline_${model_name}_exp18_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi

###################### Exp 18 ########################################
if [ $exp_num -eq 2 ]; then

  model_name="resnet50"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Baseline/resnet50_exp18"

  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"



  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Baseline_${model_name}_exp18_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Baseline_${model_name}_exp18_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi


###################### Exp 18 ########################################
if [ $exp_num -eq 3 ]; then

  model_name="resnet50_at"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Baseline/resnet50_at_exp18"

  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"


  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Baseline_${model_name}_exp18_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Baseline_${model_name}_exp18_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"


       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi



###################### Exp 19 ########################################
if [ $exp_num -eq 4 ]; then

  model_name="resnet50"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"

  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"




  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi

if [ $exp_num -eq 5 ]; then

  model_name="resnet50_timm_pretrained"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"

  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"


  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi

if [ $exp_num -eq 6 ]; then

  model_name="resnet50_at"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"

  echo "Exp 19 with $model_name"
  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"




  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi


###################### Exp 20 ########################################
if [ $exp_num -eq 7 ]; then

  model_name="resnet50"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp20"

  echo "Exp 20 with $model_name"
  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"




  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi

if [ $exp_num -eq 8 ]; then

  model_name="resnet50_timm_pretrained"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp20"

  echo "Exp 20 with $model_name"

  echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"



  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do


      echo "Starting finetuning with checkpoint: $ckpt_path"
      ckpt_name=$(basename "$ckpt_path" .pth)
      wandb_exp_name="FT_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
      out_dir="FT_Results/ssl_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"


       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi

if [ $exp_num -eq 9 ]; then

  model_name="resnet50_at"
  ckpt_dir="/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/hidisc_pytorch/Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20"
  echo "Exp 20 with $model_name"

    echo "Finetuning experiment with $model_name and finetuning type: $finetuning_type. Attack: $train_attack. Class Balance: $class_balance. Eps: $eps. PGD Steps: $pgd_steps"




  # loop over all the checkpoints in the directory ending with .pth
  for ckpt_path in $ckpt_dir/checkpoint.pth; do

       echo "Starting finetuning with checkpoint: $ckpt_path"
       # Extract checkpoint filename without extension
       ckpt_name=$(basename "$ckpt_path" .pth)
       wandb_exp_name="FT_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20_train_${finetuning_type}_adv_${train_attack}_mlp_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"
       out_dir="FT_Results/ssl_Adv_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20_train_${finetuning_type}_adv_${train_attack}_class_balanced_${class_balance}_eps_${eps}_pgd_steps_${pgd_steps}_${ckpt_name}"

       python finetuning.py distributed.single_gpu=True data.db_root=$DATA_PATH data.balance_study_per_class=$class_balance \
       model.backbone=$model_name model.start_from_ssl_ckpt=$ckpt_path model.finetuning=$finetuning_type model.num_classes=7 \
       training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=$train_attack training.attack_eps=$eps training.attack_steps=$pgd_steps \
       out_dir=$out_dir \
       wandb.exp_name=$wandb_exp_name wandb.use=True

  done


fi


#
#if [ $exp_num -eq 1 ]
#then
#
#    # Standard Linear Finetuning Using Baseline SSL Checkpoint
#    wandb_exp_name="FT_backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
#    out_dir="FT/pretrain_${parentname}/backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
#
#    echo "Saving Results in  $out_dir"
#
#    python finetuning.py distributed.single_gpu=True data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.balance_study_per_class=false \
#    model.backbone=$model_name model.start_from_ssl_ckpt=$ssl_ckpt model.finetuning=linear model.num_classes=7 \
#    training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=pgd training.attack_eps=8.0 training.attack_steps=7 \
#    out_dir=$out_dir \
#    wandb.exp_name=$wandb_exp_name wandb.use=True
#fi
#
#if [ $exp_num -eq 2 ]
#then
#    # Standard Adv. Full Finetuning From Scratch
#    wandb_exp_name="FT_backbone_${model_name}_ft_scratch_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
#    out_dir="FT/from_scratch/backbone_${model_name}_ft_scratch_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
#
#    echo "Saving Results in  $out_dir"
#
#    python finetuning.py distributed.single_gpu=True data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.balance_study_per_class=false \
#    model.backbone=$model_name model.start_from_ssl_ckpt=false model.finetuning=full model.num_classes=7 \
#    training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=pgd training.attack_eps=8.0 training.attack_steps=7 \
#    out_dir=$out_dir \
#    wandb.exp_name=$wandb_exp_name wandb.use=True
#
#fi
#
#if [ $exp_num -eq 3 ]
#then
#    # Standard Full Finetuning From Scratch
#    wandb_exp_name="FT_backbone_${model_name}_ft_scratch_class_balanced_false_exp_${exp_num}"
#    out_dir="FT/from_scratch/backbone_${model_name}_ft_scratch_class_balanced_exp_${exp_num}"
#
#    echo "Saving Results in  $out_dir"
#
#    python finetuning.py distributed.single_gpu=True data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.balance_study_per_class=false \
#    model.backbone=$model_name model.start_from_ssl_ckpt=false model.finetuning=full model.num_classes=7 \
#    training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=false training.attack_eps=8.0 training.attack_steps=7 \
#    out_dir=$out_dir \
#    wandb.exp_name=$wandb_exp_name wandb.use=True
#
#fi
#
#if [ $exp_num -eq 4 ]
#then
#    # Standard Full Finetuning From Baseline SSL Checkpoint
#    wandb_exp_name="FT_backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
#    out_dir="FT/pretrain_${parentname}/backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
#
#    echo "Saving Results in  $out_dir"
#
#    python finetuning.py distributed.single_gpu=True data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.balance_study_per_class=false \
#    model.backbone=$model_name model.start_from_ssl_ckpt=$ssl_ckpt model.finetuning=linear model.num_classes=7 \
#    training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=pgd training.attack_eps=8.0 training.attack_steps=7 \
#    out_dir=$out_dir \
#    wandb.exp_name=$wandb_exp_name wandb.use=True
#
#fi
