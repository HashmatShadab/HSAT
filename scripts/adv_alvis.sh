#!/bin/bash


NUM_GPUS=$1
BATCH_SIZE=$2
exp_num=$3
model_name=$4
random=${5:-25900}  # if random is not provided, use 25900 as default

# if exp_num == 1, run the firstscript


if [ $exp_num -eq 1 ]
then
  echo "Running Adv Training with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_True wandb.use=True
fi

if [ $exp_num -eq 2 ]
then
    echo "Running Adv Training  with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=False"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=False model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_False\
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_False wandb.use=True
fi

if [ $exp_num -eq 3 ]
then
    echo "Running Adv Training  with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd_2  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_2_eps_8_dynaug_True \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_2_eps_8_dynamic_aug_True wandb.use=True
fi

if [ $exp_num -eq 4 ]
then
    echo "Running Adv Training  with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=False"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=False model.backbone=$model_name \
    training.attack.name=pgd_2  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_2_eps_8_dynaug_False\
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_2_eps_8_dynamic_aug_False wandb.use=True
fi

if [ $exp_num -eq 5 ]
then
  echo "Running Adv Training with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True v1"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_v1_model_train_before_loss \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_True_v1_model_train_before_loss wandb.use=True data.dynamic_aug_version=v1
fi

if [ $exp_num -eq 6 ]
then
  echo "Running Adv Training with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True v1"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd_2  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_2_eps_8_dynaug_True_v1_model_train_before_loss \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_2_eps_8_dynamic_aug_True_v1_model_train_before_loss wandb.use=True data.dynamic_aug_version=v1
fi

if [ $exp_num -eq 7 ]
then
  echo "Running Adv Training with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_sanity_check \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_True_sanity_check wandb.use=True
fi

if [ $exp_num -eq 8 ]
then
  echo "Running Adv Training with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_sanity_check_only_adv_loss \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_True_sanity_check_only_adv_loss wandb.use=True training.only_adv=True
fi

if [ $exp_num -eq 9 ]
then
  echo "Running Adv Training with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=False"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=False  model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_False_sanity_check_only_adv_loss \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_False_sanity_check_only_adv_loss wandb.use=True training.only_adv=True
fi

if [ $exp_num -eq 10 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_exp10 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_exp10 wandb.use=True

fi

if [ $exp_num -eq 100 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=pgd_2  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_2_eps_8_dynaug_True_exp10 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_2_eps_8_dynaug_True_exp10 wandb.use=True

fi

if [ $exp_num -eq 11 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=2000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_warmup_eps_2000_exp11 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_warmup_eps_2000_exp11 wandb.use=True
fi

if [ $exp_num -eq 12 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_warmup_eps_5000_exp12 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_warmup_eps_5000_exp12 wandb.use=True

fi

if [ $exp_num -eq 13 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=10000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_warmup_eps_10000_exp13 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_warmup_eps_10000_exp13 wandb.use=True

fi

if [ $exp_num -eq 14 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_exp14 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_exp14 wandb.use=True

fi

if [ $exp_num -eq 144 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd_2  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_2_eps_8_dynaug_True_only_adv_exp14 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_2_eps_8_dynaug_True_only_adv_exp14 wandb.use=True

fi

if [ $exp_num -eq 15 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=2000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_warmup_eps_2000_exp15 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_warmup_eps_2000_exp15 wandb.use=True

fi

if [ $exp_num -eq 16 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_warmup_eps_5000_exp16 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_warmup_eps_5000_exp16 wandb.use=True

fi

if [ $exp_num -eq 17 ]
then
     # Dynamic Augmentation True, Attack PGD, Epsilon 8, Dynamic Augmentation Version v0, only_adv False, attack loss_type p_s_pt, attack_warmup epsilon false
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=10000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_warmup_eps_10000_exp17 \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynaug_True_only_adv_warmup_eps_10000_exp17 wandb.use=True

fi

if [ $exp_num -eq 18 ] # Baseline
then
    echo "Running Experiment 18"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=False data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=none  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Baseline/${model_name}_exp18 \
    wandb.exp_name=Baseline_backbone_${model_name}_exp18 wandb.use=True

fi

if [ $exp_num -eq 19 ]
then
    echo "Running Experiment 19"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19 wandb.use=True

fi




if [ $exp_num -eq 20 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False training.num_epochs=80000 \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp20 wandb.use=True

fi




#################################################Single GPU #########################################################
if [ $exp_num -eq 21 ]
then
    python main.py distributed.single_gpu=True \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=False data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=none  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Baseline/${model_name}_single_gpu_exp21 \
    wandb.exp_name=Baseline_backbone_${model_name}_single_gpu_exp21 wandb.use=True

fi

if [ $exp_num -eq 22 ]
then
    python main.py distributed.single_gpu=True \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_single_gpu_exp22 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_single_gpu_exp22 wandb.use=True

fi


if [ $exp_num -eq 23 ]
then
    python main.py distributed.single_gpu=True \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_single_gpu_exp23 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_single_gpu_exp23 wandb.use=True

fi

##############################################################################################################

# Experiment 24-27 : Are for testing the effect of embedding size on the adversarial training: Exp 19 is repeated with different embedding sizes
if [ $exp_num -eq 24 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=256 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256 wandb.use=True

fi

if [ $exp_num -eq 244 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=128 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp244_with_embedding128 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp244_with_embedding128 wandb.use=True

fi

if [ $exp_num -eq 25 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=512 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512 wandb.use=True

fi

if [ $exp_num -eq 26 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=768 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp26_with_embedding768 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp26_with_embedding768 wandb.use=True

fi

if [ $exp_num -eq 27 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=1024 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp27_with_embedding1024 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp27_with_embedding1024 wandb.use=True

fi

#######################################################################################################################
# Experiment 28: Testing the effect of patch loss only on the adversarial training, Increase the batch size accordingly by 4 times. Exp 19 is repeated with patch loss only
if [ $exp_num -eq 28 ] # ran this one, but kept the batch size 16
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    data.hidisc.num_slide_samples=1 data.hidisc.num_patch_samples=1 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp28 wandb.use=True

fi

if [ $exp_num -eq 281 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    data.hidisc.num_slide_samples=1 data.hidisc.num_patch_samples=1 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp281 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_patch_loss_exp281 wandb.use=True

fi

#######################################################################################################################
# Experiment 29-30 Testing the effect of multi-layer MLP on the adversarial training. Exp 29 is based on NeuriPS method and Exp 30 is based on the baseline code.
# Exp 19 is repeated with multi-layer MLP

if [ $exp_num -eq 29 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.proj_head=True model.num_embedding_out=2048 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp29 wandb.use=True

fi

if [ $exp_num -eq 30 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp30 wandb.use=True

fi


####################################################################################################################################



##############################################################################################################
# Experiment 31-34 : Exp 19 is repeated with different  attack loss : pt and s_pt
if [ $exp_num -eq 31 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp31_with_adv_loss_pt wandb.use=True

fi

if [ $exp_num -eq 32 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp32_with_adv_loss_s_pt wandb.use=True


fi





# Experiment 33: HAT-Patch Testing the effect of patch loss only on the adversarial training, Increase the batch size accordingly by 4 times.
if [ $exp_num -eq 33 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    data.hidisc.num_slide_samples=1 data.hidisc.num_patch_samples=1 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp33 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp33 wandb.use=True

fi


# Experiment 34: HAT-Patch Testing the effect of patch loss only on the adversarial training, Increase the batch size accordingly by 4 times.
if [ $exp_num -eq 34 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data data.dynamic_aug=True data.dynamic_aug_version=v0 \
    data.hidisc.num_patch_samples=1 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp34 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp34 wandb.use=True

fi




















