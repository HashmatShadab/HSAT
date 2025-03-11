#!/bin/bash


NUM_GPUS=$1
BATCH_SIZE=$2
exp_num=$3
model_name=$4
data_path=$5
random=${6:-25900}  # if random is not provided, use 25900 as default

# if exp_num == 1, run the firstscript


if [ $exp_num -eq 1 ] # Baseline
then
    echo "Running Baseline Experiment"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=False data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False \
    training.attack.name=none  training.attack.eps=8 training.attack.warmup_epochs=0 training.attack.loss_type=p_s_pt \
    out_dir=Results/Baseline/${model_name}_exp1 \
    wandb.exp_name=Baseline_backbone_${model_name}_exp1 wandb.use=True

fi

if [ $exp_num -eq 2 ]
then
    echo "Running HSAT Experiment"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2 wandb.use=True

fi




if [ $exp_num -eq 3 ]
then
    echo "Running HSAT Experiment with Clean Samples also in the training"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name \
    training.batch_size=$BATCH_SIZE training.only_adv=False training.num_epochs=80000 \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_exp3 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_exp3 wandb.use=True

fi

##############################################################################################################

# Experiment 4-8 : Are for testing the effect of embedding size on the adversarial training: Exp 2 is repeated with different embedding sizes
if [ $exp_num -eq 4 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=256 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp4_with_embedding256 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp4_with_embedding256 wandb.use=True

fi

if [ $exp_num -eq 5 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=128 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp5_with_embedding128 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp5_with_embedding128 wandb.use=True

fi

if [ $exp_num -eq 6 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=512 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp6_with_embedding512 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp6_with_embedding512 wandb.use=True

fi

if [ $exp_num -eq 7 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=768 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp7_with_embedding768 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp7_with_embedding768 wandb.use=True

fi

if [ $exp_num -eq 8 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.num_embedding_out=1024 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp8_with_embedding1024 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp8_with_embedding1024 wandb.use=True

fi


#######################################################################################################################
# Experiment 9-10 Testing the effect of multi-layer MLP on the adversarial training. 

if [ $exp_num -eq 9 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.proj_head=True model.num_embedding_out=2048 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp9 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp9 wandb.use=True

fi

if [ $exp_num -eq 10 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name model.mlp_hidden=[2048,2048] model.num_embedding_out=2048 \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp10 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_proj_head_exp10 wandb.use=True

fi


####################################################################################################################################



##############################################################################################################
# Experiment 11-12 : Exp 2 is repeated with different  attack loss : pt and s_pt
if [ $exp_num -eq 11 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp11_with_adv_loss_pt \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp11_with_adv_loss_pt wandb.use=True

fi

if [ $exp_num -eq 12 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp12_with_adv_loss_s_pt \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_exp12_with_adv_loss_s_pt wandb.use=True


fi





# Experiment 13: HAT-Patch Testing the effect of patch loss only on the adversarial training. Increase the batch size accordingly to keep the same effective batch size.
if [ $exp_num -eq 13 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    data.hidisc.num_slide_samples=1 data.hidisc.num_patch_samples=1 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp13 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp13 wandb.use=True

fi


# Experiment 14: HAT-Slide Testing the effect of patch-slide loss only on the adversarial training. Increase the batch size accordingly to keep the same effective batch size.
if [ $exp_num -eq 14 ]
then
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$random main.py \
    data.db_root=$data_pathh data.dynamic_aug=True data.dynamic_aug_version=v0 \
    data.hidisc.num_patch_samples=1 \
    model.backbone=$model_name  \
    training.batch_size=$BATCH_SIZE training.only_adv=True \
    training.attack.name=pgd  training.attack.eps=8 training.attack.warmup_epochs=5000 training.attack.loss_type=p_s_pt \
    out_dir=Results/Adv/${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp14 \
    wandb.exp_name=Adv_backbone_${model_name}_dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp14 wandb.use=True

fi




















