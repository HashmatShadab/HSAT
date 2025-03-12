#!/bin/bash

DATA_PATH=/path/to/dataset


source_model=${1:-"resnet50"}
source_exp_no=${2:-1}
source_ckpt_dir=${3:-"Results/Baseline/resnet50_exp1"}
epsilon=${4:-8}
steps=${5:-7}
load_source_from_ssl=${6:-"True"}



############### Target models from Experiment 1 ####################


target_model="resnet50"
target_exp_no=1
target_ckpt_dir="Results/Baseline/resnet50_exp1/checkpoint_40000.pth"


echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="resnet50_at"
target_exp_no=1
target_ckpt_dir="Results/Baseline/resnet50_at_exp1/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done


target_model="resnet50_timm_pretrained"
target_exp_no=1
target_ckpt_dir="Results/Baseline/resnet50_timm_pretrained_exp1/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done


target_model="wresnet50_normal"
target_exp_no=1
target_ckpt_dir="Results/Baseline/wresnet50_normal_exp1/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done



target_model="wresnet50_at"
target_exp_no=1
target_ckpt_dir="Results/Baseline/wresnet50_at_exp1/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done


target_model="resnet101_normal"
target_exp_no=1
target_ckpt_dir="Results/Baseline/resnet101_normal_exp1/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done



target_model="resnet101_at"
target_exp_no=1
target_ckpt_dir="Results/Baseline/resnet101_at_exp1/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done
############### Target models from Experiment 2 ####################


target_model="resnet50"
target_exp_no=2
target_ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"


echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="resnet50_at"
target_exp_no=2
target_ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"


echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="resnet50_timm_pretrained"
target_exp_no=2
target_ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done



target_model="wresnet50_normal"
target_exp_no=2
target_ckpt_dir="Results/Adv/wresnet50_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="wresnet50_at"
target_exp_no=2
target_ckpt_dir="Results/Adv/wresnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="resnet101_normal"
target_exp_no=2
target_ckpt_dir="Results/Adv/resnet101_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="resnet101_at"
target_exp_no=2
target_ckpt_dir="Results/Adv/resnet101_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp2/checkpoint_40000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done


############### Target models from Experiment 3 ####################

target_model="resnet50"
target_exp_no=3
target_ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp3/checkpoint_80000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done

target_model="resnet50_at"
target_exp_no=3
target_ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp3/checkpoint_80000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done


target_model="resnet50_timm_pretrained"
target_exp_no=3
target_ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp3/checkpoint_80000.pth"

echo "Evaluating Transferability of adversarial examples generated on ${source_model} source models trained using  (EXP ${source_exp_no})  with checkpoint path ${source_ckpt_dir}"
echo "on ${target_model} target models trained using (EXP ${target_exp_no}) with checkpoint path ${target_ckpt_dir}"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $source_ckpt_dir/checkpoint_40000.pth; do

    echo $ckpt_path

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone $source_model --source_exp_no $source_exp_no   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps $epsilon --steps $steps  --load_source_from_ssl $load_source_from_ssl

done


