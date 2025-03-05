#!/bin/bash

DATA_PATH=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data
#DATA_PATH="F:/Code/datasets/hidisc_data_small"





############### Target models from Experiment 18 ####################

#target_model="resnet50"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/resnet50_exp18/checkpoint_40000.pth"

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done



#target_model="resnet50_at"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/resnet50_at_exp18/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
##
### loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done


#target_model="resnet50_timm_pretrained"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/resnet50_timm_pretrained_exp18/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#
#target_model="wresnet50_normal"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
##
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done



#target_model="wresnet50_at"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/wresnet50_at_exp18/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
### loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done


#target_model="resnet101_normal"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/resnet101_normal_exp18/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done



#target_model="resnet101_at"
#target_exp_no=18
#target_ckpt_dir="Results/Baseline/resnet101_at_exp18/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
############### Target models from Experiment 19 ####################


#target_model="resnet50"
#target_exp_no=19
#target_ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
### loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#target_model="resnet50_at"
#target_exp_no=19
#target_ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
### loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#target_model="resnet50_timm_pretrained"
#target_exp_no=19
#target_ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
### loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#
#
#target_model="wresnet50_normal"
#target_exp_no=19
#target_ckpt_dir="Results/Adv/wresnet50_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#target_model="wresnet50_at"
#target_exp_no=19
#target_ckpt_dir="Results/Adv/wresnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

target_model="resnet101_normal"
target_exp_no=19
target_ckpt_dir="Results/Adv/resnet101_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint.pth"

## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do

    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"

    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True

done

#target_model="resnet101_at"
#target_exp_no=19
#target_ckpt_dir="Results/Adv/resnet101_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done

# loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done


############### Target models from Experiment 20 ####################
#
#target_model="resnet50"
#target_exp_no=20
#target_ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp20/checkpoint_80000.pth"
#
## loop over all the checkpoints in the directory ending with .pth
##for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
##
##    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
##
##    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
##    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
##    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
##
##done
##
## loop over all the checkpoints in the directory ending with .pth
##for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
##
##    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
##
##    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
##    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
##    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
##
##done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#target_model="resnet50_at"
#target_exp_no=20
#target_ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20/checkpoint_80000.pth"
#
## loop over all the checkpoints in the directory ending with .pth
##for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
##
##    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
##
##    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
##    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
##    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
##
##done
##
### loop over all the checkpoints in the directory ending with .pth
##for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
##
##    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
##
##    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
##    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
##    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
##
##done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#
#target_model="resnet50_timm_pretrained"
#target_exp_no=20
#target_ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp20/checkpoint_80000.pth"
#
### loop over all the checkpoints in the directory ending with .pth
##for ckpt_path in Results/Baseline/wresnet50_at_exp18/*_checkpoint.pth; do
##
##    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
##
##    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
##    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
##    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
##
##done
##
### loop over all the checkpoints in the directory ending with .pth
##for ckpt_path in Results/Baseline/wresnet50_at_exp18/checkpoint.pth; do
##
##    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
##
##    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
##    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
##    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
##
##done
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth; do
#
#    echo "Source ckpt $ckpt_path,  Target ckpt $target_ckpt_dir"
#
#    python adv_eval_knn_transf.py --data_db_root $DATA_PATH  --source_model_backbone wresnet50_normal --source_exp_no 18   \
#    --source_ckpt_path $ckpt_path --target_model_backbone $target_model --target_exp_no $target_exp_no \
#    --target_ckpt_path $target_ckpt_dir  --save_results_path  transf_eval_knn_results --eps 8 --steps 10  --load_source_from_ssl True
#
#done
#
#
