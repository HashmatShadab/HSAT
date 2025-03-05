#!/bin/bash

DATA_PATH=/lustre/mlnvme/data/swasim_hpc-datasets/naseer/Projects/data
#DATA_PATH="F:/Code/datasets/hidisc_data_small"



# echo all above values
echo "epsilon: 1"
echo "steps: 10"
echo "batch_size: 64"
echo "data_fraction: 0.2"

############### Target models from Experiment 18 ####################

source_model="resnet50"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet50_exp18/checkpoint_40000.pth"

echo "Running evaluation scripts for experiment 18 source models"

python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  



source_model="resnet50_at"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet50_at_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




source_model="resnet50_timm_pretrained"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet50_timm_pretrained_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2




source_model="wresnet50_normal"
source_exp_no=18
source_ckpt_dir="Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




source_model="wresnet50_at"
source_exp_no=18
source_ckpt_dir="Results/Baseline/wresnet50_at_exp18/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




source_model="resnet101_normal"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet101_normal_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




source_model="resnet101_at"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet101_at_exp18/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  



############### Target models from Experiment 19 ####################


source_model="resnet50"
source_exp_no=19
source_ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"

echo "Running evaluation scripts for experiment 19 source models"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




source_model="resnet50_at"
source_exp_no=19
source_ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  



source_model="resnet50_timm_pretrained"
source_exp_no=19
source_ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




source_model="wresnet50_normal"
source_exp_no=19
source_ckpt_dir="Results/Adv/wresnet50_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  


source_model="wresnet50_at"
source_exp_no=19
source_ckpt_dir="Results/Adv/wresnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  



source_model="resnet101_normal"
source_exp_no=19
source_ckpt_dir="Results/Adv/resnet101_normal_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  


source_model="resnet101_at"
source_exp_no=19
source_ckpt_dir="Results/Adv/resnet101_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  


############### Target models from Experiment 20 ####################

source_model="resnet50"
source_exp_no=20
source_ckpt_dir="Results/Adv/resnet50_dynamicaug_true_epsilon_warmup_5000_exp20/checkpoint_40000.pth"

echo "Running evaluation scripts for experiment 20 source models"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  

source_model="resnet50_at"
source_exp_no=20
source_ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_exp20/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  


source_model="resnet50_timm_pretrained"
source_exp_no=20
source_ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_exp20/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  




############### Target models from ImageNet Models ####################

source_model="resnet50"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet50_exp18/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False



source_model="resnet50_at"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet50_at_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False




source_model="resnet50_timm_pretrained"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet50_timm_pretrained_exp18/checkpoint_40000.pth"


  python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
  --source_exp_no $source_exp_no \
  --source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False




source_model="wresnet50_normal"
source_exp_no=18
source_ckpt_dir="Results/Baseline/wresnet50_normal_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False




source_model="wresnet50_at"
source_exp_no=18
source_ckpt_dir="Results/Baseline/wresnet50_at_exp18/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False




source_model="resnet101_normal"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet101_normal_exp18/checkpoint_40000.pth"


python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False




source_model="resnet101_at"
source_exp_no=18
source_ckpt_dir="Results/Baseline/resnet101_at_exp18/checkpoint_40000.pth"



python tsne.py --data_db_root $DATA_PATH  --source_model_backbone $source_model    \
--source_exp_no $source_exp_no \
--source_ckpt_path $source_ckpt_dir  --save_results_path  Results/tsne_eval_knn_results --eps 1 --steps 10 --eval_predict_batch_size 64 --data_fraction 0.2  --load_source_from_ssl False
