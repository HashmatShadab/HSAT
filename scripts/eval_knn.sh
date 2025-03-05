#!/bin/bash

DATA_PATH=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH

model_name=$1
ckpt_dir=$2

echo "ckpt_dir: $ckpt_dir"


echo "saving results to: $ckpt_dir/eval_knn_results"


# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $ckpt_dir/*.pth; do
    echo $ckpt_path
    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results

done

#

