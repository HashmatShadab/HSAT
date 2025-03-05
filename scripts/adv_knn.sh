#!/bin/bash


# Default values for arguments
EXP_NUM=${1:-18}
EPSILON=${2:-8}


# Run evaluation scripts
bash scripts/eval_adv_knn.sh $EXP_NUM true $EPSILON 7
bash scripts/eval_adv_knn.sh $EXP_NUM true $EPSILON 10
bash scripts/eval_adv_knn.sh $EXP_NUM true $EPSILON 20


bash scripts/eval_adv_knn.sh $EXP_NUM false $EPSILON 7

echo "All evaluation scripts executed successfully!"
