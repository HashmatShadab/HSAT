#!/bin/bash


# Default values for arguments
EXP_NUM=${1:-1}
EPSILON=${2:-8}
STEP_SIZE=${3:-10}
BATCH_SIZE=${4:-32}


# Run evaluation scripts

bash scripts/adv_knn.sh $EXP_NUM false
bash scripts/adv_knn.sh $EXP_NUM true $EPSILON $STEP_SIZE $BATCH_SIZE "bimr_knn"
bash scripts/adv_knn.sh $EXP_NUM true $EPSILON $STEP_SIZE $BATCH_SIZE "pgd_knn"
bash scripts/adv_knn.sh $EXP_NUM true $EPSILON $STEP_SIZE $BATCH_SIZE "mifgsmr_knn"


echo "All evaluation scripts executed successfully!"
