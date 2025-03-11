#!/bin/bash


# Default values for arguments
EXP_NUM=${1:-1}
MODEL_NAME=${2:-"resnet50"}
EPSILON=${3:-8}
STEP_SIZE=${4:-10}
BATCH_SIZE=${5:-32}



# Run evaluation scripts

bash scripts/adv_knn.sh $EXP_NUM $MODEL_NAME false
bash scripts/adv_knn.sh $EXP_NUM $MODEL_NAME true $EPSILON $STEP_SIZE $BATCH_SIZE "bimr_knn"
bash scripts/adv_knn.sh $EXP_NUM $MODEL_NAME true $EPSILON $STEP_SIZE $BATCH_SIZE "pgd_knn"
bash scripts/adv_knn.sh $EXP_NUM $MODEL_NAME true $EPSILON $STEP_SIZE $BATCH_SIZE "mifgsmr_knn"


echo "All evaluation scripts executed successfully!"
