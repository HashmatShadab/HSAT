#!/bin/bash


# Default values for arguments
EXP_NUM=${1:-18}
EPSILON=${2:-8}


# Run evaluation scripts
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 7 64 "ffgsmr_knn"

bash scripts/eval_adv_knn_40k_any_attack.sh 18 false
bash scripts/eval_adv_knn_40k_any_attack.sh 181 false
bash scripts/eval_adv_knn_40k_any_attack.sh 19 false
bash scripts/eval_adv_knn_40k_any_attack.sh 191 false
bash scripts/eval_adv_knn_40k_any_attack.sh 20 false
bash scripts/eval_adv_knn_40k_any_attack.sh 201 false
bash scripts/eval_adv_knn_40k_any_attack.sh 202 false
bash scripts/eval_adv_knn_40k_any_attack.sh 33 false
bash scripts/eval_adv_knn_40k_any_attack.sh 34 false
bash scripts/eval_adv_knn_40k_any_attack.sh 31 false
bash scripts/eval_adv_knn_40k_any_attack.sh 32 false
bash scripts/eval_adv_knn_40k_any_attack.sh 28 false
bash scripts/eval_adv_knn_40k_any_attack.sh 244 false
bash scripts/eval_adv_knn_40k_any_attack.sh 25 false
bash scripts/eval_adv_knn_40k_any_attack.sh 26 false
bash scripts/eval_adv_knn_40k_any_attack.sh 27 false



#
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 20 64 "mifgsmr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 20 64 "mifgsmr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 20 64 "bimr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 20 64 "bimr_knn"


echo "All evaluation scripts executed successfully!"
