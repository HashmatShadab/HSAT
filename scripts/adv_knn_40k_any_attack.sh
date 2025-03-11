#!/bin/bash


# Default values for arguments
EXP_NUM=${1:-18}
EPSILON=${2:-8}


# Run evaluation scripts
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 7 64 "ffgsmr_knn"

#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM false
#
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 10 32 "bimr_knn"
#
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 10 32 "mifgsmr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 10 32 "bimr_knn"
#
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 10 32 "pgd_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 10 32 "pgd_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 10 32 "mifgsmr_knn"


bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 20 32 "pgd_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 40 32 "pgd_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 60 32 "pgd_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 10 32 "pgd_knn"



#
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 20 64 "mifgsmr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 20 64 "mifgsmr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 4 20 64 "bimr_knn"
#bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true 8 20 64 "bimr_knn"


echo "All evaluation scripts executed successfully!"
