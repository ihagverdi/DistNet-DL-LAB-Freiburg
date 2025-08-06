#!/usr/bin/env bash
source "C:/ProgramData/anaconda3/etc/profile.d/conda.sh"
conda activate tabpfn

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

SCENARIO=(
  #clasp_factoring
  #saps-CVVAR
  lpg-zeno
  #spear_qcp
  #spear_swgcp
  yalsat_qcp
  #yalsat_swgcp
)
SAVE_DIR=./TEST_100

for SCEN in "${SCENARIO[@]}"
do
  # Loop over the folds (0-9)
  for FOLD in {0..9}
  do
    echo "Running: Scenario=$SCEN, Fold=$FOLD"
    python scripts/eval_lognormal_nn.py --scenario $SCEN --fold $FOLD --save $SAVE_DIR
    echo "Completed: Scenario=$SCEN, Fold=$FOLD"
  done
done
echo All experiments completed!