#!/bin/bash

scenarios=(
  "clasp_factoring"
  "saps-CVVAR"
  "spear_qcp"
  "yalsat_qcp"
  "spear_swgcp"
  "yalsat_swgcp"
  "lpg-zeno"
)

for scenario in "${scenarios[@]}"; do
    for fold in {0..9}; do
        echo "Running: --model lognormal_distfit.floc --scenario $scenario --fold $fold"
        python3 -m scripts.eval_model \
            --model lognormal_distfit.floc \
            --scenario "$scenario" \
            --fold "$fold" \
            --save "./paper_repr_table3"
    done
done