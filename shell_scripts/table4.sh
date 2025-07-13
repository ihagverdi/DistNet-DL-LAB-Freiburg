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
        echo "Running: lognormal_nn.floc --scenario $scenario --fold $fold"
        python3 -m scripts.eval_lognormal_nn \
            --scenario "$scenario" \
            --fold "$fold" \
            --epochs 1000 \
            --batch_size 256 \
            --save "./paper_repr_table4"
    done
done