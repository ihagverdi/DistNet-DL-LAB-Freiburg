#!/bin/bash

models=(
    "invgauss_nn.floc"
    "lognormal_nn.floc"
)

scenarios=(
  "clasp_factoring"
  "saps-CVVAR"
  "spear_qcp"
  "yalsat_qcp"
  "spear_swgcp"
  "yalsat_swgcp"
  "lpg-zeno"
)

for model in "${models[@]}"; do
    for scenario in "${scenarios[@]}"; do
        for fold in {0..9}; do
            echo "Running: --model $model --scenario $scenario --fold $fold"
            python3 -m scripts.eval_model \
                --model "$model" \
                --scenario "$scenario" \
                --fold "$fold" \
                --save "./paper_repr_table4"
        done
    done
done