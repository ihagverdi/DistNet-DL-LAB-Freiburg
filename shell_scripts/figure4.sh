#!/bin/bash

scenarios=(
  "lpg-zeno"
  "yalsat_qcp"
)

# steps=(1 2 4 8 16 64)
steps=(3 5 6 7)
seeds=(100 200 300 400 500 600 700 800 900 1000)

for step in "${steps[@]}"; do
  for scenario in "${scenarios[@]}"; do
      for fold in {0..9}; do
        for seed in "${seeds[@]}"; do
            echo "Running: lognormal_nn.floc --scenario $scenario --fold $fold -seed $seed --num_train_samples $step"
            python3 -m scripts.eval_lognormal_nn \
                --scenario "$scenario" \
                --num_train_samples "$step" \
                --fold "$fold" \
                --seed "$seed" \
                --epochs 1000 \
                --batch_size 256 \
                --save "./figure4_${step}"
        done
      done
  done
done