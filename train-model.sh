#!/bin/bash

# Loop through seed values from 1 to 10
for seed in {1..10}
do
    echo "Running with seed=$seed"
    python main.py --model='TD3_Distill' --seed=$seed --share_scale=0 --n_students=6 --episode=100
done
