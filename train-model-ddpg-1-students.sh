#!/bin/bash

# Loop through seed values from 1 to 10
for seed in {1..10}
do
    echo "Running with seed=$seed"
    python main.py --model='DDPG_Distill' --seed=$seed --share_scale=1 --n_students=1 --episode=200
done
