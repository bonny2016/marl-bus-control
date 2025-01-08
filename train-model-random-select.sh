#!/bin/bash

# Loop through seed values from 1 to 10
for seed in {2..10}
do
    echo "Running with seed=$seed"
    python main.py --model='TD3' --seed=$seed --share_scale=0
done