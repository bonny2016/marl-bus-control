@echo off

REM Loop through seed values from 1 to 10
FOR /L %%S IN (3, 1, 10) DO (
    echo Running with seed=%%S
    python main.py --model=DDPG_Distill --seed=%%S --share_scale=0 --n_students=4 --episode=100 --split_by_region=0
)
