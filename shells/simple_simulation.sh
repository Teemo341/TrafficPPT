#!/bin/bash

python -m dataloader.dataset --city boston --simulation_from 0 --simulation_num 50 --random_sample_num 5 2>&1 | tee ./results/simple_simulation.out
# python -m dataloader.dataset --city jinan --simulation_from 0 --simulation_num 50 --random_sample_num 5 --max_connection 9 2>&1 | tee ./results/simple_simulation.out