#!/bin/bash

python dataset.py --data_dir ./data_temp --simulation_from 0 --simulation_num 500000 --total_trajectories 5 2>&1 | tee dataset.out