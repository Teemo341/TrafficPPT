#!/bin/bash

python -u train_mae.py --city jinan --simulation_num 800000 --test_simulation_num 163125 --max_connection 9  --block_size 60 --n_layer 8 --batch_size 256 --max_epochs 50  --learning_rate 1e-2 --save_freq 5 --eval_freq 10  --load_dir_id 0 --load_dir ./checkpoints_jinan/ --device cuda:3 2>&1 | tee train_jinan.out