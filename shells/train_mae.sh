#!/bin/bash

python -u train_mae.py --block_size 50 --n_layer 8 --batch_size 256 --max_epochs 10  --learning_rate 1e-2 --save_freq 5 --eval_freq 10  --load_dir_id 0 --load_dir ./checkpoints_none/ --device cuda:0 2>&1 | tee train_none.out