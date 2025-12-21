#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# real jinan data including 963125 trajectories
python -u -m train \
    --city jinan \
    --data_type real \
    --data_num 900_000 \
    --test_data_num 63_125 \
    --weight_quantization_scale 30 \
    --max_connection 9  \
    --block_size 60 \
    --batch_size 32 \
    --store True \
    \
    --n_embd 64 \
    --n_head 16 \
    --n_layer 8 \
    --dropout 0.1 \
    --use_condition True \
    --condition_observable False \
    --use_adj_table True \
    --adj_type bv1h \
    --use_timestep False \
    \
    --max_epochs 50  \
    --learning_rate 1e-2 \
    --lr_decay 0.01 \
    --observe_ratio 0.5 \
    --reset_observation False \
    # \
    # --resume_dir \
    # --pretrain_model_dir ./weights/best_model.ckpt \
    2>&1 | tee train_jinan.out