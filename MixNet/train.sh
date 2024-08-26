#!/bin/bash

export WANDB_API_KEY="0a7cca3a906f5c34a06fe63623461725e2278ef3"
export WANDB_ENTITY="hero981001"

# nohup env CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# accelerate launch --main_process_port=29500 train.py \
# --exp_name 240823_e2e_b2_LN_H_M_mid --lr 1e-5 \
# --batch_size 2 --start_epoch 0 --num_workers 12 --input_size 1024 --net FSNet_H_M --num_points 10 \
# --custom_data_root data/custom_datas --open_data_root data/open_datas \
# --select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
# --select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended,marketing \
# --save_freq 5 --mid True \
# --resume /mnt/hdd1/sgh/MixNet/MixNet/model/240823_e2e_b2_LN_H_M/MixNet_FSNet_H_M_15.pth \
# > training_output.log 2>&1 &


nohup env CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
accelerate launch --main_process_port=29510 train.py \
--exp_name 240823_e2e_b2_LN_H_M --lr 1e-5 \
--batch_size 2 --start_epoch 16 --num_workers 12 --input_size 1024 --net FSNet_H_M --num_points 10 \
--custom_data_root data/custom_datas --open_data_root data/open_datas \
--select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
--select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended,marketing \
--save_freq 5 \
--resume /mnt/hdd1/sgh/MixNet/MixNet/model/240823_e2e_b2_LN_H_M/MixNet_FSNet_H_M_15.pth \
> training_output2.log 2>&1 &

# --resume /mnt/hdd1/sgh/MixNet/MixNet/model/240822_ln_test_b2_e2e/MixNet_FSNet_H_M_5.pth \
# --resume /mnt/hdd1/sgh/MixNet/MixNet/model/240820_b2_g1_768_from_only_bnk/MixNet_FSNet_H_M_250.pth
# 

# --select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
# CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch --main_process_port=29502 train.py \
# --exp_name 240812_e2e_b1_g2_s2_2048 --lr 0.0001 \
# --batch_size 1 --start_epoch 0 --num_workers 16 --input_size 2048 --net FSNet_M --num_points 20 \
# --custom_data_root data/custom_datas --open_data_root data/open_datas \
# --select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
# --select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended \


# CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch --main_process_port=29510 train.py \
# --exp_name Test --lr 1e-5 \
# --batch_size 2 --start_epoch 16 --num_workers 12 --input_size 640 --net FSNet_M --num_points 10 \
# --custom_data_root data/custom_datas --open_data_root data/open_datas \
# --select_open_data . \
# --select_custom_data hdec1 \
# --save_freq 5 \