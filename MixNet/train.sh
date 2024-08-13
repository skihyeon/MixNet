
CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch --main_process_port=29509 train.py \
--exp_name 240812_b2e_b2_g1_1024_filtered --lr 1e-5 \
--batch_size 2 --start_epoch 4 --num_workers 16 --input_size 1024 --net FSNet_M --num_points 20 \
--custom_data_root data/custom_datas --open_data_root data/open_datas \
--select_open_data . \
--select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended \
--resume ./model/240812_b2e_b2_g1_1024_filtered/MixNet_FSNet_M_3.pth --save_freq 1


# --select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
# CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch --main_process_port=29502 train.py \
# --exp_name 240812_e2e_b1_g2_s2_2048 --lr 0.0001 \
# --batch_size 1 --start_epoch 0 --num_workers 16 --input_size 2048 --net FSNet_M --num_points 20 \
# --custom_data_root data/custom_datas --open_data_root data/open_datas \
# --select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
# --select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended \