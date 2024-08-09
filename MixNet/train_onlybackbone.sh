## Train for backbone only
# CUDA_VISIBLE_DEVICES=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --main_process_port=29501 train.py \
# --exp_name 240808_pretrain_backbone_b2 --lr 0.01 \
# --batch_size 2 --start_epoch 0 --num_workers 16 --input_size 768 --net FSNet_M --num_points 20 \
# --custom_data_root data/custom_datas --open_data_root data/open_datas \
# --select_open_data totaltext,ctw1500,MSRA-TD500 \
# --select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended \
# --onlybackbone True
# --select_custom_data hdec1,hdec2,joyworks,무환샘플,통관용,bnk,kor_extended \
# --onlybackbone True \
# --select_open_data totaltext,ctw1500,MSRA-TD500 \

CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 accelerate launch --main_process_port=29502 train.py \
--exp_name 240808_pretrain_backbone_b2 --lr 0.001 \
--batch_size 2 --start_epoch 6 --num_workers 16 --input_size 1024 --net FSNet_M --num_points 20 \
--custom_data_root data/custom_datas --open_data_root data/open_datas \
--select_open_data totaltext,ctw1500,MSRA-TD500,XFUND,SROIE2019 \
--select_custom_data bnk,hdec1,hdec2,joyworks,무환샘플,통관용,kor_extended \
--onlybackbone True \
--resume /mnt/hdd1/sgh/MixNet/MixNet/model/240808_pretrain_backbone_b2/MixNet_FSNet_M_5.pth