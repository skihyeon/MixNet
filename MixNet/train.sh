NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch train.py --exp_name t --lr 1e-5 \
--batch_size 1 --start_epoch 0 --num_workers 16 --input_size 480 \
--custom_data_root data/custom_datas --open_data_root data/open_datas \
--select_custom_data kor_extended,bnk \
--select_open_data totaltext,MSRA-TD500,ctw1500,FUNSD,XFUND,SROIE2019 \


# --resume ./model/240710_only_kor/MixNet_FSNet_H_M_180.pth \