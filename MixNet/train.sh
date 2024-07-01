accelerate launch train.py --exp_name noaug_all \
--lr 1e-3 --batch_size 16 --start_epoch 1 --num_workers 12 --load_memory True --input_size 1024
