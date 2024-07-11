# python inference.py --exp_name noaug_all --checkepoch 15 --num_workers 4 --infer_path infer_test_datas/images --num_points 100  --test_size 1024 1024
# python eval.py --exp_name cwk --checkepoch 105 --num_workers 4 --eval_dataset my --num_points 100 --mid True


# exp_name="noaug_all"
# eval_dataset="All"

# for i in 15
# do
#     echo "" >> iou.log
#     echo "epoch: $i, exp_name: $exp_name data: $eval_dataset" >> iou.log
#     python eval.py --exp_name $exp_name --checkepoch $i --num_workers 4 --eval_dataset $eval_dataset --num_points 100 --gpu_num 0 --test_size 1024 1024
#     python cal_IoU.py --pred_root ./output/$exp_name --gt_root ./data/open_datas/gts >> iou.log
# done

# exp_name="only_kor_H_M_mid_extended_later"
# eval_dataset="All"

# for i in 235 240 245
# do
#     echo "" >> iou.log
#     echo "epoch: $i, exp_name: $exp_name data: $eval_dataset" >> iou.log
#     python eval.py --exp_name $exp_name --checkepoch $i --num_workers 4 --eval_dataset $eval_dataset --num_points 100 --mid True --gpu_num 0 --test_size 2048 2048
#     python cal_IoU.py --pred_root ./output/$exp_name --gt_root ./data/open_datas/gts >> iou.log
# done


# for i in 65 70 75 80 85 90 95 100 105 110
# do
#     echo "" >> iou.log
#     echo "epoch: $i, exp_name: cwk data: gghj_part" >> iou.log
#     python inference.py --exp_name cwk --checkepoch $i --num_workers 4 --infer_path infer_test_datas/gghj_part/image --num_points 100 --mid True
#     python cal_IoU.py --pred_root ./infer_test_datas/gghj_part/image/cwk_result/text/ --gt_root ./infer_test_datas/gghj_part/gt/ >> iou.log
# done

# exp_name="Test"
# eval_dataset="gghj"

# for i in 940
# do
#     echo "" >> iou.log
#     echo "epoch: $i, exp_name: $exp_name data: $eval_dataset" >> iou.log
#     python inference.py --exp_name $exp_name --checkepoch $i --num_workers 4 --infer_path infer_test_datas/gghj_part/image --num_points 100 --test_size 1024 1024 --net FSNet_hor --mid True
#     python cal_IoU.py --pred_root ./infer_test_datas/gghj_part/image/${exp_name}_result/text/ --gt_root ./infer_test_datas/gghj_part/gt/ >> iou.log
# done

exp_name="official_ext_kor"
eval_dataset="All"

for i in 940
do
    echo "" >> iou.log
    echo "epoch: $i, exp_name: $exp_name data: $eval_dataset" >> iou.log
    python eval.py --exp_name $exp_name --checkepoch $i --num_workers 4 --eval_dataset $eval_dataset --num_points 100 --mid True --gpu_num 0 --test_size 1024 1024 --net FSNet_hor
    python cal_IoU.py --pred_root ./output/$exp_name --gt_root ./data/open_datas/gts >> iou.log
done