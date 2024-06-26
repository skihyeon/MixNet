# python inference.py --exp_name ConcatDatas --checkepoch 60 --num_workers 4 --infer_path infer_test_datas/images --num_points 100 
# python eval.py --exp_name cwk --checkepoch 105 --num_workers 4 --eval_dataset my --num_points 100 --mid True


exp_name="cwk"
eval_dataset="All"

for i in 65 70 75 80 85 90 95 100 105
do
    echo "" >> iou.log
    echo "epoch: $i, exp_name: $exp_name data: $eval_dataset" >> iou.log
    python eval.py --exp_name $exp_name --checkepoch $i --num_workers 4 --eval_dataset $eval_dataset --num_points 100 --mid True
    python cal_IoU.py --pred_root ./output/$exp_name --gt_root ./data/kor_extended/Test/gt/ >> iou.log
done

exp_name="ConcatDatas"
eval_dataset="All"

for i in 60
do
    echo "" >> iou.log
    echo "epoch: $i, exp_name: $exp_name data: $eval_dataset" >> iou.log
    python eval.py --exp_name $exp_name --checkepoch $i --num_workers 4 --eval_dataset $eval_dataset --num_points 100
    python cal_IoU.py --pred_root ./output/$exp_name --gt_root ./data/open_datas/gts/ >> iou.log
done


# for i in 65 70 75 80 85 90 95 100 105 110
# do
#     echo "" >> iou.log
#     echo "epoch: $i, exp_name: cwk data: gghj_part" >> iou.log
#     python inference.py --exp_name cwk --checkepoch $i --num_workers 4 --infer_path infer_test_datas/gghj_part/image --num_points 100 --mid True
#     python cal_IoU.py --pred_root ./infer_test_datas/gghj_part/image/cwk_result/text/ --gt_root ./infer_test_datas/gghj_part/gt/ >> iou.log
# done

# for i in 200 240 245
# do
#     echo "" >> iou.log
#     echo "epoch: $i, exp_name: only_kor_H_M_mid_extended_later data: gghj_part" >> iou.log
#     python inference.py --exp_name only_kor_H_M_mid_extended_later --checkepoch $i --num_workers 4 --infer_path infer_test_datas/gghj_part/image --num_points 100 --mid True
#     python cal_IoU.py --pred_root ./infer_test_datas/gghj_part/image/only_kor_H_M_mid_extended_later_result/text/ --gt_root ./infer_test_datas/gghj_part/gt/ >> iou.log
# done
