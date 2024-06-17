exp_name="only_kor_H_M_mid_extended_later"
echo "\n \n start $exp_name" >> cal_IoU_res.txt

for i in 190 195 196 200 240 245 250

do 
python eval.py --exp_name $exp_name --mid True --checkepoch $i --mid True --eval_dataset my --num_points 100 --num_workers 4
echo "epoch_"$i >> cal_IoU_res.txt
python cal_IoU.py $exp_name >> cal_IoU_res.txt
done