DATE=`date '+%Y%m%d-%H%M%S'`
echo $DATE

CUDA_VISIBLE_DEVICES=0 \
python3 \
train.py \
--batchsize 6 \
--savepath "../model/" \
--datapath "/workspace/dataset/DIS5K/DIS-TR" \
2>&1 | tee log$DATE.log

