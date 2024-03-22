DATE=`date '+%Y%m%d-%H%M%S'`
echo $DATE

CUDA_VISIBLE_DEVICES=0 \
python3 \
test.py \
--batchsize 1 \
--model "model-56.pth" \
--datapath "/workspace/dataset/DIS5K/DIS-VD" \
2>&1 | tee log$DATE.log

