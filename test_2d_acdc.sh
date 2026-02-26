# nvidia-smi

# - - - - - - - -     Testing      - - - - - - - # 
expname="ACDC_runs"
version="ema"
numlb=7 # 3, 7, 14
gpuid=1

python3 ./code/test_performance_2d.py \
    --root_path /home/chenyu/SSMIS/data/ACDC \
    --res_path ./results/ACDC \
    --gpu_id=${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model unet  \
    --model_ext unet_unet \
    --model_i model1

python3 ./code/test_performance_2d.py \
    --root_path /home/chenyu/SSMIS/data/ACDC \
    --res_path ./results/ACDC \
    --gpu_id=${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model unet  \
    --model_ext unet_unet \
    --model_i model2