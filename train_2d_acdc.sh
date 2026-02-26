# nvidia-smi

# - - - - - - - - - - - - - - - - - - - - - - - - #

expname="ACDC_runs"
log_dir=./logs/${expname}
mkdir -p ${log_dir}


gpuid=1
cr_weight=0.1     # 0.001 0.01 0.1 1 10

version=ema

nohup python3 ./code/train_post_2d.py \
    --gpu_id=${gpuid} \
    --cfg config_2d_aut.yml \
    --exp=${expname}/v${version} \
    --cr_weight=${cr_weight} \
    >${log_dir}/log_v${version}.log 2>&1 &

