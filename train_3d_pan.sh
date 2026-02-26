# nvidia-smi
# # # - - - - - - - - - - - Pancrease - - - - - - - - - - - - - #

expname="Pancrease_runs"
log_dir=./logs/${expname}
mkdir -p ${log_dir}

gpuid=3
cr_weight=0.1    # 0.001 0.01 0.1 1 10
version=main

nohup python3 ./code/train_post_3d.py \
    --gpu_id=${gpuid} \
    --cfg config_3d_pan_aut.yml \
    --patch_size 96 96 96 \
    --exp ${expname}/v${version} \
    --cr_weight=${cr_weight} \
    >${log_dir}/log_v${version}.log 2>&1 &