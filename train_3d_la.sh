# nvidia-smi

# - - - - - - - - - - - - LA - - - - - - - - - - - - #

expname="LA_runs"
log_dir=./logs/${expname}
mkdir -p ${log_dir}

gpuid=1
cr_weight=0.1    # 0.001 0.01 0.1 1 10
version=base

nohup python3 ./code/train_post_3d.py \
    --gpu_id=${gpuid} \
    --cfg config_3d_la_aut.yml \
    --exp ${expname}/v${version} \
    --cr_weight=${cr_weight} \
    >${log_dir}/log_v${version}.log 2>&1 &

# python3 ./code/train_post_3d.py \
#     --gpu_id=${gpuid} \
#     --cfg config_3d_la_aut.yml \
#     --exp ${expname}/v${version} \
#     --cr_weight=${cr_weight} \
