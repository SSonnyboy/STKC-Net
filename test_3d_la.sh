# # - - - - - - - -      Testing      - - - - - - - # 

# nvidia-smi

##############################################################

# - - - - - - - - - - - - - - - - - - - - - # 
#                   LA
# - - - - - - - - - - - - - - - - - - - - - # 

expname="LA_runs"
version="test"
numlb=4 # 4, 8, 16
gpuid=1
# /home/chenyu/SSMIS/data/LA/data/UPT6DX9IQY9JAZ7HJKA7/mri_norm2.h5
python3 ./code/test_performance_3d.py \
    --root_path /home/chenyu/SSMIS/data/LA \
    --res_path ./results/LA \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model vnet  \
    --model_ext vnet_res18vnet \
    --model_i model1
python3 ./code/test_performance_3d.py \
    --root_path /home/chenyu/SSMIS/data/LA \
    --res_path ./results/LA \
    --gpu ${gpuid} \
    --exp ${expname}/v${version} \
    --labeled_num ${numlb} \
    --model res18vnet  \
    --model_ext vnet_res18vnet \
    --model_i model2
