gpu_id=$1 
wandb_project="wdtd3-panelty-cheetah6"
use_wandb=true
task="CheetahRun"
for seed in 1 2 3
do 
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb
done 
for seed in 1 2 3
do 
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb
done 
for seed in 1 2 3
do 
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb
done 
for seed in 1 2 3
do
    for omega_distance_threshold in 5.0
    do
        CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="m2td3" wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb omega_distance_threshold=$omega_distance_threshold
    done
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="m2td3" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb omega_distance_threshold=5.0
done
for seed in 1 2 3
do
    for n_nominals in 5
        do
            for init_lmbda in 0.001 0.05 0.1 0.5 1
            do
                CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdtd3" n_nominals=$n_nominals delta=0\
                    lambda_update_steps=1 distance_type="wass" wandb_project=$wandb_project asymmetric_critic=false init_lmbda=$init_lmbda task=$task \
                    seed=$seed use_wandb=$use_wandb single_lambda=true lmbda_lr=0
            done
        done
done
for seed in 1 2 3
do 
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
done 
# for seed in 1 2 3
# do
#     for omega_distance_threshold in 2.0 5.0 10.0 15.0 20.0
#     do
#         CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="m2td3" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb omega_distance_threshold=$omega_distance_threshold
#     done
# done
for delta in 0.001 0.05 
do
    for n_nominals in 5 10 15
        do
            for seed in 1 2 3
            do
                CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdtd3" n_nominals=$n_nominals delta=0 \
                    lambda_update_steps=1 distance_type="wass" wandb_project=$wandb_project asymmetric_critic=true init_lmbda=$init_lmbda task=$task \
                    seed=$seed use_wandb=$use_wandb single_lambda=true lmbda_lr=0
            done
        done
done

