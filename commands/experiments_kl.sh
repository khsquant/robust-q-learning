gpu_id=$1 
wandb_project="wdtd3-kl6-cheetah"
use_wandb=true
task="CheetahRun"
for seed in 1 2 3
do 
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb
done 
for seed in 1 2 3
do 
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb
done 
for delta in 0.001 0.05 
do
    for n_nominals in 5 
        do
            for seed in 1 2 3
            do
                CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdtd3" n_nominals=$n_nominals delta=$delta single_lambda=false \
                    lambda_update_steps=10 distance_type="kl" wandb_project=$wandb_project asymmetric_critic=false init_lmbda=10 task=$task \
                    seed=$seed use_wandb=$use_wandb
            done
        done
done

for delta in 0.001 0.05 
do
    for n_nominals in 5 
        do
            for seed in 1 2 3
            do
                CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="wdtd3" n_nominals=$n_nominals delta=$delta single_lambda=false \
                    lambda_update_steps=10 distance_type="kl" wandb_project=$wandb_project asymmetric_critic=true init_lmbda=10 task=$task \
                    seed=$seed use_wandb=$use_wandb
            done
        done
done