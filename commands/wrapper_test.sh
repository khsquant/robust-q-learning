gpu_id=$1 
wandb_project="wrapper_test_td3"
use_wandb=true
task="WalkerWalk"
dr_train_ratio=1.0
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true  task=$task seed=$seed use_wandb=$use_wandb 
done
task="HopperHop"
dr_train_ratio=1.0
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
done
task="CheetahRun"

for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
done
