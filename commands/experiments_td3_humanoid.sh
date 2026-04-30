gpu_id=$1
wandb_project="td3-humanoid"
use_wandb=true
task="T1JoystickRoughTerrain"

for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
        wandb_project=$wandb_project asymmetric_critic=true \
        task=$task seed=$seed use_wandb=$use_wandb
done

for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
        wandb_project=$wandb_project asymmetric_critic=true \
        task=$task seed=$seed use_wandb=$use_wandb
done

for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
        wandb_project=$wandb_project asymmetric_critic=true \
        task=$task seed=$seed use_wandb=$use_wandb
done
