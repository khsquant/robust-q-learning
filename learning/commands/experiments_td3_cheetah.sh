gpu_id=$1 
wandb_project="td3-drwrapper-test-cheetah5"
use_wandb=true
task="CheetahRun"
dr_train_ratio=1.0
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb 
done
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
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="td3" \
                            wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
done

wandb_project="sac-drwrapper-cheetah-test5"
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=false task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do    
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=true  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=true  task=$task seed=$seed use_wandb=$use_wandb 
done
for seed in 1 2 3
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py policy="sac" \
                            wandb_project=$wandb_project asymmetric_critic=true task=$task seed=$seed use_wandb=$use_wandb 
done