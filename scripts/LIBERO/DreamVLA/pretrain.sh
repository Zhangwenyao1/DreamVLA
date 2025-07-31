#!/bin/bash
### NEED TO CHANGE ###
save_checkpoint_path="checkpoints/pretrain_DreamVLA_small_libero/"
root_dir='.'
vit_checkpoint_path="/inspire/hdd/project/robotsimulation/guchun-240107140023/Hanzhe/Seer/checkpoints/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
libero_path="libero_90_converted"
### NEED TO CHANGE ###
calvin_dataset_path="/inspire/hdd/global_user/guchun-240107140023/task_ABC_D/" # change to your data path
node=1
node_num=8
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 8 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 30 \
    --seed 42 \
    --batch_size 10 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --save_checkpoint \
    --finetune_type libero_pretrain \
    --root_dir ${root_dir} \
    --wandb_project seer \
    --weight_decay 1e-4 \
    --num_resampler_query 6 \
    --num_obs_token_per_image 9 \
    --run_name pretrain_Dreamvla-small_libero \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --phase "pretrain" \
    --obs_pred \
    --sequence_length 11 \
    --action_pred_steps 3 \
    --future_steps 3 \
    --atten_goal 4 \
    --window_size 11 \
    --loss_image \
    --loss_action \
    --gripper_width \
    --atten_only_obs \
    --atten_goal_state \
    --mask_l_obs_ratio 0.5 \
    --warmup_epochs 1 \
    --attn_robot_proprio_state \
    --attn_implementation "sdpa" \
    --libero_path ${libero_path} \
    --report_to_wandb \