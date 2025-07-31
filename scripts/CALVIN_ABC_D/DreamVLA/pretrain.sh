#!/bin/bash
### need to change to your path ###
calvin_dataset_path="" # change to your data path
save_checkpoint_path="checkpoints/pretrain_DreamVLA_calvin_abc_d/"
vit_checkpoint_path="./checkpoints/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
node=1
node_num=2

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 8 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 2 \
    --precision fp32 \
    --learning_rate 1e-4 \
    --finetune_type "calvin" \
    --wandb_project dreamvla_calvin_pretrain \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --num_obs_token_per_image 9 \
    --run_name pretrain_dreamvla_calvin_abc_d \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase "pretrain" \
    --action_pred_steps 3 \
    --sequence_length 14 \
    --future_steps 3 \
    --window_size 17 \
    --obs_pred \
    --loss_image \
    --loss_action \
    --atten_goal 4 \
    --atten_goal_state \
    --atten_only_obs \
    --attn_robot_proprio_state \
    --except_lang \
    --attn_implementation "sdpa" \
    --save_checkpoint \
    --report_to_wandb \
