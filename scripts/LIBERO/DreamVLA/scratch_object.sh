#!/bin/bash
export PATH="/root/miniconda3/bin:$PATH" 
__conda_setup="$('/root/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)" 
eval "$__conda_setup"
conda activate dreamvla
export WANDB_MODE=offline

### NEED TO CHANGE ###
save_checkpoint_path="path to save checkpoint"
root_dir="."
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth" # path to mae ckpt
libero_path="libero_object_converted" # path to converted data

node=1
node_num=8
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 4 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 40 \
    --seed 42 \
    --batch_size 16 \
    --precision fp32 \
    --learning_rate 2e-4 \
    --save_checkpoint \
    --finetune_type libero_finetune \
    --root_dir ${root_dir} \
    --wandb_project dreamvla \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --run_name scratch_libero_object \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase "finetune" \
    --obs_pred \
    --action_pred_steps 3 \
    --sequence_length 7 \
    --future_steps 3 \
    --window_size 10 \
    --loss_image \
    --loss_action \
    --reset_action_token \
    --reset_obs_token \
    --save_checkpoint_seq 1 \
    --start_save_checkpoint 25 \
    --gripper_width \
    --warmup_epochs 5 \
    --libero_path ${libero_path} \
    --report_to_wandb \
    --use_dit_head \
    --load_track_labels \
    --load_sam_features \
    --sam_feat_pred \
    --loss_sam_feat \
    --flow_as_mask \
    --attn_implementation "sdpa"
