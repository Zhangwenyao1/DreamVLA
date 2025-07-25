finetune_from_pretrained_ckpt="pretrian_weight_path"
calvin_dataset_path="" # your data path
save_checkpoint_path="./checkpoints/"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing

node=1
node_num=8
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10211 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 2 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --finetune_type "calvin" \
    --wandb_project dreamvla \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --num_obs_token_per_image 9 \
    --run_name finetune_dreamvla_calvin_abc_d_flow_as_mask_depth \
    --save_checkpoint \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase "finetune" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --obs_pred \
    --depth_pred \
    --use_dit_head \
    --track_label_patch_size 8 \
    --load_track_labels \
    --loss_image \
    --loss_action \
    --loss_depth \
    --loss_sam_feat \
    --load_dino_features \
    --sam_feat_pred \
    --load_sam_features \
    --flow_as_mask \
    --attn_implementation "sdpa" \
    --reset_obs_token \
    --reset_action_decoder \
    --report_to_wandb \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \
