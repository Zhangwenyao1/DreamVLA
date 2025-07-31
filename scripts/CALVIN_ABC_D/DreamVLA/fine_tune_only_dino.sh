finetune_from_pretrained_ckpt="/inspire/hdd/project/robotsimulation/guchun-240107140023/Wenyao/seer_pretrain.pth"
calvin_dataset_path="/inspire/hdd/global_user/guchun-240107140023/task_ABC_D/" # your data path
save_checkpoint_path="./checkpoints/"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth" # downloaded from https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing
node=1
node_num=8
torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10225 train.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --workers 32 \
    --lr_scheduler cosine \
    --save_every_iter 100000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 8 \
    --precision fp32 \
    --learning_rate 1e-3 \
    --finetune_type "calvin" \
    --wandb_project dreamvla \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --num_obs_token_per_image 9 \
    --run_name finetune_dreamvla_calvin_abc_d_dit_b_only_dino \
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
    --use_dit_head \
    --loss_dino_feat \
    --dino_feat_pred \
    --load_dino_features \
    --dino_features_path "/inspire/hdd/project/robotsimulation/guchun-240107140023/Wenyao/DreamVLA/dinov2/calvin_dino/task_ABC_D" \
    --loss_action \
    --attn_implementation "sdpa" \
    --reset_obs_token \
    --report_to_wandb \
    --finetune_from_pretrained_ckpt ${finetune_from_pretrained_ckpt} \

    # --load_sam_feat \
    # --sam_features_path "/inspire/hdd/project/robotsimulation/guchun-240107140023/Wenyao/DreamVLA/segment-anything/calvin_sam/task_ABC_D" \
        # --loss_sam_feat \
    # --sam_feat_pred \
    # --load_sam_features \