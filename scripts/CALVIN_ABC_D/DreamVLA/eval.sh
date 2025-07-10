#!/bin/bash

export GIT_PYTHON_REFRESH=quiet
calvin_dataset_path= # your path/to/CALVIN_dataset
calvin_conf_path="calvin/calvin_models/conf"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
save_checkpoint_path="checkpoints/"

node=1
node_num=8


resume_from_checkpoint= # e.g., checkpoints/finetune_dreamvla.pth

IFS='/' read -ra path_parts <<< "$resume_from_checkpoint"
run_name="${path_parts[-3]}/${path_parts[-2]}"
log_name="${path_parts[-1]}"
log_folder="eval_logs/$run_name"
mkdir -p "$log_folder"
log_file="eval_logs/$run_name/evaluate_$log_name.log"

echo
echo "=============================================="
echo "Evaluating checkpoint: $resume_from_checkpoint"
echo "  run_name = $run_name"
echo "  log_name = $log_name"
echo "  log_file = $log_file"
echo "=============================================="

torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10219 eval_calvin.py \
    --traj_cons \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --gradient_accumulation_steps 1 \
    --bf16_module "vision_encoder" \
    --vit_checkpoint_path ${vit_checkpoint_path} \
    --calvin_dataset ${calvin_dataset_path} \
    --calvin_conf_path ${calvin_conf_path} \
    --workers 16 \
    --lr_scheduler cosine \
    --save_every_iter 50000 \
    --num_epochs 20 \
    --seed 42 \
    --batch_size 64 \
    --precision fp32 \
    --weight_decay 1e-4 \
    --num_resampler_query 16 \
    --num_obs_token_per_image 9 \
    --run_name ${run_name} \
    --save_checkpoint_path ${save_checkpoint_path} \
    --transformer_layers 24 \
    --hidden_dim 1024 \
    --transformer_heads 16 \
    --phase "evaluate" \
    --finetune_type "calvin" \
    --action_pred_steps 3 \
    --sequence_length 10 \
    --future_steps 3 \
    --window_size 13 \
    --obs_pred \
    --depth_pred \
    --sam_feat_pred \
    --pred_num 1 \
    --use_dit_head \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    | tee "${log_file}"


