#!/bin/bash

export GIT_PYTHON_REFRESH=quiet
calvin_dataset_path="/inspire/hdd/global_user/guchun-240107140023/task_ABC_D/"
calvin_conf_path="/inspire/hdd/global_user/guchun-240107140023/calvin/calvin_models/conf"
vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
save_checkpoint_path="/mnt/afs/wenyaozhang/dreamvla_ckpt/checkpoints/"

node=1
node_num=8

# 从 e15 到 e19 循环
for epoch in {9..18}; do
    # 1. 把 resume_from_checkpoint 动态改成 e${epoch}.pth
    resume_from_checkpoint="/inspire/ssd/project/robotsimulation/guchun-240107140023/Wenyao/checkpoints/scratch_dreamvla_calvin_abc_d_mlp_siglipdino/${epoch}.pth"

    # 2. 和你原来一模一样的拆分逻辑，生成 run_name / log_name / log_folder / log_file
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

    # 3. 原封不动地调用 torchrun 并把输出 tee 到对应的 log_file
    torchrun --nnodes=${node} --nproc_per_node=${node_num} --master_port=10253 eval_calvin.py \
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
        --use_dit_head \
        --use_dinosiglip \
        --obs_pred \
        --attn_implementation "sdpa" \
        --pred_num 1 \
        --resume_from_checkpoint ${resume_from_checkpoint} \
        | tee "${log_file}"

    echo "Finished evaluating e${epoch}."
done

echo
echo ">>> All done: epochs 15–19 have been evaluated. <<<"
