export PATH="/root/miniconda3/bin:$PATH" 
__conda_setup="$('/root/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)" 
eval "$__conda_setup"
conda activate dreamvla
export WANDB_MODE=offline

# pthlist=("32")
seeds=("42" "66" "88")
for seed in "${seeds[@]}"; do
    resume_from_checkpoint=$1
    vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
    this_resume_from_checkpoint=${resume_from_checkpoint}
    save_checkpoint_path="checkpoints/finetune_DreamVLA_small_libero/eval"
    ckpt_id=$(basename "$resume_from_checkpoint" .pth)
    dir_name=$(basename $(dirname "$resume_from_checkpoint"))
    LOG_DIR="./eval_500/${dir_name}"
    mkdir -p ${LOG_DIR}
    logfile="${LOG_DIR}/${ckpt_id}_seed${seed}.log"

    node=1
    node_num=8

    python -m torch.distributed.run  --nnodes=${node} --nproc_per_node=${node_num} --master_port=10133 eval_libero_500.py \
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path ${vit_checkpoint_path} \
        --calvin_dataset "" \
        --libero_path "LIBERO" \
        --workers 16 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed ${seed} \
        --batch_size 64 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --run_name test \
        --transformer_layers 24 \
        --phase "evaluate" \
        --finetune_type ${2:-"libero_10"} \
        --save_checkpoint_path ${save_checkpoint_path} \
        --action_pred_steps 3 \
        --future_steps 3 \
        --sequence_length 7 \
        --obs_pred \
        --gripper_width \
        --eval_libero_ensembling \
        --use_dit_head \
        --load_track_labels \
        --load_sam_features \
        --sam_feat_pred \
        --loss_sam_feat \
        --flow_as_mask \
        --resume_from_checkpoint ${this_resume_from_checkpoint} | tee ${logfile}
done