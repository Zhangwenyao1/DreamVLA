import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
import clip
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from models.dreamvla_model import DreamVLA
from utils.train_utils import get_checkpoint, train_one_epoch_calvin, get_ckpt_name
from utils.arguments_utils import get_parser
from utils.data_utils import get_calvin_dataset, get_calvin_val_dataset, get_droid_dataset, get_libero_pretrain_dataset, get_libero_finetune_dataset, get_real_finetune_dataset, get_oxe_dataset
from utils.distributed_utils import init_distributed_device, world_info_from_env  


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def count_parameters(model):
    total_params = 0
    trainable_params = 0
    trainable_names = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
    return total_params, trainable_params, trainable_names

@record
def main(args):
    os.environ["WANDB_DIR"] = f"{os.path.abspath(args.save_checkpoint_path)}"
    if args.save_checkpoints_to_wandb and args.save_checkpoint and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    random_seed(args.seed)
    ptbs = args.world_size * args.batch_size * args.gradient_accumulation_steps
    print("training batch size:", ptbs)
    args.run_name = args.run_name.replace("dreamvla", f"dreamvla_{ptbs}_{args.transformer_layers}layers_{args.transformer_heads}heads_hd{args.hidden_dim}")
    print("run_name:", args.run_name)
    model = DreamVLA(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        
        pred_num = args.pred_num,
        depth_pred = args.depth_pred,
        use_dpt_head= args.use_dpt_head,
        use_depth_query = args.use_depth_query,
        trajectory_pred = args.trajectory_pred,
        use_trajectory_query = args.use_trajectory_query,
        track_label_patch_size=args.track_label_patch_size,

        dino_feat_pred=args.dino_feat_pred,
        sam_feat_pred=args.sam_feat_pred,

        use_dinosiglip = args.use_dinosiglip,
        use_dit_head = args.use_dit_head,
        no_pred_gripper_traj= args.no_pred_gripper_traj,
        no_unshuffle=args.no_unshuffle,
        use_gpt2_pretrained = args.use_gpt2_pretrained,
        share_query=args.share_query,
        attn_implementation= args.attn_implementation
    )
    if args.finetune_type == "calvin":
        calvin_dataset = get_calvin_dataset(args, model.image_processor, clip, epoch=0, except_lang=args.except_lang)
    elif args.finetune_type == "droid":
        calvin_dataset = get_droid_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_pretrain":
        calvin_dataset = get_libero_pretrain_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "libero_finetune":
        calvin_dataset = get_libero_finetune_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "real":
        calvin_dataset = get_real_finetune_dataset(args, model.image_processor, clip, epoch=0)
    elif args.finetune_type == "oxe":
        calvin_dataset = get_oxe_dataset(args, model.image_processor, clip, epoch=0)
    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        print("wandb_project :", args.wandb_project)
        print("wandb_entity :", args.wandb_entity)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if 'vision_encoder' in args.bf16_module:
            if args.use_dinosiglip:
                model.dino_featurizer.bfloat16()
                model.siglip_featurizer.bfloat16()
                model.dino_featurizer.requires_grad_(False)
                model.siglip_featurizer.requires_grad_(False)
            else:
                model.vision_encoder.bfloat16()
                model.vision_encoder.requires_grad_(False)
        if "image_primary_projector" in args.bf16_module:
            model.image_primary_projector.bfloat16()
            model.cls_token_primary_projector.bfloat16()
        if "image_wrist_projector" in args.bf16_module:
            model.image_wrist_projector.bfloat16()
            model.cls_token_wrist_projector.bfloat16()
        if "perceiver_resampler" in args.bf16_module:
            model.perceiver_resampler.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module and args.obs_pred:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()
        if "depth_decoder" in args.bf16_module and args.depth_pred:
            model.depth_decoder.bfloat16()
            model.depth_decoder_obs_pred_projector.bfloat16()
        if "action_decoder" in args.bf16_module:
            model.action_decoder.bfloat16()
            model.action_decoder_obs_pred_projector.bfloat16()
        if "dino_decoder" in args.bf16_module and args.dino_feat_pred:
            model.dino_decoder.bfloat16()
            model.dino_decoder_obs_pred_projector.bfloat16()
        if "sam_decoder" in args.bf16_module and args.sam_feat_pred:
            model.sam_decoder.bfloat16()
            model.sam_decoder_obs_pred_projector.bfloat16()
        if "text_encoder" in args.bf16_module:
            model.clip_model.bfloat16()
    model.clip_model.requires_grad_(False)
    
    total_params, trainable_params, trainable_names = count_parameters(model)
    if args.rank == 0:
        print("total_params: {} M".format(total_params/1024/1024))
        print("trainable_params: {} M".format(trainable_params/1024/1024))
        print("trainable names: ", trainable_names)
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
    optimizer = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)  # TODO make sure the parameters which need to be optimized are passing
    total_training_steps = calvin_dataset.dataloader.num_batches * args.num_epochs
    args.warmup_steps = calvin_dataset.dataloader.num_batches * args.warmup_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")
    if args.lr_scheduler == "linear":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == "cosine":
        if args.gradient_accumulation_steps > 1:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps + 1,
                num_training_steps=total_training_steps // args.gradient_accumulation_steps + 1,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_training_steps,
            )
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    resume_from_epoch = 0
    if args.finetune_from_pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Starting finetuning from pretrained checkpoint {args.finetune_from_pretrained_ckpt}")    
        checkpoint = torch.load(args.finetune_from_pretrained_ckpt, map_location="cpu")
        image_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if "image_decoder" in k]
        projector_keys = [k for k in checkpoint["model_state_dict"].keys() if "projector" in k]
        image_decoder_obs_pred_projector_keys = [k for k in checkpoint["model_state_dict"].keys() if "image_decoder_obs_pred_projector" in k]
        action_decoder_keys = [k for k in checkpoint["model_state_dict"].keys() if "action_decoder" in k]
        resampler_keys = [k for k in checkpoint["model_state_dict"].keys() if "perceiver_resampler" in k]
        if args.reset_action_token:
            del checkpoint["model_state_dict"]["module.action_pred_token"] 
        if args.reset_obs_token:
            del checkpoint["model_state_dict"]["module.obs_tokens"] 
        if args.reset_mask_token:
            del checkpoint["model_state_dict"]["module.mask_token"] 
        if args.reset_image_decoder:
            for k in image_decoder_keys:
                if k in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][k]
        if args.reset_action_decoder:
            for k in action_decoder_keys:
                if k in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][k]
        if args.share_query:
            for k in image_decoder_obs_pred_projector_keys:
                if k in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][k]
        if args.reset_resampler:
            for k in resampler_keys:
                if k in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][k]
            del checkpoint["model_state_dict"]["module.image_primary_projector.weight"]
            del checkpoint["model_state_dict"]["module.cls_token_primary_projector.weight"]
            del checkpoint["model_state_dict"]["module.image_wrist_projector.weight"]
            del checkpoint["model_state_dict"]["module.cls_token_wrist_projector.weight"]
        if checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"].shape != ddp_model.module.transformer_backbone_position_embedding.shape:
            checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"] = checkpoint["model_state_dict"]["module.transformer_backbone_position_embedding"][:, :args.sequence_length, :, :]
        print("loading pretrained weights :", checkpoint["model_state_dict"].keys())
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    ckpt_dir = os.path.join(f"{args.save_checkpoint_path}", args.run_name)
    if args.rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    ddp_model.train()
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)

        calvin_loader = calvin_dataset.dataloader
        train_one_epoch_calvin(
            args=args,
            model=ddp_model,
            epoch=epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calvin_loader=calvin_loader,
            device_id=device_id,
            wandb=wandb,
        )
        if args.rank == 0 and args.save_checkpoint and epoch % args.save_checkpoint_seq == 0 and epoch > args.start_save_checkpoint:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    