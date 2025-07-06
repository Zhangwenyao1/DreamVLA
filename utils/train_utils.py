import time
from contextlib import suppress
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist
from utils.sigloss import SiLogLoss
from .visualize_utils import visualize_optical_flow
from PIL import Image
def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()
    
    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # images
        images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # depths
        depths_primary = batch_calvin[6].to(device_id, dtype=cast_dtype, non_blocking=True)
        depths_wrist = batch_calvin[7].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # dino & sam
        dino_feat_primary = batch_calvin[8].to(device_id, dtype=cast_dtype, non_blocking=True) if batch_calvin[8] is not None else None
        dino_feat_wrist = batch_calvin[9].to(device_id, dtype=cast_dtype, non_blocking=True) if batch_calvin[9] is not None else None
        sam_feat_primary = batch_calvin[10].to(device_id, dtype=cast_dtype, non_blocking=True) if batch_calvin[10] is not None else None
        sam_feat_wrist = batch_calvin[11].to(device_id, dtype=cast_dtype, non_blocking=True) if batch_calvin[11] is not None else None

        # track
        track_infos = batch_calvin[12]
        for k in list(track_infos.keys()):
            track_infos[k+'_l'] = track_infos[k].clone()
        for k, v in track_infos.items():
            if not k.endswith('_l'):
                track_infos[k] = track_infos[k].to(device_id, dtype=cast_dtype, non_blocking=True)
                track_infos[k] = track_infos[k][:, :args.sequence_length, :]
        
        # bs, seq_len
        bs, seq_len = images_primary.shape[:2]
        # text tokens
        text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
        
        # states
        states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2

        
        # actions
        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # label. [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        input_image_primary = images_primary[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]

        # label action
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 

        with autocast():  # image_primary, image_wrist, state, language_instruction
            arm_pred_action, gripper_pred_action, image_pred, arm_pred_state, gripper_pred_state, loss_arm_action, depth_pred, traj_pred, dino_feat_pred, sam_feat_pred = model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=actions[:, :args.sequence_length, :],
                track_infos=track_infos,
                action_label = label_actions[:, :args.sequence_length-args.atten_goal].detach()
            )
        # loss_action
        if args.loss_action and args.action_pred_steps and not args.use_dit_head:
            loss_arm_action = torch.nn.functional.smooth_l1_loss(
                            arm_pred_action[:, :args.sequence_length-args.atten_goal], 
                            label_actions[:, :args.sequence_length-args.atten_goal, :, :6].detach())
            loss_gripper_action = torch.nn.functional.binary_cross_entropy(
                            gripper_pred_action[:, :args.sequence_length-args.atten_goal], 
                            label_actions[:, :args.sequence_length-args.atten_goal, :, 6:].detach())
        elif not args.use_dit_head:
            loss_arm_action = torch.tensor([0.0]).to(device_id)
            loss_gripper_action = torch.tensor([0.0]).to(device_id)
        elif args.use_dit_head:
            loss_arm_action = arm_pred_action
            loss_gripper_action = torch.tensor([0.0]).to(device_id)
        # loss_image 
        if args.loss_image and args.obs_pred:
            # label_image_primary = images_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_primary = images_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :].flatten(0,1)
            # label_image_wrist = images_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_wrist = images_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :].flatten(0,1)
            label_image_primary = patchify(label_image_primary, patch_size=args.patch_size)
            label_image_wrist = patchify(label_image_wrist, patch_size=args.patch_size)

            label_image_primary = normalize_patchfied_image(label_image_primary)
            label_image_wrist = normalize_patchfied_image(label_image_wrist)
            
            label_image_primary = label_image_primary.view(bs, args.sequence_length-args.atten_goal+args.pred_num-1, *label_image_primary.shape[1:])
            label_image_primary = label_image_primary.unfold(1, args.pred_num, 1).permute(0, 1, 4, 2, 3).flatten(0, 1)
            
            
            label_image_wrist = label_image_wrist.view(bs, args.sequence_length-args.atten_goal+args.pred_num-1, *label_image_wrist.shape[1:])
            label_image_wrist = label_image_wrist.unfold(1, args.pred_num, 1).permute(0, 1, 4, 2, 3).flatten(0, 1)
            
            # image_pred = image_pred.reshape(-1, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
            # image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
            # image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
            
            image_pred = image_pred.reshape(bs, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
            image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
            image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4], image_pred.shape[5])
            
            pred_image_example_primary = unpatchify(image_pred[:, 0, :, :, :])[0][0].permute(1, 2, 0).detach().cpu().float().numpy()
            pred_image_example_primary = (pred_image_example_primary - pred_image_example_primary.min()) / (pred_image_example_primary.max() - pred_image_example_primary.min())
            # pred_image_example_primary = (pred_image_example_primary * 255).astype(np.uint8)
            
            label_image_example_primary = unpatchify(label_image_primary)[0][0].permute(1, 2, 0).detach().cpu().float().numpy()
            label_image_example_primary = (label_image_example_primary - label_image_example_primary.min()) / (label_image_example_primary.max() - label_image_example_primary.min())
            # pred_image_example_primary = (label_image_example_primary * 255).astype(np.uint8)
            
            
            pred_image_example_secondary = unpatchify(image_pred[:, 1, :, :, :])[0][0].permute(1, 2, 0).detach().cpu().float().numpy()
            pred_image_example_secondary = (pred_image_example_secondary - pred_image_example_secondary.min()) / (pred_image_example_secondary.max() - pred_image_example_secondary.min())
            # pred_image_example_secondary = (pred_image_example_secondary * 255).astype(np.uint8)
            
            label_image_wrist_example = unpatchify(label_image_wrist)[0][0].permute(1, 2, 0).detach().cpu().float().numpy()
            label_image_wrist_example = (label_image_wrist_example - label_image_wrist_example.min()) / (label_image_wrist_example.max() - label_image_wrist_example.min())
            # label_image_wrist_example = (label_image_wrist_example * 255).astype(np.uint8)
            
            
            '''
            # predict optical flow and image
            
            # label_tracks_primary = track_infos['tracks_l'][:, 0:args.sequence_length-args.atten_goal+args.pred_num-1, :].cuda()
            # [b, p, _, c] = label_tracks_primary.shape
            # h = w = int(math.sqrt(label_tracks_primary.shape[-2]))
            
            # new_array = np.zeros((14, 14, 2))
            # label_tracks_primary = rearrange(label_tracks_primary, 'b p (h w) c -> b p h w c', h=h, w=w)
            # masks = torch.zeros((b, p, 14, 14), dtype=torch.bool)
            # for b_idx in range(b):
            #     for p_idx in range(p):
            #         new_array = torch.zeros((14, 14, 2))
            #         for k in range(2):  # 遍历每个通道
            #             for i in range(14):
            #                 for j in range(14):
            #                     new_array[i, j, k] = label_tracks_primary[b_idx, p_idx, 2*i:2*i+2, 2*j:2*j+2, k].mean()
            #         np_norm = torch.norm(new_array, dim=2)
            #         masks[b_idx, p_idx] = np_norm > 1
            #         temp_mask = masks[b_idx, p_idx].clone()
            #         for i in range(14):
            #             for j in range(14):
            #                 if masks[b_idx, p_idx][i, j]:
            #                     if i > 0:
            #                         temp_mask[i-1, j] = True
            #                     if i < 13:
            #                         temp_mask[i+1, j] = True
            #                     if j > 0:
            #                         temp_mask[i, j-1] = True
            #                     if j < 13:
            #                         temp_mask[i, j+1] = True
            #         masks[b_idx, p_idx] = temp_mask
                
            # masks = masks.unsqueeze(1)
            # mask_reshape = (masks.flatten(0,1).reshape(b*p, -1, 1)).unsqueeze(1)

            # label_tracks_wrist = track_infos['tracks_gripper_l'][:, 0:args.sequence_length-args.atten_goal+args.pred_num-1, :].cuda()
            # [b, p, _, c] = label_tracks_wrist.shape
            # h = w = int(math.sqrt(label_tracks_wrist.shape[-2]))
            
            # new_array = np.zeros((14, 14, 2))
            # label_tracks_wrist = rearrange(label_tracks_wrist, 'b p (h w) c -> b p h w c', h=h, w=w)
            # wrist_masks = torch.zeros((b, p, 14, 14), dtype=torch.bool)
            # for b_idx in range(b):
            #     for p_idx in range(p):
            #         new_array = torch.zeros((14, 14, 2))
            #         for k in range(2):
            #             for i in range(14):
            #                 for j in range(14):
            #                     new_array[i, j, k] = label_tracks_wrist[b_idx, p_idx, 2*i:2*i+2, 2*j:2*j+2, k].mean()
            #         np_norm = torch.norm(new_array, dim=2)
            #         wrist_masks[b_idx, p_idx] = np_norm > 1
                
            # wrist_masks = wrist_masks.unsqueeze(1)
            # wrist_masks_reshape = (wrist_masks.flatten(0,1).reshape(b*p, -1, 1)).unsqueeze(1)
            '''
            
            
            if args.flow_as_mask and 'tracks_l' in track_infos.keys():
                tracks_primary = track_infos['tracks_l'][
                    :, : args.sequence_length - args.atten_goal + args.pred_num - 1, :
                ] 
                B, P, HW, C = tracks_primary.shape
                H = W = int(HW ** 0.5)

                # 1. reshape + permute -> (B*P, 2, H, W)
                tp = tracks_primary.reshape(B * P, H, W, C).permute(0, 3, 1, 2)  # (B*P, 2, H, W)

                pooled = F.avg_pool2d(tp, kernel_size=2, stride=2)

                norm = torch.norm(pooled, dim=1)  #xw

                threshold = 1.0
                mask = (norm > threshold).unsqueeze(1).float()

                dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

                # 6. flatten -> (B*P, 1, L) with L=(H//2)*(W//2)
                h2, w2 = pooled.shape[2], pooled.shape[3]
                mask_reshape = dilated.reshape(B * P, 1, h2 * w2, 1) 

                # ————————————— gripper track —————————————

                tracks_wrist = track_infos['tracks_gripper_l'][
                    :, : args.sequence_length - args.atten_goal + args.pred_num - 1, :
                ]
                # reshape + permute
                tw = tracks_wrist.reshape(B * P, H, W, C).permute(0, 3, 1, 2)  # (B*P, 2, H, W)

                # 2×2 pool
                pooled_w = F.avg_pool2d(tw, kernel_size=2, stride=2)
                norm_w   = torch.norm(pooled_w, dim=1)
                mask_w   = (norm_w > threshold).unsqueeze(1).float()

                # dilated_w = F.max_pool2d(mask_w, kernel_size=3, stride=1, padding=1)

                # flatten -> (B*P, 1, L)
                wrist_masks_reshape = mask_w.reshape(B * P, 1, h2 * w2, 1)
                
                
                
                
                # mask_img = label_image_primary.detach().cpu() * mask_reshape
                # label_image_example_primary = unpatchify(mask_img)[0][0].permute(1, 2, 0).detach().cpu().numpy()
                # mask_img_np = (label_image_example_primary - label_image_example_primary.min()) / (label_image_example_primary.max() - label_image_example_primary.min()) * 255
                # mask_img_np = mask_img_np.astype(np.uint8)

                # image = Image.fromarray(mask_img_np)
                # image.save("./mask_image_2.png")
                loss_image = 0.5 * (torch.nn.functional.mse_loss(
                                image_pred[:, 0, :, :, :]*mask_reshape.to(image_pred.device), 
                                (label_image_primary*(mask_reshape.to(image_pred.device))).detach()) + 
                                torch.nn.functional.mse_loss(
                                image_pred[:, 1, :, :, :]*wrist_masks_reshape.to(image_pred.device), 
                                (label_image_wrist*wrist_masks_reshape.to(image_pred.device)).detach()))
            else:
                loss_image = 0.5 * (torch.nn.functional.mse_loss(
                    image_pred[:, 0, :, :], 
                    label_image_primary.detach()) + 
                    torch.nn.functional.mse_loss(
                    image_pred[:, 1, :, :], 
                    label_image_wrist.detach()))
        else:
            loss_image = torch.tensor([0.0]).to(device_id)
            
        if args.loss_depth and args.depth_pred:
            label_depths_primary = depths_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :]
            label_depths_wrist = depths_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :]
            depth_loss_head = SiLogLoss()
            depth_pred = depth_pred.reshape(bs, args.sequence_length, depth_pred.shape[1], depth_pred.shape[2], depth_pred.shape[3], depth_pred.shape[4])
            depth_pred = depth_pred[:, :args.sequence_length-args.atten_goal]
            depth_pred = depth_pred.reshape(-1, depth_pred.shape[2], depth_pred.shape[3], depth_pred.shape[4], depth_pred.shape[5])
            depth_x_pred, depth_gripper_pred  = torch.split(depth_pred, 1, dim=1)
            # if nedd，we can use squeeze
            depth_x_pred = depth_x_pred.squeeze(1)   # [B*S, P, C, 768]
            depth_gripper_pred = depth_gripper_pred.squeeze(1) # [B*S, P, 196, 768]    
            
            
            
            if args.use_dpt_head:
                depth_x_pred = depth_x_pred.unsqueeze(2)
                depth_gripper_pred = depth_gripper_pred.unsqueeze(2)
                label_depths_primary = label_depths_primary.unfold(1, args.pred_num, 1).permute(0, 1, 5, 2, 3, 4).flatten(0, 1)
                label_depths_wrist = label_depths_wrist.unfold(1, args.pred_num, 1).permute(0, 1, 5, 2, 3, 4).flatten(0, 1)
            else:
                # label_depths_primary = label_depths_primary.view(bs, args.sequence_length-args.atten_goal+args.pred_num-1, *label_depths_primary.shape[1:])
                label_depths_primary = label_depths_primary.unfold(1, args.pred_num, 1).permute(0, 1, 5, 2, 3, 4).flatten(0, 1)
                
                # label_depths_wrist = label_depths_wrist.view(bs, args.sequence_length-args.atten_goal+args.pred_num-1, *label_depths_wrist.shape[1:])
                label_depths_wrist = label_depths_wrist.unfold(1, args.pred_num, 1).permute(0, 1, 5, 2, 3, 4).flatten(0, 1)
                depth_x_pred = unpatchify(depth_x_pred)
                depth_gripper_pred = unpatchify(depth_gripper_pred)
                
                
            loss_pred_depth_x = depth_loss_head(depth_x_pred, label_depths_primary)
            loss_pred_depth_gripper = depth_loss_head(depth_gripper_pred, label_depths_wrist)
            
            # loss_pred_depth = 0.5 * (torch.nn.functional.mse_loss(
            #         depth_x_pred, 
            #         label_depths_primary.detach()) + 
            #         torch.nn.functional.mse_loss(
            #         depth_gripper_pred, 
            #         label_depths_wrist.detach()))
            
            loss_pred_depth = 0.5*(loss_pred_depth_x + loss_pred_depth_gripper)    

            pred_depth_example_primary = depth_x_pred[0][0].permute(1, 2, 0).detach().cpu().float().numpy()
            pred_depth_example_primary = (pred_depth_example_primary - pred_depth_example_primary.min()) / (pred_depth_example_primary.max() - pred_depth_example_primary.min())
            # pred_image_example_primary = (pred_image_example_primary * 255).astype(np.uint8)
            
            label_depth_example_primary = label_depths_primary[0][0].permute(1, 2, 0).detach().cpu().float().numpy()
            label_depth_example_primary = (label_depth_example_primary - label_depth_example_primary.min()) / (label_depth_example_primary.max() - label_depth_example_primary.min())
            # pred_image_example_primary = (label_image_example_primary * 255).astype(np.uint8)
            
            
            pred_depth_example_secondary = depth_gripper_pred[0][0].permute(1, 2, 0).detach().float().cpu().numpy()
            pred_depth_example_secondary = (pred_depth_example_secondary - pred_depth_example_secondary.min()) / (pred_depth_example_secondary.max() - pred_depth_example_secondary.min())
            # pred_image_example_secondary = (pred_image_example_secondary * 255).astype(np.uint8)
            
            label_depth_wrist_example = label_depths_wrist[0][0].permute(1, 2, 0).detach().float().cpu().numpy()
            label_depth_wrist_example = (label_depth_wrist_example - label_depth_wrist_example.min()) / (label_depth_wrist_example.max() - label_depth_wrist_example.min())
            # label_image_wrist_example = (label_image_wrist_example * 255).astype(np.uint8)        
            
            
            
            
        else:
            loss_pred_depth_x = torch.tensor([0.0]).to(device_id)
            loss_pred_depth_gripper = torch.tensor([0.0]).to(device_id)
            loss_pred_depth = torch.tensor([0.0]).to(device_id)

        if args.loss_dino_feat and args.dino_feat_pred:
            label_dino_feat_primary = dino_feat_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :]
            label_dino_feat_wrist = dino_feat_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :]
            dino_feat_pred = dino_feat_pred.reshape(bs, args.sequence_length, dino_feat_pred.shape[1], dino_feat_pred.shape[2], dino_feat_pred.shape[3], dino_feat_pred.shape[4])
            dino_feat_pred = dino_feat_pred[:, :args.sequence_length-args.atten_goal]
            dino_feat_pred = dino_feat_pred.reshape(-1, dino_feat_pred.shape[2], dino_feat_pred.shape[3], dino_feat_pred.shape[4], dino_feat_pred.shape[5])
            dino_feat_x_pred, dino_feat_gripper_pred  = torch.split(dino_feat_pred, 1, dim=1)
            dino_feat_x_pred = dino_feat_x_pred.squeeze(1)   # [B*S, P, C, 768]
            dino_feat_gripper_pred = dino_feat_gripper_pred.squeeze(1) # [B*S, P, 196, 768]    
            
            # cosine loss
            label_dino_feat_primary = label_dino_feat_primary.reshape(-1, *label_dino_feat_primary.shape[2:])
            label_dino_feat_wrist = label_dino_feat_wrist.reshape(-1, *label_dino_feat_wrist.shape[2:])
            assert dino_feat_x_pred.shape[1] == 1
            dino_feat_x_pred = dino_feat_x_pred.squeeze(1)
            dino_feat_gripper_pred = dino_feat_gripper_pred.squeeze(1)
            loss_pred_dino_feat_x = (1 - F.cosine_similarity(dino_feat_x_pred, label_dino_feat_primary, dim=-1)).mean()
            loss_pred_dino_feat_gripper = (1 - F.cosine_similarity(dino_feat_gripper_pred, label_dino_feat_wrist, dim=-1)).mean()
            loss_pred_dino_feat = 0.5*(loss_pred_dino_feat_x + loss_pred_dino_feat_gripper)
            
        else:
            loss_pred_dino_feat_x = torch.tensor([0.0]).to(device_id)
            loss_pred_dino_feat_gripper = torch.tensor([0.0]).to(device_id)
            loss_pred_dino_feat = torch.tensor([0.0]).to(device_id)

        if args.loss_sam_feat and args.sam_feat_pred:
            label_sam_feat_primary = sam_feat_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :]
            label_sam_feat_wrist = sam_feat_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :]
            sam_feat_pred = sam_feat_pred.reshape(bs, args.sequence_length, sam_feat_pred.shape[1], sam_feat_pred.shape[2], sam_feat_pred.shape[3], sam_feat_pred.shape[4])
            sam_feat_pred = sam_feat_pred[:, :args.sequence_length-args.atten_goal]
            sam_feat_pred = sam_feat_pred.reshape(-1, sam_feat_pred.shape[2], sam_feat_pred.shape[3], sam_feat_pred.shape[4], sam_feat_pred.shape[5])
            sam_feat_x_pred, sam_feat_gripper_pred  = torch.split(sam_feat_pred, 1, dim=1)
            sam_feat_x_pred = sam_feat_x_pred.squeeze(1)   # [B*S, P, C, 768]
            sam_feat_gripper_pred = sam_feat_gripper_pred.squeeze(1) # [B*S, P, 196, 768]    

            # cosine loss
            label_sam_feat_primary = label_sam_feat_primary.reshape(-1, *label_sam_feat_primary.shape[2:])
            label_sam_feat_wrist = label_sam_feat_wrist.reshape(-1, *label_sam_feat_wrist.shape[2:])
            assert sam_feat_x_pred.shape[1] == 1
            sam_feat_x_pred = sam_feat_x_pred.squeeze(1)
            sam_feat_gripper_pred = sam_feat_gripper_pred.squeeze(1)
            loss_pred_sam_feat_x = (1 - F.cosine_similarity(sam_feat_x_pred, label_sam_feat_primary, dim=-1)).mean()
            loss_pred_sam_feat_gripper = (1 - F.cosine_similarity(sam_feat_gripper_pred, label_sam_feat_wrist, dim=-1)).mean()
            loss_pred_sam_feat = 0.5*(loss_pred_sam_feat_x + loss_pred_sam_feat_gripper)
            
        else:
            loss_pred_sam_feat_x = torch.tensor([0.0]).to(device_id)
            loss_pred_sam_feat_gripper = torch.tensor([0.0]).to(device_id)
            loss_pred_sam_feat = torch.tensor([0.0]).to(device_id)
            
        if args.loss_trajectory and args.trajectory_pred:
            
            label_tracks_primary = track_infos['tracks_l'][:, 0:args.sequence_length-args.atten_goal+args.pred_num-1, :].cuda()
            label_tracks_wrist = track_infos['tracks_gripper_l'][:, 0:args.sequence_length-args.atten_goal+args.pred_num-1, :].cuda()
            h = w = int(math.sqrt(label_tracks_primary.shape[-2]))

            label_traj_example_primary = rearrange(label_tracks_primary[0][0], '(h w) c -> h w c', h=h, w=w).detach().cpu().numpy()
            label_traj_example_primary = visualize_optical_flow(label_traj_example_primary) / 255 # [h, w, 3]
            label_traj_example_secondary = rearrange(label_tracks_wrist[0][0], '(h w) c -> h w c', h=h, w=w).detach().cpu().numpy()
            label_traj_example_secondary = visualize_optical_flow(label_traj_example_secondary) / 255 # [h, w, 3]
            
            if args.no_unshuffle:
                label_tracks_primary = label_tracks_primary
                label_tracks_wrist = label_tracks_wrist
            
            else:
                label_tracks_primary = rearrange(label_tracks_primary, 'b p (h w) c -> b p c h w', h=h, w=w)
                label_tracks_primary = F.pixel_unshuffle(label_tracks_primary, downscale_factor=h//14)
                
                label_tracks_wrist = rearrange(label_tracks_wrist, 'b p (h w) c -> b p c h w', h=h, w=w)
                label_tracks_wrist = F.pixel_unshuffle(label_tracks_wrist, downscale_factor=h//14)
            
                label_tracks_primary = rearrange(label_tracks_primary, 'b p c h w -> b p (h w) c')
                label_tracks_wrist = rearrange(label_tracks_wrist, 'b p c h w -> b p (h w) c')
                
            label_tracks_primary = label_tracks_primary.unfold(1, args.pred_num, 1).permute(0, 1, 4, 2, 3).flatten(0, 1)

            label_tracks_wrist = label_tracks_wrist.unfold(1, args.pred_num, 1).permute(0, 1, 4, 2, 3).flatten(0, 1)
            

            traj_pred = traj_pred.reshape(bs, args.sequence_length, traj_pred.shape[1], traj_pred.shape[2], traj_pred.shape[3], traj_pred.shape[4])
            traj_pred = traj_pred[:, :args.sequence_length-args.atten_goal]
            traj_pred = traj_pred.reshape(-1, traj_pred.shape[2], traj_pred.shape[3], traj_pred.shape[4], traj_pred.shape[5])
            if args.no_pred_gripper_traj:
                traj_pred_parimary = traj_pred
                traj_pred_wrist = torch.zeros_like(traj_pred)
            else:
                traj_pred_parimary, traj_pred_wrist = torch.split(traj_pred, 1, dim=1)
            traj_pred_parimary = traj_pred_parimary.squeeze(1)
            traj_pred_wrist = traj_pred_wrist.squeeze(1)

            
            loss_pred_traj_x = F.mse_loss(traj_pred_parimary, label_tracks_primary)
            
            loss_pred_traj_gripper = 0 if args.no_pred_gripper_traj else F.mse_loss(traj_pred_wrist, label_tracks_wrist)
            loss_pred_trajectory = 0.1 * (loss_pred_traj_x + loss_pred_traj_gripper)
            
            # if args.no_unshuffle:
            #     h = w = 28
            # else:
            #     h=w=14
            # pred_traj_example_primary = rearrange(traj_pred_parimary[0][0], '(h w) c -> c h w', h=h, w=w)
            # pred_traj_example_primary = F.pixel_shuffle(pred_traj_example_primary, upscale_factor=h//h).permute(1, 2, 0).detach().cpu().numpy()
            # pred_traj_example_primary = visualize_optical_flow(pred_traj_example_primary) / 255 # [h, w, 3]
            # pred_traj_example_secondary = rearrange(traj_pred_wrist[0][0], '(h w) c -> c h w', h=h, w=w)
            # pred_traj_example_secondary = F.pixel_shuffle(pred_traj_example_secondary, upscale_factor=h//h).permute(1, 2, 0).detach().cpu().numpy()
            # pred_traj_example_secondary = visualize_optical_flow(pred_traj_example_secondary) / 255 # [h, w, 3]
            
            # new_array = np.zeros((14, 14, 2))
            # for k in range(2):  # 遍历每个通道
            #     for i in range(14):
            #         for j in range(14):
            #             new_array[i, j, k] = np.mean(label_tracks_primary[:, :, 2*i:2*i+2, 2*j:2*j+2, k])
            # np_norm  = np.linalg.norm(new_array, axis=2)
            # mask = np_norm > 1
            
            # for i in range(14):
            #     for j in range(14):
            #         if mask[i, j]:
            #             if (np.round(new_array[i, j, 0] / 8).astype(int))!= 0 or (np.round(new_array[i, j, 1] / 8).astype(int)) != 0:
            #                 target_i = i + np.round(new_array[i, j, 0] / 8).astype(int)
            #                 target_j = j + np.round(new_array[i, j, 1] / 8).astype(int)
            #                 if 0 <= target_i < 14 and 0 <= target_j < 14:
            #                     if mask[target_i, target_j] == False:
            #                         mask[target_i, target_j] = True
            # temp_mask = np.copy(mask)
            # for i in range(14):
            #     for j in range(14):
            #         if mask[i, j]:
            #             # 将当前元素的上下左右也设置为 True
            #             if i > 0:
            #                 temp_mask[i-1, j] = True
            #             if i < 13:
            #                 temp_mask[i+1, j] = True
            #             if j > 0:
            #                 temp_mask[i, j-1] = True
            #             if j < 13:
            #                 temp_mask[i, j+1] = True

            # mask = temp_mask
            # mask_reshaped = mask.reshape(1, -1, 1) 
            # label_image_primary = images_primary[:, :args.sequence_length].flatten(0,1)
            # label_image_primary = patchify(label_image_primary, patch_size=args.patch_size)
            # mask_img = label_image_primary[0].detach().cpu().numpy()* mask_reshaped
            # masked_label_image = np.expand_dims(mask_img, axis=1)
            # masked_label_image = torch.tensor(masked_label_image)
            # mask_img = unpatchify(masked_label_image)
            
            # mask_img_np = mask_img.squeeze().permute(1, 2, 0).numpy()

            # mask_img_np = (mask_img_np - mask_img_np.min()) / (mask_img_np.max() - mask_img_np.min()) * 255
            # mask_img_np = mask_img_np.astype(np.uint8)

            # image = Image.fromarray(mask_img_np)
            # image.save("./mask_image_2.png")
            
        else:
            loss_pred_trajectory = torch.tensor([0.0]).to(device_id)
        
        
        
        
        # mask = torch.tensor(mask).to(images_primary.device)
        # mask_3d = torch.stack([mask, mask, mask], dim=0)
        # mask_img_np = (images_primary[:, :args.sequence_length].flatten(0,1)[0]*mask_3d).permute(1, 2, 0).detach().cpu().numpy()
        
        # from PIL import Image

        # mask_img_np = (mask_img_np - mask_img_np.min()) / (mask_img_np.max() - mask_img_np.min()) * 255
        # mask_img_np = mask_img_np.astype(np.uint8)

        # image = Image.fromarray(mask_img_np)
        # image.save("./mask_image_1.png")
       

        # from torchvision.utils import save_image
        # save_image(images_primary[:, 0:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :][0][0], './output_future.png', normalize=True)
        # save_image(images_primary[:, 0:args.future_steps+args.sequence_length-args.atten_goal+args.pred_num-1, :][0][0], './output_current.png', normalize=True)
        loss_calvin = args.loss_arm_action_ratio * loss_arm_action + args.loss_gripper_action_ratio * loss_gripper_action + 0.1 * loss_image + 0.001*loss_pred_depth + 0.1 * loss_pred_trajectory + 0.01 * loss_pred_dino_feat + 0.01 * loss_pred_sam_feat

        # gradient_accumulation_steps        
        loss = loss_calvin / args.gradient_accumulation_steps
        loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
        loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        loss_depth = loss_pred_depth / args.gradient_accumulation_steps
        loss_dino_feat = loss_pred_dino_feat / args.gradient_accumulation_steps
        loss_sam_feat = loss_pred_sam_feat / args.gradient_accumulation_steps
        loss_pred_trajectory = loss_pred_trajectory / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        ### backward pass ###
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )
                
                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                step_time_m.reset()
                data_time_m.reset()

                # log image
                if args.obs_pred and num_steps%10000==0:
                    wandb.log(
                        {
                            "pred_image": wandb.Image(
                                pred_image_example_primary,
                                caption="pred_image_example_primary",
                            ),
                            "pred_image_wrist": wandb.Image(
                                pred_image_example_secondary,
                                caption="pred_image_example_wrist",
                            ),
                            
                            "label_image": wandb.Image(
                                label_image_example_primary,
                                caption="label_image_example_primary",
                            ),
                            "label_image_wrist": wandb.Image(
                                label_image_wrist_example,
                                caption="label_image_example_wrist",
                            ),
                            
                        }
                    )
                if args.depth_pred and num_steps%10000==0:
                    wandb.log(
                        {
                            "pred_depth": wandb.Image(
                                pred_depth_example_primary,
                                caption="pred_depth_example_primary",
                            ),
                            "pred_depth_wrist": wandb.Image(
                                pred_depth_example_secondary,
                                caption="pred_depth_example_wrist",
                            ),
                            
                            "label_depth": wandb.Image(
                                label_depth_example_primary,
                                caption="label_depth_example_primary",
                            ),
                            "label_depth_wrist": wandb.Image(
                                label_depth_wrist_example,
                                caption="label_depth_example_wrist",
                            ),
                            
                        }
                    )
                if args.trajectory_pred and num_steps%10000==0:
                    wandb.log(
                        {
                            "pred_traj": wandb.Image(
                                pred_traj_example_primary,
                                caption="pred_traj_primary",
                            ),
                            "pred_traj_wrist": wandb.Image(
                                pred_traj_example_secondary,
                                caption="pred_traj_wrist",
                            ),
                            
                            "label_traj": wandb.Image(
                                label_traj_example_primary,
                                caption="label_traj_primary",
                            ),
                            "label_traj_wrist": wandb.Image(
                                label_traj_example_secondary,
                                caption="label_traj_wrist",
                            ),
                            
                        }
                    )
                
                wandb.log(
                    {
                        "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                        "loss_arm_action": loss_arm_action.item() * args.gradient_accumulation_steps,
                        "loss_gripper_action": loss_gripper_action.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                        "loss_depth": loss_pred_depth.item() * args.gradient_accumulation_steps,
                        "loss_dino_feat": loss_dino_feat.item() * args.gradient_accumulation_steps,
                        "loss_sam_feat": loss_sam_feat.item() * args.gradient_accumulation_steps,
                        "loss_pred_trajectory": loss_pred_trajectory.item() * args.gradient_accumulation_steps,
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                )

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "loss_image": loss_image.item(), "loss_depth":loss_depth.item(), "loss_arm_action": loss_arm_action.item(), "loss_gripper_action": loss_gripper_action.item(), "loss_pred_trajectory": loss_pred_trajectory.item(), "loss_dino_feat":loss_dino_feat.item(), "loss_sam_feat":loss_sam_feat.item()})

        # if args.save_every_iter != -1 and args.save_checkpoint and global_step % args.save_every_iter == 0 and global_step > 0:
                
        #     if args.rank == 0:
        #         import os
        #         if not os.path.exists(f"{args.save_checkpoint_path}/exp/{args.run_name}"):
        #             os.makedirs(f"{args.save_checkpoint_path}/exp/{args.run_name}")

        #         checkpoint_dict = {
        #             "epoch": epoch,
        #             "model_state_dict": get_checkpoint(model),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         }

        #         ckpt_name = get_ckpt_name(args, global_step)
        #         ckpt_path = os.path.join(f"{args.save_checkpoint_path}/exp", args.run_name, ckpt_name)
        #         print(f"Saving checkpoint to {ckpt_path}")
        #         torch.save(checkpoint_dict, ckpt_path)
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0:
        #                 os.remove(ckpt_path)

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def unpatchify(patches, patch_size=16, img_size=[224, 224]):
    """
    patches: [B, S, P, 196, 768] ->  [B, S, P, C, H, W]
    patch_size
    img_size: (H, W)
    """
    B, P, num_patches, patch_dim = patches.shape
    H, W = img_size
    grid_size = int(num_patches ** 0.5)  # 14 if num_patches=196

    C = patch_dim // (patch_size * patch_size)  
    patches = patches.view(B, P, grid_size, grid_size, patch_size, patch_size, C)

    img = patches.permute(0, 1, 6, 2, 4, 3, 5).contiguous()
    img = img.view(B, P, C, H, W)  

    return img

def mask_img_base_traj(label_tracks, h=28, w=28):
    
    new_array = np.zeros((14, 14, 2))
    label_traj_example_primary = rearrange(label_tracks, '(h w) c -> h w c', h=h, w=w).detach().cpu().numpy()
    
    for k in range(2):  
        for i in range(14):
            for j in range(14):
                new_array[i, j, k] = np.mean(label_traj_example_primary[2*i:2*i+2, 2*j:2*j+2, k])
    np_norm  = np.linalg.norm(new_array, axis=2)
    mask = np_norm > 1
    
    for i in range(14):
        for j in range(14):
            if mask[i, j]:
                if (np.round(new_array[i, j, 0] / 8).astype(int))!= 0 or (np.round(new_array[i, j, 1] / 8).astype(int)) != 0:
                    target_i = i + np.round(new_array[i, j, 0] / 8).astype(int)
                    target_j = j + np.round(new_array[i, j, 1] / 8).astype(int)
                    if 0 <= target_i < 14 and 0 <= target_j < 14:
                        if mask[target_i, target_j] == False:
                            mask[target_i, target_j] = True
    temp_mask = np.copy(mask)
    for i in range(14):
        for j in range(14):
            if mask[i, j]:
                if i > 0:
                    temp_mask[i-1, j] = True
                if i < 13:
                    temp_mask[i+1, j] = True
                if j > 0:
                    temp_mask[i, j-1] = True
                if j < 13:
                    temp_mask[i, j+1] = True

    
    return temp_mask