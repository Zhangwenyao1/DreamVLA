import os
import random
from functools import partial
from copy import deepcopy
from timm.models.vision_transformer import Block
import torch
import time
from torch import nn
import torch.nn.functional as F
import clip
import numpy as np
from models.vit_mae import MaskedAutoencoderViT
from models.perceiver_resampler import PerceiverResampler
from models.gpt2 import GPT2Model
from transformers import GPT2Config
from pdb import set_trace
import random 
from models.action_model.action_model import  ActionModel
from utils.sigloss import SiLogLoss
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union
from torch.cuda.amp import autocast

# def generate_attention_mask(K, num_A, num_B, atten_goal, atten_goal_state,
#                             atten_only_obs,
#                             attn_robot_proprio_state,
#                             mask_l_obs_ratio,
#                             num_obs_token, action_pred_steps, num_obs, num_depth, num_traj, action_atten_self=False, flow_atten_depth=False):
#     # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
#     # num_A: text, state, image_embedding, image_cls_token_embedding
#     # num_B: self.NUM_OBS_TOKEN+self.action_pred_steps
#     # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
#     sequence_length = (num_A + num_B) * K
#     attention_mask = torch.zeros((sequence_length, sequence_length))
#     for i in range(K):
#         start_index = i * (num_A + num_B)
#         end_index = start_index + num_A + num_B
        
#         # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
#         attention_mask[start_index:end_index, end_index:] = -float('inf')
        
#         # the sub-sub-sequence B can not be attended to
#         attention_mask[:, start_index+num_A:end_index] = -float('inf')
        
#         # if obs_token exists, action_pred_token should attend to it
#         if num_obs_token > 0 and action_pred_steps:
#             if not action_atten_self:
#                 attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
#             else:
#                 attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token+action_pred_steps] = 0.0 
#         if num_obs_token > 0 and num_depth:
#             attention_mask[start_index+num_A+num_obs:start_index+num_A+num_obs+num_depth, start_index+num_A+num_obs:start_index+num_A+num_obs+num_depth] = 0.0
#         if num_obs_token > 0 and num_traj:
#             if not flow_atten_depth:
#                 attention_mask[start_index+num_A+num_obs+num_depth:start_index+num_A+num_obs+num_depth+num_traj, start_index+num_A+num_obs+num_depth:start_index+num_A+num_obs+num_depth+num_traj] = 0.0
#             else:
#                 attention_mask[start_index+num_A+num_obs+num_depth:start_index+num_A+num_obs+num_depth+num_traj, start_index+num_A+num_obs:start_index+num_A+num_obs+num_depth+num_traj] = 0.0
#         if num_obs_token > 0 and atten_only_obs and action_pred_steps:
#             attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps] = -float('inf')
#             attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+2:start_index+num_A] = 0.0
#             attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
#             if attn_robot_proprio_state:
#                 attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+1:start_index+2] = 0.0
#             if mask_l_obs_ratio > 0:
#                 count = int(mask_l_obs_ratio * (num_obs_token))
#                 selected_numbers = np.random.choice(range(num_obs_token), size=count, replace=False)
#                 for num in selected_numbers:
#                     attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A+num] = -float('inf')
#         if num_obs_token > 0 and atten_goal:
#             if i < K - atten_goal:
#                 pred_end_index = (i + atten_goal) * (num_A + num_B)
#                 if atten_goal_state:
#                     attention_mask[start_index+num_A:start_index+num_A+num_obs_token,pred_end_index+1:pred_end_index+2] = 0.0

#     return attention_mask

def generate_attention_mask(K, num_A, num_B, atten_goal, atten_goal_state,
                            atten_only_obs,
                            attn_robot_proprio_state,
                            mask_l_obs_ratio,
                            num_obs_token, action_pred_steps):
    # num_A: 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
    # num_A: text, state, image_embedding, image_cls_token_embedding
    # num_B: self.NUM_OBS_TOKEN+self.action_pred_steps
    # num_B: obs_tokens(if exists), action_pred_token, state_pred_token (if exists)
    sequence_length = (num_A + num_B) * K
    attention_mask = torch.zeros((sequence_length, sequence_length))
    for i in range(K):
        start_index = i * (num_A + num_B)
        end_index = start_index + num_A + num_B
        
        # the i-th sub-sequence can not attend to the sub-sequences that after the i-th
        attention_mask[start_index:end_index, end_index:] = -float('inf')
        
        # the sub-sub-sequence B can not be attended to
        attention_mask[:, start_index+num_A:end_index] = -float('inf')
        
        # if obs_token exists, action_pred_token should attend to it
        if num_obs_token > 0 and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
        if num_obs_token > 0 and atten_only_obs and action_pred_steps:
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps] = -float('inf')
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+2:start_index+num_A] = 0.0
            attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A:start_index+num_A+num_obs_token] = 0.0 
            if attn_robot_proprio_state:
                attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+1:start_index+2] = 0.0
            if mask_l_obs_ratio > 0:
                count = int(mask_l_obs_ratio * (num_obs_token))
                selected_numbers = np.random.choice(range(num_obs_token), size=count, replace=False)
                for num in selected_numbers:
                    attention_mask[start_index+num_A+num_obs_token:start_index+num_A+num_obs_token+action_pred_steps, start_index+num_A+num] = -float('inf')
        if num_obs_token > 0 and atten_goal:
            if i < K - atten_goal:
                pred_end_index = (i + atten_goal) * (num_A + num_B)
                if atten_goal_state:
                    attention_mask[start_index+num_A:start_index+num_A+num_obs_token,pred_end_index+1:pred_end_index+2] = 0.0

    return attention_mask




def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

class DreamVLA(nn.Module):
    def __init__(
        self,
        finetune_type,
        clip_device,
        vit_checkpoint_path,
        sequence_length=10,
        num_resampler_query=9,
        num_obs_token_per_image=10,
        obs_pred=False,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=False,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        calvin_input_image_size=224,
        patch_size=16,
        mask_ratio=0.0,
        num_token_per_timestep=41,
        input_self=False,
        action_pred_steps=1,
        transformer_layers=12,
        hidden_dim=384,
        transformer_heads=12,
        phase="",
        gripper_width=False,
        
        pred_num = 1, 
        depth_pred=False,
        trajectory_pred=False,
        use_depth_query=False,
        use_dpt_head=False,
        use_trajectory_query=False,
        track_label_patch_size=4,
        dino_feat_pred=False,
        sam_feat_pred=False,
        use_dinosiglip=False,
        use_dit_head = False,
        use_gpt2_pretrained = False,
        no_pred_gripper_traj = False,
        no_unshuffle = False,
        share_query = False,
        attn_implementation = False
    ):
        super().__init__()
        self.finetune_type = finetune_type
        self.device = clip_device
        self.sequence_length = sequence_length
        self.action_pred_steps = action_pred_steps
        self.obs_pred = obs_pred
        self.depth_pred = depth_pred
        self.dino_feat_pred = dino_feat_pred
        self.sam_feat_pred = sam_feat_pred
        self.trajectory_pred = trajectory_pred
        self.atten_goal = atten_goal
        self.atten_goal_state = atten_goal_state
        self.atten_only_obs = atten_only_obs
        self.attn_robot_proprio_state = attn_robot_proprio_state
        self.mask_l_obs_ratio = mask_l_obs_ratio
        self.hidden_dim = hidden_dim
        self.phase = phase
        assert self.phase in ["pretrain", "finetune", "evaluate"]
        
        self.share_query = share_query
        
        
        self.gripper_width = gripper_width
        self.vit_checkpoint_path = vit_checkpoint_path
        self.pred_num = pred_num
        # text projector
        self.text_projector = nn.Linear(512, self.hidden_dim)        

        # state encoder
        ARM_STATE_FEATURE_DIM = self.hidden_dim 
        GRIPPER_STATE_FEATURE_DIM = self.hidden_dim
        self.arm_state_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.gripper_state_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.state_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim)

        # action encoder
        self.action_pose_encoder = nn.Linear(6, ARM_STATE_FEATURE_DIM)
        self.action_gripper_position_encoder = nn.Linear(2, GRIPPER_STATE_FEATURE_DIM)
        self.action_projector = nn.Linear(ARM_STATE_FEATURE_DIM + GRIPPER_STATE_FEATURE_DIM, self.hidden_dim)

        self.use_dinosiglip = use_dinosiglip
        
        # vision encoder (frozen)
        if not self.use_dinosiglip:
            self.vision_encoder = MaskedAutoencoderViT(
                patch_size=16, embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            self.RESAMPLER_hidden_dim = 768  
            self.NUM_RESAMPLER_QUERY = num_resampler_query
            self.perceiver_resampler = PerceiverResampler(dim=self.RESAMPLER_hidden_dim, num_latents=self.NUM_RESAMPLER_QUERY, depth=3)
            self.image_primary_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
            self.cls_token_primary_projector = nn.Linear(768, self.hidden_dim)
            self.image_wrist_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
            self.cls_token_wrist_projector = nn.Linear(768, self.hidden_dim)
        else:
            self.RESAMPLER_hidden_dim = 2176  
            self.NUM_RESAMPLER_QUERY = num_resampler_query
            self.perceiver_resampler = PerceiverResampler(dim=self.RESAMPLER_hidden_dim, num_latents=self.NUM_RESAMPLER_QUERY, depth=3)
            self.image_primary_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
            self.cls_token_primary_projector = nn.Linear(1024, self.hidden_dim)
            self.image_wrist_projector = nn.Linear(self.RESAMPLER_hidden_dim, self.hidden_dim)
            self.cls_token_wrist_projector = nn.Linear(1024, self.hidden_dim)
        # resampler
        


        # action_pred_token
        if self.action_pred_steps > 0:
            self.action_pred_token = nn.Parameter(torch.zeros(1, 1, self.action_pred_steps, self.hidden_dim))

        # obs_token
        
        self.NUM_OBS_TOKEN = 0
        self.NUM_DEPTH_TOKEN = 0
        self.NUM_TRAJ_TOKEN = 0
        self.NUM_DINO_TOKEN = 0
        self.NUM_SAM_TOKEN = 0
        

        if self.obs_pred:
            self.NUM_OBS_TOKEN_PER_IMAGE = num_obs_token_per_image
            self.NUM_OBS_TOKEN = self.NUM_OBS_TOKEN_PER_IMAGE * 2
            self.obs_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_OBS_TOKEN, self.hidden_dim))
        if self.depth_pred:
            self.NUM_OBS_TOKEN_PER_DEPTH = num_obs_token_per_image
            self.NUM_DEPTH_TOKEN = self.NUM_OBS_TOKEN_PER_DEPTH * 2
        if self.dino_feat_pred:
            self.NUM_OBS_TOKEN_PER_DINO = num_obs_token_per_image
            self.NUM_DINO_TOKEN = self.NUM_OBS_TOKEN_PER_DINO * 2
        if self.sam_feat_pred:
            self.NUM_OBS_TOKEN_PER_SAM = num_obs_token_per_image
            self.NUM_SAM_TOKEN = self.NUM_OBS_TOKEN_PER_SAM * 2
        if self.trajectory_pred:
            self.NUM_OBS_TOKEN_PER_TRAJ = num_obs_token_per_image
            if no_pred_gripper_traj:
                self.NUM_TRAJ_TOKEN = self.NUM_OBS_TOKEN_PER_TRAJ
            else:
                self.NUM_TRAJ_TOKEN = self.NUM_OBS_TOKEN_PER_TRAJ * 2
        if not self.share_query:
            if self.depth_pred:
                self.depth_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_DEPTH_TOKEN, self.hidden_dim))
            if self.dino_feat_pred:
                self.dino_feat_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_DINO_TOKEN, self.hidden_dim))
            if self.sam_feat_pred:
                self.sam_feat_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_SAM_TOKEN, self.hidden_dim))
            if trajectory_pred:
                self.trajectory_tokens = nn.Parameter(torch.zeros(1, 1, self.NUM_TRAJ_TOKEN, self.hidden_dim))
            
            
        # causal transformer
        self.embedding_layer_norm = nn.LayerNorm(self.hidden_dim)
        if self.share_query:
            this_num_obs_token = self.NUM_OBS_TOKEN 
        elif self.obs_pred or self.depth_pred or self.trajectory_pred or self.dino_feat_pred or self.sam_feat_pred:
            this_num_obs_token = self.NUM_OBS_TOKEN + self.NUM_DEPTH_TOKEN + self.NUM_TRAJ_TOKEN + self.NUM_DINO_TOKEN + self.NUM_SAM_TOKEN
        else:
            this_num_obs_token = 0
        self.attention_mask = nn.Parameter(generate_attention_mask(
                                    K=self.sequence_length, 
                                    num_A=1+1+self.NUM_RESAMPLER_QUERY*2+1*2, 
                                    num_B=this_num_obs_token+self.action_pred_steps,
                                    atten_goal=self.atten_goal,
                                    atten_goal_state=self.atten_goal_state,
                                    atten_only_obs=self.atten_only_obs,
                                    attn_robot_proprio_state = self.attn_robot_proprio_state,
                                    mask_l_obs_ratio=self.mask_l_obs_ratio,
                                    num_obs_token=this_num_obs_token,
                                    action_pred_steps=self.action_pred_steps,
                                    # num_obs=self.NUM_OBS_TOKEN,
                                    # num_depth=self.NUM_DEPTH_TOKEN,
                                    # num_traj=self.NUM_TRAJ_TOKEN,
                                    ), 
                                    requires_grad=False)
        num_non_learnable_token_per_timestep = 1+1+self.NUM_RESAMPLER_QUERY*2+1*2
        self.transformer_backbone_position_embedding = nn.Parameter(torch.zeros(1, self.sequence_length, 1, self.hidden_dim), requires_grad=True)  # TODO How to initialize this embedding
        config = GPT2Config()
        if not use_gpt2_pretrained:
            config.hidden_size = self.hidden_dim
            config.n_layer = transformer_layers
            config.vocab_size = 1
            config.n_head = transformer_heads
            self.attn_implementation = config.attn_implementation = attn_implementation
            self.transformer_backbone = GPT2Model(config)
        else:
            config = GPT2Config.from_pretrained("gpt2-medium")
            config.vocab_size = 1

        


        # action decoder
        MLP_hidden_dim = self.hidden_dim // 2


        self.recon_state_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
        ) # not used
        self.recon_arm_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 6),
            torch.nn.Tanh(),
        ) # not used
        self.recon_gripper_state_decoder = nn.Sequential(
            nn.Linear(MLP_hidden_dim, 1),
            torch.nn.Sigmoid(),
        ) # not used

        
        if self.obs_pred:
            self.obs_projector = nn.Linear(self.hidden_dim, 128)
            self.IMAGE_DECODER_hidden_dim = 128
            # self.IMAGE_DECODER_hidden_dim = self.hidden_dim
            self.NUM_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size) * self.pred_num      # i.e. num_patch
            self.PATCH_SIZE = patch_size
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.IMAGE_DECODER_hidden_dim))
            self.image_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.IMAGE_DECODER_hidden_dim)
            self.image_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_IMAGE + self.NUM_MASK_TOKEN, self.IMAGE_DECODER_hidden_dim), requires_grad=False)  # fixed sin-cos embedding #   cls_token is alse passed to the decoder in mae
            self.image_decoder = nn.Sequential(
                Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                # Block(self.IMAGE_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                )
            self.image_decoder_norm = nn.LayerNorm(self.IMAGE_DECODER_hidden_dim)
            self.image_decoder_pred = nn.Linear(self.IMAGE_DECODER_hidden_dim, self.PATCH_SIZE**2 * 3)


        # depth prediction
        if self.depth_pred:
            self.depth_projecor = nn.Linear(self.hidden_dim, 128)
            self.DEPTH_DECODER_hidden_dim = 128
            self.use_dpt_head = use_dpt_head
            # self.DEPTH_DECODER_hidden_dim = self.hidden_dim
            self.NUM_DEPTH_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size) * self.pred_num  
            self.PATCH_SIZE = patch_size
            self.depth_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.DEPTH_DECODER_hidden_dim) # Is this layer necessary？
            if self.use_dpt_head:
                self.NUM_DEPTH_MASK_TOKEN = 256*self.pred_num
            self.depth_decoder = nn.Sequential(
                Block(self.DEPTH_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                # Block(self.DEPTH_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                )
            self.depth_decoder_norm = nn.LayerNorm(self.DEPTH_DECODER_hidden_dim)
            self.depth_decoder_pred = nn.Linear(self.DEPTH_DECODER_hidden_dim, self.PATCH_SIZE**2 * 1) # depth is one
            self.depth_loss_head = SiLogLoss()
            self.depth_mask_token = nn.Parameter(torch.zeros(1, 1, self.DEPTH_DECODER_hidden_dim))
            self.depth_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_DEPTH + self.NUM_DEPTH_MASK_TOKEN, self.DEPTH_DECODER_hidden_dim), requires_grad=False)  # fixed sin-

        if self.dino_feat_pred:
            self.dino_projector = nn.Linear(self.hidden_dim, 128)
            self.DINO_DECODER_hidden_dim = 128
            # self.DINO_DECODER_hidden_dim = self.hidden_dim
            self.NUM_DINO_MASK_TOKEN = 256 * self.pred_num # int(calvin_input_image_size**2 / patch_size / patch_size) * self.pred_num  
            self.dino_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.DINO_DECODER_hidden_dim) # Is this layer necessary？
            self.dino_feat_decoder = nn.Sequential(
                Block(self.DINO_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                # Block(self.DINO_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                )
            self.dino_decoder_norm = nn.LayerNorm(self.DINO_DECODER_hidden_dim)
            self.dino_decoder_pred = nn.Linear(self.DINO_DECODER_hidden_dim, 768) # dino特征维度768
            self.dino_loss_head = SiLogLoss()
            self.dino_mask_token = nn.Parameter(torch.zeros(1, 1, self.DINO_DECODER_hidden_dim))
            self.dino_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_DINO + self.NUM_DINO_MASK_TOKEN, self.DINO_DECODER_hidden_dim), requires_grad=False)

        if self.sam_feat_pred:
            # self.sam_projector = nn.Linear(self.hidden_dim, 128)
            self.SAM_DECODER_hidden_dim = self.hidden_dim
            # self.SAM_DECODER_hidden_dim = 128
            self.NUM_SAM_MASK_TOKEN = 256 * self.pred_num # int(calvin_input_image_size**2 / patch_size / patch_size) * self.pred_num  
            self.sam_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.SAM_DECODER_hidden_dim) # Is this layer necessary？
            self.sam_feat_decoder = nn.Sequential(
                Block(self.SAM_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                # Block(self.SAM_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                )
            self.sam_decoder_norm = nn.LayerNorm(self.SAM_DECODER_hidden_dim)
            self.sam_decoder_pred = nn.Linear(self.SAM_DECODER_hidden_dim, 256) # sam特征维度256
            self.sam_loss_head = SiLogLoss()
            self.sam_mask_token = nn.Parameter(torch.zeros(1, 1, self.SAM_DECODER_hidden_dim))
            self.sam_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_SAM + self.NUM_SAM_MASK_TOKEN, self.SAM_DECODER_hidden_dim), requires_grad=False)

        if self.trajectory_pred:
            self.use_traj_query = use_trajectory_query
            self.track_label_patch_size = track_label_patch_size
            self.TRAJ_DECODER_hidden_dim = self.hidden_dim
            if no_unshuffle:
                self.NUM_TRAJ_MASK_TOKEN = 784 * self.pred_num
                self.traj_decoder_pred = nn.Linear(self.TRAJ_DECODER_hidden_dim, 2) 
            else:
                self.NUM_TRAJ_MASK_TOKEN = int(calvin_input_image_size**2 / patch_size / patch_size) * self.pred_num  
                self.traj_decoder_pred = nn.Linear(self.TRAJ_DECODER_hidden_dim, (patch_size//track_label_patch_size)**2 * 2) 
            self.PATCH_SIZE = patch_size
            self.traj_decoder_obs_pred_projector = nn.Linear(self.hidden_dim, self.TRAJ_DECODER_hidden_dim) # Is this layer necessary？
            self.traj_decoder = nn.Sequential(
                Block(self.TRAJ_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                Block(self.TRAJ_DECODER_hidden_dim, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm),
                )
            self.traj_decoder_norm = nn.LayerNorm(self.TRAJ_DECODER_hidden_dim)
            # depth is one
            # self.traj_loss_head = SiLogLoss()
            self.traj_mask_token = nn.Parameter(torch.zeros(1, 1, self.TRAJ_DECODER_hidden_dim))
            torch.nn.init.normal_(self.traj_mask_token, std=.02)

            self.traj_decoder_position_embedding = nn.Parameter(torch.zeros(1, self.NUM_OBS_TOKEN_PER_TRAJ + self.NUM_TRAJ_MASK_TOKEN, self.TRAJ_DECODER_hidden_dim), requires_grad=False)  # fixed sin-cos embedding #   cls_token is alse passed to the decoder in mae


        self.use_dit_head = use_dit_head
        if self.use_dit_head:

            action_model_type = "DiT-B"
            token_size = self.hidden_dim
            action_dim = 7
            future_action_window_size = self.action_pred_steps - 1
            past_action_window_size = 0
            self.action_model = ActionModel(model_type = action_model_type, 
                                            token_size = token_size, 
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size).to(torch.float32)
            
        else:
            self.action_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
            nn.Linear(MLP_hidden_dim, MLP_hidden_dim),
            nn.ReLU(),
            )
            self.arm_action_decoder = nn.Sequential(
                nn.Linear(MLP_hidden_dim, 6),
                torch.nn.Tanh(),
            )
            self.gripper_action_decoder = nn.Sequential(
                nn.Linear(MLP_hidden_dim, 1),
                torch.nn.Sigmoid(),
            )
            # initialize network
        self.initialize_weights()

        # freeze vision encoder
        if not self.use_dinosiglip:
            vit_checkpoint = torch.load(self.vit_checkpoint_path, map_location='cpu')
            msg = self.vision_encoder.load_state_dict(vit_checkpoint['model'], strict=False)
        else:
            import timm
            from timm.models.vision_transformer import VisionTransformer
            DINOSigLIP_VISION_BACKBONES = {
            "dinosiglip-vit-so-224px": {
                "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
                "siglip": "vit_so400m_patch14_siglip_224",
            },
            "dinosiglip-vit-so-384px": {
                "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
                "siglip": "vit_so400m_patch14_siglip_384",
            },
            }
            self.dino_timm_path_or_url = DINOSigLIP_VISION_BACKBONES['dinosiglip-vit-so-224px']["dino"]
            self.siglip_timm_path_or_url = DINOSigLIP_VISION_BACKBONES['dinosiglip-vit-so-224px']["siglip"]

            # Initialize both Featurizers (ViTs) by downloading from HF / TIMM Hub if necessary
            self.dino_featurizer: VisionTransformer = timm.create_model(
                self.dino_timm_path_or_url, pretrained=True, num_classes=0, img_size=224
            )
            self.dino_featurizer.eval()
            self.siglip_featurizer: VisionTransformer = timm.create_model(
                self.siglip_timm_path_or_url, pretrained=True, num_classes=0, img_size=224
            )
            
            self.dino_featurizer.forward = unpack_tuple(
                partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
            )
            self.siglip_featurizer.forward = unpack_tuple(
                partial(self.siglip_featurizer.get_intermediate_layers, n={len(self.siglip_featurizer.blocks) - 2})
            )
        # # freeze text encoder
        if os.path.exists("checkpoints/clip/ViT-B-32.pt"):
            self.clip_model, self.image_processor = clip.load("checkpoints/clip/ViT-B-32.pt", device=clip_device)
        else:
            self.clip_model, self.image_processor = clip.load("ViT-B/32", device=clip_device)
        
        if self.depth_pred and self.use_dpt_head:
            from utils.Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DPTHead, DepthAnythingV2
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            encoder = 'vits' # or 'vits', 'vitb', 'vitg'

            depth_model_full = DepthAnythingV2(**model_configs[encoder])
            depth_ckpt = torch.load(f'./checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu')
            depth_model_full.load_state_dict(depth_ckpt)
            del depth_ckpt

            # 2. 只提取 depth_head
            self.dpt_head = depth_model_full.depth_head
            # self.dpt_head.eval()
            # self.dpt_head.requires_grad_(False)

            # 3. 释放 full model
            del depth_model_full
            
        if use_gpt2_pretrained:
            self.transformer_backbone = GPT2Model.from_pretrained("gpt2-medium", config=config)
            from transformers import AutoModelForCausalLM
            self.transformer_backbone = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.obs_pred:
            image_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.IMAGE_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_IMAGE**.5), cls_token=False)
            image_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.IMAGE_DECODER_hidden_dim, int(self.NUM_MASK_TOKEN**.5), cls_token=False)
            image_decoder_position_embedding = np.concatenate((image_decoder_position_embedding_obs, image_decoder_position_embedding_mask), axis=0)
            self.image_decoder_position_embedding.data.copy_(torch.from_numpy(image_decoder_position_embedding).float().unsqueeze(0))
            torch.nn.init.normal_(self.mask_token, std=.02)
        if self.depth_pred:
            depth_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.DEPTH_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_DEPTH**.5), cls_token=False)
            depth_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.DEPTH_DECODER_hidden_dim, int(self.NUM_DEPTH_MASK_TOKEN**.5), cls_token=False)
            depth_decoder_position_embedding = np.concatenate((depth_decoder_position_embedding_obs, depth_decoder_position_embedding_mask), axis=0)
            self.depth_decoder_position_embedding.data.copy_(torch.from_numpy(depth_decoder_position_embedding).float().unsqueeze(0))
            torch.nn.init.normal_(self.depth_mask_token, std=.02)
        if self.dino_feat_pred:
            dino_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.DINO_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_DINO**.5), cls_token=False)
            dino_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.DINO_DECODER_hidden_dim, int(self.NUM_DINO_MASK_TOKEN**.5), cls_token=False)
            dino_decoder_position_embedding = np.concatenate((dino_decoder_position_embedding_obs, dino_decoder_position_embedding_mask), axis=0)
            self.dino_decoder_position_embedding.data.copy_(torch.from_numpy(dino_decoder_position_embedding).float().unsqueeze(0))
            torch.nn.init.normal_(self.dino_mask_token, std=.02)
        if self.trajectory_pred:
            traj_decoder_position_embedding_obs = get_2d_sincos_pos_embed(self.TRAJ_DECODER_hidden_dim, int(self.NUM_OBS_TOKEN_PER_TRAJ**.5), cls_token=False)
            traj_decoder_position_embedding_mask = get_2d_sincos_pos_embed(self.TRAJ_DECODER_hidden_dim, int(self.NUM_TRAJ_MASK_TOKEN**.5), cls_token=False)
            traj_decoder_position_embedding = np.concatenate((traj_decoder_position_embedding_obs, traj_decoder_position_embedding_mask), axis=0)
            self.traj_decoder_position_embedding.data.copy_(torch.from_numpy(traj_decoder_position_embedding).float().unsqueeze(0))
            torch.nn.init.normal_(self.traj_mask_token, std=.02)
        torch.nn.init.normal_(self.transformer_backbone_position_embedding, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_model_type(self):
        if not self.use_dinosiglip:
            self.vision_encoder_type = next(self.vision_encoder.parameters()).type()
        else:
            self.vision_encoder_type = next(self.dino_featurizer.parameters()).type()
        self.perceiver_resampler_type = next(self.perceiver_resampler.parameters()).type()
        self.transformer_backbone_type = next(self.transformer_backbone.parameters()).type()
        if not self.use_dit_head:
            self.action_decoder_type = next(self.action_decoder.parameters()).type()
        # else:
        #     self.action_decoder_type = next(self.action_model.parameters()).type()

    def run_multi_decoder_parallel(self, transformer_output, pred_token_start_idx, B, S, mode):
        """
        并发执行 image / depth / dino / sam 分支 decoder，使用独立 decoder 和 CUDA stream。
        """
        cur_query_start_idx = 0
        outputs = {}
        streams = {}
        tasks = {}

        def prepare(task_name, num_token, tokens_per_input, projector, mask_token,
                    pos_embed, decoder, norm, pred_head, num_mask_token):
            nonlocal cur_query_start_idx
            if mode == 'train':
                token_start = pred_token_start_idx + cur_query_start_idx
                token_end = token_start + num_token
                feat = transformer_output[:, :, token_start:token_end, :].reshape(-1, self.hidden_dim)
                projected = projector(feat).view(B * S * (num_token // tokens_per_input), tokens_per_input, -1)
                masks = mask_token.repeat(projected.shape[0], num_mask_token, 1)
                decoder_input = torch.cat([projected, masks], dim=1) + pos_embed
                tasks[task_name] = (decoder_input, decoder, norm, pred_head, num_token, tokens_per_input, num_mask_token)
                streams[task_name] = torch.cuda.Stream()
                if not self.share_query:
                    cur_query_start_idx += num_token
                else:
                    cur_query_start_idx = 0

        # 注册任务
        if self.obs_pred:
            prepare("image", self.NUM_OBS_TOKEN, self.NUM_OBS_TOKEN_PER_IMAGE,
                self.image_decoder_obs_pred_projector, self.mask_token,
                self.image_decoder_position_embedding, self.image_decoder,
                self.image_decoder_norm, self.image_decoder_pred, self.NUM_MASK_TOKEN)
        if self.depth_pred:
            prepare("depth", self.NUM_DEPTH_TOKEN, self.NUM_OBS_TOKEN_PER_DEPTH,
                self.depth_decoder_obs_pred_projector, self.depth_mask_token,
                self.depth_decoder_position_embedding, self.depth_decoder,
                self.depth_decoder_norm, self.depth_decoder_pred, self.NUM_DEPTH_MASK_TOKEN)
        if self.dino_feat_pred:
            prepare("dino", self.NUM_DINO_TOKEN, self.NUM_OBS_TOKEN_PER_DINO,
                self.dino_decoder_obs_pred_projector, self.dino_mask_token,
                self.dino_decoder_position_embedding, self.dino_feat_decoder,
                self.dino_decoder_norm, self.dino_decoder_pred, self.NUM_DINO_MASK_TOKEN)
        if self.sam_feat_pred:
            prepare("sam", self.NUM_SAM_TOKEN, self.NUM_OBS_TOKEN_PER_SAM,
                self.sam_decoder_obs_pred_projector, self.sam_mask_token,
                self.sam_decoder_position_embedding, self.sam_feat_decoder,
                self.sam_decoder_norm, self.sam_decoder_pred, self.NUM_SAM_MASK_TOKEN)

        # 执行所有 decoder 的 forward
        for name, (decoder_input, decoder, norm, pred, num_token, per_input, mask_num) in tasks.items():
            with torch.cuda.stream(streams[name]):
                out = decoder(decoder_input)
                out_feat = norm(out[:, -mask_num:, :].reshape(-1, out.shape[-1]))
                if name == 'depth':
                    pred_out = F.relu(pred(out_feat)).view(B * S, num_token // per_input, self.pred_num, mask_num // self.pred_num, -1)
                else:
                    pred_out = pred(out_feat).view(B * S, num_token // per_input, self.pred_num, mask_num // self.pred_num, -1)
                outputs[name] = pred_out

        torch.cuda.synchronize()
        return outputs



    def forward(self, image_primary, image_wrist, state, text_token, action=None, track_infos=None, action_label=None, mode = 'train'):  
        if self.training and self.phase == "pretrain":
            if self.obs_pred or self.depth_pred or self.trajectory_pred or self.dino_feat_pred or self.sam_feat_pred:
                this_num_obs_token = self.NUM_OBS_TOKEN + self.NUM_DEPTH_TOKEN + self.NUM_TRAJ_TOKEN + self.NUM_DINO_TOKEN + self.NUM_SAM_TOKEN
            else:
                this_num_obs_token = 0
            
            self.attention_mask = nn.Parameter(generate_attention_mask(
                            K=self.sequence_length, 
                            num_A=1+1+self.NUM_RESAMPLER_QUERY*2+1*2, 
                            num_B=this_num_obs_token+self.action_pred_steps,
                            atten_goal=self.atten_goal,
                            atten_goal_state=self.atten_goal_state,
                            atten_only_obs=self.atten_only_obs,
                            attn_robot_proprio_state = self.attn_robot_proprio_state,
                            mask_l_obs_ratio=self.mask_l_obs_ratio,
                            num_obs_token=this_num_obs_token,
                            action_pred_steps=self.action_pred_steps,
                            # num_obs=self.NUM_OBS_TOKEN,
                            # num_depth=self.NUM_DEPTH_TOKEN,
                            # num_traj=self.NUM_TRAJ_TOKEN
                            ).to(self.device), 
                            requires_grad=False)
        B, S, _ = state.shape
        device = image_primary.device
        S_AND_FUTURE = image_primary.shape[1]
        image_pred = None
        depth_pred = None
        traj_pred = None
        dino_pred = None
        sam_pred = None
        
        arm_pred_action, gripper_pred_action = None, None 
        arm_pred_state, gripper_pred_state = None, None
        loss_arm_action = None
        
        # text embedding
        with torch.no_grad():
            if next(self.clip_model.parameters()).type() == 'torch.cuda.BFloat16Tensor':
                with autocast(dtype=torch.bfloat16):
                    text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
            else:
                text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
            text_feature = text_feature.type(state.type())
            # text_feature = self.clip_model.encode_text(text_token.flatten(0, 1))
            # text_feature = text_feature.type(state.type())
        text_embedding = self.text_projector(text_feature)
        text_embedding = text_embedding.view(B, S, -1, self.hidden_dim) 

        # state embedding
        state = state.flatten(0, 1)
        arm_state_feature = self.arm_state_encoder(state[:, :6])
        if not self.gripper_width:
            gripper_state_one_hot = torch.nn.functional.one_hot(torch.where(state[:, 6:].flatten() < 1, torch.tensor(0).to(device), torch.tensor(1).to(device)), num_classes=2)
            gripper_state_feature = self.gripper_state_encoder(gripper_state_one_hot.type_as(state))
        else:
            gripper_state_feature = self.gripper_state_encoder(state[:, 6:])
        state_embedding = self.state_projector(torch.cat((arm_state_feature, gripper_state_feature), dim=1))
        state_embedding = state_embedding.view(B, S, -1, self.hidden_dim) 

        # image feature 
        if image_primary.type() != self.vision_encoder_type:
            image_primary = image_primary.type(self.vision_encoder_type)
            image_wrist = image_wrist.type(self.vision_encoder_type)
        with torch.no_grad():
            if not self.use_dinosiglip:
                image_primary_feature, _, _ = self.vision_encoder.forward_encoder(image_primary.flatten(0, 1), mask_ratio=0.0)
                image_wrist_feature, _, _ = self.vision_encoder.forward_encoder(image_wrist.flatten(0, 1), mask_ratio=0.0)
            else:
                dino_primary_feature = self.dino_featurizer(image_primary.flatten(0, 1), return_prefix_tokens=True)[0]
                dino_primary_cls_token = dino_primary_feature[1][:, 0]
                dino_primary_patches = dino_primary_feature[0]
                
                siglip_primary_features = self.siglip_featurizer(image_primary.flatten(0, 1))[0]
                siglip_primary_patches = siglip_primary_features
                
                image_primary_feature = torch.cat([dino_primary_patches, siglip_primary_patches], dim=2)
                image_primary_cls_token = dino_primary_cls_token
                
                
                
                dino_wrist_feature= self.dino_featurizer(image_wrist.flatten(0, 1), return_prefix_tokens=True)[0]
                dino_wrist_cls_token = dino_wrist_feature[1][:, 0]
                dino_wrist_patches = dino_wrist_feature[0]
                
                
                siglip_primary_features = self.siglip_featurizer(image_wrist.flatten(0, 1))[0]
                siglip_wrist_patches = siglip_primary_features
                
                
                image_wrist_feature = torch.cat([dino_wrist_patches, siglip_wrist_patches], dim=2)
                image_wrist_cls_token = dino_wrist_cls_token

        if image_primary_feature.type() != self.perceiver_resampler_type:
            image_primary_feature = image_primary_feature.type(self.perceiver_resampler_type)
            image_wrist_feature = image_wrist_feature.type(self.perceiver_resampler_type)
        
        
        image_primary_feature = image_primary_feature.view(B, S_AND_FUTURE, image_primary_feature.shape[-2], image_primary_feature.shape[-1])
        image_wrist_feature = image_wrist_feature.view(B, S_AND_FUTURE, image_wrist_feature.shape[-2], image_wrist_feature.shape[-1])

        # perceiver resampler
        if not self.use_dinosiglip:
            image_primary_cls_token = image_primary_feature[:, :, :1, :]
            image_wrist_cls_token = image_wrist_feature[:, :, :1, :]
            image_primary_feature = image_primary_feature[:, :, 1:, :]
            image_wrist_feature = image_wrist_feature[:, :, 1:, :]
            label_image_primary_feature = image_primary_feature.clone()
            label_image_wrist_feature = image_wrist_feature.clone()
            image_primary_feature = self.perceiver_resampler(image_primary_feature.reshape(B*S, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))  # mae vit outputs 196 tokens
            image_wrist_feature = self.perceiver_resampler(image_wrist_feature.reshape(B*S, 196, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
            image_primary_embedding = self.image_primary_projector(image_primary_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            image_wrist_embedding = self.image_wrist_projector(image_wrist_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            image_embedding = torch.cat((image_primary_embedding, image_wrist_embedding), dim=2)
            
            image_cls_token_primary_embedding = self.cls_token_primary_projector(image_primary_cls_token.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            image_cls_token_wrist_embedding = self.cls_token_wrist_projector(image_wrist_cls_token.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            image_cls_token_embedding = torch.cat((image_cls_token_primary_embedding, image_cls_token_wrist_embedding), dim=2)
        else:

            image_primary_feature = self.perceiver_resampler(image_primary_feature.reshape(B*S, 256, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))  # mae vit outputs 196 tokens
            image_wrist_feature = self.perceiver_resampler(image_wrist_feature.reshape(B*S, 256, self.RESAMPLER_hidden_dim).unsqueeze(1).unsqueeze(1))
            image_primary_embedding = self.image_primary_projector(image_primary_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            image_wrist_embedding = self.image_wrist_projector(image_wrist_feature.flatten(0, 2)).view(B, S, -1, self.hidden_dim)
            image_embedding = torch.cat((image_primary_embedding, image_wrist_embedding), dim=2)
            
            image_primary_cls_token = image_primary_cls_token.type(self.perceiver_resampler_type)
            image_wrist_cls_token = image_wrist_cls_token.type(self.perceiver_resampler_type)
            image_cls_token_primary_embedding = self.cls_token_primary_projector(image_primary_cls_token).view(B, S, -1, self.hidden_dim)
            image_cls_token_wrist_embedding = self.cls_token_wrist_projector(image_wrist_cls_token).view(B, S, -1, self.hidden_dim)
            image_cls_token_embedding = torch.cat((image_cls_token_primary_embedding, image_cls_token_wrist_embedding), dim=2)
        # aggregate embeddings and add timestep position encoding
        embeddings = torch.cat((text_embedding, state_embedding, image_embedding, image_cls_token_embedding), dim=2)
        pred_token_start_idx = embeddings.shape[2]
        transformer_input_list = [embeddings]
        if self.obs_pred:
            transformer_input_list.append(self.obs_tokens.repeat(B, S, 1, 1))
        if not self.share_query:
            if self.depth_pred:
                transformer_input_list.append(self.depth_tokens.repeat(B, S, 1, 1))
            if self.dino_feat_pred:
                transformer_input_list.append(self.dino_feat_tokens.repeat(B, S, 1, 1))
            if self.sam_feat_pred:
                transformer_input_list.append(self.sam_feat_tokens.repeat(B, S, 1, 1))
            if self.trajectory_pred:
                transformer_input_list.append(self.trajectory_tokens.repeat(B, S, 1, 1))
        if self.action_pred_steps > 0:
            transformer_input_list.append(self.action_pred_token.repeat(B, S, 1, 1))
                
                
        transformer_input = torch.cat(transformer_input_list, dim=2)  
        transformer_input = transformer_input + self.transformer_backbone_position_embedding.repeat(B, 1, transformer_input.shape[-2], 1)
        transformer_input = transformer_input.flatten(1, 2)

        # causal transformer forward
        if transformer_input.type() != self.transformer_backbone_type:
            transformer_input = transformer_input.type(self.transformer_backbone_type)
        
        if self.transformer_backbone_type == 'torch.cuda.BFloat16Tensor':
            with autocast(dtype=torch.bfloat16):
                transformer_input = self.embedding_layer_norm(transformer_input).to(dtype=torch.bfloat16)
                if self.attn_implementation == 'sdpa':
                    mask_4d = (
                        self.attention_mask               # (550, 550)
                        .unsqueeze(0)           # → (1, 550, 550)   插入 batch 维
                        .unsqueeze(0)           # → (1, 1, 550, 550) 插入 head 维 (这里先设为1)
                        .expand(transformer_input.shape[0], -1, -1, -1) # → (bs, 1, 550, 550) 复制到所有样本
                    ).contiguous()
                    transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=mask_4d)
                else:
                    transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask)
        else:
            transformer_input = self.embedding_layer_norm(transformer_input)
            if self.attn_implementation == 'sdpa':
                mask_4d = (
                    self.attention_mask               # (550, 550)
                    .unsqueeze(0)           # → (1, 550, 550)   插入 batch 维
                    .unsqueeze(0)           # → (1, 1, 550, 550) 插入 head 维 (这里先设为1)
                    .expand(transformer_input.shape[0], -1, -1, -1) # → (bs, 1, 550, 550) 复制到所有样本
                ).contiguous()
                transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=mask_4d)
            else:
                transformer_output = self.transformer_backbone(inputs_embeds=transformer_input, attention_mask=self.attention_mask)
        transformer_output = transformer_output.view(B, S, -1, self.hidden_dim)

  
        decoder_outputs = self.run_multi_decoder_parallel(
            transformer_output, pred_token_start_idx, B, S, mode
        )

        image_pred = decoder_outputs.get("image", None)
        depth_pred = decoder_outputs.get("depth", None)
        dino_pred  = decoder_outputs.get("dino", None)
        sam_pred   = decoder_outputs.get("sam", None)
        
        if self.action_pred_steps > 0:
            if self.share_query:
                this_num_obs_token = self.NUM_OBS_TOKEN
            elif self.obs_pred or self.depth_pred or self.trajectory_pred or self.dino_feat_pred or self.sam_feat_pred:
                this_num_obs_token = self.NUM_OBS_TOKEN + self.NUM_DEPTH_TOKEN + self.NUM_TRAJ_TOKEN + self.NUM_DINO_TOKEN + self.NUM_SAM_TOKEN
            else:
                this_num_obs_token = 0
            action_pred_feature = transformer_output[:, :, pred_token_start_idx+this_num_obs_token:pred_token_start_idx+this_num_obs_token+self.action_pred_steps, :]
            if not self.use_dit_head:
                action_pred_feature = self.action_decoder(action_pred_feature)
                arm_pred_action = self.arm_action_decoder(action_pred_feature)
                gripper_pred_action = self.gripper_action_decoder(action_pred_feature)
            elif self.use_dit_head and mode=='train':
                action_pred_feature = action_pred_feature[:, :self.sequence_length-self.atten_goal].flatten(0, 1)
                action_labels = action_label.flatten(0, 1)
                repeated_diffusion_steps = 8
                actions_repeated = action_labels.repeat(repeated_diffusion_steps, 1, 1)
                cognition_features_repeated = action_pred_feature.repeat(repeated_diffusion_steps, 1, 1) # [r
                arm_pred_action = self.action_model.loss(actions_repeated, cognition_features_repeated)
                gripper_pred_action = arm_pred_action
            elif self.use_dit_head and mode=='test':
                repeated_diffusion_steps = 8
                bs = image_primary.flatten(0,1).shape[0]
                # cognition_features = action_logits.unsqueeze(1).to(self.model.dtype)  # [B, 1, D]
                action_pred_feature = action_pred_feature.flatten(0, 1)
                cognition_features = action_pred_feature
                cfg_scale = 1.5
                # Sample random noise
                
                noise = torch.randn(bs, 3, self.action_model.in_channels, device=cognition_features.device).to(cognition_features.dtype)  #[B, T, D]
                using_cfg = cfg_scale > 1.0
                use_ddim = True
                num_ddim_steps = 10
                # Setup classifier-free guidance:
                if using_cfg:
                    noise = torch.cat([noise, noise], 0)
                    uncondition = self.action_model.net.z_embedder.uncondition
                    uncondition = uncondition.unsqueeze(0)  #[1, D]
                    uncondition = uncondition.expand(bs, 3, -1) #[B, 1, D]
                    z = torch.cat([cognition_features, uncondition], 0)
                    cfg_scale = cfg_scale
                    model_kwargs = dict(z=z, cfg_scale=cfg_scale)
                    sample_fn = self.action_model.net.forward_with_cfg
                else:
                    model_kwargs = dict(z=cognition_features)
                    sample_fn = self.action_model.net.forward
                
                # DDIM Sampling
                if use_ddim and num_ddim_steps is not None:
                    if self.action_model.ddim_diffusion is None:
                        self.action_model.create_ddim(ddim_step=num_ddim_steps)
                    samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                        noise.shape, 
                                                                        noise, 
                                                                        clip_denoised=False,
                                                                        model_kwargs=model_kwargs,
                                                                        progress=False,
                                                                        device=action_pred_feature.device,
                                                                        eta=0.0
                                                                        )
                else:
                    # DDPM Sampling
                    samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                            noise.shape, 
                                                                            noise, 
                                                                            clip_denoised=False,
                                                                            model_kwargs=model_kwargs,
                                                                            progress=False,
                                                                            device=self.model.device
                                                                            )
                # import pdb; pdb.set_trace()
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                # import pdb; pdb.set_trace()
                arm_pred_action,  gripper_pred_action= samples.unsqueeze(0)[...,:6], samples.unsqueeze(0)[...,6:]
                
                # ensemble
                
        return arm_pred_action, gripper_pred_action, image_pred, arm_pred_state, gripper_pred_state, loss_arm_action, depth_pred, traj_pred, dino_pred, sam_pred





def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

