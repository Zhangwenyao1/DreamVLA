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