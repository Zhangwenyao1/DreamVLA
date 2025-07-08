# Post-process
Self-collected data and pre-training datasets often exhibit certain challenges that can negatively affect model performance. Below, we outline these issues and provide standardized solutions for data formatting and post-processing.
* :warning: **Varied Action Labels:** 
Different embodiments, and sometimes even identical ones, may use diverse action labels such as:
    * **Absolute target joint positions** (qpos)
    * **Absolute target joint velocites** (qvel)
    * **Absolute end-effector poses** (ee-pose)
    * **Delta target end-effector poses** (delta ee-pose)
    Furthermore, rotation representations may vary, including quaternions, Euler angles, rotation vectors, and rotation matrices.
* :warning: **Jittering and Long Pauses:** Fresh data collectors often introduce hesitation, leading to long pauses or jittering during data collection. Without proper filtering, such data significantly degrades model performance.
* :warning: **Quick Gripper Open/Close Actions:** A frequency mismatch between camera capture and gripper control often results in abrupt changes in gripper states, especially during grasping or releasing motions.

To address these issues, we recommend a uniform, clear, and effective format for saving self-collected data and provide tools for post-processing.

## :exclamation: Data Format
For each task, we collect 100 demos. The recommended directory structure is:
```
0000 (exp_id)
|—— 000000 (episode_id)
    |—— steps 
        |—— 0000 (timestep_id, start)
            |—— image_primary.jpg (Eye-on-Base camera rgb image)
            |—— image_wrist.jpg (Eye-on-Hand camera rgb image)
            └── other.npz (robot state, language, action)
        |—— ......
        └── xxxx (timestep_id, end)
|—— 000001 (episode_id)
    |—— steps
        |—— ......
|—— ......
└── 000099 (episode_id)
    |—— steps
        |—— ......
```
### File Details:
* **image_primary.jpg** and **image_wrist.jpg**: Images saved with a resolution of 640 x 480 pixels.
* **other.npz**: Contains key robot metadata. An example of the saved format is:
```python
# at each timestep i
npz_path = f"other.npz"

# absolute current gripper pose in robot space, position + euler angles, the unit is m and rad.
gripper_pose = np.array([x, y, z, euler_x, euler_y, euler_z])

# absolute current gripper open state
gripper_open_state = np.array([1.0]) if gripper is opened else np.array([-1.0]) 

# absolute current joints position (qpos)
joints = np.array([q0, q1, q2, q3, q4, q5, q6])

# language instruction
language_instruction = f"Pick the apple." 

# absolute target pose action label (target_gripper_open_or_close is 1.0 if targetting open, else -1.0)
action_gripper_pose = np.array([target_x, target_y, target_z, target_euler_x, target_euler_y, target_euler_z, target_gripper_open_or_close])

# delta pose action label 
delta_cur_2_last_action = np.array([target_delta_x, target_delta_y, target_delta_z, target_delta_euler_x, target_delta_euler_y, target_delta_euler_z, target_gripper_open_or_close])

# save npz
np.savez_compressed(
    npz_path,
    joints=joints,
    gripper_pose=gripper_pose,
    gripper_open_state=gripper_open_state,
    action_gripper_pose=action_gripper_pose,
    delta_cur_2_last_action=delta_cur_2_last_action,
    language_instruction=language_instruction,
)
```
For most robotic systems, all metadata except delta_cur_2_last_action can be directly extracted. We provide a helper function to compute the delta pose action label in the [script](../utils/real_ft_data.py):
```python
compute_delta_action(
    data_list,
)
```

## :star: Post-processing Self-collected Data
* **Filtering Jitter and Pauses:** To filter out jittering and long pauses, use the following function in the [script](../utils/real_ft_data.py):
```python
filter_real_data(
    exp_id, 
    root_path, # path to your raw data 
    save_data_path, # a desired path to save filterd data
    save_gif_path # a desired path to save the filtered gif (only for visualization and debugging)
)
```
* **Data Augmentation for Gripper Actions:** To augment data by increasing sampling ratios during gripper open/close events, use the following function in the same [script](../utils/real_ft_data.py):
```python
make_aug_short_real_dataset_info(
    root_path, # path to your filterd data 
    root_info_path, # path to your data info, it should be like xxx/Seer/data_info
    dataset_name, # your dataset name, e.g. "ft"
    select_ratio=1.0,
    sequence_length=7, 
    action_pred_steps=3, 
    replicate_steps=10
)
```

