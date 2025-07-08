from real_controller.controller import SeerController
import time 
import numpy as np
from scipy.spatial.transform import Rotation as R


def _6d_to_pose(
    pose6d,
    degrees=False
):
    pose = np.eye(4)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()

    return pose

def pose_to_6d(
    pose, 
    degrees=False
):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)

    return pose6d

### pseudocode example to deploy dreamvla ###

# Recommend control frequency: 15Hz, same as Droid
control_freq = 15 # Hz
max_rel_pos = 0.02 # magic number, same as training
max_rel_orn = 0.05 # magic number, same as training

# set a controller
controller = SeerController()
last2robot_pose = env.get_robot_state()["pose"] # absolute 4x4 pose matrix in robot space
# warm up
for i in range(3):
    obs = {}
    obs["robot_state"] = env.get_robot_state()
    obs["color_image"] = env.get_color_images()
    target_pos, target_euler, target_gripper, _ = controller.forward(obs, include_info=True)

while True:
    # at each time step t
    torch.cuda.synchronize()
    t1 = time.time()

    obs["robot_state"] = env.get_robot_state()
    obs["color_image"] = env.get_color_images()
    target_pos, target_euler, target_gripper, _ = controller.forward(obs, include_info=True)

    # delta-action-2-absolute-action
    target_pos *= self.max_rel_pos 
    target_euler *= self.max_rel_orn
    cur2last_pose = _6d_to_pose(np.concatenate([target_pos, target_euler]))
    last2robot_pose = last2robot_pose @ cur2last_pose 
    target_pose = pose_to_6d(last2robot_pose)

    torch.cuda.synchronize()
    t2 = time.time()
    sleep_left = 1. / control_freq - (t2 - t1)

    if sleep_left > 0:
        time.sleep(sleep_left)
    
    env.step(target_pose, target_gripper)

### pseudocode example to deploy dreamvla ###





