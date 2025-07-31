import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['MUJOCO_GL'] = 'osmesa'

from pathlib import Path
import copy
import io
import distutils.dir_util
import numpy as np
import time
import torch
from torch.distributed import gather
from collections import deque
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.data_utils import preprocess_image, preprocess_text_calvin
from utils.train_utils import get_cast_dtype

# libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from pdb import set_trace 

def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)
    
    return euler

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

class ModelWrapper:
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10, 
                use_ensembling=False, ensembling_temp=0.01, libero_eval_max_steps=600, action_pred_steps=3, gripper_width=False):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.libero_eval_max_steps = libero_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)
        self.cnt = 0
        self.gripper_width = gripper_width
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)
        self.gripper_state = np.array([-1.0])
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                    [
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        7,
                    ]
                ).to(self.device)

        self.cnt += 1

    def step(self, obs, goal, timestep):
        # preprocess image
        image = obs["agentview_image"][::-1]
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["robot0_eye_in_hand_image"]
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type) 

        # expand text dimension
        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])

        if not self.gripper_width:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, self.gripper_state])).to(dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        else:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']])).to(dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 8]

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_queue.append(image_x)  # TODO find out how the policy completes the 5 sub-tasks. the obs of the later task will be appended after the former?
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  # the instruction does not change
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)
            
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)

            num_step = image_primary.shape[1]
            if num_step < self.history_len:  # padding
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state
            
            arm_action, gripper_action, image_pred, arm_pred_state, gripper_pred_state, _ , depth_pred, trajectory_pred, dino_pred, sam_pred = self.model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7).to(input_state.device),
                mode='test'
            )

            # This need to align libero environment 
            if self.use_ensembling:
                if num_step < self.history_len:
                    selected_step = num_step - 1
                else:
                    selected_step = -1
                action = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1) # (1, action_pred_steps, 7)
                self.all_time_actions[timestep:timestep+1,timestep:timestep+self.action_pred_steps] = action 
                actions_for_curr_step = self.all_time_actions[:, timestep]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = self.ensembling_temp
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
                action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
                action = action.detach().cpu().numpy()[-1]

        self.gripper_state = np.array([action[-1]])
        return action

def evaluate_libero_task(task, env, obs, args, model):
    steps = 0
    success = 0
    model.reset()
    goal = task.language
    # video = []
    with torch.no_grad():
        while steps < args.libero_eval_max_steps: # default
            action = model.step(obs, goal, steps) 
            steps += 1
            
            obs, reward, done, info = env.step(action)
            
            # video.append(Image.fromarray(obs['agentview_image'][::-1]))
            if done:
                success = 1
                break 
    # import imageio
    # imageio.mimsave("demo.gif", video, 'GIF', duration=0.1)
    env.close()
    return success

def evaluate_policy_ddp(args, model):
    pass 
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    results = []
    if "libero" in args.finetune_type:
        # if args.finetune_type == "libero_10":
        global num_eval_episodes 
        global task_num
        num_eval_episodes = 50
        task_num = 10
            
        NUM_SEQUENCES = num_eval_episodes * task_num 
        eval_sequences = list(range(NUM_SEQUENCES))
        # assert NUM_SEQUENCES % device_num == 0
        # interval_len = int(NUM_SEQUENCES // device_num)
        import math
        interval_len = math.ceil(NUM_SEQUENCES / device_num)
        eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
        
        eval_sequences = tqdm(eval_sequences)
        # else:
        #     raise NotImplementedError
    else:
        raise NotImplementedError
    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes 
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files", task.problem_folder, task.bddl_file)
        env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": args.libero_img_size,
        "camera_widths": args.libero_img_size,
        "render_gpu_device_id":device_id
        }
        print("device_id :", device_id)
        env = OffScreenRenderEnv(**env_args)
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = args.finetune_type
        env.reset()
        env.seed(args.seed)

        # set initial state
        init_states_path = os.path.join(
            f"{args.libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros(7))

        result = evaluate_libero_task(task, env, obs, args, model)
        results.append(result) 
        print("rank", torch.distributed.get_rank(), "results :", results)
    
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    eval_sequences = extract_iter_from_tqdm(eval_sequences)
    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list.sort(key=lambda x: x[1])
        print_and_save(res_tup_list, task_suite)

def print_and_save(result_list, task_suite):
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        print("this_result_list :", this_result_list)
        this_result_list = np.array(this_result_list)
        avg_success = np.mean(this_result_list, axis=0)[0]
        task = task_suite.get_task(j)
        task_name = task.name
        print(f"Success rates for task {j} {task_name}:")
        print(f"{avg_success * 100:.1f}%")

def eval_one_epoch_libero_ddp(args, model, image_processor, tokenizer):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        use_ensembling=args.eval_libero_ensembling,
                        ensembling_temp=args.ensembling_temp,
                        libero_eval_max_steps=args.libero_eval_max_steps,
                        action_pred_steps = args.action_pred_steps,
                        gripper_width=args.gripper_width)
    evaluate_policy_ddp(args, wrapped_model)