from collections import defaultdict, namedtuple
import logging
import os, json, random
from pathlib import Path
import sys
import time
import PIL.Image as Image
import copy
from collections import deque
from moviepy.editor import ImageSequenceClip
from calvin.calvin_models.calvin_agent.models.calvin_base_model import CalvinBaseModel
import time
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin.calvin_models.calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_env_state_for_initial_condition,
    get_log_dir,
    print_and_save,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from termcolor import colored
import torch
from tqdm.auto import tqdm
from calvin_env.envs.play_table_env import get_env
from utils.data_utils import preprocess_image, preprocess_text_calvin
import functools
from utils.train_utils import get_cast_dtype
import cv2


os.environ['PYOPENGL_PLATFORM'] = 'egl'
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 1000

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    return env

class ModelWrapper(CalvinBaseModel):
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10, 
                calvin_eval_max_steps=360, action_pred_steps=3):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.use_diff = False
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.calvin_eval_max_steps = calvin_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len-1)
        self.ensemble_action = False
        if self.ensemble_action:
            from models.action_ensemble import AdaptiveEnsembler
            self.action_ensembler = AdaptiveEnsembler(pred_action_horizon=self.action_pred_steps, adaptive_ensemble_alpha=0.1)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len-1)
        if self.ensemble_action:
            self.action_ensembler.reset()
    def step(self, obs, goal, timestep):
        image = obs["rgb_obs"]['rgb_static']
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["rgb_obs"]['rgb_gripper']
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)

        state = obs['robot_obs']
        state = torch.from_numpy(np.stack([state]))
        state = state.unsqueeze(1).to(dtype=self.cast_type)
        state = torch.cat([state[..., :6], state[..., [-1]]], dim=-1)

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)
            self.img_queue.append(image_x)  
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)
            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)
            num_step = image_primary.shape[1]
            if num_step < self.history_len:  
                input_image_primary = torch.cat([image_primary, image_primary[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat([image_wrist, image_wrist[:, -1].repeat(1, self.history_len-num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len-num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state
            arm_action, gripper_action, image_pred, arm_pred_state, gripper_pred_state, _ , depth_pred, trajectory_pred, dino_pred, sam_pred= self.model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=torch.zeros(1, self.history_len, 7).to(input_state.device),
                mode="test",
            )
            action = torch.concat((arm_action[0, :, 0, :], gripper_action[0, :, 0, :] > 0.5), dim=-1)
            action[:, -1] = (action[:, -1] - 0.5) * 2  # scale to -1 or 1
            action = action.cpu().detach().to(dtype=torch.float16).numpy()
            if self.ensemble_action:
                action = self.action_ensembler.ensemble_action(action)
                action[-1] = 1 if action[-1] > 0 else -1
            else:
                if num_step < self.history_len:
                    action = action[num_step - 1]
                else:
                    action = action[-1]
                
        return action

def evaluate_policy_ddp(model, env, epoch, calvin_conf_path, eval_log_dir=None, debug=False, create_plan_tsne=False, reset=False, diverse_inst=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    
    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open('./utils/lang_annotation_cache.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open('./utils/eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    assert NUM_SEQUENCES % device_num == 0
    interval_len = int(NUM_SEQUENCES // device_num)
    eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, NUM_SEQUENCES)]
    results = []
    plans = defaultdict(list)
    local_sequence_i = 0
    base_sequence_i = device_id * interval_len

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir, base_sequence_i+local_sequence_i, reset=reset, diverse_inst=diverse_inst)
        results.append(result)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
        )
        local_sequence_i += 1
    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]
    
    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = extract_iter_from_tqdm(eval_sequences)

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequences)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_list = [_[0] for _ in res_tup_list]
        eval_seq_list = [_[1] for _ in res_tup_list]
        print_and_save(res_list, eval_seq_list, eval_log_dir, epoch)

    return results

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug, eval_log_dir='', sequence_i=-1, reset=False, diverse_inst=False):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0

    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst, robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug, eval_log_dir, subtask_i, sequence_i, diverse_inst=diverse_inst)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, eval_log_dir='', subtask_i=-1, sequence_i=-1, robot_obs=None, scene_obs=None, diverse_inst=False):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    planned_actions = []
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split('\n')[0]
    if '\u2019' in lang_annotation:
        lang_annotation.replace('\u2019', '\'')
    model.reset()
    start_info = env.get_info()
    # img_list = [] 
    # print(f"{subtask} ", end="")
    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation, step)
        if len(planned_actions) == 0:
            if action.shape == (7,):
                planned_actions.append(action)
            else:
                planned_actions.extend([action[i] for i in range(action.shape[0])])
        action = planned_actions.pop(0)
        obs, _, _, current_info = env.step(action)
        # img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
        # img_list.append(img_copy)
        if step == 0:
            collect_plan(model, plans, subtask)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            
            # print(colored("success", "green"), end=" ")
            # clip = ImageSequenceClip(img_list, fps=30)
            # clip.write_gif(
            #     os.path.join(
            #         eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"
            #     ),
            #     fps=30,
            # )

            return True
    # print(colored("fail", "red"), end=" ")
    # clip = ImageSequenceClip(img_list, fps=30)
    # clip.write_gif(
    #     os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"),
    #     fps=30,
    # )
    return False

def eval_one_epoch_calvin_ddp(args, model, dataset_path, image_processor, tokenizer, eval_log_dir=None, debug=False, future_act_len=-1, reset=False, diverse_inst=False):
    env = make_env(dataset_path)
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    wrapped_model = ModelWrapper(
                        model, 
                        tokenizer, 
                        image_processor, 
                        cast_dtype, 
                        history_len=hist_len, 
                        calvin_eval_max_steps=EP_LEN,
                        action_pred_steps = args.action_pred_steps)
    evaluate_policy_ddp(wrapped_model, env, 0, args.calvin_conf_path, eval_log_dir=eval_log_dir, debug=debug, reset=reset, diverse_inst=diverse_inst)
