# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import re
import random
import uuid
from pathlib import Path
import train
from utils.arguments_utils import get_parser
import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for DINO", parents=[get_parser()])
    print("!!!")
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=4, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72000, type=int, help="Duration of the job")
    parser.add_argument("--partition", default="mozi-S1", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument("--exclude", default="", type=str, help="Nodes to exclude")
    parser.add_argument("--output_dir", default="/mnt/petrelfs/tianyang/Code/ICLR_Manipulation/out", type=str)
    return parser.parse_args()

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path(f"/ailab/user/{user}/").is_dir():
        p = Path(f"/ailab/user/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def _get_master_port(seed):
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)
    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)
    return int(master_port_str)

def _parse_slurm_node_list(s):
    nodes = []
    # Extract "hostname", "hostname[1-2,3,4-5]," substrings
    p = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for m in p.finditer(s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        prefix_list = prefix.split(',')
        if len(prefix_list) > 1:
            nodes += prefix_list[:-1]
            prefix = prefix_list[-1]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                for i in range(start, end):
                    nodes.append(prefix + f"{i:0{width}}")

    return nodes

class Trainer(object):
    def __init__(self, args):
        self.args = args
    def __call__(self):
        # import run_beit_pretraining
        self._setup_gpu_args()
        train.main(self.args)

    def checkpoint(self):
        import os
        import submitit
        # self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path
        job_id = int(os.environ["SLURM_JOB_ID"])
        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        print("node_list :", os.environ["SLURM_JOB_NODELIST"])
        nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])
        print("node_count :", node_count)
        print("nodes :", nodes)
        assert len(nodes) == node_count
        master_addr = nodes[0]
        master_port = _get_master_port(seed=job_id)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

def main():
    args = parse_args()
    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.SlurmExecutor(folder=args.output_dir, max_num_timeout=30)
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment
    if args.exclude:
        kwargs["exclude"] = args.exclude
    executor.update_parameters(
        gres=f"gpu:{num_gpus_per_node}",
        ntasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=6,
        nodes=nodes,
        time=timeout_min,
        # Below are cluster dependent parameters
        signal_delay_s=120,
        partition=partition,
        **kwargs
    )
    executor.update_parameters(job_name="seer")
    # args.dist_url = get_init_file().as_uri()
    trainer = Trainer(args)
    job = executor.submit(trainer)
    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")

if __name__ == "__main__":
    main()