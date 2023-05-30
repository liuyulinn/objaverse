import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import os
import boto3
import tyro
import argparse
#import wandb


@dataclass
class Args:
    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    workers_per_cpu: int = 1
    """number of workers per gpu"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_cpus: int = -1
    """number of gpus to use. -1 means all available cpus"""

    start: int = 0
    '''continue to render images'''


# def worker(
#     queue: multiprocessing.JoinableQueue,
#     count: multiprocessing.Value,
#     cpu: int,
# ) -> None:
#     while True:
#         item = queue.get()
#         if item is None:
#             break

#         # Perform some operation on the item
#         # group = item["group"]
#         # uid = item["uid"]
#         # path = os.path.join('data', group, f'{uid}.glb')

#         print(item, cpu)
#         # command = (
#         #     f"export DISPLAY=:0.{gpu} &&"
#         #     f" blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
#         #     f" --object_path {item}"
#         # )
#         # command = (
#         #     f"export DISPLAY=:0.{gpu} &&"
#         #     f" blenderproc run objaverse1.py --"
#         #     f" --object_path {path}"
#         # )
#         subprocess.run(item, shell=True)

#         # if args.upload_to_s3:
#         #     if item.startswith("http"):
#         #         uid = item.split("/")[-1].split(".")[0]
#         #         for f in glob.glob(f"views/{uid}/*"):
#         #             s3.upload_file(
#         #                 f, "objaverse-images", f"{uid}/{f.split('/')[-1]}"
#         #             )
#         #     # remove the views/uid directory
#         #     shutil.rmtree(f"views/{uid}")

#         with count.get_lock():
#             count.value += 1

#         queue.task_done()

def worker(
    i, item
) -> None:
    # while True:
    #     #item = queue.get()
    if item is None:
        return 

    print(f'rendering {i}/46205 images', f'name is {item}')
    subprocess.run(item, shell=True)
        # with count.get_lock():
        #     count.value += 1

        # queue.task_done()

# parser = argparse.ArgumentParser()
# parser.add_argument("--input-models-path", type=str, required=True)
# # parser.add_argument("--output-dir", type=str, required=True)
# parser.add_argument("--resolution", type=int, default=256)
# parser.add_argument("--scale", type=float, default=1.0)
# parser.add_argument("--radius", type=float, default=2)
# parser.add_argument("--num-views", type=int, default=50)
# parser.add_argument("--seed", type=int)
# parser.add_argument("--engine", type=str, default="cycles")
# parser.add_argument("--light-energy", type=float, default=10)
# parser.add_argument("--no-depth", action="store_true")
# parser.add_argument("--start", type=int, default=0)
# args = parser.parse_args()

if __name__ == "__main__":
    args = tyro.cli(Args)

    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)

    cmds = []
    script_file = os.path.join(os.path.dirname(__file__), "objaverse1.py")

    for item in model_paths:
        group = item["group"]
        uid = item["uid"]
        path = os.path.join('data', group, f'{uid}.glb')

        command = f'blenderproc run {script_file} --object-path {path} --use-gpu 0'
        #command = f'blenderproc run {script_file} --object-path {item} --use-gpu 0'
        cmds.append(command)

    cores = multiprocessing.cpu_count()
    print(f'cores total: {cores}')
    pool = multiprocessing.Pool(processes=cores)
    pool_list = []
    #result_list = []

    for i in range(args.start, len(cmds)):
        pool_list.append(pool.apply_async(worker, (i, cmds[i])))
    # args = tyro.cli(Args)

    pool.close()
    pool.join()

    # s3 = boto3.client("s3") if args.upload_to_s3 else None
    # queue = multiprocessing.JoinableQueue()
    # count = multiprocessing.Value("i", 0)

    # if args.log_to_wandb:
    #     wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    #for cpu_i in range(args.num_cpus):
    # for worker_i in range(args.workers_per_cpu):
    #     worker_i = worker_i #cpu_i * args.workers_per_gpu + 
    #     process = multiprocessing.Process(
    #         target=worker, args=(queue, count, 0)
    #     )
    #     process.daemon = True
    #     process.start()

    # Add items to the queue
    # with open(args.input_models_path, "r") as f:
    #     model_paths = json.load(f)
    # for i in range(args.start, len(cmds)):
    #     queue.put(cmds[i])

    # update the wandb count
    # if args.log_to_wandb:
    #     while True:
    #         time.sleep(5)
    #         wandb.log(
    #             {
    #                 "count": count.value,
    #                 "total": len(model_paths),
    #                 "progress": count.value / len(model_paths),
    #             }
    #         )
    #         if count.value == len(model_paths):
    #             break

    # Wait for all tasks to be completed
    # queue.join()

    # # Add sentinels to the queue to stop the worker processes
    # for i in range(args.num_cpus * args.workers_per_cpu):
    #     queue.put(None)
