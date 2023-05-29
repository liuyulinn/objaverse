import json
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input-models-path", type=str, required=True)
# parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--radius", type=float, default=2)
parser.add_argument("--num-views", type=int, default=50)
parser.add_argument("--seed", type=int)
parser.add_argument("--engine", type=str, default="cycles")
parser.add_argument("--light-energy", type=float, default=10)
parser.add_argument("--no-depth", action="store_true")
args = parser.parse_args()


with open(args.input_models_path, "r") as f:
    model_paths = json.load(f)

cmds = []
script_file = os.path.join(os.path.dirname(__file__), "objaverse1.py")

for item in model_paths:
    group = item["group"]
    uid = item["uid"]
    path = os.path.join('data', group, f'{uid}.glb')

    command = f'blenderproc run {script_file} --object-path {path} --num-views {args.num_views} --resolution {args.resolution}'
    cmds.append(command)

print(f'total mount of objs: {len(cmds)}')

for i in range(len(cmds)):
    print(f'rendering {i} / {len(cmds)} images!')
    ret = os.system(cmds[i])
    if ret == 2:
        print("KeyboardInterrupt")
        break
    elif ret != 0:
        print("Non-zero return", ret)
