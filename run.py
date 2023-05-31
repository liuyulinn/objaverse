import json
import argparse
import os
import sys
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
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)
args = parser.parse_args()


with open(args.input_models_path, "r") as f:
    model_paths = json.load(f)
    #model_paths.sort()

cmds = []
uids = []
script_file = os.path.join(os.path.dirname(__file__), "objaverse1.py")

for item in model_paths:
    group = item["group"]
    uid = item["uid"]
    path = os.path.join('data', group, f'{uid}.glb')

    command = f'blenderproc run {script_file} --object-path {path} --num-views {args.num_views} --resolution {args.resolution}'
    cmds.append(command)
    uids.append(uid)

print(f'total mount of objs: {len(cmds)}')

if args.end == 0:
    args.end = len(cmds)
for i in range(args.start, args.end):
    print(f'rendering {i} / {len(cmds)} images!, {cmds[i]}')
    if os.path.exist(f'/yulin/objaverse/views_{uids[i]}'):
        continue

    try:
        ret = os.system(cmds[i])
        if ret == 2:
            print("KeyboardInterrupt")
            break
        elif ret != 0:
            print("Non-zero return", ret)
    except: 
        info=sys.exc_info() 
        print(info[0],":",info[1] )

        os.system("echo 'fail to render {i}' > /yulin/objaverse.log".format(i))

