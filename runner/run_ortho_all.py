import json
import argparse
import os
import sys
import subprocess
import glob
import threading
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("--input-models-path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--num-views", type=int, default=50)
parser.add_argument("--seed", type=int)
parser.add_argument("--engine", type=str, default="cycles")
parser.add_argument("--light-energy", type=float, default=10)
parser.add_argument("--no-depth", type=int, default=1)
parser.add_argument("--no-normal", type=int, default=1)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)
parser.add_argument("--omit", type=int, default=1)
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--random_angle", type=int, default=0)
parser.add_argument("--use-gpu", type=int, default=1)
parser.add_argument("--ortho", type=int, default=1)
args = parser.parse_args()


def dump(local_path, remote_path):
    with zipfile.ZipFile(remote_path, "w") as fo:
        for f in glob.glob(os.path.join(local_path, "*.*")):
            fo.write(f, os.path.basename(f))

def dispatch_dump(local_path, remote_path):
    threading.Thread(target=dump, args=(local_path, remote_path)).start()


with open(args.input_models_path, "r") as f:
    model_paths = json.load(f)
    #model_paths.sort()

cmds = []
uids = []
saves = []
local_saves = []
if args.ortho:
    script_file = "scripts/objaverse_ortho.py"
else:
    script_file = "scripts/objaverse_all.py"


for item in model_paths:
    _, group, uid = model_paths[item].split('/')
    # group = item["group"]
    # uid = item["uid"]
    path = os.path.join('data', group, uid)
    # print(path)
    
    save_dir = os.path.abspath(f'{group}')
    local_saves.append(save_dir)
    command = f'blenderproc run {script_file} --object-path {path} --output_dir {save_dir}  --num-views {args.num_views} --resolution {args.resolution} --radius {args.radius} --scale {args.scale} --random {args.random} --use-gpu {args.use_gpu} --random_angle {args.random_angle} --no-depth {args.no_depth} --no-normal {args.no_normal}'
    # print("save_path", f'{args.output_dir}/{group}')
    cmds.append(command)

    #cmds.append(item)
    uids.append(uid)
    saves.append(os.path.join(args.output_dir, group))

print(f'total mount of objs: {len(cmds)}')

if args.end == 0:
    args.end = len(cmds)
for i in range(args.start, args.end):
    if args.omit:
        if os.path.exists(f'{saves[i]}/views_{uids[i].split(".")[0]}.zip'):
            print(f'already rendered {i} / {len(cmds)}', flush=True)
            continue

    try:
        ret = subprocess.call(cmds[i], stderr=subprocess.STDOUT, timeout=120)
        # ret = os.system(cmds[i])
        if ret == 2:
            print("KeyboardInterrupt", flush=True)
            break
        os.makedirs(saves[i], exist_ok=True)
        dispatch_dump(os.path.join(local_saves[i], f'views_{uids[i].split(".")[0]}'), os.path.join(saves[i], f'views_{uids[i].split(".")[0]}.zip'))
        # elif ret != 0:
        #     print("Non-zero return", ret)
        print(f'rendering {i} / {len(cmds)} {uids[i]} to {saves[i]}/views_{uids[i]}', flush=True) #>> /yulin/loglog.tx')
    except Exception as exc:
        print(uids[i], repr(exc), flush=True)
        os.system("echo 'fail to render {i}' > /yulin/objaverse.log".format(i=uids[i]))
