import json
import argparse
import os
import sys
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input-models-path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--radius", type=float, default=1.2)
parser.add_argument("--num-views", type=int, default=50)
parser.add_argument("--seed", type=int)
parser.add_argument("--engine", type=str, default="cycles")
parser.add_argument("--light-energy", type=float, default=10)
parser.add_argument("--no-depth", action="store_true")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=0)
parser.add_argument("--use-gpu", type=int, default=1)
parser.add_argument("--omit", type=int, default=1)
args = parser.parse_args()

sample_dir_list = os.listdir(args.input_models_path)
sample_dir_list.sort()
# with open(args.input_models_path, "r") as f:
#     model_paths = json.load(f)
    #model_paths.sort()

cmds = []
uids = []
script_file = "scripts/shapenet_ortho.py"

for name in sample_dir_list:
    sample_dir = os.path.join(args.input_models_path, name)
    if os.path.exists(os.path.join(sample_dir, 'model_normalized.obj')):
        path = os.path.join(sample_dir, 'model_normalized.obj')
    else:
        path = os.path.join(sample_dir, "models", "model_normalized.obj")

    command = f'blenderproc run {script_file} --object-path {path} --output_dir {args.output_dir} --uid {name} --num-views {args.num_views} --resolution {args.resolution} --radius {args.radius} --scale {args.scale} --use-gpu {args.use_gpu}'
    cmds.append(command)
    #cmds.append(item)
    uids.append(name)

print(f'total mount of objs: {len(cmds)}')

if args.end == 0:
    args.end = len(cmds)
for i in range(args.start, args.end):
    if args.omit:
    #print(f'rendering {i} / {len(cmds)} images!, {cmds[i]}')
        if os.path.exists(f'{args.output_dir}/views_{uids[i]}'):
            os.system(f'echo already rendered {i} / {len(cmds)}')
            os.system(f'echo fail to render {i} ') #>> /yulin/log_already.txt')
            continue

    try:
        ret = os.system(cmds[i])
        if ret == 2:
            print("KeyboardInterrupt")
            break
        # elif ret != 0:
        #     print("Non-zero return", ret)
        os.system(f'echo rendering {i} / {len(cmds)} {uids[i]} ') #>> /yulin/loglog.tx')
    except: 
        info=sys.exc_info() 
        print(info[0],":",info[1] )

        os.system("echo 'fail to render {i}' > /yulin/objaverse.log".format(i))

