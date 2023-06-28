import json
import argparse
import os
import sys
import subprocess
import gzip
import numpy as np

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

data_root = args.input_models_path
render_data_dir = f'{data_root}/3dmodels/original'
listing_dir = f'{data_root}/listings/metadata'
model_meta = f'{data_root}/3dmodels/metadata/3dmodels.csv.gz'
# sample_dir_list.sort()
# with open(args.input_models_path, "r") as f:
#     model_paths = json.load(f)
    #model_paths.sort()

cmds = []
uids = []
script_file = "scripts/abo_ortho.py"

model_paths = {}
with gzip.open(model_meta, 'rt') as f:
    entries = f.readlines()[1:]
    for entry in entries:
        meta_data = entry.strip().split(',')
        model = meta_data[0]
        model_path = meta_data[1]
        # extent = np.array([float(data) for data in meta_data[-3:]], dtype=np.float32)
        model_paths.update({model: model_path})
# print(model_paths.keys())

for listing in os.listdir(listing_dir):
    with gzip.open(os.path.join(listing_dir, listing), 'rb') as f:
        listing_lines = f.readlines()
    for line in listing_lines:
        data = json.loads(line, encoding='utf-8')
        if data['product_type'][0]['value'].lower() != 'table':
            continue
        if not (data["item_id"] in model_paths.keys()):
            continue
        #     print('no exist in ')
        # print()
        path = os.path.join(render_data_dir, model_paths[data['item_id']] )
        name = data['item_id']
        command = f'blenderproc run {script_file} --object-path {path} --output_dir {args.output_dir} --uid {name} --num-views {args.num_views} --resolution {args.resolution} --radius {args.radius} --scale {args.scale} --use-gpu {args.use_gpu}'
        cmds.append(command)
        #cmds.append(item)
        uids.append(name)


# for name in sample_dir_list:
#     sample_dir = os.path.join(args.input_models_path, name)
    # if os.path.exists(os.path.join(sample_dir, 'model_normalized.obj')):
    #     path = os.path.join(sample_dir, 'model_normalized.obj')
    # else:
    #     path = os.path.join(sample_dir, "models", "model_normalized.obj")

    

print(f'total mount of objs: {len(cmds)}')

if args.end == 0:
    args.end = len(cmds)
for i in range(args.start, args.end):
    #print(f'rendering {i} / {len(cmds)} images!, {cmds[i]}')
    if args.omit:
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

