import os
import gzip
import json
import numpy as np
import cv2
from PIL import Image


render_data_dir = 'data/abo/abo-benchmark-material'
listing_dir = 'data/abo/listings/metadata'
model_meta = 'data/abo/3dmodels/metadata/3dmodels.csv.gz'
out_train_dir = 'data/abo/tables_train'
out_test_dir = 'data/abo/tables_test'
mask_correction = (1, 254)
in_size = 512
out_size = 128

save_img = True
save_pose = True


def sort_fun(x):
    y = os.path.splitext(x)[0].split('_')[1]
    return int(y)


img_scale = out_size / in_size
os.makedirs(out_train_dir, exist_ok=True)
os.makedirs(out_test_dir, exist_ok=True)

split_meta = {}
split_file = os.path.join(render_data_dir, 'train_test_split.csv')
with open(split_file, 'r') as f:
    entries = f.readlines()[1:]
    for entry in entries:
        model, split, _, _ = entry.strip().split(',')
        split_meta.update({model: split})

model_extents = {}
with gzip.open(model_meta, 'rt') as f:
    entries = f.readlines()[1:]
    for entry in entries:
        meta_data = entry.strip().split(',')
        model = meta_data[0]
        extent = np.array([float(data) for data in meta_data[-3:]], dtype=np.float32)
        model_extents.update({model: extent})

train_instances = []
test_instances = []
view_distance = []

scene_count = 0
for listing in os.listdir(listing_dir):
    with gzip.open(os.path.join(listing_dir, listing), 'rb') as f:
        listing_lines = f.readlines()
    for line in listing_lines:
        data = json.loads(line, encoding='utf-8')
        if data['product_type'][0]['value'].lower() != 'table':
            continue
        render_base_dir = os.path.join(render_data_dir, data['item_id'] + '/render')
        # segmentation_dir = os.path.join(render_data_dir, data['item_id'] + '/segmentation')
        if not os.path.exists(render_base_dir):
            continue
        # assert data['item_id'] in split_meta
        # if save_img:
        #     seg_masks = []
        #     seg_files = os.listdir(segmentation_dir)
        #     seg_files.sort(key=sort_fun)
        #     for seg_file in seg_files:
        #         mask = cv2.imread(os.path.join(segmentation_dir, seg_file), 0)
        #         mask[mask <= mask_correction[0]] = 0
        #         mask[mask >= mask_correction[1]] = 255
        #         seg_masks.append(mask.astype(np.float32) / 255)
        # for env_id in os.listdir(render_base_dir):
        #     render_dir = os.path.join(render_base_dir, env_id)
        #     meta_path = os.path.join(render_data_dir, data['item_id'] + '/metadata.json')
        #     item = dict(
        #         item_id=data['item_id'],
        #         env_id=env_id,
        #         render_dir=render_dir,
        #         segmentation_dir=segmentation_dir,
        #         meta_path=meta_path)
        #     if split_meta[data['item_id']].lower() == 'train':
        #         train_instances.append(item)
        #     else:
        #         test_instances.append(item)
            # instance_out_dir = os.path.join(
            #     out_train_dir if split_meta[data['item_id']].lower() == 'train' else out_test_dir,
            #     data['item_id'] + '_' + env_id)
            # if save_img:
            #     img_out_dir = os.path.join(instance_out_dir, 'rgb')
            #     os.makedirs(img_out_dir, exist_ok=True)
            #     render_files = os.listdir(render_dir)
            #     render_files.sort(key=sort_fun)
            #     for i, render_file in enumerate(render_files):
            #         img = cv2.imread(os.path.join(render_dir, render_file))
            #         mask = seg_masks[i][..., None]
            #         img = img * mask + 255 * (1 - mask)
            #         img = np.round(img).astype(np.uint8)
            #         img = Image.fromarray(img)
            #         img = np.asarray(img.resize((out_size, out_size)))
            #         cv2.imwrite(os.path.join(img_out_dir, '{:06d}.png'.format(i)), img)
        # with open(meta_path, 'r') as f:
        #     pose_meta = json.loads(f.read())
        # if save_pose:
        #     intrinsics = pose_meta['views'][0]['K']
        #     f = intrinsics[0] * img_scale
        #     cx = intrinsics[2] * img_scale
        #     cy = intrinsics[5] * img_scale
        #     out_intrinsics = '{:.6f} {:.6f} {:.6f} 0.\n0. 0. 0.\n1.\n{} {}\n'.format(f, cx, cy, out_size, out_size)
        #     with open(os.path.join(instance_out_dir, 'intrinsics.txt'), 'w') as f:
        #         f.write(out_intrinsics)
        #     pose_out_dir = os.path.join(instance_out_dir, 'pose')
        #     os.makedirs(pose_out_dir, exist_ok=True)
        # for i, view in enumerate(pose_meta['views']):
        #     pose = np.array(view['pose']).reshape(4, 4)
        #     pose[:3, 3] /= np.linalg.norm(model_extents[data['item_id']])  # scale normalization
        #     pose[:3, 1:3] = -pose[:3, 1:3]  # looks like abo has a different definition of camera coordinate system
        #     view_distance.append(np.linalg.norm(pose[:3, 3]))
        #     if save_pose:
        #         np.savetxt(os.path.join(pose_out_dir, '{:06d}.txt'.format(i)), pose.reshape(1, -1))
        scene_count += 1
        if scene_count % 10 == 0:
            print('Processed {} scenes'.format(scene_count))

num_train = len(train_instances)
num_test = len(test_instances)
num_data = num_train + num_test
print('num_data = {}, num_train = {}, num_test = {}'.format(num_data, num_train, num_test))

dist_mean = np.mean(view_distance)
dist_std = np.std(view_distance)
print('distance_mean = {:.4f}\ndistance_std = {:.4f}'.format(dist_mean, dist_std))