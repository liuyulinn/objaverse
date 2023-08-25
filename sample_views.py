import random
import matplotlib.pyplot as plotlib
import numpy
import cv2
import os
# from lib.datasets.parallel_zip import ParallelZipFile as ZipFile

row = 5
column = 10

image_dir = 'rendered_shapenet_50/views_shapenet'
# submap = os.listdir(image_dir)
# submap.sort()
# submap = {os.path.split(u)[0]: archive for archive in archives for u in archive.namelist()}
# subs = [item for item in submap.keys()]

img_list = []
for i in range(row):
    imgs = []
    for j in range(column):
        idx = row * i + j
        # sub = submap[idx]
        # sub = f'{image_dir}/{sub}'
        # print(idx, sub)
        # archive = submap[sub]

        # frame = random.randint(0, 49)
        # with io.BytesIO(archive.read(f"{sub}/rgb_{frame:04}.png")) as fi:
        img = plotlib.imread(f"{image_dir}/rgb_{idx:04}.png")

        img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
        img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
        imgs.append(img[None])
    imgs = numpy.concatenate(imgs, axis = 0)
    img_list.append(imgs[None])

img_list = numpy.concatenate(img_list, axis = 0).transpose(0, 2, 1, 3, 4).reshape(row * 256, column * 256, 3)

plotlib.imsave('samples.png', img_list)
