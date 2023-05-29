import json
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

def unproject_depth(depth_image: np.ndarray, intrinsic: np.ndarray, offset):
    """Unproject a depth image to a 3D poin cloud.
    Args:
        depth_image: [H, W]
        intrinsic: [3, 3]
        offset: offset of x and y indices.
    Returns:
        points: [H, W, 3]
    """
    v, u = np.indices(depth_image.shape)  # [H, W], [H, W]
    z = depth_image  # [H, W]
    uv1 = np.stack([u + offset, v + offset, np.ones_like(z)], axis=-1)
    points = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]
    return points


def get_intrinsic_matrix(width, height, fov, degree=False):
    """Get the camera intrinsic matrix according to image size and fov."""
    if degree:
        fov = np.deg2rad(fov)
    f = (width / 2.0) / np.tan(fov / 2.0)
    xc = (width - 1.0) / 2.0
    yc = (height - 1.0) / 2.0
    K = np.array([[f, 0, xc], [0, f, yc], [0, 0, 1.0]])
    return K

root_dir = Path("/home/yulin/data/objaverse/views_0")

with open(root_dir / "meta.json") as f:
    meta = json.load(f)


K = get_intrinsic_matrix(256, 256, meta["fovy"])

all_points = []
all_points_rgb = []

for i, frame in enumerate(meta["frames"]):
    # print(i, frame["file_path"])
    depth = imageio.imread(root_dir / f"depth_{i:04d}.exr")
    color = imageio.imread(root_dir / f"rgb_{i:04d}.png")
    depth = depth[..., 0]
    depth[depth > 10.0] = 0.0
    rgb = color[..., :3]

    import matplotlib.pyplot as plt
    plt.imshow(depth)
    plt.show()

    points = unproject_depth(depth, K, 0.5)
    points = points[depth > 0]
    points_rgb = rgb[depth > 0]

    import trimesh
    trimesh.PointCloud(points, points_rgb).show()

    cam2world = np.array(frame["transform_matrix"])
    cam2world[:, 1:3] *= -1  # opencv to blender
    all_points.append(points @ cam2world[:3, :3].T + cam2world[:3, 3])
    all_points_rgb.append(points_rgb)
    import trimesh
    trimesh.PointCloud(all_points[-1], points_rgb).show()


all_points = np.concatenate(all_points, axis=0)
all_points_rgb = np.concatenate(all_points_rgb, axis=0)
import trimesh
trimesh.PointCloud(all_points, all_points_rgb).show()

# Ortho projection

