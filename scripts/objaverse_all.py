# isort: off
import blenderproc as bproc
from blenderproc.python.utility.SetupUtility import SetupUtility
from blenderproc.python.utility.Utility import Utility
import bpy

# isort: on
import argparse
import json
from pathlib import Path
import math
import numpy as np
from mathutils import Vector
from scipy.spatial.transform import Rotation as R


def disable_all_denoiser():
    """ Disables all denoiser.

    At the moment this includes the cycles and the intel denoiser.
    """
    # Disable cycles denoiser
    bpy.context.view_layer.cycles.use_denoising = False
    bpy.context.scene.cycles.use_denoising = False

    # Disable intel denoiser
    if bpy.context.scene.use_nodes:
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        # Go through all existing denoiser nodes
        for denoiser_node in Utility.get_nodes_with_type(nodes, 'CompositorNodeDenoise'):
            in_node = denoiser_node.inputs['Image']
            out_node = denoiser_node.outputs['Image']

            # If it is fully included into the node tree
            if in_node.is_linked and out_node.is_linked:
                # There is always only one input link
                in_link = in_node.links[0]
                # Connect from_socket of the incoming link with all to_sockets of the out going links
                for link in out_node.links:
                    links.new(in_link.from_socket, link.to_socket)

            # Finally remove the denoiser node
            nodes.remove(denoiser_node)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # r, i, j, k = np.unbind(quaternions, -1)
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis = -1, keepdims = True)
    # angles = axis_angle.norm(p = 2, dim = -1, keepdim = True)
    # angles = np.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))




# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--object-path", type=str, required=True)
parser.add_argument("--use-gpu", type=int, default = 1)
parser.add_argument("--output_dir", type=str, default="/objaverse-processed/rendered_ortho")
# parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--num-views", type=int, default=50)
parser.add_argument("--seed", type=int)
parser.add_argument("--engine", type=str, default="cycles")
parser.add_argument("--light-energy", type=float, default=10)
parser.add_argument("--no-depth", type=int, default=1)
parser.add_argument("--no-normal", type=int, default=1)
parser.add_argument("--random", type=int, default=0)
parser.add_argument("--random_angle", type=int, default=0)
args = parser.parse_args()

n_threads = 16
#args.object_path = "/home/yulin/data/objaverse/005c71d003e24a588bc203d578de416c.glb"

#args.object_path = "/home/yulin/data/objaverse/000074a334c541878360457c672b6c2e.glb"
#args.object_path = "/home/yulin/data/objaverse/model_normalized.obj"
#args.object_path = "origin.obj"
#args.object_path = "/home/yulin/data/objaverse/000074a334c541878360457c672b6c2e.obj"
uid = args.object_path.split("/")[-1].split(".")[0]

args.output_dir = f'{args.output_dir}/views_{uid}'
# args.output_dir = f'views_{uid}'
# args.no_depth = 0


np.random.seed(args.seed)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------- #
# Initialize bproc
# ---------------------------------------------------------------------------- #
bproc.init() #compute_device = 'GPU')

# Renderer setting (following GET3D)
if args.engine == "cycles":
    #bproc.renderer.set_render_devices('GPU')
    if args.use_gpu:
        bpy.context.scene.cycles.device = "GPU"
        # bproc.renderer.set_render_devices(use_only_gpu=True)
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            if d["name"] == "Intel Xeon Platinum 8255C CPU @ 2.50GHz":
                d["use"] = 0
            elif d["name"] == "AMD EPYC 7502 32-Core Processor":
                d["use"] = 0
            else:
                d["use"] = 1
            print(d["name"], d["use"])
    else:
        #bproc.python.utility.Initializer.init() #compute_device='CPU') #, compute_device_type=None, use_experimental_features=False, clean_up_scene=True)
        bproc.renderer.set_render_devices(use_only_cpu=True)
        bproc.renderer.set_cpu_threads(n_threads)
    #bpy.context.preferences.addons["cycles"].preferences.get_devices()
    #bproc.renderer.set_denoiser("OPTIX")
    disable_all_denoiser()
    bpy.context.scene.use_nodes = False
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.view_layer.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.scene.cycles.filter_width = 1.0
    bpy.context.scene.cycles.denoising_prefilter = 'FAST'
    bpy.context.view_layer.use_pass_normal = False
    bpy.context.view_layer.use_pass_diffuse_color = False
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.set_light_bounces(
        diffuse_bounces=1,
        glossy_bounces=1,
        transmission_bounces=3,
        transparent_max_bounces=3,
    )
    bproc.renderer.set_max_amount_of_samples(32)
else:
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bproc.renderer.set_output_format(enable_transparency=True)

# ---------------------------------------------------------------------------- #
# Load model
# ---------------------------------------------------------------------------- #
if args.object_path.endswith(".glb"):
    bpy.ops.import_scene.gltf(filepath = args.object_path, merge_vertices=True)
elif args.object_path.endswith(".fbx"):
    bpy.ops.import_scene.fbx(filepath=args.object_path)
elif args.object_path.endswith(".obj"):
    objs = bproc.loader.load_obj(
        args.object_path,
        use_legacy_obj_import=True,
        use_edges=False,
        use_smooth_groups=False,
        split_mode="OFF",
    )
    assert len(objs) == 1, len(objs)
    obj = objs[0]
else:
    raise ValueError(f"Unsupported file type: {args.object_path}")
    #print(obj)

# NOTE(jigu): Following GET3D to split custom normals, but do not know how it affects
# obj.blender_obj.select_set(True)
# bpy.context.view_layer.objects.active = obj.blender_obj
# bpy.ops.object.mode_set(mode="EDIT")
# bpy.ops.mesh.split_normals()
# bpy.ops.object.mode_set(mode="OBJECT")
# obj.blender_obj.select_set(False)
def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

bbox_min, bbox_max = scene_bbox()
scale = 1 / max(bbox_max - bbox_min)
for obj in scene_root_objects():
    obj.scale = obj.scale * scale * args.scale
# Apply scale to matrix_world.
bpy.context.view_layer.update()
bbox_min, bbox_max = scene_bbox()
offset = -(bbox_min + bbox_max) / 2
for obj in scene_root_objects():
    obj.matrix_world.translation += offset
bpy.ops.object.select_all(action="DESELECT")

# Scale the object
# bbox_8x3 = obj.get_bound_box()
# # print(bbox_8x3)
# size = max(np.max(bbox_8x3, 0) - np.min(bbox_8x3, 0))
# scale_factor = args.scale / size
# # Note that `obj.set_scale` needs to be called at each frame instead.
# obj.blender_obj.scale = np.array([scale_factor] * 3)

# NOTE(jigu): Following `bproc.loader.load_shapenet`
# removes the x axis rotation found in all ShapeNet objects, this is caused by importing .obj files
# the object has the same pose as before, just that the rotation_euler is now [0, 0, 0]
#obj.persist_transformation_into_mesh(location=False, rotation=True, scale=True)

# ---------------------------------------------------------------------------- #
# Render
# ---------------------------------------------------------------------------- #
# Set a light source
light = bproc.types.Light("AREA")
# CAUTION: The default value is 0.25, which can lead to lighter images compared to bpy.ops.object.light_add.
light.blender_obj.data.size = 1.0
light.set_energy(energy=args.light_energy)
light.set_location([0, 0, 0.5])

# Set rendering parameters
fovy = np.arctan(32 / 2 / 35) * 2
bproc.camera.set_intrinsics_from_blender_params(fovy, lens_unit="FOV")
bproc.camera.set_resolution(args.resolution, args.resolution)

# # Compute current axis-aligned bounding box
# aabb = [[-args.scale / 2] * 3, [args.scale / 2] * 3]
# bbox_8x3 = obj.get_bound_box()
# # print(bbox_8x3)
# aabb = [np.min(bbox_8x3, 0).tolist(), np.max(bbox_8x3, 0).tolist()]
bbox_min, bbox_max = scene_bbox()
aabb = [np.array(bbox_min).tolist(), np.array(bbox_max).tolist()]

meta = {"fovy": fovy,  "aabb": aabb}
frames = []
azimuths = []
elevations = []

# Sample camera pose
poi = np.zeros(3)
for i in range(args.num_views):
    # Sample random camera location above objects
    azimuth = np.random.uniform(0, 2 * np.pi)
    elevation = np.arccos(np.random.uniform(-1, 1))

    x = np.cos(azimuth) * np.sin(elevation)
    y = np.sin(azimuth) * np.sin(elevation)
    z = np.cos(elevation)
    location = np.array([x, y, z]) * args.radius

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

    # Add homogeneous cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix, frame=i)

    # NOTE(jigu): blenderproc save {%04d}.png.
    frame = dict(
        transform_matrix=cam2world_matrix.tolist(),
        azimuth=azimuth,
        elevation=elevation,
    )
    frames.append(frame)

    azimuths.append(azimuth)
    elevations.append(elevation)

elevations = np.array(elevations)
print(elevations.max(), elevations.min())

if not args.no_depth:
    bproc.renderer.enable_depth_output(False, output_dir=str(output_dir))

if not args.no_normal:
    bproc.renderer.enable_normals_output(output_dir=str(output_dir))
# Render RGB images
bproc.renderer.set_cpu_threads(n_threads)
data = bproc.renderer.render(output_dir=str(output_dir), return_data=False)

meta["frames"] = frames
with open(output_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=4)

# # -------------------------------------------------------------------------- #
# # Render orthographic views
# # -------------------------------------------------------------------------- #
bproc.utility.reset_keyframes()

cam_ob = bpy.context.scene.camera
cam_ob.data.type = "ORTHO"
cam_ob.data.ortho_scale = 1.0

poi = np.zeros(3)
locations = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
] 

# locations = [
#     [args.radius, 0, 0],
#     [-args.radius, 0, 0],
#     [0, args.radius, 0],
#     [0, -args.radius, 0],
#     [0, 0, args.radius],
#     [0, 0, -args.radius],
# ] #* args.radius

# bbox_min, bbox_max = scene_bbox()
# aabb = [np.array(bbox_min).tolist(), np.array(bbox_max).tolist()]



# meta = {"fovy": fovy,  "aabb": aabb}
# frames = []

# if args.random:
#     random_rotation = R.random()

for i, location in enumerate(locations):
    # Compute rotation based on vector going from location towards poi
    # print(location)
    # if args.random:
    #     azimuth = np.random.uniform(0, 2 * np.pi)
    #     elevation = np.arccos(np.random.uniform(-1, 1))
    #     x = np.cos(azimuth) * np.sin(elevation)
    #     y = np.sin(azimuth) * np.sin(elevation)
    #     z = np.cos(elevation)
    #     location1 = np.array([x, y, z])
    #     random_vector = location1 * 30 / 180 * np.pi
    #     random_rotation_plus = axis_angle_to_matrix(random_vector)

    # print(random_rotation_plus.shape)
    # location = random_rotation_plus @ (random_rotation @ location)
    # rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

    # location = random_rotation_plus @ random_rotation.as_matrix() @ np.array(location)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

    # Add homogeneous cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix, frame=i)

    # frame = dict(
#         transform_matrix = cam2world_matrix.tolist(),
#         # azimuth = azimuth,
#         # elevation = elevation,
#     )

#     frames.append(frame)

# if not args.no_depth:
#     bproc.renderer.enable_depth_output(False, output_dir=str(output_dir))

# bproc.renderer.enable_normals_output(output_dir=str(output_dir))
# # Render RGB images
# data = bproc.renderer.render(output_dir=str(output_dir), return_data=False)

bproc.renderer.set_cpu_threads(n_threads)
bproc.renderer.render(
    output_dir=str(output_dir),
    file_prefix="ortho_",
    return_data=False,
    load_keys={"colors"},
    output_key=None,
)

# Save meta info
