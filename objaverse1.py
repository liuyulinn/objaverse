# isort: off
import blenderproc as bproc
from blenderproc.python.utility.SetupUtility import SetupUtility
import bpy

# isort: on
import argparse
import json
from pathlib import Path
import math
import numpy as np
from mathutils import Vector
# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True)
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

#args.object_path = "/home/yulin/data/objaverse/005c71d003e24a588bc203d578de416c.glb"

#args.object_path = "/home/yulin/data/objaverse/000074a334c541878360457c672b6c2e.glb"
#args.object_path = "/home/yulin/data/objaverse/model_normalized.obj"
#args.object_path = "origin.obj"
#args.object_path = "/home/yulin/data/objaverse/000074a334c541878360457c672b6c2e.obj"
uid = args.object_path.split("/")[-1].split(".")[0]

args.output_dir = f'/yulin/objaverse/views_{uid}'
args.no_depth = 0


np.random.seed(args.seed)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------- #
# Initialize bproc
# ---------------------------------------------------------------------------- #
bproc.init()

# Renderer setting (following GET3D)
if args.engine == "cycles":
    #bproc.renderer.set_render_devices('GPU')
    bpy.context.scene.cycles.device = "GPU"
    #bproc.renderer.set_denoiser("OPTIX")
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.filter_width = 0.01
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
else:
    objs = bproc.loader.load_obj(
        args.object_path,
        use_legacy_obj_import=True,
        use_edges=False,
        use_smooth_groups=False,
        split_mode="OFF",
    )
    assert len(objs) == 1, len(objs)
    obj = objs[0]
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
    obj.scale = obj.scale * scale * 0.8
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

# Compute current axis-aligned bounding box
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

bproc.renderer.enable_normals_output(output_dir=str(output_dir))
# Render RGB images
data = bproc.renderer.render(output_dir=str(output_dir), return_data=False)

# Save meta info
meta["frames"] = frames
with open(output_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=4)


# # -------------------------------------------------------------------------- #
# # Render orthographic views
# # -------------------------------------------------------------------------- #
# bproc.utility.reset_keyframes()

# cam_ob = bpy.context.scene.camera
# cam_ob.data.type = "ORTHO"
# cam_ob.data.ortho_scale = 1.0

# poi = np.zeros(3)
# locations = [
#     [1, 0, 0],
#     [-1, 0, 0],
#     [0, 1, 0],
#     [0, -1, 0],
#     [0, 0, 1],
#     [0, 0, -1],
# ]

# for i, location in enumerate(locations):
#     # Compute rotation based on vector going from location towards poi
#     rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

#     # Add homogeneous cam pose based on location an rotation
#     cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
#     bproc.camera.add_camera_pose(cam2world_matrix, frame=i)

# bproc.renderer.render(
#     output_dir=str(output_dir),
#     file_prefix="ortho_",
#     return_data=False,
#     load_keys={"colors"},
#     output_key=None,
# )
