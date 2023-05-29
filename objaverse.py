import blenderproc as bproc
import argparse
import math
import os
import random
import numpy as np
import sys
import time
import urllib.request
from typing import Tuple
import blenderproc.python.renderer.RendererUtility as RendererUtility
from blenderproc.scripts.saveAsImg import save_array_as_image
import json

"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    #required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=4)
parser.add_argument("--radius", type=float, default=1.5)

#argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args()

#args.object_path = "origin.glb"
#args.object_path = "000074a334c541878360457c672b6c2e.obj"
args.object_path = "/home/yulin/data/objaverse/model_normalized.obj"
#args.object_path = "000074a334c541878360457c672b6c2e.glb"
args.output_dir = "./views_shapenet"
args.resolution = 512

bproc.init()

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
# render.resolution_x = 512
# render.resolution_y = 512
render.resolution_percentage = 100

#scene.cycles.device = "CPU" #"GPU"
bproc.renderer.set_render_devices()
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


# def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
#     theta = random.random() * 2 * math.pi
#     phi = math.acos(2 * random.random() - 1)
#     return (
#         radius * math.sin(phi) * math.cos(theta),
#         radius * math.sin(phi) * math.sin(theta),
#         radius * math.cos(phi),
#     )


def add_lighting() -> None:
    # delete the default light
    #bpy.data.objects["Light"].select_set(True)
    #bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        print('import .glb file')
        return bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        #print("???")
    elif object_path.endswith(".obj"):
        return bproc.loader.load_obj(
            object_path,
            use_legacy_obj_import=True,
            use_edges=False,
            use_smooth_groups=False,
            split_mode="OFF",
        )
    elif object_path.endswith(".fbx"):
        return bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


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


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * 0.8
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    # offset = -(bbox_min + bbox_max) / 2
    # for obj in scene_root_objects():
    #     obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def sample_camera_loc(azimuth=None, elevation=None, r=3.5):
    #phi = np.random.uniform(np.pi / 3, np.pi / 3 * 2)
    #theta = np.random.uniform(0, np.pi * 2)
    # x = r * np.sin(phi) * np.cos(theta)
    # y = r * np.sin(phi) * np.sin(theta)
    # z = r * np.cos(phi)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    #reset_scene()
    # load the object
    objs = load_object(object_file)
    # for obj in bpy.context.scene.objects.values():
    #     if not obj.parent:
    #         break
    # print('obj,', obj)

    
    object_uid = os.path.basename(object_file).split(".")[0]
    assert len(objs) == 1, len(objs)
    obj = objs[0]
    #bbox_8 = obj.get_bound_box()

    obj.persist_transformation_into_mesh(location=False, rotation=True, scale=True)

    #normalize_scene()
    add_lighting()

    

    fovy = np.arctan(32 / 2 / 35) * 2
    bproc.camera.set_intrinsics_from_blender_params(fovy, lens_unit="FOV")
    bproc.camera.set_resolution(args.resolution, args.resolution)

    
    bbox_min, bbox_max = scene_bbox()
    aabb = [np.array(bbox_min).tolist(), np.array(bbox_max).tolist()]
    #cam, cam_constraint = setup_camera()
    # create an empty object to track
    # empty = bpy.data.objects.new("Empty", None)
    # scene.collection.objects.link(empty)
    # cam_constraint.target = empty
    '''for i in range(args.num_images):
        # set the camera position
        theta = (i / args.num_images) * math.pi * 2
        phi = math.radians(60)
        point = (
            args.camera_dist * math.sin(phi) * math.cos(theta),
            args.camera_dist * math.sin(phi) * math.sin(theta),
            args.camera_dist * math.cos(phi),
        )
        cam.location = point
        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)'''


    # views = [[np.pi / 3, np.pi / 6],
    #     [np.pi / 3, np.pi / 6 * 4],
    #     [np.pi / 3, np.pi / 6 * 7], 
    #     [np.pi / 3, np.pi / 6 * 10], 
    #     [np.pi / 2, np.pi / 6 * 2],
    #     [np.pi / 2, np.pi / 6 * 5],
    #     [np.pi / 2, np.pi / 6 * 8],
    #     [np.pi / 2, np.pi / 6 * 11]]

    meta = {"fovy": fovy, "aabb": aabb}
    frames = []

    #poi = [-0.02598, -0.02080, 0]
    poi = np.zeros(3)
    for i in range(args.num_images):
        #print(i)
        # Sample random camera location around the object
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(-np.pi / 6, np.pi / 3)

        x = np.cos(azimuth) * np.cos(elevation)
        y = np.sin(azimuth) * np.cos(elevation)
        z = np.sin(elevation)
        location = np.array([x, y, z]) * args.radius

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)

        # Add homogeneous cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix, frame=i)        
        frame = dict(
            transform_matrix=cam2world_matrix.tolist(),
            azimuth=azimuth,
            elevation=elevation,
        )
        frames.append(frame)

    bproc.renderer.enable_normals_output(output_dir=str(args.output_dir))
    bproc.renderer.enable_depth_output(False, output_dir=str(args.output_dir))
    bproc.renderer.set_output_format(enable_transparency=True)

    #print("start rendering")
    data = bproc.renderer.render(output_dir=str(args.output_dir), return_data=False) #verbose=True)
    # for index, image in enumerate(data["colors"]):
    #     render_path = os.path.join(args.output_dir, f"{index:03d}.png")
    #     save_array_as_image(image, "colors", render_path)
    # for index, image in enumerate(data["normals"]):
    #     render_path = os.path.join(args.output_dir, f"{index:03d}_normal.png")
    #     save_array_as_image(image, "normals", render_path)

    meta["frames"] = frames
    with open(os.path.join(args.output_dir ,"meta.json"), "w") as f:
        json.dump(meta, f, indent = 4)
    #print('render done!')


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
