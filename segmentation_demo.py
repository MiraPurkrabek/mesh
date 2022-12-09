import os
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
from psbody.mesh import MeshViewers
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
import json
from copy import deepcopy

TEST_IMG = "occlusion_E_00"
IMG_SIZE = 1024
UV_MAP_TYPE = "BF" # 'SMPL' or 'BF'
UV_MAP_SIZE = 1024
NUMBER_OF_SUBDIVISIONS = 0

if UV_MAP_TYPE.upper() == "SMPL":
    UV_MAP_PATH = os.path.join("SMPL", "smpl_uv.obj")
elif UV_MAP_TYPE.upper() == "BF":
    UV_MAP_PATH = os.path.join("SMPL", "smpl_boundry_free_template.obj")
else:
    raise ValueError("Unknown UV Map Type")

DATA_PATH = os.path.join("data", TEST_IMG)

# ---------------------------------------------------------------------------------------------------------------
# ----- Load data from original SMPL mesh -----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
segmented_mesh = Mesh(
    filename = UV_MAP_PATH,
)
segmented_mesh.set_vertex_colors("gray20")
gradient_mesh = Mesh(
    filename = UV_MAP_PATH,
)
gradient_mesh.set_vertex_colors("gray20")
marked_mesh = Mesh(
    filename = UV_MAP_PATH,
)
marked_mesh.set_vertex_colors("gray20")

# ---------------------------------------------------------------------------------------------------------------
# ----- Segment the mesh by body parts --------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
with open(os.path.join("SMPL", "SMPL_segmentation.json"), "r") as seg_file:
    segmentation = json.load(seg_file)

with open(os.path.join("SMPL", "SMPL_segmentation_colors.json"), "r") as seg_col_file:
    seg_colors = json.load(seg_col_file)

assert(segmentation.keys() == seg_colors.keys())

for part_name in segmentation.keys():
    segmented_mesh.set_vertex_colors(
        seg_colors[part_name],
        vertex_indices=segmentation[part_name]
    )

# ---------------------------------------------------------------------------------------------------------------
# ----- Segment the mesh by body parts --------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
with open(os.path.join("SMPL", "SMPL_segmentation.json"), "r") as seg_file:
    segmentation = json.load(seg_file)

with open(os.path.join("SMPL", "SMPL_points.json"), "r") as points_file:
    SMPL_points = json.load(points_file)

with open(os.path.join("SMPL", "SMPL_segmentation_colors.json"), "r") as seg_col_file:
    seg_colors = json.load(seg_col_file)

assert(segmentation.keys() == seg_colors.keys())

for part_name in segmentation.keys():
    segmented_mesh.set_vertex_colors(
        seg_colors[part_name],
        vertex_indices=segmentation[part_name]
    )

# ---------------------------------------------------------------------------------------------------------------
# ----- Mark random vertices with red ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
colors = np.zeros((len(segmentation["rightShoulder"]), 3))
colors[:, 0] = range(len(segmentation["rightShoulder"]))
colors /= len(segmentation["rightShoulder"])
gradient_mesh.set_vertex_colors(
    colors,
    vertex_indices = segmentation["rightShoulder"]
)

# ---------------------------------------------------------------------------------------------------------------
# ----- Mark random vertices with red ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
body_part = "hips"
colors = [
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "magenta",
    "brown"
]

# for color in colors:
#     random_vertices = np.random.randint(
#         low = 0,
#         high = len(segmentation[body_part]),
#         size = 1
#     )
#     random_vertices = segmentation[body_part][random_vertices[0]]
#     marked_mesh.set_vertex_colors(
#         color,
#         vertex_indices=random_vertices
#     )
#     print(color, random_vertices)

marked_mesh.set_vertex_colors(
    "red",
    vertex_indices=[SMPL_points["rightShoulderFront"]]
)
marked_mesh.set_vertex_colors(
    "red",
    vertex_indices=SMPL_points["leftShoulderFront"]
)
marked_mesh.set_vertex_colors(
    "red",
    vertex_indices=[SMPL_points["leftHipFront"]]
)
marked_mesh.set_vertex_colors(
    "red",
    vertex_indices=[SMPL_points["rightHipFront"]]
)

marked_mesh.set_vertex_colors(
    "green",
    vertex_indices=[SMPL_points["rightShoulderBack"]]
)
marked_mesh.set_vertex_colors(
    "green",
    vertex_indices=SMPL_points["leftShoulderBack"]
)
marked_mesh.set_vertex_colors(
    "green",
    vertex_indices=SMPL_points["leftHipBack"]
)
marked_mesh.set_vertex_colors(
    "green",
    vertex_indices=[SMPL_points["rightHipBack"]]
)



# ---------------------------------------------------------------------------------------------------------------
# ----- Visualization -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:
    mvs = MeshViewers(shape=(1, 1))
    # mvs[0][0].set_static_meshes([segmented_mesh])
    # mvs[0][1].set_static_meshes([gradient_mesh])
    mvs[0][0].set_static_meshes([marked_mesh])
    
