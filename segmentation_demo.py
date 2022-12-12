import os
from psbody.mesh import Mesh
from psbody.mesh.colors import name_to_rgb
from psbody.mesh import MeshViewers
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import plotly.graph_objects as go
import torch
import json
from copy import deepcopy

from matplotlib import path
import matplotlib.pyplot as plt


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
# ----- Show manual patches -------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

v_indices = np.array(list(range(segmented_mesh.v.shape[0])))
patches_dict = {}
for key, item in SMPL_points.items():
    if key.startswith("_"):
        continue
    
    # Draw border points to the mesh
    color = np.random.random(3)
    marked_mesh.set_vertex_colors(
        color,
        vertex_indices=item
    )

    patches_dict[key] = {}

    # Compute all 'inside' points of the patch
    if key.startswith("front"):
        front_flags = np.squeeze(marked_mesh.v[:, 2] >= 0)
        front_pts = marked_mesh.v[front_flags, :2]
        front_indices = v_indices[front_flags]

        p = path.Path(marked_mesh.v[item, :2])
        flags = p.contains_points(front_pts)
        inside_indices = front_indices[flags]
        gradient_mesh.set_vertex_colors(
            color,
            vertex_indices=inside_indices
        )
        projected_pts = front_pts[flags, :]
    # Compute all 'inside' points of the patch
    elif key.startswith("back"):
        back_flags = np.squeeze(marked_mesh.v[:, 2] < 0)
        back_pts = marked_mesh.v[back_flags, :2]
        back_indices = v_indices[back_flags]

        p = path.Path(marked_mesh.v[item, :2])
        flags = p.contains_points(back_pts)
        inside_indices = back_indices[flags]
        gradient_mesh.set_vertex_colors(
            color,
            vertex_indices=inside_indices
        )
    
    # Scale points between 0 and 1 for easier use in image reprojection
    projected_pts = marked_mesh.v[inside_indices, :2]
    # print(np.min(projected_pts, axis=0), np.max(projected_pts, axis=0))
    projected_pts -= np.min(projected_pts, axis=0)
    # print(np.min(projected_pts, axis=0), np.max(projected_pts, axis=0))
    projected_pts /= np.max(projected_pts, axis=0)
    # print(np.min(projected_pts, axis=0), np.max(projected_pts, axis=0))
    patches_dict[key]["projected_points"] = projected_pts.tolist()
    patches_dict[key]["indices"] = inside_indices.tolist()

# ---------------------------------------------------------------------------------------------------------------
# ----- Save the patches in one dict ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
with open(os.path.join("SMPL", "SMPL_patches.json"), "w") as seg_ptch_file:
    json.dump(patches_dict, seg_ptch_file, indent=2)

# ---------------------------------------------------------------------------------------------------------------
# ----- Visualization -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:

    fig = go.Figure(
        data=[go.Scatter3d(
            x=segmented_mesh.v[:, 0],
            y=segmented_mesh.v[:, 1],
            z=segmented_mesh.v[:, 2],
            mode='markers',
            marker=dict(
                size=1,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                color=v_indices,
                opacity=0.8,
            ),
            text=v_indices,
        )],
    )
    fig.update_layout(
        scene = dict(
                # aspectmode = 'cube',
                zaxis = dict(nticks=4, range=[-10,0],),
            ),
        )
    # fig.show()

    
    mvs = MeshViewers(shape=(1, 3))
    mvs[0][0].set_static_meshes([segmented_mesh])
    mvs[0][1].set_static_meshes([marked_mesh])
    mvs[0][2].set_static_meshes([gradient_mesh])
    
