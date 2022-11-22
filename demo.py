import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
import numpy as np
import cv2

CAMERA=[100.0, 50.0, 0.0] # positive number means: [LEFT, TOP, FRONT]

# Load the original mesh
original_mesh = Mesh(
    filename = os.path.join("demo_data", "smpl_uv.obj"),
)
original_mesh.set_vertex_colors("white")
original_mesh.set_texture_image(os.path.join("demo_data", "img_uvmap.png"))

# Compute visbility of both mesh and texture
partial_mesh, texture_image = original_mesh.visibile_mesh(camera=CAMERA, return_texture=True)

# Save the partial UV map to load as a new texture
cv2.imwrite(os.path.join("demo_data", "partial_uvmap.png"), texture_image)

# Load a new mesh with edited texture
result_mesh = Mesh(
    filename = os.path.join("demo_data", "smpl_uv.obj"),
)
result_mesh.set_vertex_colors("white")
result_mesh.set_texture_image(os.path.join("demo_data", "partial_uvmap.png"))


# Visualize all meshes net to each other
mvs = MeshViewers(shape=(1, 3))
mvs[0][0].set_static_meshes([original_mesh])
mvs[0][1].set_static_meshes([partial_mesh])
mvs[0][2].set_static_meshes([result_mesh])

