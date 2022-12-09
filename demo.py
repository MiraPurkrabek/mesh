import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
import numpy as np
import cv2

import matplotlib.pyplot as plt

CAMERA=[0.0, 1.0, 1.0] # positive number means: [LEFT, TOP, FRONT]

# Normalize the camera to be on the unit sphere
np_camera = np.array(CAMERA)
np_camera = np_camera / (np.linalg.norm(np_camera) + 1.0e-12)
CAMERA = np_camera.tolist()

print(CAMERA)

# Load the original mesh
original_mesh = Mesh(
    filename = os.path.join("SMPL", "smpl_uv.obj"),
)
original_mesh.set_vertex_colors("white")
original_mesh.set_texture_image(os.path.join("data", "demo", "img.png"))

# Compute visbility of both mesh and texture
partial_mesh, texture_image = original_mesh.visibile_mesh(camera=CAMERA, return_texture=True)

# Save the partial UV map to load as a new texture
cv2.imwrite(os.path.join("data", "demo", "partial_uvmap.png"), texture_image)

# Load a new mesh with edited texture
result_mesh = Mesh(
    filename = os.path.join("SMPL", "smpl_uv.obj"),
)
result_mesh.set_vertex_colors("white")
result_mesh.set_texture_image(os.path.join("data", "demo", "partial_uvmap.png"))


print(np.min(original_mesh.v, axis=0), np.mean(original_mesh.v, axis=0), np.max(original_mesh.v, axis=0))
pts = partial_mesh.project_to_camera(camera=CAMERA)
# print(np.min(pts, axis=0), np.max(pts, axis=0))
# print(pts.shape)


# Visualize all meshes net to each other
if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:
    mvs = MeshViewers(shape=(1, 3))
    variable=mvs[0][0].parent_window
    print(type(variable), dir(variable))

    # cameras = mvs[0][0].on_draw(want_cameras=True)
    # print(cameras)
    mvs[0][0].set_static_meshes([original_mesh])
    mvs[0][1].set_static_meshes([partial_mesh])
    mvs[0][2].set_static_meshes([result_mesh])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # v = original_mesh.v
    # ax.scatter(v[:, 0], v[:, 1], v[:, 2])
    # ax.scatter(CAMERA[0], CAMERA[1], CAMERA[2], color="red")
    # plt.show()

