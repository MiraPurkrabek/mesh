import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
from psbody.mesh.visibility import visibility_compute
import numpy as np


# Load mesh
my_mesh = Mesh(
    filename = os.path.join("demo_data", "smpl_uv.obj"),
)
my_mesh.set_vertex_colors("white")
my_mesh.set_texture_image(os.path.join("demo_data", "smpl_uv.png"))

# creates a grid of 2x2 mesh viewers
# mvs = MeshViewers()

vis1, n_dot_cam1 = visibility_compute(
    v=my_mesh.v,
    f=my_mesh.f,
    cams=np.array([[0.0, 0.0, 0.0]])
)

# vis = my_mesh.vertex_visibility(camera=[0.0, 0.0, 0.0])
vis1 = np.squeeze(vis1).astype(bool)
print(my_mesh.v.shape, my_mesh.v.dtype)
print(my_mesh.f.shape, my_mesh.f.dtype)
print(np.array([[0.0, 0.0, 0.0]]).shape)

print(vis1.shape)
# print(n_dot_cam.shape)

my_mesh.set_vertex_colors("green", vertex_indices=vis1)
my_mesh.set_vertex_colors("red", vertex_indices=~vis1)

print(my_mesh.texture_image)

# sets the first (top-left) mesh to my_mesh
# mvs[0][0].set_static_meshes([my_mesh])

