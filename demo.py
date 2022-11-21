import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers

# Load mesh
my_mesh = Mesh(
    filename = os.path.join("demo_data", "smpl_uv.obj")
)

# creates a grid of 2x2 mesh viewers
mvs = MeshViewers(shape=[2, 2])

# sets the first (top-left) mesh to my_mesh
mvs[0][0].set_static_meshes([my_mesh])

