import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from copy import deepcopy

TEST_IMG = "classic_E_03"
IMG_SIZE = 224

transform_img = transforms.Compose([           
            transforms.Resize(IMG_SIZE, max_size=225),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225])
            ])

# ---------------------------------------------------------------------------------------------------------------
# ----- Load data from original SMPL mesh -----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
original_mesh = Mesh(
    filename = os.path.join("data", "demo", "smpl_uv.obj"),
    # filename = os.path.join("data", "demo", "smpl_boundry_free_template.obj"),
)

# ---------------------------------------------------------------------------------------------------------------
# ----- Load data from the METRO --------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
orig_vertices = np.load(os.path.join("data", TEST_IMG, "pred_vertices.npy")).astype(np.float64)

orig_camera = np.load(os.path.join("data", TEST_IMG, "pred_camera.npy"))
orig_camera = orig_camera.tolist()
# Taken from ???, gives good results
camera = [orig_camera[1], orig_camera[2], 2*1000.0/IMG_SIZE*orig_camera[0]]

# ---------------------------------------------------------------------------------------------------------------
# ----- Set the estimated pose ----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# Visibility needs negative vertices, projection will use original ones
original_mesh.v = -orig_vertices.copy()

# ---------------------------------------------------------------------------------------------------------------
# ----- Uniquify the original mesh ------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
original_mesh.print("Original mesh")

# Estimate its normals for proper uniquification
original_mesh.vn = original_mesh.estimate_vertex_normals()

unique_mesh = original_mesh.uniquified_mesh()
unique_mesh.set_texture_image(os.path.join("data", "demo", "img_uvmap.png"))
unique_mesh.print("Unique mesh")

# ---------------------------------------------------------------------------------------------------------------
# ----- Compute visibility --------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
partial_mesh, partial_texture_image = unique_mesh.visibile_mesh(
    camera=camera,
    return_texture=True,
    criterion=np.all,
)
# Save the partial UV map to load as a new texture
cv2.imwrite(os.path.join("data", "demo", "partial_uvmap.png"), partial_texture_image)
partial_mesh.set_texture_image(os.path.join("data", "demo", "partial_uvmap.png"))

# Uniquify the mesh again so that I can color faces afterwards
partial_mesh.print("Partial mesh")
partial_mesh = partial_mesh.uniquified_mesh()
partial_mesh.print("Partial mesh after uniquification")

# ---------------------------------------------------------------------------------------------------------------
# ----- Subdivide triangles to get finer texture ----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
partial_mesh.subdivide_triangles()
partial_mesh.vn = partial_mesh.estimate_vertex_normals()
partial_mesh.print("Partial mesh after subdivision")

# ---------------------------------------------------------------------------------------------------------------
# ----- Project points to the camera ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
projection_mesh = deepcopy(partial_mesh)
projection_mesh.v = - projection_mesh.v
pts = projection_mesh.project_to_camera(camera=orig_camera)
pts += 1
pts *= (IMG_SIZE/2)
print("Projected points", pts.shape)

# ---------------------------------------------------------------------------------------------------------------
# ----- Show projection in the original image -------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
input_image_PIL = transform_img(Image.open(os.path.join("data", TEST_IMG, "img.png"))) 
input_image = input_image_PIL.numpy().transpose(1, 2, 0) * 255

projected_image = input_image.copy()
pts_int = pts.astype(int)
projected_image[pts_int[:, 1], pts_int[:, 0], :] = (255, 0, 0)
cv2.imwrite("projection_test.png", projected_image[:, :, ::-1])
    
# ---------------------------------------------------------------------------------------------------------------
# ----- Create the colored mesh ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
colored_mesh = Mesh(
    # filename = os.path.join("data", "demo", "smpl_uv.obj"),
    filename = os.path.join("data", "demo", "smpl_boundry_free_template.obj"),
)
# colored_mesh.set_texture_image(os.path.join("data", "demo", "img_uvmap.png"))
# Copy vertices and faces from the unique (fully visible) mesh
colored_mesh.v = partial_mesh.v
colored_mesh.f = partial_mesh.f
colored_mesh.vt = partial_mesh.vt
colored_mesh.ft = partial_mesh.ft
colored_mesh.print("Colored mesh")

# ---------------------------------------------------------------------------------------------------------------
# ----- Sample color from the input image -----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
inpt = torch.tensor(input_image.transpose(2, 0, 1).astype(float))[None, :, :, :]
grid = torch.tensor(pts.astype(float))[None, :, None, :]

# Scale to (-1, 1)
grid -= torch.min(grid)
grid /= grid.max() /2
grid -= 1

sampled_colors = np.squeeze(torch.nn.functional.grid_sample(
    inpt/255,
    grid
).numpy()).transpose()

colored_mesh.fc = np.zeros((colored_mesh.f.shape[0], 3))
for fi, f in enumerate(colored_mesh.f):
    v1, v2, v3 = f
    c = np.mean(sampled_colors[[v1, v2, v3], :], axis=0)
    c = np.clip(c, 0, 1)
    colored_mesh.fc[fi, :] = c

# ---------------------------------------------------------------------------------------------------------------
# ----- Visualization -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:
    mvs = MeshViewers(shape=(1, 3))
    mvs[0][0].set_static_meshes([original_mesh])
    mvs[0][1].set_static_meshes([partial_mesh])
    mvs[0][2].set_static_meshes([colored_mesh])
