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
IMG_SIZE = 1024
UV_MAP_TYPE = "BF" # 'SMPL' or 'BF'
UV_MAP_SIZE = 1024
NUMBER_OF_SUBDIVISIONS = 0

if UV_MAP_TYPE.upper() == "SMPL":
    UV_MAP_PATH = os.path.join("data", "demo", "smpl_uv.obj")
elif UV_MAP_TYPE.upper() == "BF":
    UV_MAP_PATH = os.path.join("data", "demo", "smpl_boundry_free_template.obj")
else:
    raise ValueError("Unknown UV Map Type")

DATA_PATH = os.path.join("data", TEST_IMG)

transform_img = transforms.Compose([           
            transforms.Resize(IMG_SIZE, max_size=IMG_SIZE+1),
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
    filename = UV_MAP_PATH,
)

control_mesh = Mesh(
    texturetype = UV_MAP_TYPE.upper(),
)

# ---------------------------------------------------------------------------------------------------------------
# ----- Load the input image ------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
input_image_PIL = transform_img(Image.open(os.path.join(DATA_PATH, "img.png"))) 
input_image = input_image_PIL.numpy().transpose(1, 2, 0) * 255

# ---------------------------------------------------------------------------------------------------------------
# ----- Load data from the METRO --------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
orig_vertices = np.load(os.path.join(DATA_PATH, "pred_vertices.npy")).astype(np.float64)

orig_camera = np.load(os.path.join(DATA_PATH, "pred_camera.npy"))
orig_camera = orig_camera.tolist()
# Taken from ???, gives good results
camera = [orig_camera[1], orig_camera[2], 2*1000.0/IMG_SIZE*orig_camera[0]]

# ---------------------------------------------------------------------------------------------------------------
# ----- Estimate the new texture with built-in function ---------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
control_mesh.v = -orig_vertices.copy()
control_texture, control_reprojection = control_mesh.create_texture_fom_image(
    input_image,
    camera,
    projection_camera=orig_camera,
    texture_size=UV_MAP_SIZE,
    n_subdivisions=NUMBER_OF_SUBDIVISIONS,
    return_reprojection_image=True,
)

cv2.imwrite(
    os.path.join(DATA_PATH, "new_texture_{}_control.png".format(UV_MAP_TYPE.lower())),
    control_texture
)
cv2.imwrite(
    os.path.join(DATA_PATH, "projection_test_control.png"),
    control_reprojection[:, :, ::-1]
)

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
cv2.imwrite(os.path.join(DATA_PATH, "partial_uvmap.png"), partial_texture_image)
partial_mesh.set_texture_image(os.path.join(DATA_PATH, "partial_uvmap.png"))

# Uniquify the mesh again so that I can color faces afterwards
partial_mesh.print("Partial mesh")
partial_mesh = partial_mesh.uniquified_mesh()
partial_mesh.print("Partial mesh after uniquification")

# ---------------------------------------------------------------------------------------------------------------
# ----- Subdivide triangles to get finer texture ----------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
for si in range(NUMBER_OF_SUBDIVISIONS):
    partial_mesh.subdivide_triangles()
    partial_mesh.vn = partial_mesh.estimate_vertex_normals()
    partial_mesh.print("Partial mesh after subdivision {:d}".format(si+1))

# ---------------------------------------------------------------------------------------------------------------
# ----- Project points to the camera ----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
projection_mesh = deepcopy(partial_mesh)
projection_mesh.v = - projection_mesh.v
raw_pts = projection_mesh.project_to_camera(camera=orig_camera)
pts = raw_pts + 1
pts *= (IMG_SIZE/2)

# ---------------------------------------------------------------------------------------------------------------
# ----- Show projection in the original image -------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
projected_image = input_image.copy()
pts_int = pts.astype(int)
projected_image[pts_int[:, 1], pts_int[:, 0], :] = (255, 0, 0)
cv2.imwrite(
    os.path.join(DATA_PATH, "projection_test.png"),
    projected_image[:, :, ::-1]
)
    
# ---------------------------------------------------------------------------------------------------------------
# ----- Create the colored mesh ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
colored_mesh = Mesh(
    filename = UV_MAP_PATH,
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
grid = torch.tensor(raw_pts.astype(float))[None, :, None, :]

sampled_colors = np.squeeze(torch.nn.functional.grid_sample(
    inpt/255,
    grid
).numpy()).transpose()

face_colors = np.mean(sampled_colors[colored_mesh.f, :], axis=1)
colored_mesh.fc = np.clip(face_colors, 0, 1)

# ---------------------------------------------------------------------------------------------------------------
# ----- Create new texture --------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
colored_mesh.print("Colored mesh after coloring")
new_texture = colored_mesh.create_texture_from_fc(texture_size=UV_MAP_SIZE)
cv2.imwrite(
    os.path.join(DATA_PATH, "new_texture_{}.png".format(UV_MAP_TYPE.lower())),
    new_texture
)

# ---------------------------------------------------------------------------------------------------------------
# ----- Create a mesh with the new texture ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
textured_mesh = Mesh(
    filename = UV_MAP_PATH,
)
textured_mesh.v = original_mesh.v
textured_mesh.set_vertex_colors("white")
textured_mesh.set_texture_image(os.path.join(DATA_PATH, "new_texture_{}_control.png".format(UV_MAP_TYPE.lower())))


# ---------------------------------------------------------------------------------------------------------------
# ----- Visualization -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:
    mvs = MeshViewers(shape=(1, 3))
    mvs[0][0].set_static_meshes([original_mesh])
    # mvs[0][1].set_static_meshes([partial_mesh])
    mvs[0][1].set_static_meshes([colored_mesh])
    mvs[0][2].set_static_meshes([textured_mesh])

