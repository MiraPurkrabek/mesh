import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from copy import deepcopy

MULTIVIEW_NUM=0
UV_MAP_TYPE = "BF" # 'SMPL' or 'BF'
IMG_SIZE=256

TEST_IMG = "multiview{:d}".format(MULTIVIEW_NUM)
DATA_PATH = os.path.join("data", TEST_IMG)
if UV_MAP_TYPE.upper() == "SMPL":
    UV_MAP_PATH = os.path.join("SMPL", "smpl_uv.obj")
    UV_MAPS_FOLDER = os.path.join(DATA_PATH, "SMPL_UVMaps")
elif UV_MAP_TYPE.upper() == "BF":
    UV_MAP_PATH = os.path.join("SMPL", "smpl_boundry_free_template.obj")
    UV_MAPS_FOLDER = os.path.join(DATA_PATH, "BF_UVMaps")
else:
    raise ValueError("Unknown UV Map Type")

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
mesh = Mesh(
    texturetype = UV_MAP_TYPE.upper(),
)

# ---------------------------------------------------------------------------------------------------------------
# ----- Load the input images -----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
images = []
vertices = []
orig_cameras = []
cameras = []
for file in os.listdir(DATA_PATH):
    if not (file.lower().endswith(".png") or file.lower().endswith(".jpg")):
        continue

    if not file.lower().startswith("img_"):
        continue

    img_filepath = os.path.join(DATA_PATH, file)
    vert_filepath = os.path.join(DATA_PATH, file[:-4]+"_pred_vertices.npy")
    cam_filepath = os.path.join(DATA_PATH, file[:-4]+"_pred_camera.npy")
    
    input_image_PIL = transform_img(Image.open(img_filepath)) 
    input_image = input_image_PIL.numpy().transpose(1, 2, 0) * 255
    images.append(input_image)

    orig_vert = np.load(vert_filepath).astype(np.float64)
    orig_camera = np.load(cam_filepath)
    camera = [orig_camera[1], orig_camera[2], 2*1000.0/IMG_SIZE*orig_camera[0]]
    orig_camera = orig_camera.tolist()

    vertices.append(-orig_vert.copy())
    orig_cameras.append(orig_camera)
    cameras.append(camera)

# ---------------------------------------------------------------------------------------------------------------
# ----- Estimate the new texture from multiple images -----------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

merged_texture, partial_textures = mesh.merge_texture_from_image(
        images,
        vertices,
        vis_cameras=cameras,
        proj_cameras=orig_cameras,
        texture_size = 1024,
        normal_threshold = None,
        # normal_threshold = 0.0,
        return_partial_textures = True,
        verbose = True,
    )


# ---------------------------------------------------------------------------------------------------------------
# ----- Save the estimated textures -----------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

cv2.imwrite(
    os.path.join(
        DATA_PATH,
        "merged_multiview_{:s}_texture.png".format(UV_MAP_TYPE.upper())
    ),
    merged_texture
)

for pi, pmesh in enumerate(partial_textures):
    cv2.imwrite(
        os.path.join(
            DATA_PATH,
            "partial_{:s}_texture_c{:d}.png".format(UV_MAP_TYPE.upper(), pi)
        ),
        pmesh
    )

# ---------------------------------------------------------------------------------------------------------------
# ----- Create a mesh with the new texture ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

merged_mesh = Mesh(
    filename = UV_MAP_PATH,
)
merged_mesh.set_vertex_colors("white")
merged_mesh.set_texture_image(os.path.join(
    DATA_PATH,
    "merged_multiview_{:s}_texture.png".format(UV_MAP_TYPE.upper())
))

# ---------------------------------------------------------------------------------------------------------------
# ----- Create a mesh with partial textures ---------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
partial_meshes = []
for i in range(4):
    pmesh = Mesh(
        filename = UV_MAP_PATH,
    )
    pmesh.set_vertex_colors("white")
    pmesh.set_texture_image(os.path.join(
            DATA_PATH,
            "partial_{:s}_texture_c{:d}.png".format(UV_MAP_TYPE.upper(), i)
    ))
    # pmesh.v = - vertices[i].copy()
    partial_meshes.append(pmesh)
    

# ---------------------------------------------------------------------------------------------------------------
# ----- Visualization -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:
    mvs = MeshViewers(shape=(2, 3))
    
    # Partial meshes
    mvs[1][0].set_static_meshes([partial_meshes[0]])
    mvs[1][1].set_static_meshes([partial_meshes[1]])
    mvs[1][2].set_static_meshes([partial_meshes[2]])
    mvs[0][0].set_static_meshes([partial_meshes[3]])
    
    # Merged mesh
    mvs[0][2].set_static_meshes([merged_mesh])

