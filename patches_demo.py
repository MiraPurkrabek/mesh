import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from copy import deepcopy

TEST_IMG = "occlusion_E_00"
SRC_IMG_SIZE = 224
TARGET_IMG_SIZE = 256

UV_MAP_PATH = os.path.join("SMPL", "smpl_uv.obj")
DATA_PATH = os.path.join("data", TEST_IMG)

transform_img = transforms.Compose([           
            transforms.Resize(SRC_IMG_SIZE, max_size=SRC_IMG_SIZE+1),
            transforms.CenterCrop(SRC_IMG_SIZE),
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
camera = [orig_camera[1], orig_camera[2], 2*1000.0/SRC_IMG_SIZE*orig_camera[0]]

# ---------------------------------------------------------------------------------------------------------------
# ----- Estimate the new texture with built-in function ---------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
original_mesh.v = orig_vertices.copy()
ret = original_mesh.extract_patches_from_image(
    input_image,
    projection_camera = orig_camera,
    visibility_camera = camera,
    target_size=TARGET_IMG_SIZE,
)

# ---------------------------------------------------------------------------------------------------------------
# ----- Visualization -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# if "DISPLAY" in os.environ.keys() and os.environ["DISPLAY"]:
    
#     mvs = MeshViewers(shape=(1, 3))
#     mvs[0][0].set_static_meshes([original_mesh])

