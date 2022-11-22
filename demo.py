import os
from psbody.mesh import Mesh
from psbody.mesh import MeshViewers
from psbody.mesh.visibility import visibility_compute
import numpy as np
import cv2

CAMERA=[[0.0, 0.0, 100.0]]

# Load mesh
my_mesh = Mesh(
    filename = os.path.join("demo_data", "smpl_uv.obj"),
)
my_mesh.set_vertex_colors("white")
my_mesh.set_texture_image(os.path.join("demo_data", "img_uvmap.png"))
# my_mesh.set_texture_image(os.path.join("demo_data", "test_uvmap.png"))

texture_image = cv2.imread(os.path.join("demo_data", "img_uvmap.png"))
# texture_image = my_mesh.texture_image


vis, n_dot_cam = visibility_compute(
    v=my_mesh.v,
    f=my_mesh.f,
    cams=np.array(CAMERA)
)
vis = np.squeeze(vis)
print(vis.shape)

faces_to_keep = filter(lambda face: vis[face[0]] * vis[face[1]] * vis[face[2]], my_mesh.f)
vertex_indices_to_keep = np.nonzero(vis)[0]
vertices_to_keep = my_mesh.v[vertex_indices_to_keep]
old_to_new_indices = np.zeros(len(vis))
old_to_new_indices[vertex_indices_to_keep] = range(len(vertex_indices_to_keep))

partial_mesh = Mesh(
    v=vertices_to_keep,
    f=np.array([old_to_new_indices[face] for face in faces_to_keep]),
)

# print(partial_mesh.texture_image)
partial_mesh.set_vertex_colors("white")
# partial_mesh.set_texture_image(os.path.join("demo_data", "img_uvmap.png"))
# partial_mesh.reload_texture_image()
# texture_image = partial_mesh.texture_image

# vis = my_mesh.vertex_visibility(camera=[0.0, 0.0, 0.0])
vis1 = np.squeeze(
    vis).astype(bool)


# print(vis.shape)
# print(n_dot_cam.shape)

# my_mesh.set_vertex_colors("green", vertex_indices=vis1)
# my_mesh.set_vertex_colors("red", vertex_indices=~vis1)

# print(np.sum(texture_image == 0))
# print(texture_image.shape)
# faces_to_keep = filter(lambda face: vis[face[0]] * vis[face[1]] * vis[face[2]], my_mesh.f)

unq_mesh = my_mesh.uniquified_mesh()

unq_vis, _ = visibility_compute(
    v=my_mesh.v,
    f=my_mesh.f,
    cams=np.array(CAMERA)
)
unq_vis = np.squeeze(unq_vis)
ver_to_tex = my_mesh.texture_coordinates_by_vertex()
h, w = np.array(texture_image.shape[:2]) - 1  
for face in my_mesh.f:
    visible = unq_vis[face[0]] * unq_vis[face[1]] * unq_vis[face[2]]
    if not visible:
        vertices = []
        for f in face:
            x = int(h * (1.0 - ver_to_tex[f][0][1]))
            y = int(w * (ver_to_tex[f][0][0]))
            vertices.append([y, x])

        # v1 = (- np.array(ver_to_tex[face[0]]) + np.array([0, 1]))# * np.array(w, h)
        # v1 = np.squeeze(v1)
        # v2 = (- np.array(ver_to_tex[face[1]]) + np.array([0, 1]))# * np.array(w, h)
        # v2 = np.squeeze(v2)
        # v3 = (- np.array(ver_to_tex[face[2]]) + np.array([0, 1]))# * np.array(w, h)
        # v3 = np.squeeze(v3)
        # v2 = np.squeeze(np.array(ver_to_tex[face[1]]) * np.array(texture_image.shape[:2]))
        # v3 = np.squeeze(np.array(ver_to_tex[face[2]]) * np.array(texture_image.shape[:2]))

        
        # for vv1 in v1:
            # print(vv1)
            # texture_image[int(vv1[1]), int(vv1[0]), :] = (0, 0, 255)
            # texture_image = cv2.circle(
            #     texture_image,
            #     tuple(vv1.astype(int)),
            #     radius=1,
            #     color=(0, 0, 255),
            # )

        triangle_cnt = np.array(vertices, dtype=np.int32)
        # triangle_cnt = np.stack([v1, v2, v3], axis=1).astype(np.int32).transpose()
        # print(triangle_cnt.shape)
        # print(triangle_cnt)
        texture_image = cv2.drawContours(texture_image, [triangle_cnt], 0, (0, 0, 0), -1)
        # break

        # texture_image = cv2.polylines(
        #     texture_image,
        #     [triangle_cnt],
        #     isClosed = True,
        #     color = (0, 0, 255),
        #     thickness = -1
        # )
        # break

# print(my_mesh.ft.shape)
# # print(len(ver_to_tex[0]))
# unq_ver_to_tex = unq_mesh.texture_coordinates_by_vertex()
# print(unq_mesh.f.shape)
# print(len(unq_ver_to_tex), len(unq_ver_to_tex[0]))
# print(texture_image.shape[0] * texture_image.shape[1])

# h, w = np.array(texture_image.shape[:2]) - 1  
# for v, tex_coors in zip(vis, my_mesh.texture_coordinates_by_vertex()):
#     if not v:
#         for c in tex_coors:
#             x = int(c[0] * texture_image.shape[0])
#             y = int(c[1] * texture_image.shape[1])

#             x = int(h * (1.0 - c[1]))
#             y = int(w * (c[0]))
#             texture_image[x, y, :] = [0, 0, 255]

# print(np.sum(texture_image == 0))
# my_mesh.texture_image = texture_image

cv2.imwrite("test.png", texture_image)

result_mesh = Mesh(
    filename = os.path.join("demo_data", "smpl_uv.obj"),
)
result_mesh.set_vertex_colors("white")
result_mesh.set_texture_image("test.png")

# creates a grid of 2x2 mesh viewers
mvs = MeshViewers(shape=(1, 3))
mvs[0][0].set_static_meshes([my_mesh])
mvs[0][1].set_static_meshes([partial_mesh])
mvs[0][2].set_static_meshes([result_mesh])

