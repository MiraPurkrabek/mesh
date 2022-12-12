#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2013 Max Planck Society. All rights reserved.
# Created by Matthew Loper on 2013-02-20.


import numpy as np
import cv2
import time
import json
import matplotlib.pyplot as plt

"""
texture.py

"""

__all__ = ['texture_coordinates_by_vertex', ]


def texture_coordinates_by_vertex(self):
    texture_coordinates_by_vertex = [[] for i in range(len(self.v))]
    for i, face in enumerate(self.f):
        for j in [0, 1, 2]:
            texture_coordinates_by_vertex[face[j]].append(self.vt[self.ft[i][j]])
    return texture_coordinates_by_vertex


def reload_texture_image(self):
    # image is loaded as image_height-by-image_width-by-3 array in BGR color order.
    self._texture_image = cv2.imread(self.texture_filepath) if self.texture_filepath else None
    texture_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    if self._texture_image is not None and (self._texture_image.shape[0] != self._texture_image.shape[1] or
       self._texture_image.shape[0] not in texture_sizes or
       self._texture_image.shape[0] not in texture_sizes):
        closest_texture_size_idx = (np.abs(np.array(texture_sizes) - max(self._texture_image.shape))).argmin()
        sz = texture_sizes[closest_texture_size_idx]
        self._texture_image = cv2.resize(self._texture_image, (sz, sz))


def load_texture(self, texture_version):
    '''
    Expect a texture version number as an integer, load the texture version from 'texture_path' (global variable to the
    package).
    Currently there are versions [0,1,2,3] available.
    '''
    import os
    from . import texture_path

    lowres_tex_template = os.path.join(texture_path, 'textured_template_low_v%d.obj' % texture_version)
    highres_tex_template = os.path.join(texture_path, 'textured_template_high_v%d.obj' % texture_version)
    from .mesh import Mesh

    mesh_with_texture = Mesh(filename=lowres_tex_template)
    if not np.all(mesh_with_texture.f.shape == self.f.shape):
        mesh_with_texture = Mesh(filename=highres_tex_template)
    self.transfer_texture(mesh_with_texture)


def transfer_texture(self, mesh_with_texture):
    if not np.all(mesh_with_texture.f.shape == self.f.shape):
        raise Exception('Mesh topology mismatch')

    self.vt = mesh_with_texture.vt.copy()
    self.ft = mesh_with_texture.ft.copy()

    if not np.all(mesh_with_texture.f == self.f):
        if np.all(mesh_with_texture.f == np.fliplr(self.f)):
            self.ft = np.fliplr(self.ft)
        else:
            # Same shape let's see if it's face ordering this could be a bit faster...
            face_mapping = {}
            for f, ii in zip(self.f, range(len(self.f))):
                face_mapping[" ".join([str(x) for x in sorted(f)])] = ii
            self.ft = np.zeros(self.f.shape, dtype=np.uint32)

            for f, ft in zip(mesh_with_texture.f, mesh_with_texture.ft):
                k = " ".join([str(x) for x in sorted(f)])
                if k not in face_mapping:
                    raise Exception('Mesh topology mismatch')
                # the vertex order can be arbitrary...
                ids = []
                for f_id in f:
                    ids.append(np.where(self.f[face_mapping[k]] == f_id)[0][0])
                ids = np.array(ids)
                self.ft[face_mapping[k]] = np.array(ft[ids])

    self.texture_filepath = mesh_with_texture.texture_filepath
    self._texture_image = None


def set_texture_image(self, path_to_texture):
    self.texture_filepath = path_to_texture


def texture_rgb(self, texture_coordinate):
    h, w = np.array(self.texture_image.shape[:2]) - 1
    return np.double(self.texture_image[int(h * (1.0 - texture_coordinate[1]))][int(w * (texture_coordinate[0]))])[::-1]


def texture_rgb_vec(self, texture_coordinates):
    h, w = np.array(self.texture_image.shape[:2]) - 1
    n_ch = self.texture_image.shape[2]
    # XXX texture_coordinates can be lower than 0! clip needed!
    d1 = (h * (1.0 - np.clip(texture_coordinates[:, 1], 0, 1))).astype(np.int)
    d0 = (w * (np.clip(texture_coordinates[:, 0], 0, 1))).astype(np.int)
    flat_texture = self.texture_image.flatten()
    indices = np.hstack([((d1 * (w + 1) * n_ch) + (d0 * n_ch) + (2 - i)).reshape(-1, 1) for i in range(n_ch)])
    return flat_texture[indices]


def edit_texture(self, to_edit, color=(0, 0, 0), criterion=np.any):
    texture_image = self._texture_image.copy()
    h, w = np.array(texture_image.shape[:2]) - 1
    texture_coors = texture_coordinates_by_vertex(self)

    # Take always the first point. Not clear why returning more than one point
    # as all points for one vertex are the same. Would maybe work differently for 
    # different UV map
    texture_coors = np.array(list(map(lambda x: x[0], texture_coors)))

    for face in self.f:
        edit_face = criterion(to_edit[face])
        if edit_face:
            
            face_points = texture_coors[face, :]
            pts = np.array([w, h]) * ( face_points * np.array([1, -1]) + np.array([0, 1]) )
            pts = pts.astype(np.int32)

            texture_image = cv2.drawContours(
                texture_image,
                [pts],
                contourIdx=0,
                color=color,
                thickness=-1
            )
    return texture_image

def project_to_camera(self, camera):
    """Perform orthographic projection of 3D points self.v using the camera parameters
    Taken from the MeshGraphormer repo.
    Args:
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    if isinstance(camera, list):
        camera = np.array(camera)
    camera = camera.reshape(-1, 3)
    # camera /= np.linalg.norm(camera)
    X_trans = self.v[:, :2] + camera[:, 1:]
    # shape = X_trans.shape
    # X_2d = (camera[:, 0] * X_trans.reshape(shape[0], -1))
    # X_2d = X_2d.view(shape)
    X_2d = camera[:, 0] * X_trans
    # X_2d = X_trans
    return X_2d

def create_texture_from_fc(self, texture_size=128):
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    if not hasattr(self, 'fc') or not hasattr(self, 'vt') or not hasattr(self, 'f'):
        return texture

    texture_coors = texture_coordinates_by_vertex(self)
    # Take always the first point. Not clear why returning more than one point
    # as all points for one vertex are the same. Would maybe work differently for 
    # different UV map
    texture_coors = np.array(list(map(lambda x: x[0] if len(x) else [0, 0], texture_coors)))
    for fi, face in enumerate(self.f):
        face_points = texture_coors[face, :]
        pts = np.array([texture_size-1, texture_size-1]) * ( face_points * np.array([1, -1]) + np.array([0, 1]) )
        pts = pts.astype(np.int32)

        texture = cv2.drawContours(
            texture,
            [pts],
            contourIdx=0,
            color=self.fc[fi, ::-1] * 255,
            thickness=-1
        )

    return texture

def extract_image_patches(
    self,
    image,
    camera,
    patches_dict,
    visible_vertices=None,
    target_size=256,
    visibility_threshold=0.7
):

    assert isinstance(patches_dict, dict), "Patches dict must be a dictionary. It is {}".format(type(patches_dict))

    if visible_vertices is None:
        # If the visibility is not given, take all vertices
        visible_vertices = np.ones(self.v.shape[0], dtype=bool)
    else:
        visible_vertices = np.array(visible_vertices)

    _, image_size, _ = image.shape

    output_dict = {}

    projected_pts = project_to_camera(self, camera)
    projected_pts = projected_pts + 1
    projected_pts *= (image_size/2)

    for patch_name, patch_subdict in patches_dict.items():
        if patch_name.startswith("_"):
            continue
        
        patch_idx = np.array(patch_subdict["indices"])
        patch_proj_pts = np.array(patch_subdict["projected_points"]) * (target_size-1)

        # Ignore not visible patches
        patch_visibility = (visible_vertices[patch_idx]).astype(bool)
        patch_visible = np.sum(patch_visibility) / len(patch_visibility)
        if patch_visible < visibility_threshold:
            output_dict[patch_name] = None
            continue

        # Filter out non-visible vertices
        # non_patch_idx = patch_idx[~ patch_visibility]
        patch_idx = patch_idx[patch_visibility]
        patch_proj_pts = patch_proj_pts[patch_visibility]

        patch_image_coors = projected_pts[patch_idx, :]
        patch_image_coors = patch_image_coors.astype(np.float32)
        # non_patch_image_coors = projected_pts[non_patch_idx, :]
        # non_patch_image_coors = non_patch_image_coors.astype(np.float32)

        # plt.imshow(image/255)
        # plt.plot(patch_image_coors[:, 0], patch_image_coors[:, 1], 'r.')
        # plt.plot(non_patch_image_coors[:, 0], non_patch_image_coors[:, 1], 'b.')
        # plt.show()

        warpMat, _ = cv2.findHomography(
            patch_image_coors,
            patch_proj_pts.astype(np.float32),
            method=cv2.LMEDS
        )
        warped_patch = cv2.warpPerspective(
            image,
            warpMat,
            (target_size, target_size),
            flags=cv2.INTER_CUBIC,
        )
        warped_patch = np.clip(warped_patch, 0, 255)

        # plt.imshow(warped_patch[::-1, ::-1, :]/255)
        # plt.plot(patch_proj_pts[:, 0], patch_proj_pts[:, 1], 'rx')
        # plt.show()

        output_dict[patch_name] = warped_patch
    return output_dict


def create_texture_from_image(self, pts, image, texture_size=128):
    # Re-scale pts from 0 to 1
    if np.min(pts) < 0 or np.max(pts) > 1:
        pts += 1
        pts /= 2

    assert (np.min(pts) >= 0 and np.max(pts) <= 1)
    
    _, image_size, _ = image.shape

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    
    texture_coors = texture_coordinates_by_vertex(self)
    # Take always the first point. Not clear why returning more than one point
    # as all points for one vertex are the same. Would maybe work differently for 
    # different UV map
    texture_coors = np.array(list(map(lambda x: x[0] if len(x) else [0, 0], texture_coors)))
    for fi, face in enumerate(self.f):
        start_time = time.perf_counter()
        
        image_tri = (pts[face, :] * image_size).astype(np.float32)
        texture_tri = (texture_coors[face] * texture_size).astype(np.float32)

        pts_sampling_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        
        image_rect = cv2.boundingRect(image_tri)
        texture_rect = cv2.boundingRect(texture_tri)

        image_tri_rectified = image_tri - np.array([image_rect[0], image_rect[1]], dtype=np.float32)
        texture_tri_rectified = texture_tri - np.array([texture_rect[0], texture_rect[1]], dtype=np.float32)

        image_rectified = image[
            image_rect[1]:image_rect[1]+image_rect[3],
            image_rect[0]:image_rect[0]+image_rect[2],
            :
        ]

        warpMat = cv2.getAffineTransform(
            image_tri_rectified,
            texture_tri_rectified,
        )

        warpMat_time = time.perf_counter() - start_time
        start_time = time.perf_counter()

        warped_rect = cv2.warpAffine(
            image_rectified,
            warpMat,
            (texture_rect[2], texture_rect[3]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        warping_time = time.perf_counter() - start_time
        start_time = time.perf_counter()

        mask = np.zeros((texture_rect[3], texture_rect[2], 3), dtype = np.float32)
        
        mask = cv2.fillConvexPoly(
            mask,
            texture_tri_rectified.astype(np.int32),
            color=(1.0, 1.0, 1.0),
        )
        warped_rect = warped_rect * mask

        texture[
            texture_rect[1]:texture_rect[1]+texture_rect[3],
            texture_rect[0]:texture_rect[0]+texture_rect[2],
        ] *= ((1.0, 1.0, 1.0) - mask).astype(texture.dtype)
        texture[
            texture_rect[1]:texture_rect[1]+texture_rect[3],
            texture_rect[0]:texture_rect[0]+texture_rect[2],
        ] += warped_rect.astype(texture.dtype)
        
        masking_time = time.perf_counter() - start_time
        start_time = time.perf_counter()

        # if fi%100 == 0:
        #     print("{:d} ({:.2f}%)".format(fi, fi/len(self.f)*100))
        #     print("\t{:.2f} s - sampling".format(pts_sampling_time))
        #     print("\t{:.2f} s - warpMat time".format(warpMat_time))
        #     print("\t{:.2f} s - warping".format(warping_time))
        #     print("\t{:.2f} s - masking".format(masking_time))
            # break
    
    # fig = plt.figure(figsize=(10, 7))
    # rows = 1
    # columns = 4
    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(image/255)
    # plt.plot(image_tri[:, 0], image_tri[:, 1], 'b.')
    # plt.plot(image_rect[0]              , image_rect[1]              , 'g.')
    # plt.plot(image_rect[0]+image_rect[2], image_rect[1]              , 'g.')
    # plt.plot(image_rect[0]              , image_rect[1]+image_rect[3], 'g.')
    # plt.plot(image_rect[0]+image_rect[2], image_rect[1]+image_rect[3], 'g.')
    # fig.add_subplot(rows, columns, 2)
    # plt.imshow(image_rectified/255)
    # fig.add_subplot(rows, columns, 3)
    # plt.imshow(warped_rect/255)
    # fig.add_subplot(rows, columns, 4)
    # plt.imshow(texture/255)
    # plt.plot(texture_tri[:, 0], texture_tri[:, 1], 'r.')
    # plt.show()
    
    return texture[::-1, :, ::-1]
    
