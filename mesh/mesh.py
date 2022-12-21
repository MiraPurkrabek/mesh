#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2012 Max Planck Society. All rights reserved.

"""
Mesh module
-----------

"""


import os
import json
from functools import reduce
import time

import torch
import numpy as np

from . import colors
from . import search

try:
    from .serialization import serialization
except ImportError:
    pass

from . import landmarks
from . import texture
from . import processing
from .visibility import visibility_compute
from .texture_types import BF_objfile_path, SMPL_objfile_path
from .manual_patches import SMPL_manual_patches


__all__ = ["Mesh"]


class Mesh(object):
    """3d Triangulated Mesh class

    Attributes:
        v: Vx3 array of vertices
        f: Fx3 array of faces

    Optional attributes:
        fc: Fx3 array of face colors
        vc: Vx3 array of vertex colors
        vn: Vx3 array of vertex normals
        segm: dictionary of part names to triangle indices

    """
    def __init__(self,
                 v=None,
                 f=None,
                 segm=None,
                 filename=None,
                 texturetype=None,
                 ppfilename=None,
                 lmrkfilename=None,
                 basename=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 landmarks=None):
        """
        :param v: vertices
        :param f: faces
        :param filename: a filename from which a mesh is loaded
        """

        if texturetype is not None and texturetype.upper() in ["SMPL", "BF"]:
            self.texturetype = texturetype.upper()
            if texturetype.upper() == "BF":
                filename = BF_objfile_path
            else:
                filename = SMPL_objfile_path
        if filename is not None:
            self.load_from_file(filename)
            if hasattr(self, 'f'):
                self.f = np.require(self.f, dtype=np.uint32)
            self.v = np.require(self.v, dtype=np.float64)
            self.filename = filename
            if vscale is not None:
                self.v *= vscale
        if v is not None:
            self.v = np.array(v, dtype=np.float64)
            if vscale is not None:
                self.v *= vscale
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

        self.basename = basename
        if self.basename is None and filename is not None:
            self.basename = os.path.splitext(os.path.basename(filename))[0]

        if segm is not None:
            self.segm = segm
        if landmarks is not None:
            self.set_landmark_indices_from_any(landmarks)
        if ppfilename is not None:
            self.set_landmark_indices_from_ppfile(ppfilename)
        if lmrkfilename is not None:
            self.set_landmark_indices_from_lmrkfile(lmrkfilename)

        if vc is not None:
            self.set_vertex_colors(vc)

        if fc is not None:
            self.set_face_colors(fc)

    def __del__(self):
        if hasattr(self, 'textureID'):
            from OpenGL.GL import glDeleteTextures
            glDeleteTextures([self.textureID])

    def print(self, name="Mesh"):
        print("{:s}:".format(name))

        if hasattr(self, 'v'):
            print("\tVertices: {}".format(self.v.shape))
        if hasattr(self, 'vt'):
            print("\tVT      : {}".format(self.vt.shape))
        if hasattr(self, 'vn'):
            print("\tVN      : {}".format(self.vn.shape))
        if hasattr(self, 'vc'):
            print("\tVC      : {}".format(self.vc.shape))
        if hasattr(self, 'f'):
            print("\tFaces   : {}".format(self.f.shape))
        if hasattr(self, 'ft'):
            print("\tFT      : {}".format(self.ft.shape))
        if hasattr(self, 'fc'):
            print("\tFC      : {}".format(self.fc.shape))

    def uniquified_mesh(self):
        """This function returns a copy of the mesh in which vertices are copied such that
        each vertex appears in only one face, and hence has only one texture"""
        new_mesh = Mesh(v=self.v[self.f.flatten()], f=np.array(range(len(self.f.flatten()))).reshape(-1, 3))

        if not hasattr(self, 'vn'):
            self.reset_normals()
        new_mesh.vn = self.vn[self.f.flatten()]

        if hasattr(self, 'vt'):
            new_mesh.vt = self.vt[self.ft.flatten()]
            new_mesh.ft = new_mesh.f.copy()

        if hasattr(self, '_texture_image'):
            new_mesh._texture_image = self._texture_image.copy()
        
        return new_mesh

    def edges_as_lines(self, copy_vertices=False):
        from .lines import Lines
        edges = self.f[:, [0, 1, 1, 2, 2, 0]].flatten().reshape(-1, 2)
        verts = self.v.copy() if copy_vertices else self.v
        return Lines(v=verts, e=edges)

    def show(self, mv=None, meshes=[], lines=[]):
        from .meshviewer import MeshViewer
        from .utils import row

        if mv is None:
            mv = MeshViewer(keepalive=True)

        if hasattr(self, 'landm'):
            from .sphere import Sphere
            sphere = Sphere(np.zeros((3)), 1.).to_mesh()
            scalefactor = 1e-2 * np.max(np.max(self.v) - np.min(self.v)) / np.max(np.max(sphere.v) - np.min(sphere.v))
            sphere.v = sphere.v * scalefactor
            spheres = [Mesh(vc='SteelBlue', f=sphere.f, v=sphere.v + row(np.array(self.landm_raw_xyz[k]))) for k in self.landm.keys()]
            mv.set_dynamic_meshes([self] + spheres + meshes, blocking=True)
        else:
            mv.set_dynamic_meshes([self] + meshes, blocking=True)
        mv.set_dynamic_lines(lines)
        return mv

    def colors_like(self, color, arr=None):
        from .utils import row, col

        if arr is None:
            arr = np.zeros(self.v.shape)

        # if arr is single-dim, reshape it
        if arr.ndim == 1 or arr.shape[1] == 1:
            arr = arr.reshape(-1, 3)

        if isinstance(color, str):
            color = colors.name_to_rgb[color]
        elif isinstance(color, list):
            color = np.array(color)

        if color.shape[0] == arr.shape[0] and color.shape[0] == color.size:
            def jet(v):
                fourValue = 4 * v
                red = min(fourValue - 1.5, -fourValue + 4.5)
                green = min(fourValue - 0.5, -fourValue + 3.5)
                blue = min(fourValue + 0.5, -fourValue + 2.5)
                result = np.array([red, green, blue])
                result[result > 1.0] = 1.0
                result[result < 0.0] = 0.0
                return row(result)
            color = col(color)
            color = np.concatenate([jet(color[i]) for i in range(color.size)], axis=0)

        return np.ones_like(arr) * color

    def set_vertex_colors(self, vc, vertex_indices=None):
        if vertex_indices is not None:
            self.vc[vertex_indices] = self.colors_like(vc, self.v[vertex_indices])
        else:
            self.vc = self.colors_like(vc, self.v)
        return self

    def set_vertex_colors_from_weights(self, weights, scale_to_range_1=True, color=True):
        # from numpy import ones_like
        if weights is None:
            return self
        if scale_to_range_1:
            weights = weights - np.min(weights)
            weights = (1.0 - 0.0) * weights / np.max(weights) + 0.0
        if color:
            from matplotlib import cm
            self.vc = cm.jet(weights)[:, :3]
        else:
            self.vc = np.tile(np.reshape(weights, (len(weights), 1)), (1, 3))  # *ones_like(self.v)
        return self

    def scale_vertex_colors(self, weights, w_min=0.0, w_max=1.0):
        if weights is None:
            return self
        weights = weights - np.min(weights)
        weights = (w_max - w_min) * weights / np.max(weights) + w_min
        self.vc = (weights * self.vc.T).T if weights is not None else self.vc
        return self

    def set_face_colors(self, fc):
        self.fc = self.colors_like(fc, self.f)
        return self

    def faces_by_vertex(self, as_sparse_matrix=False):
        import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(self.v))]
            for i, face in enumerate(self.f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = self.f.flatten()
            col = np.array([range(self.f.shape[0])] * 3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(self.v.shape[0], self.f.shape[0]))
        return faces_by_vertex

    def estimate_vertex_normals(self, face_to_verts_sparse_matrix=None):
        from .geometry.tri_normals import TriNormalsScaled

        face_normals = TriNormalsScaled(self.v, self.f).reshape(-1, 3)
        ftov = face_to_verts_sparse_matrix if face_to_verts_sparse_matrix else self.faces_by_vertex(as_sparse_matrix=True)
        non_scaled_normals = ftov * face_normals
        norms = (np.sum(non_scaled_normals ** 2.0, axis=1) ** 0.5).T
        norms[norms == 0] = 1.0
        return (non_scaled_normals.T / norms).T

    def barycentric_coordinates_for_points(self, points, face_indices):
        from .geometry.barycentric_coordinates_of_projection import barycentric_coordinates_of_projection
        vertex_indices = self.f[face_indices.flatten(), :]
        tri_vertices = np.array([self.v[vertex_indices[:, 0]], self.v[vertex_indices[:, 1]], self.v[vertex_indices[:, 2]]])
        return vertex_indices, barycentric_coordinates_of_projection(points, tri_vertices[0, :], tri_vertices[1, :] - tri_vertices[0, :], tri_vertices[2, :] - tri_vertices[0, :])

    def transfer_segm(self, mesh, exclude_empty_parts=True):
        self.segm = {}
        if hasattr(mesh, 'segm'):
            face_centers = np.array([self.v[face, :].mean(axis=0) for face in self.f])
            (closest_faces, closest_points) = mesh.closest_faces_and_points(face_centers)
            mesh_parts_by_face = mesh.parts_by_face()
            parts_by_face = [mesh_parts_by_face[face] for face in closest_faces.flatten()]
            self.segm = dict([(part, []) for part in mesh.segm.keys()])
            for face, part in enumerate(parts_by_face):
                self.segm[part].append(face)
            for part in self.segm.keys():
                self.segm[part].sort()
                if exclude_empty_parts and not self.segm[part]:
                    del self.segm[part]

    @property
    def verts_by_segm(self):
        return dict((segment, sorted(set(self.f[indices].flatten()))) for segment, indices in self.segm.items())

    def parts_by_face(self):
        segments_by_face = [''] * len(self.f)
        for part in self.segm.keys():
            for face in self.segm[part]:
                segments_by_face[face] = part
        return segments_by_face

    def verts_in_common(self, segments):
        """
        returns array of all vertex indices common to each segment in segments"""
        return sorted(reduce(lambda s0, s1: s0.intersection(s1),
                             [set(self.verts_by_segm[segm]) for segm in segments]))
        # # indices of vertices in the faces of the first segment
        # indices = self.verts_by_segm[segments[0]]
        # for segment in segments[1:] :
        #    indices = sorted([index for index in self.verts_by_segm[segment] if index in indices]) # Intersect current segment with current indices
        # return sorted(set(indices))

    @property
    def joint_names(self):
        return self.joint_regressors.keys()

    @property
    def joint_xyz(self):
        joint_locations = {}
        for name in self.joint_names:
            joint_locations[name] = self.joint_regressors[name]['offset'] + \
                np.sum(self.v[self.joint_regressors[name]['v_indices']].T * self.joint_regressors[name]['coeff'], axis=1)
        return joint_locations

    # creates joint_regressors from a list of joint names and a per joint list of vertex indices (e.g. a ring of vertices)
    # For the regression coefficients, all vertices for a given joint are given equal weight
    def set_joints(self, joint_names, vertex_indices):
        self.joint_regressors = {}
        for name, indices in zip(joint_names, vertex_indices):
            self.joint_regressors[name] = {'v_indices': indices,
                                           'coeff': [1.0 / len(indices)] * len(indices),
                                           'offset': np.array([0., 0., 0.])}

    def vertex_visibility(self, camera, normal_threshold=None, omni_directional_camera=False, binary_visiblity=True):

        vis, n_dot_cam = self.vertex_visibility_and_normals(camera, omni_directional_camera)

        if normal_threshold is not None:
            vis = np.logical_and(vis, n_dot_cam > normal_threshold)

        return np.squeeze(vis) if binary_visiblity else np.squeeze(vis * n_dot_cam)

    def vertex_visibility_and_normals(self, camera, omni_directional_camera=False):        
        if isinstance(camera, list):
            camera_origin = camera
            assert omni_directional_camera, "Omnidrectional camera is not used but camera is of type 'list'"
        else:
            camera_origin = camera.origin.flatten()
        
        arguments = {'v': self.v,
                     'f': self.f,
                     'cams': np.array([camera_origin])}

        if not omni_directional_camera:
            arguments['sensors'] = np.array([camera.sensor_axis.flatten()])

        arguments['n'] = self.vn if hasattr(self, 'vn') else self.estimate_vertex_normals()

        return(visibility_compute(**arguments))

    def visibile_mesh(
        self,
        camera=[0.0, 0.0, 0.0],
        color=(0, 0, 0),
        return_texture=False,
        criterion=np.any,
        normal_threshold=None,
        vertices_indices=None,
    ):
        if vertices_indices is not None:
            vis = vertices_indices
        else:
            vis = self.vertex_visibility(camera, normal_threshold=normal_threshold, omni_directional_camera=True, binary_visiblity=True)
        faces_to_keep = list(filter(lambda face: vis[face[0]] * vis[face[1]] * vis[face[2]], self.f))
        vertex_indices_to_keep = np.nonzero(vis)[0]
        vertices_to_keep = self.v[vertex_indices_to_keep]
        old_to_new_indices = np.zeros(len(vis), dtype=int)
        old_to_new_indices[vertex_indices_to_keep] = range(len(vertex_indices_to_keep))

        new_mesh = Mesh(
            v=vertices_to_keep,
            f=np.array([old_to_new_indices[face] for face in faces_to_keep])
        )

        if hasattr(self, 'vt'):
            new_mesh.vt = self.vt[vertex_indices_to_keep]
        if hasattr(self, 'ft'):
            new_mesh.ft = np.array(old_to_new_indices[faces_to_keep])

        if return_texture:
            if not hasattr(self, '_texture_image'):
                return (new_mesh, None)
            else:

                # Uniquify the mesh first to get clear vertex - texture correspondence
                unique_mesh = self.uniquified_mesh()
                unique_vis = unique_mesh.vertex_visibility(
                    camera,
                    omni_directional_camera=True,
                    binary_visiblity=True
                )
                not_visible = ~ (unique_vis).astype(bool)
                texture_image = texture.edit_texture(unique_mesh, not_visible, color=color, criterion=criterion)
                return (new_mesh, texture_image)
        else:
            return new_mesh

    def create_texture_from_fc(self, texture_size=128):
        return texture.create_texture_from_fc(self, texture_size)

    def extract_patches_from_image(
        self, image, projection_camera=[1, 0, 0], visibility_camera=None, target_size=256, patches_dict=None
    ):
        if patches_dict is None:
            with open(
               SMPL_manual_patches,
                "r",
            ) as points_file:
                patches_dict = json.load(points_file)
        if visibility_camera is None:
            visibility_camera = projection_camera

        self.v = - self.v
        vis = self.vertex_visibility(
            visibility_camera,
            normal_threshold=None,
            omni_directional_camera=True,
            binary_visiblity=True
        )
        self.v = - self.v

        return texture.extract_image_patches(
            self,
            image,
            camera=projection_camera,
            target_size=target_size,
            patches_dict=patches_dict,
            visible_vertices=vis
        )

    def merge_texture_from_image(
        self,
        images: list,
        vertices: list,
        vis_cameras: list,
        proj_cameras=None,
        texture_size = 256,
        normal_threshold = -0.3,
        return_partial_textures = False,
    ):
        if proj_cameras is None:
            proj_cameras = vis_cameras

        assert len(images) == len(vertices), "Number of images and vertices must be the same"
        assert len(images) == len(vis_cameras), "Number of images and visibility cameras must be the same"
        assert len(images) == len(proj_cameras), "Number of images and projection cameras must be the same"

        merged_texture = np.random.random(size=(texture_size, texture_size, 3)) * 255
        partial_textures = [
            np.random.random(size=(texture_size, texture_size, 3)) * 255 for _ in range(4)
        ]

        # Do some shit here

        if return_partial_textures:
            return merged_texture, partial_textures
        else:
            return merged_texture
    
    def create_texture_from_image(
        self, 
        image,
        visibility_camera,
        projection_camera=None,
        texture_size=256,
        return_reprojection_image=False,
        verbose=True,
    ):
        if verbose:
            self.print("[DEBUG] Self mesh")

        h, w, c = image.shape
        assert h == w, "The function expects square image"

        assert hasattr(self, 'texturetype'), "Texture type must be specified"
        
        if projection_camera is None:
            projection_camera = visibility_camera
        
        unique_mesh = self.uniquified_mesh()
        if verbose:
            unique_mesh.print("[DEBUG] Unique mesh")

        # Create partial mesh by visibility function
        partial_mesh = unique_mesh.visibile_mesh(
            camera=visibility_camera,
            criterion=np.all,
        )
        if verbose:
            partial_mesh.print("[DEBUG] Partial mesh")

        # Unique mesh for easier sampling
        partial_mesh = partial_mesh.uniquified_mesh()
        if verbose:
            partial_mesh.print("[DEBUG] Partial mesh after uniquification")

        # Orthogonal projection
        partial_mesh.v = - partial_mesh.v
        raw_pts = partial_mesh.project_to_camera(camera=projection_camera)
        
        # Remove vertices that are not in the image
        valid_pts = np.all(raw_pts <= 1, axis=1)
        valid_pts = np.all(raw_pts >= -1, axis=1) & valid_pts
        raw_pts = raw_pts[valid_pts, :]
        
        # Remove invalid points from the mesh for consistency
        partial_mesh = partial_mesh.visibile_mesh(vertices_indices=valid_pts)
        partial_mesh.v = - partial_mesh.v

        # Draw reprojection image
        if return_reprojection_image:
            projected_image = image.copy()
            pts = raw_pts + 1
            pts *= (h/2)
            pts_int = pts.astype(int)
            projected_image[pts_int[:, 1], pts_int[:, 0], :] = (255, 0, 0)

        if verbose:
            partial_mesh.print("[DEBUG] 'Colored' mesh")

        start = time.perf_counter()
        new_texture = texture.create_texture_from_image(partial_mesh, raw_pts, image, texture_size=texture_size)
        stop = time.perf_counter()
        if verbose:
            print("The Texture commputation took {:.2f} seconds ({:.2f} minutes)".format(
                stop-start,
                (stop-start)/60
            ))
        
        if return_reprojection_image:
            return new_texture, projected_image
        else:
            return new_texture

    def estimate_circumference(self, plane_normal, plane_distance, partNamesAllowed=None, want_edges=False):
        raise Exception('estimate_circumference function has moved to body.mesh.metrics.circumferences')

    # ######################################################
    # Processing
    def reset_normals(self, face_to_verts_sparse_matrix=None, reset_face_normals=False):
        return processing.reset_normals(self, face_to_verts_sparse_matrix, reset_face_normals)

    def reset_face_normals(self):
        return processing.reset_face_normals(self)

    def keep_vertices(self, keep_list):
        return processing.keep_vertices(self, keep_list)

    def remove_vertices(self, v_list):
        return self.keep_vertices(np.setdiff1d(np.arange(self.v.shape[0]), v_list))

    def point_cloud(self):
        return Mesh(v=self.v, f=[], vc=self.vc) if hasattr(self, 'vc') else Mesh(v=self.v, f=[])

    def remove_faces(self, face_indices_to_remove):
        return processing.remove_faces(self, face_indices_to_remove)

    def scale_vertices(self, scale_factor):
        return processing.scale_vertices(self, scale_factor)

    def rotate_vertices(self, rotation):
        return processing.rotate_vertices(self, rotation)

    def translate_vertices(self, translation):
        return processing.translate_vertices(self, translation)

    def flip_faces(self):
        return processing.flip_faces(self)

    def simplified(self, factor=None, n_verts_desired=None):
        from .topology import qslim_decimator
        return qslim_decimator(self, factor, n_verts_desired)

    def subdivide_triangles(self):
        return processing.subdivide_triangles(self)

    def concatenate_mesh(self, mesh):
        return processing.concatenate_mesh(self, mesh)

    # new_ordering specifies the new index of each vertex. If new_ordering[i] = j,
    # vertex i should now be the j^th vertex. As such, each entry in new_ordering should be unique.
    def reorder_vertices(self, new_ordering, new_normal_ordering=None):
        processing.reorder_vertices(self, new_ordering, new_normal_ordering)

    # ######################################################
    # Landmark methods

    @property
    def landm_names(self):
        names = []
        if hasattr(self, 'landm_regressors') or hasattr(self, 'landm'):
            names = self.landm_regressors.keys() if hasattr(self, 'landm_regressors') else self.landm.keys()
        return list(names)

    @property
    def landm_xyz(self, ordering=None):
        landmark_order = ordering if ordering else self.landm_names
        landmark_vertex_locations = (self.landm_xyz_linear_transform(landmark_order) * self.v.flatten()).reshape(-1, 3) if landmark_order else np.zeros((0, 0))
        return dict([(landmark_order[i], xyz) for i, xyz in enumerate(landmark_vertex_locations)]) if landmark_order else {}

    def set_landmarks_from_xyz(self, landm_raw_xyz):
        self.landm_raw_xyz = landm_raw_xyz if hasattr(landm_raw_xyz, 'keys') else dict((str(i), l) for i, l in enumerate(landm_raw_xyz))
        self.recompute_landmark_indices()

    def landm_xyz_linear_transform(self, ordering=None):
        return landmarks.landm_xyz_linear_transform(self, ordering)

    def recompute_landmark_xyz(self):
        self.landm_raw_xyz = dict((name, self.v[ind]) for name, ind in self.landm.items())

    def recompute_landmark_indices(self, landmark_fname=None, safe_mode=True):
        landmarks.recompute_landmark_indices(self, landmark_fname, safe_mode)

    def set_landmarks_from_regressors(self, regressors):
        self.landm_regressors = regressors

    def set_landmark_indices_from_any(self, landmark_file_or_values):
        serialization.set_landmark_indices_from_any(self, landmark_file_or_values)

    def set_landmarks_from_raw(self, landmark_file_or_values):
        landmarks.set_landmarks_from_raw(self, landmark_file_or_values)

    def project_to_camera(self, camera=[1, 0, 0], focal_length=1000):
        return texture.project_to_camera(self, camera=camera)

    #######################################################
    # Texture methods

    @property
    def texture_image(self):
        if not hasattr(self, '_texture_image'):
            self.reload_texture_image()
        return self._texture_image

    def set_texture_image(self, path_to_texture):
        self.texture_filepath = path_to_texture
        self.reload_texture_image()

    def texture_coordinates_by_vertex(self):
        return texture.texture_coordinates_by_vertex(self)

    def reload_texture_image(self):
        texture.reload_texture_image(self)

    def transfer_texture(self, mesh_with_texture):
        texture.transfer_texture(self, mesh_with_texture)

    def load_texture(self, texture_version):
        texture.load_texture(self, texture_version)

    def texture_rgb(self, texture_coordinate):
        return texture.texture_rgb(self, texture_coordinate)

    def texture_rgb_vec(self, texture_coordinates):
        return texture.texture_rgb_vec(self, texture_coordinates)

    #######################################################
    # Search methods

    def compute_aabb_tree(self):
        return search.AabbTree(self)

    def compute_aabb_normals_tree(self):
        return search.AabbNormalsTree(self)

    def compute_closest_point_tree(self, use_cgal=False):
        return search.CGALClosestPointTree(self) if use_cgal else search.ClosestPointTree(self)

    def closest_vertices(self, vertices, use_cgal=False):
        return self.compute_closest_point_tree(use_cgal).nearest(vertices)

    def closest_points(self, vertices):
        return self.closest_faces_and_points(vertices)[1]

    def closest_faces_and_points(self, vertices):
        return self.compute_aabb_tree().nearest(vertices)

    #######################################################
    # Serialization methods

    def load_from_file(self, filename):
        serialization.load_from_file(self, filename)

    def load_from_ply(self, filename):
        serialization.load_from_ply(self, filename)

    def load_from_obj(self, filename):
        serialization.load_from_obj(self, filename)

    def write_json(self, filename, header="", footer="", name="", include_faces=True, texture_mode=True):
        serialization.write_json(self, filename, header, footer, name, include_faces, texture_mode)

    def write_three_json(self, filename, name=""):
        serialization.write_three_json(self, filename, name)

    def write_ply(self, filename, flip_faces=False, ascii=False, little_endian=True, comments=[]):
        serialization.write_ply(self, filename, flip_faces, ascii, little_endian, comments)

    def write_mtl(self, path, material_name, texture_name):
        """Serializes a material attributes file"""
        serialization.write_mtl(self, path, material_name, texture_name)

    def write_obj(self, filename, flip_faces=False, group=False, comments=None):
        serialization.write_obj(self, filename, flip_faces, group, comments)

    def load_from_obj_cpp(self, filename):
        serialization.load_from_obj_cpp(self, filename)

    def set_landmark_indices_from_ppfile(self, ppfilename):
        serialization.set_landmark_indices_from_ppfile(self, ppfilename)

    def set_landmark_indices_from_lmrkfile(self, lmrkfilename):
        serialization.set_landmark_indices_from_lmrkfile(self, lmrkfilename)
