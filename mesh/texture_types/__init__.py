import os

_this_folder = os.path.dirname(os.path.abspath(__file__))

SMPL_objfile_path = os.path.join(_this_folder, "smpl_uv.obj")
BF_objfile_path = os.path.join(_this_folder, "bf_uv.obj")

__all__ = ["SMPL_objfile_path", "BF_objfile_path"]
