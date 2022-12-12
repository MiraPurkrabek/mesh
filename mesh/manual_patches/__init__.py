import os

_this_folder = os.path.dirname(os.path.abspath(__file__))

SMPL_manual_patches = os.path.join(_this_folder, "smpl_patches.json")

__all__ = ["SMPL_manual_patches"]