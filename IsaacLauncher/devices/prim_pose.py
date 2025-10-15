import carb
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Usd, UsdGeom, Gf
from typing import Dict, Optional, Tuple
import numpy as np
import omni.usd
from scipy.spatial.transform import Rotation as R

from utils.world_model import TransformUtils, Pose

class PrimPose:
    """
    A utility to retrieve the pose (translation and orientation) of a specified prim.
    """
    def __init__(self, prim_path: str):
        self._prim_path = prim_path
        self._pose_util = Pose(prim_path)

    def reset(self):
        """
        Clears the internal transform cache. This should be called once per frame.
        """
        # Pose类已处理缓存清理
        pass

    def get_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieves the world pose of the prim.

        Returns:
            A tuple containing the translation (3,) and orientation quaternion (4,) as numpy arrays.
            Returns None if the prim is not valid.
        """
        return self._pose_util.get_pose()

    def set_pose(self, translation: np.ndarray, orientation: np.ndarray):
        """
        Sets the world pose of the prim.

        Args:
            translation: Translation vector as numpy array (3,)
            orientation: Orientation quaternion as numpy array (4,)
        """
        if not self._prim or not self._prim.IsValid():
            carb.log_warn(f"Attempted to set pose on an invalid prim at '{self._prim_path}'")
            return

        xformable = UsdGeom.Xformable(self._prim)
        xform_op = xformable.AddTransformOp()
        
        # Create transform matrix from translation and rotation
        translation_vec = Gf.Vec3d(translation[0], translation[1], translation[2])
        rotation_quat = Gf.Quatd(orientation[0], Gf.Vec3d(orientation[1], orientation[2], orientation[3]))
        rotation_mat = Gf.Matrix4d().SetRotate(rotation_quat)
        
        transform_mat = Gf.Matrix4d()
        transform_mat.SetTranslate(translation_vec)
        transform_mat = transform_mat * rotation_mat
        
        xform_op.Set(transform_mat)

    def get_size(self) -> Optional[np.ndarray]:
        """
        Retrieves the bounding box size of the prim.

        Returns:
            Size vector as numpy array (3,) or None if prim is invalid
        """
        if not self._prim or not self._prim.IsValid():
            carb.log_warn(f"Attempted to get size from an invalid prim at '{self._prim_path}'")
            return None

        # Get bounding box in local space
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        bbox = bbox_cache.ComputeWorldBound(self._prim.GetPath())
        
        if bbox.GetRange().IsEmpty():
            return np.array([0.0, 0.0, 0.0])
            
        size = bbox.GetRange().GetSize()
        return np.array([size[0], size[1], size[2]])

    def set_size(self, size: np.ndarray):
        """
        通过计算当前尺寸和期望尺寸之间的缩放比例因子来设置prim的尺寸。
        
        Args:
            size: 期望的尺寸向量，numpy数组 (3,)
        """
        if not self._prim or not self._prim.IsValid():
            carb.log_warn(f"Attempted to set size on an invalid prim at '{self._prim_path}'")
            return

        # 获取当前尺寸
        current_size = self.get_size()
        if current_size is None or np.any(current_size == 0):
            carb.log_warn(f"Cannot set size for prim at '{self._prim_path}' - invalid current size")
            return

        # 计算缩放比例因子
        scale_factors = size / current_size
        
        # 应用缩放操作
        xformable = UsdGeom.Xformable(self._prim)
        
        # 清除现有的缩放操作
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                xformable.ClearXformOpOrder()
                break
        
        # 添加新的缩放操作并设置缩放因子
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(scale_factors[0], scale_factors[1], scale_factors[2]))
       