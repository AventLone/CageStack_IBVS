import carb
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Usd, UsdGeom, Gf
from typing import Dict, Optional, Tuple, Union
import numpy as np
import omni.usd
from scipy.spatial.transform import Rotation as R
from .logger_utils import LoggerUtil

# 创建logger实例
logger = LoggerUtil.get_logger("pose")

class TransformUtils:
    """
    统一的变换工具类，处理所有变换相关的操作，包括scale检查和修正
    """
    
    @staticmethod
    def get_validated_transform(transform: Gf.Matrix4d) -> Gf.Matrix4d:
        """
        验证变换矩阵的缩放比例，如果缩放不为1则修正为1
        
        Args:
            transform: 输入的4x4变换矩阵
            
        Returns:
            修正后的变换矩阵，缩放比例为1
        """
        # 提取平移和旋转分量
        translation = transform.ExtractTranslation()
        rotation = transform.ExtractRotation()
        
        # 手动计算缩放分量 - 从矩阵的对角线元素获取
        # 对于变换矩阵，缩放信息通常在对角线的前3个元素中
        scale_x = transform[0][0]
        scale_y = transform[1][1] 
        scale_z = transform[2][2]
        
        # 检查缩放是否接近1
        scale_magnitude = np.linalg.norm(np.array([scale_x, scale_y, scale_z]))
        
        if abs(scale_magnitude - 1.0) > 1e-6:
            logger.warning(f"检测到缩放问题，缩放幅度 = {scale_magnitude}，正在修正...")
            
            # 创建新的变换矩阵，缩放设置为1
            corrected_transform = Gf.Matrix4d(1.0)
            corrected_transform.SetRotate(rotation)
            corrected_transform.SetTranslateOnly(translation)
            
            logger.info("变换矩阵已修正，缩放比例设置为1")
            return corrected_transform
        
        return transform

    @staticmethod
    def get_prim_transform(prim_path: str, stage: Optional[Usd.Stage] = None) -> Optional[Gf.Matrix4d]:
        """
        获取Prim的变换矩阵，并进行scale验证
        
        Args:
            prim_path: USD prim路径
            stage: USD stage（如果为None，则使用当前stage）
            
        Returns:
            修正后的变换矩阵，或None如果失败
        """
        try:
            if stage is None:
                stage = omni.usd.get_context().get_stage()
            
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                logger.warning(f"Prim at path '{prim_path}' is not valid")
                return None
            
            # 使用UsdGeom.XformCache获取变换
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            
            if world_transform:
                # 验证并修正变换矩阵
                return TransformUtils.get_validated_transform(world_transform)
            else:
                logger.warning(f"XformCache failed to get transform for prim '{prim_path}'")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get transform for prim '{prim_path}': {str(e)}")
            return None

    @staticmethod
    def matrix_to_pose(transform: Gf.Matrix4d) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        将变换矩阵转换为位姿（平移和四元数）
        
        Args:
            transform: 4x4变换矩阵
            
        Returns:
            元组包含平移(3,)和四元数(4,)，或None如果失败
        """
        if not transform:
            return None
            
        try:
            translation = transform.ExtractTranslation()
            rotation = transform.ExtractRotation()
            orientation_quat = rotation.GetQuat()
            
            return (
                np.array([translation[0], translation[1], translation[2]]),
                np.array([orientation_quat.GetReal(), orientation_quat.GetImaginary()[0], 
                         orientation_quat.GetImaginary()[1], orientation_quat.GetImaginary()[2]])
            )
        except Exception as e:
            logger.error(f"Failed to convert matrix to pose: {e}")
            return None

    @staticmethod
    def pose_to_matrix(translation: np.ndarray, orientation: np.ndarray) -> Gf.Matrix4d:
        """
        将位姿转换为变换矩阵
        
        Args:
            translation: 平移向量(3,)
            orientation: 四元数(4,) [w, x, y, z]
            
        Returns:
            4x4变换矩阵
        """
        transform = Gf.Matrix4d(1)
        transform.SetRotate(Gf.Quatd(orientation[0].item(), Gf.Vec3d(*orientation[1:])))
        transform.SetTranslate(Gf.Vec3d(*translation))
        return transform

class Pose:
    """
    A utility to retrieve the pose (translation and orientation) of a specified prim.
    """
    def __init__(self, prim_path: str):
        self._prim_path = prim_path
        self._prim = get_prim_at_path(self._prim_path)
        if not self._prim.IsValid():
            logger.warning(f"Pose utility prim at path '{self._prim_path}' is not valid.")
        
        # 检测prim类型
        self._prim_type = self._detect_prim_type()

    def _detect_prim_type(self) -> str:
        """检测prim的类型以选择合适的API方法"""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._prim_path)
            
            if not prim.IsValid():
                return "unknown"
            
            # 检查是否是物理刚体
            if prim.HasAPI("PhysxRigidBodyAPI"):
                return "rigid_body"
            
            # 检查是否是关节系统
            if prim.HasAPI("PhysxArticulationAPI"):
                return "articulation"
            
            # 检查是否是相机
            if prim.GetTypeName() == "Camera":
                return "camera"
            
            # 检查是否是机器人
            prim_name = prim.GetName().lower()
            prim_path_lower = self._prim_path.lower()
            robot_keywords = ['robot', 'vehicle', 'forklift', 'car', 'truck', 'drone']
            
            # 检查prim名称或路径中是否包含机器人关键词
            if any(keyword in prim_name for keyword in robot_keywords) or \
               any(keyword in prim_path_lower for keyword in robot_keywords):
                return "robot"
            
            # 检查是否有变换能力
            if UsdGeom.Xformable(prim):
                return "xformable"
            
            return "generic"
            
        except Exception as e:
            carb.log_warn(f"Failed to detect prim type: {e}")
            return "unknown"

    def reset(self):
        """Clears the internal transform cache. This should be called once per frame."""
        pass

    def get_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieves the world pose of the prim.

        Returns:
            A tuple containing the translation (3,) and orientation quaternion (4,) as numpy arrays.
            Returns None if the prim is not valid.
        """
        if not self._prim or not self._prim.IsValid():
            carb.log_warn(f"Attempted to get pose from an invalid prim at '{self._prim_path}'")
            return None

        # 打印当前prim类型
        # logger.debug(f"Prim类型: {self._prim_type}")
        # # 打印当前prim路径
        # logger.debug(f"Prim路径: {self._prim_path}")

        if self._prim_type == "robot":
            # 机器人优先使用Robot API
            result = self._get_pose_robot_api()
            if result is not None:
                logger.debug("使用Robot API")
                logger.debug(f"Robot API result: {result}")
                return result

        if self._prim_type == "articulation":
            # 关节系统优先使用Articulation API
            result = self._get_pose_articulation()
            if result is not None:
                logger.debug("使用Articulation API")
                logger.debug(f"Articulation API result: {result}")
                return result

        result = self._get_pose_isaac_core()
        if result is not None or not np.all(result == 0):
            logger.debug("使用Isaac Core API")
            # logger.debug(f"Isaac Core API result: {result}")
            return result

        # 根据prim类型选择合适的方法
        if self._prim_type == "rigid_body":
            # 物理刚体优先使用RigidPrim API
            result = self._get_pose_rigid_prim()
            if result is not None:
                logger.debug("使用RigidPrim API")
                logger.debug(f"RigidPrim API result: {result}")
                return result
            
        if self._prim_type == "camera":
            # 相机优先使用Camera API
            result = self._get_pose_camera()
            if result is not None:
                logger.debug("使用Camera API")
                return result

        # 如果特定类型方法失败，尝试通用方法
        result = self._get_pose_xform_cache()
        if result is not None:
            logger.debug("使用XformCache API")
            return result
            
        result = self._get_pose_direct_xformable()
        if result is not None:
            logger.debug("使用Direct Xformable API")
            return result

        logger.warning("所有API方法均失败")
        return None

    def get_pose_with_comparison(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取位姿并打印不同方法的对比结果（用于调试）
        根据prim类型只执行相关的方法，保持简洁性
        """
        if not self._prim or not self._prim.IsValid():
            carb.log_warn(f"Attempted to get pose from an invalid prim at '{self._prim_path}'")
            return None

        results = {}
        
        # 总是尝试通用方法
        results['isaac_core'] = self._get_pose_isaac_core()
        results['xform_cache'] = self._get_pose_xform_cache()
        results['direct_xformable'] = self._get_pose_direct_xformable()
        results['articulation'] = self._get_pose_articulation()
        results['robot_api'] = self._get_pose_robot_api()
        # 根据prim类型选择特定的API方法
        if self._prim_type == "robot":
            results['robot_api'] = self._get_pose_robot_api()
        # 1.物理引擎限制：PhysX不允许在启用的刚体下再有其他启用的刚体
        # 2.性能优化：单个刚体比多个刚体层级更高效
        # 3.模拟稳定性：避免复杂的物理交互导致的不可预测行为
        elif self._prim_type == "rigid_body":
            results['rigid_prim'] = self._get_pose_rigid_prim()
        elif self._prim_type == "articulation":
            results['articulation'] = self._get_pose_articulation()
        elif self._prim_type == "camera":
            results['camera'] = self._get_pose_camera()
        
        # 打印对比结果
        self._print_pose_comparison(results)
        
        # 返回第一个有效的结果
        for method_name, result in results.items():
            if result is not None:
                logger.debug(f"Using pose from {method_name} method")
                logger.debug(f"result: {result}")
        
        return None

    def _get_pose_isaac_core(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """使用Isaac Sim 5.0的Core API获取位姿"""
        try:
            from omni.isaac.core.utils.xforms import get_world_pose
            position, orientation = get_world_pose(self._prim_path)
            
            if position is not None and orientation is not None:
                return (
                    np.array(position, dtype=np.float64),
                    np.array(orientation, dtype=np.float64)  # [w, x, y, z]格式
                )
            else:
                carb.log_warn(f"Isaac Core get_world_pose failed for prim '{self._prim_path}'")
                return None
                
        except ImportError:
            carb.log_info("Isaac Core API not available")
            return None
        except Exception as e:
            carb.log_warn(f"Isaac Core API failed: {e}")
            return None

    def _get_pose_xform_cache(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """使用UsdGeom.XformCache获取位姿"""
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                carb.log_error("Unable to get USD stage")
                return None
            
            prim = stage.GetPrimAtPath(self._prim_path)
            if not prim.IsValid():
                carb.log_error(f"Prim at path '{self._prim_path}' is not valid")
                return None
            
            # 使用UsdGeom.XformCache获取变换
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            
            if world_transform:
                # 验证并修正变换矩阵
                validated_transform = TransformUtils.get_validated_transform(world_transform)
                return TransformUtils.matrix_to_pose(validated_transform)
            else:
                carb.log_error(f"Failed to get transform matrix for prim '{self._prim_path}'")
                return None
                
        except Exception as e:
            carb.log_error(f"UsdGeom.XformCache method failed: {e}")
            return None

    def _get_pose_direct_xformable(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """直接从Xformable获取位姿"""
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._prim_path)
            
            if UsdGeom.Xformable(prim):
                xformable = UsdGeom.Xformable(prim)
                
                # 获取世界变换矩阵，而不是本地变换
                xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                world_transform = xform_cache.GetLocalToWorldTransform(prim)
                
                if world_transform:
                    validated_transform = TransformUtils.get_validated_transform(world_transform)
                    return TransformUtils.matrix_to_pose(validated_transform)
            
            carb.log_error(f"Direct Xformable method failed for prim '{self._prim_path}'")
            return None
            
        except Exception as e:
            carb.log_error(f"Direct Xformable method failed: {e}")
            return None

    def _get_pose_robot_api(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """使用Robot API获取位姿"""
        try:
            # 尝试使用Robot API (如果可用)
            from omni.isaac.core.robots import Robot
            from omni.isaac.core.utils.prims import is_prim_path_valid
            
            if is_prim_path_valid(self._prim_path):
                robot = Robot(prim_path=self._prim_path)
                if hasattr(robot, 'get_world_pose'):
                    position, orientation = robot.get_world_pose()
                    if position is not None and orientation is not None:
                        return (
                            np.array(position, dtype=np.float64),
                            np.array(orientation, dtype=np.float64)
                        )
                
                # 备用方法：通过Robot的关节或链接获取
                if hasattr(robot, 'articulation_root'):
                    root = robot.articulation_root
                    if root:
                        position = root.get_world_pose()[0]
                        orientation = root.get_world_pose()[1]
                        if position is not None and orientation is not None:
                            return (
                                np.array(position, dtype=np.float64),
                                np.array(orientation, dtype=np.float64)
                            )
            
            carb.log_info("Robot API method not available or failed")
            return None
            
        except ImportError:
            carb.log_info("Robot API not available")
            return None
        except Exception as e:
            carb.log_warn(f"Robot API failed: {e}")
            return None

    def _get_pose_rigid_prim(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """使用RigidPrim API获取位姿（适用于物理刚体）"""
        try:
            from omni.isaac.core.prims import RigidPrim
            from omni.isaac.core.utils.prims import is_prim_path_valid
            
            if is_prim_path_valid(self._prim_path):
                rigid_prim = RigidPrim(prim_path=self._prim_path)
                if hasattr(rigid_prim, 'get_world_pose'):
                    position, orientation = rigid_prim.get_world_pose()
                    if position is not None and orientation is not None:
                        return (
                            np.array(position, dtype=np.float64),
                            np.array(orientation, dtype=np.float64)
                        )
            
            carb.log_info("RigidPrim API method not available or failed")
            return None
            
        except ImportError:
            carb.log_info("RigidPrim API not available")
            return None
        except Exception as e:
            carb.log_warn(f"RigidPrim API failed: {e}")
            return None

    def _get_pose_articulation(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """使用Articulation API获取位姿（适用于机器人关节）"""
        try:
            from omni.isaac.core.articulations import Articulation
            from omni.isaac.core.utils.prims import is_prim_path_valid
            
            if is_prim_path_valid(self._prim_path):
                articulation = Articulation(prim_path=self._prim_path)
                if hasattr(articulation, 'get_world_pose'):
                    position, orientation = articulation.get_world_pose()
                    if position is not None and orientation is not None:
                        return (
                            np.array(position, dtype=np.float64),
                            np.array(orientation, dtype=np.float64)
                        )
                
                # 备用方法：获取基座标位姿
                if hasattr(articulation, 'get_base_pose'):
                    position, orientation = articulation.get_base_pose()
                    if position is not None and orientation is not None:
                        return (
                            np.array(position, dtype=np.float64),
                            np.array(orientation, dtype=np.float64)
                        )
            
            carb.log_info("Articulation API method not available or failed")
            return None
            
        except ImportError:
            carb.log_info("Articulation API not available")
            return None
        except Exception as e:
            carb.log_warn(f"Articulation API failed: {e}")
            return None

    def _get_pose_camera(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """使用Camera API获取位姿（适用于相机传感器）"""
        try:
            from omni.isaac.sensor import Camera
            from omni.isaac.core.utils.prims import is_prim_path_valid
            
            if is_prim_path_valid(self._prim_path):
                camera = Camera(prim_path=self._prim_path)
                if hasattr(camera, 'get_world_pose'):
                    position, orientation = camera.get_world_pose()
                    if position is not None and orientation is not None:
                        return (
                            np.array(position, dtype=np.float64),
                            np.array(orientation, dtype=np.float64)
                        )
            
            carb.log_info("Camera API method not available or failed")
            return None
            
        except ImportError:
            carb.log_info("Camera API not available")
            return None
        except Exception as e:
            carb.log_warn(f"Camera API failed: {e}")
            return None

    def _print_pose_comparison(self, results: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]]):
        """打印不同方法的位姿对比结果"""
        logger.info(f"\n=== Pose Comparison for '{self._prim_path}' ===")
        
        valid_methods = {}
        for method_name, result in results.items():
            if result is not None:
                position, orientation = result
                valid_methods[method_name] = {
                    'position': position,
                    'orientation': orientation
                }
                logger.info(f"{method_name}: position={position}, orientation={orientation}")
            else:
                logger.warning(f"{method_name}: FAILED")
        
        # 如果有多个有效方法，比较它们之间的差异
        if len(valid_methods) > 1:
            logger.info("\n--- Differences between methods ---")
            method_names = list(valid_methods.keys())
            
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    method1 = method_names[i]
                    method2 = method_names[j]
                    
                    pos1 = valid_methods[method1]['position']
                    pos2 = valid_methods[method2]['position']
                    ori1 = valid_methods[method1]['orientation']
                    ori2 = valid_methods[method2]['orientation']
                    
                    pos_diff = np.linalg.norm(pos1 - pos2)
                    ori_diff = np.arccos(2 * np.dot(ori1, ori2)**2 - 1)  # 角度差异
                    
                    logger.info(f"{method1} vs {method2}: pos_diff={pos_diff:.6f}, ori_diff={np.degrees(ori_diff):.6f}°")
        
        logger.info("=== End of Comparison ===\n")

# 保留相对位姿计算功能作为静态工具方法
def get_relative_pose(base_pose: Dict[str, np.ndarray], target_pose: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    计算目标相对于基座的位姿
    
    Args:
        base_pose: 基座标位姿 ({'translation': ..., 'orientation': ...})
        target_pose: 目标位姿 ({'translation': ..., 'orientation': ...})
        
    Returns:
        包含相对'translation'和'orientation'的字典
    """
    # 使用统一的变换工具创建变换矩阵
    base_transform = TransformUtils.pose_to_matrix(base_pose['translation'], base_pose['orientation'])
    target_transform = TransformUtils.pose_to_matrix(target_pose['translation'], target_pose['orientation'])

    # Relative transform = Inverse(Base) * Target
    relative_transform = base_transform.GetInverse() * target_transform

    # 使用统一的变换工具转换回位姿
    relative_pose = TransformUtils.matrix_to_pose(relative_transform)
    if relative_pose:
        return {
            'translation': relative_pose[0],
            'orientation': relative_pose[1]
        }
    else:
        logger.error("Failed to calculate relative pose")
        return {'translation': np.zeros(3), 'orientation': np.array([1.0, 0.0, 0.0, 0.0])}

class PalletInitializer:
    """
    Handles pallet pose initialization functionality based on static USD data.
    This class is designed to set the initial pose of pallets relative to a vehicle
    before the physics simulation starts.
    """
    
    def __init__(self, vehicle_prim_path: str):
        """
        Initialize the PalletInitializer.
        
        Args:
            vehicle_prim_path: USD prim path of the vehicle for relative positioning.
        """
        if not vehicle_prim_path:
            raise ValueError("vehicle_prim_path cannot be None or empty.")
        self._vehicle_prim_path = vehicle_prim_path
        self._stage = omni.usd.get_context().get_stage()
    
    def _get_prim_static_transform(self, prim_path: str) -> Optional[Gf.Matrix4d]:
        """
        Get the static world transform of a prim using unified transform utils.
        
        Args:
            prim_path: The USD prim path of the object.
            
        Returns:
            A Gf.Matrix4d representing the world transform, or None if not found.
        """
        return TransformUtils.get_prim_transform(prim_path, self._stage)

    def initialize_pallet_statically(self, pallet_config: dict) -> bool:
        """
        Initialize pallet pose with front face center relative to vehicle steering center.
        
        Args:
            pallet_config: Dictionary containing 'prim_path' and 'initial_pose'.
                         
        Returns:
            True if initialization is successful, False otherwise.
        """
        prim_path = pallet_config.get('prim_path')
        initial_pose = pallet_config.get('initial_pose')
        
        if not prim_path or not initial_pose:
            logger.warning(f"Missing prim_path or initial_pose for pallet.")
            return False

        # Get vehicle transform using unified method
        vehicle_transform = self._get_prim_static_transform(self._vehicle_prim_path)
        if vehicle_transform is None:
            logger.error("Failed to get vehicle transform")
            return False
            
        vehicle_position = vehicle_transform.ExtractTranslation()
        vehicle_rotation = vehicle_transform.ExtractRotationMatrix()
        
        # Calculate relative position from config
        relative_pos = np.array(initial_pose.get('translation', [0, 0, 0]))
        relative_quat = initial_pose.get('orientation', [1, 0, 0, 0])  # w,x,y,z
        
        # Apply vehicle orientation to relative position
        world_relative_pos = vehicle_rotation * Gf.Vec3d(*relative_pos)
        
        # Get pallet dimensions for front face offset calculation
        pallet_dimensions = self._get_pallet_dimensions(prim_path)
        
        # Calculate front face center offset (assuming front is -Y direction)
        front_face_offset = np.array([0, pallet_dimensions[1]/2, 0])
        world_front_offset = vehicle_rotation * Gf.Vec3d(*front_face_offset)
        
        # Get vehicle steering center position
        steering_center = self._get_steering_center_position()
        if steering_center is None:
            logger.error("Failed to get vehicle steering center position")
            return False
        
        # Final pallet position calculation
        world_relative_pos_array = np.array([world_relative_pos[0], world_relative_pos[1], world_relative_pos[2]])
        world_front_offset_array = np.array([world_front_offset[0], world_front_offset[1], world_front_offset[2]])
        
        final_position = steering_center + world_relative_pos_array - world_front_offset_array
        
        # Set pallet transform
        return self._set_pallet_transform(prim_path, final_position, relative_quat, vehicle_transform)
    
    def _get_steering_center_position(self) -> Optional[np.ndarray]:
        """Get vehicle steering center position (front axle center)."""
        try:
            vehicle_transform = self._get_prim_static_transform(self._vehicle_prim_path)
            if vehicle_transform is None:
                logger.error("无法获取车辆变换矩阵")
                return None
                
            vehicle_pos = vehicle_transform.ExtractTranslation()
            vehicle_rotation = vehicle_transform.ExtractRotationMatrix()
            
            # Default wheelbase for steering center calculation
            wheelbase = 1.044  # meters
            local_offset = Gf.Vec3d(wheelbase, 0, 0)  # forward offset
            
            world_offset = vehicle_rotation * local_offset
            
            steering_center = np.array([vehicle_pos[0] + world_offset[0], 
                                      vehicle_pos[1] + world_offset[1], 
                                      0.0])  # Keep Z at ground level
            return steering_center
        except Exception as e:
            logger.error(f"Failed to calculate steering center: {e}")
            return None
    
    def _get_pallet_dimensions(self, prim_path: str) -> np.ndarray:
        """Get pallet dimensions, return default if failed."""
        try:
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                bbox = bbox_cache.ComputeWorldBound(prim)
                if not bbox.IsEmpty():
                    bbox_range = bbox.ComputeAlignedRange()
                    size = bbox_range.GetSize()
                    return np.array([size[0], size[1], size[2]])
        except Exception as e:
            logger.warning(f"Failed to get pallet dimensions: {e}")
        
        # 默认托盘尺寸
        return np.array([1.2, 1.0, 0.15])  # 长、宽、高
    
    def _set_pallet_transform(self, prim_path: str, position: np.ndarray, 
                             orientation: np.ndarray, vehicle_transform: Gf.Matrix4d) -> bool:
        """Set pallet transform using unified transform utils."""
        try:
            # 使用统一的变换工具设置位姿
            pallet_prim = self._stage.GetPrimAtPath(prim_path)
            if not pallet_prim.IsValid():
                logger.warning(f"Pallet prim at path '{prim_path}' is not valid")
                return False
            
            # 创建变换矩阵
            pallet_transform = TransformUtils.pose_to_matrix(position, orientation)
            
            # 设置prim的变换
            xformable = UsdGeom.Xformable(pallet_prim)
            ops = xformable.GetOrderedXformOps()
            
            # 清除现有变换操作
            while len(ops) > 0:
                xformable.RemoveXformOp(ops[0])
                ops = xformable.GetOrderedXformOps()
            
            # 添加新的变换操作
            xformable.AddTransformOp().Set(pallet_transform)
            
            logger.info(f"Successfully initialized pallet at '{prim_path}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set pallet transform: {e}")
            return False
        
        return True
