import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.stage import get_current_stage
import random

class ObjRandomizer:
    def __init__(self, obj_prim_path: dict) -> None:
        # self._material_pool = None
        # self._pose_flag: bool = config["pose_flag"]
        # self._color_flag: bool = config["color_flag"]
        # self._material_flag: bool = config["material_flag"]

        # self._obj_prim_path = config["obj_prim_path"]

        stage = get_current_stage()
        self._obj_prim = stage.GetPrimAtPath(obj_prim_path)

        self.rep_obj_prim = rep.get.prim_at_path(obj_prim_path)

        self._mat_pool = rep.create.material_omnipbr(diffuse=rep.distribution.uniform((0.2, 0.1, 0.3), (0.6, 0.6, 0.7)),
                                                     roughness=random.uniform(0.1, 0.9),
                                                     metallic=random.uniform(0.1, 0.9),
                                                     count=100)
        
    @property
    def obj_position(self):
        return omni.usd.get_world_transform_matrix(self._obj_prim).ExtractTranslation()
    
    def _randomizeObj(self):
        with self.rep_obj_prim:
            # 颜色随机化（对当前材质做颜色扰动）
            rep.randomizer.color(colors=rep.distribution.uniform((0.01, 0.01, 0.01), (1.0, 1.0, 1.0)))
            # 材质随机化（从材质池里抽一个绑定）
            # rep.randomizer.materials(self._mat_pool)
            # 位姿随机化（位置/欧拉角/缩放）
            rep.modify.pose(
                position=rep.distribution.uniform((-15.0, -2.0, 0.0), (-5.0, 20.0, 0.0)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),  # 度
                # scale=rep.distribution.uniform((0.8, 0.8, 0.8), (1.2, 1.2, 1.2))
            )
            
        return self.rep_obj_prim.node