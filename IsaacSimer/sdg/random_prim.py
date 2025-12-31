import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
from isaacsim.core.utils import prims, stage, xforms
from isaacsim.core.experimental.objects import RectLight
from isaacsim.core.experimental.materials import OmniPbrMaterial, PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Mesh
import uuid
from typing import Dict, List

class PrimRandomizer:
    def __init__(self) -> None:
        self._prims_dict = dict()         # Prims which are used to change pose
        self._visual_prims_dict: Dict[str, Mesh] = dict()  # Prims which are used to chage apperance
        self._rect_lights = None
        self._visual_materials: List[OmniPbrMaterial] = list()

        light_prims = []
        for prim in stage.traverse_stage():
            if prim.GetTypeName() == "RectLight":
                light_prims.append(str(prim.GetPath()))
        if not len(light_prims) == 0:
            self._rect_lights = RectLight(light_prims)

        self._light_count = len(light_prims)
        self._light_indices = list(range(0, self._light_count))

        self._material_count = 100
        self._create_random_pbr_materials(n=self._material_count)

        np.random.seed(123)

    def add_prim(self, prim_path: str, name: str):
        self._prims_dict[name] = prims.get_prim_at_path(prim_path)

    def add_mesh_prim(self, prim_path: str, name: str):
        self._visual_prims_dict[name] = Mesh(paths=[prim_path])

    def randomize_position(self, name: str, position_range: tuple, z = 0.0):
        """
        position_range[0]: x range;
        position_range[1]: y range.
        """
        attr = self._prims_dict[name].GetAttribute("xformOp:translate")
        if not attr.GetTypeName():
            attr = self._prims_dict[name].CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3)
        attr.Set(Gf.Vec3d(np.random.uniform(*position_range[0]), np.random.uniform(*position_range[1]), z))

    def randomize_position_relative_to(self, name: str, reference: str, position_range: tuple, z = 0.0):
        """
        position_range[0]: x range;
        position_range[1]: y range.
        """

        reference_position, _ = xforms.get_world_pose(self._prims_dict[reference].GetPath())
        # new_position = np.zeros(3, dtype=np.float32)
        # new_position[:2] =
        attr = self._prims_dict[name].GetAttribute("xformOp:translate")
        if not attr.GetTypeName():
            attr = self._prims_dict[name].CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3)

        attr.Set(Gf.Vec3d(reference_position[0] + np.random.uniform(*position_range[0]),
                reference_position[1] +np.random.uniform(*position_range[1]), z))

    def randomize_orientation(self, name: str, yaw_range: tuple):
        """
        yaw_range: unit: degree
        """
        orient_attr = self._prims_dict[name].GetAttribute("xformOp:orient")
        if not orient_attr.GetTypeName():
            orient_attr=self._prims_dict[name].CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf)

        rz = Gf.Rotation(Gf.Vec3d(0, 0, 1), np.random.uniform(*yaw_range))
        qd = rz.GetQuat()
        orient_attr.Set(Gf.Quatf(float(qd.GetReal()),
                                 float(qd.GetImaginary()[0]),
                                 float(qd.GetImaginary()[1]),
                                 float(qd.GetImaginary()[2])))

    def randomize_light(self):
        if self._rect_lights is not None:
            self._rect_lights.set_color_temperatures(np.random.uniform(1000.0, 8000.0, self._light_count))
            self._rect_lights.set_intensities(np.random.uniform(500.0, 30000.0, self._light_count))
            self._rect_lights.set_colors([np.random.rand(1, 3)] * self._light_count)

    def randomize_apperance(self, name: str):
        chosen_index = int(np.random.choice(self._material_count))
        self._visual_prims_dict[name].apply_visual_materials(materials=self._visual_materials[chosen_index])

    def _create_random_pbr_materials(self, n=4, base_path="/World/VisualMaterials") -> None:
        # 生成 N 个唯一的材质 prim 路径
        for i in range(n):
            path = f"{base_path}/rand_pbr_{i}_{uuid.uuid4().hex[:6]}"
            visual_material = OmniPbrMaterial(path)  # 会在 stage 上创建这些 OmniPBR 材质（若不存在）
            # 随机化基础参数（shape 要匹配 API 要求）
            colors = (np.random.rand(1, 3)).tolist()                         # diffuse_color_constant (N,3)
            metallic = (np.random.rand(1, 1) * 1.0).tolist()                 # metallic_constant (N,1)
            roughness = (np.clip(np.random.rand(1, 1), 0.02, 0.9)).tolist()  # reflection_roughness_constant (N,1)
            emissive_flag = [False]                                          # enable_emission (N,1)
            emissive_color = np.zeros((1, 3)).tolist()                       # emissive_color (N,3) - all zero by default

            # 写入材质输入
            visual_material.set_input_values(name="diffuse_color_constant", values=colors)
            visual_material.set_input_values(name="metallic_constant", values=metallic)
            visual_material.set_input_values(name="reflection_roughness_constant", values=roughness)
            visual_material.set_input_values(name="enable_emission", values=emissive_flag)
            visual_material.set_input_values(name="emissive_color", values=emissive_color)
            self._visual_materials.append(visual_material)
