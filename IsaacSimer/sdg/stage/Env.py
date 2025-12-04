import numpy as np
import os
import time
import random
import math
import json
import carb
from tqdm import tqdm
import glob
from typing import List, Tuple

# 导入其他必要的模块
from isaacsim.core.api import World
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage, add_reference_to_stage
from isaacsim.core.utils.bounds import compute_aabb, compute_obb, create_bbox_cache, get_obb_corners
from isaacsim.storage.native import get_assets_root_path

import omni.replicator.core as rep
import omni.usd
from omni.physx import get_physx_simulation_interface
from pxr import (
    Gf,
    Vt,
    Tf,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdSkel,
    UsdShade,
    UsdUtils,
    Semantics,
    PhysicsSchemaTools,
    PhysxSchema,
)

from imgMap import MaskManager


class enviroment:
    def __init__(self, pathList, config, science=0, writeResult=True):
        """
        随机环境生成
        Args:
            pathList (_type_): 车辆运行轨迹(货架场景)
            config (_type_): 配置文件
            science (int, optional): 明眸场景(0)，货架场景(1). Defaults to 0.
            writeResult (bool, optional): 是否保存图片. Defaults to True.
        """
        # Create a stage
        omni.usd.get_context().new_stage()
        # 生成一个高质量随机种子，取 32 位以内
        seed = int.from_bytes(os.urandom(8), "little") % (2**32)
        print(f"[INFO] Using random seed: {seed}")

        random.seed(seed)
        np.random.seed(seed)
        rep.set_global_seed(seed)
        self.stage = omni.usd.get_context().get_stage()
        self.world = World(stage_units_in_meters=1.0)
        self.path = pathList
        self.config = config
        self.science=science
        self.WIDTH_, self.LENGTH_, self.HEIGHT_ = self.config["warehouseSize"]
        self.envxform = UsdGeom.Xform.Define(self.stage, "/World/env")
        self.Shelfxform = UsdGeom.Xform.Define(self.stage, "/World/env/shelf")
        self.Goodsxform = UsdGeom.Xform.Define(self.stage, "/World/env/goods")
        self.Shelfgoodsxform = UsdGeom.Xform.Define(self.stage, "/World/env/shelf/goods")
        self.movxform = UsdGeom.Xform.Define(self.stage, "/World/mobile")
        self.Peoxform = UsdGeom.Xform.Define(self.stage, "/World/Characters")
        self.Forkxform = UsdGeom.Xform.Define(self.stage, "/World/mobile/forklift")
        self.Testxform = UsdGeom.Xform.Define(self.stage, "/World/test")
        self.distantLight = rep.create.light(position=(0, 0, 1000), look_at=(0,0,0), intensity=3000, light_type="distant")
        self.assets_root_path = "/home/visionnav/application/isaacSim/isaac-sim-assets-environments-5.0.0/Assets/Isaac/5.0"
        self.id = 0 # 用于测量尺寸
        self.InitMap()
        self.GoodstmplePaths = {}
        self.ReferObjPaths = {}
        
        self.rps = []
        self.move_rps = []
    
    def generate_controlled_random_color(self, color_type="natural"):
        """
        生成受控的随机颜色，避免过于刺眼的颜色
        
        参数:
        color_type: "natural"自然光, "warm"暖色, "cool"冷色, "any"任何颜色
        """
        if color_type == "natural":
            # 自然光范围：偏白色和浅黄色
            r = random.uniform(0.8, 1.0)
            g = random.uniform(0.8, 1.0)
            b = random.uniform(0.6, 0.9)
        elif color_type == "warm":
            # 暖色调：红黄为主
            r = random.uniform(0.7, 1.0)
            g = random.uniform(0.4, 0.8)
            b = random.uniform(0.0, 0.3)
        elif color_type == "cool":
            # 冷色调：蓝绿为主
            r = random.uniform(0.0, 0.3)
            g = random.uniform(0.4, 0.8)
            b = random.uniform(0.7, 1.0)
        else:
            # 自然光范围：偏白色和浅黄色
            r = random.uniform(0.8, 1.0)
            g = random.uniform(0.8, 1.0)
            b = random.uniform(0.6, 0.9)
        
        return (r, g, b)
        
    def CreateLight(self):
        color = self.generate_controlled_random_color()
        intensity = random.uniform(500, 10000)
        # exposure = random.uniform(1, 0.5)
        exposure = 1
        temperature = random.gauss(6500, 2000)
        while temperature < 0:
            temperature = random.gauss(6500, 4000)


        look_xy  = (random.uniform(-1, 1), random.uniform(-1, 1))
        scale_xy = (random.uniform(0, 10), random.uniform(0, 10))
        self.rectlight = rep.create.light(
            position=(0, 0, 6), 
            look_at= (look_xy[0], look_xy[1], 0), 
            scale = (self.WIDTH_ - scale_xy[0], self.LENGTH_ - scale_xy[1], 1),
            color = color, 
            intensity=intensity,
            exposure = exposure,
            temperature = temperature,
            light_type="rect")

    def set_or_update_semantics(self, prim, new_label, semantic_type="class"):
        # 获取或应用 API
        flag = -1
        for prop in prim.GetProperties():
            is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath())
            if is_semantic:
                name = prop.SplitName()[1]
                sem = Semantics.SemanticsAPI.Get(prim, name)
                # sem = Semantics.SemanticsAPI.Get(prim, "Semantics")
                # if not sem:
                #     sem = Semantics.SemanticsAPI.Get(prim, "Semantics_yMv0")
                # if not sem:
                #     sem = Semantics.SemanticsAPI.Get(prim, "Semantics_80x0")
                flag = 1

        if flag == -1:
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        # 确保属性存在
        type_attr = sem.GetSemanticTypeAttr()
        data_attr = sem.GetSemanticDataAttr()
        if not type_attr:
            type_attr = sem.CreateSemanticTypeAttr()
        if not data_attr:
            data_attr = sem.CreateSemanticDataAttr()

        # 修改语义属性
        type_attr.Set(semantic_type)
        data_attr.Set(new_label)
    
    def set_semantics_recursive(self, root_prim, label: str):
        """
        递归为 root_prim 及其所有子节点设置语义。
        """
        if not root_prim or not root_prim.IsValid():
            return
        
        def trace_child(base_xform):
            for child in base_xform.GetChildren():
                if child.GetTypeName() == "Xform":
                    trace_child(child)
                
                if child.GetTypeName() == "Mesh":
                    self.set_or_update_semantics(child, label)
        
        trace_child(root_prim)
                    
                
                
    def CreateWarehouse(self):
        self.CreateLight()
        y_l = self.LENGTH_ / 2
        x_l = self.WIDTH_ / 2
        # floor count
        count = 1
        for x in np.arange(-x_l,x_l,6):
            for y in np.arange(-y_l,y_l,6):
                
                position = (x+3, y+3, 0)
                position_cell = (x+3, y+3, 9)
                floor_prim = prims.create_prim(
                    prim_path=f"/World/env/Floor_{count}",
                    position=position,
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path + self.config["floor"]["url"],
                    semantic_label=self.config["floor"]["class"],
                )
                self.set_semantics_recursive(floor_prim, self.config["floor"]["class"])
                self.add_colliders(floor_prim)
                
                cell_prim = prims.create_prim(
                    prim_path=f"/World/env/Cell_{count}",
                    position=position_cell,
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path + self.config["cell"]["url"],
                    semantic_label=self.config["cell"]["class"],
                )
                self.set_semantics_recursive(cell_prim, self.config["cell"]["class"])
                count += 1

        #wall_A
        count = 1
        left_num = int(np.ceil(self.WIDTH_ / 6))
        front_num = int(np.ceil(self.LENGTH_ / 6))
        pillarLen_num = int(np.ceil((self.WIDTH_ -6) / 9)) if self.WIDTH_ > 6 else 0
        beam_num = int(np.ceil(self.LENGTH_ / 9))

        count_all = 2 * (left_num + front_num)

        for i in range(count_all):
            if i < left_num:
                if 6 * (i +1) <= self.WIDTH_:
                    x = - (self.WIDTH_ / 2) + 6 * i + 3
                else:
                    x =  (self.WIDTH_ / 2) - 3
                y = - (self.LENGTH_ /2)
                rotation_wall = [0, 0, np.pi/2]
            elif i < left_num + front_num:
                if  6 * (i +1 - left_num)  <= self.LENGTH_:
                    y = - (self.LENGTH_ / 2) + 6 * ((i - left_num)) + 3
                else:
                    y = (self.LENGTH_ / 2) - 3
                x = (self.WIDTH_ /2)
                rotation_wall = [0, 0, np.pi]       
            elif i <  2 * left_num + front_num:
                if 6 * (i +1 - left_num - front_num) <= self.WIDTH_:
                    x = - (self.WIDTH_ / 2) + 6 * (i - left_num - front_num) + 3
                else:
                    x =  (self.WIDTH_ / 2) - 3
                y = (self.LENGTH_ /2)
                rotation_wall = [0, 0, -np.pi/2]  
            else:
                if  6 * (i +1 - 2 * left_num - front_num)  <= self.LENGTH_:
                    y = - (self.LENGTH_ / 2) + 6 * ((i - 2 * left_num - front_num)) + 3
                else:
                    y = (self.LENGTH_ / 2) - 3
                x = -(self.WIDTH_ /2) 
                rotation_wall = [0, 0, 0]  
            
            position_wall_a = (x, y, 0)
            position_wall_b = (x, y, 3)
            position_wall_b_up = (x, y, 6)
            position_pillar = (x, y, 0)
            
            wallA_prim = prims.create_prim(
                prim_path=f"/World/env/wall_A_{i}",
                position=position_wall_a,
                orientation=euler_angles_to_quat(rotation_wall),
                usd_path=self.assets_root_path + self.config["walldown_B"]["url"],
                semantic_label=self.config["walldown_B"]["class"],
            )
            self.set_semantics_recursive(wallA_prim, self.config["walldown_B"]["class"])
            wallB_prim = prims.create_prim(
                prim_path=f"/World/env/wall_B_{i}",
                position=position_wall_b,
                orientation=euler_angles_to_quat(rotation_wall),
                usd_path=self.assets_root_path + self.config["walluppon_B"]["url"],
                semantic_label=self.config["walluppon_B"]["class"],
            )
            self.set_semantics_recursive(wallB_prim, self.config["walluppon_B"]["class"])
            wallB_up_prim = prims.create_prim(
                prim_path=f"/World/env/wall_B_up_{i}",
                position=position_wall_b_up,
                orientation=euler_angles_to_quat(rotation_wall),
                usd_path=self.assets_root_path + self.config["walluppon_B"]["url"],
                semantic_label=self.config["walluppon_B"]["class"],
            )
            self.set_semantics_recursive(wallB_up_prim, self.config["walluppon_B"]["class"])
            if(i < left_num-1):
                for t in range(beam_num):
                    if (-self.LENGTH_/2 + t * 9) > self.LENGTH_ / 2 - 9:
                        beam_position = (-self.WIDTH_/2 + (i+1) * 6, self.LENGTH_/2 - 9, 9)
                    else: 
                        beam_position = (-self.WIDTH_/2 + (i+1) * 6, -self.LENGTH_/2 +  t * 9, 9) 
                    beam_prim = prims.create_prim(
                        prim_path=f"/World/env/Beam_{i}_{t}",
                        position=beam_position,
                        orientation=euler_angles_to_quat([0, 0, np.pi/2]),
                        usd_path=self.assets_root_path + self.config["beam"]["url"],
                        semantic_label=self.config["beam"]["class"],
                    )
                    self.set_semantics_recursive(beam_prim, self.config["beam"]["class"])
            if (left_num < i < left_num + front_num - 1):
                pillar_prim = prims.create_prim(
                    prim_path=f"/World/env/Pillar_{i}",
                    position=position_pillar,
                    orientation=euler_angles_to_quat(rotation_wall),
                    usd_path=self.assets_root_path + self.config["pillar"]["url"],
                    semantic_label=self.config["pillar"]["class"],
                )
                self.set_semantics_recursive(pillar_prim, self.config["pillar"]["class"])
                for t in range(pillarLen_num):
                    position_pillar_len = (3 + (t+1) * 9 - self.WIDTH_ / 2, y, 9)
                    pillar_len_prim = prims.create_prim(
                        prim_path=f"/World/env/Pillar_len_{i}_{t}",
                        position=position_pillar_len,
                        orientation=euler_angles_to_quat([0, -np.pi/2, 0]),
                        usd_path=self.assets_root_path + self.config["pillarLen"]["url"],
                        semantic_label=self.config["pillarLen"]["class"],
                    )
                    self.set_semantics_recursive(pillar_len_prim, self.config["pillarLen"]["class"])
                    
                    
            elif (2 * left_num + front_num < i < count_all - 1):
                pillar_prim = prims.create_prim(
                    prim_path=f"/World/env/Pillar_{i}",
                    position=position_pillar,
                    orientation=euler_angles_to_quat(rotation_wall),
                    usd_path=self.assets_root_path + self.config["pillar"]["url"],
                    semantic_label=self.config["pillar"]["class"],
                )
                self.set_semantics_recursive(pillar_prim, self.config["pillar"]["class"])
        count = 1
        for i in np.arange(-self.WIDTH_/2 + 5, self.WIDTH_/2, 8):
            for j in np.arange(-self.LENGTH_ / 2 + 5, self.LENGTH_/2, 6):
                
                light_position = (i, j, 9)
                light_prim = prims.create_prim(
                    prim_path=f"/World/env/Light_prim_{count}",
                    position=light_position,
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path + self.config["light"]["url"],
                    semantic_label=self.config["light"]["class"],
                )
                self.set_semantics_recursive(light_prim, self.config["light"]["class"])
                count += 1
        
        
        wall_classes = ["wall_A", "wall_B", "cell"]
        for wall_class in wall_classes:
            self.RandomWallColor(wall_class)
        
        self.CreateNavmesh()
    
    ### 墙面颜色随机化    
    def RandomWallColor(self, class_:str, aim_color:List[float]=[]):
        
        if class_ == "wall_A":
            aim_mesh = "Section0"
        elif class_ == "wall_B":
            aim_mesh = "SM_WallB_6M"
        elif class_ == "floor":
            aim_mesh = "SM_floor02"
        elif class_ == "cell":
            aim_mesh = "SM_CeilingA_6X6"
        else:
            aim_mesh = "all"
            
        if aim_color == []:
            aim_color = Gf.Vec4f(random.random(), random.random(), random.random(), random.uniform(0.5, 0.9))
                        
        target_prims = self.find_aimPrim_paths(class_)
        
        stage = omni.usd.get_context().get_stage()
        for target_prim in target_prims:
            for prim in Usd.PrimRange(target_prim):
                path = prim.GetPath().pathString
                name = prim.GetName().lower()
                if prim.GetTypeName() == "Mesh" and (os.path.basename(path) == aim_mesh or aim_mesh == "all"):
                    mesh_path = prim.GetPath().pathString
                    binding_api = UsdShade.MaterialBindingAPI(prim)
                    bound_mat, _ = binding_api.ComputeBoundMaterial()
                    if bound_mat:
                        # 找到连接的 Shader
                        for child in bound_mat.GetPrim().GetChildren():
                            shader = UsdShade.Shader(child)
                            if not shader:
                                continue

                            # 检查有没有 ColorAlbedo 输入
                            color_input = shader.GetInput("inputs:ColorAlbedo") or shader.GetInput("ColorAlbedo")
                            if color_input:
                                color_input.Set(aim_color)
                                print(f"[OK] 修改 {shader.GetPath()} ColorAlbedo = {aim_color}")
                            else:
                                print(f"[INFO] {shader.GetPath()} 没有 ColorAlbedo 输入")
                    else:
                        print("未绑定材质")
                        
    ### 料笼颜色随机化
    def RandomCageColor(self, target_prim, aim_color:List[float]=[]):
        
        def find_mesh_prims(root_prim, depth_limit=10):
            """
            递归查找 root_prim 下第 depth_limit 层及以下的所有 Mesh prim。
            """
            result = []

            def traverse(prim, depth):
                print(prim.GetPath().pathString)
                if not prim.IsValid():
                    print(f"{prim.GetPath().pathString} is un valid")
                    return
                if prim.GetTypeName() == "Mesh" or depth >= depth_limit:
                    result.append(prim)
                for child in prim.GetChildren():
                    traverse(child, depth + 1)

            traverse(root_prim, 0)
            return result
            
        if aim_color == []:
            aim_color = Gf.Vec4f(random.random(), random.random(), random.random(), random.uniform(0.5, 0.9))
                        
        # target_prims = self.find_aimPrim_paths("cage")
        
        stage = omni.usd.get_context().get_stage()
        base_prim_path = target_prim.GetPath().pathString
        # scope_path = os.path.join(base_prim_path, "Prototypes/Looks")
        scope_path = os.path.join(base_prim_path, "Prototypes")
        scope_prim = stage.GetPrimAtPath(scope_path)
        if not scope_prim.IsValid():
            print(f"❌ Scope {scope_path} 不存在")
        else:
            print(f"✅ 遍历 Scope: {scope_path}")
        # for mat_prim in Usd.PrimRange(scope_prim):
        #     if mat_prim.GetTypeName() == "Material":
        #         print("找到材料！！！")
        #     else:
        #         print(mat_prim.GetTypeName())

        #     # 获取绑定的 Shader（如果有）
        #     for shader_prim in Usd.PrimRange(mat_prim):
        #         shader = UsdShade.Shader(shader_prim)
        #         if not shader:
        #             print(" no shader.", " +++++++++++++++ ")
        #             continue

        #         # 检查有没有 ColorAlbedo 输入
        #         color_input = shader.GetInput("inputs:diffuseColor") or shader.GetInput("diffuseColor")
        #         if color_input:
        #             print("color_input: ", color_input)
        #             color_input.Set(aim_color)
        #             print(f"[OK] 修改 {shader.GetPath()} diffuseColor = {aim_color}")
        #         else:
        #             print(f"[INFO] {shader.GetPath()} 没有 diffuseColor 输入")
                
        meshprim_lists = find_mesh_prims(scope_prim)
        print(meshprim_lists)
        for prim in meshprim_lists:
            path = prim.GetPath().pathString
            name = prim.GetName().lower()
            
            if prim.GetTypeName() == "Mesh":
                mesh_path = prim.GetPath().pathString
                binding_api = UsdShade.MaterialBindingAPI(prim)
                bound_mat, _ = binding_api.ComputeBoundMaterial()
                if bound_mat:
                    # 找到连接的 Shader
                    for child in bound_mat.GetPrim().GetChildren():
                        print(child.GetPath().pathString, " ----------- ")
                        shader = UsdShade.Shader(child)
                        if not shader:
                            print(" no shader.", " +++++++++++++++ ")
                            continue

                        # 检查有没有 ColorAlbedo 输入
                        color_input = shader.GetInput("inputs:diffuseColor") or shader.GetInput("diffuseColor")
                        if color_input:
                            print("color_input: ", color_input)
                            color_input.Set(aim_color)
                            print(f"[OK] 修改 {shader.GetPath()} diffuseColor = {aim_color}")
                        else:
                            print(f"[INFO] {shader.GetPath()} 没有 diffuseColor 输入")
                else:
                    print("未绑定材质")
                    
    ### 初始化占用地图             
    def InitMap(self):
        self.map = MaskManager(self.WIDTH_, self.LENGTH_)
        self.setRandomShelfRigion()
        self.roadwidth = 3
        self.shelf_half_extent = self.measure_shelf_extent()
        print(f"self.shelf_half_extent: {self.shelf_half_extent}")
        self.path_points = None
        
        self.box_name = [
            "SM_FuseBox_01.usd",
            "SM_FuseBox_04.usd",
            "SM_CardBoxA_01.usd",
            "SM_CardBoxA_02.usd",
            "SM_CardBoxB_01.usd",
            "SM_CardBoxB_02.usd",
            "SM_CardBoxC_01.usd",
            "SM_CardBoxC_02.usd",
            "SM_CardBoxD_01.usd",
            "SM_CardBoxD_02.usd",
            "SM_CardBoxD_03.usd",
            "SM_CardBoxD_04.usd",
            "SM_CardBoxD_05.usd",
        ]
        self.basePath = self.assets_root_path +"/Isaac/Environments/Simple_Warehouse/Props"
        self.boxFiles = [os.path.join(self.basePath, file) for file in self.box_name]

        self.updatePeopleUrl()
        
    
    def extract_xy_rotation(self, R):
        R = np.array(R)
        theta_z = np.arctan2(R[1, 0], R[0, 0])  # 索引从0开始
        return theta_z  # 返回弧度值
    
    def setRandomShelfRigion(self):
        min_x = random.uniform(0, self.WIDTH_* 2/8)
        min_y = random.uniform(0, self.LENGTH_* 2/8)
        max_x = random.uniform(self.WIDTH_*5/8, self.WIDTH_*6/8)
        max_y = random.uniform(self.LENGTH_*5/8, self.LENGTH_*6/8)
        
        self.shelfrigion = [(min_x, min_y), (max_x, max_y)]
        print(f"self.shelfrigion: {self.shelfrigion}")
    
    def getShelfslocation(self):
        orintationFLag = random.randint(0,1)
        # orintationFLag = 0
        
        if orintationFLag == 1:
            orintation = "vertical"
        else:
            orintation = "horizontal"

        shelves = self.map.generate_shelves_in_region(
        path_points=self.path_points,
        path_half_width=self.roadwidth/2,
        region_min=self.shelfrigion[0],
        region_max=self.shelfrigion[1],
        shelf_length=self.shelf_half_extent[0] * 2,
        shelf_depth=self.shelf_half_extent[1] * 2,
        safety_margin=0.2,
        orientation=orintation)
        
        return shelves
    
    def GetReferObjInformation(self, objName:str, url:str=None, seglabel:str=None):
        if objName in self.ReferObjPaths:
            pass 
        else:
            if url != None:
                objurl = url
                if seglabel == None:
                    seglabel = "unlabel"
            else:
                if seglabel == None:
                    if self.config[objName]["class"]:
                        seglabel = self.config[objName]["class"]
                    else:
                        seglabel = "unlabel"
                objurl =self.assets_root_path + self.config[objName]["url"]
            box_half_size = self.measure_half_extent(objurl)
            if seglabel == "pallet":
                self.pallet_half_size = box_half_size
                print("pallet half size: " , self.pallet_half_size)
            count = len(self.ReferObjPaths)
            tmp_path = Sdf.Path(f"/_class_/ObjReferTemplate_{count}")
            self.ReferObjPaths[objName] = {
                "path": tmp_path,
                "size": box_half_size,
                "label":seglabel
            }
            if not self.stage.GetPrimAtPath(tmp_path).IsValid():
                proto_xform = UsdGeom.Xform.Define(self.stage, tmp_path)
                proto_xform.GetPrim().GetReferences().AddReference(objurl)  # 引用本地 usd 文件
                if seglabel == "cage":
                    self.RandomCageColor(proto_xform.GetPrim())
                self.set_semantics_recursive(proto_xform.GetPrim(), seglabel)
                proto_xform.GetPrim().SetInstanceable(True)
                
                UsdGeom.XformCommonAPI(proto_xform).SetTranslate((10000, 10000, 10000))
        
        return self.ReferObjPaths[objName]
    
    def CreateShelf(self, shelves):
        self.GetReferObjInformation("sideshelf")
        self.GetReferObjInformation("rackshield")
        self.GetReferObjInformation("railshelf")
        self.GetReferObjInformation("signcver")
        self.GetReferObjInformation("pallet")
        
        
        refer_side_url_path   = self.ReferObjPaths["sideshelf"]["path"]
        refer_shield_url_path = self.ReferObjPaths["rackshield"]["path"]
        refer_shelf_url_path  = self.ReferObjPaths["railshelf"]["path"]
        refer_shelf_sign_url_path  = self.ReferObjPaths["signcver"]["path"]
        refer_pallet_url_path  = self.ReferObjPaths["pallet"]["path"]
        
        
        floor_decal_redrec_url_path = self.assets_root_path + self.config["floordecal_recred"]["url"]
        floor_decal_line_url_path = self.assets_root_path + self.config["floordecal_stripfull"]["url"]
        for id, shelf_item in enumerate(shelves):
            shelf_center = shelf_item["center"]
            shelf_lenth = shelf_item["length"]
            shelf_orin = shelf_item["orientation"]
            print(" == " * 100)
            print(shelf_orin)
            print(" == " * 100)
            shelf_xform_path = f"/World/env/shelf/SHELF_{id}"
            if shelf_orin == "vertical":
                shelf_min_length_location = shelf_center[1] - shelf_lenth / 2
                nums = int(shelf_lenth / self.shelf_half_extent[0] / 2)
                
                shelf_sheild_left_prim_path = f"{shelf_xform_path}/shelf_sheild_0"
                shelf_sheild_left_position = (shelf_center[0] - self.WIDTH_/ 2, shelf_min_length_location - self.LENGTH_/ 2, 0)
                shelf_sheild_left_oritation = [0, 0, np.pi/2]
                shelf_sheild_prim = self.create_objs_instance(shelf_sheild_left_prim_path, shelf_sheild_left_position, shelf_sheild_left_oritation, refer_shield_url_path, self.ReferObjPaths["rackshield"]["label"])

                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/env/shelf/shelf_floordecal_redrec_{id}_0",
                    position=(shelf_center[0] - self.WIDTH_/ 2, shelf_min_length_location - self.LENGTH_/ 2, 0.001),
                    orientation=euler_angles_to_quat([0, 0, np.pi/2]),
                    scale=(1.1, 1.1, 1),
                    usd_path=floor_decal_redrec_url_path,
                    semantic_label=self.config["floordecal_recred"]["class"],
                )
                self.set_semantics_recursive(floordecal_recred_prims, self.config["floordecal_recred"]["class"])

                for num in range(nums):
                    location_x = shelf_center[0]
                    location_y = shelf_min_length_location + num * (self.shelf_half_extent[0] * 2)
                    
                    for t in range(2):
                        shelf_side_prims_path = f"{shelf_xform_path}/shelf_side_{num+1}_{t}"
                        shelf_side_position = (location_x- self.WIDTH_/ 2, location_y- self.LENGTH_/ 2, t * 2 * self.shelf_half_extent[2])
                        shelf_side_oritation = [0, 0, np.pi/2]
                        
                        shelf_side_prim = self.create_objs_instance(shelf_side_prims_path, shelf_side_position, shelf_side_oritation, refer_side_url_path, self.ReferObjPaths["sideshelf"]["label"])

                    
                    for t in range(3):
                        shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{num+1}_{t}_0"
                        shelf_side_sign_position = (location_x- self.WIDTH_/ 2 - (self.shelf_half_extent[1] - 0.005), location_y- self.LENGTH_/ 2, (t+1) * 7 / 4 )
                        shelf_side_sign_oritation = [0, 0, np.pi/2]
                        shelf_side_sign_prim = self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])

                        
                        shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{num+1}_{t}_1"
                        shelf_side_sign_position = (location_x- self.WIDTH_/ 2 + (self.shelf_half_extent[1] - 0.005), location_y- self.LENGTH_/ 2, (t+1) * 7 / 4 )
                        shelf_side_sign_oritation = [0, 0, -np.pi/2]
                        shelf_side_sign_prim = self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])

                        
                    floordecal_line_prims = prims.create_prim(
                        prim_path=f"/World/env/shelf/shelf_floordecal_line_{id}_{num+1}_0",
                        position=(location_x - self.shelf_half_extent[1] - self.WIDTH_/ 2, location_y - self.LENGTH_/ 2, 0.001),
                        orientation=euler_angles_to_quat([0, 0, np.pi/2]),
                        usd_path=floor_decal_line_url_path,
                        semantic_label=self.config["floordecal_stripfull"]["class"],
                        ) 
                    self.set_semantics_recursive(floordecal_line_prims, self.config["floordecal_stripfull"]["class"])
                    
                    floordecal_line_prims = prims.create_prim(
                        prim_path=f"/World/env/shelf/shelf_floordecal_line_{id}_{num+1}_1",
                        position=(location_x + self.shelf_half_extent[1] - self.WIDTH_/ 2, location_y - self.LENGTH_/ 2, 0.001),
                        orientation=euler_angles_to_quat([0, 0, np.pi/2]),
                        usd_path=floor_decal_line_url_path,
                        semantic_label=self.config["floordecal_stripfull"]["class"],
                        )
                    self.set_semantics_recursive(floordecal_line_prims, self.config["floordecal_stripfull"]["class"])
                    
                    shelf_nums = random.randint(2,3)
                    shelf_gaps = [round(random.uniform(1.5, 2), 1) for _ in range(shelf_nums)]
                    total_gap = 0
                    for j, gap in enumerate(shelf_gaps):
                        total_gap += gap
                        shelf_prim_path = f"{shelf_xform_path}/shelf_{num+1}_{j}"
                        position=(location_x - self.WIDTH_/ 2, location_y + self.shelf_half_extent[0] - self.LENGTH_/ 2, total_gap)
                        shelf_prim_oritation = [0, 0, np.pi/2]
                        shelf_prim = self.create_objs_instance(shelf_prim_path, position, shelf_prim_oritation, refer_shelf_url_path, self.ReferObjPaths["railshelf"]["label"])

                        position_tmp=(position[0], position[1], position[2]-gap)
                        
                        # ##### 增加托盘
                        add_pallet_flag = random.random() > 0.5
                        if  add_pallet_flag:
                            num_pallet = int(self.shelf_half_extent[0] // (self.pallet_half_size[0]))
                            for i in range(num_pallet):
                                pallet_prim_path = f"{shelf_xform_path}/pallet_{num+1}_{j}_{i}"
                                position_pallet = (position[0], location_y + self.pallet_half_size[0] * (2*i + 1) - self.LENGTH_/ 2 + 0.1, position[2]-gap)
                                pallet_oritation = [0, 0, np.pi/2]
                                shelf_pallet_prim = self.create_objs_instance(pallet_prim_path, position_pallet, pallet_oritation, refer_pallet_url_path, self.ReferObjPaths["pallet"]["label"])

                            position_goods = (position_tmp[0], position_tmp[1], position_tmp[2] + self.pallet_half_size[2] * 2)
                            gap = gap - self.pallet_half_size[2] * 2
                        else:
                            position_goods=position_tmp
                        shelf_goods_base_xfrom = f"/World/env/shelf/ShelfGoods_{id}_{num+1}_{j}"
                        self.CreateShelfGoods(shelflocation=position_goods, shelfGap=gap, base_xform=shelf_goods_base_xfrom, oritation="vertical")
                        
                        
                        if j == (len(shelf_gaps) -1):
                            self.CreateShelfGoods(shelflocation=position, shelfGap=1.0, base_xform=f"/World/env/shelf/ShelfGoods_{id}_{num+1}_{j+1}", oritation="vertical")

                
                for t in range(2):
                    tmp_prims_path = f"{shelf_xform_path}/shelf_side_{nums+1}_{t}"
                    tmp_position = (location_x- self.WIDTH_/ 2, shelf_min_length_location + shelf_lenth - self.LENGTH_/ 2, t * 2 * self.shelf_half_extent[2])
                    tmp_oritation = [0, 0, np.pi/2]
                    shelf_side_prim = self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_side_url_path, self.ReferObjPaths["sideshelf"]["label"])
                
                
                for t in range(3):
                    shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{nums+1}_{t}_0"
                    shelf_side_sign_position = (location_x- self.WIDTH_/ 2 - (self.shelf_half_extent[1] - 0.005), location_y- self.LENGTH_/ 2, (t+1) * 7 / 4 )
                    shelf_side_sign_oritation = [0, 0, np.pi/2]
                    self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])
                    
                    shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{nums+1}_{t}_1"
                    shelf_side_sign_position = (location_x- self.WIDTH_/ 2 + (self.shelf_half_extent[1] - 0.005), location_y- self.LENGTH_/ 2, (t+1) * 7 / 4 )
                    shelf_side_sign_oritation = [0, 0, -np.pi/2]
                    self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])
                
                
                tmp_prims_path = f"{shelf_xform_path}/shelf_sheild_1"
                tmp_position = (shelf_center[0] - self.WIDTH_/ 2, shelf_min_length_location + shelf_lenth - self.LENGTH_/ 2, 0)
                tmp_oritation = [0, 0, -np.pi/2]
                self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_shield_url_path, self.ReferObjPaths["rackshield"]["label"])
                
                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/env/shelf/shelf_floordecal_redrec_{id}_1",
                    position=(shelf_center[0] - self.WIDTH_/ 2, shelf_min_length_location + shelf_lenth - self.LENGTH_/ 2, 0.001),
                    orientation=euler_angles_to_quat([0, 0, -np.pi/2]),
                    scale=(1.1, 1.1, 1),
                    usd_path=floor_decal_redrec_url_path,
                    semantic_label=self.config["floordecal_recred"]["class"],
                ) 
                self.set_semantics_recursive(floordecal_recred_prims, self.config["floordecal_recred"]["class"])        
            else:
                shelf_min_length_location = shelf_center[0] - shelf_lenth / 2
                nums = int(shelf_lenth / self.shelf_half_extent[0] / 2)
                
                tmp_prims_path = f"{shelf_xform_path}/shelf_sheild_0"
                tmp_position = (shelf_min_length_location - self.WIDTH_/ 2, shelf_center[1] - self.LENGTH_/ 2, 0)
                tmp_oritation = [0, 0, 0]
                self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_shield_url_path, self.ReferObjPaths["rackshield"]["label"])
                
                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/env/shelf/shelf_floordecal_redrec_{id}_0",
                    position=(shelf_min_length_location - self.WIDTH_/ 2, shelf_center[1] - self.LENGTH_/ 2, 0.001),
                    scale=(1.1, 1.1, 1),
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=floor_decal_redrec_url_path,
                    semantic_label=self.config["floordecal_recred"]["class"],
                )
                self.set_semantics_recursive(floordecal_recred_prims, self.config["floordecal_recred"]["class"])   
                
                for num in range(nums):
                    location_y = shelf_center[1]
                    location_x = shelf_min_length_location + num * (self.shelf_half_extent[0] * 2)
                    
                    for t in range(2):
                        tmp_prims_path =f"{shelf_xform_path}/shelf_side_{num+1}_{t}"
                        tmp_position =(location_x- self.WIDTH_/ 2, location_y- self.LENGTH_/ 2, t * 2 * self.shelf_half_extent[2])
                        tmp_oritation = [0, 0, 0]
                        self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_side_url_path, self.ReferObjPaths["sideshelf"]["label"])
                        
                    for t in range(3):
                        shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{num+1}_{t}_0"
                        shelf_side_sign_position = (location_x- self.WIDTH_/ 2, location_y- self.LENGTH_/ 2 + (self.shelf_half_extent[1] - 0.005), (t+1) * 7 / 4 )
                        shelf_side_sign_oritation = [0, 0, np.pi]
                        self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])
                        
                        shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{num+1}_{t}_1"
                        shelf_side_sign_position = (location_x- self.WIDTH_/ 2, location_y- self.LENGTH_/ 2 - (self.shelf_half_extent[1] - 0.005), (t+1) * 7 / 4 )
                        shelf_side_sign_oritation = [0, 0, 0]
                        self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])
                    
                    floordecal_line_prims1 = prims.create_prim(
                        prim_path=f"/World/env/shelf/shelf_floordecal_line_{id}_{num+1}_{0}",
                        position=(location_x- self.WIDTH_/ 2, location_y - self.shelf_half_extent[1] - self.LENGTH_/ 2, 0.001),
                        orientation=euler_angles_to_quat([0, 0, 0]),
                        usd_path=floor_decal_line_url_path,
                        semantic_label=self.config["floordecal_stripfull"]["class"],
                        )
                    
                    self.set_semantics_recursive(floordecal_line_prims1, self.config["floordecal_stripfull"]["class"])
                    
                    floordecal_line_prims2 = prims.create_prim(
                        prim_path=f"/World/env/shelf/shelf_floordecal_line_{id}_{num+1}_{1}",
                        position=(location_x- self.WIDTH_/ 2, location_y + self.shelf_half_extent[1] - self.LENGTH_/ 2, 0.001),
                        orientation=euler_angles_to_quat([0, 0, 0]),
                        usd_path=floor_decal_line_url_path,
                        semantic_label=self.config["floordecal_stripfull"]["class"],
                        ) 
                    self.set_semantics_recursive(floordecal_line_prims2, self.config["floordecal_stripfull"]["class"]) 
                    
                    shelf_nums = random.randint(2,3)
                    shelf_gaps = [round(random.uniform(1,2), 1) for _ in range(shelf_nums)]
                    total_gap = 0
                    for j, gap in enumerate(shelf_gaps):
                        total_gap += gap
                        position=(location_x + self.shelf_half_extent[0] - self.WIDTH_/ 2, location_y- self.LENGTH_/ 2, total_gap)
                        tmp_prims_path =f"{shelf_xform_path}/shelf_{num+1}_{j}"
                        tmp_position =position
                        tmp_oritation = [0, 0, 0]
                        self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_shelf_url_path, self.ReferObjPaths["railshelf"]["label"])
                        
                        position_tmp=(position[0], position[1], position[2]-gap)
                        
                        # ##### 增加托盘
                        add_pallet_flag = random.random() > 0.5
                        if  add_pallet_flag:
                            num_pallet = int(self.shelf_half_extent[0] // (self.pallet_half_size[0]))
                            for i in range(num_pallet):
                                pallet_prim_path = f"{shelf_xform_path}/pallet_{num+1}_{j}_{i}"
                                position_pallet = (location_x + self.pallet_half_size[0] * (2 * i + 1) - self.WIDTH_/ 2 + 0.1, location_y- self.LENGTH_/ 2, position[2]-gap)
                                pallet_oritation = [0, 0, 0]
                                shelf_pallet_prim = self.create_objs_instance(pallet_prim_path, position_pallet, pallet_oritation, refer_pallet_url_path, self.ReferObjPaths["pallet"]["label"])
                            position_goods = (position_tmp[0], position_tmp[1], position_tmp[2] + self.pallet_half_size[2] * 2)
                        else:
                            position_goods=position_tmp
                        shelf_goods_base_xfrom = f"/World/env/shelf/ShelfGoods_{id}_{num+1}_{j}"
                        self.CreateShelfGoods(shelflocation=position_goods, shelfGap=gap, base_xform=shelf_goods_base_xfrom, oritation="horizontal")
                        
                        if j == (len(shelf_gaps) -1):
                            self.CreateShelfGoods(shelflocation=position, shelfGap=1.0, base_xform=f"/World/env/shelf/ShelfGoods_{id}_{num+1}_{j+1}", oritation="horizontal")
                            
                
                for t in range(2):
                    tmp_prims_path =f"{shelf_xform_path}/shelf_side_{nums+1}_{t}"
                    tmp_position =(shelf_min_length_location + shelf_lenth - self.WIDTH_/ 2, shelf_center[1] - self.LENGTH_/ 2, t * 2 * self.shelf_half_extent[2])
                    tmp_oritation = [0, 0, 0]
                    self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_side_url_path, self.ReferObjPaths["sideshelf"]["label"])
                
                for t in range(3):
                    shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{nums+1}_{t}_0"
                    shelf_side_sign_position = (location_x- self.WIDTH_/ 2, location_y- self.LENGTH_/ 2 + (self.shelf_half_extent[1] - 0.005), (t+1) * 7 / 4 )
                    shelf_side_sign_oritation = [0, 0, np.pi]
                    self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])
                    
                    shelf_side_sign_prims_path = f"/World/env/shelf/shelf_side_sign_{id}_{nums+1}_{t}_1"
                    shelf_side_sign_position = (location_x- self.WIDTH_/ 2, location_y- self.LENGTH_/ 2 - (self.shelf_half_extent[1] - 0.005), (t+1) * 7 / 4 )
                    shelf_side_sign_oritation = [0, 0, 0]
                    self.create_objs_instance(shelf_side_sign_prims_path, shelf_side_sign_position, shelf_side_sign_oritation, refer_shelf_sign_url_path, self.ReferObjPaths["signcver"]["label"])
                

                tmp_prims_path =f"{shelf_xform_path}/shelf_sheild_1"
                tmp_position =(shelf_min_length_location + shelf_lenth - self.WIDTH_/ 2, shelf_center[1] - self.LENGTH_/ 2, 0)
                tmp_oritation = [0, 0, np.pi]
                self.create_objs_instance(tmp_prims_path, tmp_position, tmp_oritation, refer_shield_url_path, self.ReferObjPaths["rackshield"]["label"])
                
                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/env/shelf/shelf_floordecal_redrec_{id}_1",
                    position=(shelf_min_length_location + shelf_lenth - self.WIDTH_/ 2, shelf_center[1] - self.LENGTH_/ 2, 0.001),
                    orientation=euler_angles_to_quat([0, 0, np.pi]),
                    scale=(1.1, 1.1, 1),
                    usd_path=floor_decal_redrec_url_path,
                    semantic_label=self.config["floordecal_recred"]["class"],
                )
                self.set_semantics_recursive(floordecal_recred_prims, self.config["floordecal_recred"]["class"])
            stage = omni.usd.get_context().get_stage()
            shelf_prim = stage.GetPrimAtPath(shelf_xform_path)
            self.add_colliders(shelf_prim)


    def CreateShelfGoods(self, shelflocation:List[float], shelfGap:float, base_xform:str, oritation:str="horizontal"):
        boxurl = random.choice(self.boxFiles)
        if boxurl in self.GoodstmplePaths:
            tmp_path = self.GoodstmplePaths[boxurl]["path"]
            box_half_size = self.GoodstmplePaths[boxurl]["size"]
        else:
            box_half_size = self.measure_half_extent(boxurl)
            count = len(self.GoodstmplePaths)
            tmp_path = Sdf.Path(f"/_class_/GoodsTemplate_{count}")
            # tmp_path = Sdf.Path(f"/World/Prototypes/GoodsTemplate_{count}")
            self.GoodstmplePaths[boxurl] = {
                "path": tmp_path,
                "size": box_half_size
            }
            if not self.stage.GetPrimAtPath(tmp_path).IsValid():
                proto_xform = UsdGeom.Xform.Define(self.stage, tmp_path)
                proto_xform.GetPrim().GetReferences().AddReference(boxurl)  # 引用本地 usd 文件
                self.set_semantics_recursive(proto_xform.GetPrim(), "Goods")
                proto_xform.GetPrim().SetInstanceable(True)
                
                UsdGeom.XformCommonAPI(proto_xform).SetTranslate((10000, 10000, 10000))
        
        
        if oritation == "horizontal":
            gap_lenth = self.shelf_half_extent[0] * 2 - 0.1 * 2
            gap_width = self.shelf_half_extent[1] * 2
            shelf_min_x = shelflocation[0] - self.shelf_half_extent[0] + 0.1
            shelf_min_y = shelflocation[1] - self.shelf_half_extent[1]
        else:
            gap_lenth = self.shelf_half_extent[1] * 2
            gap_width = self.shelf_half_extent[0] * 2 - 0.1 * 2
            shelf_min_x = shelflocation[0] - self.shelf_half_extent[1]
            shelf_min_y = shelflocation[1] - self.shelf_half_extent[0] + 0.1
        gap_height = shelfGap
        
        shelf_min_z = shelflocation[2]
        
        goods_orination = np.pi/4 if random.randint == 1 else 0
        
        if goods_orination == 0:
            box_len = box_half_size[0] * 2
            box_wid = box_half_size[1] * 2
        else:
            box_len = box_half_size[1] * 2
            box_wid = box_half_size[0] * 2
                
        good_len_nums = math.floor(gap_lenth / box_len)
        good_len_gap = (gap_lenth - box_len * good_len_nums) / 2
        
        good_wid_nums = math.floor(gap_width / box_wid)
        good_wid_gap = (gap_width - box_wid * good_wid_nums) / 2
        
        good_hei_nums = math.floor((gap_height - 0.05) / (box_half_size[2] * 2))
        if good_hei_nums > 1:
            good_actural_hei_nums = random.randint(1, good_hei_nums)
        else:
            good_actural_hei_nums = good_hei_nums
        
        for h in range(good_actural_hei_nums):
            for w in range(good_wid_nums):
                for l in range(good_len_nums):
                    if h == (good_actural_hei_nums - 1):
                        if (w == 0 or w == (good_wid_nums-1) or l == 0 or l == (good_len_nums-1)):
                            continueFlag = True if random.choices([0, 1], weights=[4, 1])[0] == 1 else False
                            if continueFlag:
                                continue
                    locate_x = shelf_min_x + good_len_gap + (2*l + 1) * box_len / 2
                    locate_y = shelf_min_y + good_wid_gap + (2*w + 1) * box_wid / 2
                    locate_z = shelf_min_z + 2 * h * box_half_size[2]
                    
                    self.create_goods_instance(base_xform=base_xform, 
                                               position=(locate_x, locate_y, locate_z), 
                                               goods_orientation=goods_orination, 
                                               template_path=tmp_path,
                                               PrimEnds=f"_{l}_{w}_{h}")
    
    def create_goods_instance(self, base_xform:str, position:List[float], goods_orientation:float, template_path:str, PrimEnds:str=None):
        prim_path = f"{base_xform}/Goods{PrimEnds}"
        xform = UsdGeom.Xform.Define(self.stage, prim_path)

        # 设置位姿
        xform.AddTranslateOp().Set(position)
        quat = euler_angles_to_quat([0, 0, goods_orientation])
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

        # 引用模板
        xform.GetPrim().GetReferences().AddInternalReference(template_path)

        # 设置语义标签
        xform.GetPrim().CreateAttribute("semantics:semanticLabel", Sdf.ValueTypeNames.String).Set("Goods")
        return xform
    
    def create_objs_instance(self, prim_path:str, position:List[float], oritiation:List[float], template_path:str, className:str):

        xform = UsdGeom.Xform.Define(self.stage, prim_path)

        # 设置位姿
        xform.AddTranslateOp().Set(position)
        quat = euler_angles_to_quat(oritiation)
        xform.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

        # 引用模板
        xform.GetPrim().GetReferences().AddInternalReference(template_path)
        # 设置语义标签
        xform.GetPrim().CreateAttribute("semantics:semanticLabel", Sdf.ValueTypeNames.String).Set(className)
                
        return xform.GetPrim()
        
    def SetTrace(self, start_point:Tuple[float, float], end_point:Tuple[float, float], 
                 priority:float=1.0, 
                 max_trans_points:int=15):
        
        shelf_rigion_min_x, shelf_rigion_min_y = self.shelfrigion[0]
        shelf_rigion_max_x, shelf_rigion_max_y = self.shelfrigion[1]
        shelf_center = ((shelf_rigion_max_x + shelf_rigion_min_x)/2, (shelf_rigion_max_y + shelf_rigion_min_y)/2)
        shelf_rigion_half_width =  (shelf_rigion_max_x - shelf_rigion_min_x)/2
        shelf_rigion_half_height =  (shelf_rigion_max_y - shelf_rigion_min_y)/2
        
        road_rot_flag = random.randint(0,1)
        if road_rot_flag == 1:
            orintation = "vertical"
        else:
            orintation = "horizontal"
            
        # 货架区域示例定义
        shelf_zone_example = {"min": self.shelfrigion[0], "max":self.shelfrigion[1]}

        # 使用示例
        path_points = self.map.generate_constrained_path_v2(
            start_point = start_point,
            end_point = end_point,
            shelf_zone=shelf_zone_example,  # 传入货架区域
            path_half_width=self.roadwidth/2,
            shelf_half_length=self.shelf_half_extent[0],
            shelf_half_depth=self.shelf_half_extent[1],
            first_direction=orintation,
            priority=priority,
            gap=0.2,
            max_points=max_trans_points
        )
        
        self.path_points = path_points
        
        return path_points


    def generate_camera_trajectory_3d(self, path_3d: np.ndarray, speed: float, fps: int, angle_offset_deg: float, look_ahead_dist: float = 1.5) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """
        生成相机沿 3D 路径运动的位置和 look_at 向量序列。

        参数:
        - path_3d: np.ndarray, shape=(N, 3), 路径点 (x, y, z)
        - speed: float, 相机速度 (单位 m/s)
        - fps: int, 帧率
        - angle_offset_deg: float, 相机与轨迹方向夹角（度）
        - look_ahead_dist: float, 相机 look_at 指向距离（单位 m）

        返回:
        - positions: List[(x, y, z)] 每帧相机位置
        - look_ats: List[(x, y, z)] 每帧相机 look_at 目标位置
        """
        angle_offset_rad = math.radians(angle_offset_deg)
        positions = []
        look_ats_left = []
        look_ats_right = []

        # 计算每段方向向量和长度
        segment_vectors = path_3d[1:] - path_3d[:-1]
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        segment_directions_xy = np.arctan2(segment_vectors[:,1], segment_vectors[:,0])

        # 每帧移动距离
        delta_s = speed / fps

        # 初始化
        current_seg = 0
        seg_pos = 0.0

        while current_seg < len(segment_lengths):
            start = path_3d[current_seg]
            vec = segment_vectors[current_seg]
            seg_len = segment_lengths[current_seg]
            dir_angle = segment_directions_xy[current_seg]

            # 当前帧位置
            t = seg_pos / (seg_len + 1e-6)
            pos = start + t * vec
            positions.append(tuple(pos))

            # 当前帧 look_at 目标（沿轨迹方向 + 偏角）
            look_at_x = pos[0] + math.cos(dir_angle + angle_offset_rad) * look_ahead_dist
            look_at_y = pos[1] + math.sin(dir_angle + angle_offset_rad) * look_ahead_dist
            look_at_z = pos[2]  # z方向保持与相机相同高度
            look_ats_left.append((look_at_x, look_at_y, look_at_z))
            
            look_at_right_x = pos[0] + math.cos(dir_angle - angle_offset_rad) * look_ahead_dist
            look_at_right_y = pos[1] + math.sin(dir_angle - angle_offset_rad) * look_ahead_dist
            look_at_right_z = pos[2]  # z方向保持与相机相同高度
            look_ats_right.append((look_at_right_x, look_at_right_y, look_at_right_z))

            # 前进
            seg_pos += delta_s
            if seg_pos >= seg_len:
                seg_pos -= seg_len
                current_seg += 1

        return positions, look_ats_left, look_ats_right        


    def CreateShelfCamera(self, Road:List[float], angle_offset_deg:float=45):
        
        ## 控制帧率
        speed = 5   #m/s
        fps = 30
        
        tmp_road = np.array(Road) - np.array([self.WIDTH_/2, self.LENGTH_/2])
        pos_z = 1.5
        z_array = np.full((tmp_road.shape[0], 1),pos_z)
        road_3d = np.hstack((tmp_road, z_array))
        cam_pos, cam_left_look, cam_right_look = self.generate_camera_trajectory_3d(road_3d, speed, fps, angle_offset_deg, look_ahead_dist=pos_z)
        self.move_cam_pos = [(float(x), float(y), float(z)) for (x, y, z) in cam_pos]
        self.move_cam_top_pos = [(float(x), float(y), 8.0) for (x, y, z) in cam_pos]
        self.move_cam_left_look = [(float(x), float(y), float(z)) for (x, y, z) in cam_left_look]
        self.move_cam_right_look = [(float(x), float(y), float(z)) for (x, y, z) in cam_right_look]
        position = cam_pos[0]
        
        look_left = cam_left_look[0]
        look_right = cam_right_look[0]
        self.move_top_camera = rep.create.camera(
            position=(position[0], position[1], 8), look_at=position,
            focus_distance=400.0, focal_length=4.0, clipping_range=(0.1, 10000000.0), name="move_top_camera"
        )
        self.move_left_camera = rep.create.camera(
            position=position, look_at=look_left,
            focus_distance=400.0, focal_length=8.0, clipping_range=(0.1, 10000000.0), name="move_left_camera"
        )
        self.move_right_camera = rep.create.camera(
            position=position, look_at=look_right,
            focus_distance=400.0, focal_length=8.0, clipping_range=(0.1, 10000000.0), name="move_right_camera"
        )

        resolution = self.config.get("resolution", (512, 512))
        unique_tag = str(int(time.time() * 1000))  # 或者 UUID
        move_top_rp  = rep.create.render_product(self.move_top_camera, resolution,  name=f"move_top_View_{unique_tag}")
        move_left_rp  = rep.create.render_product(self.move_left_camera, resolution,  name=f"move_left_View_{unique_tag}")
        move_right_rp = rep.create.render_product(self.move_right_camera, resolution, name=f"move_right_View_{unique_tag}")

        rps = [move_top_rp, move_left_rp, move_right_rp]
        for rp in rps:
            rp.hydra_texture.set_updates_enabled(False)
            self.move_rps.append(rp)
    
    ### 导航网格创建    
    def CreateNavmesh(self):
        # 获取当前场景
        stage = omni.usd.get_context().get_stage()
        print(f"stage.GetRootLayer(): {stage.GetRootLayer()}")
        omni.kit.commands.execute(
            "CreateNavMeshVolumeCommand",
            parent_prim_path=Sdf.Path("/"),
            volume_type = 0,
            layer=stage.GetRootLayer())
        
        navmesh_prim = stage.GetPrimAtPath("/NavMeshVolume")
        
        xform = UsdGeom.Xformable(navmesh_prim)

        # 查找已有的 transform 操作
        ops = xform.GetOrderedXformOps()

        # 如果已有 translate，就直接改值
        translate_ops = [op for op in ops if op.GetOpName() == "xformOp:translate"]
        if translate_ops:
            print("start set translate_ops")
            translate_ops[0].Set(Gf.Vec3f(0, 0, 1))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1))

        # 同理处理 scale
        scale_ops = [op for op in ops if op.GetOpName() == "xformOp:scale"]
        if scale_ops:
            print("start set scale_ops")
            scale_ops[0].Set(Gf.Vec3f(self.WIDTH_, self.LENGTH_, self.HEIGHT_))
        else:
            xform.AddScaleOp().Set(Gf.Vec3f(self.WIDTH_, self.LENGTH_, self.HEIGHT_))
        
      
    def RandomUpdateMap(self):
        if self.science == 1:
            
            if len(self.path) >= 2:
                self.path_points = self.path
            else:
                #轨迹
                road_start_x = round(random.uniform(self.shelfrigion[0][0], self.shelfrigion[0][0] + 1), 1)
                road_start_y = round(random.uniform(self.shelfrigion[0][1], self.shelfrigion[0][1] + 1), 1)
                road_end_x = round(random.uniform(self.shelfrigion[1][0] - 1, self.shelfrigion[1][0]), 1)
                road_end_y = round(random.uniform(self.shelfrigion[1][1] - 1, self.shelfrigion[1][1]), 1)
                start_point = (road_start_x, road_start_y)
                end_point = (road_end_x, road_end_y)
                self.SetTrace(start_point, end_point)
            
            #货架
            shelves = self.getShelfslocation()
            self.CreateShelf(shelves)
            self.CreateShelfCamera(self.path_points)
        
        #库位
        linear_range_count = random.randint(3, 10)
        linear_range_arrange = random.choice([(3,2), (4,2), (5,2), (6, 2), (1, 1)])
        # linear_range_arrange = random.choice([(1, 3)]) # 160场景
        linear_range_goods_nums = linear_range_arrange[0] * linear_range_arrange[1]
        # self.linear_range_goods_counts = [random.randint(4, linear_range_goods_nums * 2) for _ in range(linear_range_count)]
        self.linear_range_goods_counts = [random.randint(1, linear_range_goods_nums) for _ in range(linear_range_count)]
        self.linear_range_goods_prim_paths = [self.randomchoiseGoods_in_arange() for _ in range(linear_range_count)] 
        self.Update_linear_range_2_map(linear_range_count, linear_range_arrange)
        
        #人车
        self.human_count = random.randint(0, 10)
        self.random_PF("character")
        
        self.forklift_count = random.randint(1, 3)
        self.random_PF("forklift")
        
        self.map.visualize_all_in_one()
        
        #杂物
    
    ### 人物url随机
    def updatePeopleUrl(self):
        people_base_path = self.assets_root_path + "/Isaac/People/Characters"
        people_files = os.listdir(people_base_path)
        peoples = []
        for file in people_files:
            tmp_base = os.path.join(people_base_path, file)
            if os.path.isdir(tmp_base):
                names = os.listdir(tmp_base)
                if "textures" in names:
                    for name in names:
                        if ".usd" in name:
                            usdPath = os.path.join(tmp_base, name)
                            peoples.append(usdPath)
                            break
        
        self.peoples = peoples
        
        animation_base_path = self.assets_root_path + "/Isaac/People/Animations"
        animation_files = os.listdir(animation_base_path)
        self.animations = []
        for animation in animation_files:
            animation_path = os.path.join(animation_base_path, animation)
            self.animations.append(animation_path)

 
    def find_aimPrim_paths(self, target_keyword:str):
        """找出所有路径中包含目标关键词的 prim

        Args:
            target_keyword (str): 目标名称

        Returns:
            _type_: prims列表
        """
        
        stage = omni.usd.get_context().get_stage()
        # 
        target_prims = [prim for prim in stage.Traverse() if target_keyword in os.path.basename(prim.GetPath().pathString)]

        print(f"找到 {len(target_prims)} 个 prim，路径中包含 '{target_keyword}':")
        for p in target_prims:
            print(" -", p.GetPath().pathString)
        
        return target_prims

    def random_color_by_prim_path(self, root_path:str, label:str="Character"):
        
        # 打开 Stage
        stage = omni.usd.get_context().get_stage()

        # 要匹配的关键词（路径中包含这些字符串的子部件才会被修改）
        if label == "Character":
            target_keywords = ["hardhat", "policecap", "shirt", "workboots", "shoes", "vest", "jeans", "pants"]
            visibal_keywords = ["hardhat", "policecap","vest"]
        elif label == "forklift":
            pass
        else:
            pass

        # 随机颜色函数
        def random_color():
            return [random.random(), random.random(), random.random()]
        
        def try_set_material_color(prim, color):
            """
            尝试修改Prim的材质颜色，若Prim没有有效的材质或Surface节点则跳过。
            """
            from pxr import UsdShade, Gf

            bound_material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not bound_material:
                print(f"[WARN] No material bound to {prim.GetPath()}")
                return False

            surface = bound_material.GetSurfaceOutput()
            if not surface:
                print(f"[WARN] No surface output in material {bound_material.GetPath()}")
                return False

            connected = surface.GetConnectedSource()
            if not connected:
                print(f"[WARN] No connected shader on surface for {bound_material.GetPath()}")
                return False

            shader_prim = connected[0]
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                print(f"[WARN] Shader not found for {shader_prim.GetPath()}")
                return False

            # 常见属性名可能是“diffuseColor”、“base_color”、“baseColor”等
            for param_name in ["diffuseColor", "base_color", "baseColor", "albedo_color"]:
                if shader.GetInput(param_name):
                    shader.GetInput(param_name).Set(Gf.Vec3f(*color))
                    print(f"[OK] Set color of {prim.GetPath()} via {param_name}")
                    return True

            print(f"[WARN] No known color input on shader {shader_prim.GetPath()}")
            return False


        for prim in Usd.PrimRange(root_path):
            path = prim.GetPath().pathString
            name = prim.GetName().lower()

            # 二级匹配结构名
            if any(k in name for k in target_keywords):
                if prim.IsA(UsdGeom.Gprim):
                    geom = UsdGeom.Gprim(prim)
                    color = random_color()

                    # 优先修改材质，否则使用 displayColor
                    success = try_set_material_color(prim, color)
                    if not success:
                        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])

                    if any(k in name for k in visibal_keywords):
                        # 可见性控制
                        visibility = random.choices(["inherited", "invisible"], weights=[9, 1], k=1)[0]
                        geom.CreateVisibilityAttr(visibility)

                        print(f"�� 修改 {path} -> 颜色={color}, 可见性={visibility}")
    
    def Randomcolor(self, target_keyword:str, label:str="character"):
        target_base_prims = self.find_aimPrim_paths(target_keyword)
        for base_prim in target_base_prims:
            self.random_color_by_prim_path(base_prim, label)
    
    def random_Forklift_color(self):
        stage = omni.usd.get_context().get_stage()
        
        target_keyword = "forklift_"
        target_base_prims = self.find_aimPrim_paths(target_keyword)
        change_prim = target_base_prims[0]
        mesh_prim = stage.GetPrimAtPath(change_prim.GetPath().pathString + "/S_ForkliftBody")
        
        mesh = UsdGeom.Mesh(mesh_prim)
        color_attr = mesh.GetDisplayColorPrimvar()

        if color_attr:
            print("Has displayColor primvar!")
            colors = color_attr.Get()
            print(f"Color count: {len(colors)}")
        else:
            print("No displayColor primvar found.")
        
        
    def random_PF(self, label):
        if label == "Character":
            
            url_path = self.assets_root_path + self.config["Character"]["url"]
            peo_half_extent = self.measure_half_extent(url_path)
            people_items = self.map.add_multiple_masks(peo_half_extent[0], peo_half_extent[1], 1, self.human_count)
            for i,item in enumerate(people_items):
                
                # 随机选人物
                url = random.choice(self.peoples)
                
                position_xy, rot_z = item["center_physical"], (np.pi/ 180 * item["rotation"])
                person_path = f"/World/Characters/Character_{(i+1):02d}"
                People_prims = prims.create_prim(
                    prim_path=person_path,
                    position=(position_xy[0]- self.WIDTH_/2, position_xy[1]- self.LENGTH_/2, 0),
                    orientation=euler_angles_to_quat([0, 0, rot_z]),
                    usd_path=url,
                    semantic_label=self.config["Character"]["class"],
                )
                self.add_colliders(People_prims)
            
            # 随机化人物装饰和颜色
            self.Randomcolor(target_keyword="Character_")
            
                
        elif label == "forklift":
            url_path = self.assets_root_path + self.config["forklift"]["url"]
            fork_half_extent = self.measure_half_extent(url_path)
            # for i in range(self.forklift_count):
                
            #     position_xy, rot_z = self.map.place_single_mask(fork_half_extent[0], fork_half_extent[1], 1)
            forklift_items = self.map.add_multiple_masks(fork_half_extent[0], fork_half_extent[1], 1, self.forklift_count)
            for i,item in enumerate(forklift_items):
                position_xy, rot_z = item["center_physical"], (np.pi/ 180 * item["rotation"])
                Forklift_prims = prims.create_prim(
                    prim_path=f"/World/mobile/forklift/forklift_{i+1}",
                    position=(position_xy[0]- self.WIDTH_/2, position_xy[1]- self.LENGTH_/2, 0),
                    orientation=euler_angles_to_quat([0, 0, rot_z]),
                    usd_path=url_path,
                    semantic_label=self.config["forklift"]["class"])
                self.set_semantics_recursive(Forklift_prims, self.config["forklift"]["class"])
                self.add_colliders(Forklift_prims)
                # self.add_rigid_body_dynamics(Forklift_prims)
                # rep.get.prim_at_path()
                # rep.randomizer.texture(textures=[
                #     '/home/visionnav/application/isaacSim/isaac-sim-assets-environments-5.0.0/Assets/Isaac/5.0/Isaac/Props/Forklift/Materials/Textures/T_Forklift_D.png',
                #     '/home/visionnav/application/isaacSim/isaac-sim-assets-environments-5.0.0/Assets/Isaac/5.0/Isaac/Props/Forklift/Materials/Textures/T_Forklift_N.png',
                #     '/home/visionnav/application/isaacSim/isaac-sim-assets-environments-5.0.0/Assets/Isaac/5.0/Isaac/Props/Forklift/Materials/Textures/T_Forklift_ORM.png',
                #     ], input_prims=Forklift_prims)
        
    # def update_linear_range_2_map(self, count:int, ranks:List, half_extents:List[List]):
    def Update_linear_range_2_map(self, count:int, ranks:List):
        floor_decal_line_url_path = self.assets_root_path + self.config["floordecal_stripfull"]["url"]
        for id in range(count):
            # goods_height = half_extents[id][2] * 2
            url_path = self.linear_range_goods_prim_paths[id]
            goodName = os.path.basename(url_path).split('.')[0]
            half_extents = self.ReferObjPaths[goodName]["size"]
            goods_height = half_extents[2] * 2
            regions = self.map.generate_batch_grid_regions(half_extents[0], half_extents[1], ranks[0], ranks[1], num_regions=1)
            
            if regions:
                mask_count = 0
                for count, selected_region in enumerate(regions):
                    if selected_region["direction"] == "horizontal":
                        tmp_region_min_x = selected_region["center"][0] - selected_region["half_width"]  - self.WIDTH_ / 2 - 0.1
                        tmp_region_max_x = selected_region["center"][0] + selected_region["half_width"]  - self.WIDTH_ / 2 + 0.1
                        tmp_region_min_y = selected_region["center"][1] - selected_region["half_height"] - self.LENGTH_ / 2 - 0.1
                        tmp_region_max_y = selected_region["center"][1] + selected_region["half_height"] - self.LENGTH_ / 2 + 0.1
                    else:
                        tmp_region_min_x = selected_region["center"][0] - selected_region["half_width"] - self.WIDTH_ /2 - 0.1
                        tmp_region_max_x = selected_region["center"][0] + selected_region["half_width"] - self.WIDTH_ /2 + 0.1
                        tmp_region_min_y = selected_region["center"][1] - selected_region["half_height"] - self.LENGTH_ / 2 - 0.1
                        tmp_region_max_y = selected_region["center"][1] + selected_region["half_height"] - self.LENGTH_ / 2 + 0.1
                    region_outlines = [(tmp_region_min_x, tmp_region_min_y), (tmp_region_min_x, tmp_region_max_y), (tmp_region_max_x, tmp_region_max_y), (tmp_region_max_x, tmp_region_min_y)]
                    for rid in range(len(region_outlines)):
                        if rid % 2 == 0:
                            rid_len = tmp_region_max_y - tmp_region_min_y
                            scale = rid_len / 4
                            roz_z = np.pi/2
                        else:
                            rid_len = tmp_region_max_x - tmp_region_min_x
                            scale = rid_len / 4
                            roz_z = 0
                        
                        if rid == 0:
                            r_locate = (tmp_region_min_x, tmp_region_min_y, 0.001)
                        elif rid == 1:
                            r_locate = (tmp_region_min_x, tmp_region_max_y, 0.001)
                        elif rid == 2:
                            r_locate = (tmp_region_max_x, tmp_region_min_y, 0.001)
                        elif rid == 3:
                            r_locate = (tmp_region_min_x, tmp_region_min_y, 0.001)

                        floordecal_line_prims = prims.create_prim(
                        prim_path=f"/World/env/Goods_regoopm_floordecal_line_{id}_{count}_{rid}",
                        position=r_locate,
                        scale=(scale, 1, 1),
                        orientation=euler_angles_to_quat([0, 0, roz_z]),
                        usd_path=floor_decal_line_url_path,
                        semantic_label=self.config["floordecal_stripfull"]["class"],
                        ) 
                        self.set_semantics_recursive(floordecal_line_prims, self.config["floordecal_stripfull"]["class"])
                        
                        
                    masked = random.randint(0, 1)
                    if masked:
                        mask_count += 1
                        masks = self.map.fill_selected_region(
                            selected_region,
                            num_masks=self.linear_range_goods_counts[id],     # 生成8个mask
                            priority=0.7,    # 优先级0.7
                            max_per_cell=2   # 每个网格单元最多2个mask
                        )
                        
                        print(f"区域中心: {selected_region['center']}")
                        print(f"区域尺寸: {selected_region['half_width']*2:.2f}x{selected_region['half_height']*2:.2f}")
                        print(f"成功添加mask: {len(masks)}个")
                        
                        # 检查每个mask的位置
                        tested_mask_center_set = {}
                        for i, mask in enumerate(masks):
                            print(f"Mask {i}: {mask['center_physical']}")
                            if mask['center_physical'] in tested_mask_center_set:
                                position = (mask['center_physical'][0] - self.WIDTH_/2, mask['center_physical'][1]-self.LENGTH_/2, goods_height * tested_mask_center_set[mask['center_physical']])
                                tested_mask_center_set[mask['center_physical']] += 1
                            else:
                                position = (mask['center_physical'][0] - self.WIDTH_/2, mask['center_physical'][1]-self.LENGTH_/2, 0)
                                tested_mask_center_set[mask['center_physical']] = 1
                            
                            
                            tmp_prims_path = f"/World/env/goods/cage_{id+1}_{mask_count}_{i+1}"
                            tmp_position = position
                            tmp_oritation = [0, 0, 0]
                            good_xform = self.create_objs_instance(
                                tmp_prims_path, tmp_position, tmp_oritation, 
                                self.ReferObjPaths[goodName]["path"], 
                                self.ReferObjPaths[goodName]["label"])
                            self.add_colliders(good_xform)
                            # self.add_rigid_body_dynamics(Goods_prims)
                        
                        #### 添加相机
                        self.CreateBrightEyesCamera(selected_region['center'], f"Region_{id}_{mask_count}")
                        
                                  
    def randomchoiseGoods_in_arange(self):
        cage_or_box = random.randint(0,1)
        if (cage_or_box == 0):
            halfPath = self.config["goodChoise"]["cage"]
        else:
            halfPath = self.config["goodChoise"]["cage"]
        rootPath = os.path.join(self.assets_root_path, halfPath)
        # 构建匹配模式，** 表示匹配所有子目录
        pattern = os.path.join(rootPath, "**", "*.usd*")
        # 进行递归匹配
        self.goods_in_arange_usd_files = glob.glob(pattern, recursive=True)

        goodsPath = random.choice(self.goods_in_arange_usd_files)
        
        goodName = os.path.basename(goodsPath).split('.')[0]
        
        ref_result = self.GetReferObjInformation(goodName, goodsPath, "cage")
        
        return goodsPath
    
    def measure_half_extent(self, prim_path):
        test_prim = prims.create_prim(
            prim_path=f"/World/test/test_{self.id}",
            position=(0, 0, 0),
            orientation=euler_angles_to_quat([0, 0, 0]),
            usd_path=prim_path,
            semantic_label="test",
        )
        
        bb_cache = create_bbox_cache()
        _, _, half_extent = compute_obb(bb_cache, test_prim.GetPrimPath())
        if min(half_extent) > min(self.WIDTH_, self.LENGTH_) * 0.2:
            half_extent = [half_extent[0] / 100, half_extent[1] / 100, half_extent[2] / 100]
        
        # 获取当前USD舞台（Stage）
        stage = omni.usd.get_context().get_stage()

        # 指定要删除的Prim路径
        prim_path = Sdf.Path(f"/World/test/test_{self.id}")

        # 检查该路径下是否存在Prim，然后删除
        if stage.GetPrimAtPath(prim_path):
            # 方法1: 使用DeletePrims命令（更推荐，类似于界面操作）
            omni.kit.commands.execute("DeletePrims", paths=[prim_path])
            
            # 方法2: 直接使用Stage的RemovePrim方法
            # stage.RemovePrim(prim_path)
            print(f"已删除Prim: {prim_path}")
        else:
            print(f"未找到指定路径的Prim: {prim_path}")
        
        self.id += 1
                
                
        return half_extent
    
    def measure_shelf_extent(self):
        usd_side_path=self.assets_root_path + self.config["sideshelf"]['url']
        usd_shelf_path=self.assets_root_path + self.config["railshelf"]['url']
        
        side_extent = self.measure_half_extent(usd_side_path)
        shelf_extent = self.measure_half_extent(usd_shelf_path)
        
        print(f"side_extent: {side_extent}")
        print(f"shelf_extent: {shelf_extent}")
        
        half_height = side_extent[2]
        half_lenth,half_width = shelf_extent[0], shelf_extent[1]
        
        shelf_half_extent = [half_lenth, half_width, half_height]
                       
        return shelf_half_extent
    
    def CreateBrightEyesCamera(self, center:Tuple[float], name:str):
        center_x = center[0] - self.WIDTH_ / 2
        center_y = center[1] - self.LENGTH_ / 2
        
        position_up = (center_x, center_y, 6)
        position_left_up = (-self.WIDTH_/2 + 1, center_y, 6)
        position_right_up = (self.WIDTH_/2 - 1, center_y, 6)
        position_front_up = (center_x, -self.LENGTH_/2 + 1, 6)
        position_rear_up = (center_x, self.LENGTH_/2 - 1, 6)
        
        look_postion = (center_x, center_y, 0)
        
        tmp_up_camera = rep.create.camera(
            position=position_up, look_at=look_postion,
            focus_distance=400.0, focal_length=4.0, clipping_range=(0.1, 10000000.0), name=f"{name}_up"
        )
        tmp_left_up_camera = rep.create.camera(
            position=position_left_up, look_at=look_postion,
            focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name=f"{name}_left_up"
        )
        tmp_right_up_camera = rep.create.camera(
            position=position_right_up, look_at=look_postion,
            focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name=f"{name}_right_up"
        )
        tmp_front_up_camera = rep.create.camera(
            position=position_front_up, look_at=look_postion,
            focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name=f"{name}_front_up"
        )
        tmp_rear_up_camera = rep.create.camera(
            position=position_rear_up, look_at=look_postion,
            focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name=f"{name}_rear_up"
        )

        resolution = self.config.get("resolution", (512, 512))
        unique_tag = str(int(time.time() * 1000))  # 或者 UUID
        top_rp = rep.create.render_product(tmp_up_camera, resolution, name=f"{name}_TopView_{unique_tag}")
        left_rp = rep.create.render_product(tmp_left_up_camera, resolution, name=f"{name}_LeftView_{unique_tag}")
        right_rp = rep.create.render_product(tmp_right_up_camera, resolution, name=f"{name}_RightView_{unique_tag}")
        front_rp = rep.create.render_product(tmp_front_up_camera, resolution, name=f"{name}_FrontView_{unique_tag}")
        rear_rp = rep.create.render_product(tmp_rear_up_camera, resolution, name=f"{name}_RearView_{unique_tag}")
        # Disable the render products until SDG to improve perf by avoiding unnecessary rendering
        rps = [top_rp, left_rp, right_rp, front_rp, rear_rp]
        for rp in rps:
            rp.hydra_texture.set_updates_enabled(False)
            self.rps.append(rp)
    
    def WriteImages(self):
        # 初始化场景
        # self.world.reset()

        # # 关键改动：先播放时间线，让动画系统初始化
        # self.timeline.set_current_time(0.0)
        # self.timeline.play()
        
        # # 更新多帧让动画系统完全初始化
        # for _ in range(20):  # 增加更新次数
        #     omni.kit.app.get_app().update()
        
        # # 跳转到随机时间点
        # target_time = random.uniform(0.5, 2.0)
        # self.timeline.set_current_time(target_time)
        
        # # 再次更新确保动画姿态应用
        # for _ in range(10):
        #     omni.kit.app.get_app().update()
        
        # # 暂停并最后更新一次
        # self.timeline.stop()
        # for _ in range(3):
        #     omni.kit.app.get_app().update()


        # If output directory is relative, set it relative to the current working directory
        if not os.path.isabs(self.config["writer_config"]["output_dir"]):
            self.config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), self.config["writer_config"]["output_dir"])
        print(f"[scene_based_sdg] Output directory={self.config['writer_config']['output_dir']}")

        # Make sure the writer type is in the registry
        writer_type = self.config.get("writer", "BasicWriter")
        if writer_type not in rep.WriterRegistry.get_writers():
            # carb.log_error(f"Writer type {writer_type} not found in the registry, closing application..")
            return
        #     simulation_app.close()

        # Get the writer from the registry and initialize it with the given config parameters
        writer = rep.WriterRegistry.get(writer_type)
        writer_kwargs = self.config["writer_config"]
        print(f"[scene_based_sdg] Initializing {writer_type} with: {writer_kwargs}")
        writer.initialize(**writer_kwargs)

        # Attach writer to the render products
        writer.attach(self.rps)

        rt_subframes = self.config.get("rt_subframes", -1)

        # Enable the render products for SDG
        for rp in self.rps:
            rp.hydra_texture.set_updates_enabled(True)

        # Start the SDG
        num_frames = self.config.get("num_frames", 0)
        print(f"[scene_based_sdg] Running SDG for {num_frames} frames")
        for i in range(num_frames):
            print(f"[scene_based_sdg] \t Capturing frame {i}")
            # Trigger the custom event to randomize the cones at specific frames
            # if i % 2 == 0:
            #     rep.utils.send_og_event(event_name="randomize_cones")
            # Trigger any on_frame registered randomizers and the writers (delta_time=0.0 to avoid advancing the timeline)
            rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)

        # Wait for the data to be written to disk
        rep.orchestrator.wait_until_complete()

        # Cleanup writer and render products
        writer.detach()
        for rp in self.rps:
            rp.destroy()
    
    def WriteMoveImages(self):
        # # 初始化场景
        # self.world.reset()

        # If output directory is relative, set it relative to the current working directory
        if not os.path.isabs(self.config["writer_config"]["output_dir"]):
            self.config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), self.config["writer_config"]["output_dir"])
        print(f"[scene_based_sdg] Output directory={self.config['writer_config']['output_dir']}")

        # Make sure the writer type is in the registry
        writer_type = self.config.get("writer", "BasicWriter")
        if writer_type not in rep.WriterRegistry.get_writers():
            # carb.log_error(f"Writer type {writer_type} not found in the registry, closing application..")
            return
        #     simulation_app.close()

        # Get the writer from the registry and initialize it with the given config parameters
        writer = rep.WriterRegistry.get(writer_type)
        writer.async_write=False
        writer_kwargs = self.config["writer_config"]
        print(f"[scene_based_sdg] Initializing {writer_type} with: {writer_kwargs}")
        writer.initialize(**writer_kwargs)

        # Attach writer to the render products
        writer.attach(self.move_rps)
        
        with rep.trigger.on_frame():
            position = rep.distribution.sequence(self.move_cam_pos)
            look_left = rep.distribution.sequence(self.move_cam_left_look)
            look_right = rep.distribution.sequence(self.move_cam_right_look)
            with self.move_left_camera: 
                rep.modify.pose(
                    position=position,
                    look_at=look_left,
                )
            with self.move_right_camera: 
                rep.modify.pose(
                    position=position,
                    look_at=look_right,
                )
        
        with rep.trigger.on_frame():
            position_up = rep.distribution.sequence(self.move_cam_top_pos)
            position = rep.distribution.sequence(self.move_cam_pos)
            with self.move_top_camera: 
                rep.modify.pose(
                    position=position_up,
                    look_at=position,
                )
        

        rt_subframes = self.config.get("rt_subframes", -1)

        # Enable the render products for SDG
        for rp in self.move_rps:
            rp.hydra_texture.set_updates_enabled(True)

        # Start the SDG
        num_move_frames = len(self.move_cam_pos)
        print(f"[scene_based_sdg] Running SDG for {num_move_frames} frames")
        for i in range(num_move_frames):
            print(f"[scene_based_sdg] \t Capturing frame {i}")
            # # Trigger the custom event to randomize the cones at specific frames
            # if i % 2 == 0:
            #     rep.utils.send_og_event(event_name="randomize_cones")
            # Trigger any on_frame registered randomizers and the writers (delta_time=0.0 to avoid advancing the timeline)
            rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, wait_for_render=True)

        # Wait for the data to be written to disk
        rep.orchestrator.wait_until_complete()

        # Cleanup writer and render products
        writer.detach()
        for rp in self.move_rps:
            rp.destroy()

    
    
    # Enables collisions with the asset (without rigid body dynamics the asset will be static)
    def add_colliders(self, root_prim):
        # Iterate descendant prims (including root) and add colliders to mesh or primitive types
        for desc_prim in Usd.PrimRange(root_prim):
            if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim) or desc_prim.IsA(UsdGeom.Xform):
                # Physics
                if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
                else:
                    collision_api = UsdPhysics.CollisionAPI(desc_prim)
                collision_api.CreateCollisionEnabledAttr(True)
                # PhysX
                if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
                else:
                    physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
                # Set PhysX specific properties
                physx_collision_api.CreateContactOffsetAttr(0.001)
                physx_collision_api.CreateRestOffsetAttr(0.0)

            # Add mesh specific collision properties only to mesh types
            if desc_prim.IsA(UsdGeom.Mesh):
                # Add mesh collision properties to the mesh (e.g. collider aproximation type)
                if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
                else:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
                mesh_collision_api.CreateApproximationAttr().Set("convexHull")

    # Check if prim (or its descendants) has colliders
    def has_colliders(self, root_prim):
        for desc_prim in Usd.PrimRange(root_prim):
            if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                return True
        return False

    # Enables rigid body dynamics (physics simulation) on the prim
    def add_rigid_body_dynamics(self, prim, disable_gravity=False, angular_damping=None):
        if self.has_colliders(prim):
            # Physics
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            else:
                rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            rigid_body_api.CreateRigidBodyEnabledAttr(True)
            # PhysX
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            else:
                physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
            if angular_damping is not None:
                physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
        else:
            print(f"Prim '{prim.GetPath()}' has no colliders. Skipping rigid body dynamics properties.")
    
    def Close_stage(self):
        import omni.usd

        usd_context = omni.usd.get_context()

        # 关闭当前 stage
        usd_context.close_stage()

        # 重置区域以便重新使用
        self.map.reset_region()
        
        # 清除所有掩码
        self.map.clear_masks()
        

        print("✅ 已关闭当前 stage 并新建空白 Stage")
        

def ResolveJson(jsonPath):
    with open(jsonPath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return convert_string_bools(data)

def convert_string_bools(obj):
    """
    递归遍历数据结构，将字符串 "False" 和 "True" 转换为 Python 的布尔值。
    同样也可以处理 "None" 或其他需要转换的字符串。
    """
    if isinstance(obj, dict):
        # 如果是字典，遍历每个键值对
        for key, value in obj.items():
            obj[key] = convert_string_bools(value)
        return obj
    elif isinstance(obj, list):
        # 如果是列表，遍历每个元素
        return [convert_string_bools(item) for item in obj]
    elif isinstance(obj, str):
        # 如果是字符串，检查是否是特定的需要转换的值
        if obj.lower() == "false": # 处理大小写可能不一致的情况
            return False
        elif obj.lower() == "true":
            return True
        elif obj.lower() == "none" or obj.lower() == "null":
            return None
        # 添加其他你需要的字符串转换规则...
        else:
            return obj
    else:
        # 如果不是字典、列表或字符串，直接返回（如数字、真正的布尔值等）
        return obj