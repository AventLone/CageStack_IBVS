import numpy as np
import random, math, os
from typing import Callable

import omni.kit.app
import omni.usd
from isaacsim.core.utils import prims, xforms
import isaacsim
import omni.replicator.core as rep
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.core.utils.bounds import (
    compute_combined_aabb,
    compute_obb,
    create_bbox_cache,
    get_obb_corners
)
from isaacsim.core.prims import SingleRigidPrim

from pxr import (
    Gf,
    Vt,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdSkel,
    UsdShade,
    UsdUtils,
    PhysicsSchemaTools,
    PhysxSchema
)

omni.usd.get_context().new_stage()
prims.create_prim("/World", prim_type="Xform")

class EnvProxy:
    def __init__(self, env_name: str) -> None:
        self._env_name = env_name
        self._prims_path_dict = dict()
        self._prims_dict = dict()
        prims.create_prim(f"/World/{env_name}", prim_type="Xform")

    def __call__(self, construct: Callable, prim_name: str) -> None:
        if prim_name not in self._prims_dict:
            self._prims_path_dict[prim_name], self._prims_dict[prim_name] = construct(self._env_name, prim_name)
        else:
            raise KeyError(f"Failed to construct the prim, as {prim_name} has already existed!")

    def get_prim(self, prim_name: str):
        return self._prims_dict[prim_name]

    def get_prim_pose(self, prim_name: str):
        return xforms.get_world_pose(self._prims_path_dict[prim_name])

    def set_prim_position(self, prim_name: str, position) -> None:
        self._prims_dict[prim_name].GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))

    def set_prim_orientation(self, prim_name: str, orientation) -> None:
        self._prims_dict[prim_name].GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(*orientation))



class EnvConstructor:
    def __init__(self, config) -> None:
        self.config = config
        self.assets_root_path = self.config["root_path"]

        self.width = 30
        self.length = 30

        self.occupied_areas = []
        self.env_id = 0
        self.shelf_id = 0
        self.forklift_id = 0
        self.container_id = 0
        self.box_id = 0
        self.debris_id = 0
        self.offset = 0

    def create_warehouse(self, env_name: str, prim_name: str, width=42, length=42):
        self.width = width
        self.length = length
        x_min = -width / 2 + self.offset
        x_max = width / 2 + self.offset
        y_min = -length / 2
        y_max = length / 2

        parent_prim_path = f"/World/{env_name}/{prim_name}"
        parent_prim = prims.create_prim(parent_prim_path, prim_type="Xform")

        # floor count
        count = 1
        for x in np.arange(x_min, x_max, 6):
            for y in np.arange(y_min, y_max, 6):

                position = (x + 3, y + 3, 0)
                position_cell = (x + 3, y + 3, 9)
                floor_prim = prims.create_prim(
                    prim_path=f"{parent_prim_path}/Floor_{count}",
                    position=position,
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path
                    + self.config["warehouse"]["floor"]["url"],
                    semantic_label=self.config["warehouse"]["floor"]["class"],
                )
                self._add_colliders(floor_prim, approx_type="none")
                ceil_prim = prims.create_prim(
                    prim_path=f"{parent_prim_path}/Cell_{count}",
                    position=position_cell,
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path
                    + self.config["warehouse"]["ceil"]["url"],
                    semantic_label=self.config["warehouse"]["ceil"]["class"],
                )
                count += 1

        # wall_A
        count = 1
        left_num = int(np.ceil(width / 6))
        front_num = int(np.ceil(length / 6))
        pillarLen_num = int(np.ceil((width - 6) / 9)) if width > 6 else 0
        beam_num = int(np.ceil(length / 9))

        count_all = 2 * (left_num + front_num)

        for i in range(count_all):
            if i < left_num:
                if 6 * (i + 1) <= width:
                    x = x_min + 6 * i + 3
                else:
                    x = x_max - 3
                y = y_min
                rotation_wall = [0, 0, np.pi / 2]
            elif i < left_num + front_num:
                if 6 * (i + 1 - left_num) <= length:
                    y = y_min + 6 * ((i - left_num)) + 3
                else:
                    y = y_max - 3
                x = x_max
                rotation_wall = [0, 0, np.pi]
            elif i < 2 * left_num + front_num:
                if 6 * (i + 1 - left_num - front_num) <= width:
                    x = x_min + 6 * (i - left_num - front_num) + 3
                else:
                    x = x_max - 3
                y = y_max
                rotation_wall = [0, 0, -np.pi / 2]
            else:
                if 6 * (i + 1 - 2 * left_num - front_num) <= length:
                    y = y_min + 6 * ((i - 2 * left_num - front_num)) + 3
                else:
                    y = y_max - 3
                x = x_min
                rotation_wall = [0, 0, 0]

            position_wall_a = (x, y, 0)
            position_wall_b = (x, y, 3)
            position_wall_b_up = (x, y, 6)
            position_pillar = (x, y, 0)

            wallA_prim = prims.create_prim(
                prim_path=f"{parent_prim_path}/wall_A_{i}",
                position=position_wall_a,
                orientation=euler_angles_to_quat(rotation_wall),
                usd_path=self.assets_root_path
                + self.config["warehouse"]["walldown"]["url"],
                semantic_label=self.config["warehouse"]["walldown"]["class"],
            )
            wallB_prim = prims.create_prim(
                prim_path=f"{parent_prim_path}/wall_B_{i}",
                position=position_wall_b,
                orientation=euler_angles_to_quat(rotation_wall),
                usd_path=self.assets_root_path
                + self.config["warehouse"]["walluppon"]["url"],
                semantic_label=self.config["warehouse"]["walluppon"]["class"],
            )
            wallB_up_prim = prims.create_prim(
                prim_path=f"{parent_prim_path}/wall_B_up_{i}",
                position=position_wall_b_up,
                orientation=euler_angles_to_quat(rotation_wall),
                usd_path=self.assets_root_path
                + self.config["warehouse"]["walluppon"]["url"],
                semantic_label=self.config["warehouse"]["walluppon"]["class"],
            )
            if i < left_num - 1:
                for t in range(beam_num):
                    if (y_min + t * 9) > y_max - 9:
                        beam_position = (
                            x_min + (i + 1) * 6,
                            y_max - 9,
                            9,
                        )
                    else:
                        beam_position = (
                            x_min + (i + 1) * 6,
                            y_min + t * 9,
                            9,
                        )
                    beam_prim = prims.create_prim(
                        prim_path=f"{parent_prim_path}/Beam_{i}_{t}",
                        position=beam_position,
                        orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                        usd_path=self.assets_root_path
                        + self.config["warehouse"]["beam"]["url"],
                        semantic_label=self.config["warehouse"]["beam"]["class"],
                    )
            if left_num < i < left_num + front_num - 1:
                pillar_prim = prims.create_prim(
                    prim_path=f"{parent_prim_path}/Pillar_{i}",
                    position=position_pillar,
                    orientation=euler_angles_to_quat(rotation_wall),
                    usd_path=self.assets_root_path
                    + self.config["warehouse"]["pillar"]["url"],
                    semantic_label=self.config["warehouse"]["pillar"]["class"],
                )
                for t in range(pillarLen_num):
                    position_pillar_len = (3 + (t + 1) * 9 + x_min, y, 9)
                    pillar_len_prim = prims.create_prim(
                        prim_path=f"{parent_prim_path}/Pillar_len_{i}_{t}",
                        position=position_pillar_len,
                        orientation=euler_angles_to_quat([0, -np.pi / 2, 0]),
                        usd_path=self.assets_root_path
                        + self.config["warehouse"]["pillarLen"]["url"],
                        semantic_label=self.config["warehouse"]["pillarLen"]["class"],
                    )

            elif 2 * left_num + front_num < i < count_all - 1:
                pillar_prim = prims.create_prim(
                    prim_path=f"{parent_prim_path}/Pillar_{i}",
                    position=position_pillar,
                    orientation=euler_angles_to_quat(rotation_wall),
                    usd_path=self.assets_root_path
                    + self.config["warehouse"]["pillar"]["url"],
                    semantic_label=self.config["warehouse"]["pillar"]["class"],
                )
        count = 1
        for i in np.arange(x_min + 5, x_max, 8):
            for j in np.arange(y_min + 5, y_max, 6):

                light_position = (i, j, 9)
                light_prim = prims.create_prim(
                    prim_path=f"{parent_prim_path}/Light_prim_{count}",
                    position=light_position,
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path
                    + self.config["warehouse"]["light"]["url"],
                    semantic_label=self.config["warehouse"]["light"]["class"],
                )
                count += 1

        self.env_id += 1
        self.offset += width

        return parent_prim_path, parent_prim

    # def create_react_light(self):


    def create_forklift(self, env_name: str, prim_name: str,
                        center=(0, 0),
                        half_width=7.5,
                        half_length=7.5,
                        yaw=(-math.pi / 2, -math.pi / 2)):
        cx, cy = center
        offset = 0
        if (self.env_id - 1) != 0:
            offset = self.offset - self.width
            cx += offset

        if not (-self.width / 2 + offset <= cx <= self.width / 2 + offset):
            raise ValueError(
                f"Center X={cx} 超出场景范围 [-{self.width/2 + offset}, {self.width/2 + offset}]"
            )
        if not (-self.length / 2 <= cy <= self.length / 2):
            raise ValueError(
                f"Center Y={cy} 超出场景范围 [-{self.length/2}, {self.length/2}]"
            )

        min_x = max(cx - half_width, -self.width / 2 + offset)
        max_x = min(cx + half_width, self.width / 2 + offset)
        if min_x != cx - half_width or max_x != cx + half_width:
            print("X 方向：所设置的范围超出场景范围，已对边界更新")

        min_y = max(cy - half_length, -self.length / 2)
        max_y = min(cy + half_length, self.length / 2)
        if min_y != cy - half_length or max_y != cy + half_length:
            print("Y 方向：所设置的范围超出场景范围，已对边界更新")

        range_x = (min_x, max_x)
        range_y = (min_y, max_y)

        # self._forklift_prim = f"/World/Forklift_{self.forklift_id}"

        prim_path = f"/World/{env_name}/{prim_name}"


        forklift_prim = prims.create_prim(
            prim_path=prim_path,
            position=(
                random.uniform(*range_x),
                random.uniform(*range_y),
                0.1,
            ),
            orientation=euler_angles_to_quat([0, 0, random.uniform(*yaw)]),
            usd_path=self.assets_root_path + self.config["forklift"]["url"],
            semantic_label=self.config["forklift"]["class"],
        )

        min_pt, max_pt = self._get_prim_world_size(forklift_prim)
        if not self._check_collision(min_pt, max_pt):
            self.occupied_areas.append((min_pt, max_pt))
        else:
            raise RuntimeError("叉车无法放置，生成位置有误，请更改设置条件")

        self.forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
        self.forklift_id += 1

        return prim_path, forklift_prim

    def create_containers(
        self,
        env_name: str, prim_name: str,
        min_pt=(-1, 1, 0),
        max_pt=(-3, -5, 0),
        yaw=(0, 0),
        name="pallet",
        has_cargo=True,
        cargo_nums=12
    ):
        self.container_name = name

        offset = 0
        if (self.env_id - 1) != 0:
            offset = self.offset - self.width

        pos = (
            random.uniform(min_pt[0], max_pt[0]) + offset,
            random.uniform(min_pt[1], max_pt[1]),
            0,
        )
        yaw_ = random.uniform(*yaw)
        self._containers_prim = f"/World/container_{self.container_id}"

        prim_path = f"/World/{env_name}/{prim_name}"

        container_prim = prims.create_prim(
            prim_path=prim_path,
            position=pos,
            orientation=euler_angles_to_quat([0, 0, yaw_]),
            usd_path=self.assets_root_path
            + self.config["target_containers"][name]["url"],
            semantic_label=self.config["target_containers"][name]["class"],
        )
        self._rep_target_container = rep.get.prim_at_path(self._containers_prim)

        materials = self._create_materials(10)
        rand_mat = random.choice(materials)

        mesh_prims = [
            prim
            for prim in Usd.PrimRange(container_prim)
            if prim.GetTypeName() == "Mesh"
        ]

        for mesh_prim in mesh_prims:
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(
                rand_mat, UsdShade.Tokens.strongerThanDescendants
            )

            self._add_colliders(mesh_prim, approx_type="convexDecomposition")
            container_rigid_prim = SingleRigidPrim(
                prim_path=str(mesh_prim.GetPrimPath())
            )
            container_rigid_prim.enable_rigid_body_physics()

        if has_cargo:
            self._create_container_goods(
                prim_path=self._containers_prim,
                container_prim=container_prim,
                yaw=yaw_,
                num_boxes=cargo_nums,
            )

        min_pt, max_pt = self._get_prim_world_size(container_prim)

        f_min, f_max = self.occupied_areas.pop(0)
        min_pt = Gf.Vec2d(min(f_min[0], min_pt[0]), min(f_min[1], min_pt[1]))
        max_pt = Gf.Vec2d(max(f_max[0], max_pt[0]), max(f_max[1], max_pt[1]))
        self.occupied_areas.append((min_pt, max_pt))

        self.container_id += 1

        return prim_path, container_prim

    def create_shelf(
        self,
        env_name, prim_name,
        shelf_orin="vertical",
        shelf_bays=(2, 3),
        shelf_rows=(2, 6),
        row_gap=(2, 3),
    ):
        # shelf_orin = random.choice(["vertical", "horizon"])
        shelf_bays = random.randint(*shelf_bays)
        shelf_rows = random.randint(*shelf_rows)
        row_gap = random.uniform(*row_gap)
        self.shelf_half_extent = self._measure_shelf_extent()
        shelf_length = 2 * shelf_bays * self.shelf_half_extent[0]
        shelf_width = 2 * self.shelf_half_extent[1]

        group_base = shelf_rows * shelf_width + (shelf_rows - 1) * row_gap

        group_size_x = group_base if shelf_orin == "vertical" else shelf_length
        group_size_y = shelf_length if shelf_orin == "vertical" else group_base

        shelf_center = self._find_free_center(group_size_x + 3, group_size_y + 3)
        if shelf_orin == "vertical":
            shelf_center = (
                shelf_center[0] - group_size_x / 2 + shelf_width / 2,
                shelf_center[1] - shelf_length / 2,
            )
        else:
            shelf_center = (
                shelf_center[0] - shelf_length / 2,
                shelf_center[1] - group_size_x / 2 + shelf_width / 2,
            )

        self._shelf_prim = f"/World/shelf_{self.shelf_id}"
        prim_path = f"/World/{env_name}/{prim_name}"
        prims.create_prim(
            # self._shelf_prim,
            prim_path=prim_path,
            prim_type="Xform",
            position=(shelf_center[0], shelf_center[1], 0),
        )
        for id in range(shelf_rows):
            if shelf_orin == "vertical":
                shelf_min_length_location = shelf_center[1]
                nums = int(shelf_length / self.shelf_half_extent[0] / 2)

                shelf_sheild_left_prim = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_sheild_{id}_0",
                    position=(
                        shelf_center[0],
                        shelf_min_length_location,
                        0,
                    ),
                    orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["rackshield"]["url"],
                    semantic_label=self.config["shelf"]["rackshield"]["class"],
                )

                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_redrec_{id}_0",
                    position=(
                        shelf_center[0],
                        shelf_min_length_location,
                        0.001,
                    ),
                    orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                    scale=(1.1, 1.1, 1),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["floordecal_recred"]["url"],
                    semantic_label=self.config["shelf"]["floordecal_recred"]["class"],
                )

                for num in range(nums):
                    location_x = shelf_center[0]
                    location_y = shelf_min_length_location + num * (
                        self.shelf_half_extent[0] * 2
                    )

                    for t in range(2):
                        shelf_side_prims = prims.create_prim(
                            prim_path=f"/World/shelf_{self.shelf_id}/shelf_side_{id}_{num}_{t}",
                            position=(
                                location_x,
                                location_y,
                                t * 2 * self.shelf_half_extent[2],
                            ),
                            orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                            usd_path=self.assets_root_path
                            + self.config["shelf"]["sideshelf"]["url"],
                            semantic_label=self.config["shelf"]["sideshelf"]["class"],
                        )

                    floordecal_line_prims = prims.create_prim(
                        prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_line_{id}_{num+1}_0",
                        position=(
                            location_x - self.shelf_half_extent[1],
                            location_y,
                            0.001,
                        ),
                        orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                        usd_path=self.assets_root_path
                        + self.config["shelf"]["floordecal_stripfull"]["url"],
                        semantic_label=self.config["shelf"]["floordecal_stripfull"][
                            "class"
                        ],
                    )

                    floordecal_line_prims = prims.create_prim(
                        prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_line_{id}_{num+1}_1",
                        position=(
                            location_x + self.shelf_half_extent[1],
                            location_y,
                            0.001,
                        ),
                        orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                        usd_path=self.assets_root_path
                        + self.config["shelf"]["floordecal_stripfull"]["url"],
                        semantic_label=self.config["shelf"]["floordecal_stripfull"][
                            "class"
                        ],
                    )

                    shelf_nums = random.randint(2, 3)
                    shelf_gaps = [
                        round(random.uniform(1.5, 2), 1) for _ in range(shelf_nums)
                    ]
                    total_gap = 0
                    for j, gap in enumerate(shelf_gaps):
                        total_gap += gap

                        shelf_rail_prims = prims.create_prim(
                            prim_path=f"/World/shelf_{self.shelf_id}/shelf_rail_{id}_{num+1}_{j}",
                            position=(
                                location_x,
                                location_y + self.shelf_half_extent[0],
                                total_gap,
                            ),
                            orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                            usd_path=self.assets_root_path
                            + self.config["shelf"]["railshelf"]["url"],
                            semantic_label=self.config["shelf"]["railshelf"]["class"],
                        )

                        self._create_shelfgoods(
                            shelflocation=(
                                location_x,
                                location_y + self.shelf_half_extent[0],
                                total_gap - gap,
                            ),
                            shelfGap=gap,
                            base_xform=f"/World/shelf_{self.shelf_id}/shelfgoods_{id}_{num+1}_{j}",
                            oritation="vertical",
                        )

                        if j == (len(shelf_gaps) - 1):
                            self._create_shelfgoods(
                                shelflocation=(
                                    location_x,
                                    location_y + self.shelf_half_extent[0],
                                    total_gap,
                                ),
                                shelfGap=1.0,
                                base_xform=f"/World/shelf_{self.shelf_id}/shelfgoods_{id}_{num+1}_{j+1}",
                                oritation="vertical",
                            )

                for t in range(2):
                    shelf_side2_prims = prims.create_prim(
                        prim_path=f"/World/shelf_{self.shelf_id}/shelf_side_{id}_{nums}_{t}",
                        position=(
                            location_x,
                            shelf_min_length_location + shelf_length,
                            t * 2 * self.shelf_half_extent[2],
                        ),
                        orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                        usd_path=self.assets_root_path
                        + self.config["shelf"]["sideshelf"]["url"],
                        semantic_label=self.config["shelf"]["sideshelf"]["class"],
                    )

                shelf_sheild_right_prims = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_sheild_{id}_1",
                    position=(
                        shelf_center[0],
                        shelf_min_length_location + shelf_length,
                        0,
                    ),
                    orientation=euler_angles_to_quat([0, 0, np.pi / 2]),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["rackshield"]["url"],
                    semantic_label=self.config["shelf"]["rackshield"]["class"],
                )
                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_redrec_{id}_1",
                    position=(
                        shelf_center[0],
                        shelf_min_length_location + shelf_length,
                        0.001,
                    ),
                    orientation=euler_angles_to_quat([0, 0, -np.pi / 2]),
                    scale=(1.1, 1.1, 1),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["floordecal_recred"]["url"],
                    semantic_label=self.config["shelf"]["floordecal_recred"]["class"],
                )
            else:
                shelf_min_length_location = shelf_center[0]
                nums = int(shelf_length / self.shelf_half_extent[0] / 2)

                shelf_sheild_left_prim = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_sheild_{id}_0",
                    position=(
                        shelf_min_length_location,
                        shelf_center[1],
                        0,
                    ),
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["rackshield"]["url"],
                    semantic_label=self.config["shelf"]["rackshield"]["class"],
                )

                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_redrec_{id}_0",
                    position=(
                        shelf_min_length_location,
                        shelf_center[1],
                        0.001,
                    ),
                    scale=(1.1, 1.1, 1),
                    orientation=euler_angles_to_quat([0, 0, 0]),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["floordecal_recred"]["url"],
                    semantic_label=self.config["shelf"]["floordecal_recred"]["class"],
                )

                for num in range(nums):
                    location_y = shelf_center[1]
                    location_x = shelf_min_length_location + num * (
                        self.shelf_half_extent[0] * 2
                    )

                    for t in range(2):
                        shelf_side_prims = prims.create_prim(
                            prim_path=f"/World/shelf_{self.shelf_id}/shelf_side_{id}_{num}_{t}",
                            position=(
                                location_x,
                                location_y,
                                t * 2 * self.shelf_half_extent[2],
                            ),
                            orientation=euler_angles_to_quat([0, 0, 0]),
                            usd_path=self.assets_root_path
                            + self.config["shelf"]["sideshelf"]["url"],
                            semantic_label=self.config["shelf"]["sideshelf"]["class"],
                        )

                    floordecal_line_prims1 = prims.create_prim(
                        prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_line_{id}_{num+1}_{0}",
                        position=(
                            location_x,
                            location_y - self.shelf_half_extent[1],
                            0.001,
                        ),
                        orientation=euler_angles_to_quat([0, 0, 0]),
                        usd_path=self.assets_root_path
                        + self.config["shelf"]["floordecal_stripfull"]["url"],
                        semantic_label=self.config["shelf"]["floordecal_stripfull"][
                            "class"
                        ],
                    )

                    floordecal_line_prims2 = prims.create_prim(
                        prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_line_{id}_{num+1}_{1}",
                        position=(
                            location_x,
                            location_y + self.shelf_half_extent[1],
                            0.001,
                        ),
                        orientation=euler_angles_to_quat([0, 0, 0]),
                        usd_path=self.assets_root_path
                        + self.config["shelf"]["floordecal_stripfull"]["url"],
                        semantic_label=self.config["shelf"]["floordecal_stripfull"][
                            "class"
                        ],
                    )

                    shelf_nums = random.randint(2, 3)
                    shelf_gaps = [
                        round(random.uniform(1, 2), 1) for _ in range(shelf_nums)
                    ]
                    total_gap = 0
                    for j, gap in enumerate(shelf_gaps):
                        total_gap += gap

                        shelf_rail_prims = prims.create_prim(
                            prim_path=f"/World/shelf_{self.shelf_id}/shelf_rail_{id}_{num+1}_{j}",
                            position=(
                                location_x + self.shelf_half_extent[0],
                                location_y,
                                total_gap,
                            ),
                            orientation=euler_angles_to_quat([0, 0, 0]),
                            usd_path=self.assets_root_path
                            + self.config["shelf"]["railshelf"]["url"],
                            semantic_label=self.config["shelf"]["railshelf"]["class"],
                        )

                        self._create_shelfgoods(
                            shelflocation=(
                                location_x + self.shelf_half_extent[0],
                                location_y,
                                total_gap - gap,
                            ),
                            shelfGap=gap,
                            base_xform=f"/World/shelf_{self.shelf_id}/shelfgoods_{id}_{num+1}_{j}",
                            oritation="horizon",
                        )

                        if j == (len(shelf_gaps) - 1):
                            self._create_shelfgoods(
                                shelflocation=(
                                    location_x + self.shelf_half_extent[0],
                                    location_y,
                                    total_gap,
                                ),
                                shelfGap=1.0,
                                base_xform=f"/World/shelf_{self.shelf_id}/shelfgoods_{id}_{num+1}_{j+1}",
                                oritation="horizon",
                            )

                for t in range(2):
                    shelf_side2_prims = prims.create_prim(
                        prim_path=f"/World/shelf_{self.shelf_id}/shelf_side_{id}_{nums}_{t}",
                        position=(
                            shelf_min_length_location + shelf_length,
                            shelf_center[1],
                            t * 2 * self.shelf_half_extent[2],
                        ),
                        orientation=euler_angles_to_quat([0, 0, 0]),
                        usd_path=self.assets_root_path
                        + self.config["shelf"]["sideshelf"]["url"],
                        semantic_label=self.config["shelf"]["sideshelf"]["class"],
                    )

                shelf_sheild_right_prims = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_sheild_{id}_1",
                    position=(
                        shelf_min_length_location + shelf_length,
                        shelf_center[1],
                        0,
                    ),
                    orientation=euler_angles_to_quat([0, 0, np.pi]),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["rackshield"]["url"],
                    semantic_label=self.config["shelf"]["rackshield"]["class"],
                )
                floordecal_recred_prims = prims.create_prim(
                    prim_path=f"/World/shelf_{self.shelf_id}/shelf_floordecal_redrec_{id}_1",
                    position=(
                        shelf_min_length_location + shelf_length,
                        shelf_center[1],
                        0.001,
                    ),
                    orientation=euler_angles_to_quat([0, 0, np.pi]),
                    scale=(1.1, 1.1, 1),
                    usd_path=self.assets_root_path
                    + self.config["shelf"]["floordecal_recred"]["url"],
                    semantic_label=self.config["shelf"]["floordecal_recred"]["class"],
                )
            if shelf_orin == "vertical":
                shelf_center = (
                    shelf_center[0] + row_gap + shelf_width,
                    shelf_center[1],
                )
            else:
                shelf_center = (
                    shelf_center[0],
                    shelf_center[1] + row_gap + shelf_width,
                )

        # stage = omni.usd.get_context().get_stage()
        shelf = stage.GetPrimAtPath(f"/World/shelf_{self.shelf_id}")
        min_pt, max_pt = self._get_prim_world_size(shelf)
        # if not self._check_collision(min_pt, max_pt):
        self.occupied_areas.append((min_pt, max_pt))
        # else:
        #     raise RuntimeError("货架无法放置，生成位置有误，请更改设置条件")

        self.shelf_id += 1

    def create_debris(self, width=10, length=10, nums=(30, 50), yaw=(0, np.pi)):
        debris_files = self.config["debris"]

        nums = random.randint(*nums)

        position_xy = self._find_free_center(width, length)
        min_pt = (position_xy[0] - width / 2, position_xy[1] - length / 2)
        max_pt = (position_xy[0] + width / 2, position_xy[1] + length / 2)

        self._debris_prim = f"/World/debris_{self.debris_id}"
        prims.create_prim(
            self._debris_prim,
            prim_type="Xform",
            position=(position_xy[0], position_xy[1], 0),
        )
        for i in range(nums):
            debris_key = random.choice(list(debris_files.keys()))
            debrisurl = self.assets_root_path + debris_files[debris_key]["url"]

            debris_path = f"/World/debris_{self.debris_id}/debris_{i+1}"

            pos = (
                random.uniform(min_pt[0], max_pt[0]),
                random.uniform(min_pt[1], max_pt[1]),
                0,
            )
            debris_prims = prims.create_prim(
                prim_path=debris_path,
                position=(
                    pos[0],
                    pos[1],
                    0,
                ),
                orientation=euler_angles_to_quat([0, 0, random.uniform(*yaw)]),
                usd_path=debrisurl,
                semantic_label=debris_files[debris_key]["class"],
            )
            self._add_colliders(debris_prims)
            container_rigid_prim = SingleRigidPrim(
                prim_path=str(debris_prims.GetPrimPath())
            )
            container_rigid_prim.enable_rigid_body_physics()

        self.debris_id += 1

    def create_light(self, intensity=(500, 10000), temperature=(6500, 2000)):
        color = (0.8, 0.8, 0.8)
        offset = 0
        if (self.env_id - 1) != 0:
            offset = self.offset - self.width
        self.rectlight = rep.create.light(
            position=(0 + offset, 0, 6),
            look_at=(0, 0, 0),
            scale=(self.length - 10, self.width - 10, 1),
            color=color,
            intensity=random.uniform(*intensity),
            exposure=1,
            temperature=max(0, random.gauss(*temperature)),
            light_type="rect",
        )

    def _create_container_goods(self, prim_path, container_prim, yaw, num_boxes):
        bb_cache = create_bbox_cache()
        container_size = bb_cache.ComputeLocalBound(container_prim).GetRange().GetSize()
        matrix = omni.usd.get_world_transform_matrix(container_prim)
        container_pos = matrix.ExtractTranslation()

        box_files = self.config["container_boxs"]
        boxurl = self.assets_root_path + random.choice(box_files)
        box_size = self._measure_half_extent(boxurl) * 2

        num_x = int(container_size[0] // box_size[0])
        num_y = int(container_size[1] // box_size[1])

        gap_x = container_size[0] - num_x * box_size[0]
        gap_y = container_size[1] - num_y * box_size[1]

        if gap_x > box_size[0] / 2:
            num_x += 1
            gap_x = container_size[0] - num_x * box_size[0]
        if gap_y > box_size[1] / 2:
            num_y += 1
            gap_y = container_size[1] - num_y * box_size[1]

        total_per_layer = num_x * num_y

        indices = np.arange(num_boxes)
        ix = indices % num_x
        iy = (indices // num_x) % num_y
        iz = indices // total_per_layer

        x_offsets = (
            -container_size[0] / 2 + box_size[0] / 2 + ix * box_size[0] + gap_x / 2
        )
        y_offsets = (
            -container_size[1] / 2 + box_size[1] / 2 + iy * box_size[1] + gap_y / 2
        )
        z_offsets = container_size[2] * 1.1 + iz * box_size[2] * 1.1

        prim_path = prim_path + "/box"
        prims.create_prim(prim_path, prim_type="Xform", position=container_pos)
        for i in range(num_boxes):
            local_offset = Gf.Vec3d(x_offsets[i], y_offsets[i], z_offsets[i])
            rotated_offset = matrix.Transform(local_offset) - container_pos
            pos = container_pos + rotated_offset
            i += self.box_id
            box_prim = prims.create_prim(
                prim_path=prim_path + f"/box_{i}",
                position=pos,
                orientation=euler_angles_to_quat([0, 0, yaw]),
                usd_path=boxurl,
                semantic_label="goods",
            )
            self._add_colliders(box_prim, approx_type="boundingCube")
            box_rigid_prim = SingleRigidPrim(str(box_prim.GetPrimPath()))
            box_rigid_prim.enable_rigid_body_physics()

        self.box_id += num_boxes

    def _create_falling_goods(self, container_prim, num_boxes=8):
        bb_cache = create_bbox_cache()
        spawn_height = (
            bb_cache.ComputeLocalBound(container_prim).GetRange().GetSize()[2] * 1.1
        )
        container_pos = omni.usd.get_world_transform_matrix(
            container_prim
        ).ExtractTranslation()

        box_files = self.config["container_boxs"]
        boxurl = self.assets_root_path + random.choice(box_files)

        for i in range(num_boxes):
            box_prim = prims.create_prim(
                prim_path=f"/World/SimulatedCardbox_{i}",
                position=container_pos
                + Gf.Vec3d(
                    random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), spawn_height
                ),
                orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
                usd_path=boxurl,
                semantic_label="goods",
            )

            spawn_height += (
                bb_cache.ComputeLocalBound(box_prim).GetRange().GetSize()[2] * 1.1
            )

            self._add_colliders(box_prim, approx_type="boundingCube")
            box_rigid_prim = SingleRigidPrim(prim_path=str(box_prim.GetPrimPath()))
            box_rigid_prim.enable_rigid_body_physics()

    def _create_shelfgoods(
        self,
        shelflocation,
        shelfGap,
        base_xform,
        oritation,
    ):
        box_files = self.config["goods"]
        boxurl = self.assets_root_path + random.choice(box_files)
        box_half_size = self._measure_half_extent(boxurl)

        if oritation == "vertical":
            gap_lenth = self.shelf_half_extent[1] * 2
            gap_width = self.shelf_half_extent[0] * 2 - 0.1 * 2
            shelf_min_x = shelflocation[0] - self.shelf_half_extent[1]
            shelf_min_y = shelflocation[1] - self.shelf_half_extent[0] + 0.1
        else:
            gap_lenth = self.shelf_half_extent[0] * 2 - 0.1 * 2
            gap_width = self.shelf_half_extent[1] * 2
            shelf_min_x = shelflocation[0] - self.shelf_half_extent[0] + 0.1
            shelf_min_y = shelflocation[1] - self.shelf_half_extent[1]
        gap_height = shelfGap

        shelf_min_z = shelflocation[2]

        goods_orination = np.pi / 4 if random.randint == 1 else 0

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
                        if (
                            w == 0
                            or w == (good_wid_nums - 1)
                            or l == 0
                            or l == (good_len_nums - 1)
                        ):
                            continueFlag = (
                                True
                                if random.choices([0, 1], weights=[4, 1])[0] == 1
                                else False
                            )
                            if continueFlag:
                                continue
                    locate_x = shelf_min_x + good_len_gap + (2 * l + 1) * box_len / 2
                    locate_y = shelf_min_y + good_wid_gap + (2 * w + 1) * box_wid / 2
                    locate_z = shelf_min_z + 2 * h * box_half_size[2]

                    shelf_goods_prims = prims.create_prim(
                        prim_path=f"{base_xform}/goods_{l}_{w}_{h}",
                        position=(
                            locate_x,
                            locate_y,
                            locate_z,
                        ),
                        orientation=euler_angles_to_quat([0, 0, goods_orination]),
                        usd_path=boxurl,
                        semantic_label="goods",
                    )

    def _add_colliders(self, root_prim, approx_type="convexHull"):
        # Iterate descendant prims (including root) and add colliders to mesh or primitive types
        for desc_prim in Usd.PrimRange(root_prim):
            if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
                # Physics
                if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
                else:
                    collision_api = UsdPhysics.CollisionAPI(desc_prim)
                collision_api.CreateCollisionEnabledAttr(True)
            # Add mesh specific collision properties only to mesh types
            if desc_prim.IsA(UsdGeom.Mesh):
                # Add mesh collision properties to the mesh (e.g. collider aproximation type)
                if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
                else:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
                mesh_collision_api.CreateApproximationAttr().Set(approx_type)

    def _get_prim_world_size(self, prim):
        bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])

        world_bbox = bbox_cache.ComputeWorldBound(prim)
        world_range = world_bbox.ComputeAlignedRange()

        min_pt = world_range.GetMin()
        max_pt = world_range.GetMax()
        return min_pt, max_pt

    def _check_collision(self, new_min, new_max):
        if (
            new_min[0] < -self.width / 2
            and new_max[0] > self.width / 2
            and new_min[1] < -self.length / 2
            and new_max[1] > self.length / 2
        ):
            return False

        for min_pt, max_pt in self.occupied_areas:
            if not (
                new_max[0] < min_pt[0]
                or new_min[0] > max_pt[0]
                or new_max[1] < min_pt[1]
                or new_min[1] > max_pt[1]
            ):
                return True
        return False

    def _measure_half_extent(self, url_path):
        test_prim = prims.create_prim(
            prim_path=f"/World/test",
            position=(0, 0, 0),
            orientation=euler_angles_to_quat([0, 0, 0]),
            usd_path=url_path,
            semantic_label="test",
        )

        min_pt, max_pt = self._get_prim_world_size(test_prim)
        half_extent = max_pt - min_pt

        prim_path = Sdf.Path(f"/World/test")
        omni.kit.commands.execute("DeletePrims", paths=[prim_path])

        return half_extent / 2

    def _measure_shelf_extent(self):
        usd_side_path = self.assets_root_path + self.config["shelf"]["sideshelf"]["url"]
        usd_shelf_path = (
            self.assets_root_path + self.config["shelf"]["railshelf"]["url"]
        )

        side_extent = self._measure_half_extent(usd_side_path)
        shelf_extent = self._measure_half_extent(usd_shelf_path)

        half_height = side_extent[2]
        half_lenth, half_width = shelf_extent[0], shelf_extent[1]

        shelf_half_extent = [half_lenth, half_width, half_height]

        return shelf_half_extent

    def _find_free_center(self, obj_width, obj_length, max_trials=1000):
        offset = 0
        if (self.env_id - 1) != 0:
            offset = self.offset - self.width
        for _ in range(max_trials):
            cx = random.uniform(
                -self.width / 2 + obj_width / 2 + offset,
                self.width / 2 - obj_width / 2 + offset,
            )
            cy = random.uniform(
                -self.length / 2 + obj_length / 2, self.length / 2 - obj_length / 2
            )
            min_pt = (cx - obj_width / 2, cy - obj_length / 2)
            max_pt = (cx + obj_width / 2, cy + obj_length / 2)

            if not self._check_collision(min_pt, max_pt):
                return (cx, cy)

        raise RuntimeError("无法在场景范围内找到空闲位置")

    def _create_omnipbr_material(self, mtl_url, mtl_name, mtl_path, texture_list):

        stage = omni.usd.get_context().get_stage()

        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=mtl_url,
            mtl_name=mtl_name,
            mtl_path=mtl_path,
        )

        material_prim = stage.GetPrimAtPath(mtl_path)

        shader = UsdShade.Shader(
            omni.usd.get_shader_from_material(material_prim, get_prim=True)
        )
        use_texture = random.choice([True, False])

        # Add value inputs
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float)
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float)
        if not use_texture:
            shader.GetInput("diffuse_color_constant").Set(
                Gf.Vec3f(random.random(), random.random(), random.random())
            )
            shader.GetInput("metallic_constant").Set(random.random())
            shader.GetInput("reflection_roughness_constant").Set(
                random.uniform(0.05, 0.9)
            )

        # Add texture inputs
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
        shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
        shader.CreateInput("metallic_texture", Sdf.ValueTypeNames.Asset)
        if use_texture:
            shader.GetInput("diffuse_texture").Set(random.choice(texture_list))
            shader.GetInput("reflectionroughness_texture").Set(
                random.choice(texture_list)
            )
            shader.GetInput("metallic_texture").Set(random.choice(texture_list))

        # Add other attributes
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)
        shader.GetInput("project_uvw").Set(random.choice([True, False]))

        # Add texture scale and rotate
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
        shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)
        shader.GetInput("texture_scale").Set(
            (random.uniform(0.1, 2.0), random.uniform(0.1, 2.0))
        )
        shader.GetInput("texture_rotate").Set(random.uniform(0, 360))

        material = UsdShade.Material(material_prim)

        return material

    def _create_materials(self, num):

        MDL = "OmniPBR.mdl"
        mtl_name, _ = os.path.splitext(MDL)
        MAT_PATH = "/World/Looks"
        materials = []

        texture_dir = self.assets_root_path + "/Isaac/Materials/Textures/Synthetic/"
        texture_list = []

        # 遍历目录，把所有 .png 文件加入列表
        for f in os.listdir(texture_dir):
            if f.lower().endswith(".png"):
                texture_path = os.path.join(texture_dir, f)
                texture_list.append(texture_path)

        for _ in range(num):

            prim_path = omni.usd.get_stage_next_free_path(
                stage, f"{MAT_PATH}/{mtl_name}", False
            )

            mat = self._create_omnipbr_material(
                mtl_url=MDL,
                mtl_name=mtl_name,
                mtl_path=prim_path,
                texture_list=texture_list,
            )

            materials.append(mat)

        return materials

    def create(
        self,
        forklift=True,
        containers=True,
        shelf=True,
        debris=True,
        light=True,
    ):
        prim_list = {}
        self.create_warehouse()

        if forklift:
            self.create_forklift()
            prim_list["forklift"] = self._forklift_prim
        if containers:
            self.create_containers()
            prim_list["containers"] = self._containers_prim
        if shelf:
            self.create_shelf()
            prim_list["shelf"] = self._shelf_prim
        if debris:
            self.create_debris()
            prim_list["debris"] = self._debris_prim
        if light:
            self.create_light()

        return Scence(prim_list)


class Scence(EnvConstructor):
    def __init__(self, prim_list):
        if "forklift" in prim_list:
            self._forklift_prim = prim_list["forklift"]
        if "containers" in prim_list:
            self._containers_prim = prim_list["containers"]
        if "shelf" in prim_list:
            self._shelf_prim = prim_list["shelf"]
        if "debris" in prim_list:
            self._debris_prim = prim_list["debris"]

    def get_forklift_prim(self):
        return self._forklift_prim

    def get_containers_prim(self):
        return self._containers_prim

    def get_shelf_prim(self):
        return self._shelf_prim

    def get_debris_prim(self):
        return self._debris_prim
