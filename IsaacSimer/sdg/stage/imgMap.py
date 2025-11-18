import numpy as np
import cv2
import math
import random
from typing import List, Tuple, Dict, Optional
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union    

class MaskManager:
    def __init__(self, physical_width: float, physical_height: float, grid_size: float = 0.1):
        """
        初始化掩码管理器
        
        :param physical_width: 物理区域宽度
        :param physical_height: 物理区域高度
        :param grid_size: 网格尺寸（默认0.1）
        """
        self.physical_width = physical_width
        self.physical_height = physical_height
        self.grid_size = grid_size
        
        # 计算网格单元数量
        self.grid_size_x = int(physical_width / grid_size)
        self.grid_size_y = int(physical_height / grid_size)
        
        # 存储所有掩码
        self.masks = []
        
        # 存储所有路径
        self.paths = []
        
        # 掩码ID计数器
        self._next_mask_id = 0
        
        # 创建网格可视化图像（白色背景）
        self.grid_image = np.ones((self.grid_size_y, self.grid_size_x, 3), dtype=np.uint8) * 255
        
        self.cell_mask_count = {} # 用于记录每个网格单元的mask数量
        
        self.generated_regions = []
        
        # 存储系统状态
        self.shelves = []
        self.column_groups = {}  # 直接在主类中管理列组信息
        
        self.path_polygons = []  # 存储路径形成的多边形区域
        self.region_bounds = None  # 存储区域边界

    def add_mask(self, center_x: float, center_y: float, rotation: float, 
                half_width: float, half_height: float, priority: float) -> bool:
        """
        添加一个新掩码
        
        :return: 如果添加成功返回True，否则返回False
        """
        # 将物理中心点转换为网格坐标
        center_grid_x, center_grid_y = self.physical_to_grid(center_x, center_y)
        
        # 将半宽半高转换为网格单位
        half_width_grid = half_width / self.grid_size
        half_height_grid = half_height / self.grid_size
        
        # 计算旋转后的矩形四个顶点（在网格坐标系中）
        angle_rad = np.radians(rotation)
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # 矩形的原始顶点（相对于中心）
        points = np.array([
            [-half_width_grid, -half_height_grid],
            [half_width_grid, -half_height_grid],
            [half_width_grid, half_height_grid],
            [-half_width_grid, half_height_grid]
        ])
        
        # 旋转顶点
        rotated_points = []
        for point in points:
            x_new = point[0] * cos_val - point[1] * sin_val
            y_new = point[0] * sin_val + point[1] * cos_val
            rotated_points.append([x_new + center_grid_x, y_new + center_grid_y])
        
        new_mask = {
            'id': self._next_mask_id,  # 添加唯一ID
            'points': np.array(rotated_points, dtype=np.int32),
            'priority': priority,
            'center_grid': (center_grid_x, center_grid_y),
            'center_physical': (center_x, center_y),
            'half_width': half_width,
            'half_height': half_height,
            'rotation': rotation
        }
        self._next_mask_id += 1
        
        # 检查新掩码是否与现有掩码相交（根据优先级规则）
        overlap_result = self._check_overlap(new_mask)
        if overlap_result['overlap']:
            # 如果是0.7与0.7重叠，进行特殊处理
            if (new_mask['priority'] == 0.7 and 
                overlap_result['existing_mask'] is not None and
                ((overlap_result['existing_mask']['priority'] * 10 ) % 7 ==  0)):
                return self._handle_07_overlap(new_mask, overlap_result['existing_mask'])
            if (new_mask['priority'] == 1.1 and overlap_result['existing_mask'] is not None and (overlap_result["existing_mask"]["priority"] == 1.1 or overlap_result["existing_mask"]["priority"] == 1.0)):
                self.masks.append(new_mask)
                self._draw_mask(new_mask)
                return True
            if ((overlap_result["existing_mask"]["priority"] == 1.1)):
                self.masks.append(new_mask)
                self._draw_mask(new_mask)
                return True
            else:
                return False
        
        # 如果不相交，添加新掩码并更新可视化
        self.masks.append(new_mask)
        self._draw_mask(new_mask)
        return True

    ##### 单个货物
    def physical_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return int(round(x / self.grid_size)), int(round(y / self.grid_size))

    def _rotate_rect_points(self, cx: float, cy: float, half_w: float, half_h: float, rotation: float):
        angle_rad = math.radians(rotation)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        points = [
            (-half_w, -half_h),
            ( half_w, -half_h),
            ( half_w,  half_h),
            (-half_w,  half_h)
        ]
        rotated = []
        for x, y in points:
            xr = x * cos_val - y * sin_val + cx
            yr = x * sin_val + y * cos_val + cy
            rotated.append((xr, yr))
        return rotated

    def _is_mask_within_bounds_(self, center_x: float, center_y: float,
                               half_width: float, half_height: float, rotation: float) -> bool:
        rotated_points = self._rotate_rect_points(center_x, center_y, half_width, half_height, rotation)
        rotated_points_grid = [self.physical_to_grid(x, y) for x, y in rotated_points]
        xs = [p[0] for p in rotated_points_grid]
        ys = [p[1] for p in rotated_points_grid]
        return (min(xs) >= 0 and max(xs) <= self.grid_size_x and
                min(ys) >= 0 and max(ys) <= self.grid_size_y)

    def _polygons_intersect(self, poly1, poly2) -> bool:
        def project(polygon, axis):
            dots = [p[0]*axis[0] + p[1]*axis[1] for p in polygon]
            return min(dots), max(dots)

        def overlap(proj1, proj2):
            return proj1[0] <= proj2[1] and proj2[0] <= proj1[1]

        polygons = [poly1, poly2]
        for polygon in polygons:
            for i in range(len(polygon)):
                p1 = polygon[i]
                p2 = polygon[(i+1) % len(polygon)]
                edge = (p2[0] - p1[0], p2[1] - p1[1])
                axis = (-edge[1], edge[0])
                proj1 = project(poly1, axis)
                proj2 = project(poly2, axis)
                if not overlap(proj1, proj2):
                    return False
        return True

    def _check_overlap_(self, new_mask):
        for mask in self.masks:
            if self._polygons_intersect(new_mask['points'], mask['points']):
                return True
        return False

    def _check_path_overlap(self, new_mask) -> bool:
        mask_points = new_mask['points']
        for path in self.paths:
            path_points = [(self.physical_to_grid(x, y)[0], self.physical_to_grid(x, y)[1]) for x, y in path['points']]
            # 将路径每两个点形成的矩形看作线宽矩形
            path_width = path.get('width', 1)
            for i in range(len(path_points) - 1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = max(math.hypot(dx, dy), 0.001)
                dir_x = dx / length
                dir_y = dy / length
                norm_x = -dir_y
                norm_y = dir_x
                half_width = path_width / 2
                rect = [
                    (p1[0] - norm_x*half_width, p1[1] - norm_y*half_width),
                    (p1[0] + norm_x*half_width, p1[1] + norm_y*half_width),
                    (p2[0] + norm_x*half_width, p2[1] + norm_y*half_width),
                    (p2[0] - norm_x*half_width, p2[1] - norm_y*half_width)
                ]
                if self._polygons_intersect(mask_points, rect):
                    # 优先级判断
                    if path.get('priority', 1) >= new_mask['priority']:
                        return True
        return False

    def add_multiple_masks(self, half_width: float, half_height: float, priority: float, count: int, max_attempts: int = 200):
        added_masks = []
        for _ in range(count):
            success = False
            for attempt in range(max_attempts):
                center_x = random.uniform(half_width, self.physical_width - half_width)
                center_y = random.uniform(half_height, self.physical_height - half_height)
                rotation = random.uniform(0, 360)

                rotated_points = self._rotate_rect_points(center_x, center_y, half_width, half_height, rotation)
                rotated_points_grid = [self.physical_to_grid(x, y) for x, y in rotated_points]

                new_mask = {
                    'id': self._next_mask_id,
                    'points': np.array(rotated_points_grid, dtype=np.int32),
                    'priority': priority,
                    'center_grid': self.physical_to_grid(center_x, center_y),
                    'center_physical': (center_x, center_y),
                    'half_width': half_width,
                    'half_height': half_height,
                    'rotation': rotation
                }

                if not self._is_mask_within_bounds_(center_x, center_y, half_width, half_height, rotation):
                    continue
                if self._check_overlap_(new_mask):
                    continue
                if self._check_path_overlap(new_mask):
                    continue

                self.masks.append(new_mask)
                self._draw_mask(new_mask)
                self._next_mask_id += 1
                added_masks.append(new_mask)
                success = True
                break
            if not success:
                print(f"无法放置 mask {_+1}，尝试了 {max_attempts} 次仍失败")
        return added_masks
    #### 上述为单个货物代码
    
    #####基于初始点和终点的路径随机化
    def generate_constrained_path_v2(self,
                                     start_point: Tuple[float, float], 
                                     end_point: Tuple[float, float],
                                     shelf_zone: Optional[Dict] = None,
                                     path_half_width: float = 1.5,
                                     shelf_half_length: float = 2,
                                     shelf_half_depth: float = 0.5,
                                     first_direction: str = "horizontal",
                                     priority: float = 1,
                                     gap: float = 0.2,
                                     max_points: int = 20,
                                     max_attempts: int = 100) -> Optional[List[Tuple[float, float]]]:
        """
        生成满足约束的随机正交路径
        
        参数:
            start_point: 起始点 (x, y)
            end_point: 结束点 (x, y)
            shelf_zone: 货架区域定义，格式 {'min': (x, y), 'max': (x, y)}
            path_half_width: 路径半宽度
            shelf_half_length: 货架半长度（第一方向最小长度）
            shelf_half_depth: 货架半深度（第二方向最小长度）
            first_direction: 初始方向 'horizontal' 或 'vertical'
            priority: 优先级系数（未使用）
            gap: 平行线段最小间隙
            max_points: 最大点数限制
            max_attempts: 最大尝试次数
        
        返回:
            路径点列表，如果失败返回 None
        """
        # 计算内缩区域（考虑路径宽度）
        if shelf_zone:
            min_x, min_y = shelf_zone['min']
            max_x, max_y = shelf_zone['max']
            inner_min_x = start_point[0]
            inner_max_x = end_point[0]
            inner_min_y = start_point[1]
            inner_max_y = end_point[1]
        else:
            inner_min_x = inner_min_y = -10**9
            inner_max_x = inner_max_y = 10**9
        
        for _ in range(max_attempts):
            points = [start_point]
            segments_h = []  # 水平线段 (y, x_min, x_max)
            segments_v = []  # 垂直线段 (x, y_min, y_max)
            current_dir = first_direction
            current = start_point
            segment_count = 0  # 已生成的线段数
            
            while len(points) < max_points:
                # 尝试直接连接到终点
                if self.try_connect_to_end(current, end_point, current_dir, segment_count,
                                          shelf_half_length, shelf_half_depth,
                                          inner_min_x, inner_max_x, inner_min_y, inner_max_y,
                                          segments_h, segments_v, gap):
                    points.append(end_point)
                    self.add_path_to_map(points, path_width=1.0, priority=1.0)
                    return points
                
                # 生成下一步
                next_dir = "vertical" if current_dir=="horizontal" else "horizontal"
                min_length = (shelf_half_length*2 + path_half_width) if segment_count % 2 == 0 else (shelf_half_depth + path_half_width) * 2
                
                found = False
                if current_dir == "horizontal":
                    # 水平移动
                    d_min = min_length
                    d_max = inner_max_x - current[0]
                    
                    if d_max < d_min:
                        d = d_max
                    else:
                        t_min = (shelf_half_depth + path_half_width) * 2 if segment_count % 2 == 0 else (shelf_half_length*2 + path_half_width)
                        t_max = inner_max_y - current[1]
                        
                        if t_max <= t_min:
                            d = d_max
                        else:
                            d = round(random.uniform(d_min, d_max),1)
                            if d_max -d < d_min:
                                d = d_max
                    new_point = (current[0] + d, current[1])
                else:
                    # 垂直移动
                    d_min = min_length
                    d_max = inner_max_y - current[1]
                    
                    if d_max < d_min:
                        d = d_max
                    else:
                        t_min = (shelf_half_depth + path_half_width) * 2 if segment_count % 2 == 0 else (shelf_half_length*2 + path_half_width)
                        t_max = inner_max_x - current[0]
                        
                        if t_max <= t_min:
                            d = d_max
                        else:
                            d = round(random.uniform(d_min, d_max), 1)
                            if d_max -d < d_min:
                                d = d_max
                    new_point = (current[0], current[1] + d)
                
                # # 检查新点是否在内缩区域内
                # if not (inner_min_x <= new_point[0] <= inner_max_x and 
                #         inner_min_y <= new_point[1] <= inner_max_y):
                #     continue
                
                # 创建新线段并检查间隙
                if current_dir == "horizontal":
                    x_min = min(current[0], new_point[0])
                    x_max = max(current[0], new_point[0])
                    new_segment = (current[1], x_min, x_max)
                    if not self.check_segment_gap(segments_h, new_segment, gap, False):
                        continue
                else:
                    y_min = min(current[1], new_point[1])
                    y_max = max(current[1], new_point[1])
                    new_segment = (current[0], y_min, y_max)
                    if not self.check_segment_gap(segments_v, new_segment, gap, True):
                        continue
                
                # 添加新点和线段
                points.append(new_point)
                if current_dir == "horizontal":
                    segments_h.append(new_segment)
                else:
                    segments_v.append(new_segment)
                
                current = new_point
                current_dir = next_dir
                segment_count += 1
            
            # 检查是否可以在最后一点连接到终点
            if len(points) < max_points:
                if self.try_connect_to_end(current, end_point, current_dir, segment_count,
                                          shelf_half_length, shelf_half_depth,
                                          inner_min_x, inner_max_x, inner_min_y, inner_max_y,
                                          segments_h, segments_v, gap):
                    points.append(end_point)
                    self.add_path_to_map(points, path_width=1.0, priority=1.0)
                    return points
        
        return None

    def try_connect_to_end(self,
                          current: Tuple[float, float],
                          end_point: Tuple[float, float],
                          current_dir: str,
                          segment_count: int,
                          shelf_half_length: float,
                          shelf_half_depth: float,
                          inner_min_x: float,
                          inner_max_x: float,
                          inner_min_y: float,
                          inner_max_y: float,
                          segments_h: list,
                          segments_v: list,
                          gap: float) -> bool:
        """尝试从当前点直接连接到终点"""
        # 检查方向和对齐
        if current_dir == "horizontal" and current[1] == end_point[1]:
            dist = abs(end_point[0] - current[0])
            # min_len = shelf_half_length if segment_count % 2 == 0 else shelf_half_depth
            min_len = 0.1
            if dist < min_len:
                return False
        elif current_dir == "vertical" and current[0] == end_point[0]:
            dist = abs(end_point[1] - current[1])
            # min_len = shelf_half_length if segment_count % 2 == 0 else shelf_half_depth
            min_len = 0.1
            if dist < min_len:
                return False
        else:
            return False
        
        # 检查终点在区域内
        if not (inner_min_x <= end_point[0] <= inner_max_x and 
                inner_min_y <= end_point[1] <= inner_max_y):
            print(" test ---- 004")
            return False
        
        # 创建线段并检查间隙
        if current_dir == "horizontal":
            x_min = min(current[0], end_point[0])
            x_max = max(current[0], end_point[0])
            segment = (current[1], x_min, x_max)
            if not self.check_segment_gap(segments_h, segment, gap, False):
                print(" test ---- 005")
                return False
        else:
            y_min = min(current[1], end_point[1])
            y_max = max(current[1], end_point[1])
            segment = (current[0], y_min, y_max)
            if not self.check_segment_gap(segments_v, segment, gap, True):
                print(" test ---- 006")
                return False
        
        return True

    def check_segment_gap(self,
                         segments: list,
                         new_segment: tuple,
                         gap: float,
                         is_vertical: bool) -> bool:
        """检查新线段与同方向线段的间隙"""
        for seg in segments:
            if is_vertical:
                # 垂直线段: (x, y_min, y_max)
                dist = abs(seg[0] - new_segment[0])
                if dist < gap:
                    # 检查y轴重叠
                    y_overlap = (seg[1] < new_segment[2] and 
                                new_segment[1] < seg[2])
                    if y_overlap:
                        return False
            else:
                # 水平线段: (y, x_min, x_max)
                dist = abs(seg[0] - new_segment[0])
                if dist < gap:
                    # 检查x轴重叠
                    x_overlap = (seg[1] < new_segment[2] and 
                                new_segment[1] < seg[2])
                    if x_overlap:
                        return False
        return True
    #####上述为路径代码

    ####基于路径的货架随机布置      
    # # ==================== 货架添加到地图 ====================
    
    def _add_shelf_to_map(self, shelf: Dict, priority: float = 1.0) -> bool:
        """
        将货架添加到地图掩码系统中
        """
        try:
            center_x, center_y = shelf['center']

            # 将 orientation 转成角度
            orientation_str = shelf['orientation']
            if isinstance(orientation_str, str):
                if orientation_str.lower() == "horizontal":
                    orientation = 0.0
                elif orientation_str.lower() == "vertical":
                    orientation = 90.0
                else:
                    orientation = 0.0
            else:
                orientation = float(orientation_str)

            half_length = shelf['length'] / 2
            half_depth = shelf['depth'] / 2

            success = self.add_mask(center_x, center_y, orientation,
                                    half_length, half_depth, priority)

            if success and self.masks:
                # 标记为货架类型
                self.masks[-1]['type'] = 'shelf'
                self.masks[-1]['shelf_info'] = shelf
                print(f"货架添加到地图: 位置({center_x:.2f}, {center_y:.2f}), 优先级{priority}")

            return success

        except Exception as e:
            print(f"错误: 添加货架到地图失败 - {e}")
            return False
    
    def _update_masks_with_merged_shelves(self, merged_shelves: List[Dict]):
        """
        用合并后的货架更新masks
        """
        # 移除所有原有货架
        self.masks = [mask for mask in self.masks if mask.get('type') != 'shelf']
        
        # 添加合并后的货架
        for shelf in merged_shelves:
            self._add_shelf_to_map(shelf)

    def add_path_from_route_in_region(
        self,
        route_points: List[Tuple[float, float]],
        route_width: float,
        region_min: Tuple[float, float],
        region_max: Tuple[float, float],
        priority: float = 1.1
    ) -> List[Dict]:
        """
        根据路线点生成路径掩码，并添加到 self.masks。
        路线为水平或垂直折线，沿每段两端延伸直到与货架区域边界相交。
        返回：空白区域列表（每个包含 min/max 坐标）
        """

        x_min, y_min = region_min
        x_max, y_max = region_max

        def segment_orientation(p1, p2):
            """判断路线段方向（水平或垂直）"""
            if abs(p1[1] - p2[1]) < 1e-5:
                return 'horizontal'
            elif abs(p1[0] - p2[0]) < 1e-5:
                return 'vertical'
            else:
                print(" == " * 100)
                print(f"p1: {p1}, p2: {p2}")
                print(" == " * 100)
                raise ValueError("路线段不是水平或垂直方向")

        def extend_to_region(p1, p2, direction):
            """沿线段两端延伸直到货架区域边界"""
            x1, y1 = p1
            x2, y2 = p2
            if direction == 'horizontal':
                extended_p1 = (x_min, y1)
                extended_p2 = (x_max, y2)
            else:
                extended_p1 = (x1, y_min)
                extended_p2 = (x2, y_max)
            return extended_p1, extended_p2

        path_polygons = []

        # 遍历每一段路线
        for i in range(len(route_points) - 1):
            p1 = route_points[i]
            p2 = route_points[i + 1]
            direction = segment_orientation(p1, p2)

            # 计算延伸后的两端点
            p1_ext, p2_ext = extend_to_region(p1, p2, direction)

            # 计算路径中心与尺寸
            center_x = (p1_ext[0] + p2_ext[0]) / 2
            center_y = (p1_ext[1] + p2_ext[1]) / 2
            length = np.hypot(p2_ext[0] - p1_ext[0], p2_ext[1] - p1_ext[1])
            # length = np.max(p2_ext[0] - p1_ext[0], p2_ext[1] - p1_ext[1])

            # 路径尺寸
            half_width = route_width / 2
            half_height = length / 2
            rotation = 0 if direction == 'horizontal' else 90

            # 添加路径mask
            success = self.add_mask(
                center_x=center_x,
                center_y=center_y,
                rotation=rotation,
                half_width=half_height,   # 长边沿路径方向
                half_height=half_width,   # 短边为路径宽度
                priority=priority
            )

            # 同步路径的几何信息
            if success:
                if direction == 'horizontal':
                    rect = box(
                        center_x - half_height,
                        center_y - half_width,
                        center_x + half_height,
                        center_y + half_width
                    )
                else:
                    rect = box(
                        center_x - half_width,
                        center_y - half_height,
                        center_x + half_width,
                        center_y + half_height
                    )
                path_polygons.append(rect)

        # ========== 计算空白区域 ==========
        region_poly = box(x_min, y_min, x_max, y_max)
        if path_polygons:
            path_union = unary_union(path_polygons)
            free_area = region_poly.difference(path_union)
        else:
            free_area = region_poly  # 没有路径则全区域为空白

        free_regions = []
        if free_area.is_empty:
            print("⚠️ 无空白区域（路径覆盖全部）")
        else:
            # free_area 可能是 MultiPolygon 或 Polygon
            polys = [free_area] if isinstance(free_area, Polygon) else list(free_area.geoms)
            for poly in polys:
                minx, miny, maxx, maxy = poly.bounds
                free_regions.append({
                    'min': (minx+0.2, miny+0.2),
                    'max': (maxx-0.2, maxy-0.2),
                    'area': poly.area
                })

        print(f"✅ 生成路径掩码完成，共 {len(free_regions)} 个空白区域")
        
        return free_regions
     
    def place_shelves_in_empty_regions(
        self,
        empty_regions: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        shelf_length_min: float,
        shelf_width: float,
        shelf_gap: float,
        orientation: str = "horizontal",
        safety_margin: float = 0.2,
        priority: float = 1.0
    ) -> List[Dict]:
        """
        在空白区域中放置货架，所有货架方向一致（横放或竖放）。
        货架之间留有指定间隙，只能放在单个空白区域内。
        """

        shelves = []
        for region in empty_regions:
            
            x_min, y_min = region["min"]
            x_max, y_max = region["max"]

            region_length_x = x_max - x_min
            region_length_y = y_max - y_min

            if orientation == "horizontal":
                # 每层沿 y 方向放置
                current_y = y_min + shelf_width / 2.0 + safety_margin / 2
                while current_y + shelf_width / 2.0 <= y_max - safety_margin / 2:
                    # 横向货架带：从左到右，可能只放一条
                    usable_length = region_length_x - safety_margin*2
                    if usable_length < shelf_length_min:
                        break
                    
                    add_gap = usable_length % shelf_length_min
                    
                    if add_gap == 0:
                        usable_length = usable_length
                    else:
                        usable_length = usable_length - (usable_length % shelf_length_min)

                    center_x = (x_min + x_max) / 2.0
                    center_y = current_y
                    half_width = usable_length / 2.0
                    half_height = shelf_width / 2.0
                    rotation = 0.0

                    # 添加到掩码系统
                    if self.add_mask(
                        center_x=center_x,
                        center_y=center_y,
                        rotation=rotation,
                        half_width=half_width,
                        half_height=half_height,
                        priority=priority
                    ):
                        shelves.append({
                            "center": (center_x, center_y),
                            "length": usable_length,
                            "depth": shelf_width,
                            "orientation":orientation,
                            "half_width": half_width,
                            "half_height": half_height,
                            "rotation": rotation
                        })

                        current_y += shelf_width + shelf_gap
                    else:
                        current_y += 0.1

            elif orientation == "vertical":
                # 每列沿 x 方向放置
                current_x = x_min + shelf_width / 2.0 + safety_margin / 2
                while current_x + shelf_width / 2.0 <= x_max - safety_margin / 2:
                    usable_length = region_length_y - safety_margin
                    if usable_length < shelf_length_min:
                        break
                    
                    add_gap = usable_length % shelf_length_min
                    
                    if add_gap == 0:
                        usable_length = usable_length
                    else:
                        usable_length = usable_length - (usable_length % shelf_length_min)

                    center_x = current_x
                    center_y = (y_min + y_max) / 2.0
                    half_width = usable_length / 2.0
                    half_height = shelf_width / 2.0
                    rotation = 90.0

                    if self.add_mask(
                        center_x=center_x,
                        center_y=center_y,
                        rotation=rotation,
                        half_width=half_width,
                        half_height=half_height,
                        priority=priority
                    ):
                        shelves.append({
                            "center": (center_x, center_y),
                            "length": usable_length,
                            "depth": shelf_width,
                            "orientation":orientation,
                            "half_width": half_width,
                            "half_height": half_height,
                            "rotation": rotation
                        })

                        current_x += shelf_width + shelf_gap
                    else:
                        current_x += 0.1

            else:
                raise ValueError("orientation must be 'horizontal' or 'vertical'")
        

        return shelves

    def generate_shelves_in_region(
        self,
        path_points: List[Tuple[float, float]],
        path_half_width: float,
        region_min: Tuple[float, float],
        region_max: Tuple[float, float],
        shelf_length: float,
        shelf_depth: float,
        safety_margin: float = 0.2,
        orientation: str = "vertical"
    ) -> List[Dict]:
        """
        在区域内生成货架，确保不与路径相交
        路径会延伸到区域边界，根据路径交点合并货架
        
        参数:
            path_points: 路径点列表
            path_half_width: 路径半宽
            region_min: 区域最小坐标 (min_x, min_y)
            region_max: 区域最大坐标 (max_x, max_y)
            shelf_length: 货架长度
            shelf_depth: 货架深度
            safety_margin: 安全间距
            orientation: 货架方向 ("vertical" 或 "horizontal")
        
        返回:
            生成的货架列表，每个货架包含 center, length, depth, orientation 等信息
        """
        
        x_min, y_min = region_min
        x_max, y_max = region_max
             
        free_regions = self.add_path_from_route_in_region(
            route_points=path_points,
            route_width=path_half_width * 2,
            region_min=region_min,
            region_max=region_max,
            priority=1.1
        )
        print(free_regions)
        
        orientation = "horizontal" if random.random() > 0.5 else "vertical"
        
        shelves = self.place_shelves_in_empty_regions(
            empty_regions=free_regions,
            shelf_length_min=shelf_length,
            shelf_width=shelf_depth,
            shelf_gap=path_half_width*2,
            orientation=orientation,
            priority=1.0,
            safety_margin=safety_margin
        )
        
        # 创建区域信息（用于记录和可视化）
        region_info = {
            'center': ((x_max + x_min) / 2, (y_max + y_min) / 2),
            'half_width': (x_max - x_min) / 2,
            'half_height': (y_max - y_min) / 2,
            'rows': 1,
            'cols': 1,
            'gap_x': 0.2,
            'gap_y': 0.2,
            'priority': 1.0,
            'filled': True,
            'placement_attempts': 1,
            'path_half_width': path_half_width,
            'safety_margin': 0.2
        }
        
        self.generated_regions.append(region_info)
        
        return shelves
    ####UP货架代码区域
    
    
    

    def _check_overlap(self, new_mask: Dict) -> Dict:
        """
        检查新掩码是否与现有掩码相交（考虑更新后的优先级规则）
        
        更新后的规则:
        - 优先级 0.5: 可被任何优先级覆盖
        - 优先级 0.7: 只能被其他0.7优先级覆盖
        - 优先级 1.0: 不能被任何优先级覆盖
        - 优先级 1.1：只能相互覆盖覆盖后优先级不变
        - 优先级 1.4: 与0.7优先级规则一致（只能被其他0.7优先级覆盖）
        """
        new_priority = new_mask['priority']
        new_poly = new_mask['points']
        
        for existing_mask in self.masks:
            exist_priority = existing_mask['priority']
            exist_poly = existing_mask['points']
            
            # 检查几何重叠
            if self._polygons_overlap(new_poly, exist_poly):
                # 应用更新后的优先级规则
                if exist_priority == 0.5:
                    return {'overlap': False, 'existing_mask': None}
                elif (exist_priority * 10) % 7 == 0:
                    return {'overlap': True, 'existing_mask': existing_mask}
                elif exist_priority == 1.0:
                    # 1.0不能与任何优先级的掩码重叠
                    return {'overlap': True, 'existing_mask': existing_mask}
                elif exist_priority == 1.1:
                    return {'overlap': True, 'existing_mask': existing_mask}
                
                # if new_priority == 0.5:
                #     # 0.5可以被任何掩码覆盖
                #     return {'overlap': True, 'existing_mask': existing_mask}
                # elif new_priority == 0.7:
                #     # 0.7只能被其他0.7优先级覆盖
                #     if (exist_priority * 10) % 7 == 0:
                #         return {'overlap': True, 'existing_mask': existing_mask}
                # elif new_priority == 1.0:
                #     # 1.0不能与任何优先级的掩码重叠
                #     return {'overlap': True, 'existing_mask': existing_mask}
                # elif new_priority == 1.1:
                #     # 1.1只能相互重叠
                #     return {'overlap': True, 'existing_mask': existing_mask}
                # elif new_priority == 1.4:
                #     # 1.4只能被其他0.7优先级覆盖（与0.7规则一致）
                #     if exist_priority == 0.7 or exist_priority == 1.4:
                #         return {'overlap': True, 'existing_mask': existing_mask}
                # # 检查现有掩码是否会阻止新掩码
                # elif exist_priority == 1.0:
                #     # 现有掩码是1.0，不能被任何新掩码覆盖
                #     return {'overlap': True, 'existing_mask': existing_mask}
                # elif  (exist_priority * 10) % 7 == 0 and new_priority != 0.7:
                #     # 现有掩码是0.7，只能被0.7或1.4覆盖，其他优先级不能覆盖它
                #     return {'overlap': True, 'existing_mask': existing_mask}
        
        return {'overlap': False, 'existing_mask': None}

    def _handle_07_overlap(self, new_mask: Dict, existing_mask: Dict) -> bool:
        """
        处理两个0.7优先级掩码重叠的特殊情况
        
        先调整后者的中心点和朝向与被覆盖区域一致，
        然后计算这两个调整后区域的并集，
        并将合并后的区域优先级设置为1.4
        """
        # 先调整新掩码的位姿（中心点和旋转角度）与现有掩码一致
        adjusted_mask = self._adjust_mask_to_existing(new_mask, existing_mask)
        
        # 计算调整后的掩码与现有掩码的并集[6,7](@ref)
        union_poly = self._compute_union(adjusted_mask['points'], existing_mask['points'])
        
        if union_poly is None or len(union_poly) == 0:
            return False
        
        # 计算并集的最小外接矩形
        rect = cv2.minAreaRect(union_poly)
        center, size, angle = rect
        
        # 设置合并后的参数
        new_center_x, new_center_y = center
        new_rotation = angle
        new_half_width = size[0] * self.grid_size / 2
        new_half_height = size[1] * self.grid_size / 2
        new_priority = existing_mask['priority'] + 0.7  # 设置新的优先级
        
        # 将物理坐标转换回网格坐标
        new_center_physical_x, new_center_physical_y = self.grid_to_physical(new_center_x, new_center_y)
        
        # 创建新的合并掩码
        merged_mask = {
            'id': self._next_mask_id,  # 为新掩码分配新ID
            'points': union_poly,
            'priority': new_priority,
            'center_grid': (int(new_center_x), int(new_center_y)),
            'center_physical': (new_center_physical_x, new_center_physical_y),
            'half_width': new_half_width,
            'half_height': new_half_height,
            'rotation': new_rotation
        }
        self._next_mask_id += 1
        
        # 移除原有的两个掩码（通过ID比较而不是直接比较字典）
        self.masks = [mask for mask in self.masks if mask['id'] not in [existing_mask['id'], new_mask['id']]]
        
        # 添加新的合并掩码
        self.masks.append(merged_mask)
        
        # 重新绘制网格
        self.grid_image = np.ones((self.grid_size_y, self.grid_size_x, 3), dtype=np.uint8) * 255
        for mask in self.masks:
            self._draw_mask(mask)
        
        return True

    def _adjust_mask_to_existing(self, new_mask: Dict, existing_mask: Dict) -> Dict:
        """
        调整新掩码的位姿（中心点和旋转角度）与现有掩码一致
        """
        # 创建调整后的掩码副本
        adjusted_mask = new_mask.copy()
        
        # 调整中心点与现有掩码一致
        adjusted_mask['center_physical'] = existing_mask['center_physical']
        adjusted_mask['center_grid'] = existing_mask['center_grid']
        
        # 调整旋转角度与现有掩码一致
        adjusted_mask['rotation'] = existing_mask['rotation']
        
        # 重新计算旋转后的顶点
        center_grid_x, center_grid_y = existing_mask['center_grid']
        half_width_grid = new_mask['half_width'] / self.grid_size
        half_height_grid = new_mask['half_height'] / self.grid_size
        
        angle_rad = np.radians(existing_mask['rotation'])
        cos_val = np.cos(angle_rad)
        sin_val = np.sin(angle_rad)
        
        # 矩形的原始顶点（相对于中心）
        points = np.array([
            [-half_width_grid, -half_height_grid],
            [half_width_grid, -half_height_grid],
            [half_width_grid, half_height_grid],
            [-half_width_grid, half_height_grid]
        ])
        
        # 旋转顶点
        rotated_points = []
        for point in points:
            x_new = point[0] * cos_val - point[1] * sin_val
            y_new = point[0] * sin_val + point[1] * cos_val
            rotated_points.append([x_new + center_grid_x, y_new + center_grid_y])
        
        adjusted_mask['points'] = np.array(rotated_points, dtype=np.int32)
        
        return adjusted_mask

    def _compute_union(self, poly1: np.ndarray, poly2: np.ndarray) -> Optional[np.ndarray]:
        """
        计算两个多边形的并集[6,7](@ref)
        """
        try:
            # 将多边形转换为轮廓格式
            poly1_contour = poly1.reshape((-1, 1, 2)).astype(np.int32)
            poly2_contour = poly2.reshape((-1, 1, 2)).astype(np.int32)
            
            # 创建临时图像用于计算并集
            temp_image = np.zeros((self.grid_size_y, self.grid_size_x), dtype=np.uint8)
            
            # 绘制多边形[2](@ref)
            cv2.fillPoly(temp_image, [poly1_contour], 255)
            cv2.fillPoly(temp_image, [poly2_contour], 255)
            
            # 查找并集的轮廓
            contours, _ = cv2.findContours(temp_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 返回最大的轮廓（并集）
                largest_contour = max(contours, key=cv2.contourArea)
                return largest_contour.reshape(-1, 2)
            
            return None
        except Exception as e:
            print(f"计算并集时出错: {e}")
            return None

    def _polygons_overlap(self, poly1: np.ndarray, poly2: np.ndarray) -> bool:
        """
        使用分离轴定理判断两个凸多边形是否相交
        """
        if len(poly1) == 0 or len(poly2) == 0:
            return False
        
        # 检查所有多边形的所有边
        polygons = [poly1, poly2]
        for i, poly in enumerate(polygons):
            for j in range(len(poly)):
                # 获取边的法向量
                p1 = poly[j]
                p2 = poly[(j + 1) % len(poly)]
                edge = [p2[0] - p1[0], p2[1] - p1[1]]
                normal = [-edge[1], edge[0]]  # 法向量
                
                # 投影两个多边形到法向量
                min1, max1 = self._project_polygon(poly1, normal)
                min2, max2 = self._project_polygon(poly2, normal)
                
                # 检查投影是否重叠
                if max1 < min2 or max2 < min1:
                    return False
        return True
    

    def _check_min_spacing(self, center_x: float, center_y: float, 
                        half_width: float, half_height: float, 
                        min_spacing: float) -> bool:
        """
        检查新掩码与所有现有掩码之间的最小间距
        
        :param center_x: 新掩码中心的x坐标
        :param center_y: 新掩码中心的y坐标
        :param half_width: 新掩码的半宽
        :param half_height: 新掩码的半高
        :param min_spacing: 要求的最小间距
        :return: 如果满足最小间距要求返回True，否则返回False
        """
        for mask in self.masks:
            # 计算两个掩码中心点之间的距离[6,8](@ref)
            exist_center_x, exist_center_y = mask['center_physical']
            distance = math.sqrt((center_x - exist_center_x)**2 + (center_y - exist_center_y)**2)
            
            # 计算两个掩码的最小可能距离（考虑它们的尺寸）
            min_possible_distance = (half_width + mask['half_width'] + 
                                half_height + mask['half_height']) / 2
            
            # 如果实际距离小于要求的最小间距，返回False
            if distance - min_possible_distance < min_spacing:
                return False
        
        return True
    
    def _is_mask_within_bounds(self, center_x: float, center_y: float,
        half_width: float, half_height: float, rotation: float) -> bool:
        """
        使用旋转矩形的四个顶点来检测是否完全在物理边界内
        :param center_x: mask中心的x坐标
        :param center_y: mask中心的y坐标
        :param half_width: mask的半宽
        :param half_height: mask的半高
        :param rotation: mask的旋转角度
        :return: 如果mask完全在边界内返回True，否则返回False
        """
        pts = self._rotate_rect_points(center_x, center_y, half_width, half_height, rotation)
        for (x, y) in pts:
            if x < 0 or x > self.physical_width or y < 0 or y > self.physical_height:
                return False
        return True

    def generate_orthogonal_path(self, start_point: Tuple[float, float], end_point: Tuple[float, float], 
                               path_width: float = 1.5, num_turns: int = 1, 
                               min_segment_length: float = 5.0) -> List[Tuple[float, float]]:
        """
        生成连贯的直角随机路径（曼哈顿路径）
        """
        if num_turns < 0:
            raise ValueError("转折次数不能为负")
        
        path_points = [start_point]
        current_point = start_point
        
        # 计算总体方向
        total_dx = end_point[0] - start_point[0]
        total_dy = end_point[1] - start_point[1]
        
        # 随机决定第一步的方向
        move_horizontal = random.choice([True, False])
        
        # 生成转折
        for turn in range(num_turns):
            # 计算到终点的剩余距离
            remaining_dx = end_point[0] - current_point[0]
            remaining_dy = end_point[1] - current_point[1]
            
            # 如果已经很接近终点，提前结束
            if abs(remaining_dx) < min_segment_length and abs(remaining_dy) < min_segment_length:
                break
            
            # 确定移动方向和距离
            if move_horizontal:
                # 水平移动
                if abs(remaining_dx) > min_segment_length:
                    # 随机移动距离，但不超过到终点的距离
                    max_move = min(abs(remaining_dx), self.physical_width * 0.3)
                    move_distance = random.uniform(min_segment_length, max_move)
                else:
                    # 如果水平距离很小，直接移动到终点的x坐标
                    move_distance = abs(remaining_dx)
                
                # 确定移动方向（朝向终点）
                direction = 1 if remaining_dx >= 0 else -1
                new_x = current_point[0] + move_distance * direction
                new_point = (new_x, current_point[1])
            else:
                # 垂直移动
                if abs(remaining_dy) > min_segment_length:
                    # 随机移动距离，但不超过到终点的距离
                    max_move = min(abs(remaining_dy), self.physical_height * 0.3)
                    move_distance = random.uniform(min_segment_length, max_move)
                else:
                    # 如果垂直距离很小，直接移动到终点的y坐标
                    move_distance = abs(remaining_dy)
                
                # 确定移动方向（朝向终点）
                direction = 1 if remaining_dy >= 0 else -1
                new_y = current_point[1] + move_distance * direction
                new_point = (current_point[0], new_y)
            
            # 检查新点是否在边界内
            if self._is_point_within_bounds(new_point, path_width):
                path_points.append(new_point)
                current_point = new_point
                # 切换方向
                move_horizontal = not move_horizontal
            else:
                # 如果超出边界，调整移动距离
                if move_horizontal:
                    if new_x < path_width/2:
                        new_x = path_width/2
                    elif new_x > self.physical_width - path_width/2:
                        new_x = self.physical_width - path_width/2
                    new_point = (new_x, current_point[1])
                else:
                    if new_y < path_width/2:
                        new_y = path_width/2
                    elif new_y > self.physical_height - path_width/2:
                        new_y = self.physical_height - path_width/2
                    new_point = (current_point[0], new_y)
                
                path_points.append(new_point)
                current_point = new_point
                # 切换方向
                move_horizontal = not move_horizontal
        
        # 添加终点
        path_points.append(end_point)
        
        return path_points

    def add_path_to_map(self, path_points: List[Tuple[float, float]], path_width: float = 1.5, 
                       priority: float = 0.5) -> bool:
        """
        将路径区域添加到掩码地图中
        """
        if len(path_points) < 2:
            raise ValueError("路径点集至少需要2个点")
        
        success = True
        
        # 为路径的每个段创建矩形掩码
        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            
            # 计算段的中心点
            center_x = (start_point[0] + end_point[0]) / 2
            center_y = (start_point[1] + end_point[1]) / 2
            
            # 计算段的长度和角度
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            length = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 确定矩形的半宽和半高
            half_width_segment = length / 2
            half_height_segment = path_width / 2
            
            # 添加矩形掩码代表路径段
            if not self.add_mask(center_x, center_y, angle, half_width_segment, half_height_segment, priority):
                success = False
                print(f"警告: 无法添加从 {start_point} 到 {end_point} 的路径段")
        
        # 存储路径信息
        self.paths.append({
            'points': path_points,
            'width': path_width,
            'priority': priority
        })
        
        return success

    def _is_point_within_bounds(self, point: Tuple[float, float], path_width: float = 0.5) -> bool:
        """
        检查点是否在物理边界内（考虑路径宽度作为安全边距）
        """
        x, y = point
        margin = path_width / 2
        return (margin <= x <= self.physical_width - margin and 
                margin <= y <= self.physical_height - margin)

    def _draw_mask(self, mask: Dict):
        """
        绘制单个掩码到网格图像
        """
        priority = mask['priority']
        points = mask['points']
        
        # 根据优先级选择颜色
        if priority == 0.5:
            color = (255, 0, 0)  # 红色
        elif priority == 0.7:
            color = (0, 255, 0)  # 绿色
        elif priority == 1.0:
            color = (0, 0, 255)  # 蓝色
        elif priority == 1.1:
            color = (0, 255, 255)  # 蓝色
        elif priority == 1.4:
            color = (255, 255, 0)  # 黄色（1.4优先级）
        else:
            color = (128, 128, 128)  # 灰色（未知优先级）
        
        # 绘制填充的多边形
        cv2.fillPoly(self.grid_image, [points], color)
        # # 绘制边界
        # cv2.polylines(self.grid_image, [points], True, (0, 0, 0), 1)
        
        # 计算该mask所在网格单元（假设网格单元大小已知或可通过其他方式计算）
        cell_x = int(mask['center_physical'][0] / self.grid_size)
        cell_y = int(mask['center_physical'][1] / self.grid_size)
        
        # 统计该网格单元内的mask数量
        if (cell_x, cell_y) not in self.cell_mask_count:
            self.cell_mask_count[(cell_x, cell_y)] = 0
        self.cell_mask_count[(cell_x, cell_y)] += 1
        
        # 标注优先级和尺寸信息
        center = mask['center_grid']
        info_text = f"P:{priority} W:{mask['half_width']*2:.1f} H:{mask['half_height']*2:.1f}"
        cv2.putText(self.grid_image, info_text, (center[0]-20, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    def grid_to_physical(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        将网格坐标转换为物理坐标
        """
        physical_x = grid_x * self.grid_size
        physical_y = grid_y * self.grid_size
        return physical_x, physical_y
    

    def _check_region_path_intersection(self, center_x: float, center_y: float,
                                        region_half_width: float, region_half_height: float,
                                        safety_buffer: float = 0.0) -> bool:
        left = center_x - region_half_width - safety_buffer
        right = center_x + region_half_width + safety_buffer
        bottom = center_y - region_half_height - safety_buffer
        top = center_y + region_half_height + safety_buffer

        for path in self.paths:
            path_width = path['width']
            for i in range(len(path['points']) - 1):
                start_point = path['points'][i]
                end_point = path['points'][i + 1]
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                length = math.sqrt(dx ** 2 + dy ** 2)
                if length == 0:
                    continue

                dir_x = dx / length
                dir_y = dy / length
                norm_x = -dir_y
                norm_y = dir_x

                half_width = path_width / 2
                path_rect_points = [
                    (start_point[0] - norm_x * half_width, start_point[1] - norm_y * half_width),
                    (start_point[0] + norm_x * half_width, start_point[1] + norm_y * half_width),
                    (end_point[0] + norm_x * half_width, end_point[1] + norm_y * half_width),
                    (end_point[0] - norm_x * half_width, end_point[1] - norm_y * half_width)
                ]

                if self._rect_rect_intersection((left, bottom, right, top), path_rect_points):
                    return True
        return False

    def _rect_rect_intersection(self, rect1: Tuple[float, float, float, float],
                                rect2_points: List[Tuple[float, float]], tolerance: float = 1e-6) -> bool:
        left, bottom, right, top = rect1
        rect1_points = [(left, bottom), (right, bottom), (right, top), (left, top)]

        rects = [rect1_points, rect2_points]
        for rect in rects:
            for j in range(len(rect)):
                p1 = rect[j]
                p2 = rect[(j + 1) % len(rect)]
                edge = [p2[0] - p1[0], p2[1] - p1[1]]
                normal = [-edge[1], edge[0]]

                min1, max1 = self._project_polygon(np.array(rect1_points), normal)
                min2, max2 = self._project_polygon(np.array(rect2_points), normal)

                if max1 < min2 - tolerance or max2 < min1 - tolerance:
                    return False
        return True

    def _project_polygon(self, points: np.ndarray, axis: List[float]):
        axis = np.array(axis)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        projections = points @ axis
        return projections.min(), projections.max()

    def calculate_intelligent_offset(self, collision_info: Dict,
                                     base_shift_x: float, base_shift_y: float,
                                     attempt_count: int) -> Tuple[float, float]:
        base_offset_x = attempt_count * base_shift_x
        base_offset_y = attempt_count * base_shift_y

        if not collision_info.get('overlap', False):
            return (base_offset_x, base_offset_y)

        collision_dx, collision_dy = collision_info.get('collision_direction', (0, 0))
        overlap_area = collision_info.get('overlap_area', 1.0)

        perpendicular_x = -collision_dy
        perpendicular_y = collision_dx

        area_factor = min(overlap_area / 100.0, 3.0)

        intelligent_offset_x = (perpendicular_x * 0.8 + collision_dx * 0.2) * area_factor
        intelligent_offset_y = (perpendicular_y * 0.8 + collision_dy * 0.2) * area_factor

        total_offset_x = base_offset_x + intelligent_offset_x * 0.5
        total_offset_y = base_offset_y + intelligent_offset_y * 0.5

        return (total_offset_x, total_offset_y)

    def spiral_offset(self, attempt_count: int, step: float = 1.0) -> Tuple[float, float]:
        angle = attempt_count * 2.399963
        radius = step * math.sqrt(attempt_count)
        return radius * math.cos(angle), radius * math.sin(angle)

    def adaptive_offset_strategy(self, collision_history: List[Dict], attempt_count: int) -> Tuple[float, float]:
        if not collision_history:
            return self.calculate_intelligent_offset({'overlap': False}, 0, 0, attempt_count)

        recent_collisions = collision_history[-5:]
        avg_direction_x, avg_direction_y = 0, 0
        collision_count = 0
        for collision in recent_collisions:
            if 'collision_direction' in collision:
                dx, dy = collision['collision_direction']
                avg_direction_x += dx
                avg_direction_y += dy
                collision_count += 1

        if collision_count > 0 and attempt_count < 10:
            avg_direction_x /= collision_count
            avg_direction_y /= collision_count
            avoid_direction_x = -avg_direction_x
            avoid_direction_y = -avg_direction_y

            random_factor_x = random.uniform(-0.5, 0.5)
            random_factor_y = random.uniform(-0.5, 0.5)

            intelligent_offset_x = avoid_direction_x + random_factor_x
            intelligent_offset_y = avoid_direction_y + random_factor_y

            length = max((intelligent_offset_x ** 2 + intelligent_offset_y ** 2) ** 0.5, 0.001)
            intelligent_offset_x = intelligent_offset_x / length * min(attempt_count * 0.5, 5.0)
            intelligent_offset_y = intelligent_offset_y / length * min(attempt_count * 0.5, 5.0)

            return (intelligent_offset_x, intelligent_offset_y)
        else:
            return self.spiral_offset(attempt_count, step=1.0)

    def generate_batch_grid_regions(self, half_width: float, half_height: float,
                                    rows: int, cols: int,
                                    gap_x: float = 0.5, gap_y: float = 0.1,
                                    num_regions: int = 1, direction: str = 'horizontal',
                                    region_priority: float = 0.7,
                                    min_region_spacing: float = 2.0,
                                    max_collision_attempts: int = 200,
                                    debug: bool = False) -> List[Dict]:

        tmp_regions = []
        region_width = cols * (half_width * 2) + (cols - 1) * gap_x + gap_x * 2
        region_height = rows * (half_height * 2) + (rows - 1) * gap_y + gap_y * 2

        if direction == 'horizontal':
            step_x = region_width + min_region_spacing
            step_y = 0
            start_x = region_width / 2 + min_region_spacing
            start_y = self.physical_height / 2
        else:
            step_x = 0
            step_y = region_height + min_region_spacing
            start_x = self.physical_width / 2
            start_y = region_height / 2 + min_region_spacing

        for i in range(num_regions):
            attempt_count = 0
            placement_successful = False
            collision_history = []

            while attempt_count < max_collision_attempts and not placement_successful:
                base_region_center_x = start_x + i * step_x
                base_region_center_y = start_y + i * step_y

                intelligent_offset_x, intelligent_offset_y = self.adaptive_offset_strategy(collision_history, attempt_count)
                region_center_x = base_region_center_x + intelligent_offset_x
                region_center_y = base_region_center_y + intelligent_offset_y

                if not self._is_region_within_bounds(region_center_x, region_center_y,
                                                     region_width / 2, region_height / 2):
                    collision_info = {'overlap': True, 'collision_type': 'boundary'}
                    collision_history.append(collision_info)
                    attempt_count += 1
                    continue

                if self._check_region_path_intersection(region_center_x, region_center_y,
                                                        region_width / 2, region_height / 2):
                    collision_info = {'overlap': True, 'collision_type': 'path'}
                    collision_history.append(collision_info)
                    attempt_count += 1
                    continue

                region_collision_info = self._check_region_region_intersection(region_center_x, region_center_y,
                                                                              region_width / 2, region_height / 2,
                                                                              self.generated_regions,
                                                                              min_region_spacing)
                if region_collision_info['overlap']:
                    collision_history.append(region_collision_info)
                    attempt_count += 1
                    continue

                placement_successful = True
                region_info = {
                    'center': (region_center_x, region_center_y),
                    'half_width': region_width / 2,
                    'half_height': region_height / 2,
                    'rows': rows,
                    'cols': cols,
                    'gap_x': gap_x,
                    'gap_y': gap_y,
                    'mask_half_width': half_width,
                    'mask_half_height': half_height,
                    'priority': region_priority,
                    'direction': direction,
                    'filled': False,
                    'masks': [],
                    'placement_attempts': attempt_count + 1
                }
                self.generated_regions.append(region_info)
                tmp_regions.append(region_info)

                if debug:
                    print(f"成功生成区域 {i + 1}: 中心=({region_center_x:.2f}, {region_center_y:.2f}), 尺寸={region_width:.2f}x{region_height:.2f}, 尝试次数={attempt_count + 1}")

            if not placement_successful and debug:
                print(f"警告: 区域 {i + 1} 在 {max_collision_attempts} 次尝试后仍无法放置")
        
        # 存储区域信息
        if not hasattr(self, 'grid_regions'):
            self.grid_regions = []
        self.grid_regions.extend(self.generated_regions)

        return tmp_regions

    def _is_region_within_bounds(self, center_x: float, center_y: float,
                                 half_width: float, half_height: float) -> bool:
        left = center_x - half_width
        right = center_x + half_width
        bottom = center_y - half_height
        top = center_y + half_height
        return (left >= 0 and right <= self.physical_width and
                bottom >= 0 and top <= self.physical_height)

    def _check_region_region_intersection(self, center_x: float, center_y: float,
                                          region_half_width: float, region_half_height: float,
                                          existing_regions: List[Dict],
                                          min_spacing: float = 0.0) -> Dict:
        new_left = center_x - region_half_width - min_spacing
        new_right = center_x + region_half_width + min_spacing
        new_bottom = center_y - region_half_height - min_spacing
        new_top = center_y + region_half_height + min_spacing

        max_overlap_area = 0
        best_direction = (0, 0)
        nearest_region_idx = -1

        for idx, region in enumerate(existing_regions):
            exist_center_x, exist_center_y = region['center']
            exist_left = exist_center_x - region['half_width']
            exist_right = exist_center_x + region['half_width']
            exist_bottom = exist_center_y - region['half_height']
            exist_top = exist_center_y + region['half_height']

            x_overlap = max(0, min(new_right, exist_right) - max(new_left, exist_left))
            y_overlap = max(0, min(new_top, exist_top) - max(new_bottom, exist_bottom))
            overlap_area = x_overlap * y_overlap

            if overlap_area > 0:
                dx = center_x - exist_center_x
                dy = center_y - exist_center_y
                length = max((dx ** 2 + dy ** 2) ** 0.5, 0.001)
                direction = (dx / length, dy / length)

                if overlap_area > max_overlap_area:
                    max_overlap_area = overlap_area
                    best_direction = direction
                    nearest_region_idx = idx

        if max_overlap_area > 0:
            return {
                'overlap': True,
                'collision_direction': best_direction,
                'overlap_area': max_overlap_area,
                'nearest_region_index': nearest_region_idx
            }
        else:
            return {'overlap': False}


    def fill_selected_region(self, region_info: Dict, num_masks: int = 5,
                        priority: float = 0.7, max_per_cell: int = 2) -> List[Dict]:
        """
        在选定的区域中随机放置mask（修复坐标计算问题）
        """
        if region_info.get('filled', False):
            print("警告: 该区域已被填充")
            return []
        
        successful_masks = []
        attempt_count = 0
        
        # 获取区域参数
        region_center_x, region_center_y = region_info['center']
        region_half_width = region_info['half_width']
        region_half_height = region_info['half_height']
        mask_half_width = region_info['mask_half_width']
        mask_half_height = region_info['mask_half_height']
        rows = region_info['rows']
        cols = region_info['cols']
        gap_x = region_info['gap_x']
        gap_y = region_info['gap_y']
        
        # 计算每个网格单元的尺寸和位置
        cell_width = (mask_half_width * 2 + gap_x)
        cell_height = (mask_half_height * 2 + gap_y)
        
        # 计算区域左上角起点（相对于区域中心）
        start_x = region_center_x - region_half_width + mask_half_width
        start_y = region_center_y - region_half_height + mask_half_height
        
        # 生成网格单元中心点
        grid_cells = []
        for row in range(rows):
            for col in range(cols):
                cell_x = start_x + col * cell_width
                cell_y = start_y + row * cell_height
                grid_cells.append((cell_x, cell_y))
        
        # 初始化每个网格单元的mask计数
        cell_counts = [0] * len(grid_cells)
        # print(" ------ ", f"{cell_counts}, {rows}x{cols}")
        
        # 随机生成指定数量的mask
        for _ in range(num_masks):
            attempt_count += 1
            if attempt_count > 100:  # 防止无限循环
                print("达到最大尝试次数，停止生成")
                break
            
            # 找出还有空间的网格单元
            available_cells = [i for i, count in enumerate(cell_counts) if count < max_per_cell]
            
            if not available_cells:
                print("所有网格单元已达到最大mask数量限制")
                break
            
            # 随机选择一个网格单元
            cell_idx = random.choice(available_cells)
            cell_center_x, cell_center_y = grid_cells[cell_idx]
            
            # 在网格单元内随机偏移（限制在单元空间内）
            max_offset_x = min(gap_x / 2, mask_half_width * 0.5)
            max_offset_y = min(gap_y / 2, mask_half_height * 0.5)
            
            offset_x = random.uniform(-max_offset_x, max_offset_x)
            offset_y = random.uniform(-max_offset_y, max_offset_y)
            
            mask_center_x = cell_center_x
            mask_center_y = cell_center_y
            
            # 随机生成旋转角度
            # rotation = random.uniform(0, 360)
            rotation = 0
            
            # 检查是否在区域边界内
            if not self._is_point_in_region(mask_center_x, mask_center_y, 
                                        region_center_x, region_center_y,
                                        region_half_width, region_half_height,
                                        mask_half_width, mask_half_height):
                continue
            
            # # 检查是否在物理边界内
            # if not self._is_mask_within_bounds(mask_center_x, mask_center_y, 
            #                                 mask_half_width, mask_half_height):
            #     continue
            
            # # 检查与现有掩码的最小间距
            # min_spacing = max(gap_x, gap_y) * 0.2  # 使用间隙的80%作为最小间距
            # if not self._check_min_spacing(mask_center_x, mask_center_y, 
            #                             mask_half_width, mask_half_height, 
            #                             min_spacing):
            #     continue
            
            # 尝试添加mask
            if self.add_mask(mask_center_x, mask_center_y, rotation, 
                            mask_half_width, mask_half_height, priority):
                mask_info = {
                    'center_physical': (mask_center_x, mask_center_y),
                    'rotation': rotation,
                    'priority': priority
                }
                successful_masks.append(mask_info)
                cell_counts[cell_idx] += 1
        
        # 更新区域信息
        region_info['filled'] = True
        region_info['masks'].extend(successful_masks)
        
        print(f"在区域中成功添加了 {len(successful_masks)} 个mask")
        return successful_masks
    
    def _is_point_in_region(self, point_x: float, point_y: float,
                       region_center_x: float, region_center_y: float,
                       region_half_width: float, region_half_height: float,
                       mask_half_width: float, mask_half_height: float) -> bool:
        """
        检查点是否在区域边界内（考虑mask的尺寸）
        """
        # 计算mask的边界
        mask_left = point_x - mask_half_width
        mask_right = point_x + mask_half_width
        mask_bottom = point_y - mask_half_height
        mask_top = point_y + mask_half_height
        
        # 计算区域的边界
        region_left = region_center_x - region_half_width
        region_right = region_center_x + region_half_width
        region_bottom = region_center_y - region_half_height
        region_top = region_center_y + region_half_height
        
        # 检查mask是否完全在区域内
        return (mask_left >= region_left and mask_right <= region_right and
                mask_bottom >= region_bottom and mask_top <= region_top)

    def reset_region(self, region_index: int=-1) -> bool:
        """
        重置指定区域，允许再次填充
        
        :param region_index: 区域索引
        :return: 成功重置返回True，否则返回False
        """
        if not hasattr(self, 'grid_regions') or region_index >= len(self.grid_regions):
            print(f"错误: 区域索引 {region_index} 无效")
            return False
        if region_index >= 0:
            region = self.grid_regions[region_index]
            region['filled'] = False
            region['masks'] = []
        else:
            for i in range(len(self.grid_regions)):
                region = self.grid_regions[i]
                region['filled'] = False
                region['masks'] = []
        
        print(f"区域 {region_index} 已重置，可以重新填充")
        return True

    def _draw_path(self, image: np.ndarray, path: Dict):
        """
        在图像上绘制路径
        """
        points = path['points']
        width = path['width']
        priority = path['priority']
        
        # 根据优先级选择颜色
        if priority == 0.5:
            color = (0, 0, 0)  # 红色
        elif priority == 0.7:
            color = (0, 255, 0)  # 绿色
        elif priority == 1.0:
            color = (0, 0, 255)  # 蓝色
        elif priority == 1.4:
            color = (255, 255, 0)  # 黄色（1.4优先级）
        else:
            color = (128, 128, 128)  # 灰色（未知优先级）
        
        # 绘制路径线段
        for i in range(len(points) - 1):
            start_grid = self.physical_to_grid(points[i][0], points[i][1])
            end_grid = self.physical_to_grid(points[i+1][0], points[i+1][1])
            
            # 绘制线段
            line_thickness = max(1, int(width / self.grid_size))
            cv2.line(image, start_grid, end_grid, color, line_thickness)
        
        # 绘制路径点
        for point in points:
            grid_point = self.physical_to_grid(point[0], point[1])
            cv2.circle(image, grid_point, 3, color, -1)

    def _add_legend(self, image: np.ndarray) -> np.ndarray:
        """
        添加图例到图像
        """
        height, width = image.shape[:2]
        
        # 创建图例区域
        legend_height = 100
        legend = np.ones((legend_height, width, 3), dtype=np.uint8) * 255
        
        # 添加标题
        cv2.putText(legend, "Mask Legend - Updated Priority Rules", (10, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 添加优先级说明
        cv2.putText(legend, "Priority 0.5 (Red): Can be covered by any mask", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(legend, "Priority 0.7 (Green): Can be covered by 0.7 only", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(legend, "Priority 1.0 (Blue): Cannot be covered by any mask", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(legend, "Priority 1.4 (Yellow): Merged 0.7 areas, same rules as 0.7", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(legend, "0.7+0.7 overlap → merged as 1.4 with adjusted center/orientation", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # 将图例添加到图像底部
        result = np.vstack([image, legend])
        return result
    
    def visualize_all_in_one(self, scale: int = 10, show_info: bool = True, 
                        highlight_mask_id: Optional[int] = None,
                        highlight_color: Tuple[int, int, int] = (255, 0, 255)):
        """
        在一张图上可视化所有结果：掩码、路径和区域，可突出显示单个掩码
        
        :param scale: 图像缩放比例
        :param show_info: 是否显示图例信息
        :param highlight_mask_id: 要突出显示的掩码ID（可选）
        :param highlight_color: 突出显示的颜色（BGR格式，默认洋红色）
        """
        # 创建副本图像用于显示
        display_image = self.grid_image.copy()
        
        # 如果指定了要突出显示的掩码ID，先绘制该掩码
        if highlight_mask_id is not None:
            highlight_mask = None
            for mask in self.masks:
                if mask['id'] == highlight_mask_id:
                    highlight_mask = mask
                    break
            
            if highlight_mask:
                self._draw_single_mask(display_image, highlight_mask, highlight_color)
                print(f"已突出显示掩码 ID: {highlight_mask_id}")
            else:
                print(f"警告: 未找到ID为 {highlight_mask_id} 的掩码")
        
        for mask in self.masks:
            self._draw_mask(mask)
        
        # 1. 绘制所有路径
        for path in self.paths:
            self._draw_path(display_image, path)
        
        # 2. 绘制所有区域边界
        if hasattr(self, 'grid_regions'):
            for region in self.grid_regions:
                self._draw_region(display_image, region)
        
        # # 3. 绘制图例
        # if show_info:
        #     display_image = self._add_legend(display_image)
        
        # 放大图像以便查看
        height, width = display_image.shape[:2]
        large_image = cv2.resize(display_image, (width * scale, height * scale), 
                            interpolation=cv2.INTER_NEAREST)
        
        # cv2.imshow('All in One Visualization', large_image)
        cv2.imwrite("test.jpg", large_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def _draw_single_mask(self, image: np.ndarray, mask: Dict, 
                        color: Tuple[int, int, int] = (255, 0, 255)):
        """
        在图像上绘制单个掩码，使用指定颜色
        
        :param image: 要绘制的图像
        :param mask: 掩码字典
        :param color: 绘制颜色（BGR格式）
        """
        priority = mask['priority']
        points = mask['points']
        
        # 使用指定颜色绘制填充的多边形
        cv2.fillPoly(image, [points], color)
        
        # 绘制边界（使用黑色边框）
        cv2.polylines(image, [points], True, (0, 0, 0), 2)
        
        # 标注优先级和尺寸信息
        center = mask['center_grid']
        info_text = f"ID:{mask['id']} P:{priority} W:{mask['half_width']*2:.1f} H:{mask['half_height']*2:.1f}"
        cv2.putText(image, info_text, (center[0]-30, center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 在中心点绘制一个标记
        cv2.drawMarker(image, center, (0, 0, 0), cv2.MARKER_CROSS, 10, 2)

    def _draw_region(self, image: np.ndarray, region: Dict):
        """
        在图像上绘制区域边界
        """
        center_x, center_y = region['center']
        half_width = region['half_width']
        half_height = region['half_height']
        
        # 将物理坐标转换为网格坐标
        left = int((center_x - half_width) / self.grid_size)
        right = int((center_x + half_width) / self.grid_size)
        top = int((center_y - half_height) / self.grid_size)
        bottom = int((center_y + half_height) / self.grid_size)
        
        # 根据填充状态选择颜色
        color = (0, 255, 0) if not region.get('filled', False) else (0, 0, 255)  # 绿色:未填充, 蓝色:已填充
        
        # 绘制区域边界
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        
        # 标注区域信息
        region_id = self.grid_regions.index(region)
        info_text = f"Region {region_id}: {'Filled' if region.get('filled', False) else 'Empty'}"
        cv2.putText(image, info_text, (left + 5, top + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def get_mask_info(self) -> List[Dict]:
        """
        获取所有掩码的详细信息
        """
        info_list = []
        for mask in self.masks:
            info_list.append({
                'id': mask['id'],
                'center_physical': mask['center_physical'],
                'center_grid': mask['center_grid'],
                'rotation': mask['rotation'],
                'half_width': mask['half_width'],
                'half_height': mask['half_height'],
                'priority': mask['priority']
            })
        return info_list

    def get_path_info(self) -> List[Dict]:
        """
        获取所有路径的详细信息
        """
        return self.paths

    def clear_masks(self):
        """
        清除所有掩码
        """
        self.masks = []
        self.paths = []
        self._next_mask_id = 0  # 重置ID计数器
        self.grid_image = np.ones((self.grid_size_y, self.grid_size_x, 3), dtype=np.uint8) * 255

    def print_detailed_info(self):
        """
        打印所有掩码和路径的详细信息
        """
        print("\n=== 掩码详细信息 ===")
        print(f"物理区域尺寸: {self.physical_width} x {self.physical_height}")
        print(f"网格尺寸: {self.grid_size}")
        print(f"网格单元数量: {self.grid_size_x} x {self.grid_size_y}")
        print(f"掩码总数: {len(self.masks)}")
        
        # 按优先级分类统计
        priority_counts = {0.5: 0, 0.7: 0, 1.0: 0, 1.4: 0}
        for mask in self.masks:
            priority = mask['priority']
            if priority in priority_counts:
                priority_counts[priority] += 1
        
        print(f"\n按优先级分类:")
        for priority, count in priority_counts.items():
            print(f"  优先级 {priority}: {count} 个掩码")
        
        # 打印每个掩码的详细信息
        print(f"\n每个掩码的详细信息:")
        for i, mask in enumerate(self.masks):
            print(f"  掩码 {i+1} (ID: {mask['id']}):")
            print(f"    物理中心: {mask['center_physical']}")
            print(f"    网格中心: {mask['center_grid']}")
            print(f"    尺寸: {mask['half_width']*2:.2f} x {mask['half_height']*2:.2f}")
            print(f"    旋转角度: {mask['rotation']:.1f}°")
            print(f"    优先级: {mask['priority']}")
        
        # 打印路径信息
        print(f"\n=== 路径详细信息 ===")
        print(f"路径总数: {len(self.paths)}")
        for i, path in enumerate(self.paths):
            print(f"  路径 {i+1}:")
            print(f"    点数: {len(path['points'])}")
            print(f"    宽度: {path['width']}")
            print(f"    优先级: {path['priority']}")
            print(f"    起点: {path['points'][0]}")
            print(f"    终点: {path['points'][-1]}")

# 使用示例
if __name__ == "__main__":
    # 创建管理器
    manager = MaskManager(50, 80, 0.1)
    
    # # 货架区域示例定义
    # shelf_zone_example = {
    #     'center': (25, 25),      # 区域中心坐标
    #     'half_width': 25,        # 区域半宽
    #     'half_height': 25,       # 区域半高
    #     'type': 'shelf_zone',    # 区域类型标识
    #     'priority': 1           # 区域优先级
    # }
    path_half_width = 1.0
    # 货架参数
    shelf_length = 4.0
    shelf_depth = 1.5

    region_min = (2,2)
    region_max = (45,45)

    # 使用示例
    path_points = manager.generate_constrained_path_v2(start_point=(5, 5),
        end_point=(45, 45),
        shelf_zone={'min': (0, 0), 'max': (50, 50)},
        path_half_width=path_half_width,
        shelf_half_length=shelf_length/2,
        shelf_half_depth=shelf_depth/2,
        first_direction='horizontal',
        priority=1.0,
        gap=0.2,
        max_points=15)

    shelves = manager.generate_shelves_in_region(
        path_points=path_points,
        path_half_width=path_half_width,
        region_min=region_min,
        region_max=region_max,
        shelf_length=shelf_length,
        shelf_depth=shelf_depth,
        safety_margin=0.2,
        orientation="horizontal"
    )
    print(shelves)
    
    # 3. 批量生成区域
    regions = manager.generate_batch_grid_regions(
        half_width=2.0,      # 单个mask的半宽
        half_height=1.5,     # 单个mask的半高
        rows=3,              # 每个区域3行
        cols=4,              # 每个区域4列
        gap_x=0.5,           # 水平间隙
        gap_y=0.3,           # 垂直间隙
        num_regions=2,       # 生成2个区域
        direction='horizontal',  # 水平排列
        min_region_spacing=3.0  # 区域间距
    )

    # 4. 选择第一个区域并填充
    if regions:
        for selected_region in regions:
            masked = random.randint(0, 1)
            if masked:
                masks = manager.fill_selected_region(
                    selected_region,
                    num_masks=8,     # 生成8个mask
                    priority=0.7,    # 优先级0.7
                    max_per_cell=2   # 每个网格单元最多2个mask
                )
                
                print(f"区域中心: {selected_region['center']}")
                print(f"区域尺寸: {selected_region['half_width']*2:.2f}x{selected_region['half_height']*2:.2f}")
                print(f"成功添加mask: {len(masks)}个")
                
                # 检查每个mask的位置
                for i, mask in enumerate(masks):
                    print(f"Mask {i}: {mask['center_physical']}")
    else:
        print("没有成功生成任何区域")
    result = manager.add_multiple_masks(half_width=8.0, half_height=4.0, priority=1.0, count=3) 

    manager.visualize_all_in_one()
    
    # 打印详细信息
    manager.print_detailed_info()

    # 6. 重置区域以便重新使用
    manager.reset_region()
    
    # 清除所有掩码
    manager.clear_masks()