import random
from isaacsim.util.debug_draw import _debug_draw
import math

draw = _debug_draw.acquire_debug_draw_interface()

class Example():
    def create(self):
        self.draw = _debug_draw.acquire_debug_draw_interface()
        N = 500
        self.point_list = [(random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0)) for _ in range(N)]
        self.color_list = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(N)]
        self.size_list = [10.0 for _ in range(N)]

    def update(self):
        # modify the point list
        for i in range(len(self.point_list)):
            self.point_list[i] = (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0))

        # draw the points
        self.draw.clear_points()
        self.draw.draw_points(self.point_list, self.color_list, self.size_list)

import asyncio
import omni
example = Example()
example.create()

async def update_points():
    # Update 10 times, waiting 10 frames between each update
    for _ in range(10):
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        example.update()

# asyncio.ensure_future(update_points())
from pxr import Gf
import numpy as np

def drawArrow(origin, direction, length=0.4, head_length=0.2, head_width=0.06, color=(1,0,0,1), width=6.0):
    """
    origin: (3,) 起点
    direction: (3,) 方向向量（不必归一化）
    length: 箭杆总长度
    head_length: 箭头沿方向的长度（<= length）
    head_width: 箭头两侧偏移量（大致宽度）
    color: rgba 元组
    width: 线宽
    """
    o = np.array(origin, dtype=float)
    d = np.array(direction, dtype=float)
    nd = d / (np.linalg.norm(d) + 1e-12)

    tip = o + nd * length
    shaft_end = o + nd * (length - head_length)

    # 找一个与 nd 不平行的向量作为参考，构造侧向基
    arbitrary = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(nd, arbitrary)) > 0.99:
        arbitrary = np.array((0.0, 1.0, 0.0))
    side = np.cross(nd, arbitrary)
    side /= (np.linalg.norm(side) + 1e-12)
    up = np.cross(side, nd)
    up /= (np.linalg.norm(up) + 1e-12)

    # 箭头两条边的端点
    head_point1 = tip - nd * head_length + side * head_width
    head_point2 = tip - nd * head_length - side * head_width
    # 也可以用 up 旋转来画“V”形，这里用 side 做左右偏移（效果清晰）

    # 需要传给 debug_draw 的是两组点（每条线的 start 列表 和 end 列表）
    starts = [tuple(o), tuple(shaft_end), tuple(tip)]
    ends   = [tuple(shaft_end), tuple(tip), tuple(head_point1)]
    # 额外的一条从 tip 到 head_point2
    starts.append(tuple(tip))
    ends.append(tuple(head_point2))

    colors = [color] * len(starts)
    widths = [width] * len(starts)

    draw.draw_lines(starts, ends, colors, widths)
    # draw.clear_lines() 可以清除所有线（如果需要清理）
    # _debug_draw.release_debug_draw_interface(draw)  # 可选：释放接口（不常必要）

def drawPoses(poses):
    for pose in poses:
        origin = [pose[0], pose[1], 0.2]
        direction = [math.cos(pose[2]), math.sin(pose[2]), 0]
        drawArrow(origin, direction)

def draw_vertical_rect(center,
                       yaw,
                       width=1.0,
                       height=1.0,
                       color=(1.0, 0.0, 1.0, 1.0),
                       line_thickness=5.0):
    """
    在 world 空间画一个垂直于 XY 平面的矩形（墙）。
    center: (x,y,z)
    yaw: 绕 Z 轴的朝向（弧度）。矩形宽度沿 yaw 方向延展，垂直方向为 +Z。
    width: 矩形水平宽度（沿 yaw）
    height: 矩形竖直高度（沿 Z）
    color: rgba
    line_thickness: 屏幕像素宽度
    clear_first: 是否先 clear_lines()（常用于每帧更新）
    """
    c = np.array(center, dtype=float)
    hw = width / 2.0
    hh = height / 2.0

    # 水平方向向量（矩形的“右”方向）
    r = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)

    # 竖直向量（世界坐标Z轴）
    u = np.array([0.0, 0.0, 1.0], dtype=float)

    # 四个角（顺时针或逆时针，保证闭合）
    p0 = c - r * hw - u * hh  # 左下
    p1 = c + r * hw - u * hh  # 右下
    p2 = c + r * hw + u * hh  # 右上
    p3 = c - r * hw + u * hh  # 左上

    pts = [tuple(p0), tuple(p1), tuple(p2), tuple(p3), tuple(p0)]

    starts = [pts[0], pts[1], pts[2], pts[3]]
    ends   = [pts[1], pts[2], pts[3], pts[0]]  # 最后一条连回起点

    draw.draw_lines(starts, ends, [color] * len(starts), [line_thickness] * len(starts))

def drawPath(path, color=(0.8, 0.8, 0.8, 1), width=3.0):
    draw.clear_lines()   # 清掉这个 draw 实例上所有线
    draw.clear_points()

    color_list = [(0.9, 0.0, 0.0, 1) for _ in range(len(path))]
    draw.draw_points(path, color_list, [8.0] * len(path))

    draw.draw_lines_spline(path, color, width, False)

    
