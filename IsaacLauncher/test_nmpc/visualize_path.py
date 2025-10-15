# 可在 Isaac Sim 的 Script Editor 直接跑
from pxr import UsdGeom, Gf, Sdf, Vt
from omni.isaac.core.utils.stage import get_current_stage

def _ensure_basis_curves(stage, path_prim_path, width=0.025, color=(1.0, 0.3, 0.0)):
    curves = UsdGeom.BasisCurves.Get(stage, path_prim_path)
    if not curves:
        curves = UsdGeom.BasisCurves.Define(stage, path_prim_path)
        curves.CreateTypeAttr("linear")
        curves.CreateWidthsAttr(Vt.FloatArray([width]))
        curves.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return curves

def _remove_children(stage, parent_path: str):
    parent = stage.GetPrimAtPath(parent_path)
    if not parent:
        return
    for child in list(parent.GetChildren()):
        stage.RemovePrim(child.GetPath())

def visualize_xytheta_path_with_arrows(
    name: str,
    poses_xytheta,               # [(x, y, th), ...]  th=弧度，Z-up，+X为前
    z_lift: float = 0.02,        # 线抬高，避免与地面闪烁
    line_width: float = 0.025,
    line_color=(1.0, 0.3, 0.0),
    draw_points: bool = False,   # 是否把采样点也画出来
    point_size: float = 0.03,
    point_color=(0.2, 0.8, 0.2),
    arrow_every: int = 5,        # 每隔多少个点画一个箭头
    arrow_len: float = 0.5,      # 箭头长度（米）
    arrow_radius: float = 0.05,  # 箭头底面半径（米）
    arrow_color=(0.1, 0.6, 1.0),
):
    """
    返回已创建/更新的 Prim 路径字典
    """
    stage = get_current_stage()
    base = f"/World/{name}"
    path_prim_path = Sdf.Path(f"{base}/Path")
    arrows_parent_path = f"{base}/Arrows"
    points_prim_path = Sdf.Path(f"{base}/Points")

    # 1) 路径曲线（BasisCurves, linear）
    curves = _ensure_basis_curves(stage, path_prim_path, width=line_width, color=line_color)
    xyz = [Gf.Vec3f(x, y, z_lift) for x, y, _ in poses_xytheta]
    if len(xyz) < 2:
        # 至少两个点才能画线；只有1个点时画点代替
        xyz = xyz * 2
    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(xyz)]))
    curves.CreatePointsAttr().Set(Vt.Vec3fArray(xyz))

    # 2) （可选）采样点
    if draw_points:
        pts_prim = UsdGeom.Points.Get(stage, points_prim_path)
        if not pts_prim:
            pts_prim = UsdGeom.Points.Define(stage, points_prim_path)
            pts_prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*point_color)]))
        pts_prim.CreateWidthsAttr(Vt.FloatArray([point_size] * len(xyz)))
        pts_prim.CreatePointsAttr().Set(Vt.Vec3fArray(xyz))
    else:
        # 不画点则清理旧点
        if stage.GetPrimAtPath(points_prim_path):
            stage.RemovePrim(points_prim_path)

    # 3) 朝向箭头（用 Cone，当作“指向 +X”的箭头头部）
    #    先清空旧箭头，再逐个创建
    if not stage.GetPrimAtPath(arrows_parent_path):
        UsdGeom.Xform.Define(stage, Sdf.Path(arrows_parent_path))
    _remove_children(stage, arrows_parent_path)

    # 需要的下标（每隔 arrow_every 个，且至少取第一个和最后一个）
    idxs = list(range(0, len(poses_xytheta), max(1, arrow_every)))
    if len(poses_xytheta) >= 2 and (idxs[-1] != len(poses_xytheta) - 1):
        idxs.append(len(poses_xytheta) - 1)

    created = {
        "path": str(path_prim_path),
        "arrows_parent": arrows_parent_path,
        "arrows": [],
        "points": str(points_prim_path) if draw_points else None,
    }

    for i in idxs:
        x, y, th = poses_xytheta[i]
        z = z_lift
        arrow_xform_path = Sdf.Path(f"{arrows_parent_path}/arrow_{i:04d}")
        arrow_x = UsdGeom.Xform.Define(stage, arrow_xform_path)

        # 在父Xform上做平移 + 绕Z轴旋转（yaw=th，弧度 → 角度）
        api = UsdGeom.XformCommonAPI(arrow_x)
        api.SetTranslate(Gf.Vec3f(x, y, z))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, th * 180.0 / 3.141592653589793))

        # 创建一个朝 +X 轴的圆锥，作为箭头；把它沿 +X 平移到“底部在原点”位置
        cone_path = Sdf.Path(f"{arrow_xform_path}/Cone")
        cone = UsdGeom.Cone.Define(stage, cone_path)
        cone.CreateAxisAttr("X")                    # 让高度沿 X 轴
        cone.CreateHeightAttr(arrow_len)
        cone.CreateRadiusAttr(arrow_radius)
        cone.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*arrow_color)]))
        # 把锥体中心(默认在高度中点)沿 +X 平移一半，让“底面”落在父Xform原点
        UsdGeom.XformCommonAPI(cone).SetTranslate(Gf.Vec3f(arrow_len * 0.5, 0.0, 0.0))

        created["arrows"].append(str(arrow_xform_path))

    return created

# --- 用法示例 ---
# poses = [(0.0, 0.0, 0.0),
#          (1.0, 0.2, 0.1),
#          (2.0, 0.6, 0.2),
#          (3.0, 1.1, 0.25),
#          (4.0, 1.6, 0.30)]
# visualize_xytheta_path_with_arrows("PlannedPath",
#                                    poses,
#                                    draw_points=True,
#                                    arrow_every=2,
#                                    arrow_len=0.4)
