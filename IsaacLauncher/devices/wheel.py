from omni.isaac.dynamic_control import _dynamic_control
from pxr import UsdPhysics
from omni.usd import get_context


class VN_Wheel:
    def __init__(self, prim_path: str, joint_name: str):
        stage = get_context().get_stage()
        wheel_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(prim_path + "/" + joint_name), "angular")
        wheel_drive.GetStiffnessAttr().Set(0)
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        forklift = self.dc.get_articulation(prim_path)
        self.wheel = self.dc.find_articulation_dof(forklift, joint_name)
        
    def setVelocity(self, v):
        self.dc.set_dof_velocity_target(self.wheel, v)
        
    @property
    def state(self):
        wheel_state = self.dc.get_dof_state(self.wheel, _dynamic_control.STATE_ALL)
        return wheel_state.pos
