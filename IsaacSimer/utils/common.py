import yaml, asyncio, uuid
from isaacsim.simulation_app import SimulationApp

def load_config(yaml_file_path: str):
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def openSimuApp(config_file_path: str) -> SimulationApp:
    simulate_config = load_config(config_file_path)["simulation_app"]
    simulation_app = SimulationApp(simulate_config["config"])
    from isaacsim.core.utils.stage import open_stage
    open_stage(simulate_config["stage_file_path"])
    simulation_app.update()

    return simulation_app

class SimTimer:
    def __init__(self, world) -> None:
        self._world = world
        self._loop = asyncio.get_event_loop()

    async def sleep(self, seconds: float):
        """await 按仿真时间等待 seconds 秒"""
        assert seconds > 0, "seconds has to be greater than 0"
        
        steps = round(seconds / self._world.get_physics_dt())
        fut = self._loop.create_future()
        cb_name = f"sim_sleep_{uuid.uuid4().hex}"   # 确保并发时唯一
        count = 0
        
        def on_physics_step(step_dt: float):
            nonlocal count
            count += 1
            if count >= steps and not fut.done():
                # 这里不直接 remove 回调，交给 finally 做，避免竞态
                fut.set_result(None)

        self._world.add_physics_callback(cb_name, on_physics_step)

        try:
            await fut
        finally:
            # 移除回调，防止泄漏/重复触发
            try:
                self._world.remove_physics_callback(cb_name)
            except Exception:
                pass