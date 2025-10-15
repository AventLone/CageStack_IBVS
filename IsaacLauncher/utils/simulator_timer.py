import asyncio, uuid


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
        
        def _on_physics_step(step_dt: float):
            nonlocal count
            count += 1
            if count >= steps and not fut.done():
                # 这里不直接 remove 回调，交给 finally 做，避免竞态
                fut.set_result(None)

        self._world.add_physics_callback(cb_name, _on_physics_step)

        try:
            await fut
        finally:
            # 移除回调，防止泄漏/重复触发
            try:
                self._world.remove_physics_callback(cb_name)
            except Exception:
                pass