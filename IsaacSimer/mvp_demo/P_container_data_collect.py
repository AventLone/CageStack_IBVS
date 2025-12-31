from utils.common import openSimuApp, load_config
import math, os


class ContainerDataCollector:
    def __init__(self) -> None:
        self.simulation_app = openSimuApp("configs/e_test_2.yaml")
        config = load_config("configs/e_test_2.yaml")

        import carb.settings
        # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        from isaacsim.core.api.world import World
        # self.world = World(physics_dt=0.012,       # Physics time step
        #                    rendering_dt=0.02,   # Render timestep
        #                    stage_units_in_meters=1.0)
        self.world = World()
        self.world.reset()

        from devices.vehicles import Vehicle
        self._forklift = Vehicle(world=self.world, cfg=config["vehicle"])  # type: ignore

        from utils.common import SimTimer
        self._sim_timer = SimTimer(world=self.world)

        import omni.replicator.core as rep
        rep.orchestrator.set_capture_on_play(False)
        trigger = rep.trigger.on_frame(interval=2, num_frames=2060, rt_subframes=8)  # 每10帧触发一次 randomization/写出
        self._body_left_camera_data = rep.create.render_product(
            "/World/E_car_finish/body/cameras/left_camera/SG3S_ISX031C_GMSL2F_H190XA_01",
            (960, 768))
        self._body_right_camera_data = rep.create.render_product(
            "/World/E_car_finish/body/cameras/right_camera/SG3S_ISX031C_GMSL2F_H190XA_01",
            (960, 768))
        self.view_port = rep.create.render_product("/OmniverseKit_Persp", (1980, 1080))
        left_camera_dir = os.path.join(os.getcwd(), "data/instance_segmentation/left")
        right_camera_dir = os.path.join(os.getcwd(), "data/instance_segmentation/right")
        view_port_dir = os.path.join(os.getcwd(), "data/instance_segmentation/view_port")
        self._left_camera_writer = rep.BasicWriter(output_dir=left_camera_dir,
                                            frame_padding=5,
                                            rgb=True,
                                            instance_segmentation=True)
        self._right_camera_writer = rep.BasicWriter(output_dir=right_camera_dir,
                                    frame_padding=5,
                                    rgb=True,
                                    instance_segmentation=True)
        self.view_port_writer = rep.BasicWriter(output_dir=view_port_dir,
                                                frame_padding=5,
                                                rgb=True)
        self._left_camera_writer.attach(self._body_left_camera_data, trigger=trigger)
        self._right_camera_writer.attach(self._body_right_camera_data, trigger=trigger)
        self.view_port_writer.attach(self.view_port, trigger=trigger)


    async def _collect_data(self, duration: float, hz: int):
        period = 1.0 / hz
        epochs = round(duration / period)

        import omni.replicator.core as rep
        for _ in range(100000):
            await rep.orchestrator.step_async()
            # await asyncio.sleep(1.0)
        await rep.orchestrator.wait_until_complete_async()

        self._left_camera_writer.detach()
        self._right_camera_writer.detach()
        self._body_left_camera_data.destroy()
        self._body_right_camera_data.destroy()

    async def _control_sequence(self):

        from isaacsim.core.utils.types import ArticulationAction
        self._forklift.move(math.pi)

        await self._sim_timer.sleep(2.5)
        fork_action = ArticulationAction(joint_velocities=[-0.3],
                                         joint_indices=[self._forklift.get_dof_index("fork_z")])
        self._forklift.apply_action(fork_action)

        await self._sim_timer.sleep(1.0)
        fork_action = ArticulationAction(joint_velocities=[0.0],
                                         joint_indices=[self._forklift.get_dof_index("fork_z")])
        self._forklift.apply_action(fork_action)

        # self._forklift.move(2.0 * math.pi)
        await self._sim_timer.sleep(22.6)
        self._forklift.move(0.0)
        fork_action = ArticulationAction(joint_velocities=[0.3],
                                         joint_indices=[self._forklift.get_dof_index("fork_z")])
        self._forklift.apply_action(fork_action)


        await self._sim_timer.sleep(1.0)
        fork_action = ArticulationAction(joint_velocities=[0.0],
                                         joint_indices=[self._forklift.get_dof_index("fork_z")])
        self._forklift.apply_action(fork_action)
        self._forklift.move(-1.0 * math.pi)

        await self._sim_timer.sleep(3.0)
        self._forklift.move(-1.5 * math.pi)

        await self._sim_timer.sleep(15.0)
        self._forklift.move(0.0)

    def run(self):
        from omni.kit.async_engine import run_coroutine
        run_coroutine(self._collect_data(duration=50.0, hz=10))
        run_coroutine(self._control_sequence())
        require_reset = False
        while self.simulation_app.is_running():
            self.world.step(render=True)
            if self.world.is_stopped() and not require_reset:   # 播放/暂停与重置逻辑
                require_reset = True

            if self.world.is_playing():
                if require_reset:
                    self.world.reset()
                    require_reset = False
                    self._forklift.move(-math.pi)

        self.simulation_app.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect data")
    parser.add_argument("-e", "--epochs", help="epochs to collect data", default=100)
    args = parser.parse_args()

    collector = ContainerDataCollector()
    collector.run()
