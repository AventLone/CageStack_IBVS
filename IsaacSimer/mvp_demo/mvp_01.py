from utils.common import openSimuApp, load_config
simulation_app = openSimuApp("configs/mvp_01.yaml")

# Settings
import carb.settings
# Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 0)

import asyncio, time
from itertools import chain
import numpy as np
from isaacsim.core.api import World

from devices.vehicles import Vehicle
from utils.common import SimTimer
from omni.kit.async_engine import run_coroutine
from isaacsim.core.utils.types import ArticulationAction


class MVP():
    def __init__(self, config: dict) -> None:
        self._world: World = World()
        self._world.reset()

        self._AGVs: dict[str, Vehicle] = dict()
        self._simu_timer = SimTimer(self._world)

        self._load_agvs(config["AGVs"])


    def _load_agvs(self, config: dict):
        for agv_name, params in config.items():
            self._AGVs[agv_name] = Vehicle(self._world, params)


    def _simulate(self):
        require_reset = False
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped() and not require_reset:
                require_reset = True
            if self._world.is_playing() and require_reset:
                self._world.reset()
                require_reset = False

    
    async def _task_agv_e(self):
        while simulation_app.is_running():
            self._AGVs['E'].move(3.50)
            await self._simu_timer.sleep(4.8)
            self._AGVs['E'].move(0.0)

            fork_action = ArticulationAction(joint_velocities=[0.1],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)
            await self._simu_timer.sleep(2.0)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)

            await self._simu_timer.sleep(2.0)

            fork_action = ArticulationAction(joint_velocities=[-0.1],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)
            await self._simu_timer.sleep(2.0)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)

            self._AGVs['E'].move(-3.50)
            await self._simu_timer.sleep(4.8)
            self._AGVs['E'].move(0.0)


    async def _task_agv_p(self):
        while simulation_app.is_running():
            self._AGVs['P'].move(-3.52)
            await self._simu_timer.sleep(4.8)
            self._AGVs['P'].move(0.0)

            fork_action = ArticulationAction(joint_velocities=[0.1],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)
            await self._simu_timer.sleep(2.0)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)

            await self._simu_timer.sleep(2.0)

            fork_action = ArticulationAction(joint_velocities=[-0.1],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)
            await self._simu_timer.sleep(2.0)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)

            self._AGVs['P'].move(3.52)
            await self._simu_timer.sleep(4.8)
            self._AGVs['P'].move(0.0)

    async def _task_agv_amr(self):
        while simulation_app.is_running():
            self._AGVs["AMR"].move(4.0)
            await self._simu_timer.sleep(7.8)
            self._AGVs["AMR"].move(-4.0)
            await self._simu_timer.sleep(7.8)

    async def _task(self):
        while simulation_app.is_running():
            self._AGVs['AMR'].move(5.0)

            self._AGVs['P'].move(-3.52)
            self._AGVs['E'].move(3.3)
            await self._simu_timer.sleep(4.8)
            self._AGVs['P'].move(0.0)
            self._AGVs['E'].move(0.0)


            fork_action = ArticulationAction(joint_velocities=[0.1],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)
            fork_action = ArticulationAction(joint_velocities=[-0.1],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)
            await self._simu_timer.sleep(2.0)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)

            # await self._simu_timer.sleep(2.0)
            await self._simu_timer.sleep(1.0)
            self._AGVs["AMR"].move(-5.0)
            await self._simu_timer.sleep(1.0)

            fork_action = ArticulationAction(joint_velocities=[-0.1],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)
            fork_action = ArticulationAction(joint_velocities=[0.1],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)
            await self._simu_timer.sleep(2.0)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['P'].get_dof_index("fork2")])
            self._AGVs['P'].apply_action(fork_action)
            fork_action = ArticulationAction(joint_velocities=[0.0],
                                            joint_indices=[self._AGVs['E'].get_dof_index("fork_z")])
            self._AGVs['E'].apply_action(fork_action)

            self._AGVs['P'].move(3.52)
            self._AGVs['E'].move(-3.3)
            await self._simu_timer.sleep(4.8)
            self._AGVs['E'].move(0.0)
            self._AGVs['P'].move(0.0)

            await self._world.reset_async()


    def run(self):
        run_coroutine(self._task())

        self._simulate()
        simulation_app.close()


if __name__ == "__main__":
    config = load_config("configs/mvp_01.yaml")
    mvp = MVP(config)
    mvp.run()
