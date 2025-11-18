# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Standalone script to run an actor sdg job

"""

import argparse
import asyncio
import os
import sys

import numpy as np
from isaacsim import SimulationApp


class ActorSDG:
    def __init__(self, sim_app, config_file_path, no_random_commands):
        self._sim_app = sim_app
        # Inputs
        self.config_file_path = config_file_path
        self.no_random_commands = no_random_commands

        self._sim_manager = None
        self._setup_sim_sub = None
        self._setup_sim_succeed = False

    async def run(self):
        # Enable all required extensions
        from isaacsim.replicator.agent.core.simulation import SimulationManager
        self._enable_extensions()
        await self._sim_app.app.next_update_async()
        
        # Init SimulatonManager
        self._sim_manager = SimulationManager()

        try:
            can_load_config = self._sim_manager.load_config_file(self.config_file_path)
            if not can_load_config:
                return False

            # Set up sim
            await self._setup_sim()

            # [Optional] Generate random commands
            if not self.no_random_commands:
                await self._gen_random_commands()

            # Wait for data generation callback
            await self._sim_manager.run_data_generation_async(will_wait_until_complete=True)
            return True
        except Exception as e:
            import carb

            carb.log_error(f"Failed to load config file {e}")
            return False

    def _enable_extensions(self):
        import carb
        import omni.replicator.core as rep
        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()

        ext_manager.set_extension_enabled_immediate("omni.kit.viewport.window", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.manipulator.prim", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.property.usd", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.scripting", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.timeline", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.graph.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.retarget.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.core", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.navigation.meshtools", True)
        ext_manager.set_extension_enabled_immediate("omni.anim.people", True)
        ext_manager.set_extension_enabled_immediate("isaacsim.replicator.agent.core", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.mesh.raycast", True)
        ext_manager.set_extension_enabled_immediate("omni.physx.graph", True)  # For Conveyor Belt

    async def _setup_sim(self):
        def done_callback(e):
            self._setup_sim_succeed = True
            self._setup_sim_sub = None

        # Set up simulation and start data generation
        self._setup_sim_sub = self._sim_manager.register_set_up_simulation_done_callback(done_callback)
        # self._sim_manager.set_up_simulation_from_config_file()
        # self._sim_manager.load_assets_to_scene()
        self.load_assets_to_scene()

        while self._setup_sim_sub and not self._sim_app.is_exiting():
            await self._sim_app.app.next_update_async()

    async def _gen_random_commands(self):
        if self._sim_manager.get_config_file_valid_value("character", "command_file"):
            task = asyncio.create_task(self._sim_manager.generate_random_commands())
            await task
            commands = task.result()
            self._sim_manager.save_commands(commands)
    
    
    def load_assets_to_scene(self):
        """
        Trigger navemsh baking and load characters, robots and cameras.
        """
        async def try_bake_navmesh():
            import carb
            import omni.anim.navigation.core as nav
            # Load assets other than scene
            # It first makes sure nav mesh is ready, then load cameras and characters in navmesh ready callback
            _inav = nav.acquire_interface()
            _inav.start_navmesh_baking_and_wait()
            navmesh = _inav.get_navmesh()
            if navmesh is None:
                carb.log_error(
                    "NavMesh building failed. Please check whether the stage has a valid NavmeshVolume. "
                    "Will not load assets to scene."
                )
                return
            
            if self._sim_manager.get_config_file_section("robot"):
                self._sim_manager.load_robot_from_config_file()
                self._sim_manager.setup_anim_people_robot_command_from_config_file()
            else:
                carb.log_info("No robot section in the config file. Skip robot setup.")

            if self._sim_manager.get_config_file_section("character"):
                # self._sim_manager.load_characters_from_config_file()
                self._sim_manager.setup_all_characters()
                self._sim_manager.setup_anim_people_command_from_config_file()
            else:
                carb.log_info("No character section in the config file. Skip character setup.")

            if self._sim_manager.get_config_file_section("event"):
                self._sim_manager.setup_incidents_from_config_file()
            else:
                carb.log_info("No incident section in the config file. Skip incident setup.")

            response_section = self._sim_manager.get_config_file_section("response")
            if response_section:
                from isaacsim.replicator.agent.core.response.core import AgentResponseManager
                
                AgentResponseManager.get_instance().reset()
                AgentResponseManager.get_instance().setup_responses_from_config_file(response_section)
            else:
                carb.log_info("No response section in the config file. Skip agent response setup.")

            self._sim_manager.load_camera_from_config_file()
            self._sim_manager.load_lidar_from_config_file()
            # Mark complete
            carb.eventdispatcher.get_eventdispatcher().dispatch_event(
                event_name=self._sim_manager.SET_UP_SIMULATION_DONE_EVENT,
                payload={}
            )

        asyncio.ensure_future(try_bake_navmesh())
    


def main():
    # Read command line arguments
    config_file_path = "/home/visionnav/application/isaacSim/isaac-sim-standalone-5.0.0-linux-x86_64/tools/actor_sdg/default_config.yaml"
    no_random_commands = True


    print("Config file path: {}".format(config_file_path))
    print("Don't random commands: {}".format(no_random_commands))

    # Check files exist
    if not os.path.isfile(config_file_path):
        print("Invalid config file path. Exit.", file=sys.stderr)
        return
    
    BASE_EXP_PATH = os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.action_and_event_data_generation.base.kit")
    APP_CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1920, "height": 1080}
    sim_app = SimulationApp(launch_config=APP_CONFIG, experience=BASE_EXP_PATH)

    # Start SDG
    sdg = ActorSDG(
        sim_app,
        os.path.abspath(config_file_path),
        no_random_commands,
    )

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(sdg.run())
    try:
        while not task.done():
            sim_app.update()

        if not task.result():
            print("Failed to run SDG")

    # Close app
    finally:
        sim_app.update()


if __name__ == "__main__":
    main()
