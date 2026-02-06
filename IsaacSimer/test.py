from isaacsim.simulation_app import SimulationApp

APP_CONFIG = {"renderer": "RaytracedLighting", "headless": False}

# Example ROS2 bridge sample demonstrating the manual loading of stages and manual publishing of images
simulation_app = SimulationApp(APP_CONFIG)

from isaacsim.core.utils import extensions, stage
from isaacsim.core.api import SimulationContext

from isaacsim.core.utils import stage

# enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)
stage.open_stage("/home/avent/Desktop/IsaacAssets/Worlds/warehouse_trailer.usd")
simulation_app.update()
simulation_context.initialize_physics()


from omni_graph import og, camera_publish_graph, joint_states_graph

# Run the ROS Camera graph once to generate ROS image publishers in SDGPipeline
og.Controller.evaluate_sync(camera_publish_graph)
og.Controller.evaluate_sync(joint_states_graph)
simulation_app.update()

# Need to initialize physics getting any articulation..etc
simulation_context.play()



while simulation_app.is_running() and simulation_context.is_playing():
    simulation_context.step(render=True)   # Run with a fixed step size

simulation_context.stop()
simulation_app.close()
