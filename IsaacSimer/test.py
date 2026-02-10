from isaacsim.simulation_app import SimulationApp

APP_CONFIG = {"renderer": "RaytracedLighting", "headless": False}
simulation_app = SimulationApp(APP_CONFIG)

from isaacsim.core.utils import extensions, stage
from isaacsim.core.api import SimulationContext

extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

simulation_context = SimulationContext(stage_units_in_meters=1.0)
stage.open_stage("/home/avent/Desktop/IsaacAssets/Worlds/warehouse_trailer.usd")
simulation_app.update()
simulation_context.initialize_physics()   # Need to initialize physics getting any articulation..etc

from omni_graph import og, camera_publish_graph, joint_states_graph
# Run the ROS Camera graph once to generate ROS image publishers in SDGPipeline
og.Controller.evaluate_sync(camera_publish_graph)
og.Controller.evaluate_sync(joint_states_graph)
simulation_app.update()


simulation_context.play()
require_reset = False
while simulation_app.is_running():
    simulation_context.step(render=True)   # Run with a fixed step size
    if simulation_context.is_stopped() and not require_reset:
        require_reset = True
    if simulation_context.is_playing():
        if require_reset:
            simulation_context.reset()
            require_reset = False

simulation_context.stop()
simulation_app.close()

