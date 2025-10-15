from utils import ConfigLoader, SysTimer
from isaacsim import SimulationApp
import argparse
import time


class SimuExcutor:
    def __init__(self, yaml_file: str):
        self._config = ConfigLoader(yaml_file).load()
        simulate_config = self._config["simulation_app"]

        self._simulation_app = SimulationApp(simulate_config["config"])

        from isaacsim.core.utils.stage import open_stage
        open_stage(simulate_config["stage_file_path"])

        from isaacsim.core.api.world import World
        self._world = World(physics_dt=simulate_config["physics_dt"],       # Physics time step
                            rendering_dt=simulate_config["rendering_dt"],   # Render timestep
                            stage_units_in_meters=1.0)
        self._require_reset = False
        self._world.reset()

        import controllers
        ControlT = getattr(controllers, simulate_config["controller_class"])
        self._controller = ControlT(self._world, self._config)

    def excute(self):

        while self._simulation_app.is_running():
            start = time.perf_counter()

            self._world.step(render=True)
            # 在每一步仿真中调用控制器的run方法
            self._controller.step()

            if self._world.is_stopped() and not self._require_reset:   # 播放/暂停与重置逻辑
                self._require_reset = True

            if self._world.is_playing():
                if self._require_reset:
                    self._world.reset()
                    self._require_reset = False

            end = time.perf_counter()
            if end - start > 0.01:
                # print("WARN : simulation step period is more than 10ms !!! time is ", end - start)
                pass
        self._simulation_app.close()

def main():
    parser = argparse.ArgumentParser(
        description="Isaac Sim Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python launcher.py --config **.config",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../configs/st_test.yaml",  # Default to option 2 (Reverse Driving)
        help="config file path",
    )

    args = parser.parse_args()

    SysTimer.initialize()
    print(f"SysTimer timestamp is {SysTimer.get_timestamp()}")
    print(f"SysTimer time seqnum is {SysTimer.get_time_seqnum()}")

    simu_excutor = SimuExcutor(args.config)
    simu_excutor.excute()

if __name__ == "__main__":
    main()
