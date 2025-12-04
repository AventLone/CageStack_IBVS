from isaacsim import SimulationApp
import argparse
from tools import *
import asyncio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=False,
        default="/zclin/test/IsaacSimLauncher/sdg/obj_stage_sdg/config.json",
        help="Include specific config parameters (json or yaml))",
    )
    parser.add_argument("--gui", required=False, default=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.gui:
        cfg["launch_config"] = {"renderer": "RaytracedLighting", "headless": False}

    simulation_app = SimulationApp(cfg["launch_config"])
    from core import ObjectStageGenerator
    from isaacsim.core.api import World

    world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)

    generator = ObjectStageGenerator(cfg)

    warehouse1 = generator.create()
    warehouse2 = generator.create()
    warehouse3 = generator.create()
    warehouse4 = generator.create()

    world.reset()

    for i in range(20):
        world.step(render=False)

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
