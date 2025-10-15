from service import SerialController
import ecal.core.core as ecal_core
import sys, time
import argparse
from utils import ConfigLoader

def main(config_path):

    loader = ConfigLoader(config_path)
    config = loader.get()["simulation_app"]

    if not ecal_core.is_initialized():  # True/False
        ecal_core.initialize(sys.argv, "AGVInterfaceServer")
    # publish
    service = SerialController()
    service.initialize(config["actuators_config"],
                       config["sensors_config"])

    while ecal_core.ok():
        time.sleep(0.001)

    # 关闭 eCAL
    ecal_core.finalize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Isaac Sim Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python launcher.py --config **.config",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../configs/e_test.yaml",  # Default to option 2 (Reverse Driving)
        help="config file path",
    )
    args = parser.parse_args()

    main(args.config)
