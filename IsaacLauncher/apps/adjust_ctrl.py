from service import AdjustControlPublisher
from utils import ConfigLoader
import ecal.core.core as ecal_core
import time
import sys
import argparse


def main(config_path):
    loader = ConfigLoader(config_path)
    config = loader.get()["vehicle"]

    if not ecal_core.is_initialized():  # True/False
        ecal_core.initialize(sys.argv, "AdjustControlServer")

    ecal_pub = AdjustControlPublisher(config["adjust_ctrl_topic"], config["vehicle_state_topic"])
    ecal_pub.start()

    while ecal_core.ok():
        time.sleep(0.001)

    print("AdjustControlPublisher stop")

    ecal_pub.join()
    # 关闭 eCAL
    if ecal_core.is_initialized():
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