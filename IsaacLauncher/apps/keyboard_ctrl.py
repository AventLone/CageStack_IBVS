from service import KeyBoardPublisher
from utils import ConfigLoader
import argparse


def main(config_path):
    loader = ConfigLoader(config_path)
    config = loader.get()["vehicle"]

    ecal_pub = KeyBoardPublisher(config["vehicle_state_topic"],
                                 config["keyboard_topic"])
    ecal_pub.run()

    print("keyBoardPublisherProcess stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Isaac Sim Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python launcher.py --config **.config",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=
        "../configs/e_test.yaml",  # Default to option 2 (Reverse Driving)
        help="config file path",
    )
    args = parser.parse_args()

    main(args.config)
