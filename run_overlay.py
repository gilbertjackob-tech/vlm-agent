import argparse

from copilot.ui.overlay import launch_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the operator overlay connected to the local daemon.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765", help="Daemon base URL.")
    parser.add_argument("--run-id", default="", help="Existing daemon run id to follow.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_overlay(base_url=args.base_url, run_id=args.run_id)
