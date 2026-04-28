from __future__ import annotations

import argparse
import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from copilot.runtime.daemon import LocalDaemon


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Windows operator daemon.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"Copilot local daemon listening on http://{args.host}:{args.port}")
    LocalDaemon(host=args.host, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
