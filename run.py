"""Entry point. Launches the Gradio UI with a shareable public link."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `pipeline` and `ui` importable when run as a script from any cwd.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ui.app import launch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="ZimPrep exam paper extractor.")
    parser.add_argument("--no-share", action="store_true", help="Do not create a public share link.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    launch(share=not args.no_share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
