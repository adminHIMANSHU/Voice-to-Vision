"""Check whether the local machine can run VoiceToVision."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.diagnostics import format_preflight, run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="also probe the configured camera")
    args = parser.parse_args()

    results = run_preflight(include_camera=args.camera)
    print(format_preflight(results))
    return 0 if all(result.ok for result in results if result.required) else 1


if __name__ == "__main__":
    raise SystemExit(main())