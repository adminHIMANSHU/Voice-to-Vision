"""Benchmark the CPU-only decision pipeline without loading AI models."""

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.safety_evaluator import SafetyEvaluator
from app.core.spatial_engine import SpatialEngine


def benchmark(name, operation, iterations: int, samples: int) -> dict[str, float]:
    durations = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        durations.append(time.perf_counter() - started)

    elapsed = statistics.median(durations)
    return {
        "name": name,
        "median_seconds": elapsed,
        "microseconds_per_operation": elapsed / iterations * 1_000_000,
        "operations_per_second": iterations / elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    bbox = (400, 150, 880, 650)
    spatial_engine = SpatialEngine()
    evaluator = SafetyEvaluator()
    detection = {
        "track_id": 1,
        "name": "person",
        "confidence": 0.92,
        "direction": "CENTER",
        "distance": "NEAR",
        "motion_state": "APPROACHING",
        "bbox": bbox,
    }

    results = [
        benchmark(
            "spatial_engine",
            lambda: spatial_engine.evaluate_spatial_attributes(bbox, 1280, 720, 1),
            args.iterations,
            args.samples,
        ),
        benchmark(
            "safety_evaluator",
            lambda: evaluator.evaluate_object(detection),
            args.iterations,
            args.samples,
        ),
    ]

    print("VoiceToVision core benchmark")
    print("=" * 28)
    for result in results:
        print(
            f"{result['name']}: {result['microseconds_per_operation']:.2f} us/op, "
            f"{result['operations_per_second']:.0f} ops/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())