"""
Rigorous batch latency tests: hit /detect/batch with varying batch sizes and record response times.

Uses test data from scaffold/sample_data.json and datasets/test.json. Run with the API up:

  python -m scaffold.batch_latency_test [--base-url http://localhost:8000] [--runs 5]

Comparing ONNX vs PyTorch:
  1. Start server normally (uses ONNX if model.onnx exists), run this script, note the output.
  2. Stop server, start with PyTorch only:  FORCE_PYTORCH=1 uvicorn scaffold.server:app --host 0.0.0.0 --port 8000
  3. Run this script again and compare mean/p95 latency.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    import requests
    _USE_REQUESTS = True
except ImportError:
    _USE_REQUESTS = False

import urllib.request
import urllib.error


# Repo root: parent of scaffold/
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA = REPO_ROOT / "scaffold" / "sample_data.json"
DATASETS_TEST = REPO_ROOT / "datasets" / "test.json"
EVAL_TEST_DATA = REPO_ROOT / "eval" / "test_data.json"

# Batch sizes to test (powers of 2 and a few in between)
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_RUNS = 5
WARMUP_RUNS = 2


def load_texts(max_texts: int = 512) -> list[str]:
    """Load 'text' fields from sample_data.json and datasets/test.json up to max_texts."""
    texts: list[str] = []
    for path in (SAMPLE_DATA, DATASETS_TEST, EVAL_TEST_DATA):
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skip {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if isinstance(item, dict) and "text" in item:
                t = item["text"]
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
                    if len(texts) >= max_texts:
                        return texts
    if not texts:
        raise FileNotFoundError(
            f"No texts found in {SAMPLE_DATA}, {DATASETS_TEST}, or {EVAL_TEST_DATA}"
        )
    return texts


def run_batch(url: str, texts: list[str], timeout: float = 60.0) -> tuple[float, int]:
    """POST to /detect/batch; return (elapsed_seconds, status_code)."""
    body = json.dumps({"texts": texts}).encode("utf-8")
    start = time.perf_counter()
    try:
        if _USE_REQUESTS:
            r = requests.post(
                url,
                json={"texts": texts},
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            status = r.status_code
        else:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception:
        status = -1
    elapsed = time.perf_counter() - start
    return elapsed, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch API latency test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of timed runs per batch size (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=max(BATCH_SIZES) * 2,
        help="Max texts to load from test data",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    batch_url = f"{args.base_url.rstrip('/')}/detect/batch"
    print(f"Loading test data (max {args.max_texts} texts)...")
    texts = load_texts(max_texts=args.max_texts)
    print(f"Loaded {len(texts)} texts from {SAMPLE_DATA.name}, etc.\n")

    # Only test batch sizes we have enough data for
    sizes = [s for s in BATCH_SIZES if s <= len(texts)]
    if not sizes:
        print("Not enough texts for any batch size.", file=sys.stderr)
        return 1

    print(f"Batch sizes to test: {sizes}")
    print(f"Warmup runs: {WARMUP_RUNS}, timed runs: {args.runs}\n")
    print("-" * 80)

    results: list[dict] = []

    for batch_size in sizes:
        batch = texts[:batch_size]
        # Warmup
        for _ in range(WARMUP_RUNS):
            run_batch(batch_url, batch, timeout=args.timeout)

        # Timed runs
        times_sec: list[float] = []
        for i in range(args.runs):
            elapsed, status = run_batch(batch_url, batch, timeout=args.timeout)
            if status != 200:
                print(f"  batch_size={batch_size} run {i+1}: HTTP {status}", file=sys.stderr)
            times_sec.append(elapsed)

        times_ms = [t * 1000 for t in times_sec]
        mean_ms = statistics.mean(times_ms)
        per_item_ms = mean_ms / batch_size
        stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        p95_ms = sorted(times_ms)[int(0.95 * len(times_ms))] if times_ms else 0

        row = {
            "batch_size": batch_size,
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(stdev_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "per_item_ms": round(per_item_ms, 2),
        }
        results.append(row)
        print(
            f"batch_size={batch_size:4d}  mean={mean_ms:8.2f} ms  std={stdev_ms:6.2f} ms  "
            f"p95={p95_ms:8.2f} ms  per_item={per_item_ms:6.2f} ms"
        )

    print("-" * 80)
    print("\nSummary (mean latency vs batch size):")
    for r in results:
        print(f"  {r['batch_size']:4d}  ->  {r['mean_ms']:8.2f} ms  ({r['per_item_ms']:6.2f} ms/item)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
