#!/usr/bin/env python3
"""Evaluation harness for Sensitive Content Detection challenge.

Calls the candidate's API with a blind test set and computes metrics.

Usage:
    python run_eval.py --url http://<candidate-vm>:8000 --data test_data.json
    python run_eval.py --url http://<candidate-vm>:8000 --data test_data.json --output results.json
"""

import argparse
import json
import time
import sys
from pathlib import Path

import requests


def call_detect(url: str, text: str, timeout: float = 5.0) -> dict:
    """Call the candidate's /detect endpoint."""
    start = time.time()
    try:
        resp = requests.post(
            f"{url}/detect",
            json={"text": text},
            timeout=timeout,
        )
        latency = time.time() - start
        resp.raise_for_status()
        result = resp.json()
        result["latency_ms"] = latency * 1000
        result["error"] = None
        return result
    except Exception as e:
        return {
            "has_sensitive_content": False,
            "confidence": 0.0,
            "latency_ms": (time.time() - start) * 1000,
            "error": str(e),
        }


def compute_metrics(results: list) -> dict:
    """Compute precision, recall, F1, FP rate, and latency stats."""
    tp = fp = tn = fn = 0
    errors = 0
    latencies = []

    for r in results:
        if r["error"]:
            errors += 1
            continue

        expected = r["expected"]
        predicted = r["predicted"]
        latencies.append(r["latency_ms"])

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

    latencies.sort()
    p95_idx = int(len(latencies) * 0.95) if latencies else 0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_rate": round(fp_rate, 4),
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(latencies[p95_idx], 1) if latencies else 0,
        "max_latency_ms": round(max(latencies), 1) if latencies else 0,
        "total_samples": len(results),
        "errors": errors,
    }


def print_scorecard(metrics: dict, candidate: str = ""):
    """Print a formatted scorecard."""
    title = f"Scorecard{f' — {candidate}' if candidate else ''}"
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  F1 Score:       {metrics['f1'] * 100:6.1f}%")
    print(f"  Precision:      {metrics['precision'] * 100:6.1f}%")
    print(f"  Recall:         {metrics['recall'] * 100:6.1f}%")
    print(f"  FP Rate:        {metrics['fp_rate'] * 100:6.1f}%")
    print(f"  Accuracy:       {metrics['accuracy'] * 100:6.1f}%")
    print(f"  {'─' * 46}")
    print(f"  Avg Latency:    {metrics['avg_latency_ms']:6.1f} ms")
    print(f"  P95 Latency:    {metrics['p95_latency_ms']:6.1f} ms")
    print(f"  Max Latency:    {metrics['max_latency_ms']:6.1f} ms")
    print(f"  {'─' * 46}")
    print(f"  Confusion:      TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"                  FN={metrics['fn']}  TN={metrics['tn']}")
    if metrics["errors"] > 0:
        print(f"  Errors:         {metrics['errors']}")
    print(f"  Total Samples:  {metrics['total_samples']}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate sensitive content detection API")
    parser.add_argument("--url", required=True, help="Candidate API base URL (e.g., http://10.0.0.1:8000)")
    parser.add_argument("--data", required=True, help="Path to test data JSON file")
    parser.add_argument("--output", help="Save detailed results to JSON file")
    parser.add_argument("--candidate", default="", help="Candidate name (for scorecard label)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    # Load test data
    with open(args.data) as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test cases from {args.data}")
    print(f"Calling {args.url}/detect ...")

    # Health check
    try:
        health = requests.get(f"{args.url}/health", timeout=5)
        print(f"Health check: {health.status_code}")
    except Exception as e:
        print(f"Health check failed: {e}")
        print("Proceeding anyway...")

    # Run evaluation
    results = []
    for i, tc in enumerate(test_data):
        text = tc["text"]
        expected = tc["has_sensitive_content"]

        r = call_detect(args.url, text, timeout=args.timeout)
        r["expected"] = expected
        r["predicted"] = r.get("has_sensitive_content", False)
        r["text_preview"] = text[:80]
        r["category"] = tc.get("category", "unknown")
        results.append(r)

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
            print(f"  [{i + 1}/{len(test_data)}] ...", end="\r")

    print()

    # Compute and display metrics
    metrics = compute_metrics(results)
    print_scorecard(metrics, candidate=args.candidate)

    # Show false negatives and false positives
    fns = [r for r in results if r["expected"] and not r["predicted"] and not r["error"]]
    fps = [r for r in results if not r["expected"] and r["predicted"] and not r["error"]]

    if fns:
        print(f"\nMissed secrets ({len(fns)}):")
        for r in fns[:10]:
            print(f"  conf={r['confidence']:.3f} | {r['text_preview']}")

    if fps:
        print(f"\nFalse alarms ({len(fps)}):")
        for r in fps[:10]:
            print(f"  conf={r['confidence']:.3f} | {r['text_preview']}")

    # Save detailed results
    if args.output:
        output = {
            "candidate": args.candidate,
            "url": args.url,
            "metrics": metrics,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
