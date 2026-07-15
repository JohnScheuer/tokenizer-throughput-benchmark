"""
Saturation test: find max sustainable throughput for each SLO.

For each tokenizer and each SLO (1ms, 5ms, 10ms p99),
find the highest RPS that still meets the SLO.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEXT = (
    "In a long technical discussion about artificial intelligence, researchers "
    "analyze how language models process information."
)

# Test at multiple offered loads (RPS)
RPS_LEVELS = [100, 500, 1000, 2500, 5000, 10000, 25000, 50000]
DURATION_S = 3
SLO_TARGETS = [1.0, 5.0, 10.0, 50.0]


def load_tokenizers():
    tokenizers = {}

    from transformers import AutoTokenizer
    tokenizers["gpt2"] = {
        "type": "hf",
        "tok": AutoTokenizer.from_pretrained("gpt2"),
    }
    tokenizers["llama2"] = {
        "type": "hf",
        "tok": AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer"),
    }

    import tiktoken
    tokenizers["tiktoken-gpt-3.5"] = {
        "type": "tiktoken",
        "tok": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    }
    tokenizers["tiktoken-gpt-4o"] = {
        "type": "tiktoken",
        "tok": tiktoken.encoding_for_model("gpt-4o"),
    }

    return tokenizers


def tokenize_one(tok_info, text):
    if tok_info["type"] == "hf":
        return tok_info["tok"](text, padding=False, truncation=False)["input_ids"]
    elif tok_info["type"] == "tiktoken":
        return tok_info["tok"].encode(text)


def load_test(tok_info, text, target_rps, duration_s, n_workers=8):
    """
    Send requests at target_rps for duration_s.
    Return latency distribution and actual RPS achieved.
    """
    inter_arrival = 1.0 / target_rps
    latencies = []

    def worker():
        t0 = time.perf_counter()
        _ = tokenize_one(tok_info, text)
        return (time.perf_counter() - t0) * 1000.0

    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures = []

    start = time.perf_counter()
    next_send = start
    sent = 0

    try:
        while time.perf_counter() - start < duration_s:
            now = time.perf_counter()
            if now >= next_send:
                futures.append(executor.submit(worker))
                sent += 1
                next_send += inter_arrival
            else:
                time.sleep(min(inter_arrival * 0.5, next_send - now))
    except Exception:
        pass

    # Collect all results
    for f in futures:
        try:
            latencies.append(f.result(timeout=30))
        except Exception:
            pass

    executor.shutdown(wait=False)

    actual_duration = time.perf_counter() - start
    actual_rps = len(latencies) / actual_duration if actual_duration > 0 else 0

    return {
        "requests_sent": sent,
        "requests_completed": len(latencies),
        "actual_rps": actual_rps,
        "latencies": latencies,
    }


def main():
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rps_levels": RPS_LEVELS,
        "duration_s": DURATION_S,
        "slo_targets_ms": SLO_TARGETS,
    }
    print(json.dumps(metadata, indent=2))

    tokenizers = load_tokenizers()

    rows = []

    for tok_name, tok_info in tokenizers.items():
        print(f"\n{'='*60}")
        print(f"Tokenizer: {tok_name}")
        print(f"{'='*60}")

        for target_rps in RPS_LEVELS:
            result = load_test(tok_info, TEXT, target_rps, DURATION_S)

            if not result["latencies"]:
                continue

            lats = np.array(result["latencies"])
            p50 = np.percentile(lats, 50)
            p95 = np.percentile(lats, 95)
            p99 = np.percentile(lats, 99)
            p999 = np.percentile(lats, 99.9)

            rows.append({
                "tokenizer": tok_name,
                "target_rps": target_rps,
                "actual_rps": result["actual_rps"],
                "requests_completed": result["requests_completed"],
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "p999_ms": p999,
                "mean_ms": np.mean(lats),
            })

            print(
                f"  target_rps={target_rps:6d}  actual={result['actual_rps']:>6.0f}  "
                f"p50={p50:.2f}  p95={p95:.2f}  p99={p99:.2f}  p999={p999:.2f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "saturation_test.csv"), index=False)

    print(f"\n{'='*60}")
    print("MAX SUSTAINABLE RPS PER SLO")
    print(f"{'='*60}")

    for slo in SLO_TARGETS:
        print(f"\np99 < {slo}ms:")
        for tok in df["tokenizer"].unique():
            sub = df[(df["tokenizer"] == tok) & (df["p99_ms"] < slo)]
            if len(sub) > 0:
                max_rps = sub["actual_rps"].max()
                print(f"  {tok:30s}: {max_rps:>7.0f} RPS")
            else:
                print(f"  {tok:30s}: EXCEEDS SLO EVEN AT LOWEST RPS")

    print(f"\nSaved: {RESULTS_DIR}/saturation_test.csv")


if __name__ == "__main__":
    main()
