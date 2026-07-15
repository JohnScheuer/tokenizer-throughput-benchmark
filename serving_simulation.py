"""
Real serving simulation:
- Requests arrive over time
- Measure per-request tokenization latency
- p50, p95, p99 latency distributions
- How many concurrent requests can we handle at fixed latency budget?
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

BASE_TEXT_MEDIUM = (
    "In a long technical discussion about artificial intelligence, researchers "
    "analyze how language models process information, how attention mechanisms "
    "work, and how memory systems affect inference performance."
)

N_REQUESTS = 1000
CONCURRENCY_LEVELS = [1, 4, 16, 64]


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


def simulate_serving(tok_info, text, n_requests, concurrency):
    """
    Simulate n_requests arriving at random times, processed by `concurrency` threads.
    Returns per-request latency in ms.
    """
    latencies = []
    lock = None

    def task(_):
        t0 = time.perf_counter()
        _ = tokenize_one(tok_info, text)
        return (time.perf_counter() - t0) * 1000.0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(task, range(n_requests)))

    return results


def main():
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_requests": N_REQUESTS,
        "concurrency_levels": CONCURRENCY_LEVELS,
    }
    print(json.dumps(metadata, indent=2))

    tokenizers = load_tokenizers()

    rows = []

    for tok_name, tok_info in tokenizers.items():
        print(f"\n{'='*60}")
        print(f"Tokenizer: {tok_name}")
        print(f"{'='*60}")

        for conc in CONCURRENCY_LEVELS:
            print(f"\n  concurrency={conc}")
            latencies = simulate_serving(tok_info, BASE_TEXT_MEDIUM, N_REQUESTS, conc)

            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            mean_lat = np.mean(latencies)
            max_lat = np.max(latencies)

            throughput = N_REQUESTS / (sum(latencies) / 1000.0 / conc)

            rows.append({
                "tokenizer": tok_name,
                "concurrency": conc,
                "n_requests": N_REQUESTS,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "mean_ms": mean_lat,
                "max_ms": max_lat,
                "throughput_req_s": throughput,
            })

            print(f"    p50={p50:.3f}ms  p95={p95:.3f}ms  p99={p99:.3f}ms  mean={mean_lat:.3f}ms  max={max_lat:.3f}ms")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "serving_simulation.csv"), index=False)

    # Practical serving question: how many concurrent requests can we handle at 1ms p99?
    print(f"\n{'='*60}")
    print("PRACTICAL BUDGET: max concurrency to stay under 1ms p99")
    print(f"{'='*60}")

    for tok in df["tokenizer"].unique():
        sub = df[df["tokenizer"] == tok].sort_values("concurrency")
        under_1ms = sub[sub["p99_ms"] < 1.0]
        if len(under_1ms) > 0:
            max_conc = under_1ms["concurrency"].max()
            print(f"  {tok:30s}: max {max_conc} concurrent (p99={under_1ms[under_1ms['concurrency']==max_conc]['p99_ms'].values[0]:.3f}ms)")
        else:
            best = sub.iloc[0]
            print(f"  {tok:30s}: EXCEEDS 1ms even at conc=1 (p99={best['p99_ms']:.3f}ms)")

    print(f"\n{'='*60}")
    print("PRACTICAL BUDGET: max concurrency to stay under 10ms p99")
    print(f"{'='*60}")

    for tok in df["tokenizer"].unique():
        sub = df[df["tokenizer"] == tok].sort_values("concurrency")
        under_10ms = sub[sub["p99_ms"] < 10.0]
        if len(under_10ms) > 0:
            max_conc = under_10ms["concurrency"].max()
            print(f"  {tok:30s}: max {max_conc} concurrent")
        else:
            print(f"  {tok:30s}: EXCEEDS 10ms")

    print(f"\nSaved: {RESULTS_DIR}/serving_simulation.csv")


if __name__ == "__main__":
    main()
