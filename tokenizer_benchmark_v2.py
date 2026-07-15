"""
Extended tokenizer benchmark:
- Multi-worker parallelism (real serving simulation)
- More modern tokenizers (Qwen2, DeepSeek)
- CPU saturation analysis
"""

import gc
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("plots", exist_ok=True)

BASE_TEXT_MEDIUM = (
    "In a long technical discussion about artificial intelligence, researchers "
    "analyze how language models process information, how attention mechanisms "
    "work, and how memory systems affect inference performance. They compare "
    "latency, throughput, and quality across different serving architectures."
)

TEXTS = {
    "medium": BASE_TEXT_MEDIUM,
    "long": (BASE_TEXT_MEDIUM + " ") * 20,
}

N_WORKERS_LIST = [1, 2, 4, 8, 16]
N_ITEMS_PER_WORKER = 500
N_REPEATS = 3


def load_tokenizers():
    tokenizers = {}

    try:
        from transformers import AutoTokenizer
        tokenizers["gpt2"] = {
            "type": "hf",
            "tok": AutoTokenizer.from_pretrained("gpt2"),
            "family": "BPE (GPT-2)",
            "model_id": "gpt2",
        }
        print("[ok] gpt2")
    except Exception as e:
        print(f"[skip] gpt2: {e}")

    try:
        import tiktoken
        for model_name in ["gpt-3.5-turbo", "gpt-4o"]:
            try:
                enc = tiktoken.encoding_for_model(model_name)
                tokenizers[f"tiktoken-{model_name}"] = {
                    "type": "tiktoken",
                    "tok": enc,
                    "family": f"Tiktoken ({model_name})",
                    "model_id": model_name,
                }
                print(f"[ok] tiktoken-{model_name}")
            except Exception as e:
                print(f"[skip] tiktoken-{model_name}: {e}")
    except Exception as e:
        print(f"[skip] tiktoken: {e}")

    try:
        from transformers import AutoTokenizer
        tokenizers["llama2"] = {
            "type": "hf",
            "tok": AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer"),
            "family": "SentencePiece (LLaMA)",
            "model_id": "hf-internal-testing/llama-tokenizer",
        }
        print("[ok] llama2")
    except Exception as e:
        print(f"[skip] llama2: {e}")

    # Modern tokenizers
    try:
        from transformers import AutoTokenizer
        tokenizers["qwen2"] = {
            "type": "hf",
            "tok": AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B"),
            "family": "BPE (Qwen2)",
            "model_id": "Qwen/Qwen2-0.5B",
        }
        print("[ok] qwen2")
    except Exception as e:
        print(f"[skip] qwen2: {e}")

    return tokenizers


def encode(tok_info, texts):
    if tok_info["type"] == "hf":
        return tok_info["tok"](texts, padding=False, truncation=False)["input_ids"]
    elif tok_info["type"] == "tiktoken":
        if isinstance(texts, str):
            return tok_info["tok"].encode(texts)
        return [tok_info["tok"].encode(t) for t in texts]


def worker_task(args):
    """Worker function for multiprocessing."""
    tokenizer_type, model_id, text, n_items = args

    if tokenizer_type == "hf":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        def encode_fn(t):
            return tok(t, padding=False, truncation=False)["input_ids"]
    elif tokenizer_type == "tiktoken":
        import tiktoken
        enc = tiktoken.encoding_for_model(model_id)
        def encode_fn(t):
            return enc.encode(t)

    t0 = time.perf_counter()
    for _ in range(n_items):
        _ = encode_fn(text)
    elapsed = time.perf_counter() - t0

    return elapsed


def benchmark_multiprocess(tok_info, text, n_workers, n_items_per_worker, repeats=N_REPEATS):
    """Simulate n_workers concurrent processes each tokenizing n_items."""
    args_list = [(tok_info["type"], tok_info["model_id"], text, n_items_per_worker)] * n_workers

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            worker_times = list(executor.map(worker_task, args_list))
        total_elapsed = time.perf_counter() - t0
        times.append(total_elapsed)

    return {
        "wall_median_s": float(np.median(times)),
        "wall_min_s": float(np.min(times)),
    }


def benchmark_thread(tok_info, text, n_workers, n_items_per_worker, repeats=N_REPEATS):
    """Simulate n_workers concurrent threads (subject to GIL)."""
    def task():
        for _ in range(n_items_per_worker):
            _ = encode(tok_info, text)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(task) for _ in range(n_workers)]
            for f in futures:
                f.result()
        total_elapsed = time.perf_counter() - t0
        times.append(total_elapsed)

    return {
        "wall_median_s": float(np.median(times)),
        "wall_min_s": float(np.min(times)),
    }


def main():
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_workers": N_WORKERS_LIST,
        "n_items_per_worker": N_ITEMS_PER_WORKER,
        "n_repeats": N_REPEATS,
        "cpu_count": os.cpu_count(),
    }
    print(json.dumps(metadata, indent=2))

    print("\n=== Loading tokenizers ===")
    tokenizers = load_tokenizers()

    # Get token counts
    token_counts = {}
    for text_name, text in TEXTS.items():
        token_counts[text_name] = {}
        for tok_name, tok_info in tokenizers.items():
            ids = encode(tok_info, text)
            token_counts[text_name][tok_name] = len(ids)

    rows = []

    for text_name, text in TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Text: {text_name}")
        print(f"{'='*60}")

        for tok_name, tok_info in tokenizers.items():
            n_tokens = token_counts[text_name][tok_name]
            print(f"\n  {tok_name} ({n_tokens} tok/text)")

            for n_workers in N_WORKERS_LIST:
                # Thread pool (GIL)
                try:
                    r_thread = benchmark_thread(tok_info, text, n_workers, N_ITEMS_PER_WORKER)
                    total_tokens = n_workers * N_ITEMS_PER_WORKER * n_tokens
                    thread_tput = total_tokens / r_thread["wall_median_s"]
                except Exception as e:
                    print(f"    workers={n_workers} thread FAIL: {e}")
                    continue

                # Process pool (true parallelism)
                try:
                    r_proc = benchmark_multiprocess(tok_info, text, n_workers, N_ITEMS_PER_WORKER)
                    total_tokens = n_workers * N_ITEMS_PER_WORKER * n_tokens
                    proc_tput = total_tokens / r_proc["wall_median_s"]
                except Exception as e:
                    print(f"    workers={n_workers} process FAIL: {e}")
                    proc_tput = None

                rows.append({
                    "text_length": text_name,
                    "tokenizer": tok_name,
                    "family": tok_info["family"],
                    "n_tokens_per_text": n_tokens,
                    "n_workers": n_workers,
                    "n_items_per_worker": N_ITEMS_PER_WORKER,
                    "thread_wall_s": r_thread["wall_median_s"],
                    "process_wall_s": r_proc["wall_median_s"] if proc_tput else None,
                    "thread_tokens_per_s": thread_tput,
                    "process_tokens_per_s": proc_tput,
                    "process_vs_thread_ratio": proc_tput / thread_tput if proc_tput else None,
                })

                proc_str = f"{proc_tput:>10.0f}" if proc_tput else "     FAIL"
                ratio_str = f"{proc_tput / thread_tput:.2f}x" if proc_tput else "N/A"
                print(
                    f"    workers={n_workers:2d}  "
                    f"thread: {thread_tput:>10.0f} tok/s  "
                    f"process: {proc_str} tok/s  "
                    f"process/thread: {ratio_str}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "concurrent_benchmark.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "concurrent_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY (medium text)")
    print(f"{'='*60}")
    sub = df[df["text_length"] == "medium"]
    for tok in sub["tokenizer"].unique():
        s = sub[sub["tokenizer"] == tok].sort_values("n_workers")
        print(f"\n{tok}:")
        for _, r in s.iterrows():
            print(
                f"  workers={int(r['n_workers']):2d}: "
                f"thread={r['thread_tokens_per_s']:>10.0f}  "
                f"process={r['process_tokens_per_s'] or 0:>10.0f}"
            )


if __name__ == "__main__":
    main()
