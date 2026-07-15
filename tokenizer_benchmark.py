"""
tokenizer-throughput-benchmark

Measures tokenization throughput across:
- Different tokenizers (GPT-2, LLaMA, Qwen, Tiktoken)
- Sequential vs batch tokenization
- Various text lengths
- Detokenization performance
"""

import gc
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("plots", exist_ok=True)

BASE_TEXT_SHORT = "Hello, how are you today? This is a short text sample."

BASE_TEXT_MEDIUM = (
    "In a long technical discussion about artificial intelligence, researchers "
    "analyze how language models process information, how attention mechanisms "
    "work, and how memory systems affect inference performance. They compare "
    "latency, throughput, and quality across different serving architectures."
)

BASE_TEXT_LONG = (BASE_TEXT_MEDIUM + " ") * 20

TEXTS = {
    "short": BASE_TEXT_SHORT,
    "medium": BASE_TEXT_MEDIUM,
    "long": BASE_TEXT_LONG,
}

BATCH_SIZES = [1, 4, 16, 64, 256]
N_REPEATS = 5


def load_tokenizers():
    """Load available tokenizers, skipping any that fail."""
    tokenizers = {}

    # GPT-2 (BPE)
    try:
        from transformers import AutoTokenizer
        tokenizers["gpt2"] = {
            "type": "hf",
            "tok": AutoTokenizer.from_pretrained("gpt2"),
            "family": "BPE (GPT-2)",
        }
        print("[ok] gpt2 loaded")
    except Exception as e:
        print(f"[skip] gpt2: {e}")

    # GPT-2 medium (same BPE, different vocab size for consistency check)
    try:
        from transformers import AutoTokenizer
        tokenizers["gpt2-medium"] = {
            "type": "hf",
            "tok": AutoTokenizer.from_pretrained("gpt2-medium"),
            "family": "BPE (GPT-2)",
        }
        print("[ok] gpt2-medium loaded")
    except Exception as e:
        print(f"[skip] gpt2-medium: {e}")

    # Tiktoken (OpenAI's fast BPE)
    try:
        import tiktoken
        for model_name in ["gpt-3.5-turbo", "gpt-4o"]:
            try:
                enc = tiktoken.encoding_for_model(model_name)
                tokenizers[f"tiktoken-{model_name}"] = {
                    "type": "tiktoken",
                    "tok": enc,
                    "family": f"Tiktoken ({model_name})",
                }
                print(f"[ok] tiktoken-{model_name} loaded")
            except Exception as e:
                print(f"[skip] tiktoken-{model_name}: {e}")
    except Exception as e:
        print(f"[skip] tiktoken: {e}")

    # SentencePiece (LLaMA-style)
    try:
        from transformers import AutoTokenizer
        tokenizers["llama2"] = {
            "type": "hf",
            "tok": AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer"),
            "family": "SentencePiece (LLaMA)",
        }
        print("[ok] llama2 loaded")
    except Exception as e:
        print(f"[skip] llama2: {e}")

    return tokenizers


def encode(tok_info, texts):
    """Unified encode interface for different tokenizer backends."""
    if tok_info["type"] == "hf":
        return tok_info["tok"](texts, padding=False, truncation=False)["input_ids"]
    elif tok_info["type"] == "tiktoken":
        if isinstance(texts, str):
            return tok_info["tok"].encode(texts)
        return [tok_info["tok"].encode(t) for t in texts]


def decode(tok_info, ids_list):
    """Unified decode interface."""
    if tok_info["type"] == "hf":
        if isinstance(ids_list[0], list):
            return [tok_info["tok"].decode(ids) for ids in ids_list]
        return tok_info["tok"].decode(ids_list)
    elif tok_info["type"] == "tiktoken":
        if isinstance(ids_list[0], list):
            return [tok_info["tok"].decode(ids) for ids in ids_list]
        return tok_info["tok"].decode(ids_list)


def benchmark_sequential(tok_info, text, n_items, warmup=2, repeats=N_REPEATS):
    """Tokenize n_items copies of text one at a time."""
    for _ in range(warmup):
        for _ in range(n_items):
            _ = encode(tok_info, text)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n_items):
            _ = encode(tok_info, text)
        times.append(time.perf_counter() - t0)

    return {
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
    }


def benchmark_batch(tok_info, text, batch_size, warmup=2, repeats=N_REPEATS):
    """Tokenize a batch of texts in one call."""
    texts = [text] * batch_size

    for _ in range(warmup):
        _ = encode(tok_info, texts)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = encode(tok_info, texts)
        times.append(time.perf_counter() - t0)

    return {
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
    }


def benchmark_detokenize(tok_info, text, n_items, warmup=2, repeats=N_REPEATS):
    """Decode n_items token sequences."""
    ids = encode(tok_info, text)
    ids_list = [ids] * n_items

    for _ in range(warmup):
        _ = decode(tok_info, ids_list)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = decode(tok_info, ids_list)
        times.append(time.perf_counter() - t0)

    return {
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
    }


def main():
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_sizes": BATCH_SIZES,
        "text_lengths": list(TEXTS.keys()),
        "n_repeats": N_REPEATS,
    }
    print(json.dumps(metadata, indent=2))

    print("\n=== Loading tokenizers ===")
    tokenizers = load_tokenizers()

    print(f"\n=== Loaded {len(tokenizers)} tokenizers ===")
    for name in tokenizers:
        print(f"  - {name} ({tokenizers[name]['family']})")

    # Get token counts per text
    print("\n=== Token counts per text ===")
    token_counts = {}
    for text_name, text in TEXTS.items():
        token_counts[text_name] = {}
        for tok_name, tok_info in tokenizers.items():
            ids = encode(tok_info, text)
            token_counts[text_name][tok_name] = len(ids)
        print(f"  {text_name} (chars={len(text)}):")
        for tok_name, count in token_counts[text_name].items():
            print(f"    {tok_name:35s}: {count} tokens")

    rows = []

    for text_name, text in TEXTS.items():
        print(f"\n{'='*60}")
        print(f"Text: {text_name} ({len(text)} chars)")
        print(f"{'='*60}")

        for tok_name, tok_info in tokenizers.items():
            n_tokens = token_counts[text_name][tok_name]
            print(f"\n  Tokenizer: {tok_name} ({n_tokens} tokens/text)")

            for bs in BATCH_SIZES:
                # Sequential (one at a time)
                try:
                    r_seq = benchmark_sequential(tok_info, text, bs)
                    seq_tokens_per_s = (bs * n_tokens) / r_seq["median_s"]
                except Exception as e:
                    print(f"    bs={bs} sequential FAIL: {e}")
                    continue

                # Batch (all at once)
                try:
                    r_batch = benchmark_batch(tok_info, text, bs)
                    batch_tokens_per_s = (bs * n_tokens) / r_batch["median_s"]
                except Exception as e:
                    print(f"    bs={bs} batch FAIL: {e}")
                    continue

                speedup = r_seq["median_s"] / r_batch["median_s"] if r_batch["median_s"] > 0 else 0

                rows.append({
                    "text_length": text_name,
                    "tokenizer": tok_name,
                    "family": tok_info["family"],
                    "tokens_per_text": n_tokens,
                    "batch_size": bs,
                    "seq_median_s": r_seq["median_s"],
                    "batch_median_s": r_batch["median_s"],
                    "seq_tokens_per_s": seq_tokens_per_s,
                    "batch_tokens_per_s": batch_tokens_per_s,
                    "batch_speedup": speedup,
                })

                print(
                    f"    bs={bs:4d}  "
                    f"seq: {seq_tokens_per_s:>10.0f} tok/s  "
                    f"batch: {batch_tokens_per_s:>10.0f} tok/s  "
                    f"speedup: {speedup:.2f}x"
                )

    # Detokenization benchmark
    print(f"\n{'='*60}")
    print("Detokenization benchmark (medium text, batch=64)")
    print(f"{'='*60}")

    detok_rows = []
    for tok_name, tok_info in tokenizers.items():
        try:
            r = benchmark_detokenize(tok_info, TEXTS["medium"], 64)
            n_tokens = token_counts["medium"][tok_name]
            tokens_per_s = (64 * n_tokens) / r["median_s"]
            detok_rows.append({
                "tokenizer": tok_name,
                "family": tok_info["family"],
                "n_items": 64,
                "tokens_per_text": n_tokens,
                "median_s": r["median_s"],
                "tokens_per_s": tokens_per_s,
            })
            print(f"  {tok_name:35s}: {tokens_per_s:>10.0f} tok/s decode")
        except Exception as e:
            print(f"  {tok_name}: FAIL ({e})")

    df = pd.DataFrame(rows)
    detok_df = pd.DataFrame(detok_rows)

    df.to_csv(os.path.join(RESULTS_DIR, "tokenizer_benchmark.csv"), index=False)
    detok_df.to_csv(os.path.join(RESULTS_DIR, "detokenizer_benchmark.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY (batch tokenization throughput at batch=64)")
    print(f"{'='*60}")
    summary = df[df["batch_size"] == 64].groupby(["tokenizer", "text_length"])["batch_tokens_per_s"].mean().round(0)
    print(summary.to_string())

    print(f"\nSaved: {RESULTS_DIR}/tokenizer_benchmark.csv")


if __name__ == "__main__":
    main()
