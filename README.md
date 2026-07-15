# tokenizer-throughput-benchmark

**Author:** João Felipe De Souza

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)
![Transformers](https://img.shields.io/badge/Transformers-5.13-FFD21E?style=flat-square)
![Tiktoken](https://img.shields.io/badge/Tiktoken-0.13-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-WSL%20%7C%20Linux-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Empirical tokenization benchmark covering four dimensions of serving performance:

1. Single-thread throughput
2. Batch vs sequential
3. Concurrent load (thread vs process parallelism)
4. Capacity planning (max RPS per SLO target)

Tokenizers tested:
- GPT-2 (BPE, HuggingFace)
- LLaMA 2 (SentencePiece, HuggingFace)
- Qwen2 (BPE, HuggingFace)
- Tiktoken GPT-3.5-turbo (OpenAI Rust BPE)
- Tiktoken GPT-4o (OpenAI Rust BPE)

For architecture details, see [DESIGN.md](DESIGN.md).

---

## Why This Matters

The default assumption: "Tiktoken is faster, use it always."

The reality: Tiktoken wins on single-threaded synthetic benchmarks but
**loses badly under real concurrent serving** due to GIL contention.

This benchmark quantifies both dimensions and produces a practical capacity
planning guide.

---

## Key Findings

### Finding 1 — Single-threaded: Tiktoken dominates

Batch=64, long text:

| Tokenizer | Throughput | vs GPT-2 |
|---|---:|---:|
| **tiktoken-gpt-4o** | **5.66M tok/s** | **1.77×** |
| tiktoken-gpt-3.5 | 3.66M tok/s | 1.14× |
| llama2 | 3.50M tok/s | 1.09× |
| gpt2 | 3.20M tok/s | 1.00× |

### Finding 2 — Under concurrent load, HuggingFace wins tail latency

p99 latency at concurrency=64:

| Tokenizer | p99 latency |
|---|---:|
| **gpt2** | **3.79 ms** |
| llama2 | 6.37 ms |
| tiktoken-gpt-4o | 12.04 ms |
| tiktoken-gpt-3.5 | 12.07 ms |

HuggingFace is 3× better on tail latency under concurrency.

### Finding 3 — Capacity planning: GPT-2 sustains 6.5× more RPS at 1ms SLO

Max sustainable RPS with p99 < 1ms:

| Tokenizer | Max RPS | vs Tiktoken-3.5 |
|---|---:|---:|
| **gpt2** | **6536 RPS** | **6.5×** |
| llama2 | 4979 RPS | 5.0× |
| tiktoken-gpt-4o | 2495 RPS | 2.5× |
| tiktoken-gpt-3.5 | 999 RPS | 1.0× |

At relaxed 5ms SLO all tokenizers saturate around **7500-8800 RPS** —
the hardware limit.

### Finding 4 — GIL effect explains everything

p99 latency growth from conc=1 to conc=64:

| Tokenizer | conc=1 | conc=64 | Degradation |
|---|---:|---:|---:|
| gpt2 | 0.19 ms | 3.79 ms | 19.6× |
| llama2 | 0.19 ms | 6.37 ms | 34.4× |
| **tiktoken-gpt-3.5** | **0.08 ms** | **12.07 ms** | **143.7×** |
| tiktoken-gpt-4o | 0.09 ms | 12.04 ms | 140.1× |

Tiktoken Python bindings hold the GIL, serializing concurrent calls.
HuggingFace releases the GIL during Rust execution.

### Finding 5 — Detokenization: Tiktoken destroys HuggingFace

Decoding throughput:

| Tokenizer | Decode tok/s |
|---|---:|
| **tiktoken-gpt-3.5** | **42.8M** |
| tiktoken-gpt-4o | 20.8M |
| gpt2-medium | 4.5M |
| llama2 | 2.6M |
| gpt2 | 2.4M |

For streaming inference where detokenization runs per generated token,
Tiktoken is **10-18× faster**.

### Finding 6 — LLaMA produces 12% more tokens than GPT-2

Same text, different token counts:

| Text | GPT-2 | LLaMA | Diff |
|---|---:|---:|---:|
| medium | 43 | 49 | +14% |
| long | 861 | 962 | +12% |

More tokens = larger KV-cache = more compute downstream.

---

## Main Conclusion

Tokenizer choice depends on workload:

- **Concurrent serving (many small requests)** → HuggingFace tokenizers.
  Better tail latency, better GIL behavior, 6.5× more RPS at 1ms SLO.

- **Batch offline processing** → Tiktoken.
  2-4× faster single-threaded, hardware-optimal for batches.

- **Streaming decoding** → Always Tiktoken.
  10-18× faster detokenization matters when called per token.

The common recommendation "just use Tiktoken" is wrong for concurrent
inference serving workloads.

---

## Results Files

| File | Description |
|---|---|
| `results/tokenizer_benchmark.csv` | Sequential vs batch throughput |
| `results/detokenizer_benchmark.csv` | Detokenization benchmark |
| `results/concurrent_benchmark.csv` | Thread vs process parallelism |
| `results/serving_simulation.csv` | Per-request latency under load |
| `results/saturation_test.csv` | Max RPS at each SLO |

## Plots

| File | Description |
|---|---|
| `plots/batch_throughput.png` | Throughput vs batch size |
| `plots/tokenizer_comparison.png` | Cross-tokenizer comparison |
| `plots/detokenization.png` | Detok throughput |
| `plots/latency_vs_concurrency.png` | Latency scaling with concurrency |
| `plots/latency_percentiles.png` | p50/p95/p99 comparison |
| `plots/saturation_curve.png` | p99 vs RPS saturation curve |
| `plots/max_rps_per_slo.png` | Max RPS per SLO target |

---

## Repository Structure

~~~text
tokenizer-throughput-benchmark/
├── tokenizer_benchmark.py       # Single-thread throughput
├── tokenizer_benchmark_v2.py    # Concurrency benchmark
├── serving_simulation.py        # Per-request latency
├── saturation_test.py           # Max RPS per SLO
├── plot_tokenizer.py
├── plot_serving.py
├── plot_saturation.py
├── README.md
├── DESIGN.md
├── LICENSE
├── requirements.txt
├── results/
└── plots/
~~~

---

## How to Run

### 1. Setup

~~~bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
~~~

### 2. Run benchmarks

~~~bash
python3 tokenizer_benchmark.py       # single-thread
python3 tokenizer_benchmark_v2.py    # concurrent
python3 serving_simulation.py        # per-request latency
python3 saturation_test.py           # max RPS per SLO
~~~

### 3. Generate plots

~~~bash
python3 plot_tokenizer.py
python3 plot_serving.py
python3 plot_saturation.py
~~~

---

## Limitations

- Single-machine benchmark
- English text only
- No streaming detokenization tested (only batch decode)
- Tokenizer versions may change performance characteristics
- Actual serving includes network I/O and other overhead

---

## References

- Sennrich et al., Neural Machine Translation of Rare Words with Subword Units (2016)
- Kudo & Richardson, SentencePiece (2018)
- OpenAI Tiktoken repository
- HuggingFace Tokenizers library
- Python GIL documentation
