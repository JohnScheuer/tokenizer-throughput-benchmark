# Design — tokenizer-throughput-benchmark

For findings and usage, see [README.md](README.md).

---

## Objective

Benchmark tokenization throughput and latency across production tokenizers
under four dimensions: single-thread, batch, concurrent, and saturation.

---

## Four Benchmarks

### 1. Throughput benchmark
Measures raw encoding/decoding speed:
- Sequential vs batch
- 3 text lengths, 5 batch sizes
- 5 tokenizers (GPT-2, LLaMA, Qwen2, Tiktoken-3.5, Tiktoken-4o)

### 2. Concurrency benchmark
Measures parallelism behavior:
- Thread pool (subject to GIL)
- Process pool (true parallelism)
- 5 concurrency levels

### 3. Serving simulation
Per-request latency under concurrent load:
- 1000 requests per configuration
- p50, p95, p99 distributions
- Thread-based concurrency (simulates HTTP server)

### 4. Saturation test
Capacity planning under offered load:
- Ramps from 100 to 50000 RPS
- Measures p99 at each load level
- Finds max sustainable RPS per SLO target (1ms, 5ms, 10ms)

---

## GIL Analysis

The most important finding: the GIL effect.

- **HuggingFace tokenizers** release the GIL during Rust execution.
  Threads can run in parallel.
- **Tiktoken** Python bindings hold the GIL during Rust execution.
  Threads serialize.

This makes Tiktoken faster single-threaded but slower under concurrent
serving — the exact opposite of what naive benchmarks suggest.

Evidence:
- Thread-based Tiktoken throughput drops from 2.7M to 490k tok/s at 16 workers
- Thread-based HuggingFace throughput stays flat
- Process-based Tiktoken scales properly (bypasses GIL)

---

## Saturation Test Design

Uses a load generator that:
1. Sends requests at a target rate (inter-arrival time = 1/RPS)
2. Processes them via ThreadPoolExecutor (8 workers)
3. Records per-request latency
4. Reports actual achieved RPS

When target RPS exceeds the system's capacity:
- Actual RPS saturates below target
- Latency spikes as the queue grows

The crossover point gives the max sustainable RPS for a given SLO.

---

## Why This Matters for Serving

Most serving frameworks (FastAPI, Uvicorn, TGI, vLLM) use:
- Thread pools for CPU-bound tasks like tokenization
- Async I/O for network

If the tokenizer holds the GIL, it blocks other async handlers
and serializes the entire request pipeline.

The right tokenizer choice depends on the concurrency model, not just
the single-call speed.

---

## File Structure

~~~text
tokenizer-throughput-benchmark/
├── tokenizer_benchmark.py       # Throughput (seq vs batch)
├── tokenizer_benchmark_v2.py    # Concurrency (thread vs process)
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
