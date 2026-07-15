import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("results/saturation_test.csv")

COLORS = {
    "gpt2": "#3498DB",
    "llama2": "#2ECC71",
    "tiktoken-gpt-3.5": "#E74C3C",
    "tiktoken-gpt-4o": "#C0392B",
}

# Plot: p99 vs actual RPS
fig, ax = plt.subplots(figsize=(12, 7))
for tok in ["gpt2", "llama2", "tiktoken-gpt-3.5", "tiktoken-gpt-4o"]:
    sub = df[df["tokenizer"] == tok].sort_values("actual_rps")
    ax.plot(sub["actual_rps"], sub["p99_ms"],
            marker="o", linewidth=2, markersize=8,
            color=COLORS[tok], label=tok)

ax.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="p99 = 1ms SLO")
ax.axhline(5.0, color="orange", linestyle="--", alpha=0.5, label="p99 = 5ms SLO")
ax.set_xlabel("Actual RPS achieved")
ax.set_ylabel("p99 latency (ms)")
ax.set_title("Tokenizer capacity planning: p99 vs RPS\n(where each tokenizer breaks the SLO)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.savefig("plots/saturation_curve.png", dpi=180, bbox_inches="tight")
plt.close()
print("plot: plots/saturation_curve.png")

# Plot: max RPS per SLO bar chart
slos = [1.0, 5.0, 10.0]
tokenizers = ["gpt2", "llama2", "tiktoken-gpt-3.5", "tiktoken-gpt-4o"]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(tokenizers))
width = 0.25

for i, slo in enumerate(slos):
    max_rps = []
    for tok in tokenizers:
        sub = df[(df["tokenizer"] == tok) & (df["p99_ms"] < slo)]
        max_rps.append(sub["actual_rps"].max() if len(sub) > 0 else 0)
    ax.bar(x + (i - 1) * width, max_rps, width, label=f"p99 < {slo}ms")

ax.set_xticks(x)
ax.set_xticklabels(tokenizers, rotation=15, ha="right")
ax.set_ylabel("Max sustainable RPS")
ax.set_title("Max RPS per tokenizer per SLO target")
ax.grid(True, alpha=0.3, axis="y")
ax.legend()
plt.tight_layout()
plt.savefig("plots/max_rps_per_slo.png", dpi=180, bbox_inches="tight")
plt.close()
print("plot: plots/max_rps_per_slo.png")

print("\n=== Capacity planning summary ===")
for slo in [1.0, 5.0]:
    print(f"\nMax RPS at p99 < {slo}ms:")
    for tok in tokenizers:
        sub = df[(df["tokenizer"] == tok) & (df["p99_ms"] < slo)]
        max_rps = sub["actual_rps"].max() if len(sub) > 0 else 0
        print(f"  {tok:25s}: {max_rps:>7.0f} RPS")
