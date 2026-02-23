"""
Goodput vs GPU count for synchronous vs elastic fault tolerance.
Parameters drawn from FT-HSDP (arXiv 2602.00277) and ByteCheckpoint (NSDI '25).
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Parameters ---
# Per-server failure rate: 0.0023 failures/server/day (Meta, section 2)
# 8 GPUs per server
LAMBDA_PER_GPU = 0.0023 / 8 / 86400  # failures/GPU/second

T_ITER = 2.0          # seconds per training step
T_STALL = 0.5         # checkpoint stall (ByteCheckpoint steady-state)
T_SAVE = 40.0         # end-to-end save (background, overlapped)
T_LOAD = 150.0        # checkpoint load time

# Cold recovery: fit from Meta's Figure 2 (section 2)
# ~200s at 16K, ~600s at 64K, ~900s at 98K
def t_cold(n_gpus):
    return 50 + 0.009 * n_gpus  # rough linear fit in seconds

# Elastic recovery (FT-HSDP): ~180s at 98K GPUs
T_ELASTIC = 180.0

# Correlated failure probability: fraction of failures that take down
# the whole job even with elastic recovery
P_CORRELATED = 0.05

# --- GPU counts to sweep ---
gpu_counts = np.logspace(np.log10(1000), np.log10(200_000), 200)

def mtbf(n_gpus):
    """Cluster MTBF in seconds. Uses empirical superlinear correction."""
    # Linear: MTBF = 1 / (N * lambda)
    mtbf_linear = 1.0 / (n_gpus * LAMBDA_PER_GPU)
    # Superlinear correction: Meta observed 18 min at 100K vs 50 min linear
    # Apply a mild power-law correction: MTBF_actual = MTBF_linear * (N_ref/N)^0.3
    n_ref = 32_000
    correction = (n_ref / n_gpus) ** 0.3 if n_gpus > n_ref else 1.0
    return mtbf_linear * correction

def optimal_ckpt_interval(mtbf_s, t_stall):
    """Young's formula: optimal interval in seconds."""
    return np.sqrt(2 * mtbf_s * t_stall)

def goodput_sync(n_gpus):
    m = mtbf(n_gpus)
    ckpt_interval = optimal_ckpt_interval(m, T_STALL)
    n_steps = ckpt_interval / T_ITER
    cycle = n_steps * T_ITER + T_STALL
    # Expected wasted time per cycle
    t_wasted = (cycle / m) * (n_steps * T_ITER / 2 + t_cold(n_gpus))
    productive = n_steps * T_ITER - t_wasted
    return max(0, productive / cycle)

def goodput_elastic(n_gpus, n_replicas):
    m = mtbf(n_gpus)
    ckpt_interval = optimal_ckpt_interval(m, T_STALL)
    n_steps = ckpt_interval / T_ITER
    cycle = n_steps * T_ITER + T_STALL
    # Checkpoint overhead
    ckpt_overhead = T_STALL / cycle
    # Degraded operation fraction
    degraded_frac = T_ELASTIC / (n_replicas * m)
    # Correlated failure penalty (still need full restart)
    correlated_penalty = P_CORRELATED * (n_steps * T_ITER / 2 + t_cold(n_gpus)) / m
    return max(0, 1.0 - ckpt_overhead - degraded_frac - correlated_penalty)

# --- Compute ---
gp_sync = [goodput_sync(n) for n in gpu_counts]
gp_elastic_4 = [goodput_elastic(n, 4) for n in gpu_counts]
gp_elastic_8 = [goodput_elastic(n, 8) for n in gpu_counts]
gp_elastic_16 = [goodput_elastic(n, 16) for n in gpu_counts]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(gpu_counts / 1000, gp_sync, label="Synchronous checkpoint-restart", linewidth=2)
ax.plot(gpu_counts / 1000, gp_elastic_4, label="Elastic (R=4 replicas)", linewidth=2)
ax.plot(gpu_counts / 1000, gp_elastic_8, label="Elastic (R=8 replicas)", linewidth=2)
ax.plot(gpu_counts / 1000, gp_elastic_16, label="Elastic (R=16 replicas)", linewidth=2)

ax.set_xlabel("GPU count (thousands)", fontsize=12)
ax.set_ylabel("Goodput (fraction of wall time)", fontsize=12)
ax.set_title("Goodput vs scale: synchronous vs elastic fault tolerance", fontsize=14)
ax.set_xscale("log")
ax.set_xlim(1, 200)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")
ax.text(150, 0.91, "90% goodput", fontsize=9, color="gray", ha="right")

plt.tight_layout()
plt.savefig("journal/2026-02-21-fault-tolerance-goodput-vs-scale.png", dpi=150)
print("Saved to journal/2026-02-21-fault-tolerance-goodput-vs-scale.png")
plt.close()
