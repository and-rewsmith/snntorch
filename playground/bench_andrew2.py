import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc
import json
import os

from snntorch._neurons.leaky import Leaky
from snntorch._neurons.stateleaky import StateLeaky
from tqdm import tqdm


# Sweep configurations: (batch_size, channels)
SWEEP_CONFIGS = [
    # (100, 100),
    (20, 80),
    # (10, 40),
    (10, 20),
    (1, 5),
]
N_RUNS = 3

# Same timestep schedule as baseline
TIMESTEPS = np.logspace(1, 5, num=10, dtype=int)

device = "cuda:1"
torch.set_grad_enabled(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
try:
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

REPEATS = 7


def get_peak_bytes(cuda_device):
    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    torch.cuda.synchronize()
    gc.collect()
    return torch.cuda.memory_allocated(cuda_device)


# ------------------------------
# Benchmark kernels
# ------------------------------


def bench_leaky(
    num_steps: int, batch_size: int, channels: int, train: bool = False
) -> float:
    lif = Leaky(beta=0.9).to(device)

    if train:
        ctx = torch.enable_grad()
        x = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
        x.requires_grad_(True)
    else:
        ctx = torch.no_grad()
        x = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)

    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)

    torch.cuda.synchronize()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    with ctx:
        for step_idx in range(num_steps):
            spk, mem = lif(x[step_idx], mem=mem)
            if train and step_idx < num_steps - 1:
                mem = mem.detach()

        if train:
            loss = spk.sum() + mem.sum()
            loss.backward()
            if x.grad is not None:
                x.grad = None
            del loss
    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    if gc_was_enabled:
        gc.enable()

    del lif, x, mem, spk
    torch.cuda.synchronize()
    gc.collect()
    return elapsed_ms / 1000.0


def bench_stateleaky(
    num_steps: int,
    batch_size: int,
    channels: int,
    train: bool = False,
    input_tensor: torch.Tensor | None = None,
) -> float:
    lif = StateLeaky(beta=0.9, channels=channels).to(device)
    if input_tensor is None:
        input_tensor = torch.arange(
            1,
            num_steps * batch_size * channels + 1,
            device=device,
            dtype=torch.float32,
        ).view(num_steps, batch_size, channels)

    if train:
        input_tensor.requires_grad_(True)
        ctx = torch.enable_grad()
    else:
        ctx = torch.no_grad()

    torch.cuda.synchronize()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    with ctx:
        out = lif.forward(input_tensor)

        if train:
            if isinstance(out, tuple):
                spk, mem = out
                loss = spk.sum() + mem.sum()
            else:
                loss = out.sum()
            loss.backward()
            if input_tensor.grad is not None:
                input_tensor.grad = None
            del loss

    end_evt.record()
    torch.cuda.synchronize()
    elapsed_ms = start_evt.elapsed_time(end_evt)
    if gc_was_enabled:
        gc.enable()

    del lif, input_tensor, out
    torch.cuda.synchronize()
    gc.collect()
    return elapsed_ms / 1000.0


# ------------------------------
# Worker: run all configs for a single RUN
# ------------------------------


def run_all_configs_one_run(run_idx: int):
    results_infer_all = []
    results_train_all = []

    for cfg in SWEEP_CONFIGS:
        batch_size, channels = cfg

        results_infer = dict(
            batch_size=batch_size,
            channels=channels,
            times_leaky=[],
            times_state=[],
            mems_leaky=[],
            mems_state=[],
        )
        results_train = dict(
            batch_size=batch_size,
            channels=channels,
            times_leaky=[],
            times_state=[],
            mems_leaky=[],
            mems_state=[],
        )

        # light warmup per config to stabilize kernels/allocator
        max_steps = int(max(TIMESTEPS))
        prealloc_state_input = torch.arange(
            1,
            max_steps * batch_size * channels + 1,
            device=device,
            dtype=torch.float32,
        ).view(max_steps, batch_size, channels)
        _ = bench_leaky(16, batch_size, channels, train=False)
        _ = bench_stateleaky(
            16,
            batch_size,
            channels,
            train=False,
            input_tensor=prealloc_state_input[:16],
        )
        _ = bench_leaky(16, batch_size, channels, train=True)
        _ = bench_stateleaky(
            16,
            batch_size,
            channels,
            train=True,
            input_tensor=prealloc_state_input[:16],
        )

        for steps in tqdm(
            TIMESTEPS, desc=f"RUN{run_idx} B{batch_size}-C{channels}"
        ):
            # --- Inference ---
            # timing with repeats; memory measured once separately
            times = []
            for _ in range(REPEATS):
                tval = bench_leaky(
                    int(steps), batch_size, channels, train=False
                )
                times.append(tval)
            t1 = float(np.median(times))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base1 = get_cur_bytes(device)
            _ = bench_leaky(int(steps), batch_size, channels, train=False)
            peak1 = get_peak_bytes(device)
            d1 = max(0, peak1 - base1) / 1024**2

            times = []
            for _ in range(REPEATS):
                tval = bench_stateleaky(
                    int(steps),
                    batch_size,
                    channels,
                    train=False,
                    input_tensor=prealloc_state_input[: int(steps)],
                )
                times.append(tval)
            t2 = float(np.median(times))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base2 = get_cur_bytes(device)
            _ = bench_stateleaky(
                int(steps),
                batch_size,
                channels,
                train=False,
                input_tensor=prealloc_state_input[: int(steps)],
            )
            peak2 = get_peak_bytes(device)
            d2 = max(0, peak2 - base2) / 1024**2

            results_infer["times_leaky"].append(t1)
            results_infer["times_state"].append(t2)
            results_infer["mems_leaky"].append(d1)
            results_infer["mems_state"].append(d2)

            # --- Training ---
            times = []
            for _ in range(REPEATS):
                tval = bench_leaky(
                    int(steps), batch_size, channels, train=True
                )
                times.append(tval)
            t1 = float(np.median(times))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base1 = get_cur_bytes(device)
            _ = bench_leaky(int(steps), batch_size, channels, train=True)
            peak1 = get_peak_bytes(device)
            d1 = max(0, peak1 - base1) / 1024**2

            times = []
            for _ in range(REPEATS):
                tval = bench_stateleaky(
                    int(steps),
                    batch_size,
                    channels,
                    train=True,
                    input_tensor=prealloc_state_input[: int(steps)],
                )
                times.append(tval)
            t2 = float(np.median(times))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            base2 = get_cur_bytes(device)
            _ = bench_stateleaky(
                int(steps),
                batch_size,
                channels,
                train=True,
                input_tensor=prealloc_state_input[: int(steps)],
            )
            peak2 = get_peak_bytes(device)
            d2 = max(0, peak2 - base2) / 1024**2

            results_train["times_leaky"].append(t1)
            results_train["times_state"].append(t2)
            results_train["mems_leaky"].append(d1)
            results_train["mems_state"].append(d2)

        results_infer_all.append(results_infer)
        results_train_all.append(results_train)

    return results_infer_all, results_train_all


# ------------------------------
# Entry point
# ------------------------------

if __name__ == "__main__":
    # Hardcoded configuration that produced the smooth graph
    os.environ.pop("SNN_ISOLATE", None)  # ensure non-isolated mode
    N_RUNS = 3
    REPEATS = 1
    # Main mode: launch workers in-process
    METRIC_KEYS = [
        "times_leaky",
        "times_state",
        "mems_leaky",
        "mems_state",
    ]

    # Accumulators for mean/std across runs
    infer_sum = None
    infer_sumsq = None
    infer_meta = None  # holds batch_size/channels per cfg

    train_sum = None
    train_sumsq = None
    train_meta = None

    for run_idx in range(N_RUNS):
        infer, train = run_all_configs_one_run(run_idx)

        if infer_sum is None:
            # initialize accumulators and metadata
            infer_sum = []
            infer_sumsq = []
            infer_meta = []
            for cfg in infer:
                infer_meta.append(
                    {
                        "batch_size": cfg["batch_size"],
                        "channels": cfg["channels"],
                    }
                )
                cfg_sum = {}
                cfg_sumsq = {}
                for k in METRIC_KEYS:
                    arr = np.array(cfg[k], dtype=float)
                    cfg_sum[k] = arr.copy()
                    cfg_sumsq[k] = arr**2
                infer_sum.append(cfg_sum)
                infer_sumsq.append(cfg_sumsq)
        else:
            for cfg_idx in range(len(infer)):
                for k in METRIC_KEYS:
                    arr = np.array(infer[cfg_idx][k], dtype=float)
                    infer_sum[cfg_idx][k] = infer_sum[cfg_idx][k] + arr
                    infer_sumsq[cfg_idx][k] = infer_sumsq[cfg_idx][k] + arr**2

        if train_sum is None:
            train_sum = []
            train_sumsq = []
            train_meta = []
            for cfg in train:
                train_meta.append(
                    {
                        "batch_size": cfg["batch_size"],
                        "channels": cfg["channels"],
                    }
                )
                cfg_sum = {}
                cfg_sumsq = {}
                for k in METRIC_KEYS:
                    arr = np.array(cfg[k], dtype=float)
                    cfg_sum[k] = arr.copy()
                    cfg_sumsq[k] = arr**2
                train_sum.append(cfg_sum)
                train_sumsq.append(cfg_sumsq)
        else:
            for cfg_idx in range(len(train)):
                for k in METRIC_KEYS:
                    arr = np.array(train[cfg_idx][k], dtype=float)
                    train_sum[cfg_idx][k] = train_sum[cfg_idx][k] + arr
                    train_sumsq[cfg_idx][k] = train_sumsq[cfg_idx][k] + arr**2

    # Compute mean and std across runs
    results_infer = []
    results_train = []
    for cfg_idx in range(len(SWEEP_CONFIGS)):
        # Inference
        cfg_res = {
            "batch_size": infer_meta[cfg_idx]["batch_size"],
            "channels": infer_meta[cfg_idx]["channels"],
        }
        for k in METRIC_KEYS:
            mean_arr = infer_sum[cfg_idx][k] / max(N_RUNS, 1)
            var_arr = infer_sumsq[cfg_idx][k] / max(N_RUNS, 1) - mean_arr**2
            var_arr = np.maximum(var_arr, 0.0)
            std_arr = np.sqrt(var_arr)
            cfg_res[k] = mean_arr.tolist()
            cfg_res[f"std_{k}"] = std_arr.tolist()
        results_infer.append(cfg_res)

        # Training
        cfg_res_t = {
            "batch_size": train_meta[cfg_idx]["batch_size"],
            "channels": train_meta[cfg_idx]["channels"],
        }
        for k in METRIC_KEYS:
            mean_arr = train_sum[cfg_idx][k] / max(N_RUNS, 1)
            var_arr = train_sumsq[cfg_idx][k] / max(N_RUNS, 1) - mean_arr**2
            var_arr = np.maximum(var_arr, 0.0)
            std_arr = np.sqrt(var_arr)
            cfg_res_t[k] = mean_arr.tolist()
            cfg_res_t[f"std_{k}"] = std_arr.tolist()
        results_train.append(cfg_res_t)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_time_inf, ax_mem_inf = axes[0]
    ax_time_trn, ax_mem_trn = axes[1]

    cmap = plt.get_cmap("tab10")
    for idx, res in enumerate(results_infer):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        ax_time_inf.errorbar(
            TIMESTEPS,
            res["times_leaky"],
            yerr=res.get("std_times_leaky", None),
            fmt="-",
            color=color,
            label=f"Leaky {label_suffix}",
            capsize=3,
        )
        ax_time_inf.errorbar(
            TIMESTEPS,
            res["times_state"],
            yerr=res.get("std_times_state", None),
            fmt="--",
            color=color,
            label=f"StateLeaky {label_suffix}",
            capsize=3,
        )
        ax_mem_inf.errorbar(
            TIMESTEPS,
            res["mems_leaky"],
            yerr=res.get("std_mems_leaky", None),
            fmt="-",
            color=color,
            label=f"Leaky {label_suffix}",
            capsize=3,
        )
        ax_mem_inf.errorbar(
            TIMESTEPS,
            res["mems_state"],
            yerr=res.get("std_mems_state", None),
            fmt="--",
            color=color,
            label=f"StateLeaky {label_suffix}",
            capsize=3,
        )

    for idx, res in enumerate(results_train):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        ax_time_trn.errorbar(
            TIMESTEPS,
            res["times_leaky"],
            yerr=res.get("std_times_leaky", None),
            fmt="-",
            color=color,
            label=f"Leaky (train) {label_suffix}",
            capsize=3,
        )
        ax_time_trn.errorbar(
            TIMESTEPS,
            res["times_state"],
            yerr=res.get("std_times_state", None),
            fmt="--",
            color=color,
            label=f"StateLeaky (train) {label_suffix}",
            capsize=3,
        )
        ax_mem_trn.errorbar(
            TIMESTEPS,
            res["mems_leaky"],
            yerr=res.get("std_mems_leaky", None),
            fmt="-",
            color=color,
            label=f"Leaky (train) {label_suffix}",
            capsize=3,
        )
        ax_mem_trn.errorbar(
            TIMESTEPS,
            res["mems_state"],
            yerr=res.get("std_mems_state", None),
            fmt="--",
            color=color,
            label=f"StateLeaky (train) {label_suffix}",
            capsize=3,
        )

    for ax in (ax_time_inf, ax_mem_inf, ax_time_trn, ax_mem_trn):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)

    ax_time_inf.set_title("SNN Performance (Time) - Inference")
    ax_time_inf.set_xlabel("Timesteps")
    ax_time_inf.set_ylabel("Time (s)")
    ax_mem_inf.set_title("SNN Memory (Peak) - Inference")
    ax_mem_inf.set_xlabel("Timesteps")
    ax_mem_inf.set_ylabel("Δ Memory (MB)")
    ax_time_trn.set_title("SNN Performance (Time) - Training")
    ax_time_trn.set_xlabel("Timesteps")
    ax_time_trn.set_ylabel("Time (s)")
    ax_mem_trn.set_title("SNN Memory (Peak) - Training")
    ax_mem_trn.set_xlabel("Timesteps")
    ax_mem_trn.set_ylabel("Δ Memory (MB)")

    ax_time_inf.legend(ncol=2, fontsize=8)
    ax_mem_inf.legend(ncol=2, fontsize=8)
    ax_time_trn.legend(ncol=2, fontsize=8)
    ax_mem_trn.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig("snn_performance_comparison.png", dpi=150)
    plt.show()
