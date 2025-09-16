import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import json
import os


# ------------------------------
# Config (no argparse): mirror original bench
# ------------------------------

SWEEP_CONFIGS = [
    (64, 256),
]
N_RUNS = 1
TIMESTEPS = np.logspace(1, 4.1, num=10, dtype=int)[::2]
BATCHWISE_CHUNK_SIZE = (
    64  # will switch to 32 for two-chunk case when timing that path
)

device = "cuda:1"


def cuda_sync():
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def get_peak_bytes(cuda_device):
    import torch

    return torch.cuda.max_memory_allocated(cuda_device)


def get_cur_bytes(cuda_device):
    import torch

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated(cuda_device)


# ------------------------------
# CUDA-event timed benches
# ------------------------------


def bench_leaky_events(
    num_steps: int, batch_size: int, channels: int, train: bool = False
):
    import torch
    from snntorch._neurons.leaky import Leaky

    lif = Leaky(beta=0.9).to(device)
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    # [T, B, C]
    input_tensor = torch.arange(
        1,
        num_steps * batch_size * channels + 1,
        device=device,
        dtype=torch.float32,
    ).view(num_steps, batch_size, channels)

    if train:
        input_tensor.requires_grad_(True)

    # Warm
    _ = lif.forward(linear(input_tensor[:2, :2, :]))
    cuda_sync()

    # Time with CUDA events
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    t0 = time.time()

    mem = torch.zeros(batch_size, channels, device=device)
    spk = torch.zeros(batch_size, channels, device=device)
    spk_steps = []
    for step_idx in range(num_steps):
        z = linear(input_tensor[step_idx])
        spk, mem = lif(z, mem=mem)
        spk_steps.append(spk)
        if train and step_idx < num_steps - 1:
            mem = mem.detach()

    spk_out = torch.stack(spk_steps, dim=0)

    if train:
        loss = spk_out.sum()
        loss.backward()
        if linear.weight.grad is not None:
            linear.weight.grad = None
        if input_tensor.grad is not None:
            input_tensor.grad = None
        del loss

    if device.startswith("cuda"):
        end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
    t1 = time.time()

    # Cleanup
    del lif, linear, input_tensor, mem, spk, spk_out
    cuda_sync()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return (t1 - t0), (
        gpu_ms / 1000.0 if device.startswith("cuda") else (t1 - t0)
    )


def bench_stateleaky_events(
    num_steps: int,
    batch_size: int,
    channels: int,
    train: bool = False,
    chunk_size: int = None,
):
    import torch
    from snntorch._neurons.stateleaky import StateLeaky

    if chunk_size is None:
        chunk_size = batch_size

    lif = StateLeaky(beta=0.9, channels=channels).to(device)

    # Build [B, T, C] then precompute linear over full tensor, match original flow
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)
    input_tensor = (
        torch.arange(
            1,
            num_steps * batch_size * channels + 1,
            device=device,
            dtype=torch.float32,
        )
        .view(batch_size, num_steps, channels)
        .contiguous()
    )

    z_full = (
        linear(input_tensor.view(-1, channels))
        .view(num_steps, batch_size, channels)
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    ).detach()

    if train:
        z_full.requires_grad_(True)

    # Warm
    _ = lif.forward(z_full[:2, :1, :])
    cuda_sync()

    # Event timing
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    t0 = time.time()

    # Process by chunks along B
    for b_start in range(0, batch_size, chunk_size):
        b_end = min(b_start + chunk_size, batch_size)
        x_chunk = z_full[:, b_start:b_end, :]
        spk_chunk, _mem = lif.forward(x_chunk)
        if train:
            chunk_loss = spk_chunk.sum()
            chunk_loss.backward()
            del chunk_loss

    if train:
        if linear.weight.grad is not None:
            linear.weight.grad = None
        if input_tensor.grad is not None:
            input_tensor.grad = None

    if device.startswith("cuda"):
        end.record()
        end.synchronize()
        gpu_ms = start.elapsed_time(end)
    t1 = time.time()

    del lif, linear, input_tensor, z_full
    cuda_sync()
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return (t1 - t0), (
        gpu_ms / 1000.0 if device.startswith("cuda") else (t1 - t0)
    )


# ------------------------------
# Runner to match original plots
# ------------------------------


if __name__ == "__main__":
    import torch

    # Sanity device fallback
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

    METRIC_KEYS = [
        "times_leaky",
        "times_state",
        "mems_leaky",
        "mems_state",
    ]

    results_infer_all = []
    results_train_all = []

    for run_idx in range(N_RUNS):
        for batch_size, channels in SWEEP_CONFIGS:
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

            for steps in TIMESTEPS:
                steps = int(steps)

                # Inference — measure peak memory deltas similar to original
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                baseline_mem = (
                    get_cur_bytes(device) if device.startswith("cuda") else 0
                )
                t_leaky_wall, t_leaky = bench_leaky_events(
                    steps, batch_size, channels, train=False
                )
                peak = (
                    get_peak_bytes(device) if device.startswith("cuda") else 0
                )
                d_leaky = (
                    max(0, peak - baseline_mem) / (1024**2)
                    if device.startswith("cuda")
                    else 0
                )

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                baseline_mem = (
                    get_cur_bytes(device) if device.startswith("cuda") else 0
                )
                # One big chunk (B)
                t_state_wall, t_state = bench_stateleaky_events(
                    steps,
                    batch_size,
                    channels,
                    train=False,
                    chunk_size=batch_size,
                )
                # Two chunks — match original by switching BATCHWISE_CHUNK_SIZE=32 if B=64
                _t_state_two_wall, t_state_two = bench_stateleaky_events(
                    steps,
                    batch_size,
                    channels,
                    train=False,
                    chunk_size=max(1, batch_size // 2),
                )
                # Use the two-chunk timing for comparison to original when chunking is desired
                peak = (
                    get_peak_bytes(device) if device.startswith("cuda") else 0
                )
                d_state = (
                    max(0, peak - baseline_mem) / (1024**2)
                    if device.startswith("cuda")
                    else 0
                )

                results_infer["times_leaky"].append(t_leaky)
                results_infer["times_state"].append(t_state_two)
                results_infer["mems_leaky"].append(d_leaky)
                results_infer["mems_state"].append(d_state)

                # Training
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                baseline_mem = (
                    get_cur_bytes(device) if device.startswith("cuda") else 0
                )
                t_leaky_wall, t_leaky = bench_leaky_events(
                    steps, batch_size, channels, train=True
                )
                peak = (
                    get_peak_bytes(device) if device.startswith("cuda") else 0
                )
                d_leaky = (
                    max(0, peak - baseline_mem) / (1024**2)
                    if device.startswith("cuda")
                    else 0
                )

                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                baseline_mem = (
                    get_cur_bytes(device) if device.startswith("cuda") else 0
                )
                t_state_wall, t_state = bench_stateleaky_events(
                    steps,
                    batch_size,
                    channels,
                    train=True,
                    chunk_size=batch_size,
                )
                _t_state_two_wall, t_state_two = bench_stateleaky_events(
                    steps,
                    batch_size,
                    channels,
                    train=True,
                    chunk_size=max(1, batch_size // 2),
                )
                peak = (
                    get_peak_bytes(device) if device.startswith("cuda") else 0
                )
                d_state = (
                    max(0, peak - baseline_mem) / (1024**2)
                    if device.startswith("cuda")
                    else 0
                )

                results_train["times_leaky"].append(t_leaky)
                results_train["times_state"].append(t_state_two)
                results_train["mems_leaky"].append(d_leaky)
                results_train["mems_state"].append(d_state)

            results_infer_all.append(results_infer)
            results_train_all.append(results_train)

    # Plot same format as original
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_time_inf, ax_mem_inf = axes[0]
    ax_time_trn, ax_mem_trn = axes[1]

    cmap = plt.get_cmap("tab10")
    for idx, res in enumerate(results_infer_all):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        ax_time_inf.plot(
            TIMESTEPS,
            res["times_leaky"],
            "-",
            color=color,
            label=f"Leaky {label_suffix}",
        )
        ax_time_inf.plot(
            TIMESTEPS,
            res["times_state"],
            "--",
            color=color,
            label=f"StateLeaky {label_suffix}",
        )
        ax_mem_inf.plot(
            TIMESTEPS,
            res["mems_leaky"],
            "-",
            color=color,
            label=f"Leaky {label_suffix}",
        )
        ax_mem_inf.plot(
            TIMESTEPS,
            res["mems_state"],
            "--",
            color=color,
            label=f"StateLeaky {label_suffix}",
        )

    for idx, res in enumerate(results_train_all):
        color = cmap(idx % 10)
        label_suffix = f"B{res['batch_size']}-C{res['channels']}"
        ax_time_trn.plot(
            TIMESTEPS,
            res["times_leaky"],
            "-",
            color=color,
            label=f"Leaky (train) {label_suffix}",
        )
        ax_time_trn.plot(
            TIMESTEPS,
            res["times_state"],
            "--",
            color=color,
            label=f"StateLeaky (train) {label_suffix}",
        )
        ax_mem_trn.plot(
            TIMESTEPS,
            res["mems_leaky"],
            "-",
            color=color,
            label=f"Leaky (train) {label_suffix}",
        )
        ax_mem_trn.plot(
            TIMESTEPS,
            res["mems_state"],
            "--",
            color=color,
            label=f"StateLeaky (train) {label_suffix}",
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

    os.makedirs("snn_performance", exist_ok=True)
    plt.tight_layout()
    plt.savefig("snn_performance/snn_performance_comparison.png", dpi=150)
    plt.show()
