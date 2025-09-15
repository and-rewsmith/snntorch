import argparse
import time
import os
import sys
import torch


def time_stateleaky_direct(
    T: int, B: int, C: int, chunk: int, device: str, train: bool
):
    """Accurate GPU timing of StateLeaky using CUDA events.
    Returns a dict with per-scenario GPU/Wall timings and per-chunk breakdown.
    """
    from snntorch._neurons.stateleaky import StateLeaky

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.set_device(device)
        torch.cuda.synchronize()

    lif = StateLeaky(beta=0.9, channels=C).to(device)
    x = torch.randn(T, B, C, device=device)
    linear = torch.nn.Linear(C, C, bias=False).to(device)
    assert linear.weight.requires_grad, "Weights must require grad"

    # if train:
    #     x.requires_grad_(True)

    def timed_call(x_in):
        start = end = None
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
        t0 = time.time()
        intermediate = linear(x_in)
        intermediate = intermediate.permute(1, 2, 0)
        assert intermediate.shape == (B, C, T) or intermediate.shape == (
            chunk,
            C,
            T,
        )
        spk, mem = lif.forward(intermediate)
        if train:
            loss = spk.sum()
            loss.backward()
            del loss
        if device.startswith("cuda"):
            end.record()
            end.synchronize()
        t1 = time.time()
        gpu_ms = (
            start.elapsed_time(end) if start is not None else (t1 - t0) * 1e3
        )
        wall_ms = (t1 - t0) * 1e3
        return gpu_ms, wall_ms

    results = {}

    # Big chunk (single call)
    gpu_ms_big, wall_ms_big = timed_call(x)
    results["big"] = dict(gpu_ms=gpu_ms_big, wall_ms=wall_ms_big)

    # Two chunks along batch (B/2 + B/2)
    assert B % chunk == 0, "Chunk must divide batch size"
    per_chunk = []
    total_gpu_ms = 0.0
    total_wall_ms = 0.0
    for b0 in range(0, B, chunk):
        b1 = b0 + chunk
        x_chunk = x[:, b0:b1, :]
        if len(per_chunk) == 0:
            print(
                f"[debug] x_chunk is_contiguous={x_chunk.is_contiguous()}, stride={x_chunk.stride()}"
            )
        gms, wms = timed_call(x_chunk)
        per_chunk.append(dict(gpu_ms=gms, wall_ms=wms))
        total_gpu_ms += gms
        total_wall_ms += wms

    results["two"] = dict(
        total_gpu_ms=total_gpu_ms,
        total_wall_ms=total_wall_ms,
        per_chunk=per_chunk,
    )
    return results


def time_stateleaky_bench(
    T: int, B: int, C: int, chunk: int, device: str, train: bool
):
    """Call the existing bench function for comparison, using its signature.
    Do not modify the bench; set globals then call.
    """
    try:
        import playground.bench_andrew2 as b
    except ModuleNotFoundError:
        # Fallback when run as a script: add repo root to sys.path
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        sys.path.insert(0, repo_root)
        import playground.bench_andrew2 as b  # type: ignore

    b.device = device
    b.BATCHWISE_CHUNK_SIZE = chunk

    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.synchronize()

    mem, t = b.bench_stateleaky(T, B, C, train=train)
    return t


def time_leaky_direct(T: int, B: int, C: int, device: str, train: bool):
    """Accurate GPU timing of Leaky (baseline) using CUDA events.
    Leaky is not chunked: runs across time for the full batch.
    Returns a dict with only the big-chunk timing.
    """
    from snntorch._neurons.leaky import Leaky

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.set_device(device)
        torch.cuda.synchronize()

    lif = Leaky(beta=0.9).to(device)
    linear = torch.nn.Linear(C, C, bias=False).to(device)
    x = torch.randn(T, B, C, device=device)
    assert linear.weight.requires_grad, "Weights must require grad"

    def time_loop(x_in):
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
        t0 = time.time()
        mem = torch.zeros(x_in.shape[1], C, device=device)
        spk_steps = []
        for t in range(x_in.shape[0]):
            z = linear(x_in[t])
            spk, mem = lif(z, mem=mem)
            spk_steps.append(spk)
        spk_out = torch.stack(spk_steps, dim=0)
        if train:
            loss = spk_out.sum()
            loss.backward()
            del loss
        if device.startswith("cuda"):
            end.record()
            end.synchronize()
        t1 = time.time()
        gpu_ms = (
            start.elapsed_time(end)
            if device.startswith("cuda")
            else (t1 - t0) * 1e3
        )
        wall_ms = (t1 - t0) * 1e3
        return gpu_ms, wall_ms

    big_gpu, big_wall = time_loop(x)
    return {"big": {"gpu_ms": big_gpu, "wall_ms": big_wall}}


def time_leaky_bench(T: int, B: int, C: int, device: str, train: bool):
    try:
        import playground.bench_andrew2 as b
    except ModuleNotFoundError:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        sys.path.insert(0, repo_root)
        import playground.bench_andrew2 as b  # type: ignore

    b.device = device
    # Leaky is not chunked; no need to set chunk size

    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.synchronize()

    mem, t = b.bench_leaky(T, B, C, train=train)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--T", type=int, default=10000)
    p.add_argument("--B", type=int, default=64)
    p.add_argument("--C", type=int, default=256)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument(
        "--no-train", action="store_true", help="Only run inference"
    )
    p.add_argument("--no-infer", action="store_true", help="Only run training")
    args = p.parse_args()

    if args.device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA not available"
        print(
            f"CUDA available: {torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    print(
        f"Device: {args.device}  T={args.T} B={args.B} C={args.C} chunk={args.chunk}"
    )

    # Inference
    if not args.no_infer:
        print("\n=== Inference (Leaky direct, CUDA events) ===")
        res_leaky_inf = time_leaky_direct(
            args.T, args.B, args.C, args.device, train=False
        )
        print(
            f"Big chunk:    GPU {res_leaky_inf['big']['gpu_ms']:.2f} ms, Wall {res_leaky_inf['big']['wall_ms']:.2f} ms"
        )

        print("\n=== Inference (bench_andrew2.bench_leaky) ===")
        t_bench_leaky_big = time_leaky_bench(
            args.T, args.B, args.C, args.device, train=False
        )
        print(f"bench big chunk reported: {t_bench_leaky_big:.6f} s")

        print("\n=== Inference (StateLeaky direct, CUDA events) ===")
        res_inf = time_stateleaky_direct(
            args.T, args.B, args.C, args.chunk, args.device, train=False
        )
        print(
            f"Big chunk:    GPU {res_inf['big']['gpu_ms']:.2f} ms, "
            f"Wall {res_inf['big']['wall_ms']:.2f} ms"
        )
        print(
            f"Two chunks:   GPU {res_inf['two']['total_gpu_ms']:.2f} ms, "
            f"Wall {res_inf['two']['total_wall_ms']:.2f} ms"
        )
        for i, ch in enumerate(res_inf["two"]["per_chunk"]):
            print(
                f"  chunk {i}:  GPU {ch['gpu_ms']:.2f} ms, Wall {ch['wall_ms']:.2f} ms"
            )

        print("\n=== Inference (bench_andrew2.bench_stateleaky) ===")
        t_bench_big = time_stateleaky_bench(
            args.T, args.B, args.C, args.B, args.device, train=False
        )
        print(f"bench big chunk reported: {t_bench_big:.6f} s")
        t_bench_two = time_stateleaky_bench(
            args.T, args.B, args.C, args.chunk, args.device, train=False
        )
        print(f"bench two chunks reported: {t_bench_two:.6f} s")

    # Training
    if not args.no_train:
        print("\n=== Training (Leaky direct, CUDA events) ===")
        res_leaky_tr = time_leaky_direct(
            args.T, args.B, args.C, args.device, train=True
        )
        print(
            f"Big chunk:    GPU {res_leaky_tr['big']['gpu_ms']:.2f} ms, Wall {res_leaky_tr['big']['wall_ms']:.2f} ms"
        )

        print("\n=== Training (bench_andrew2.bench_leaky) ===")
        t_bench_leaky_big = time_leaky_bench(
            args.T, args.B, args.C, args.device, train=True
        )
        print(f"bench big chunk reported: {t_bench_leaky_big:.6f} s")

        print("\n=== Training (StateLeaky direct, CUDA events) ===")
        res_tr = time_stateleaky_direct(
            args.T, args.B, args.C, args.chunk, args.device, train=True
        )
        print(
            f"Big chunk:    GPU {res_tr['big']['gpu_ms']:.2f} ms, "
            f"Wall {res_tr['big']['wall_ms']:.2f} ms"
        )
        print(
            f"Two chunks:   GPU {res_tr['two']['total_gpu_ms']:.2f} ms, "
            f"Wall {res_tr['two']['total_wall_ms']:.2f} ms"
        )
        for i, ch in enumerate(res_tr["two"]["per_chunk"]):
            print(
                f"  chunk {i}:  GPU {ch['gpu_ms']:.2f} ms, Wall {ch['wall_ms']:.2f} ms"
            )

        print("\n=== Training (bench_andrew2.bench_stateleaky) ===")
        t_bench_big = time_stateleky_bench_safe(
            args.T, args.B, args.C, args.B, args.device, train=True
        )
        print(f"bench big chunk reported: {t_bench_big:.6f} s")
        t_bench_two = time_stateleky_bench_safe(
            args.T, args.B, args.C, args.chunk, args.device, train=True
        )
        print(f"bench two chunks reported: {t_bench_two:.6f} s")


def time_stateleky_bench_safe(T, B, C, chunk, device, train):
    """Wrap bench call in try/except so a bench failure doesn't stop direct timings."""
    try:
        return time_stateleaky_bench(T, B, C, chunk, device, train)
    except Exception as e:
        print(f"[bench error] {e}")
        return float("nan")


if __name__ == "__main__":
    main()
