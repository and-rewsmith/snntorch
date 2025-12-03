# test_conv1d_cudnn.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("cuDNN version:", torch.backends.cudnn.version())
print(
    "CUDNN_* env:",
    {k: v for k, v in os.environ.items() if k.startswith("CUDNN_")},
)

# Let cuDNN autotune algorithms for these shapes
torch.backends.cudnn.benchmark = True


def run_case(
    name: str,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    length: int,
    kernel_size: int,
    groups: int = 1,
    stride: int = 1,
    padding: int | None = None,
):
    if padding is None:
        padding = kernel_size // 2  # "same-ish" conv

    print(f"\n=== {name} ===")
    print(
        f"  shape:  x=({batch_size}, {in_channels}, {length}), "
        f"w=({out_channels}, {in_channels // groups}, {kernel_size}), "
        f"groups={groups}, stride={stride}, padding={padding}"
    )

    x = torch.randn(
        batch_size, in_channels, length, device=device, dtype=torch.float32
    )

    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
    ).to(device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        y = conv(x)
    torch.cuda.synchronize()

    # Real runs
    for _ in range(5):
        y = conv(x)
    torch.cuda.synchronize()

    print("  output shape:", tuple(y.shape))
    print("  done")


def run_stateleaky_like():
    """
    Mimic StateLeaky:
      - input: (B, C, T)
      - depthwise conv: groups = C, out_channels = C
      - very long kernel (kernel_size ~ T)
      - manual left padding: F.pad(x, (K-1, 0))
    """
    name = "conv1d_stateleaky_like"

    # You can bump these to your real values (e.g. T=10000) if memory allows.
    B = 64
    C = 256
    T = 4096  # num_steps
    K = T  # kernel_size

    print(f"\n=== {name} ===")
    print(
        f"  shape:  x=({B}, {C}, {T}), "
        f"w=({C}, 1, {K}), groups={C}, stride=1, manual_pad_left={K-1}"
    )

    x = torch.randn(B, C, T, device=device, dtype=torch.float32)

    # Depthwise conv: groups = C, out_channels = C
    conv = nn.Conv1d(
        in_channels=C,
        out_channels=C,
        kernel_size=K,
        stride=1,
        padding=0,  # we do explicit left pad to match causal conv
        groups=C,
        bias=False,
    ).to(device=device, dtype=torch.float32)

    # Warmup
    for _ in range(2):
        x_pad = F.pad(x, (K - 1, 0))
        y = conv(x_pad)
    torch.cuda.synchronize()

    # Real runs
    for _ in range(3):
        x_pad = F.pad(x, (K - 1, 0))
        y = conv(x_pad)
    torch.cuda.synchronize()

    print("  output shape:", tuple(y.shape))
    print("  done")


if __name__ == "__main__":
    # Very standard Conv1d – should hit cuDNN
    run_case(
        name="conv1d_standard",
        batch_size=32,
        in_channels=64,
        out_channels=128,
        length=1024,
        kernel_size=7,
        groups=1,
        stride=1,
    )

    # Grouped Conv1d (still cuDNN-friendly, but not depthwise)
    run_case(
        name="conv1d_grouped",
        batch_size=32,
        in_channels=64,
        out_channels=64,
        length=1024,
        kernel_size=5,
        groups=4,  # groups < in_channels
        stride=1,
    )

    # StateLeaky-like: depthwise + huge kernel + manual causal padding
    run_stateleaky_like()

    print("\nAll tests complete.")
