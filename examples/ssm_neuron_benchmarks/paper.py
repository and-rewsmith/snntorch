import torch
import snntorch as snn

# Standard snnTorch Leaky / LIF neuron
lif = snn.Leaky(beta=0.9, threshold=1.0)


def run_leaky(x):
    """
    x: (batch, time, features) = (B, T, C)
    Returns:
        spk: (B, T, C) spike sequence
        mem: (B, T, C) membrane sequence
    """
    B, T, C = x.shape
    mem = torch.zeros(B, C, device=x.device)
    spk_seq = []
    mem_seq = []

    for t in range(T):
        spk, mem = lif(x[:, t], mem)  # step-by-step recurrence
        spk_seq.append(spk)
        mem_seq.append(mem)

    spk = torch.stack(spk_seq, dim=1)
    mem = torch.stack(mem_seq, dim=1)
    return spk, mem


import torch


def iterative_rollout(alpha, u):
    """
    Simple sequential recurrence:
        x_t = alpha * x_{t-1} + u_t

    alpha: scalar or tensor broadcastable to u[t]
    u: (T, D) input sequence
    Returns:
        x: (T, D) sequence of states
    """
    T, D = u.shape
    x = torch.zeros(D, device=u.device)
    xs = []
    for t in range(T):
        x = alpha * x + u[t]
        xs.append(x)
    return torch.stack(xs, dim=0)


import torch
import torch.nn.functional as F


def conv_rollout(alpha, u):
    """
    Time-invariant diagonal recurrence as a 1D convolution.

    u: (B, T, C) input sequence
    alpha: scalar decay in (0, 1)
    Returns:
        x: (B, T, C) parallel state sequence
    """
    B, T, C = u.shape

    # exponential kernel h[k] = alpha^k
    k = torch.arange(T, device=u.device)
    h = (alpha**k).view(1, 1, T)  # (1, 1, T)

    x = u.transpose(1, 2)  # (B, C, T)
    # causal 'full' convolution, then crop back to length T
    v = F.conv1d(
        x, h.expand(C, 1, T), padding=T - 1, groups=C
    )  # (B, C, T + T - 1)
    v = v.transpose(1, 2)[:, :T, :]  # (B, T, C)
    return v


import torch


def prefix_scan(alpha, u):
    """
    Input-dependent diagonal recurrence:
        x_t = alpha_t * x_{t-1} + u_t

    alpha: (T, D) elementwise decays in (0, 1)
    u:     (T, D) inputs
    Returns:
        x:   (T, D) states, computed via prefix products + prefix sums
    """
    # prefix products p_t = prod_{i=0}^t alpha_i
    p = torch.cumprod(alpha, dim=0)  # (T, D)

    # contributions normalized by prefix products
    contrib = u / p  # (T, D)

    # prefix sums s_t = sum_{i=0}^t contrib_i
    s = torch.cumsum(contrib, dim=0)  # (T, D)

    # x_t = p_t * s_t
    x = p * s
    return x


from typing import Callable, Sequence


def associative_scan(op: Callable, elems: Sequence):
    """
    Conceptual API for an associative scan:
        y_t = op(y_{t-1}, elems[t])

    In practice, efficient *parallel* implementations of this pattern
    rely on custom kernels (e.g., JAX, Triton, Thrust) rather than
    pure Python loops.
    """
    out = []
    acc = elems[0]
    out.append(acc)
    for e in elems[1:]:
        acc = op(acc, e)
        out.append(acc)
    return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class StateLeaky(nn.Module):
    """
    Reset-less Leaky/LIF as a diagonal, time-invariant SSM,
    evaluated via 1D grouped convolution.

    I: (B, T, C) sequence of input currents
    Returns:
        spk: (B, T, C) spike outputs
        v:   (B, T, C) membrane potentials
    """

    def __init__(self, beta=0.9, v_th=1.0):
        super().__init__()
        # Parameterize beta, but keep scalar here for simplicity
        self.beta = nn.Parameter(torch.tensor(beta))
        self.v_th = v_th

    def forward(self, I):
        B, T, C = I.shape

        # Build exponential kernel h[k] = beta^k
        k = torch.arange(T, device=I.device)
        h = (self.beta**k).view(1, 1, T)  # (1, 1, T)

        x = I.transpose(1, 2)  # (B, C, T)
        v = F.conv1d(
            x,
            h.expand(C, 1, T),
            padding=T - 1,
            groups=C,
        )  # (B, C, T + T - 1)
        v = v.transpose(1, 2)[:, :T, :]  # (B, T, C)

        spk = (v > self.v_th).to(I.dtype)
        return spk, v


import torch


def benchmark_step(model, loss_fn, x, y):
    """
    Measures wall-clock time (in ms) for a single forward+backward step.

    model:  nn.Module
    loss_fn: callable mapping (logits, targets) -> scalar loss
    x:      model inputs
    y:      targets
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)  # milliseconds
