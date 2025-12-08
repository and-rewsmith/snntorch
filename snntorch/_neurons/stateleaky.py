from warnings import warn
import torch
from torch import nn
from torch.nn import functional as F
from .neurons import LIF


# ---------------------------------------------------------------------------
#  Fast causal depthwise convolution implemented using Conv2d with MANUAL padding
# ---------------------------------------------------------------------------

def causal_conv2d(input_tensor, kernel_tensor):
    """
    input_tensor:  (B, C, T)
    kernel_tensor: (C, 1, K)
    returns:       (B, C, T)
    """
    B, C, T = input_tensor.shape
    _, _, K = kernel_tensor.shape

    # Manual causal pad on time dimension (left-only)
    x = F.pad(input_tensor.unsqueeze(2), (K - 1, 0, 0, 0))  # (B, C, 1, T + K - 1)

    # Flip kernel for convolution (Conv2d performs cross-correlation by default)
    w = torch.flip(kernel_tensor, dims=[-1]).unsqueeze(2)  # (C, 1, 1, K)

    # Depthwise Conv2d
    out = F.conv2d(x, w, groups=C)  # (B, C, 1, T)
    return out.squeeze(2)  # (B, C, T)


# ---------------------------------------------------------------------------
#  StateLeaky neuron (Conv2d-accelerated LIF)
# ---------------------------------------------------------------------------

class StateLeaky(LIF):
    """
    Drop-in replacement for LIF that uses a Conv2d-based causal convolution
    for faster state computation while preserving surrogate-gradient behavior.
    """

    def __init__(
        self,
        beta,
        channels,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        learn_beta=False,
        learn_threshold=False,
        state_quant=False,
        output=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        kernel_truncation_steps=None,
    ):
        # Let LIF handle beta/tau/threshold/spike_grad/etc.
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self.channels = channels
        self.kernel_truncation_steps = kernel_truncation_steps

        if not output:
            if (
                spike_grad is not None
                or surrogate_disable
                or learn_threshold
                or learn_graded_spikes_factor
                or (
                    isinstance(graded_spikes_factor, torch.Tensor)
                    and not torch.all(graded_spikes_factor == 1.0)
                )
                or (
                    not isinstance(graded_spikes_factor, torch.Tensor)
                    and graded_spikes_factor != 1.0
                )
            ):
                warn(
                    "StateLeaky: spike-related settings are unused when output=False (no spikes emitted).",
                    UserWarning,
                )

    # ----------------------------------------------------------------------
    # Core state computation (Conv2d version)
    # ----------------------------------------------------------------------
    def _base_state_function(self, input_):
        """
        input_: (T, B, C)
        returns mem: (T, B, C)
        """
        T, B, C = input_.shape

        # (T, B, C) -> (B, C, T)
        x = input_.permute(1, 2, 0)
        device = x.device

        # kernel length
        if self.kernel_truncation_steps is None:
            K = T
        else:
            K = min(self.kernel_truncation_steps, T)

        # time indices
        t_idx = torch.arange(K, device=device).view(1, 1, K)

        # Get beta (possibly learnable) and compute tau
        beta = self.beta.to(device)
        if beta.shape in [(), (1,)]:
            tau = 1.0 / (1.0 - beta + 1e-12)
            decay_filter = torch.exp(-t_idx / tau).expand(C, 1, K)
        else:
            tau = 1.0 / (1.0 - beta + 1e-12)
            tau = tau.view(C, 1, 1)
            decay_filter = torch.exp(-t_idx / tau)

        # causal depthwise conv2d
        conv_out = causal_conv2d(x, decay_filter)  # (B, C, T)
        return conv_out.permute(2, 0, 1)  # (T, B, C)

    # ----------------------------------------------------------------------
    # Forward pass (inherits surrogate-gradient spikes from LIF)
    # ----------------------------------------------------------------------
    def forward(self, input_):
        # Compute membrane state with Conv2d causal kernel
        mem = self._base_state_function(input_)

        # Optional quantization
        if self.state_quant:
            mem = self.state_quant(mem)

        # If no spikes requested, just return membrane trace
        if not self.output:
            return mem

        # Use LIF's surrogate spike logic
        spk = self.fire(mem) * self.graded_spikes_factor
        return spk, mem
