import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile
from .neurons import LIF

import torch
from torch.autograd import Function
from torch.nn import functional as F


class _CausalExpConv(Function):
    @staticmethod
    def forward(
        ctx, x, tau, build_kernel_no_flip_fn, depthwise_corr_no_flip_fn
    ):
        """
        x:   (B, C, T)
        tau: () or (C,) tensor on same device/dtype
        build_kernel_no_flip_fn: callable(channels, T, device, dtype) -> (C,1,T) kernel with exp(-k/tau)
        depthwise_corr_no_flip_fn: callable(x, kernel) -> (B,C,T) causal correlation
        """
        B, C, T = x.shape
        device, dtype = x.device, x.dtype

        # Forward via your fast conv path (depthwise correlation with causal left pad).
        # We detach the kernel so autograd does not try to backprop through the FIR taps;
        # we'll provide custom grads wrt x and tau below.
        kernel = build_kernel_no_flip_fn(
            C, T, device, dtype
        ).detach()  # (C,1,T)
        y = depthwise_corr_no_flip_fn(x, kernel)  # (B,C,T)

        # Save for backward: y and alpha (=exp(-1/tau)), and whether tau is scalar or per-channel
        # alpha shape: (C,) after broadcast
        if tau.ndim == 0:
            alpha = (
                torch.exp(-1.0 / tau).to(device=device, dtype=dtype).repeat(C)
            )
            tau_for_grad = tau.reshape(())  # scalar shape
            tau_is_scalar = True
        else:
            alpha = torch.exp(
                -1.0 / tau.to(device=device, dtype=dtype)
            ).reshape(
                -1
            )  # (C,)
            tau_for_grad = tau
            tau_is_scalar = False

        ctx.save_for_backward(y, alpha, tau_for_grad)
        ctx.tau_is_scalar = tau_is_scalar
        return y.permute(2, 0, 1)  # (T,B,C) to match your interface

    @staticmethod
    def backward(ctx, grad_out_TBC):
        """
        grad_out_TBC: (T,B,C) = dL/dy
        returns grads for: x, tau, build_fn(None), depthwise_fn(None)
        """
        y, alpha, tau = (
            ctx.saved_tensors
        )  # y: (B,C,T), alpha: (C,), tau: () or (C,)
        B, C, T = y.shape

        gy = grad_out_TBC.permute(1, 2, 0).contiguous()  # (B,C,T)

        # ---- grad wrt input: reverse-time exponential filter ----
        # g_x[t] = g_y[t] + alpha * g_x[t+1]
        gx = torch.zeros_like(gy)  # (B,C,T)
        # vectorized across (B,C), loop over time only
        a = alpha.view(1, C, 1)  # broadcast over batch and time
        acc = torch.zeros(
            (B, C), device=gy.device, dtype=gy.dtype
        )  # g_x at "next" time
        for t in range(T - 1, -1, -1):
            acc = gy[:, :, t] + a[:, :, 0] * acc  # (B,C)
            gx[:, :, t] = acc

        # ---- grad wrt tau (if requires_grad) ----
        # We compute dL/dalpha via: dalpha = sum_t <gy[t], r[t]>, where
        # r[t] = y[t-1] + alpha * r[t-1], with r[0]=0 (define y[-1]=0).
        # Then dL/dtau = dL/dalpha * d_alpha/d_tau, with d_alpha/d_tau = -alpha / tau^2.
        if tau.requires_grad:
            r = torch.zeros(
                (B, C), device=y.device, dtype=y.dtype
            )  # r at previous step
            dalpha_per_channel = torch.zeros(
                (C,), device=y.device, dtype=y.dtype
            )

            for t in range(T):
                y_prev = y[:, :, t - 1] if t > 0 else 0.0
                r = y_prev + a[:, :, 0] * r  # (B,C)
                # accumulate per-channel: sum over batch
                dalpha_per_channel += (gy[:, :, t] * r).sum(dim=0)

            # chain rule: d_alpha/d_tau = -alpha / tau^2
            if ctx.tau_is_scalar:
                # sum over channels for scalar tau
                d_alpha_d_tau = -alpha.mean() / (
                    tau**2
                )  # mean vs sum is a design choice; use mean to match broadcasting
                dtau = (dalpha_per_channel.sum() * d_alpha_d_tau).reshape(())
            else:
                d_alpha_d_tau = -alpha / (tau**2)  # (C,)
                dtau = (dalpha_per_channel * d_alpha_d_tau).reshape_as(tau)
        else:
            dtau = None

        # No grads for the function handles
        return gx, dtau, None, None


class StateLeaky(LIF):
    """
        TODO: write some docstring similar to SNN.Leaky

         Jason wrote:
    -      beta = (1 - delta_t / tau), can probably set delta_t to "1"
    -      if tau > delta_t, then beta: (0, 1)
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
    ):
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

        self._tau_buffer(self.beta, learn_beta, channels)

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    # @profile(skip=False, stdout=False, filename="baseline.prof")
    def forward(self, input_):
        mem = self._base_state_function(input_)

        if self.state_quant:
            mem = self.state_quant(mem)

        if self.output:
            self.spk = self.fire(mem) * self.graded_spikes_factor
            # self.spk = self.spk.transpose(0, 1).contiguous().transpose(0, 1)
            return self.spk, mem

        else:
            return mem

    def _base_state_function(self, input_):
        batch, channels, num_steps = input_.shape
        device = input_.device

        # time axis shape (1, 1, num_steps)
        time_steps = torch.arange(num_steps, device=device).view(
            1, 1, num_steps
        )
        assert time_steps.shape == (1, 1, num_steps)

        # single channel case
        if self.tau.shape == ():
            # tau is scalar, broadcast across channels
            tau = self.tau.to(device)
            decay_filter = torch.exp(-time_steps / tau).expand(
                channels, 1, num_steps
            )
        else:
            # tau is (channels,), reshape to (channels, 1, 1) so it broadcasts correctly
            tau = self.tau.to(device).view(channels, 1, 1)
            assert tau.shape == (channels, 1, 1)
            decay_filter = torch.exp(
                -time_steps / tau
            )  # directly (channels, 1, num_steps)

        assert decay_filter.shape == (channels, 1, num_steps)
        assert input_.shape == (batch, channels, num_steps)

        # depthwise convolution: each channel gets its own decay filter
        conv_result = self.causal_conv1d(input_, decay_filter)
        assert conv_result.shape == (batch, channels, num_steps)

        # return conv_result.permute(2, 0, 1)  # (num_steps, batch, channels)
        return conv_result

    # def _base_state_function(self, input_):
    #     batch, channels, num_steps = input_.shape

    #     # make the decay filter
    #     time_steps = torch.arange(0, num_steps).to(input_.device)
    #     assert time_steps.shape == (num_steps,)

    #     # single channel case
    #     if self.tau.shape == ():
    #         decay_filter = (
    #             torch.exp(-time_steps / self.tau.to(input_.device))
    #             .unsqueeze(1)
    #             .expand(num_steps, channels)
    #         )
    #         assert decay_filter.shape == (num_steps, channels)
    #     # multichannel case
    #     else:
    #         # expand timesteps to be of shape (num_steps, channels)
    #         time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)
    #         # expand tau to be of shape (num_steps, channels)
    #         tau = (
    #             self.tau.unsqueeze(0)
    #             .expand(num_steps, channels)
    #             .to(input_.device)
    #         )
    #         # compute decay filter
    #         decay_filter = torch.exp(-time_steps / tau)
    #         assert decay_filter.shape == (num_steps, channels)

    #     # prepare for convolution
    #     # input_ = input_.permute(1, 2, 0)
    #     assert input_.shape == (batch, channels, num_steps)
    #     decay_filter = decay_filter.permute(1, 0).unsqueeze(1)
    #     assert decay_filter.shape == (channels, 1, num_steps)

    #     # check contiguous
    #     # input_ = input_.contiguous()
    #     # decay_filter = decay_filter.contiguous()
    #     print(f"input_.is_contiguous(): {input_.is_contiguous()}")
    #     print(f"decay_filter.is_contiguous(): {decay_filter.is_contiguous()}")
    #     # input()

    #     conv_result = self.causal_conv1d(input_, decay_filter).contiguous()
    #     assert conv_result.shape == (batch, channels, num_steps)

    #     return conv_result.permute(2, 0, 1)

    def causal_conv1d(self, input_tensor, kernel_tensor):
        # get dimensions
        batch_size, in_channels, num_steps = input_tensor.shape
        # kernel_tensor: (channels, 1, kernel_size)
        out_channels, _, kernel_size = kernel_tensor.shape

        # for causal convolution, output at time t only depends on inputs up to t
        # therefore, we pad only on the left side
        padding = kernel_size - 1
        padded_input = F.pad(input_tensor, (padding, 0))

        # kernel is flipped to turn cross-correlation performed by F.conv1d into convolution
        flipped_kernel = torch.flip(kernel_tensor, dims=[-1])

        # perform convolution with the padded input (output length = num_steps length)
        causal_conv_result = F.conv1d(
            padded_input, flipped_kernel, groups=in_channels
        )

        return causal_conv_result

    ###########################

    # def _build_decay_kernel_no_flip(self, channels, num_steps, device, dtype):
    #     """
    #     Construct a decay kernel aligned for cross-correlation:
    #     w[..., k] corresponds to lag = (num_steps - 1 - k).
    #     Output shape: (channels, 1, num_steps), contiguous by construction.
    #     """
    #     # reversed "lags": [num_steps-1, ..., 1, 0] -> causal weights for current & past
    #     # shape (1,1,T) to broadcast cleanly
    #     time_rev = torch.arange(
    #         num_steps - 1, -1, -1, device=device, dtype=dtype
    #     ).view(1, 1, num_steps)

    #     if self.tau.shape == ():  # scalar tau
    #         tau_scalar = self.tau.to(device=device, dtype=dtype)
    #         base = (
    #             -time_rev / tau_scalar
    #         ).exp()  # (1,1,T) -> new, dense, contiguous
    #         kernel = base.repeat(
    #             channels, 1, 1
    #         )  # (C,1,T) contiguous by construction
    #     else:  # channelwise tau
    #         # tau: (C,1,1) broadcasts against (1,1,T) -> elementwise exp returns a dense contiguous tensor
    #         tau = self.tau.to(device=device, dtype=dtype).view(channels, 1, 1)
    #         kernel = (
    #             -time_rev / tau
    #         ).exp()  # (C,1,T) contiguous as a fresh result

    #     # Sanity checks (no flips, no negative strides)
    #     # assert kernel.is_contiguous()
    #     return kernel

    # def _causal_depthwise_corr1d_no_flip(self, input_tensor, kernel_tensor):
    #     """
    #     input_tensor:  (batch, channels, num_steps)
    #     kernel_tensor: (channels, 1, kernel_size) aligned for *correlation* (already reversed in time)
    #     Performs causal correlation with left padding (kernel_size - 1).
    #     """
    #     # Left padding for causal dependence on past only
    #     pad_left = kernel_tensor.shape[-1] - 1
    #     padded = F.pad(input_tensor, (pad_left, 0))
    #     # Depthwise groups: each channel uses its own kernel
    #     return F.conv1d(padded, kernel_tensor, groups=input_tensor.shape[1])

    # def _base_state_function(self, input_):
    #     """
    #     input_: (batch, channels, num_steps), already in conv1d's expected layout.
    #     Returns: (num_steps, batch, channels)
    #     """
    #     batch, channels, num_steps = input_.shape
    #     device = input_.device
    #     dtype = input_.dtype

    #     # Build decay kernel directly in conv-ready, correlation-aligned layout:
    #     # For causal correlation with left padding (K-1), weight index k corresponds to lag = K-1-k.
    #     # So we construct time as [K-1, K-2, ..., 0] to avoid any flip.
    #     kernel = self._build_decay_kernel_no_flip(
    #         channels, num_steps, device, dtype
    #     )
    #     # kernel: (channels, 1, num_steps), contiguous by construction

    #     # Depthwise causal correlation: left-pad by K-1 and call conv1d with groups=channels.
    #     y = self._causal_depthwise_corr1d_no_flip(
    #         input_, kernel.detach()
    #     )  # (batch, channels, num_steps)

    #     # Return in your requested layout
    #     return y.permute(2, 0, 1)

    def _tau_buffer(self, beta, learn_beta, channels):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)

        if (
            beta.shape != (channels,)
            and beta.shape != ()
            and beta.shape != (1,)
        ):
            raise ValueError(
                f"Beta shape {beta.shape} must be either ({channels},) or (1,)"
            )

        tau = 1 / (1 - beta + 1e-12)
        if learn_beta:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)

    ###########################

    # def _base_state_function(self, input_):
    #     """
    #     input_: (batch, channels, num_steps) -> returns (num_steps, batch, channels)
    #     """
    #     # Ensure canonical contiguous layout for speed
    #     if not input_.is_contiguous():
    #         input_ = input_.contiguous()

    #     y_TBC = _CausalExpConv.apply(
    #         input_,
    #         self.tau,  # () or (C,)
    #         self._build_decay_kernel_no_flip,  # callable
    #         self._causal_depthwise_corr1d_no_flip,
    #     )
    #     return y_TBC


# TODO: throw exceptions if calling subclass methods we don't want to use
# fire_inhibition
# mem_reset, init, detach, zeros, reset_mem, init_leaky
# detach_hidden, reset_hidden


if __name__ == "__main__":
    device = "cuda"
    leaky_linear = StateLeaky(beta=0.9).to(device)
    timesteps = 5
    batch = 1
    channels = 1
    print("timesteps: ", timesteps)
    print("batch: ", batch)
    print("channels: ", channels)
    print()
    input_ = (
        torch.arange(1, timesteps * batch * channels + 1)
        .float()
        .view(timesteps, batch, channels)
        .to(device)
    )
    print("--------input tensor-----------")
    print(input_)
    print()
    out = leaky_linear.forward(input_)
    print("--------output-----------")
    print(out)
