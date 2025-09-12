import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile
from .neurons import LIF
from torch.autograd import Function


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
        # Optional cache for decay filter to avoid recomputation across batch chunks
        self.cache_decay_filter = False
        self._decay_cache_key = None
        self._decay_filter_cached = None
        # Training-time option: detach temporal dependency between steps (no BPTT)
        # When True and module in training mode, use a step-wise recurrence with
        # mem detached at every step (matches baseline Leaky training semantics).
        self.detach_time = False

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    # @profile(skip=False, stdout=False, filename="baseline.prof")
    def forward(self, input_):
        # Training-time detachment: compute mem via conv without tracking,
        # then pass gradients straight-through per time step to inputs.
        if self.training and self.detach_time:
            with torch.no_grad():
                mem_ng = self._base_state_function_conv(input_)
            self.mem = _PerStepPassThrough.apply(input_, mem_ng)
        else:
            self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            # self.spk = self.spk.transpose(0, 1).contiguous().transpose(0, 1)
            return self.spk, self.mem

        else:
            return self.mem

    def _base_state_function(self, input_):
        num_steps, batch, channels = input_.shape
        # In training with temporal detachment requested, use step-wise path
        if self.training and self.detach_time:
            return self._base_state_function_step_detach(input_)
        
        # make (or reuse) the decay filter of shape (num_steps, channels)
        device = input_.device
        dtype = input_.dtype
        cache_ok = (
            self.cache_decay_filter
            and not isinstance(self.tau, nn.Parameter)
        )
        cache_key = (
            device,
            dtype,
            int(num_steps),
            int(channels),
            tuple(self.tau.shape),
        )
        if cache_ok and self._decay_cache_key == cache_key and self._decay_filter_cached is not None:
            decay_filter = self._decay_filter_cached
        else:
            time_steps = torch.arange(0, num_steps, device=device, dtype=dtype)
            assert time_steps.shape == (num_steps,)

            # single channel case
            if self.tau.shape == ():
                decay_filter = (
                    torch.exp(-time_steps / self.tau.to(device=device, dtype=dtype))
                    .unsqueeze(1)
                    .expand(num_steps, channels)
                )
                assert decay_filter.shape == (num_steps, channels)
            # multichannel case
            else:
                # expand timesteps to be of shape (num_steps, channels)
                time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)
                # expand tau to be of shape (num_steps, channels)
                tau = (
                    self.tau.to(device=device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(num_steps, channels)
                )
                # compute decay filter
                decay_filter = torch.exp(-time_steps / tau)
                assert decay_filter.shape == (num_steps, channels)

            if cache_ok:
                self._decay_cache_key = cache_key
                self._decay_filter_cached = decay_filter

        # prepare for convolution
        input_ = input_.permute(1, 2, 0)
        assert input_.shape == (batch, channels, num_steps)
        decay_filter = decay_filter.permute(1, 0).unsqueeze(1)
        assert decay_filter.shape == (channels, 1, num_steps)

        conv_result = self.causal_conv1d(input_, decay_filter).contiguous()
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)

    def _base_state_function_conv(self, input_):
        # Pure conv path, independent of training flags
        num_steps, batch, channels = input_.shape
        device = input_.device
        dtype = input_.dtype

        # make (or reuse) the decay filter of shape (num_steps, channels)
        cache_ok = (
            self.cache_decay_filter
            and not isinstance(self.tau, nn.Parameter)
        )
        cache_key = (
            device,
            dtype,
            int(num_steps),
            int(channels),
            tuple(self.tau.shape),
        )
        if cache_ok and self._decay_cache_key == cache_key and self._decay_filter_cached is not None:
            decay_filter = self._decay_filter_cached
        else:
            time_steps = torch.arange(0, num_steps, device=device, dtype=dtype)
            assert time_steps.shape == (num_steps,)

            if self.tau.shape == ():
                decay_filter = (
                    torch.exp(-time_steps / self.tau.to(device=device, dtype=dtype))
                    .unsqueeze(1)
                    .expand(num_steps, channels)
                )
                assert decay_filter.shape == (num_steps, channels)
            else:
                time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)
                tau = (
                    self.tau.to(device=device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(num_steps, channels)
                )
                decay_filter = torch.exp(-time_steps / tau)
                assert decay_filter.shape == (num_steps, channels)

            if cache_ok:
                self._decay_cache_key = cache_key
                self._decay_filter_cached = decay_filter

        # prepare for convolution
        inp = input_.permute(1, 2, 0)
        assert inp.shape == (batch, channels, num_steps)
        k = decay_filter.permute(1, 0).unsqueeze(1)
        assert k.shape == (channels, 1, num_steps)

        conv_result = self.causal_conv1d(inp, k).contiguous()
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)

    def _base_state_function_step_detach(self, input_):
        # input_: (T, B, C)
        num_steps, batch, channels = input_.shape
        assert input_.shape == (num_steps, batch, channels)

        device = input_.device
        dtype = input_.dtype

        # decay per step: exp(-1/tau) to mirror convolution kernel exp(-t/tau)
        tau = self.tau.to(device=device, dtype=dtype)
        if tau.shape == ():
            decay = torch.exp(-1.0 / tau)
            assert decay.shape == ()
        else:
            assert tau.shape == (channels,)
            decay = torch.exp(-1.0 / tau)
            assert decay.shape == (channels,)

        mem = torch.zeros(batch, channels, device=device, dtype=dtype)
        assert mem.shape == (batch, channels)

        mem_steps = []
        for t in range(num_steps):
            x_t = input_[t]
            assert x_t.shape == (batch, channels)
            if decay.shape == ():
                mem = decay * mem + x_t
            else:
                mem = decay.unsqueeze(0) * mem + x_t
            assert mem.shape == (batch, channels)
            mem_steps.append(mem)
            # Detach temporal dependency to avoid BPTT across time
            mem = mem.detach()

        mem_seq = torch.stack(mem_steps, dim=0)
        assert mem_seq.shape == (num_steps, batch, channels)
        return mem_seq

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


class _PerStepPassThrough(Function):
    @staticmethod
    def forward(ctx, inp, mem_ng):
        # Return precomputed membrane (no-grad), but remember nothing
        return mem_ng

    @staticmethod
    def backward(ctx, grad_output):
        # Route gradients straight-through to inputs per time-step; no grad for mem_ng
        return grad_output, None

        


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
