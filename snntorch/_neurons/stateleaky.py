import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile

from .neurons import LIF
# from .linearleaky import LinearLeaky


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
        learn_decay_filter=False,
        learn_threshold=False,
        state_quant=False,
        output=True,
        graded_spikes_factor=1.0,
        max_timesteps=256,
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

        self.learn_decay_filter = learn_decay_filter
        self.max_timesteps = max_timesteps
        self._tau_buffer(self.beta, learn_beta, channels)
        self.input_dependence = nn.Linear(channels, channels, bias=True)

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    # @profile(skip=True, stdout=True, filename='baseline.prof')
    def forward(self, input_):
        self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            return self.spk, self.mem

        else:
            return self.mem

    def _base_state_function(self, input_):
        num_steps, batch, channels = input_.shape

        # confirm input shape
        assert input_.shape == (num_steps, batch, channels)

        # calc input dependence
        input_dependent_decay_modifier = self.input_dependence(input_.reshape(num_steps * batch, channels))
        input_dependent_decay_modifier = input_dependent_decay_modifier.reshape(num_steps, batch, channels)
        input_dependent_decay_modifier = input_dependent_decay_modifier.permute(2, 1, 0)
        input_dependent_decay_modifier = torch.sigmoid(input_dependent_decay_modifier)
        assert input_dependent_decay_modifier.shape == (channels, batch, num_steps)

        input_ = input_.permute(1, 2, 0)
        assert input_.shape == (batch, channels, num_steps)

        converted_tau = self.tau if self.tau.shape == (channels,) else self.tau.expand(channels).to(input_.device)
        assert converted_tau.shape == (channels,)

        # init decay filter
        # print(time_steps.shape)
        # print(self.tau.shape)

        # truncate decay filter to num_steps
        decay_filter = self.decay_filter.clone()
        assert decay_filter.shape == (self.max_timesteps, channels)
        decay_filter = decay_filter[:num_steps]
        assert decay_filter.shape == (num_steps, channels)

        # print("------------decay filter------------")
        # print(decay_filter)
        # print()

        # prepare for convolution
        decay_filter = decay_filter.permute(1, 0).unsqueeze(1)
        assert decay_filter.shape == (channels, 1, num_steps)
        decay_filter = decay_filter.expand(channels, batch, num_steps)
        assert decay_filter.shape == (channels, batch, num_steps)

        conv_result = self.full_mode_conv1d_truncated(input_, input_dependent_decay_modifier * decay_filter)
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)  # return membrane potential trace

    def _tau_buffer(self, beta, learn_beta, channels):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)

        if beta.shape != (channels,) and beta.shape != () and beta.shape != (1,):
            raise ValueError(f"Beta shape {beta.shape} must be either ({channels},) or (1,)")

        tau = 1 / (1 - beta + 1e-12)

        if learn_beta:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)

        # this is super important to make sure that decay_filter can not
        # require grad!
        tau = self.tau.clone().detach()

        converted_tau = tau if tau.shape == (channels,) else tau.expand(channels)
        assert converted_tau.shape == (channels,)

        # Create time steps array similar to _base_state_function
        time_steps = torch.arange(0, self.max_timesteps).to(converted_tau.device)
        assert time_steps.shape == (self.max_timesteps,)
        time_steps = time_steps.unsqueeze(1).expand(self.max_timesteps, channels)
        assert time_steps.shape == (self.max_timesteps, channels)

        decay_filter = torch.exp(-time_steps / converted_tau)
        assert decay_filter.shape == (self.max_timesteps, channels)

        if self.learn_decay_filter:
            self.decay_filter = nn.Parameter(decay_filter)
        else:
            self.register_buffer("decay_filter", decay_filter)

    def full_mode_conv1d_truncated(self, input_tensor, kernel_tensor):
        """
        Performs batch and channel-specific 1D convolutions using grouped convolution.

        torch.nn.functional.conv1d expects:
            - input shape: (batch_size, in_channels, length)
            - kernel shape: (out_channels, in_channels_per_group, kernel_length)
            - groups parameter: splits input channels into independent groups

        When groups=N:
            - Input channels are split into N groups
            - Each group is convolved with its own kernel independently
            - Each group's input channels = total_channels ÷ N
        """
        # input_tensor: (batch=B, channels=C, num_steps=S)
        # kernel_tensor: (channels=C, batch=B, kernel_size=K)
        kernel_tensor = torch.flip(kernel_tensor, dims=[-1]).to(input_tensor.device)

        # Get dimensions and verify shapes match
        batch_size_input, channels_input, num_steps = input_tensor.shape
        channels_kernel, batch_size_kernel, kernel_size = kernel_tensor.shape
        assert batch_size_input == batch_size_kernel
        assert channels_input == channels_kernel
        assert num_steps == kernel_size

        # We want each batch-channel combination to have its own independent convolution.
        # To achieve this, we'll use grouped convolution with groups = B*C
        # This means each input channel will be convolved with exactly one kernel.
        total_groups = batch_size_input * channels_input

        # Step 1: Reshape input for grouped convolution
        # From: (batch=B, channels=C, steps=S)
        # To:   (batch=1, total_groups=B*C, steps=S)
        # The 1 in the batch dimension is because we're treating each channel independently
        input_reshaped = input_tensor.reshape(1, total_groups, num_steps)
        assert input_reshaped.shape == (1, total_groups, num_steps)

        # Step 2: Reshape kernel for grouped convolution
        # From: (channels=C, batch=B, kernel_size=K)
        # To:   (total_groups=B*C, in_channels_per_group=1, kernel_size=K)
        # The middle 1 is required because each group processes 1 channel
        kernel_reshaped = kernel_tensor.permute(1, 0, 2)  # First get batch and channels adjacent
        assert kernel_reshaped.shape == (batch_size_kernel, channels_kernel, kernel_size)
        kernel_reshaped = kernel_reshaped.reshape(total_groups, 1, kernel_size)
        assert kernel_reshaped.shape == (total_groups, 1, kernel_size)

        # Add padding for full convolution
        padding = kernel_size - 1
        padded_input = F.pad(input_reshaped, (padding, padding))

        # Verify shapes before convolution
        assert padded_input.shape == (1, total_groups, num_steps + 2 * padding)
        assert kernel_reshaped.shape == (total_groups, 1, kernel_size)

        # Perform grouped convolution
        # - groups=total_groups means each channel gets its own independent convolution
        # - each group has 1 input channel and 1 output channel
        # - total number of groups = B*C (batch_size * channels)
        conv_result = F.conv1d(padded_input, kernel_reshaped, groups=total_groups)

        # Truncate to original sequence length
        conv_result = conv_result[..., 0:num_steps]

        # # Debug prints
        # print("Conv result shape:", conv_result.shape)
        # print("Expected shape:", (batch_size_input, channels_input, num_steps))
        # print("Total elements in conv_result:", conv_result.numel())
        # print("Total elements expected:", batch_size_input * channels_input * num_steps)

        # Reshape back to original
        # From: (1, B*C, S) -> (B, C, S)
        return conv_result.reshape(batch_size_input, channels_input, num_steps)

    # def full_mode_conv1d_truncated(self, input_tensor, kernel_tensor):
    #     # input_tensor: (batch, channels, num_steps)
    #     # kernel_tensor: (channels, 1, kernel_size)
    #     kernel_tensor = torch.flip(kernel_tensor, dims=[-1]).to(input_tensor.device)

    #     # get dimensions
    #     batch_size, in_channels, num_steps = input_tensor.shape
    #     out_channels, _, kernel_size = kernel_tensor.shape

    #     # pad the input tensor on both sides
    #     padding = kernel_size - 1
    #     padded_input = F.pad(input_tensor, (padding, padding))

    #     # print(padded_input.shape)
    #     # print(input_tensor.shape)
    #     # print("------input / kernel-------------")
    #     # print(input_tensor)
    #     # print(kernel_tensor)

    #     # perform convolution with the padded input
    #     conv_result = F.conv1d(padded_input, kernel_tensor, groups=in_channels)

    #     # truncate the result to match the original input length
    #     truncated_result = conv_result[..., 0:num_steps]

    #     return truncated_result

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
    input_ = torch.arange(1, timesteps * batch * channels + 1).float().view(timesteps, batch, channels).to(device)
    print("--------input tensor-----------")
    print(input_)
    print()
    out = leaky_linear.forward(input_)
    print("--------output-----------")
    print(out)
