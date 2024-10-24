import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile

from .neurons import LIF

def full_mode_conv1d_truncated(input_tensor, kernel_tensor):
    # input_tensor: (batch, channels, num_steps)
    # kernel_tensor: (channels, 1, kernel_size)
    kernel_tensor = torch.flip(kernel_tensor, dims=[-1])

    # get dimensions
    batch_size, in_channels, num_steps = input_tensor.shape
    out_channels, _, kernel_size = kernel_tensor.shape

    # pad the input tensor on both sides
    padding = kernel_size - 1
    padded_input = F.pad(input_tensor, (padding, padding))

    # print(padded_input.shape)
    # print(input_tensor.shape)
    # print("------input / kernel-------------")
    # print(input_tensor)
    # print(kernel_tensor)

    # perform convolution with the padded input
    conv_result = F.conv1d(padded_input, kernel_tensor, groups=in_channels)

    # truncate the result to match the original input length
    truncated_result = conv_result[..., 0:num_steps]

    return truncated_result

class LinearLeaky(LIF):
    """
    TODO: write some docstring similar to SNN.Leaky

     Jason wrote:
-      beta = (1 - delta_t / tau), can probably set delta_t to "1"
-      if tau > delta_t, then beta: (0, 1)
    """

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self._init_mem()

    def _init_mem(self):
        pass

    def reset_mem(self):
        pass

    def init_leaky(self):
        pass

    # @profile(skip=True, stdout=True, filename='baseline.prof')
    def forward(self, input_, mem=None):
        # init time steps arr
        num_steps, batch, channels = input_.shape
        time_steps = torch.arange(0, num_steps, device=input_.device)
        assert time_steps.shape == (num_steps,)
        time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)

        # init decay filter
        decay_filter = torch.exp(-time_steps / self.beta).to(input_.device)
        assert decay_filter.shape == (num_steps, channels)

        # prepare for convolution
        input_ = input_.permute(1, 2, 0)
        assert input_.shape == (batch, channels, num_steps)
        decay_filter = decay_filter.permute(1, 0).unsqueeze(1)
        assert decay_filter.shape == (channels, 1, num_steps)

        conv_result = full_mode_conv1d_truncated(input_, decay_filter)
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)

    def _base_state_function(self, input_):
        pass

    def _base_sub(self, input_):
        pass

    def _base_zero(self, input_):
        pass

    def _base_int(self, input_):
        pass

    @classmethod
    def detach_hidden(cls):
        pass

    @classmethod
    def reset_hidden(cls):
        pass


if __name__ == "__main__":
    device = "cuda"
    leaky_linear = LinearLeaky(beta=0.9).to(device)
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