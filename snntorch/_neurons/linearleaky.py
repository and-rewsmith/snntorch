from .neurons import LIF
import torch
from torch import nn

from torch.nn import functional as F


class LinearLeaky(LIF):
    """
    TODO: write some docstring similar to SNN.Leaky
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
        # reset_mechanism="subtract",
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

        # if self.reset_mechanism_val == 0:  # reset by subtraction
        #     self.state_function = self._base_sub
        # elif self.reset_mechanism_val == 1:  # reset to zero
        #     self.state_function = self._base_zero
        # elif self.reset_mechanism_val == 2:  # no reset, pure integration
        #     self.state_function = self._base_int

        # self.reset_delay = reset_delay

    def _init_mem(self):
        pass

    def reset_mem(self):
        pass

    def init_leaky(self):
        pass

    # def __valid_mode_conv1d(input_tensor, kernel_tensor):
    #     # input_tensor: (batch, channels, num_steps)
    #     # kernel_tensor: (channels, 1, kernel_size)

    #     # Get dimensions
    #     batch_size, in_channels, num_steps = input_tensor.shape
    #     out_channels, _, kernel_size = kernel_tensor.shape

    #     # Perform convolution without padding for 'valid' mode
    #     # Grouped convolution with groups=in_channels performs an independent conv per channel
    #     conv_result = F.conv1d(input_tensor, kernel_tensor, groups=in_channels)

    #     return conv_result

    def full_mode_conv1d_truncated(self, input_tensor, kernel_tensor):
        # input_tensor: (batch, channels, num_steps)
        # kernel_tensor: (channels, 1, kernel_size)
        kernel_tensor = torch.flip(kernel_tensor, dims=[-1])

        # Get dimensions
        batch_size, in_channels, num_steps = input_tensor.shape
        out_channels, _, kernel_size = kernel_tensor.shape

        # Calculate padding for 'full' mode
        padding = kernel_size - 1

        # Pad the input tensor on both sides
        padded_input = F.pad(input_tensor, (padding, padding))
        print(padded_input.shape)
        print(input_tensor.shape)
        print("-------------------")
        print(input_tensor)
        print(kernel_tensor)
        print("-------------------")

        # Perform convolution with the padded input
        conv_result = F.conv1d(padded_input, kernel_tensor, groups=in_channels)
        print(conv_result)

        # Truncate the result to match the original input length
        truncated_result = conv_result[..., 0:num_steps]
        
        return truncated_result

    # TODO: change beta to tau
    # beta = (1 - delta_t / tau), can probably set delta_t to "1"
    # if tau > delta_t, then beta: (0, 1)
    def forward_v2(self, input_, mem=None):
        # EQUATION:
        # V_t = V_0 * exp(-t/tau) + SUM_s->t(I_s * exp(-(t-s)/tau)) ??? # not sure if correct

        # SETUP:
        num_steps, batch, channels = input_.shape
        time_steps = torch.arange(0, num_steps, device=input_.device)
        assert time_steps.shape == (num_steps,)

        # DECAY FILTER WILL BE REUSED:
        decay_filter = torch.exp(-time_steps / self.beta)
        assert decay_filter.shape == (num_steps,)
        # print("decay filter:")
        # print(decay_filter.shape)

        # INITIAL MEMBRANE DECAY:
        initial_mem = torch.zeros_like(input_[0])
        # print("initial mem:")
        # print(initial_mem.shape)
        initial_state_decay_over_time = decay_filter * initial_mem
        # print("initial state decay over time:")
        # print(initial_state_decay_over_time.shape)
        assert initial_state_decay_over_time.shape == (batch, num_steps)

        # INPUT CURRENT DECAY:
        # print("input:")
        input_ = input_.permute(1, 2, 0)
        assert input_.shape == (batch, channels, num_steps)
        # print(input_.shape)
        # print("decay filter:")
        # print(decay_filter.shape)
        decay_filter = decay_filter.view(channels, 1, -1)
        assert decay_filter.shape == (channels, 1, num_steps)

        conv_result = self.full_mode_conv1d_truncated(input_, decay_filter)

        print("num steps:")
        print(num_steps)
        print("conv result:")
        print(conv_result.shape)
        print(conv_result)

        # input_decay_over_time = F.conv1d(input_, decay_filter, stride=1, padding=0)
        return conv_result

    def forward(self, input_, mem=None):

        num_steps, batch, channels = input_.shape

        time_steps = torch.arange(0, num_steps, device=input_.device)
        assert time_steps.shape == (num_steps,)

        # time x batch x channels....
        # time x time x batch x channels // TO-DO: change self.beta
        # TO-DO: add time-step offset as well.
        mem_response = input_ * torch.exp(-time_steps / self.beta)
        assert mem_response.shape == (num_steps, num_steps, batch, channels)

        mem = mem_response.sum()
        pass

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
    leaky_linear = LinearLeaky(beta=0.9)
    leaky_linear.forward_v2(torch.arange(1, 6).float().view(5, 1, 1))
    print("success")
