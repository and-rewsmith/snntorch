from .neurons import LIF
import torch
from torch import nn


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
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        pass

    def _init_mem(self):
        pass

    def reset_mem(self):
        pass

    def init_leaky(self):
        pass

    def forward(self, input_, mem=None):
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
    print("success")
