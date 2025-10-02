import pytest
import torch
import torch.nn as nn

import os

from snntorch._neurons.stateleaky import StateLeaky

"""
Tests for the StateLeaky neuron class.

Test Structure:
--------------
1. Channel Configuration Tests:
    - Single batch, single channel
    - Single batch, multiple channels
    - Multiple batches, single channel
    - Multiple batches, multiple channels

2. Learning Parameter Tests:
    - Multi-beta learning (tests learn_beta=True)

Coverage:
--------
- Input/output shape consistency
- Input value bounds (≤ 1)
- Output activation (presence of values > 1)
- Parameter learnability for learn_beta

Limitations:
-----------
1. Does not test:
   - Spike generation (output=True)
   - Threshold learning (learn_threshold=True)
   - State quantization (state_quant=True)
   - Graded spike factor learning (learn_graded_spikes_factor=True)
   - Surrogate gradient functions (spike_grad parameter)

2. Testing Scope:
   - Uses fixed timesteps (5)
   - Uses fixed channel counts (1 or 4)
   - Uses fixed batch sizes (1 or 2)
   - Uses fixed beta value (0.9)
   - Tests only forward pass, not backward pass or gradient flow

3. Input Patterns:
   - Uses simple ascending sequence inputs normalized to [0,1]
   - Does not test with random or complex input patterns
"""


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def linear_leaky_single_channel(device):
    return StateLeaky(beta=0.9, channels=1, output=False).to(device)


@pytest.fixture(scope="module")
def linear_leaky_multi_channel(device):
    return StateLeaky(beta=0.9, channels=4, output=False).to(device)


@pytest.fixture(scope="module")
def linear_leaky_multi_beta(device):
    return StateLeaky(
        beta=torch.tensor([0.9] * 4), channels=4, learn_beta=True, output=False
    ).to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_single_channel(device):
    timesteps = 5
    batch = 1
    channels = 1

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_multiple_channel(device):
    timesteps = 5
    batch = 1
    channels = 4

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_multiple_batches_single_channel(device):
    timesteps = 5
    batch = 2
    channels = 1

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_batch_multiple(device):
    timesteps = 5
    batch = 2
    channels = 4

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


# Channel configuration tests
def test_single_batch_single_channel(
    linear_leaky_single_channel, input_tensor_single_batch_single_channel
):
    output = linear_leaky_single_channel.forward(
        input_tensor_single_batch_single_channel
    )

    assert input_tensor_single_batch_single_channel.shape == output.shape
    assert (
        (input_tensor_single_batch_single_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_single_batch_single_channel.shape
    ), "Output shape does not match input shape."
    assert (
        (output > 1).any().item()
    ), "No elements in the output are greater than 1."


def test_single_batch_multi_channel(
    linear_leaky_multi_channel, input_tensor_single_batch_multiple_channel
):
    output = linear_leaky_multi_channel.forward(
        input_tensor_single_batch_multiple_channel
    )

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (
        (input_tensor_single_batch_multiple_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_single_batch_multiple_channel.shape
    ), "Output shape does not match input shape."
    assert (
        (output > 1).any().item()
    ), "No elements in the output are greater than 1."


def test_multi_batch_single_channel(
    linear_leaky_single_channel, input_tensor_multiple_batches_single_channel
):
    output = linear_leaky_single_channel.forward(
        input_tensor_multiple_batches_single_channel
    )

    assert input_tensor_multiple_batches_single_channel.shape == output.shape
    assert (
        (input_tensor_multiple_batches_single_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_multiple_batches_single_channel.shape
    ), "Output shape does not match input shape."
    assert (
        (output > 1).any().item()
    ), "No elements in the output are greater than 1."


def test_multi_batch_multi_channel(
    linear_leaky_multi_channel, input_tensor_batch_multiple
):
    output = linear_leaky_multi_channel.forward(input_tensor_batch_multiple)

    assert input_tensor_batch_multiple.shape == output.shape
    assert (
        (input_tensor_batch_multiple <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_batch_multiple.shape
    ), "Output shape does not match input shape."
    assert (
        (output > 1).any().item()
    ), "No elements in the output are greater than 1."


def test_multi_beta_forward(
    linear_leaky_multi_beta, input_tensor_single_batch_multiple_channel
):
    output = linear_leaky_multi_beta.forward(
        input_tensor_single_batch_multiple_channel
    )

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (
        (input_tensor_single_batch_multiple_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_single_batch_multiple_channel.shape
    ), "Output shape does not match input shape."
    assert (
        (output > 1).any().item()
    ), "No elements in the output are greater than 1."

    # Verify learn_beta is a learnable parameter
    assert isinstance(
        linear_leaky_multi_beta.tau, nn.Parameter
    ), "learn_beta should be a learnable parameter"


def test_chunking_with_gd():
    batch_size = 256
    chunk_size = 64
    channels = 32
    timesteps = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"

    beta = torch.full((channels,), 0.9).to(device)
    lif = StateLeaky(beta=beta, channels=channels, learn_beta=False).to(device)
    linear = torch.nn.Linear(channels, channels, bias=False).to(device)

    input_tensor = (
        torch.arange(
            1,
            timesteps * batch_size * channels + 1,
            device=device,
            dtype=torch.float32,
        )
        .view(batch_size, channels, timesteps)
        .contiguous()
        .permute(2, 0, 1)   # (T, B, C)
    )
    
    # 1. Get ground truth with no chunking
    torch.set_grad_enabled(True)
    linear.zero_grad()

    # forward pass on entire tensor
    full_input_projected = linear(input_tensor)
    spk_full, mem_full = lif(full_input_projected)

    # backward pass on entire tensor
    loss_full = spk_full.sum()
    loss_full.backward()
    grad_full = linear.weight.grad.clone()

    # 2. Get gradients with chunking
    torch.set_grad_enabled(True)
    linear.zero_grad()

    spk_chunks = []

    for b_start in range(0, batch_size, chunk_size):
        b_end = min(b_start + chunk_size, batch_size)

        # select a chunk of the input tensor
        input_chunk = input_tensor[:, b_start:b_end, :]

        # # forward pass on chunk
        chunk_input_projected = linear(input_chunk)
        spk_chunk, _ = lif(chunk_input_projected)

        spk_chunks.append(spk_chunk)

        # backward pass on chunk
        loss_chunk = spk_chunk.sum()
        loss_chunk.backward()
    
    # resemble chunks into a single tensor
    spk_chunked = torch.cat(spk_chunks, dim=1)
    grad_chunked = linear.weight.grad.clone()

    # assertions
    assert spk_full.shape == spk_chunked.shape, "Chunked spike tensor shape mismatch."
    assert torch.allclose(
        spk_full, 
        spk_chunked,
        atol=1e-6
    ), "Forward pass (spikes) do not match."
    assert torch.allclose(
        grad_full, 
        grad_chunked,
        atol=1e-6
    ), "Backward pass (gradients) do not match."

if __name__ == "__main__":
    pytest.main()
