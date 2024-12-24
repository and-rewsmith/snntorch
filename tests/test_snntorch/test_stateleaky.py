import pytest
import torch
from snntorch._neurons.stateleaky import StateLeaky


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
    return StateLeaky(beta=torch.tensor([0.9] * 4), channels=4, output=False).to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_single_channel(device):
    timesteps = 5
    batch = 1
    channels = 1

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_multiple_channel(device):
    timesteps = 5
    batch = 1
    channels = 4

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_multiple_batches_single_channel(device):
    timesteps = 5
    batch = 2
    channels = 1

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_batch_multiple(device):
    timesteps = 5
    batch = 2
    channels = 4

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


# Channel configuration tests
def test_single_batch_single_channel(
        linear_leaky_single_channel, input_tensor_single_batch_single_channel):
    output = linear_leaky_single_channel.forward(input_tensor_single_batch_single_channel)

    assert input_tensor_single_batch_single_channel.shape == output.shape
    assert (input_tensor_single_batch_single_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_single_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_single_batch_multi_channel(
        linear_leaky_multi_channel, input_tensor_single_batch_multiple_channel):
    output = linear_leaky_multi_channel.forward(input_tensor_single_batch_multiple_channel)

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (input_tensor_single_batch_multiple_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_multiple_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."

def test_multi_batch_single_channel(
        linear_leaky_single_channel, input_tensor_multiple_batches_single_channel):
    output = linear_leaky_single_channel.forward(input_tensor_multiple_batches_single_channel)

    assert input_tensor_multiple_batches_single_channel.shape == output.shape
    assert (input_tensor_multiple_batches_single_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_multiple_batches_single_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_multi_batch_multi_channel(
        linear_leaky_multi_channel, input_tensor_batch_multiple):
    output = linear_leaky_multi_channel.forward(input_tensor_batch_multiple)

    assert input_tensor_batch_multiple.shape == output.shape
    assert (input_tensor_batch_multiple <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_batch_multiple.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_multi_beta_forward(
        linear_leaky_multi_beta, input_tensor_single_batch_multiple_channel):
    output = linear_leaky_multi_beta.forward(input_tensor_single_batch_multiple_channel)

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (input_tensor_single_batch_multiple_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_multiple_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


if __name__ == "__main__":
    pytest.main()
