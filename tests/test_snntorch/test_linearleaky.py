import pytest
import torch
from snntorch._neurons.stateleaky import StateLeaky  # Adjust the import path as needed


@pytest.fixture(scope="module")
def device():
    # Fixture to use GPU if available, otherwise fall back to CPU
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def linear_leaky_instance(device):
    # Fixture to initialize the StateLeaky instance
    return StateLeaky(beta=1, output=False).to(device)

@pytest.fixture(scope="module")
def linear_leaky_instance_multi_beta(device):
    # Fixture to initialize the StateLeaky instance
    return StateLeaky(beta=torch.tensor([0.9, 0.1])).to(device)

@pytest.fixture(scope="module")
def input_tensor_single_batch_single_channel(device):
    # Fixture to create a sample input tensor for single batch, single channel
    timesteps = 5
    batch = 1
    channels = 1

    mult = timesteps * batch * channels

    input_ = torch.arange(1, mult + 1).float().view(timesteps, batch, channels).to(device) / mult
    return input_

@pytest.fixture(scope="module")
def input_tensor_single_batch_multiple_channel(device):
    # Fixture to create a sample input tensor for single batch, single channel
    timesteps = 3
    batch = 1
    channels = 2
    input_ = torch.arange(1, timesteps * batch * channels + 1).float().view(timesteps, batch, channels).to(device)
    return input_

@pytest.fixture(scope="module")
def input_tensor_multiple_batches_single_channel(device):
    # Fixture for multiple batches but single channel
    timesteps = 5
    batch = 3  # Batch size of 3
    channels = 1  # Single channel
    input_ = torch.arange(1, timesteps * batch * channels + 1).float().view(timesteps, batch, channels).to(device)
    return input_


@pytest.fixture(scope="module")
def input_tensor_batch_multiple(device):
    # Fixture for multiple batch and channels
    timesteps = 3
    batch = 2  # Batch size of 2
    channels = 4  # 3 Channels
    input_ = torch.arange(1, timesteps * batch * channels + 1).float().view(timesteps, batch, channels).to(device)
    return input_


def test_forward_method_correctness_single_batch_single_channel(
        linear_leaky_instance, input_tensor_single_batch_single_channel):
    output = linear_leaky_instance.forward(input_tensor_single_batch_single_channel)

    expected_output = torch.tensor(
        [[[1.0000]], [[2.3292]], [[3.7668]], [[5.2400]], [[6.7250]]],
        device=input_tensor_single_batch_single_channel.device)

    print(input_tensor_single_batch_single_channel)
    print("output: ", output)
    print(linear_leaky_instance.tau)
    assert torch.allclose(output, expected_output,
                          atol=1e-4), "The forward method does not produce the expected output for single batch, single channel."


# def test_forward_method_correctness_multiple_batches_multiple_channels(
#     linear_leaky_instance, input_tensor_batch_multiple
# ):
#     output = linear_leaky_instance.forward(input_tensor_batch_multiple)

#     expected_output = torch.tensor([
#         [[ 1.0000,  2.0000,  3.0000,  4.0000],
#          [ 5.0000,  6.0000,  7.0000,  8.0000]],

#         [[ 9.3292, 10.6584, 11.9876, 13.3168],
#          [14.6460, 15.9752, 17.3044, 18.6335]],

#         [[20.0711, 21.5087, 22.9462, 24.3838],
#          [25.8214, 27.2589, 28.6965, 30.1340]]],
#         device=input_tensor_batch_multiple.device
#     )

#     assert torch.allclose(output, expected_output, atol=1e-4), (
#         "The forward method does not produce the expected output for multiple batches, multiple channels."
#     )


# def test_forward_method_correctness_multiple_batches_single_channel(
#         linear_leaky_instance, input_tensor_multiple_batches_single_channel):
#     output = linear_leaky_instance.forward(input_tensor_multiple_batches_single_channel)

#     expected_output = torch.tensor([[
#          [ 1.0000],
#          [ 2.0000],
#          [ 3.0000]],

#         [[ 4.3292],
#          [ 5.6584],
#          [ 6.9876]],

#         [[ 8.4251],
#          [ 9.8627],
#          [11.3003]],

#         [[12.7735],
#          [14.2467],
#          [15.7200]],

#         [[17.2049],
#          [18.6899],
#          [20.1749]]], device=input_tensor_multiple_batches_single_channel.device)

#     # Check that the output matches the expected output
#     assert torch.allclose(output, expected_output,
#                           atol=1e-4), "The forward method does not produce the expected output for multiple batches, single channel."

# def test_forward_method_correctness_multiple_batches_single_channel_multi_beta(
#         linear_leaky_instance_multi_beta, input_tensor_single_batch_multiple_channel):
#     output = linear_leaky_instance_multi_beta.forward(input_tensor_single_batch_multiple_channel)

#     expected_output = torch.tensor([
#         [[1.0000, 2.0000]],
#         [[3.3292, 4.0001]],
#         [[6.0959, 6.0002]]], device=input_tensor_single_batch_multiple_channel.device)

#     # Check that the output matches the expected output
#     assert torch.allclose(output, expected_output,
#                           atol=1e-4), "The forward method does not produce the expected output for multiple batches, single channel."


if __name__ == "__main__":
    pytest.main()
