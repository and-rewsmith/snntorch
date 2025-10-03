# benchmark StateLeaky

import torch
import time
from snntorch._neurons.stateleaky import StateLeaky


layer = StateLeaky(beta=0.9, channels=channels).to("cuda")

# warmup
input_ = (
    torch.arange(1, TIMESTEPS * batch * channels + 1)
    .float()
    .view(TIMESTEPS, batch, channels)
    .to("cuda")
)
layer.forward(input_)

# timing
input_ = (
    torch.arange(1, TIMESTEPS * batch * channels + 1)
    .float()
    .view(TIMESTEPS, batch, channels)
    .to("cuda")
)
input_ = torch.rand_like(input_)
start_time = time.time()
layer.forward(input_)
end_time = time.time()

print(f"{end_time-start_time}")
