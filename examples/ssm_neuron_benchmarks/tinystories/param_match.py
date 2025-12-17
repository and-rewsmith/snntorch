import math

import torch
import torch.nn as nn

from snntorch._neurons.stateleaky import StateLeaky
from snntorch._neurons.associative import AssociativeLeaky


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_stateleaky_block(hidden_dim: int) -> nn.Module:
    """
    Lightweight block mirroring the TinyStories StateLeaky stack (without embeddings/output):
    - 4x LayerNorm(hidden_dim)
    - 3x Linear(hidden_dim, hidden_dim)
    - 3x StateLeaky(channels=hidden_dim, learn_beta=True)  [no params]
    """
    ln1 = nn.LayerNorm(hidden_dim)
    ln2 = nn.LayerNorm(hidden_dim)
    ln3 = nn.LayerNorm(hidden_dim)
    ln_out = nn.LayerNorm(hidden_dim)

    fc2 = nn.Linear(hidden_dim, hidden_dim)
    fc3 = nn.Linear(hidden_dim, hidden_dim)
    fc4 = nn.Linear(hidden_dim, hidden_dim)

    lif1 = StateLeaky(beta=torch.tensor(0.9), channels=hidden_dim, learn_beta=True)
    lif2 = StateLeaky(beta=torch.tensor(0.9), channels=hidden_dim, learn_beta=True)
    lif3 = StateLeaky(beta=torch.tensor(0.9), channels=hidden_dim, learn_beta=True)

    # Package into a Module for easy param counting
    block = nn.Module()
    block.ln1 = ln1
    block.ln2 = ln2
    block.ln3 = ln3
    block.ln_out = ln_out
    block.fc2 = fc2
    block.fc3 = fc3
    block.fc4 = fc4
    block.lif1 = lif1
    block.lif2 = lif2
    block.lif3 = lif3
    return block


def build_associative_block(hidden_dim: int) -> nn.Module:
    """
    Lightweight block mirroring the TinyStories Gen2 stack (without embeddings/output):
    - 4x LayerNorm(hidden_dim)
    - 3x Linear(hidden_dim, hidden_dim)
    - 3x AssociativeLeaky(..., in_dim=hidden_dim, num_spiking_neurons=hidden_dim, use_q_projection=True)
      Note: This enforces square parametrization with d = n = sqrt(hidden_dim).
    """
    # Ensure square parametrization: hidden_dim must be a perfect square
    m = int(math.isqrt(hidden_dim))
    if m * m != hidden_dim:
        raise ValueError(f"hidden_dim must be a perfect square; got {hidden_dim}")

    ln1 = nn.LayerNorm(hidden_dim)
    ln2 = nn.LayerNorm(hidden_dim)
    ln3 = nn.LayerNorm(hidden_dim)
    ln_out = nn.LayerNorm(hidden_dim)

    fc2 = nn.Linear(hidden_dim, hidden_dim)
    fc3 = nn.Linear(hidden_dim, hidden_dim)
    fc4 = nn.Linear(hidden_dim, hidden_dim)

    gen2_1 = AssociativeLeaky.from_num_spiking_neurons(
        in_dim=hidden_dim, num_spiking_neurons=hidden_dim, use_q_projection=True
    )
    gen2_2 = AssociativeLeaky.from_num_spiking_neurons(
        in_dim=hidden_dim, num_spiking_neurons=hidden_dim, use_q_projection=True
    )
    gen2_3 = AssociativeLeaky.from_num_spiking_neurons(
        in_dim=hidden_dim, num_spiking_neurons=hidden_dim, use_q_projection=True
    )

    block = nn.Module()
    block.ln1 = ln1
    block.ln2 = ln2
    block.ln3 = ln3
    block.ln_out = ln_out
    block.fc2 = fc2
    block.fc3 = fc3
    block.fc4 = fc4
    block.gen2_1 = gen2_1
    block.gen2_2 = gen2_2
    block.gen2_3 = gen2_3
    return block


def find_best_under_target_by_instantiation(stateleaky_hidden_dim: int):
    """
    Instantiate blocks and walk up perfect squares until the AssociativeLeaky
    param count exceeds the StateLeaky target. Return the last under-or-equal
    match, along with the first over-threshold candidate.

    Returns:
      best_h_under, best_params_under, target_params, delta_under,
      next_h_over, next_params_over, delta_over
    """
    # Target param count by instantiation
    target_block = build_stateleaky_block(stateleaky_hidden_dim)
    target_params = count_params(target_block)

    best_h_under = None
    best_params_under = None

    m = 1
    while True:
        h_assoc = m * m
        assoc_block = build_associative_block(h_assoc)
        assoc_params = count_params(assoc_block)

        if assoc_params <= target_params:
            best_h_under = h_assoc
            best_params_under = assoc_params
            m += 1
            # Small safety cap, but crossing should happen quickly
            if m > 4096:
                break
            continue

        # We crossed the target at this m
        next_h_over = h_assoc
        next_params_over = assoc_params

        if best_h_under is None:
            # Never under-shot; report current as over and no under
            delta_over = next_params_over - target_params
            return None, None, target_params, None, next_h_over, next_params_over, delta_over

        delta_under = target_params - best_params_under
        delta_over = next_params_over - target_params
        return (
            best_h_under,
            best_params_under,
            target_params,
            delta_under,
            next_h_over,
            next_params_over,
            delta_over,
        )


def main():
    # Target: parameter count of StateLeaky(hidden_dim=256)
    state_hidden = 256

    # Walk up perfect squares and pick last under-or-equal by instantiated params
    (
        best_h_under,
        best_params_under,
        target_params,
        delta_under,
        next_h_over,
        next_params_over,
        delta_over,
    ) = find_best_under_target_by_instantiation(state_hidden)

    print(f"StateLeaky(hidden_dim={state_hidden}) params: {target_params}")

    if best_h_under is not None:
        print(f"Best under-or-equal AssociativeLeaky hidden_dim (perfect square): {best_h_under}")
        print(f"AssociativeLeaky(hidden_dim={best_h_under}) params: {best_params_under}")
        print(f"Under delta (target - assoc): {delta_under}")
    else:
        print("No under-or-equal associative match found before crossing.")

    if next_h_over is not None:
        print(f"Next over-threshold AssociativeLeaky hidden_dim (perfect square): {next_h_over}")
        print(f"AssociativeLeaky(hidden_dim={next_h_over}) params: {next_params_over}")
        print(f"Over delta (assoc - target): {delta_over}")



if __name__ == "__main__":
    main()

