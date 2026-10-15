"""Tanh MLP with configurable depth/width.

Per-layer weights and biases are exposed as `weights` / `biases` (equivalently,
`layers[k].weight` / `layers[k].bias`) with no other computation performed on
them here. specification section 6 needs exactly these trained matrices -- e.g.
Pi_j = prod_{k<=j} ||W_k||_2 -- computed from weights alone, without ever
evaluating the network; that computation belongs to certification/network_bounds.py,
not here.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_width: int = 64, hidden_depth: int = 4):
        super().__init__()
        sizes = [in_features] + [hidden_width] * hidden_depth + [out_features]
        self.layers = nn.ModuleList(nn.Linear(sizes[k], sizes[k + 1]) for k in range(len(sizes) - 1))

    @property
    def weights(self) -> list[torch.Tensor]:
        return [layer.weight for layer in self.layers]

    @property
    def biases(self) -> list[torch.Tensor]:
        return [layer.bias for layer in self.layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)
