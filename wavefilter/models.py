from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class GlobalSoftMaxAttention(nn.Module):
    def forward(self, ampl: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return F.softmax(ampl, -1)


class Conv1dPulseFinderAttention(nn.Module):
    def __init__(self, length: int, use_amplitude: bool = True):
        super().__init__()
        self.use_amplitude = use_amplitude
        self.pulse_finder = nn.Conv1d(1, 1, length, padding="same")
        n_channels = 2 + int(use_amplitude)
        self.combine = nn.Conv1d(n_channels, 1, 5, padding="same")
        self.scale = nn.Conv1d(1, 1, 1, padding="same")

    def forward(self, ampl: torch.Tensor, original: torch.Tensor) -> Any:
        # Should be returning `torch.Tensor`:
        pf = self.pulse_finder(original)
        inputs: Tuple[torch.Tensor, ...] = (original, pf)
        if self.use_amplitude:
            inputs += (ampl,)
        concat = torch.concat(inputs, dim=-2)
        encoded = self.combine(concat)
        encoded = F.leaky_relu(encoded)
        # encoded = torch.tanh(encoded)
        encoded = self.scale(encoded)
        # encoded = f.softmax(encoded, -1)
        encoded = torch.sigmoid(encoded)
        return encoded


"""
    encoded = torch.sigmoid(encoded)
    #encoded = torch.tanh(encoded)
    encoded = self.scale(encoded)
    encoded = f.softmax(encoded, -1)
    #encoded = torch.sigmoid(encoded)
    return encoded
"""

"""
  def forward(self, ampl, original):
    pf = self.pulse_finder(original)
    inputs = (original, pf)
    if self.use_amplitude:
      inputs += (ampl,)
    concat = torch.concat(inputs, dim=-2)
    encoded = torch.tanh(F.softshrink(self.combine(concat)))
    return F.softmax(encoded, -1
"""


class ParallelWeightedModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module_weights: Dict[str, float] = {}

    def add(self, name: str, module: nn.Module, init_weight: float = 0.5) -> None:
        self.module_weights[name] = init_weight
        self.add_module(name, module)

    def increment_weight(self, name: str, delta: float, keep_sum: bool = True) -> None:
        total = sum(self.module_weights.values())
        remainder = total - self.module_weights[name]
        new_ratio = (remainder - delta) / remainder
        self.module_weights[name] += delta
        for module in self.module_weights:
            if module == name:
                continue
            self.module_weights[module] *= new_ratio

    def set_weight(self, name: str, weight: float) -> None:
        self.module_weights[name] = weight

    def get_weight(self, name: str) -> float:
        return self.module_weights[name]

    def extra_repr(self) -> str:
        # return ",".join(f"{k}_weight={v}" for k, v in self.module_weights.items())
        weights = ",".join(f"{k}={v}" for k, v in self.module_weights.items())
        return f"module_weights=({weights})"

    def forward(self, ampl: torch.Tensor, original: torch.Tensor) -> Any:
        results: List[torch.Tensor] = []
        total_weights = 0.0
        for name, module in self.named_children():
            weight = self.module_weights[name]
            total_weights += weight
            result = weight * module(ampl, original)
            results.append(result)
        return sum(results) / total_weights


# Use inspiration from https://stackoverflow.com/a/67347262
class TiedFlippedConvolve1D(nn.Module):
    def __init__(self, tied_to: nn.Conv1d) -> None:
        super().__init__()
        self.tied_to = tied_to
        self.bias = nn.parameter.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.bias, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, torch.flip(self.tied_to.weight, [-1]), self.bias, padding="same")


class Product(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class WaveFilter(nn.Module):
    def __init__(self, kernel_size: int, attend: nn.Module) -> None:
        super().__init__()
        self.convolve = nn.Conv1d(1, 1, kernel_size, padding="same")
        self.attend = attend
        self.encode = Product()
        self.reconstruct = TiedFlippedConvolve1D(self.convolve)

    def forward(self, input: torch.Tensor) -> Any:
        # Should be returning `torch.Tensor`:
        ampl = self.convolve(input)
        attend = self.attend(ampl, input)
        # attend = F.softmax(ampl, -1)
        encoded = self.encode(ampl, attend)
        output = self.reconstruct(encoded)
        # output = F.conv1d(encoded, torch.flip(self.convolve.weight, [-1]), self.out_bias, padding="same")
        return output
