import torch
import torch.nn as nn
import torch.nn.functional as F


class GRELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        # GELU derivative: φ(x) + x * Φ(x)
        # where φ = normal pdf, Φ = normal cdf
        cdf = 0.5 * (1.0 + torch.erf(input / torch.sqrt(torch.tensor(2.0, device=input.device))))
        pdf = torch.exp(-0.5 * input**2) / torch.sqrt(torch.tensor(2.0 * torch.pi, device=input.device))
        gelu_grad = cdf + input * pdf
        return grad_output * gelu_grad


class GRELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return GRELUFunction.apply(input)
