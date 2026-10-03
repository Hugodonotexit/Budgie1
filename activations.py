"""Activation functions."""

import math

import torch
from torch import nn


def _dsilu_zero():
    """The x where d/dx SiLU(x) = 0 (SiLU's minimum, about -1.278), by Newton's method."""
    x = -1.28
    for _ in range(60):
        s = 1.0 / (1.0 + math.exp(-x))
        x -= (1.0 + x * (1.0 - s)) / ((1.0 - s) - x * s * (1.0 - s))
    return x


DSILU_ZERO = _dsilu_zero()


def dsilu(x):
    """The derivative of SiLU: sigmoid(x) * (1 + x * (1 - sigmoid(x))). Its value at 0 is 0.5."""
    s = torch.sigmoid(x.float())
    return (s * (1.0 + x.float() * (1.0 - s))).to(x.dtype)


def dsilu_centered(x):
    """dSiLU shifted so that it is exactly 0 at 0: its zero crossing (x = -1.278) moved to the origin."""
    return dsilu(x + DSILU_ZERO)


class CenteredDSiLU(nn.Module):
    """dsilu_centered as a layer: ranges over (-0.0998, 1.0998), 0 at the origin."""

    def forward(self, x):
        return dsilu_centered(x)
