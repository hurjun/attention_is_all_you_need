"""Noam learning-rate schedule (paper section 5.3, Eq. 3).

    lr(step) = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

The rate increases linearly for the first ``warmup_steps`` steps and then decays
proportionally to the inverse square root of the step number.
"""

from __future__ import annotations

from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def noam_lambda(d_model: int, warmup_steps: int) -> Callable[[int], float]:
    """Return a step -> multiplier function for :class:`~torch.optim.lr_scheduler.LambdaLR`.

    The optimizer's base learning rate should be ``1.0`` so that the multiplier
    *is* the learning rate.

    Args:
        d_model: Model dimension.
        warmup_steps: Number of warmup steps.

    Returns:
        A function mapping a 0-based step index to the learning-rate multiplier.
    """

    def lr_lambda(step: int) -> float:
        # LambdaLR calls with step starting at 0; use 1-based steps so the
        # formula is well defined at the first optimizer step.
        s = step + 1
        return d_model ** -0.5 * min(s ** -0.5, s * warmup_steps ** -1.5)

    return lr_lambda


def make_noam_scheduler(
    optimizer: Optimizer, d_model: int, warmup_steps: int = 4000
) -> LambdaLR:
    """Create a :class:`~torch.optim.lr_scheduler.LambdaLR` with the Noam schedule.

    Args:
        optimizer: Optimizer whose base ``lr`` should be ``1.0``.
        d_model: Model dimension.
        warmup_steps: Number of warmup steps (paper uses 4000).

    Returns:
        A configured ``LambdaLR`` scheduler.
    """
    return LambdaLR(optimizer, lr_lambda=noam_lambda(d_model, warmup_steps))
